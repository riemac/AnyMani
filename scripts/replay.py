"""Replay the packaged student or a frozen family teacher on one selected hand."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/anymani"))

from anymani.publication.runtime import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_IMPLEMENTATION_CERTIFICATE,
    PaperRuntime,
    checkpoint_manifest_sha256,
    create_run_directory,
    evaluator_arguments_for_policy,
    resolve_student_runtime_artifacts,
    run_module_main,
    sha256_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--cohort",
        default="leap_training",
        choices=(
            "leap_training", "allegro_training", "leap_right_variant", "leap_right_mother",
            "allegro_right_variant", "allegro_right_mother",
        ),
    )
    hand = parser.add_mutually_exclusive_group()
    hand.add_argument("--asset-index", type=int, default=0, help="Index in the ready parent runtime axis.")
    hand.add_argument("--asset-id", type=str, help="Asset ID or source-qualified key, such as evaluation_only#2.")
    parser.add_argument("--policy", choices=("student", "teacher"), default="student")
    parser.add_argument("--student-checkpoint", type=Path, default=None, help="Override the packaged student state file.")
    parser.add_argument("--student-torchscript", type=Path, default=None, help="Override the packaged TorchScript actor.")
    parser.add_argument("--student-sidecar", type=Path, default=None, help="Override the actor ABI sidecar.")
    parser.add_argument("--steps", type=int, default=600, help="Policy steps; 600 is the frozen 30-second window.")
    parser.add_argument("--action-mode", choices=("mean", "sample"), default="mean")
    parser.add_argument("--action-seed", type=int, default=1042)
    parser.add_argument("--record-video", action="store_true", help="Write a silent MP4 of the selected hand.")
    parser.add_argument(
        "--video-preset",
        choices=("standard", "paper_white"),
        default="standard",
        help="Select standard viewing or the white-background paper recording preset.",
    )
    parser.add_argument("--video-resolution", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"), default=None)
    parser.add_argument("--video-eye", nargs=3, type=float, metavar=("X", "Y", "Z"), default=None)
    parser.add_argument("--video-lookat", nargs=3, type=float, metavar=("X", "Y", "Z"), default=None)
    parser.add_argument("--real-time", action="store_true", help="Pace an interactive viewer at the policy rate.")
    parser.add_argument("--headless", action="store_true", help="Run without a viewer.")
    parser.add_argument("--smoke", action="store_true", help="Use one hand, one replica, and two policy steps.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--reference", type=Path, default=None, help="Override the bundled frozen 30-second reference.")
    parser.add_argument(
        "--implementation-certificate",
        type=Path,
        default=DEFAULT_IMPLEMENTATION_CERTIFICATE,
        help="Verified static/runtime compatibility record; defaults to the repository record.",
    )
    parser.add_argument("--rl-games-root", type=Path, default=None, help="Optional pinned rl-games 1.6.5 source root.")
    return parser


def _write_evaluation_metadata(path: Path, metadata: dict) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    document["publication_runtime"] = metadata
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.implementation_certificate is not None:
        args.implementation_certificate = args.implementation_certificate.expanduser().resolve(strict=True)
    if args.policy == "student" and args.action_mode != "mean":
        raise ValueError("the packaged TorchScript student supports deterministic mean replay")
    student_overrides = (args.student_checkpoint, args.student_torchscript, args.student_sidecar)
    if args.policy == "teacher" and any(path is not None for path in student_overrides):
        raise ValueError("student artifact overrides require --policy student")
    if args.record_video and args.headless:
        raise ValueError("--record-video requires a visible Isaac viewer")
    runtime = PaperRuntime.open(args.data_dir)
    student_artifacts = (
        resolve_student_runtime_artifacts(
            runtime.models,
            checkpoint=args.student_checkpoint,
            torchscript=args.student_torchscript,
            sidecar=args.student_sidecar,
        )
        if args.policy == "student"
        else None
    )
    if args.output_dir is None:
        output_dir = create_run_directory("replay", args.cohort)
    else:
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=False)
    selector: int | str = args.asset_id if args.asset_id is not None else args.asset_index
    steps = 2 if args.smoke else args.steps
    replicas = 1
    selection = runtime.member_view(args.cohort, selector, lock_dir=output_dir / "cohort_views")
    runtime.configure_task_environment(selection, rl_games_root=args.rl_games_root)
    reference = runtime.write_reference(output_dir, args.reference)
    teacher = runtime.models.teacher(selection.family)
    encoder, encoder_sha256 = runtime.models.encoder()
    evaluation = output_dir / "replay.json"
    evaluator_args = [
        "--cohort_lock", str(selection.lock_path),
        "--reference", str(reference),
        "--output", str(evaluation),
        "--num_replicas", str(replicas),
        "--steps", str(steps),
        "--viewer_asset_index", "0",
        "--action_mode", args.action_mode,
        "--n040_artifact", str(encoder),
        "--n040_sha256", encoder_sha256,
        "--video_preset", args.video_preset,
        "--rl_games_strict",
    ]
    for option, value in (
        ("--video_resolution", args.video_resolution),
        ("--video_eye", args.video_eye),
        ("--video_lookat", args.video_lookat),
    ):
        if value is not None:
            evaluator_args.extend((option, *(str(item) for item in value)))
    if args.action_mode == "sample":
        evaluator_args.extend(("--action_seed", str(args.action_seed)))
    if checkpoint_manifest_sha256(teacher) != selection.metadata["runtime_lock_sha256"]:
        evaluator_args.append("--cohort_transfer")
    if args.implementation_certificate is not None:
        evaluator_args.extend(("--implementation_certificate", str(args.implementation_certificate.expanduser().resolve())))
    if args.headless:
        evaluator_args.append("--headless")
    if args.real_time:
        evaluator_args.append("--real-time")
    if args.record_video:
        evaluator_args.extend(("--video", str(output_dir / "replay.mp4")))

    if args.policy == "student":
        status = output_dir / "student-status.json"
        assert student_artifacts is not None
        wrapper_args = [
            "--student_checkpoint", str(student_artifacts.checkpoint),
            "--student_torchscript", str(student_artifacts.torchscript),
            "--student_sidecar", str(student_artifacts.sidecar),
            "--reference_teacher_checkpoint", str(teacher),
            "--student_status", str(status),
            "--", *evaluator_arguments_for_policy(args.policy, teacher, evaluator_args),
        ]
        exit_code = run_module_main("anymani.distill.il.evaluate_family", wrapper_args)
    else:
        exit_code = run_module_main(
            "anymani.distill.rl.evaluate_palm_rotation_mvp",
            evaluator_arguments_for_policy(args.policy, teacher, evaluator_args),
        )
    if exit_code != 0:
        return exit_code
    if not evaluation.is_file():
        raise RuntimeError("Replay completed without the evaluator JSON artifact")
    _write_evaluation_metadata(
        evaluation,
        {
            **selection.metadata,
            "policy": args.policy,
            "family": selection.family,
            "steps": steps,
            "replicas_per_asset": replicas,
            "action_mode": args.action_mode,
            "action_seed": args.action_seed if args.action_mode == "sample" else None,
            "video_configuration": {
                "record_video": args.record_video,
                "preset": args.video_preset,
                "resolution": args.video_resolution,
                "eye": args.video_eye,
                "lookat": args.video_lookat,
                "silent": True,
                "policy_rate_hz": 20,
            },
            "smoke": args.smoke,
            "n040_sha256": encoder_sha256,
            "teacher_sha256": sha256_file(teacher),
            "reference_path": str(reference),
            "reference_sha256": sha256_file(reference),
            "implementation_certificate_path": str(args.implementation_certificate),
            "implementation_certificate_sha256": sha256_file(args.implementation_certificate),
        },
    )
    if args.policy == "student":
        assert student_artifacts is not None
        _write_evaluation_metadata(
            output_dir / "student-status.json",
            {
                **selection.metadata,
                "policy": args.policy,
                "family": selection.family,
                "smoke": args.smoke,
                "n040_sha256": encoder_sha256,
                "teacher_sha256": sha256_file(teacher),
                "student_sha256": sha256_file(student_artifacts.checkpoint),
                "student_torchscript_sha256": sha256_file(student_artifacts.torchscript),
                "student_sidecar_path": str(student_artifacts.sidecar),
            },
        )
    print(f"Replay JSON: {evaluation}")
    if args.record_video:
        print(f"Replay video: {output_dir / 'replay.mp4'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
