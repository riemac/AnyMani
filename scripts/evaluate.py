"""Run fixed 30-second, R16 evaluations for a paper cohort or population."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
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

TRAIN_COHORTS = {"leap": ("leap_training",), "allegro": ("allegro_training",)}
UNSEEN_COHORTS = {
    "leap": ("leap_right_variant", "leap_right_mother"),
    "allegro": ("allegro_right_variant", "allegro_right_mother"),
}
COHORT_FAMILY = {
    **{name: "leap" for names in (TRAIN_COHORTS["leap"], UNSEEN_COHORTS["leap"]) for name in names},
    **{name: "allegro" for names in (TRAIN_COHORTS["allegro"], UNSEEN_COHORTS["allegro"]) for name in names},
}
ALL_COHORTS = (
    "leap_training", "allegro_training", "leap_right_variant", "leap_right_mother",
    "allegro_right_variant", "allegro_right_mother",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--policy", choices=("student", "teacher"), default="student")
    parser.add_argument("--student-checkpoint", type=Path, default=None, help="Override the packaged student state file.")
    parser.add_argument("--student-torchscript", type=Path, default=None, help="Override the packaged TorchScript actor.")
    parser.add_argument("--student-sidecar", type=Path, default=None, help="Override the actor ABI sidecar.")
    parser.add_argument("--population", choices=("cohort", "train", "unseen"), default="cohort")
    parser.add_argument("--cohort", choices=ALL_COHORTS, default=None)
    parser.add_argument("--family", choices=("all", "leap", "allegro"), default="all")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument(
        "--implementation-certificate",
        type=Path,
        default=DEFAULT_IMPLEMENTATION_CERTIFICATE,
        help="Verified static/runtime compatibility record; defaults to the repository record.",
    )
    parser.add_argument("--rl-games-root", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true", help="Use one member view, one replica, and two steps.")
    parser.add_argument("--_single-cohort", action="store_true", help=argparse.SUPPRESS)
    return parser


def _select_cohorts(population: str, cohort: str | None, family: str) -> tuple[str, ...]:
    if population == "cohort":
        if cohort is None:
            raise ValueError("--population cohort requires --cohort")
        if family != "all" and COHORT_FAMILY[cohort] != family:
            raise ValueError(f"cohort {cohort!r} belongs to {COHORT_FAMILY[cohort]!r}, not {family!r}")
        return (cohort,)
    if cohort is not None:
        raise ValueError("--cohort cannot be combined with --population train or unseen")
    families = ("leap", "allegro") if family == "all" else (family,)
    groups = TRAIN_COHORTS if population == "train" else UNSEEN_COHORTS
    return tuple(name for selected_family in families for name in groups[selected_family])


def _output_directory(raw: Path | None) -> Path:
    if raw is None:
        return create_run_directory("evaluation")
    output = raw.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    return output


def _ensure_empty_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(f"Evaluation output directory is not empty: {path}")


def _attach_runtime_metadata(path: Path, metadata: dict) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    document["publication_runtime"] = metadata
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run_cohort(
    *,
    runtime: PaperRuntime,
    cohort_name: str,
    policy: str,
    student_artifacts,
    output_dir: Path,
    smoke: bool,
    reference_override: Path | None,
    certificate: Path | None,
    rl_games_root: Path | None,
) -> dict:
    _ensure_empty_directory(output_dir)
    if smoke:
        selection = runtime.member_view(cohort_name, 0, lock_dir=output_dir / "cohort_views")
        replicas, steps = 1, 2
    else:
        selection = runtime.cohort(cohort_name)
        replicas, steps = 16, 600
    runtime.configure_task_environment(selection, rl_games_root=rl_games_root)

    bundle_cohort = runtime.assets.cohort(cohort_name)
    nominal_count = int(bundle_cohort["nominal_count"])
    ready_count = int(bundle_cohort["ready_count"])
    failures = tuple(bundle_cohort.get("initialization_failures", ()))
    if nominal_count - ready_count != len(failures):
        raise RuntimeError(f"Asset bundle failure list disagrees with cohort counts: {cohort_name}")

    teacher = runtime.models.teacher(selection.family)
    encoder, encoder_sha256 = runtime.models.encoder()
    reference = runtime.write_reference(output_dir, reference_override)
    result_path = output_dir / "evaluation.json"
    evaluator_args = [
        "--cohort_lock", str(selection.lock_path),
        "--reference", str(reference),
        "--output", str(result_path),
        "--num_replicas", str(replicas),
        "--steps", str(steps),
        "--action_mode", "mean",
        "--n040_artifact", str(encoder),
        "--n040_sha256", encoder_sha256,
        "--headless",
        "--rl_games_strict",
    ]
    if checkpoint_manifest_sha256(teacher) != selection.metadata["runtime_lock_sha256"]:
        evaluator_args.append("--cohort_transfer")
    if certificate is not None:
        evaluator_args.extend(("--implementation_certificate", str(certificate.expanduser().resolve(strict=True))))

    if policy == "student":
        status_path = output_dir / "student-status.json"
        if student_artifacts is None:
            raise RuntimeError("student evaluation requires resolved student runtime artifacts")
        wrapper_args = [
            "--student_checkpoint", str(student_artifacts.checkpoint),
            "--student_torchscript", str(student_artifacts.torchscript),
            "--student_sidecar", str(student_artifacts.sidecar),
            "--reference_teacher_checkpoint", str(teacher),
            "--student_status", str(status_path),
            "--", *evaluator_arguments_for_policy(policy, teacher, evaluator_args),
        ]
        exit_code = run_module_main("anymani.distill.il.evaluate_family", wrapper_args)
    else:
        exit_code = run_module_main(
            "anymani.distill.rl.evaluate_palm_rotation_mvp",
            evaluator_arguments_for_policy(policy, teacher, evaluator_args),
        )
    if exit_code != 0:
        return {"cohort": cohort_name, "status": "failed", "exit_code": exit_code}
    if not result_path.is_file():
        raise RuntimeError("Evaluation exited successfully without its JSON artifact")

    evaluation_metadata = {
        **selection.metadata,
        "policy": policy,
        "family": selection.family,
        "protocol": {"horizon_s": steps * 0.05, "replicas_per_ready_asset": replicas, "steps": steps},
        "nominal_asset_denominator": nominal_count,
        "ready_assets_evaluated": ready_count if not smoke else 1,
        "unavailable_initialization_failures": list(failures),
        "unavailable_assets_count_as_failures": True,
        "smoke": smoke,
        "n040_sha256": encoder_sha256,
        "teacher_sha256": sha256_file(teacher),
        "student_sha256": sha256_file(student_artifacts.checkpoint) if student_artifacts is not None else None,
        "student_torchscript_sha256": (
            sha256_file(student_artifacts.torchscript) if student_artifacts is not None else None
        ),
        "student_sidecar_path": str(student_artifacts.sidecar) if student_artifacts is not None else None,
        "reference_path": str(reference),
        "reference_sha256": sha256_file(reference),
        "implementation_certificate_path": str(certificate) if certificate is not None else None,
        "implementation_certificate_sha256": sha256_file(certificate) if certificate is not None else None,
    }
    _attach_runtime_metadata(result_path, evaluation_metadata)
    if policy == "student":
        _attach_runtime_metadata(output_dir / "student-status.json", evaluation_metadata)
    return {
        "cohort": cohort_name,
        "status": "completed",
        "evaluation": str(result_path),
        "nominal_asset_count": nominal_count,
        "ready_asset_count": ready_count,
        "initialization_failure_count": len(failures),
        "initialization_failures": list(failures),
        "replicas_per_ready_asset": replicas,
        "policy_steps": steps,
        "publication_runtime": evaluation_metadata,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cohorts = _select_cohorts(args.population, args.cohort, args.family)
    student_overrides = (args.student_checkpoint, args.student_torchscript, args.student_sidecar)
    if args.policy == "teacher" and any(path is not None for path in student_overrides):
        raise ValueError("student artifact overrides require --policy student")
    if args.implementation_certificate is not None:
        args.implementation_certificate = args.implementation_certificate.expanduser().resolve(strict=True)
    if args.rl_games_root is not None:
        args.rl_games_root = args.rl_games_root.expanduser().resolve(strict=True)
    if args.output_dir is None:
        output_dir = create_run_directory("evaluation", args.population)
    elif args._single_cohort:
        output_dir = args.output_dir.expanduser().resolve()
    else:
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=False)

    results = []
    if len(cohorts) == 1:
        runtime = PaperRuntime.open(args.data_dir)
        results.append(_run_cohort(
            runtime=runtime,
            cohort_name=cohorts[0],
            policy=args.policy,
            student_artifacts=(
                resolve_student_runtime_artifacts(
                    runtime.models,
                    checkpoint=args.student_checkpoint,
                    torchscript=args.student_torchscript,
                    sidecar=args.student_sidecar,
                )
                if args.policy == "student"
                else None
            ),
            output_dir=output_dir,
            smoke=args.smoke,
            reference_override=args.reference,
            certificate=args.implementation_certificate,
            rl_games_root=args.rl_games_root,
        ))
    else:
        for cohort_name in cohorts:
            cohort_output = output_dir / cohort_name
            command = [
                str(Path(__file__).resolve()),
                "--data-dir", str(args.data_dir.expanduser().resolve()),
                "--population", "cohort",
                "--cohort", cohort_name,
                "--policy", args.policy,
                "--output-dir", str(cohort_output),
                "--_single-cohort",
            ]
            if args.smoke:
                command.append("--smoke")
            if args.reference is not None:
                command.extend(("--reference", str(args.reference.expanduser().resolve(strict=True))))
            if args.implementation_certificate is not None:
                command.extend(("--implementation-certificate", str(args.implementation_certificate)))
            if args.rl_games_root is not None:
                command.extend(("--rl-games-root", str(args.rl_games_root)))
            for flag, value in (
                ("--student-checkpoint", args.student_checkpoint),
                ("--student-torchscript", args.student_torchscript),
                ("--student-sidecar", args.student_sidecar),
            ):
                if value is not None:
                    command.extend((flag, str(value.expanduser().resolve(strict=True))))
            process = subprocess.run([sys.executable, *command], check=False)
            child_summary = cohort_output / "cohort-summary.json"
            if process.returncode == 0 and child_summary.is_file():
                results.append(json.loads(child_summary.read_text(encoding="utf-8"))["results"][0])
            else:
                results.append({"cohort": cohort_name, "status": "failed", "exit_code": int(process.returncode)})
                break

    nominal_total = sum(int(result.get("nominal_asset_count", 0)) for result in results)
    ready_total = sum(int(result.get("ready_asset_count", 0)) for result in results)
    failures_total = sum(int(result.get("initialization_failure_count", 0)) for result in results)
    completed = len(results) == len(cohorts) and all(result.get("status") == "completed" for result in results)
    summary = {
        "artifact_type": "anymani.paper_fixed_evaluation_summary",
        "schema_version": "1.0.0",
        "status": "completed" if completed else "failed",
        "policy": args.policy,
        "population": args.population,
        "family": args.family,
        "protocol": {"horizon_s": 2 * 0.05 if args.smoke else 30.0, "replicas_per_ready_asset": 1 if args.smoke else 16},
        "nominal_asset_denominator": nominal_total,
        "ready_assets_evaluated": ready_total if not args.smoke else len(results),
        "unavailable_initialization_failures": failures_total,
        "unavailable_assets_count_as_failures": True,
        "results": results,
    }
    summary_path = output_dir / "cohort-summary.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Evaluation summary: {summary_path}")
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
