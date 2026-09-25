"""Collect frozen LEAP and Allegro teacher demonstrations for offline student training."""

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
    run_module_main,
    sha256_file,
)

FAMILY_COHORTS = {"leap": "leap_training", "allegro": "allegro_training"}
COLLECTION_FAMILIES = {"leap": "leap_right", "allegro": "allegro_right"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--family", choices=("all", "leap", "allegro"), default="all")
    parser.add_argument("--mode", choices=("all", "mean", "sample"), default="all")
    parser.add_argument("--output", type=Path, default=None, help="New output directory for the four HDF5 files.")
    parser.add_argument("--sample-seed", type=int, default=1042)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument(
        "--implementation-certificate",
        type=Path,
        default=DEFAULT_IMPLEMENTATION_CERTIFICATE,
        help="Verified static/runtime compatibility record; defaults to the repository record.",
    )
    parser.add_argument("--rl-games-root", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true", help="Collect one hand, one replica, and two steps.")
    parser.add_argument("--_single-job", action="store_true", help=argparse.SUPPRESS)
    return parser


def _jobs(family: str, mode: str, *, smoke: bool) -> tuple[tuple[str, str], ...]:
    families = ("leap", "allegro") if family == "all" else (family,)
    modes = ("mean", "sample") if mode == "all" else (mode,)
    selected = tuple((item_family, item_mode) for item_family in families for item_mode in modes)
    if smoke and family == "all" and mode == "all":
        return (("leap", "mean"),)
    return selected


def _child_status_code(return_code: int, summary_exists: bool) -> int:
    """Treat a missing required child summary as failure even if the process exits zero."""
    if return_code != 0:
        return int(return_code)
    return 0 if summary_exists else 1


def _run_one(
    *,
    data_dir: Path,
    output_dir: Path,
    family: str,
    mode: str,
    sample_seed: int,
    smoke: bool,
    reference_override: Path | None,
    certificate: Path | None,
    rl_games_root: Path | None,
) -> dict:
    runtime = PaperRuntime.open(data_dir)
    cohort_name = FAMILY_COHORTS[family]
    if smoke:
        selection = runtime.member_view(cohort_name, 0, lock_dir=output_dir / "cohort_views")
        replicas, steps, stride = 1, 2, 1
    else:
        selection = runtime.cohort(cohort_name)
        replicas, steps, stride = 16, 600, 4
    runtime.configure_task_environment(selection, rl_games_root=rl_games_root)

    stem = f"{family}_{mode}"
    dataset_path = output_dir / f"{stem}.h5"
    status_path = output_dir / f"{stem}.status.json"
    evaluation_path = output_dir / f"{stem}.evaluation.json"
    if any(path.exists() for path in (dataset_path, status_path, evaluation_path)):
        raise FileExistsError(f"Collection output already exists for {stem} in {output_dir}")
    reference = runtime.write_reference(output_dir, reference_override)
    teacher = runtime.models.teacher(family)
    encoder, encoder_sha256 = runtime.models.encoder()
    evaluator_args = [
        "--checkpoint", str(teacher),
        "--cohort_lock", str(selection.lock_path),
        "--reference", str(reference),
        "--output", str(evaluation_path),
        "--num_replicas", str(replicas),
        "--steps", str(steps),
        "--trace_stride", "1",
        "--trace_rewards",
        "--n040_artifact", str(encoder),
        "--n040_sha256", encoder_sha256,
        "--headless",
        "--rl_games_strict",
    ]
    if checkpoint_manifest_sha256(teacher) != selection.metadata["runtime_lock_sha256"]:
        evaluator_args.append("--cohort_transfer")
    if certificate is not None:
        evaluator_args.extend(("--implementation_certificate", str(certificate.expanduser().resolve(strict=True))))

    collection_args = [
        "--dataset_output", str(dataset_path),
        "--collection_family", COLLECTION_FAMILIES[family],
        "--collection_action_mode", mode,
        "--dataset_stride", str(stride),
        "--collector_status", str(status_path),
    ]
    if mode == "sample":
        collection_args.extend(("--collection_action_seed", str(sample_seed)))
    collection_args.extend(("--", *evaluator_args))
    exit_code = run_module_main("anymani.distill.il.collect_family", collection_args)
    if exit_code != 0:
        return {"family": family, "mode": mode, "status": "failed", "exit_code": exit_code}

    from anymani.distill.il.family_dataset import read_family_metadata

    metadata = read_family_metadata(dataset_path)
    expected_env_count = len(selection.cohort.members) * replicas
    expected_sample_count = (steps - 1) // stride + 1
    if (
        metadata["env_count"] != expected_env_count
        or metadata["recorded_steps"] != steps
        or metadata["sample_stride"] != stride
        or metadata["sample_count"] != expected_sample_count
    ):
        raise RuntimeError(f"Collected dataset dimensions disagree with the requested protocol: {metadata}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["publication_runtime"] = {
        **selection.metadata,
        "family": family,
        "action_mode": mode,
        "action_seed": sample_seed if mode == "sample" else None,
        "replicas_per_asset": replicas,
        "policy_steps": steps,
        "dataset_stride": stride,
        "n040_sha256": encoder_sha256,
        "teacher_sha256": sha256_file(teacher),
        "reference_path": str(reference),
        "reference_sha256": sha256_file(reference),
        "implementation_certificate_path": str(certificate) if certificate is not None else None,
        "implementation_certificate_sha256": sha256_file(certificate) if certificate is not None else None,
        "smoke": smoke,
    }
    temporary = status_path.with_suffix(status_path.suffix + ".tmp")
    temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(status_path)
    return {
        "family": family,
        "mode": mode,
        "status": "completed",
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "status_path": str(status_path),
        "evaluation": str(evaluation_path),
        "frames": metadata["recorded_steps"],
        "samples": metadata["sample_count"],
        "environment_count": metadata["env_count"],
        "asset_count": metadata["asset_count"],
        "publication_runtime": status["publication_runtime"],
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    jobs = _jobs(args.family, args.mode, smoke=args.smoke)
    if args.sample_seed < 0:
        raise ValueError("--sample-seed must be non-negative")
    if args.implementation_certificate is not None:
        args.implementation_certificate = args.implementation_certificate.expanduser().resolve(strict=True)
    if args.rl_games_root is not None:
        args.rl_games_root = args.rl_games_root.expanduser().resolve(strict=True)

    output_dir = args.output.expanduser().resolve() if args.output is not None else None
    if output_dir is None:
        output_dir = create_run_directory("teacher-demonstrations")
    elif not args._single_job:
        output_dir.mkdir(parents=True, exist_ok=False)
    elif not output_dir.is_dir():
        raise FileNotFoundError(f"Parent collection output directory does not exist: {output_dir}")

    results = []
    if len(jobs) == 1:
        family, mode = jobs[0]
        results.append(_run_one(
            data_dir=args.data_dir,
            output_dir=output_dir,
            family=family,
            mode=mode,
            sample_seed=args.sample_seed,
            smoke=args.smoke,
            reference_override=args.reference,
            certificate=args.implementation_certificate,
            rl_games_root=args.rl_games_root,
        ))
    else:
        for family, mode in jobs:
            command = [
                str(Path(__file__).resolve()),
                "--data-dir", str(args.data_dir.expanduser().resolve()),
                "--family", family,
                "--mode", mode,
                "--output", str(output_dir),
                "--sample-seed", str(args.sample_seed),
                "--_single-job",
            ]
            if args.smoke:
                command.append("--smoke")
            if args.reference is not None:
                command.extend(("--reference", str(args.reference.expanduser().resolve(strict=True))))
            if args.implementation_certificate is not None:
                command.extend(("--implementation-certificate", str(args.implementation_certificate)))
            if args.rl_games_root is not None:
                command.extend(("--rl-games-root", str(args.rl_games_root)))
            process = subprocess.run([sys.executable, *command], check=False)
            child_summary = output_dir / f"{family}_{mode}.summary.json"
            if process.returncode == 0 and child_summary.is_file():
                results.extend(json.loads(child_summary.read_text(encoding="utf-8")).get("results", ()))
            else:
                results.append({
                    "family": family,
                    "mode": mode,
                    "exit_code": _child_status_code(process.returncode, child_summary.is_file()),
                })
            if process.returncode != 0:
                break

    completed = all(result.get("status") == "completed" or result.get("exit_code") == 0 for result in results)
    summary = {
        "artifact_type": "anymani.paper_teacher_collection_run",
        "schema_version": "1.0.0",
        "status": "completed" if completed else "failed",
        "data_dir": str(args.data_dir.expanduser().resolve()),
        "output_dir": str(output_dir),
        "requested_jobs": [{"family": family, "mode": mode} for family, mode in jobs],
        "smoke": args.smoke,
        "sample_seed": args.sample_seed,
        "results": results,
    }
    summary_path = (
        output_dir / f"{jobs[0][0]}_{jobs[0][1]}.summary.json"
        if args._single_job
        else output_dir / "collection-summary.json"
    )
    if summary_path.exists():
        raise FileExistsError(summary_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Collection summary: {summary_path}")
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
