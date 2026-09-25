"""Collect frozen family-teacher demonstrations with the paper protocol."""

from __future__ import annotations
import argparse
import json
import runpy
import sys
import traceback
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect a frozen family teacher for offline shared students.")
    parser.add_argument("--dataset_output", type=Path, required=True)
    parser.add_argument("--collection_family", choices=("leap_right", "allegro_right"), required=True)
    parser.add_argument("--collection_action_mode", choices=("mean", "sample"), default="mean")
    parser.add_argument("--collection_action_seed", type=int, default=None)
    parser.add_argument("--dataset_stride", type=int, default=4)
    parser.add_argument("--collector_status", type=Path, required=True)
    args, evaluator_args = parser.parse_known_args()
    if evaluator_args and evaluator_args[0] == "--":
        evaluator_args = evaluator_args[1:]
    if args.dataset_output.exists() or args.collector_status.exists():
        raise FileExistsError("collection data and status paths must be new")
    args.collector_status.parent.mkdir(parents=True, exist_ok=True)
    sys.argv = [sys.argv[0], *evaluator_args]
    backend = None
    collector = None
    code = 1
    try:
        backend = runpy.run_module("anymani.distill.rl.evaluate_palm_rotation_mvp", run_name="family_evaluator_backend")
        from anymani.distill.il.family_collection import FrozenFamilyCollection
        from anymani.distill.il.family_dataset import read_family_metadata

        collector = FrozenFamilyCollection(
            args.dataset_output,
            family=args.collection_family,
            mode=args.collection_action_mode,
            seed=args.collection_action_seed,
            sample_stride=args.dataset_stride,
        )
        backend["main"](collector=collector)
        metadata = read_family_metadata(args.dataset_output)
        args.collector_status.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "python_exit_code": 0,
                    "dataset": str(args.dataset_output.resolve()),
                    "frames": metadata["recorded_steps"],
                    "sample_steps": metadata["sample_count"],
                    "env_count": metadata["env_count"],
                    "qualified_episodes": metadata["qualified_episode_count"],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        code = 0
    except BaseException as error:
        args.collector_status.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "python_exit_code": 1,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "completed_policy_steps": collector.executed_steps if collector is not None else 0,
                    "attempted_policy_steps": collector._step if collector is not None else 0,
                    "env_count": int(collector.active.numel())
                    if collector is not None and collector.active is not None
                    else 0,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        traceback.print_exc()
        sys.stderr.flush()
    finally:
        if collector is not None:
            collector.close()
        if backend is not None:
            backend["simulation_app"].close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
