"""Train the shared BC student from the four collected teacher datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/anymani"))

DATASET_NAMES = ("leap_mean.h5", "leap_sample.h5", "allegro_mean.h5", "allegro_sample.h5")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demonstrations", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/student")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-seconds", type=float, default=7200)
    parser.add_argument("--max-ram-gib", type=float, default=16)
    parser.add_argument("--resume", type=Path, help="Resume a checkpoint inside the same output directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    datasets = [args.demonstrations.expanduser().resolve() / name for name in DATASET_NAMES]
    for path in datasets:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; first run scripts/collect.py.")

    from anymani.distill.il.family_student import export_family_student_torchscript
    from anymani.distill.il.train_family import run_family_training

    output = args.output.expanduser().resolve()
    report = run_family_training(
        datasets,
        output_dir=output,
        representation="n040",
        seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.epochs,
        max_seconds=args.max_seconds,
        max_ram_gib=args.max_ram_gib,
        device=args.device,
        resume=args.resume,
        tf32=False,
    )
    summary = {
        "status": report.get("status"),
        "training_report": str(output / "training-report.json"),
        "best_checkpoint": str(output / "best.pt"),
        "best_epoch": report.get("best_epoch"),
        "updates": report.get("updates"),
    }
    if report.get("status") == "completed" and (output / "best.pt").is_file():
        export = export_family_student_torchscript(output / "best.pt", output / "actor.ts")
        summary["export"] = export
    print(json.dumps(summary, indent=2, default=str))
    return 0 if report.get("status") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
