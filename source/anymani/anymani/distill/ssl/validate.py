r"""Schema-7 full checkpoints 的独立 validation 命令行入口。

运行入口：``python -m anymani.distill.ssl.validate``。每个 ``--checkpoint`` 都是显式候选；
命令不会扫描训练目录，也不会回写源 run。
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from .config_store import compose_validation_cfg
from .post_training import EmbodimentValidation


def _build_parser() -> argparse.ArgumentParser:
    r"""构造显式 baseline/candidate 和固定 q-bank 预算的平坦 CLI。"""

    parser = argparse.ArgumentParser(description="Validate explicit AnyMani Geometry SSL checkpoints.")
    parser.add_argument("--baseline_checkpoint", required=True)
    parser.add_argument("--checkpoint", action="append", required=True, dest="checkpoints")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--q_per_asset", type=int, default=None)
    parser.add_argument("--assets_per_minibatch", type=int, default=None)
    parser.add_argument("--q_per_asset_per_minibatch", type=int, default=None)
    parser.add_argument("--max_resident_assets", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--deterministic_algorithms", action=argparse.BooleanOptionalAction, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    r"""组合 canonical method/data，执行一次显式 checkpoint selection。"""

    args = _build_parser().parse_args(argv)
    config = compose_validation_cfg()
    validation_updates = {
        name: getattr(args, name)
        for name in (
            "q_per_asset",
            "assets_per_minibatch",
            "q_per_asset_per_minibatch",
            "max_resident_assets",
            "device",
        )
        if getattr(args, name) is not None
    }
    run_updates = {
        "baseline_checkpoint": args.baseline_checkpoint,
        "checkpoints": tuple(args.checkpoints),
        **{
            name: getattr(args, name)
            for name in ("output_dir", "experiment_name", "seed", "deterministic_algorithms")
            if getattr(args, name) is not None
        },
    }
    config = replace(
        config,
        validation=replace(config.validation, **validation_updates),
        run=replace(config.run, **run_updates),
    )
    output_dir = EmbodimentValidation(config).run()
    print(output_dir)
    return output_dir


if __name__ == "__main__":
    main()


__all__ = ["main"]
