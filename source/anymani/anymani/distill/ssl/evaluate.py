r"""Schema-7 full checkpoint 的独立 held-out evaluation 命令行入口。

运行入口：``python -m anymani.distill.ssl.evaluate``。只有显式提供 ``--baseline_checkpoint``
时才执行训练形态 q-bank 前后对比。
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from .config_store import compose_evaluation_cfg
from .post_training import EmbodimentEvaluation


def _build_parser() -> argparse.ArgumentParser:
    r"""构造目标 checkpoint、可选 baseline 和固定评估资源的平坦 CLI。"""

    parser = argparse.ArgumentParser(description="Evaluate one AnyMani Geometry SSL checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline_checkpoint", default="")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--q_per_asset", type=int, default=None)
    parser.add_argument("--assets_per_minibatch", type=int, default=None)
    parser.add_argument("--q_per_asset_per_minibatch", type=int, default=None)
    parser.add_argument("--bootstrap_replicates", type=int, default=None)
    parser.add_argument("--max_resident_assets", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--deterministic_algorithms", action=argparse.BooleanOptionalAction, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    r"""组合 canonical method/data，执行一次显式 held-out evaluation。"""

    args = _build_parser().parse_args(argv)
    config = compose_evaluation_cfg()
    evaluation_updates = {
        name: getattr(args, name)
        for name in (
            "q_per_asset",
            "assets_per_minibatch",
            "q_per_asset_per_minibatch",
            "bootstrap_replicates",
            "max_resident_assets",
            "device",
        )
        if getattr(args, name) is not None
    }
    run_updates = {
        "checkpoint": args.checkpoint,
        "baseline_checkpoint": args.baseline_checkpoint,
        **{
            name: getattr(args, name)
            for name in ("output_dir", "experiment_name", "seed", "deterministic_algorithms")
            if getattr(args, name) is not None
        },
    }
    config = replace(
        config,
        evaluation=replace(config.evaluation, **evaluation_updates),
        run=replace(config.run, **run_updates),
    )
    output_dir = EmbodimentEvaluation(config).run()
    print(output_dir)
    return output_dir


if __name__ == "__main__":
    main()


__all__ = ["main"]
