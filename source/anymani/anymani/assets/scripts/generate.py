"Provides pre-made and explicit-source post-mutate commands. Topology-subset selection is not implemented; pre-made currently uses the declared complete enumeration order."

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from ..config import AssetRunStrategyCfg
from ..generator.hand_generator import HandGeneratorCfg
from ._asset_generate_runner import (
    enumerate_post_mutate_bundles,
    enumerate_premade_bundles,
    prepare_post_mutate_run_cfg,
    print_post_mutate_result_summary,
    print_post_mutate_summary,
    print_premade_registry_summary,
    print_premade_result_summary,
)


def _load_config_module(module_name: str):

    return importlib.import_module(module_name)


def _build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(description="Unified asset generation runner.")
    parser.add_argument(
        "--stage",
        choices=("pre-made", "post-mutate"),
        required=True,
        help="Which asset generation stage to run.",
    )
    parser.add_argument(
        "--config-module",
        default="anymani.assets.config.asset_gen_cfg",
        help="Python module path containing PRE_MADE_CFG / POST_MUTATE_CFG.",
    )
    parser.add_argument("--source-path", default=None, help="Override post-mutate source topology path.")
    parser.add_argument("--n-samples", type=int, default=None, help="Override HandGeneratorCfg.n_samples.")
    parser.add_argument(
        "--post-mutate-seed",
        type=int,
        default=None,
        help="Override HandGeneratorCfg.post_mutate_seed for an independent reproducible variant set.",
    )
    parser.add_argument("--max-enumerate", type=int, default=None, help="Override pre-made max_enumerate.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override the pre-made output root.")
    return parser


def _validate_strategy(strategy: AssetRunStrategyCfg) -> None:

    # NOTE:


    if strategy.topology_selection_mode != "all":
        raise NotImplementedError(
            "Runner strategy extensions are declared but not implemented yet; "
            "current runner only supports topology_selection_mode='all'."
        )

    # NOTE:


    if strategy.topology_selection_count is not None:
        raise NotImplementedError(
            "Runner strategy extensions are declared but not implemented yet; "
            "topology_selection_count must stay None for now."
        )


def _run_premade(module, *, max_enumerate: int | None, output_dir: Path | None) -> int:

    cfg: HandGeneratorCfg = module.PRE_MADE_CFG


    if max_enumerate is not None:
        cfg = cfg.replace(max_enumerate=max_enumerate)
    if output_dir is not None:
        cfg = cfg.replace(output_dir=output_dir)


    if getattr(module, "PRE_MADE_SHOW_REGISTRY", False):
        print_premade_registry_summary(cfg)


    results = enumerate_premade_bundles(cfg)
    print_premade_result_summary(
        results,
        cfg,
        print_limit=getattr(module, "PRE_MADE_PRINT_RESULT_LIMIT", None),
    )
    return 0


def _run_post_mutate(
    module,
    *,
    source_path: str | None,
    n_samples: int | None,
    post_mutate_seed: int | None,
) -> int:

    cfg: HandGeneratorCfg = module.POST_MUTATE_CFG


    if n_samples is not None:
        cfg = cfg.replace(n_samples=n_samples)
    if post_mutate_seed is not None:
        cfg = cfg.replace(post_mutate_seed=post_mutate_seed)


    resolved_source_path = source_path or module.POST_MUTATE_SOURCE_TOPOLOGY_PATH


    prepared_cfg, source_topology_dir, planned_run_dir = prepare_post_mutate_run_cfg(
        cfg,
        source_path=resolved_source_path,
    )


    print_post_mutate_summary(
        prepared_cfg,
        source_path=resolved_source_path,
        source_topology_dir=source_topology_dir,
        planned_run_dir=planned_run_dir,
    )


    results = enumerate_post_mutate_bundles(prepared_cfg)
    print_post_mutate_result_summary(
        results,
        print_limit=getattr(module, "POST_MUTATE_PRINT_RESULT_LIMIT", None),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    "Dispatches the selected asset command and returns its process status."

    parser = _build_parser()
    args = parser.parse_args(argv)


    module = _load_config_module(args.config_module)
    strategy = getattr(module, "ASSET_RUN_STRATEGY", AssetRunStrategyCfg())
    _validate_strategy(strategy)


    if args.stage == "pre-made":
        return _run_premade(
            module,
            max_enumerate=args.max_enumerate,
            output_dir=args.output_dir,
        )

    source_path = args.source_path or getattr(module, "POST_MUTATE_SOURCE_TOPOLOGY_PATH", None)
    if source_path is None:
        parser.error("--source-path is required for post-mutate; select a pre-made topology bundle.")
    if args.output_dir is not None:
        parser.error("--output-dir is available only for the pre-made stage.")

    return _run_post_mutate(
        module,
        source_path=source_path,
        n_samples=args.n_samples,
        post_mutate_seed=args.post_mutate_seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
