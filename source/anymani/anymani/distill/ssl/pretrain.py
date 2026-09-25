"""Command-line entry point for the registered N040 geometry pretraining run."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
import hashlib
from pathlib import Path

from anymani.distill.ssl.config_store import compose_pretrain_cfg
from anymani.distill.ssl.experiment import EmbodimentPretrain
from anymani.distill.ssl.experiments import DEFAULT_EXPERIMENT_NAME, load_experiment


def _build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(description="Run AnyMani embodiment geometry pretraining.")
    parser.add_argument(
        "--config",
        default=DEFAULT_EXPERIMENT_NAME,
        help="registered experiment name or path to a Python snapshot exporting EXPERIMENT",
    )

    parser.add_argument("--run_name", "--experiment_name", dest="experiment_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--dataset-manifest", type=Path, help="Use a newly generated SSL dataset manifest.")
    parser.add_argument("--dataset-sha256", help="Expected SHA-256 of --dataset-manifest.")
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--new_run", action="store_true", help="ignore matching incomplete runs and start a new run")
    parser.add_argument(
        "--allow_worktree_change",
        "--allow-worktree-change",
        action="store_true",
        help="allow an explicitly validated dirty-worktree source fix when resuming an incomplete run",
    )
    parser.add_argument(
        "--extend_completed_run",
        "--extend-completed-run",
        action="store_true",
        help="start an independent child run by extending a completed checkpoint to a larger max_epochs budget",
    )
    parser.add_argument(
        "--extension_source_package_version",
        "--extension-source-package-version",
        type=str,
        default=None,
        help="exact source checkpoint package version for an explicitly reviewed cross-release extension",
    )
    parser.add_argument("--source_cache_root", type=str, default=None)
    parser.add_argument("--source_cache_mode", choices=("auto", "readonly", "read-write", "off"), default=None)

    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--num_minibatches", type=int, default=None)
    parser.add_argument("--assets_per_minibatch", type=int, default=None)
    parser.add_argument("--q_per_asset_per_minibatch", type=int, default=None)
    parser.add_argument("--mini_epochs", type=int, default=None)
    parser.add_argument("--microbatch_size", type=int, default=None)

    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--max_gradient_norm_per_group", type=float, default=None)
    parser.add_argument("--checkpoint_every_epochs", type=int, default=None)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sampling_seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--shuffle_assets", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--deterministic_algorithms", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resource_profile", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--emit_compression_basis", action=argparse.BooleanOptionalAction, default=None)
    return parser


def _config_overrides(args: argparse.Namespace) -> tuple[str, ...]:

    field_paths = (
        ("experiment_name", "run.experiment_name"),
        ("output_dir", "run.output_dir"),
        ("resume_checkpoint", "run.resume_checkpoint"),
        ("new_run", "run.new_run"),
        ("allow_worktree_change", "run.allow_worktree_change"),
        ("extend_completed_run", "run.extend_completed_run"),
        ("extension_source_package_version", "run.extension_source_package_version"),
        ("source_cache_root", "run.source_cache_root"),
        ("source_cache_mode", "run.source_cache_mode"),
        ("max_epochs", "trainer.max_epochs"),
        ("num_minibatches", "trainer.num_minibatches"),
        ("assets_per_minibatch", "trainer.sampling.assets_per_minibatch"),
        ("q_per_asset_per_minibatch", "trainer.sampling.q_per_asset_per_minibatch"),
        ("mini_epochs", "trainer.mini_epochs"),
        ("microbatch_size", "trainer.microbatch_size"),
        ("learning_rate", "trainer.optimizer.learning_rate"),
        ("weight_decay", "trainer.optimizer.weight_decay"),
        ("max_gradient_norm_per_group", "trainer.max_gradient_norm_per_group"),
        ("checkpoint_every_epochs", "trainer.checkpoint_every_epochs"),
        ("device", "trainer.device"),
        ("shuffle_assets", "trainer.sampling.shuffle_assets"),
        ("deterministic_algorithms", "run.deterministic_algorithms"),
        ("resource_profile", "trainer.resource_profile"),
        ("emit_compression_basis", "trainer.emit_compression_basis"),
    )
    overrides = [f"{path}={getattr(args, field)}" for field, path in field_paths if getattr(args, field) is not None]

    if args.seed is not None:
        overrides.extend((f"run.seed={args.seed}", f"trainer.sampling.seed={args.seed}"))
    if args.sampling_seed is not None:
        overrides.append(f"trainer.sampling.seed={args.sampling_seed}")
    return tuple(overrides)


def main(argv: Sequence[str] | None = None) -> Path:

    args = _build_parser().parse_args(argv)
    preset = load_experiment(args.config)
    config = compose_pretrain_cfg(_config_overrides(args), config_ref=args.config)
    if args.dataset_sha256 is not None and args.dataset_manifest is None:
        raise ValueError("--dataset-sha256 requires --dataset-manifest")
    if args.dataset_manifest is not None:
        manifest = args.dataset_manifest.expanduser().resolve(strict=True)
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        if args.dataset_sha256 is not None and digest != args.dataset_sha256.lower():
            raise ValueError("Dataset manifest SHA-256 does not match --dataset-sha256")
        config = replace(config, data=replace(config.data, manifest=str(manifest), expected_sha256=digest))
    config.validate_composed()
    output_dir = EmbodimentPretrain(
        config,
        config_identity={
            "name": preset.name,
            "module": preset.module_name,
            "path": str(preset.path),
            "sha256": preset.config_sha256,
        },
    ).run()
    print(output_dir)
    return output_dir


if __name__ == "__main__":  # ``python -m anymani.distill.ssl.pretrain``
    main()


__all__ = ["main"]
