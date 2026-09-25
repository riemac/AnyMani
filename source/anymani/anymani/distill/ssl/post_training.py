"""Separate configuration for fixed-measure post-training evaluation."""


from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from omegaconf import MISSING, OmegaConf

from .contracts import build_runtime
from .runtime.pretrainer import ExecutionPrecisionCfg

EVALUATION_SCHEMA_VERSION = "1.0.0"
"""Stable schema for independent evaluation settings and results."""


@dataclass(frozen=True)
class EvaluationCfg:


    q_per_asset: int = 64
    assets_per_minibatch: int = 2
    q_per_asset_per_minibatch: int = 2
    final_ablations: tuple[str, ...] = (
        "query_only",
        "same_asset_q_shuffle",
        "cross_asset_shuffle",
        "joint_token_shuffle",
    )
    bootstrap_replicates: int = 2_000
    evaluation_seed_offset: int = 2_000_003
    bootstrap_seed_offset: int = 4_000_003
    max_resident_assets: int = 64
    device: str = "cuda:0"
    dtype: str = "float32"
    source_cache_root: str = "logs/ssl/_cache/geometry_source/v2"
    source_cache_mode: str = "read-write"
    z_compression_ranks: tuple[int, ...] = (32, 64, 96, 128)
    execution: ExecutionPrecisionCfg = field(default_factory=ExecutionPrecisionCfg)

    def __post_init__(self) -> None:


        object.__setattr__(self, "final_ablations", tuple(self.final_ablations))
        object.__setattr__(self, "z_compression_ranks", tuple(self.z_compression_ranks))
        counts = (
            self.q_per_asset,
            self.assets_per_minibatch,
            self.q_per_asset_per_minibatch,
            self.bootstrap_replicates,
            self.max_resident_assets,
        )
        offsets = (
            self.evaluation_seed_offset,
            self.bootstrap_seed_offset,
        )
        if min(counts) < 1 or not self.final_ablations:
            raise ValueError("evaluation q/batch/bootstrap/resource budgets and ablations must be non-empty")
        if self.z_compression_ranks not in {(), (32, 64, 96, 128)}:
            raise ValueError("Z compression ranks must be disabled or exactly 32/64/96/128")
        if min(offsets) < 1 or len(set(offsets)) != len(offsets):
            raise ValueError("evaluation seed offsets must be positive and distinct")
        if self.max_resident_assets < self.assets_per_minibatch:
            raise ValueError("evaluation max_resident_assets must cover one asset minibatch")
        _validate_cuda_float32(self.device, self.dtype, role="evaluation")
        _validate_source_cache(self.source_cache_root, self.source_cache_mode, role="evaluation")


def _validate_cuda_float32(device: str, dtype: str, *, role: str) -> None:


    if not (device == "cuda" or (device.startswith("cuda:") and device[5:].isdigit())):
        raise ValueError(f"{role} device must be 'cuda' or 'cuda:<index>'")
    if dtype != "float32":
        raise ValueError(f"current Warp {role} path requires dtype='float32'")


def _validate_source_cache(root: str, mode: str, *, role: str) -> None:


    if mode not in {"readonly", "read-write", "off"}:
        raise ValueError(f"{role} source_cache_mode is invalid")
    if mode != "off" and not root:
        raise ValueError(f"{role} source_cache_root is required unless cache mode is off")


class EvaluationRun:


    def __init__(self, config: EvaluationRunCfg) -> None:


        self.config = config

    def resolve_output_dir(self, override: Path | None = None) -> Path:


        output_dir = override
        if output_dir is None:
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            output_dir = Path(self.config.output_dir) / self.config.experiment_name / timestamp
        return output_dir.expanduser().resolve(strict=False)

    def prepare_output_dir(self, override: Path | None = None) -> Path:


        output_dir = self.resolve_output_dir(override)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir


@dataclass(frozen=True)
class EvaluationRunCfg:


    runtime_type: ClassVar[type[EvaluationRun]] = EvaluationRun
    output_dir: str = "logs/ssl"
    experiment_name: str = "canonical_multi_anchor_gaussian_evaluation"
    checkpoint: str = ""
    analyses: tuple[str, ...] = ()
    compression_basis: str = ""
    seed: int = 20260813
    deterministic_algorithms: bool = True

    def __post_init__(self) -> None:


        if not self.output_dir or not self.experiment_name or self.seed < 0:
            raise ValueError("evaluation run requires output identity and non-negative seed")
        object.__setattr__(self, "analyses", tuple(self.analyses))
        unknown = set(self.analyses) - {"ablations", "compression"}
        if unknown or len(set(self.analyses)) != len(self.analyses):
            raise ValueError(f"evaluation analyses must be unique registered names, got {sorted(unknown)}")

    def validate_inputs(self) -> None:


        if not self.checkpoint:
            raise ValueError("evaluation requires --checkpoint")
        if "compression" not in self.analyses and self.compression_basis:
            raise ValueError("compression_basis is only valid with --analysis compression")


class EmbodimentEvaluation:


    def __init__(self, config: EmbodimentEvaluationCfg, *, output_dir: Path | None = None) -> None:


        self.config = config
        self.output_dir = output_dir

    def run(self) -> Path:


        self.config.validate_composed()
        data = build_runtime(self.config.data)
        method = build_runtime(self.config.method)
        run = build_runtime(self.config.run)
        from .runtime.post_training import evaluate_checkpoint

        return evaluate_checkpoint(
            data=data,
            method=method,
            config=self.config.evaluation,
            run=run,
            output_dir_override=self.output_dir,
            resolved_config=resolved_post_training_config_dict(self.config),
        )


@dataclass(frozen=True)
class EmbodimentEvaluationCfg:


    schema_version: str = EVALUATION_SCHEMA_VERSION
    data: Any = MISSING
    method: Any = MISSING
    evaluation: EvaluationCfg = MISSING
    run: Any = MISSING

    def validate_composed(self) -> None:


        _validate_root(self, schema=EVALUATION_SCHEMA_VERSION, roles=("data", "method", "run"))
        if not isinstance(self.evaluation, EvaluationCfg):
            raise TypeError("evaluation root requires a concrete EvaluationCfg")
        self.run.validate_inputs()


def _validate_root(config: Any, *, schema: str, roles: tuple[str, ...]) -> None:


    if config.schema_version != schema:
        raise ValueError(f"post-training schema must be exactly {schema}")
    missing = tuple(
        role for role in roles if getattr(config, role) == MISSING or getattr(config, role) == "???"
    )
    if missing:
        raise ValueError(f"post-training config is missing component roles: {missing}")
    invalid = tuple(
        role for role in roles if not callable(getattr(type(getattr(config, role)), "runtime_type", None))
    )
    if invalid:
        raise TypeError(f"post-training roles lack runtime_type bindings: {invalid}")


def resolved_post_training_config_dict(config: Any) -> dict[str, Any]:


    container = OmegaConf.to_container(OmegaConf.structured(config), resolve=True)
    if not isinstance(container, dict):
        raise TypeError("resolved post-training config must be a mapping")
    return {str(key): value for key, value in container.items()}


__all__ = [
    "EVALUATION_SCHEMA_VERSION",
    "EmbodimentEvaluation",
    "EmbodimentEvaluationCfg",
    "EvaluationCfg",
    "EvaluationRun",
    "EvaluationRunCfg",
    "resolved_post_training_config_dict",
]
