"""Online geometry teacher training with additive minibatch updates."""


from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from .sampling import OnlineSamplingCfg


@dataclass(frozen=True)
class AdamWCfg:


    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4

    def __post_init__(self) -> None:


        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("AdamW learning rate must be positive and weight decay non-negative")


@dataclass(frozen=True)
class ExecutionPrecisionCfg:


    teacher_dtype: str = "float32"
    parameter_dtype: str = "float32"
    model_autocast_dtype: str = "bfloat16"
    loss_dtype: str = "float32"
    fairgrad_accumulation_dtype: str = "float64"
    allow_tf32: bool = False
    compile_enabled: bool = True
    compile_mode: str = "reduce-overhead"

    def __post_init__(self) -> None:


        if self.teacher_dtype != "float32" or self.parameter_dtype != "float32" or self.loss_dtype != "float32":
            raise ValueError("Geometry SSL teacher, parameters and loss reduction must remain float32")
        if self.model_autocast_dtype not in {"bfloat16", "float32", "tf32"}:
            raise ValueError("model_autocast_dtype must be bfloat16, tf32, or float32")
        if self.fairgrad_accumulation_dtype != "float64":
            raise ValueError("FairGrad accumulation must remain float64")
        if self.compile_mode not in {"default", "reduce-overhead", "max-autotune"}:
            raise ValueError("unsupported torch.compile mode")


class EmbodimentPretrainTrainer:


    def __init__(self, config: EmbodimentPretrainTrainerCfg) -> None:


        self.config = config

    def fit(
        self,
        *,
        data: Any,
        method: Any,
        run: Any,
        output_dir_override: Path | None,
        resolved_config: dict[str, Any],
    ) -> Path:


        from .lifecycle import fit_embodiment_pretrain

        return fit_embodiment_pretrain(
            trainer=self,
            data=data,
            method=method,
            run=run,
            output_dir_override=output_dir_override,
            resolved_config=resolved_config,
        )

@dataclass(frozen=True)
class EmbodimentPretrainTrainerCfg:


    runtime_type: ClassVar[type[EmbodimentPretrainTrainer]] = EmbodimentPretrainTrainer
    sampling: OnlineSamplingCfg = field(default_factory=OnlineSamplingCfg)
    max_epochs: int = 256
    num_minibatches: int = 4
    mini_epochs: int = 1
    microbatch_size: int = 64
    optimizer: AdamWCfg = field(default_factory=AdamWCfg)
    execution: ExecutionPrecisionCfg = field(default_factory=ExecutionPrecisionCfg)
    max_gradient_norm_per_group: float = 10.0
    device: str = "cuda:0"
    checkpoint_every_epochs: int = 32
    resource_profile: bool = False
    emit_compression_basis: bool = False
    compression_q_per_asset: int = 64
    compression_q_per_asset: int = 64

    def __post_init__(self) -> None:


        counts = (
            self.max_epochs,
            self.num_minibatches,
            self.mini_epochs,
            self.microbatch_size,
            self.checkpoint_every_epochs,
            self.compression_q_per_asset,
            self.compression_q_per_asset,
        )
        if min(counts) < 1 or self.max_gradient_norm_per_group <= 0.0:
            raise ValueError("trainer update/resource/cadence values must be positive")
        minibatch_size = (
            self.sampling.assets_per_minibatch * self.sampling.q_per_asset_per_minibatch
        )
        if minibatch_size % self.microbatch_size != 0:
            raise ValueError("microbatch_size must exactly divide the full training minibatch")
        if self.microbatch_size % self.sampling.q_per_asset_per_minibatch != 0:
            raise ValueError("microbatch_size must contain complete per-asset q blocks")
        if not (self.device == "cuda" or (self.device.startswith("cuda:") and self.device[5:].isdigit())):
            raise ValueError("embodiment pretraining device must be 'cuda' or 'cuda:<index>'")
        if self.device_window_assets < 1:
            raise ValueError("microbatch_size must contain at least one complete asset q block")
        if self.checkpoint_every_epochs > self.max_epochs or self.max_epochs % self.checkpoint_every_epochs != 0:
            raise ValueError("immutable checkpoint cadence must exactly divide max_epochs")

    @property
    def device_window_assets(self) -> int:


        return self.microbatch_size // self.sampling.q_per_asset_per_minibatch


__all__ = [
    "AdamWCfg",
    "EmbodimentPretrainTrainer",
    "EmbodimentPretrainTrainerCfg",
    "ExecutionPrecisionCfg",
]
