r"""在线程序化监督的 Trainer 配置与最高 fit runtime。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from .sampling import OnlineSamplingCfg


@dataclass(frozen=True)
class AdamWCfg:
    r"""当前 canonical learned parameters 的 AdamW 更新规则。"""

    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4

    def __post_init__(self) -> None:
        r"""拒绝非正学习率与负 weight decay。"""

        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("AdamW learning rate must be positive and weight decay non-negative")


class EmbodimentPretrainTrainer:
    r"""拥有资产/q 在线日程、梯度累计、optimizer update 和 phase 调度。"""

    def __init__(self, config: EmbodimentPretrainTrainerCfg) -> None:
        r"""保存训练与资源配置；构造阶段不创建模型、optimizer 或 CUDA state。"""

        self.config = config

    def fit(
        self,
        *,
        data: Any,
        method: Any,
        evaluation: Any,
        run: Any,
        output_dir_override: Path | None,
        resolved_config: dict[str, Any],
    ) -> Path:
        r"""把五个 role runtime 交给显式 lifecycle 内核。"""

        from .lifecycle import fit_embodiment_pretrain

        return fit_embodiment_pretrain(
            trainer=self,
            data=data,
            method=method,
            evaluation=evaluation,
            run=run,
            output_dir_override=output_dir_override,
            resolved_config=resolved_config,
        )


@dataclass(frozen=True)
class EmbodimentPretrainTrainerCfg:
    r"""在线 sampling/update 算法、optimizer、设备资源和记录 cadence。"""

    runtime_type: ClassVar[type[EmbodimentPretrainTrainer]] = EmbodimentPretrainTrainer
    sampling: OnlineSamplingCfg = field(default_factory=OnlineSamplingCfg)
    optimizer: AdamWCfg = field(default_factory=AdamWCfg)
    gradient_accumulation_steps: int = 4  # 最后一个 update group 可以少于该值
    max_gradient_norm: float = 10.0
    max_resident_assets: int = 20  # 资源上限，不改变全资产 shuffle 的统计顺序
    device: str = "cuda:0"
    dtype: str = "float32"
    log_every_updates: int = 10
    checkpoint_every_updates: int = 1_000
    run_safety_step_limit: int = 30_000

    def __post_init__(self) -> None:
        r"""验证 update、设备与记录轴，不要求 epoch minibatches 可整除 accumulation。"""

        counts = (
            self.gradient_accumulation_steps,
            self.max_resident_assets,
            self.log_every_updates,
            self.checkpoint_every_updates,
            self.run_safety_step_limit,
        )
        if min(counts) < 1 or self.max_gradient_norm <= 0.0:
            raise ValueError("trainer update/resource/cadence values must be positive")
        if not (self.device == "cuda" or (self.device.startswith("cuda:") and self.device[5:].isdigit())):
            raise ValueError("embodiment pretraining device must be 'cuda' or 'cuda:<index>'")
        if self.dtype != "float32":
            raise ValueError("current Warp online supervision requires trainer dtype='float32'")


__all__ = ["AdamWCfg", "EmbodimentPretrainTrainer", "EmbodimentPretrainTrainerCfg"]
