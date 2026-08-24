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
    r"""拥有资产/q 在线日程、显存分块、optimizer update 和训练 checkpoint。"""

    def __init__(self, config: EmbodimentPretrainTrainerCfg) -> None:
        r"""保存训练与资源配置；构造阶段不创建模型、optimizer 或 CUDA state。"""

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
        r"""把 data/method/run 与 Trainer 自身交给显式 lifecycle 内核。"""

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
    r"""在线 epoch、新 minibatch、全局复用遍数、显存切片与记录 cadence。

    canonical 数值锚点为 ``32 epochs × 4 minibatches × 64 assets × 8 q``，即生成
    65536 个不同 ``(asset,q)`` pairs。每个 512-pair minibatch 独立更新一次；
    ``microbatch_size=64`` 只切 forward/backward，不改变完整 minibatch 统计目标。
    """

    runtime_type: ClassVar[type[EmbodimentPretrainTrainer]] = EmbodimentPretrainTrainer
    sampling: OnlineSamplingCfg = field(default_factory=OnlineSamplingCfg)
    max_epochs: int = 32  # 外层训练回合上限；不表示完整资产 catalog 遍历
    num_minibatches: int = 4  # 每个 epoch 新生成的 minibatch 数
    mini_epochs: int = 1  # 对本 epoch 全部 minibatches 的完整遍历次数
    microbatch_size: int = 64  # 一次模型 forward/backward 的 $(asset,q)$ pair 数
    optimizer: AdamWCfg = field(default_factory=AdamWCfg)
    max_gradient_norm: float = 10.0
    max_resident_assets: int = 64  # 首个 preset 恰好驻留一个 64-asset 训练 minibatch
    device: str = "cuda:0"
    dtype: str = "float32"
    checkpoint_every_epochs: int = 1  # 每个 epoch 保留完整状态，支持事后曲线与 checkpoint 分析

    def __post_init__(self) -> None:
        r"""验证新数据预算、复用次数、设备资源与记录轴严格为正。"""

        counts = (
            self.max_epochs,
            self.num_minibatches,
            self.mini_epochs,
            self.microbatch_size,
            self.max_resident_assets,
            self.checkpoint_every_epochs,
        )
        if min(counts) < 1 or self.max_gradient_norm <= 0.0:
            raise ValueError("trainer update/resource/cadence values must be positive")
        if self.max_resident_assets < self.sampling.assets_per_minibatch:
            raise ValueError("max_resident_assets must cover one training asset minibatch")
        minibatch_size = (
            self.sampling.assets_per_minibatch * self.sampling.q_per_asset_per_minibatch
        )  # $B_{mb}=N_{asset}^{mb}N_q^{mb}$
        if minibatch_size % self.microbatch_size != 0:
            raise ValueError("microbatch_size must exactly divide the full training minibatch")
        if self.microbatch_size % self.sampling.q_per_asset_per_minibatch != 0:
            raise ValueError("microbatch_size must contain complete per-asset q blocks")
        if not (self.device == "cuda" or (self.device.startswith("cuda:") and self.device[5:].isdigit())):
            raise ValueError("embodiment pretraining device must be 'cuda' or 'cuda:<index>'")
        if self.dtype != "float32":
            raise ValueError("current Warp online supervision requires trainer dtype='float32'")


__all__ = [
    "AdamWCfg",
    "EmbodimentPretrainTrainer",
    "EmbodimentPretrainTrainerCfg",
]
