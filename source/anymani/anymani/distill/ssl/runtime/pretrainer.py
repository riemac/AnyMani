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


@dataclass(frozen=True)
class ExecutionPrecisionCfg:
    r"""Geometry teacher、learned model、loss/FairGrad 与 compile 的数值边界。

    几何 teacher 的 FK、Warp closest-face、barycentric、$\kappa$、Gaussian target 与 mask 固定
    FP32；learned encoder/readers 可在 BF16 autocast 中运行。参数与 AdamW master state 仍为 FP32，
    FairGrad 范数和点积仍以 FP64 累计，因此低精度只作用于 learned GEMM/activation。
    """

    teacher_dtype: str = "float32"  # Warp/几何真值执行精度
    parameter_dtype: str = "float32"  # 参数、AdamW master state 与持久化 retained state
    model_autocast_dtype: str = "bfloat16"  # learned forward 的 CUDA autocast dtype
    loss_dtype: str = "float32"  # objective numerator/denominator 与 private gradient dtype
    fairgrad_accumulation_dtype: str = "float64"  # shared task-gradient norm/dot 累计精度
    allow_tf32: bool = False  # 全局关闭，避免 teacher FP32 matmul 被透明降精度
    compile_enabled: bool = True  # 固定 64-pair learned region 的正式执行策略
    compile_mode: str = "reduce-overhead"  # CUDA graph/private-pool 适合固定 shape 热路径

    def __post_init__(self) -> None:
        r"""拒绝会改变 teacher 或 optimizer master-state 数值语义的组合。"""

        if self.teacher_dtype != "float32" or self.parameter_dtype != "float32" or self.loss_dtype != "float32":
            raise ValueError("Geometry SSL teacher, parameters and loss reduction must remain float32")
        if self.model_autocast_dtype not in {"bfloat16", "float32", "tf32"}:
            raise ValueError("model_autocast_dtype must be bfloat16, tf32, or float32")
        if self.fairgrad_accumulation_dtype != "float64":
            raise ValueError("FairGrad accumulation must remain float64")
        if self.compile_mode not in {"default", "reduce-overhead", "max-autotune"}:
            raise ValueError("unsupported torch.compile mode")


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

    canonical 数值锚点为 ``256 epochs × 4 minibatches × 64 assets × 8 q``，即生成
    524288 个不同 ``(asset,q)`` pairs。每个 512-pair minibatch 独立更新一次；
    ``microbatch_size=64`` 只切 forward/backward，不改变完整 minibatch 统计目标。
    """

    runtime_type: ClassVar[type[EmbodimentPretrainTrainer]] = EmbodimentPretrainTrainer
    sampling: OnlineSamplingCfg = field(default_factory=OnlineSamplingCfg)
    max_epochs: int = 256  # 1024 updates，恰好覆盖 8192-asset catalog 的 8 个完整 cycles
    num_minibatches: int = 4  # 每个 epoch 新生成的 minibatch 数
    mini_epochs: int = 1  # 对本 epoch 全部 minibatches 的完整遍历次数
    microbatch_size: int = 64  # 一次模型 forward/backward 的 $(asset,q)$ pair 数
    optimizer: AdamWCfg = field(default_factory=AdamWCfg)
    execution: ExecutionPrecisionCfg = field(default_factory=ExecutionPrecisionCfg)
    max_gradient_norm_per_group: float = 10.0  # shared/density-private/kappa-private 分别裁剪
    device: str = "cuda:0"
    checkpoint_every_epochs: int = 32  # 每个 8192-asset catalog cycle 保存一次 immutable full state
    resource_profile: bool = False  # 默认热路径禁止显存快照和显式 CUDA synchronize
    emit_compression_basis: bool = False  # 正式 snapshot 可在最终 train state 上发布 train-derived PCA basis
    compression_q_per_asset: int = 64  # train-derived basis 的固定 q-bank，正式评估与此保持同一测度
    compression_q_per_asset: int = 64  # train-derived basis 的固定 q-bank，正式评估与此保持同一测度

    def __post_init__(self) -> None:
        r"""验证新数据预算、复用次数、设备资源与记录轴严格为正。"""

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
        )  # $B_{mb}=N_{asset}^{mb}N_q^{mb}$
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
        r"""返回一个 64-pair stream unit 的资产数，正式配置为 $64/8=8$。"""

        return self.microbatch_size // self.sampling.q_per_asset_per_minibatch


__all__ = [
    "AdamWCfg",
    "EmbodimentPretrainTrainer",
    "EmbodimentPretrainTrainerCfg",
    "ExecutionPrecisionCfg",
]
