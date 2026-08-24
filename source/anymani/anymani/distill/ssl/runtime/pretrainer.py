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

    def selection_baseline(self, metrics: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        r"""按 validation suite 独立冻结三项重建指标的初始化尺度。"""

        if not metrics:
            raise ValueError("validation selection requires at least one named suite")
        baseline: dict[str, dict[str, float]] = {}
        for suite_name, suite_metrics in metrics.items():
            missing = set(self.config.validation.selection_metrics) - suite_metrics.keys()
            if missing:
                raise ValueError(f"validation suite {suite_name!r} lacks selection terms: {sorted(missing)}")
            baseline[suite_name] = {
                name: float(suite_metrics[name]) for name in self.config.validation.selection_metrics
            }
        if any(value <= 0.0 for suite in baseline.values() for value in suite.values()):
            raise FloatingPointError("initial validation selection metrics must be positive")
        return baseline

    def normalized_validation_score(
        self,
        metrics: dict[str, dict[str, float]],
        baseline: dict[str, dict[str, float]],
    ) -> float:
        r"""先对三项重建指标等权，再对 validation suites 等权形成 promotion score。"""

        if set(metrics) != set(baseline):
            raise ValueError("validation metrics and initialization baseline suites do not match")
        suite_scores = [
            sum(
                metrics[suite_name][name] / suite_baseline[name]
                for name in self.config.validation.selection_metrics
            )
            / len(self.config.validation.selection_metrics)
            for suite_name, suite_baseline in baseline.items()
        ]
        return sum(suite_scores) / len(suite_scores)


@dataclass(frozen=True)
class ValidationCfg:
    r"""训练中固定 validation bank、执行 cadence 与 best-checkpoint 选择协议。"""

    q_per_asset: int = 64
    assets_per_minibatch: int = 2
    q_per_asset_per_minibatch: int = 2
    every_epochs: int = 8
    selection_metrics: tuple[str, ...] = ("density", "kappa", "derived_field")
    seed_offset: int = 1_000_003

    def __post_init__(self) -> None:
        r"""验证固定 bank 的显式 q/batch 轴和三项 selection 指标。"""

        object.__setattr__(self, "selection_metrics", tuple(self.selection_metrics))
        counts = (
            self.q_per_asset,
            self.assets_per_minibatch,
            self.q_per_asset_per_minibatch,
            self.every_epochs,
            self.seed_offset,
        )
        if min(counts) < 1 or not self.selection_metrics:
            raise ValueError("validation q/batch/cadence/seed values and selection metrics must be non-empty")


@dataclass(frozen=True)
class FinalEvaluationCfg:
    r"""冻结 best checkpoint 后的 unseen-suite、独立 q-bank 与消融报告协议。"""

    q_per_asset: int = 64
    assets_per_minibatch: int = 2
    q_per_asset_per_minibatch: int = 2
    final_ablations: tuple[str, ...] = (
        "query_only",
        "same_asset_q_shuffle",
        "cross_asset_shuffle",
        "first_order_zero",
        "first_order_joint_shuffle",
        "first_order_sign_flip",
    )
    bootstrap_replicates: int = 2_000
    evaluation_seed_offset: int = 2_000_003
    training_q_bank_seed_offset: int = 3_000_003
    bootstrap_seed_offset: int = 4_000_003

    def __post_init__(self) -> None:
        r"""验证冻结评估预算、消融集合与互不重叠的随机域。"""

        object.__setattr__(self, "final_ablations", tuple(self.final_ablations))
        counts = (
            self.q_per_asset,
            self.assets_per_minibatch,
            self.q_per_asset_per_minibatch,
            self.bootstrap_replicates,
        )
        offsets = (
            self.evaluation_seed_offset,
            self.training_q_bank_seed_offset,
            self.bootstrap_seed_offset,
        )
        if min(counts) < 1 or not self.final_ablations:
            raise ValueError("final evaluation q/batch/bootstrap budgets and ablations must be non-empty")
        if min(offsets) < 1 or len(set(offsets)) != len(offsets):
            raise ValueError("final evaluation seed offsets must be positive and distinct")


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
    validation: ValidationCfg = field(default_factory=ValidationCfg)
    final_evaluation: FinalEvaluationCfg = field(default_factory=FinalEvaluationCfg)
    optimizer: AdamWCfg = field(default_factory=AdamWCfg)
    max_gradient_norm: float = 10.0
    max_resident_assets: int = 64  # 首个 preset 恰好驻留一个 64-asset 训练 minibatch
    device: str = "cuda:0"
    dtype: str = "float32"
    checkpoint_every_epochs: int = 8

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
    "FinalEvaluationCfg",
    "ValidationCfg",
]
