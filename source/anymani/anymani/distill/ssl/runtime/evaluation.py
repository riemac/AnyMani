r"""多锚点方法的固定 validation、诊断和 checkpoint selection 声明。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from .sampling import OnlineSamplingCfg


class MultiAnchorEvaluation:
    r"""保存固定 evaluation 协议；具体前向由 Trainer 调用 method 完成。"""

    def __init__(self, config: MultiAnchorEvaluationCfg) -> None:
        r"""绑定 validation/q-bank/ablation 数值，不物化任何资产。"""

        self.config = config

    def validation_sampling(self, *, trainer_sampling: Any, run_seed: int, asset_count: int) -> OnlineSamplingCfg:
        r"""构造固定 held-out bank 的独立 q schedule，不复用 train cursor。"""

        if asset_count < 1:
            raise ValueError("validation sampling requires at least one held-out asset")
        return OnlineSamplingCfg(
            epochs=1,
            q_per_asset_per_epoch=self.config.q_per_asset,
            assets_per_minibatch=min(trainer_sampling.assets_per_minibatch, asset_count),
            q_per_asset_per_minibatch=trainer_sampling.q_per_asset_per_minibatch,
            shuffle_assets=False,
            seed=run_seed + self.config.validation_seed_offset,
        )

    def selection_baseline(self, metrics: dict[str, float]) -> dict[str, float]:
        r"""只保留 checkpoint selection 声明的初始化归一化分母。"""

        missing = set(self.config.selection_metrics) - metrics.keys()
        if missing:
            raise ValueError(f"validation metrics lack selection terms: {sorted(missing)}")
        baseline = {name: float(metrics[name]) for name in self.config.selection_metrics}
        if any(value <= 0.0 for value in baseline.values()):
            raise FloatingPointError("initial validation selection metrics must be positive")
        return baseline

    def normalized_score(self, metrics: dict[str, float], baseline: dict[str, float]) -> float:
        r"""按 initialization-normalized metrics 等权计算 promotion score。"""

        return sum(metrics[name] / baseline[name] for name in self.config.selection_metrics) / len(
            self.config.selection_metrics
        )

    def require_ablation_contract(self, reported: tuple[str, ...]) -> None:
        r"""拒绝 YAML 声明与 concrete evaluator 实际执行的 ablation 集漂移。"""

        if reported != self.config.final_ablations:
            raise ValueError("evaluation final_ablations do not match the concrete multi-anchor evaluator")

    @property
    def validation_seed(self) -> int:
        r"""返回相对 run root seed 的 validation domain offset。"""

        return self.config.validation_seed_offset

    @property
    def q_bank_seed(self) -> int:
        r"""返回相对 run root seed 的 training-morphology q-bank offset。"""

        return self.config.training_q_bank_seed_offset

    @property
    def bootstrap_seed(self) -> int:
        r"""返回相对 run root seed 的 paired bootstrap offset。"""

        return self.config.bootstrap_seed_offset


@dataclass(frozen=True)
class MultiAnchorEvaluationCfg:
    r"""当前 method 的 held-out bank、评估 cadence、选择指标和最终消融。"""

    runtime_type: ClassVar[type[MultiAnchorEvaluation]] = MultiAnchorEvaluation
    q_per_asset: int = 64
    every_optimizer_updates: int = 250
    selection_metrics: tuple[str, ...] = ("density", "kappa", "derived_field")
    final_ablations: tuple[str, ...] = (
        "query_only",
        "same_asset_q_shuffle",
        "cross_asset_shuffle",
        "first_order_zero",
        "first_order_joint_shuffle",
        "first_order_sign_flip",
    )
    bootstrap_replicates: int = 2_000
    validation_seed_offset: int = 1_000_003
    bootstrap_seed_offset: int = 2_000_003
    training_q_bank_seed_offset: int = 3_000_003

    def __post_init__(self) -> None:
        r"""规范 YAML sequences，并验证当前可执行 evaluation 协议。"""

        object.__setattr__(self, "selection_metrics", tuple(self.selection_metrics))
        object.__setattr__(self, "final_ablations", tuple(self.final_ablations))
        if self.q_per_asset < 1 or self.every_optimizer_updates < 1 or self.bootstrap_replicates < 1:
            raise ValueError("evaluation q/cadence/bootstrap budgets must be positive")
        if not self.selection_metrics or not self.final_ablations:
            raise ValueError("evaluation requires selection metrics and final ablations")
        offsets = (
            self.validation_seed_offset,
            self.bootstrap_seed_offset,
            self.training_q_bank_seed_offset,
        )
        if min(offsets) < 1 or len(set(offsets)) != len(offsets):
            raise ValueError("evaluation seed offsets must be positive and distinct")


__all__ = ["MultiAnchorEvaluation", "MultiAnchorEvaluationCfg"]
