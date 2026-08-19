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

    def selection_baseline(
        self,
        metrics: dict[str, dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        r"""保存每条 validation suite 的初始化归一化分母。

        suite 轴不能在 baseline 前合并。若先按资产数扁平平均，扩大某条 suite 会
        同时改变统计精度和 checkpoint objective；这里对每条 suite 独立冻结三项指标。
        """

        if not metrics:
            raise ValueError("validation selection requires at least one named suite")
        baseline: dict[str, dict[str, float]] = {}
        for suite_name, suite_metrics in metrics.items():
            missing = set(self.config.selection_metrics) - suite_metrics.keys()
            if missing:
                raise ValueError(f"validation suite {suite_name!r} lacks selection terms: {sorted(missing)}")
            baseline[suite_name] = {
                name: float(suite_metrics[name]) for name in self.config.selection_metrics
            }  # 每个 suite 固定同一组 metric 分母，suite 间不共享尺度
        if any(value <= 0.0 for suite in baseline.values() for value in suite.values()):
            raise FloatingPointError("initial validation selection metrics must be positive")
        return baseline

    def normalized_score(
        self,
        metrics: dict[str, dict[str, float]],
        baseline: dict[str, dict[str, float]],
    ) -> float:
        r"""先对 metric 等权，再对 validation suite 等权计算 promotion score。

        对 suite $s$ 的归一化分数为
        $S_s=|M|^{-1}\sum_{m\in M}L_{s,m}/L^{(0)}_{s,m}$；最终
        $S=|S|^{-1}\sum_s S_s$。因此 suite 资产数不进入 checkpoint 权重。
        """

        if set(metrics) != set(baseline):
            raise ValueError("validation metrics and initialization baseline suites do not match")
        suite_scores = []
        for suite_name, suite_baseline in baseline.items():
            suite_metrics = metrics[suite_name]
            suite_scores.append(
                sum(suite_metrics[name] / suite_baseline[name] for name in self.config.selection_metrics)
                / len(self.config.selection_metrics)
            )
        return sum(suite_scores) / len(suite_scores)

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
