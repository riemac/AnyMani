r"""Pure LEAP-right PPO的Critic-first证据门与干预顺序。

本模块只消费已记录的per-asset advantage/return/value/gradient统计，不运行optimizer，也不自动改变训练。它把
Learning Forensics顺序编码成可审计建议：先保证value-normalizer purity，再处理Actor advantage尺度；Actor
PCGrad要求健康Critic和可靠full gradients，PopArt/FairGrad-Critic只处理修复后仍持续的Critic失配。
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class CriticCheckpointEvidence:
    r"""一个checkpoint的逐资产Critic尺度、拟合质量与稀疏gradient proxy。"""

    update: int
    return_target_std: tuple[float, ...]  # 每资产物理return target标准差
    explained_variance: tuple[float, ...]  # 每资产$1-\mathrm{MSE}/\mathrm{Var}(R)$
    advantage_std: tuple[float, ...] = ()  # 每资产Actor advantage标准差；用于scale calibration而非Critic健康替身
    critic_gradient_norm: tuple[float, ...] = ()  # head-level proxy $\|g_i^c\|_2$
    critic_top1_norm_fraction: float | None = None  # $\max_i\|g_i\|/\sum_i\|g_i\|$
    actor_negative_pair_fraction: float | None = None  # Actor head-gradient余弦<0的pair比例
    gradient_probe_seconds: float | None = None
    epoch_total_seconds: float | None = None
    value_normalizer_pure: bool = True  # diagnostic/readout不得修改running value moments
    actor_full_gradient_reliable: bool = False  # 独立rollout/replica halves已确认Actor gradient方向可复现
    critic_full_gradient_reliable: bool = False  # FairGrad只能消费已确认可靠的完整Critic task gradients

    def __post_init__(self) -> None:
        r"""拒绝空资产轴、非有限值、错位gradient轴与非法比例/时间。"""

        if self.update < 1 or not self.return_target_std or len(self.return_target_std) != len(self.explained_variance):
            raise ValueError("critic checkpoint evidence requires aligned non-empty asset arrays")
        values = (*self.return_target_std, *self.explained_variance, *self.advantage_std, *self.critic_gradient_norm)
        if not all(math.isfinite(value) for value in values) or any(value < 0.0 for value in self.return_target_std):
            raise ValueError("critic checkpoint arrays must be finite and return scales non-negative")
        if self.critic_gradient_norm and len(self.critic_gradient_norm) != len(self.return_target_std):
            raise ValueError("critic gradient norms must align with the asset axis")
        if self.advantage_std and len(self.advantage_std) != len(self.return_target_std):
            raise ValueError("advantage standard deviations must align with the asset axis")
        if any(value < 0.0 for value in self.advantage_std):
            raise ValueError("advantage standard deviations must be non-negative")
        for name, value in (
            ("critic_top1_norm_fraction", self.critic_top1_norm_fraction),
            ("actor_negative_pair_fraction", self.actor_negative_pair_fraction),
        ):
            if value is not None and (not math.isfinite(value) or not 0.0 <= value <= 1.0):
                raise ValueError(f"{name} must lie in [0,1]")
        if self.gradient_probe_seconds is not None and (
            not math.isfinite(self.gradient_probe_seconds) or self.gradient_probe_seconds < 0.0
        ):
            raise ValueError("gradient probe seconds must be finite and non-negative")
        if self.epoch_total_seconds is not None and (
            not math.isfinite(self.epoch_total_seconds) or self.epoch_total_seconds <= 0.0
        ):
            raise ValueError("epoch total seconds must be finite and positive")

    @property
    def critic_gradient_span(self) -> float | None:
        r"""返回非零head-gradient norm的max/min；存在零贡献时为正无穷。"""

        if not self.critic_gradient_norm:
            return None
        maximum = max(self.critic_gradient_norm)
        positive = tuple(value for value in self.critic_gradient_norm if value > 1.0e-12)
        if maximum <= 1.0e-12:
            return 1.0  # 全零表示probe无信号，不伪造尺度分裂
        return math.inf if len(positive) != len(self.critic_gradient_norm) else maximum / min(positive)

    @property
    def return_target_std_span(self) -> float:
        r"""返回非零return std的max/min；部分零方差资产使跨度为正无穷。"""

        maximum = max(self.return_target_std)
        positive = tuple(value for value in self.return_target_std if value > 1.0e-12)
        if maximum <= 1.0e-12:
            return 1.0
        return math.inf if len(positive) != len(self.return_target_std) else maximum / min(positive)

    @property
    def advantage_std_span(self) -> float | None:
        r"""返回逐资产Actor advantage std的max/min；未记录时为None。"""

        if not self.advantage_std:
            return None
        maximum = max(self.advantage_std)
        positive = tuple(value for value in self.advantage_std if value > 1.0e-12)
        if maximum <= 1.0e-12:
            return 1.0
        return math.inf if len(positive) != len(self.advantage_std) else maximum / min(positive)

    @property
    def explained_variance_quartile_means(self) -> tuple[float, float]:
        r"""返回排序后bottom/top quartile普通均值，至少各取一项。"""

        ordered = sorted(self.explained_variance)
        width = max(1, len(ordered) // 4)
        return statistics.fmean(ordered[:width]), statistics.fmean(ordered[-width:])

    @property
    def probe_overhead_fraction(self) -> float | None:
        r"""返回probe wall/epoch wall；缺任一计时则为None。"""

        if self.gradient_probe_seconds is None or self.epoch_total_seconds is None:
            return None
        return self.gradient_probe_seconds / self.epoch_total_seconds


@dataclass(frozen=True)
class CriticInterventionDecision:
    r"""基于已有阶段与证据形成的一项非执行性建议。"""

    action: str  # fix purity / per-asset norm / PopArt / FairGrad-c / PCGrad-a / stop / none
    reasons: tuple[str, ...]
    latest_update: int


def recommend_critic_intervention(
    checkpoints: Sequence[CriticCheckpointEvidence],
    *,
    applied_interventions: Sequence[str] = (),
    critic_healthy: bool = False,
) -> CriticInterventionDecision:
    r"""按预注册条件返回下一项Critic-first干预建议，不修改训练状态。

    Conditions:
        1. 最新证据显示value normalizer不纯，首先建议修复；
        2. 连续两个checkpoint的Actor advantage std span均$\ge10$，先建议逐资产归一化；
        3. 已归一化且Critic健康时，只有可靠full Actor gradients仍有$>25\%$负pair才建议PCGrad-a；
        4. Critic不健康且return/gradient/EV连续失配时先建议PopArt，可靠full Critic gradients仍失衡才建议FairGrad-c；
        5. 任何常驻gradient solver若实测开销$>20\%$ epoch wall，建议停止。
    """

    evidence = tuple(checkpoints)
    if not evidence:
        raise ValueError("critic intervention decision requires at least one checkpoint")
    if any(right.update <= left.update for left, right in zip(evidence, evidence[1:])):
        raise ValueError("critic checkpoint evidence must be strictly update-ordered")
    latest = evidence[-1]
    applied = set(applied_interventions)
    overhead = latest.probe_overhead_fraction
    if overhead is not None and overhead > 0.20 and applied & {"fairgrad_critic", "pcgrad_actor"}:
        return CriticInterventionDecision(
            action="stop_gradient_solver",
            reasons=(f"gradient solver overhead {overhead:.3f} exceeds 0.20",),
            latest_update=latest.update,
        )
    if not latest.value_normalizer_pure:
        return CriticInterventionDecision(
            action="fix_value_normalizer_purity",
            reasons=("diagnostic or readout path mutates value running moments",),
            latest_update=latest.update,
        )

    if "per_asset_advantage_normalization" not in applied and len(evidence) >= 2:
        persistent = True
        reasons: list[str] = []
        for checkpoint in evidence[-2:]:
            span = checkpoint.advantage_std_span
            persistent &= span is not None and span >= 10.0
            reasons.append(f"u{checkpoint.update}: advantage_std_span={span}")
        if persistent:
            return CriticInterventionDecision(
                action="per_asset_advantage_normalization",
                reasons=tuple(reasons),
                latest_update=latest.update,
            )

    if "per_asset_advantage_normalization" in applied and critic_healthy:
        negative = latest.actor_negative_pair_fraction
        if latest.actor_full_gradient_reliable and "pcgrad_actor" not in applied and negative is not None and negative > 0.25:
            return CriticInterventionDecision(
                action="pcgrad_actor",
                reasons=(f"reliable full Actor gradients retain negative-pair fraction {negative:.3f} > 0.25",),
                latest_update=latest.update,
            )
        return CriticInterventionDecision(
            action="none",
            reasons=("critic is healthy; Actor surgery lacks reliable persistent full-gradient conflict",),
            latest_update=latest.update,
        )

    persistent_critic_mismatch = False
    if len(evidence) >= 2:
        checks = []
        for checkpoint in evidence[-2:]:
            gradient_span = checkpoint.critic_gradient_span
            bottom, top = checkpoint.explained_variance_quartile_means
            checks.append(
                checkpoint.return_target_std_span >= 10.0
                and gradient_span is not None
                and gradient_span >= 10.0
                and bottom < 0.0
                and top > 0.2
            )
        persistent_critic_mismatch = all(checks)
    if "per_asset_advantage_normalization" in applied and "popart" not in applied and persistent_critic_mismatch:
        return CriticInterventionDecision(
            action="popart",
            reasons=("return scale, Critic gradient span and EV split persist after Actor scale calibration",),
            latest_update=latest.update,
        )

    if "popart" in applied and "fairgrad_critic" not in applied:
        span = latest.critic_gradient_span
        top1 = latest.critic_top1_norm_fraction
        if latest.critic_full_gradient_reliable and (
            (span is not None and span >= 10.0) or (top1 is not None and top1 > 0.5)
        ):
            return CriticInterventionDecision(
                action="fairgrad_critic",
                reasons=(f"post-normalization critic span={span}, top1_norm_fraction={top1}",),
                latest_update=latest.update,
            )
    return CriticInterventionDecision(action="none", reasons=("no registered intervention condition met",), latest_update=latest.update)


__all__ = [
    "CriticCheckpointEvidence",
    "CriticInterventionDecision",
    "recommend_critic_intervention",
]
