'Reduce recorded per-asset Critic and gradient evidence into non-executing review gates.'

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class CriticCheckpointEvidence:
    'Contract for Critic checkpoint evidence.'

    update: int
    return_target_std: tuple[float, ...]
    explained_variance: tuple[float, ...]
    advantage_std: tuple[float, ...] = ()
    critic_gradient_norm: tuple[float, ...] = ()
    critic_top1_norm_fraction: float | None = None
    actor_negative_pair_fraction: float | None = None
    gradient_probe_seconds: float | None = None
    epoch_total_seconds: float | None = None
    value_normalizer_pure: bool = True
    actor_full_gradient_reliable: bool = False
    critic_full_gradient_reliable: bool = False

    def __post_init__(self) -> None:
        'Validate the declared contract.'

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
        'Handle Critic gradient span.'

        if not self.critic_gradient_norm:
            return None
        maximum = max(self.critic_gradient_norm)
        positive = tuple(value for value in self.critic_gradient_norm if value > 1.0e-12)
        if maximum <= 1.0e-12:
            return 1.0
        return math.inf if len(positive) != len(self.critic_gradient_norm) else maximum / min(positive)

    @property
    def return_target_std_span(self) -> float:
        'Handle return target std span.'

        maximum = max(self.return_target_std)
        positive = tuple(value for value in self.return_target_std if value > 1.0e-12)
        if maximum <= 1.0e-12:
            return 1.0
        return math.inf if len(positive) != len(self.return_target_std) else maximum / min(positive)

    @property
    def advantage_std_span(self) -> float | None:
        'Handle advantage std span.'

        if not self.advantage_std:
            return None
        maximum = max(self.advantage_std)
        positive = tuple(value for value in self.advantage_std if value > 1.0e-12)
        if maximum <= 1.0e-12:
            return 1.0
        return math.inf if len(positive) != len(self.advantage_std) else maximum / min(positive)

    @property
    def explained_variance_quartile_means(self) -> tuple[float, float]:
        'Handle explained variance quartile means.'

        ordered = sorted(self.explained_variance)
        width = max(1, len(ordered) // 4)
        return statistics.fmean(ordered[:width]), statistics.fmean(ordered[-width:])

    @property
    def probe_overhead_fraction(self) -> float | None:
        'Handle probe overhead fraction.'

        if self.gradient_probe_seconds is None or self.epoch_total_seconds is None:
            return None
        return self.gradient_probe_seconds / self.epoch_total_seconds


@dataclass(frozen=True)
class CriticInterventionDecision:
    'Contract for Critic intervention decision.'

    action: str  # fix purity / per-asset norm / PopArt / FairGrad-c / PCGrad-a / stop / none
    reasons: tuple[str, ...]
    latest_update: int


def recommend_critic_intervention(
    checkpoints: Sequence[CriticCheckpointEvidence],
    *,
    applied_interventions: Sequence[str] = (),
    critic_healthy: bool = False,
) -> CriticInterventionDecision:
    'Handle recommend Critic intervention.'

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
