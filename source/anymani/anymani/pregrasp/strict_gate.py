(
    'Single numerical admission predicate for MVP80 strict good-pregrasp. Judge '
    'reset-state quality only; do not use rotation reward, policy action, or '
    'learning results. Measure total angular speed during the first 0.2 s after '
    'writing q0=u0 and releasing freeze. A zero-action reset should not rotate '
    'spontaneously about the target axis, so gate norm(omega), not only the '
    'off-axis component.'
)

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .good_catalog import GOOD_PREGRASP_TOP_K, GoodPregraspEntry, GoodPregraspMetrics


@dataclass(frozen=True)
class StrictGoodPregraspGate:
    'Combine geometry, penetration, cold-reset stability, and PALM-support hard gates.'

    joint_margin_fraction_min: float = 0.10  # Minimum normalized margin from each active joint to its nearest limit.
    tip_center_distance_m_max: float = 0.10  # Maximum thumb-plus-two-non-thumb TIP distance to cube center, m.
    sector_min_deg: float = 30.0  # Minimum in-plane pair angle among three fingers, degrees.
    penetration_depth_m_max: float = 0.0005  # Maximum illegal penetration at initialization/cold reset, m.
    object_displacement_m_max: float = 0.005  # Maximum displacement from initial state over 1 s, m.
    object_tilt_deg_max: float = 10.0  # Maximum angle between object z and hand z, degrees.
    peak_linear_velocity_m_s_max: float = 0.25  # Peak linear speed in first 0.2 s, m/s.
    peak_angular_velocity_rad_s_max: float = 2.0  # Peak total angular speed in first 0.2 s, rad/s.
    palm_contact_fraction_min: float = 0.50  # PALM contact fraction during final 0.5 s of policy samples.

    def __post_init__(self) -> None:
        'Reject non-finite/negative thresholds and invalid fraction ranges.'

        values = tuple(float(value) for value in self.to_dict().values())
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("strict good-pregrasp thresholds must be finite and non-negative")
        if not 0.0 <= self.joint_margin_fraction_min <= 0.5:
            raise ValueError("joint margin threshold must lie in [0,0.5]")
        if not 0.0 <= self.palm_contact_fraction_min <= 1.0:
            raise ValueError("palm contact threshold must lie in [0,1]")

    def to_dict(self) -> dict[str, float]:
        'Return stable threshold mapping for generation identity.'

        return {
            "joint_margin_fraction_min": self.joint_margin_fraction_min,
            "tip_center_distance_m_max": self.tip_center_distance_m_max,
            "sector_min_deg": self.sector_min_deg,
            "penetration_depth_m_max": self.penetration_depth_m_max,
            "object_displacement_m_max": self.object_displacement_m_max,
            "object_tilt_deg_max": self.object_tilt_deg_max,
            "peak_linear_velocity_m_s_max": self.peak_linear_velocity_m_s_max,
            "peak_angular_velocity_rad_s_max": self.peak_angular_velocity_rad_s_max,
            "palm_contact_fraction_min": self.palm_contact_fraction_min,
        }

    @property
    def digest(self) -> str:
        'Return SHA-256 identity covering thresholds and total-angular-speed semantics.'

        payload: dict[str, Any] = {
            "schema_version": "1.0.0",
            "angular_velocity_kind": "total_l2",
            "thresholds": self.to_dict(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def violations(self, metrics: GoodPregraspMetrics) -> tuple[str, ...]:
        'Return every hard gate violated by a candidate; empty tuple means strict pass.'

        failures: list[str] = []
        if metrics.joint_limit_margin_fraction < self.joint_margin_fraction_min:
            failures.append("joint_margin")
        if max(metrics.envelope_tip_center_distance_m) > self.tip_center_distance_m_max:
            failures.append("tip_center_distance")
        if metrics.envelope_sector_min_deg < self.sector_min_deg:
            failures.append("sector")
        if metrics.penetration_depth_max_m > self.penetration_depth_m_max:
            failures.append("penetration")
        if metrics.object_displacement_max_m > self.object_displacement_m_max:
            failures.append("displacement")
        if metrics.object_tilt_max_deg > self.object_tilt_deg_max:
            failures.append("tilt")
        if metrics.peak_linear_velocity_m_s > self.peak_linear_velocity_m_s_max:
            failures.append("peak_linear_velocity")
        if metrics.peak_angular_velocity_rad_s is None:
            failures.append("missing_total_angular_velocity")
        elif metrics.peak_angular_velocity_rad_s > self.peak_angular_velocity_rad_s_max:
            failures.append("peak_angular_velocity")
        if metrics.palm_contact_fraction < self.palm_contact_fraction_min:
            failures.append("palm_contact_fraction")
        return tuple(failures)

    def accepts(self, metrics: GoodPregraspMetrics) -> bool:
        'Return True only when all nine strict conditions pass.'

        return not self.violations(metrics)

    def validate_entry(self, entry: GoodPregraspEntry) -> None:
        'Require all eight members of a schema-3 Top-8 entry to pass strict gate.'

        rejected = [(member.rank, self.violations(member.metrics)) for member in entry.members]
        rejected = [(rank, violations) for rank, violations in rejected if violations]
        if rejected:
            raise ValueError(f"strict good-pregrasp entry contains rejected Top-8 members: {rejected}")


MVP80_STRICT_GOOD_PREGRASP_GATE = StrictGoodPregraspGate()


def top8_publication_indices(candidate_counts: Sequence[int], *, allow_partial: bool = False) -> tuple[int, ...]:
    (
        'Choose asset rows based on complete Top-8 counts. Default requires the whole '
        'batch ready; explicit partial preparation may return only assets with at '
        'least eight strict candidates. Preserve original asset axis for '
        'source/physical/routing identity. The completed entry still undergoes schema '
        'and per-member strict validation.'
    )

    if any(count < 0 for count in candidate_counts):
        raise ValueError("strict candidate counts must be non-negative")
    ready = tuple(index for index, count in enumerate(candidate_counts) if count >= GOOD_PREGRASP_TOP_K)
    return ready if allow_partial or len(ready) == len(candidate_counts) else ()


__all__ = ["MVP80_STRICT_GOOD_PREGRASP_GATE", "StrictGoodPregraspGate", "top8_publication_indices"]
