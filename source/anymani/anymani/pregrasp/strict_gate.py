r"""MVP80 strict good-pregrasp的唯一数值准入谓词。

该gate只判断reset初态质量，不读取rotation reward、policy action或后续学习表现。总角速度门测量写入
$q_0=u_0,T_{ho,0}$并解除冻结后的前0.2 s瞬态；因为zero-action reset不应自发绕目标轴旋转，使用
$\|\omega\|_2$而不是只看离轴分量。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from .good_catalog import GoodPregraspEntry, GoodPregraspMetrics


@dataclass(frozen=True)
class StrictGoodPregraspGate:
    r"""联合几何、penetration、cold-reset稳定性与PALM support硬门。"""

    joint_margin_fraction_min: float = 0.10  # active joint到最近limit的归一化余量
    tip_center_distance_m_max: float = 0.10  # thumb+2 non-thumb TIP到cube center最大距离，m
    sector_min_deg: float = 30.0  # 三指面内pair angle最小值，degree
    penetration_depth_m_max: float = 0.0005  # initial/cold-reset非法穿透，m
    object_displacement_m_max: float = 0.005  # 1 s相对初态最大位移，m
    object_tilt_deg_max: float = 10.0  # object z相对hand z最大倾角，degree
    peak_linear_velocity_m_s_max: float = 0.25  # 前0.2 s峰值线速度，m/s
    peak_angular_velocity_rad_s_max: float = 2.0  # 前0.2 s总角速度峰值，rad/s
    palm_contact_fraction_min: float = 0.50  # 后0.5 s PALM contact policy-sample占比

    def __post_init__(self) -> None:
        r"""拒绝non-finite、负阈值及无效fraction范围。"""

        values = tuple(float(value) for value in self.to_dict().values())
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("strict good-pregrasp thresholds must be finite and non-negative")
        if not 0.0 <= self.joint_margin_fraction_min <= 0.5:
            raise ValueError("joint margin threshold must lie in [0,0.5]")
        if not 0.0 <= self.palm_contact_fraction_min <= 1.0:
            raise ValueError("palm contact threshold must lie in [0,1]")

    def to_dict(self) -> dict[str, float]:
        r"""返回generation identity可直接嵌入的稳定阈值mapping。"""

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
        r"""返回阈值与total-angular语义的SHA-256身份。"""

        payload: dict[str, Any] = {
            "schema_version": "1.0.0",
            "angular_velocity_kind": "total_l2",
            "thresholds": self.to_dict(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def violations(self, metrics: GoodPregraspMetrics) -> tuple[str, ...]:
        r"""返回一个候选违反的全部硬门名称；空tuple即严格通过。"""

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
        r"""当且仅当九个strict conditions全部成立时返回True。"""

        return not self.violations(metrics)

    def validate_entry(self, entry: GoodPregraspEntry) -> None:
        r"""要求一个schema-3 entry的Top-8全部严格通过。"""

        rejected = [(member.rank, self.violations(member.metrics)) for member in entry.members]
        rejected = [(rank, violations) for rank, violations in rejected if violations]
        if rejected:
            raise ValueError(f"strict good-pregrasp entry contains rejected Top-8 members: {rejected}")


MVP80_STRICT_GOOD_PREGRASP_GATE = StrictGoodPregraspGate()


__all__ = ["MVP80_STRICT_GOOD_PREGRASP_GATE", "StrictGoodPregraspGate"]
