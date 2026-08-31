r"""自动 palm-supported pregrasp 的纯 Python schema 与数值接受门。

本模块不导入 IsaacLab、Kit、USD 或 policy。它定义物理搜索身份、canonical 16-slot reset 候选、
轨迹统计和确定性 gate，使离线 search、GM reset consumer、checkpoint provenance 与事后诊断共享同一
可审计语义。Palm contact 是合法支撑；坏接触只指 finger non-tip 或未声明 body contact。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

PREGRASP_SCHEMA_VERSION = "1.0.0"
"""Pregrasp result schema；任何字段/判据语义变化必须显式升级。"""

PREGRASP_RESULT_ARTIFACT_TYPE = "anymani.pregrasp.result"
"""单个 asset × scale interval × support mode 的物理搜索结果类型。"""


def _stable_digest(payload: Mapping[str, Any]) -> str:
    r"""对 JSON-safe mapping 计算canonical SHA-256。"""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()  # cache key不依赖YAML/JSON空白或字段写入顺序


def _finite_tuple(values: Sequence[float], *, length: int, field_name: str) -> tuple[float, ...]:
    r"""把固定长度数值序列转成finite float tuple。

    Args:
        values (Sequence[float]): 输入数值序列。
        length (int): 期望固定长度。
        field_name (str): 错误信息中的科研字段名。

    Returns:
        tuple[float, ...]: 长度固定且逐项finite的float tuple。
    """

    parsed = tuple(float(value) for value in values)  # JSON/YAML numeric统一成Python float
    if len(parsed) != length or not all(math.isfinite(value) for value in parsed):
        raise ValueError(f"{field_name} must contain {length} finite values")
    return parsed


@dataclass(frozen=True)
class PregraspIdentity:
    r"""与dataset row无关的手—物—scale—physics—search物理身份。

    `asset_row`、env spacing、replicas、policy与log path均不改变同一个pregrasp的物理语义，因此不进入
    本类型。相反，physical hand、cube bytes、scale interval、support mode、physics和search algorithm
    任一变化都必须产生不同digest。
    """

    asset_id: str
    source_content_hash: str
    physical_geometry_hash: str
    canonical_schema_digest: str
    cube_asset_id: str
    cube_asset_sha256: str
    scale_min: float
    scale_max: float
    support_mode: Literal["palm_supported", "tip_only"]
    physics_identity: Mapping[str, Any]
    search_identity: Mapping[str, Any]

    def __post_init__(self) -> None:
        r"""拒绝不完整身份、非法scale interval与未知support语义。"""

        text_fields = (
            self.asset_id,
            self.source_content_hash,
            self.physical_geometry_hash,
            self.canonical_schema_digest,
            self.cube_asset_id,
            self.cube_asset_sha256,
        )
        if not all(text_fields):
            raise ValueError("pregrasp physical identity fields must be non-empty")
        if not (math.isfinite(self.scale_min) and math.isfinite(self.scale_max)):
            raise ValueError("pregrasp scale interval must be finite")
        if self.scale_min <= 0.0 or self.scale_max < self.scale_min:
            raise ValueError("pregrasp scale interval must satisfy 0 < scale_min <= scale_max")
        if self.support_mode not in {"palm_supported", "tip_only"}:
            raise ValueError("pregrasp support_mode must be palm_supported or tip_only")
        if not self.physics_identity or not self.search_identity:
            raise ValueError("pregrasp physics/search identities must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        r"""返回JSON-safe identity mapping，不加入selection-local metadata。"""

        return {
            "asset_id": self.asset_id,
            "source_content_hash": self.source_content_hash,
            "physical_geometry_hash": self.physical_geometry_hash,
            "canonical_schema_digest": self.canonical_schema_digest,
            "cube_asset_id": self.cube_asset_id,
            "cube_asset_sha256": self.cube_asset_sha256,
            "scale_min": self.scale_min,
            "scale_max": self.scale_max,
            "support_mode": self.support_mode,
            "physics_identity": dict(self.physics_identity),
            "search_identity": dict(self.search_identity),
        }

    @property
    def digest(self) -> str:
        r"""返回完整物理搜索身份的canonical SHA-256 cache key。"""

        return _stable_digest(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PregraspIdentity:
        r"""从artifact mapping严格恢复identity。"""

        return cls(
            asset_id=str(payload["asset_id"]),
            source_content_hash=str(payload["source_content_hash"]),
            physical_geometry_hash=str(payload["physical_geometry_hash"]),
            canonical_schema_digest=str(payload["canonical_schema_digest"]),
            cube_asset_id=str(payload["cube_asset_id"]),
            cube_asset_sha256=str(payload["cube_asset_sha256"]),
            scale_min=float(payload["scale_min"]),
            scale_max=float(payload["scale_max"]),
            support_mode=str(payload["support_mode"]),  # type: ignore[arg-type]
            physics_identity=dict(payload["physics_identity"]),
            search_identity=dict(payload["search_identity"]),
        )


@dataclass(frozen=True)
class PregraspCandidate:
    r"""一个canonical hand与hand-frame object reset候选。

    `q_rad`与`active_joint_mask`固定为16槽canonical depth-major axis；inactive/ghost位置必须精确为零。
    Object pose定义在hand semantic frame `{h}`，position单位m，quaternion顺序固定`(w,x,y,z)`。
    """

    q_rad: tuple[float, ...]
    active_joint_mask: tuple[bool, ...]
    object_position_h_m: tuple[float, float, float]
    object_orientation_wxyz: tuple[float, float, float, float]
    object_scale: float
    seed_source: str

    def __post_init__(self) -> None:
        r"""验证canonical axes、ghost零值、quaternion和scale。"""

        q_rad = _finite_tuple(self.q_rad, length=16, field_name="q_rad")
        mask = tuple(bool(value) for value in self.active_joint_mask)
        if len(mask) != 16 or not any(mask):
            raise ValueError("active_joint_mask must contain 16 entries and at least one active joint")
        if any(not active and value != 0.0 for value, active in zip(q_rad, mask)):
            raise ValueError("pregrasp ghost joint coordinates must be exactly zero")
        position = _finite_tuple(self.object_position_h_m, length=3, field_name="object_position_h_m")
        quaternion = _finite_tuple(self.object_orientation_wxyz, length=4, field_name="object_orientation_wxyz")
        quaternion_norm = math.sqrt(sum(value * value for value in quaternion))
        if abs(quaternion_norm - 1.0) > 1.0e-5:
            raise ValueError("object_orientation_wxyz must be a unit quaternion")
        if not math.isfinite(self.object_scale) or self.object_scale <= 0.0:
            raise ValueError("object_scale must be finite and positive")
        if not self.seed_source:
            raise ValueError("pregrasp seed_source must be non-empty")
        object.__setattr__(self, "q_rad", q_rad)
        object.__setattr__(self, "active_joint_mask", mask)
        object.__setattr__(self, "object_position_h_m", position)
        object.__setattr__(self, "object_orientation_wxyz", quaternion)

    def to_dict(self) -> dict[str, Any]:
        r"""返回JSON-safe candidate mapping。"""

        return {
            "q_rad": list(self.q_rad),
            "active_joint_mask": list(self.active_joint_mask),
            "object_position_h_m": list(self.object_position_h_m),
            "object_orientation_wxyz": list(self.object_orientation_wxyz),
            "object_scale": self.object_scale,
            "seed_source": self.seed_source,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PregraspCandidate:
        r"""从artifact mapping恢复canonical candidate。"""

        return cls(
            q_rad=tuple(payload["q_rad"]),
            active_joint_mask=tuple(payload["active_joint_mask"]),
            object_position_h_m=tuple(payload["object_position_h_m"]),
            object_orientation_wxyz=tuple(payload["object_orientation_wxyz"]),
            object_scale=float(payload["object_scale"]),
            seed_source=str(payload["seed_source"]),
        )


@dataclass(frozen=True)
class PregraspMetrics:
    r"""一个候选在settle/stress窗口上的纯数值统计。"""

    finite: bool
    dropped: bool
    penetrated: bool
    tip_ge_2_fraction: float
    tip_active_count_mean: float
    palm_occupancy_fraction: float
    finger_non_tip_occupancy_fraction: float
    tip_object_center_distance_mean_m: float
    object_anchor_distance_max_m: float
    object_linear_velocity_rms_m_s: float
    object_angular_velocity_rms_rad_s: float
    joint_limit_margin_min_rad: float
    target_tracking_error_rms_rad: float

    def __post_init__(self) -> None:
        r"""验证概率、单位统计与finite边界。"""

        fractions = (
            self.tip_ge_2_fraction,
            self.palm_occupancy_fraction,
            self.finger_non_tip_occupancy_fraction,
        )
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("pregrasp occupancy/persistence fractions must lie in [0,1]")
        non_negative = (
            self.tip_active_count_mean,
            self.tip_object_center_distance_mean_m,
            self.object_anchor_distance_max_m,
            self.object_linear_velocity_rms_m_s,
            self.object_angular_velocity_rms_rad_s,
            self.target_tracking_error_rms_rad,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in non_negative):
            raise ValueError("pregrasp count/distance/velocity/tracking metrics must be finite and non-negative")
        if not math.isfinite(self.joint_limit_margin_min_rad):
            raise ValueError("pregrasp joint limit margin must be finite")

    def to_dict(self) -> dict[str, Any]:
        r"""返回JSON-safe metric mapping。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PregraspMetrics:
        r"""从artifact mapping恢复数值统计。"""

        return cls(**dict(payload))


@dataclass(frozen=True)
class PregraspAcceptanceCfg:
    r"""Easy-tier palm-supported数值接受阈值。

    N000实测表明manual/zero-q初态都由palm稳定承托，且都没有`>=2 TIP`持续接触；因此默认硬门只
    约束稳定性，TIP距离/接触和finger non-tip作为候选排序指标。更严格precision-grasp probe可显式
    覆盖前两项阈值。Palm occupancy没有上下限。
    """

    min_tip_ge_2_fraction: float = 0.0
    max_finger_non_tip_fraction: float = 1.0
    max_anchor_distance_m: float = 0.025
    max_linear_velocity_rms_m_s: float = 0.05
    max_angular_velocity_rms_rad_s: float = 0.5
    min_joint_limit_margin_rad: float = 0.0
    max_target_tracking_error_rms_rad: float = 0.1


def evaluate_pregrasp(metrics: PregraspMetrics, config: PregraspAcceptanceCfg) -> tuple[str, ...]:
    r"""按稳定顺序返回候选拒绝reason codes；空tuple表示accepted。

    Args:
        metrics (PregraspMetrics): settle/stress窗口统计。
        config (PregraspAcceptanceCfg): easy-tier数值阈值。

    Returns:
        tuple[str, ...]: 可审计、顺序稳定的拒绝原因；不包含palm contact。
    """

    reasons: list[str] = []
    if not metrics.finite:
        reasons.append("non_finite_state")
    if metrics.dropped:
        reasons.append("object_dropped")
    if metrics.penetrated:
        reasons.append("invalid_penetration")
    if metrics.tip_ge_2_fraction < config.min_tip_ge_2_fraction:
        reasons.append("insufficient_tip_persistence")
    if metrics.finger_non_tip_occupancy_fraction > config.max_finger_non_tip_fraction:
        reasons.append("finger_non_tip_contact")
    if metrics.object_anchor_distance_max_m > config.max_anchor_distance_m:
        reasons.append("object_anchor_drift")
    if metrics.object_linear_velocity_rms_m_s > config.max_linear_velocity_rms_m_s:
        reasons.append("object_linear_motion")
    if metrics.object_angular_velocity_rms_rad_s > config.max_angular_velocity_rms_rad_s:
        reasons.append("object_angular_motion")
    if metrics.joint_limit_margin_min_rad < config.min_joint_limit_margin_rad:
        reasons.append("joint_limit_margin")
    if metrics.target_tracking_error_rms_rad > config.max_target_tracking_error_rms_rad:
        reasons.append("target_tracking_error")
    return tuple(reasons)


@dataclass(frozen=True)
class PregraspResult:
    r"""单个物理身份与候选的版本化接受/拒绝artifact。"""

    identity: PregraspIdentity
    candidate: PregraspCandidate
    metrics: PregraspMetrics
    status: Literal["accepted", "rejected"]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        r"""确保status与reason codes一致，防止artifact自相矛盾。"""

        if self.status not in {"accepted", "rejected"}:
            raise ValueError("pregrasp result status must be accepted or rejected")
        if self.status == "accepted" and self.reason_codes:
            raise ValueError("accepted pregrasp result cannot contain rejection reasons")
        if self.status == "rejected" and not self.reason_codes:
            raise ValueError("rejected pregrasp result must contain at least one reason")

    def to_dict(self) -> dict[str, Any]:
        r"""返回完整JSON-safe schema-1 result document。"""

        return {
            "artifact_type": PREGRASP_RESULT_ARTIFACT_TYPE,
            "schema_version": PREGRASP_SCHEMA_VERSION,
            "identity": self.identity.to_dict(),
            "identity_digest": self.identity.digest,
            "candidate": self.candidate.to_dict(),
            "metrics": self.metrics.to_dict(),
            "status": self.status,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PregraspResult:
        r"""严格恢复schema-1 result并复核identity digest。"""

        if payload.get("artifact_type") != PREGRASP_RESULT_ARTIFACT_TYPE:
            raise ValueError("unsupported pregrasp artifact_type")
        if payload.get("schema_version") != PREGRASP_SCHEMA_VERSION:
            raise ValueError("unsupported pregrasp schema_version")
        identity = PregraspIdentity.from_dict(payload["identity"])
        if payload.get("identity_digest") != identity.digest:
            raise ValueError("pregrasp identity digest mismatch")
        return cls(
            identity=identity,
            candidate=PregraspCandidate.from_dict(payload["candidate"]),
            metrics=PregraspMetrics.from_dict(payload["metrics"]),
            status=str(payload["status"]),  # type: ignore[arg-type]
            reason_codes=tuple(str(reason) for reason in payload["reason_codes"]),
        )


__all__ = [
    "PREGRASP_RESULT_ARTIFACT_TYPE",
    "PREGRASP_SCHEMA_VERSION",
    "PregraspAcceptanceCfg",
    "PregraspCandidate",
    "PregraspIdentity",
    "PregraspMetrics",
    "PregraspResult",
    "evaluate_pregrasp",
]
