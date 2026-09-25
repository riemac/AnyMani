(
    'Identity-keyed schema-2 for heterogeneous pregrasp/contact-basin records. '
    'Pure-Python contract; imports no Isaac Lab, USD, policy, or training '
    'backend. A record separates object identity, center-point contact tier, and '
    'local perturbation-basin evidence. Tiers: rejected < support_basin < '
    'contact_basin < gravity_robust. Point coverage proves only the nominal '
    'center and cannot satisfy default basin queries; basin coverage needs a '
    'scale certificate and perturbation statistics. Palm contact is valid. '
    'Contact tier requires persistent >=2 TIP and bounded non-tip contact; '
    'gravity tier additionally requires >=3 TIP and all six hand-frame gravity '
    'directions.'
)

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Literal

PREGRASP_SCHEMA_VERSION = "2.1.0"  # 2.1 records reset state and PD preload target separately to preserve contact force.
PREGRASP_RECORD_ARTIFACT_TYPE = "anymani.pregrasp.record"  # Stable artifact type for one candidate certificate.
PREGRASP_INDEX_ARTIFACT_TYPE = "anymani.pregrasp.index"  # Stable artifact type for atomic cache index.
SCALE_ANCHORS = ("1.1", "1.2", "1.25")  # Three DexCube scale anchors sealed by P0001.
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")  # Use lowercase 64-hex SHA-256 for every physical/cache digest.


class PregraspTier(StrEnum):
    'Contact-certification tier reached by the center candidate.'

    REJECTED = "rejected"  # Did not meet even the support-stability gate.
    SUPPORT_BASIN = "support_basin"  # Palm-supported point/basin; two TIPs are not required.
    CONTACT_BASIN = "contact_basin"  # At least two TIPs and bounded non-tip contact.
    GRAVITY_ROBUST = "gravity_robust"  # Pass all six hand-frame gravity directions on top of contact tier.


class PregraspCoverage(StrEnum):
    'Whether coverage is failure, point, or local perturbation basin.'

    REJECTED = "rejected"  # Center point failed the minimum support gate.
    POINT = "point"  # Nominal candidate only; cannot satisfy require_basin.
    BASIN = "basin"  # Passed explicit-scale and local-perturbation certificate.


_TIER_RANK = {
    PregraspTier.REJECTED: 0,
    PregraspTier.SUPPORT_BASIN: 1,
    PregraspTier.CONTACT_BASIN: 2,
    PregraspTier.GRAVITY_ROBUST: 3,
}  # Tier order is only for provider minimum-tier comparison; physics per tier is fixed.


def tier_satisfies(actual: PregraspTier, required: PregraspTier) -> bool:
    'Check whether a tier meets a provider minimum-tier request.'

    return _TIER_RANK[PregraspTier(actual)] >= _TIER_RANK[PregraspTier(required)]  # Higher tiers imply lower-tier capability.


def _validate_sha256(value: str, field_name: str) -> str:
    'Strictly validate lowercase 64-character SHA-256; reject fake identities.'

    parsed = str(value)  # Input may be a JSON scalar; do not normalize case or truncate.
    if _SHA256_PATTERN.fullmatch(parsed) is None:
        raise ValueError(f"{field_name} must be a 64-character lowercase SHA-256 digest")
    return parsed


def _freeze_json(value: Any, path: str = "identity") -> Any:
    (
        'Recursively validate finite JSON and freeze containers so later mutation '
        'cannot alter the digest. Freeze dict as MappingProxyType and list/tuple as '
        'tuple; leaves are None/bool/int/finite float/str. Thaw to ordinary JSON '
        'containers only for serialization.'
    )

    if value is None or isinstance(value, (bool, str)):  # JSON-native dimensionless leaf value.
        return value
    if isinstance(value, int) and not isinstance(value, bool):  # Bool was handled by the previous branch.
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain finite JSON numbers")
        return float(value)
    if isinstance(value, Mapping):
        frozen = {str(key): _freeze_json(item, f"{path}.{key}") for key, item in sorted(value.items())}
        return MappingProxyType(frozen)  # Deep-copy then freeze so later input mutation cannot change identity.
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, f"{path}[{index}]") for index, item in enumerate(value))
    raise ValueError(f"{path} must contain finite JSON-compatible values, got {type(value).__name__}")


def _thaw_json(value: Any) -> Any:
    'Thaw frozen identity into a normal JSON-safe dict/list.'

    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    'Create canonical JSON bytes with stable field order/spacing and no NaN.'

    return json.dumps(
        _thaw_json(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")  # Digest is independent of indentation, locale, and dict insertion order.


def stable_digest(payload: Mapping[str, Any]) -> str:
    'Return SHA-256 of canonical JSON.'

    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()  # Shared digest boundary for cache/content identity.


def _finite_tuple(values: Sequence[float], *, length: int, field_name: str) -> tuple[float, ...]:
    'Normalize a fixed-length numeric sequence to finite floats.'

    parsed = tuple(float(value) for value in values)  # Normalize JSON/YAML numbers to Python float.
    if len(parsed) != length or not all(math.isfinite(value) for value in parsed):
        raise ValueError(f"{field_name} must contain {length} finite values")
    return parsed


def active_mask_digest(active_joint_mask: Sequence[bool]) -> str:
    'Digest canonical active-joint routing, excluding selection-local row.'

    mask = tuple(bool(value) for value in active_joint_mask)  # Canonical v1 has 16 slots.
    if len(mask) != 16 or not any(mask):
        raise ValueError("active_joint_mask must contain 16 entries and at least one active joint")
    return stable_digest({"active_joint_mask": list(mask)})  # Validate active action/joint structure only.


@dataclass(frozen=True)
class PregraspGate:
    (
        'Explicit numeric gates for point, contact, basin, and gravity certification. '
        'No loose defaults for TIP persistence or non-tip limit: each search must '
        'select values and bind this object digest into PregraspLookupKey. Every '
        'other threshold also enters the digest so calibration cannot silently reuse '
        'a cache.'
    )

    min_tip_ge_2_fraction: float  # Minimum time fraction with at least two TIP contacts for contact tier.
    min_tip_ge_3_fraction: float  # Minimum time fraction with at least three TIP contacts for gravity tier.
    max_finger_non_tip_fraction: float  # Maximum finger non-tip contact fraction; exclude palm.
    max_penetration_depth_m: float  # Maximum illegal object-hand penetration, meters.
    max_anchor_distance_m: float  # Maximum object drift from candidate anchor, meters.
    max_linear_velocity_rms_m_s: float  # Linear-velocity RMS during settle/stress window, m/s.
    max_angular_velocity_rms_rad_s: float  # Object angular-velocity RMS, rad/s.
    max_object_orientation_drift_rad: float  # Maximum orientation drift from candidate, rad.
    min_joint_limit_margin_rad: float  # Minimum margin from active joints to nearest limit, rad.
    max_target_tracking_error_rms_rad: float  # PD target-tracking RMS, rad.
    max_joint_effort_rms_N_m: float  # Active-joint effort RMS, N m.
    min_basin_success_fraction: float  # Fraction of local perturbation trials passing the complete point gate.
    required_gravity_directions: int = 6  # Six fixed hand-frame stress directions for AnyRotate strong tier.

    def __post_init__(self) -> None:
        'Reject invalid probabilities, negative physical bounds, and empty gravity stress.'

        fractions = (
            self.min_tip_ge_2_fraction,
            self.min_tip_ge_3_fraction,
            self.max_finger_non_tip_fraction,
            self.min_basin_success_fraction,
        )  # All four values are time/trial fractions in [0,1].
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("pregrasp gate fractions must lie in [0,1]")
        if self.min_tip_ge_2_fraction <= 0.0:  # Sealed policy forbids reducing contact gate to support-only.
            raise ValueError("min_tip_ge_2_fraction must be explicitly non-zero")
        non_negative = (
            self.max_penetration_depth_m,
            self.max_anchor_distance_m,
            self.max_linear_velocity_rms_m_s,
            self.max_angular_velocity_rms_rad_s,
            self.max_object_orientation_drift_rad,
            self.min_joint_limit_margin_rad,
            self.max_target_tracking_error_rms_rad,
            self.max_joint_effort_rms_N_m,
        )  # All values are finite nonnegative physical quantities.
        if any(not math.isfinite(value) or value < 0.0 for value in non_negative):
            raise ValueError("pregrasp gate physical thresholds must be finite and non-negative")
        if self.required_gravity_directions < 1:
            raise ValueError("required_gravity_directions must be positive")

    def to_dict(self) -> dict[str, Any]:
        'Return JSON-safe gate document.'

        return asdict(self)

    @property
    def digest(self) -> str:
        'Return canonical digest of all admission thresholds.'

        return stable_digest(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PregraspGate:
        'Restore and revalidate gate from an artifact.'

        return cls(**dict(payload))


@dataclass(frozen=True)
class PregraspLookupKey:
    (
        'Physical cache-query identity excluding scale interval and dataset row. '
        'asset_id is human provenance only and excluded from digest. Exact lookup '
        'depends on source/physical/canonical/routing/cube, support mode, gate, '
        'physics, and search identity; any scientific change creates another lookup '
        'domain.'
    )

    asset_id: str  # Provenance label; may change with dataset naming.
    source_content_hash: str  # source bundle content SHA-256
    physical_geometry_hash: str  # SHA-256 of real geometry, excluding ghosts.
    canonical_schema_digest: str  # canonical ABI schema SHA-256
    routing_digest: str  # active joint routing SHA-256
    cube_asset_id: str  # Human-readable object identity.
    cube_asset_sha256: str  # SHA-256 of actual runtime-resolved USD/object bytes.
    support_mode: Literal["palm_supported", "tip_only"]  # Current mainline support mode is palm_supported.
    gate_digest: str  # SHA-256 of PregraspGate.
    physics_identity: Mapping[str, Any]  # Solver/material/mass/inertia/contact thresholds, etc.
    search_identity: Mapping[str, Any]  # Algorithm/version/seed/proposals/stress definition.

    def __post_init__(self) -> None:
        'Strictly validate digests, text, and recursively finite JSON.'

        if not self.asset_id or not self.cube_asset_id:
            raise ValueError("asset_id and cube_asset_id must be non-empty")
        for field_name in (
            "source_content_hash",
            "physical_geometry_hash",
            "canonical_schema_digest",
            "routing_digest",
            "cube_asset_sha256",
            "gate_digest",
        ):
            object.__setattr__(self, field_name, _validate_sha256(getattr(self, field_name), field_name))
        if self.support_mode not in {"palm_supported", "tip_only"}:
            raise ValueError("support_mode must be palm_supported or tip_only")
        if not self.physics_identity or not self.search_identity:
            raise ValueError("physics_identity and search_identity must be non-empty")
        object.__setattr__(self, "physics_identity", _freeze_json(self.physics_identity, "physics_identity"))
        object.__setattr__(self, "search_identity", _freeze_json(self.search_identity, "search_identity"))

    def to_dict(self) -> dict[str, Any]:
        'Return full JSON document including provenance asset_id.'

        return {
            "asset_id": self.asset_id,
            "source_content_hash": self.source_content_hash,
            "physical_geometry_hash": self.physical_geometry_hash,
            "canonical_schema_digest": self.canonical_schema_digest,
            "routing_digest": self.routing_digest,
            "cube_asset_id": self.cube_asset_id,
            "cube_asset_sha256": self.cube_asset_sha256,
            "support_mode": self.support_mode,
            "gate_digest": self.gate_digest,
            "physics_identity": _thaw_json(self.physics_identity),
            "search_identity": _thaw_json(self.search_identity),
        }

    def identity_dict(self) -> dict[str, Any]:
        'Return fields used in lookup digest; exclude selection/provenance asset_id.'

        document = self.to_dict()
        document.pop("asset_id")  # Renaming the same physical asset must not create a duplicate cache entry.
        return document

    @property
    def digest(self) -> str:
        'Return canonical SHA-256 of lookup domain.'

        return stable_digest(self.identity_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PregraspLookupKey:
        'Restore lookup key from artifact; do not repair invalid fields.'

        return cls(**dict(payload))


@dataclass(frozen=True)
class PregraspCandidate:
    (
        'Canonical hand-controller state and hand-frame object reset point. Contact '
        'equilibrium also depends on implicit-PD preload: error between actual q_s '
        'and target q_t produces torque that maintains fingertip normal force. Store '
        'both values. Reset writes q_state_rad, then initializes action/controller '
        'target from q_target_rad. Setting them equal removes preload and can break '
        'replayed TIP contact.'
    )

    q_state_rad: tuple[float, ...]  # Canonical actual reset state q_s, shape [16], rad.
    q_target_rad: tuple[float, ...]  # Canonical PD preload target q_t, shape [16], rad.
    active_joint_mask: tuple[bool, ...]  # True means a real active joint.
    object_position_h_m: tuple[float, float, float]  # Object origin in hand frame, meters.
    object_orientation_wxyz: tuple[float, float, float, float]  # R_ho quaternion, order wxyz.
    object_scale: float  # nominal anchor scale
    seed_source: str  # proposal/template/refinement lineage

    def __post_init__(self) -> None:
        'Validate canonical axes, zero ghosts, unit quaternion, and positive scale.'

        q_state = _finite_tuple(self.q_state_rad, length=16, field_name="q_state_rad")
        q_target = _finite_tuple(self.q_target_rad, length=16, field_name="q_target_rad")
        mask = tuple(bool(value) for value in self.active_joint_mask)
        if len(mask) != 16 or not any(mask):
            raise ValueError("active_joint_mask must contain 16 entries and at least one active joint")
        if any(
            not active and (state != 0.0 or target != 0.0)
            for state, target, active in zip(q_state, q_target, mask)
        ):
            raise ValueError("pregrasp ghost joint state and target coordinates must be exactly zero")
        position = _finite_tuple(self.object_position_h_m, length=3, field_name="object_position_h_m")
        quaternion = _finite_tuple(self.object_orientation_wxyz, length=4, field_name="object_orientation_wxyz")
        if abs(math.sqrt(sum(value * value for value in quaternion)) - 1.0) > 1.0e-5:
            raise ValueError("object_orientation_wxyz must be a unit quaternion")
        if not math.isfinite(self.object_scale) or self.object_scale <= 0.0:
            raise ValueError("object_scale must be finite and positive")
        if not self.seed_source:
            raise ValueError("seed_source must be non-empty")
        object.__setattr__(self, "q_state_rad", q_state)
        object.__setattr__(self, "q_target_rad", q_target)
        object.__setattr__(self, "active_joint_mask", mask)
        object.__setattr__(self, "object_position_h_m", position)
        object.__setattr__(self, "object_orientation_wxyz", quaternion)

    def to_dict(self) -> dict[str, Any]:
        'Return JSON-safe candidate.'

        return {
            "q_state_rad": list(self.q_state_rad),
            "q_target_rad": list(self.q_target_rad),
            "active_joint_mask": list(self.active_joint_mask),
            "object_position_h_m": list(self.object_position_h_m),
            "object_orientation_wxyz": list(self.object_orientation_wxyz),
            "object_scale": self.object_scale,
            "seed_source": self.seed_source,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PregraspCandidate:
        'Restore candidate from artifact.'

        return cls(
            q_state_rad=tuple(payload["q_state_rad"]),
            q_target_rad=tuple(payload["q_target_rad"]),
            active_joint_mask=tuple(payload["active_joint_mask"]),
            object_position_h_m=tuple(payload["object_position_h_m"]),
            object_orientation_wxyz=tuple(payload["object_orientation_wxyz"]),
            object_scale=float(payload["object_scale"]),
            seed_source=str(payload["seed_source"]),
        )


@dataclass(frozen=True)
class PregraspMetrics:
    'Physical sufficient statistics for nominal point during settle/stress window.'

    finite: bool  # Whether all gate simulator states are finite.
    dropped: bool  # Whether an explicit drop/fall boundary was crossed.
    penetration_depth_max_m: float  # Maximum illegal object-hand penetration, meters.
    tip_ge_2_fraction: float  # Fraction of time with at least two active TIPs.
    tip_ge_3_fraction: float  # Fraction of time with at least three active TIPs.
    tip_active_count_mean: float  # Time-average active TIP count.
    palm_occupancy_fraction: float  # Fraction of time with valid palm support.
    finger_non_tip_occupancy_fraction: float  # Bad finger contact fraction; exclude palm.
    tip_object_center_distance_mean_m: float  # Mean TIP-to-object-center distance, meters.
    object_anchor_distance_max_m: float  # Maximum object translation from candidate anchor, meters.
    object_linear_velocity_rms_m_s: float  # Linear-velocity RMS, m/s.
    object_angular_velocity_rms_rad_s: float  # Angular-velocity RMS, rad/s.
    object_orientation_drift_max_rad: float  # Maximum orientation drift from candidate, rad.
    joint_limit_margin_min_rad: float  # Minimum joint-limit margin, rad.
    target_tracking_error_rms_rad: float  # PD target-tracking RMS, rad.
    joint_effort_rms_N_m: float  # Active-joint effort RMS, N m.

    def __post_init__(self) -> None:
        'Validate fractions and physical values; never replace missing data with zero.'

        fractions = (self.tip_ge_2_fraction, self.tip_ge_3_fraction, self.palm_occupancy_fraction, self.finger_non_tip_occupancy_fraction)
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("pregrasp metric fractions must lie in [0,1]")
        non_negative = (
            self.penetration_depth_max_m,
            self.tip_active_count_mean,
            self.tip_object_center_distance_mean_m,
            self.object_anchor_distance_max_m,
            self.object_linear_velocity_rms_m_s,
            self.object_angular_velocity_rms_rad_s,
            self.object_orientation_drift_max_rad,
            self.target_tracking_error_rms_rad,
            self.joint_effort_rms_N_m,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in non_negative):
            raise ValueError("pregrasp physical metrics must be finite and non-negative")
        if not math.isfinite(self.joint_limit_margin_min_rad):
            raise ValueError("joint_limit_margin_min_rad must be finite")

    def to_dict(self) -> dict[str, Any]:
        'Return JSON-safe point metrics.'

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PregraspMetrics:
        'Restore point metrics from artifact.'

        return cls(**dict(payload))


@dataclass(frozen=True)
class ScaleStressSample:
    'Verification result at one explicit scale.'

    scale: float  # Actual prestartup USD scale.
    passed: bool  # Whether full point gate passes at this tier.
    reason_codes: tuple[str, ...]  # Failure reason; must be empty when passed.
    physics_snapshot: Mapping[str, Any]  # Measured mass/inertia/COM at this scale; excluded from lookup domain.

    def __post_init__(self) -> None:
        'Validate scale and pass/reason consistency.'

        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("scale stress sample must use a finite positive scale")
        reasons = tuple(str(reason) for reason in self.reason_codes)
        if bool(self.passed) == bool(reasons):
            raise ValueError("passed scale sample must have no reasons; failed sample must have reasons")
        if not self.physics_snapshot:
            raise ValueError("scale stress sample requires an actual physics snapshot")
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "physics_snapshot", _freeze_json(self.physics_snapshot, "physics_snapshot"))

    def to_dict(self) -> dict[str, Any]:
        'Return JSON-safe scale sample.'

        return {
            "scale": self.scale,
            "passed": self.passed,
            "reason_codes": list(self.reason_codes),
            "physics_snapshot": _thaw_json(self.physics_snapshot),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ScaleStressSample:
        'Restore scale sample from artifact.'

        return cls(
            scale=float(payload["scale"]),
            passed=bool(payload["passed"]),
            reason_codes=tuple(payload["reason_codes"]),
            physics_snapshot=dict(payload["physics_snapshot"]),
        )


@dataclass(frozen=True)
class ScaleCertificate:
    'Continuous scale interval and local-basin statistics for one candidate.'

    anchor: Literal["1.1", "1.2", "1.25"]  # Use decimal strings for search anchors to avoid float-key ambiguity.
    scale_min: float  # Closed interval lower bound.
    scale_max: float  # Closed interval upper bound.
    scale_samples: tuple[ScaleStressSample, ...]  # Probe points rechecked inside the interval.
    perturbation_trials: int  # Total local q/pose/velocity trials.
    perturbation_successes: int  # Number passing the full point gate.
    gravity_directions_passed: int  # Number passing all six strong stress directions; may be zero for contact tier.

    def __post_init__(self) -> None:
        'Validate anchor, closed interval, scale samples, and binomial sufficient statistics.'

        if self.anchor not in SCALE_ANCHORS:
            raise ValueError(f"scale anchor must be one of {SCALE_ANCHORS}")
        if not (math.isfinite(self.scale_min) and math.isfinite(self.scale_max)):
            raise ValueError("scale interval must be finite")
        anchor_value = float(self.anchor)  # Numeric comparison only; artifact key keeps canonical string.
        if self.scale_min <= 0.0 or self.scale_max < self.scale_min or not self.scale_min <= anchor_value <= self.scale_max:
            raise ValueError("scale interval must be positive and contain its anchor")
        samples = tuple(self.scale_samples)
        if not samples or any(not self.scale_min <= sample.scale <= self.scale_max for sample in samples):
            raise ValueError("scale samples must be non-empty and lie inside the certified interval")
        if any(not sample.passed for sample in samples):
            raise ValueError("certified scale interval cannot contain a failed stress sample")
        if not any(abs(sample.scale - anchor_value) <= 1.0e-8 for sample in samples):
            raise ValueError("scale certificate must explicitly test its anchor")
        if self.perturbation_trials < 1 or not 0 <= self.perturbation_successes <= self.perturbation_trials:
            raise ValueError("perturbation successes must lie in [0,trials] with trials>0")
        if not 0 <= self.gravity_directions_passed <= 6:
            raise ValueError("gravity_directions_passed must lie in [0,6]")
        object.__setattr__(self, "scale_samples", samples)

    @property
    def basin_success_fraction(self) -> float:
        'Return local-perturbation binomial pass fraction.'

        return self.perturbation_successes / self.perturbation_trials  # Trials were validated as positive.

    def contains(self, scale: float) -> bool:
        'Check whether runtime scale lies in the closed certified interval.'

        requested = float(scale)
        return math.isfinite(requested) and self.scale_min <= requested <= self.scale_max

    def to_dict(self) -> dict[str, Any]:
        'Return JSON-safe scale/basin certificate.'

        return {
            "anchor": self.anchor,
            "scale_min": self.scale_min,
            "scale_max": self.scale_max,
            "scale_samples": [sample.to_dict() for sample in self.scale_samples],
            "perturbation_trials": self.perturbation_trials,
            "perturbation_successes": self.perturbation_successes,
            "gravity_directions_passed": self.gravity_directions_passed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ScaleCertificate:
        'Restore scale certificate from artifact.'

        return cls(
            anchor=str(payload["anchor"]),  # type: ignore[arg-type]
            scale_min=float(payload["scale_min"]),
            scale_max=float(payload["scale_max"]),
            scale_samples=tuple(ScaleStressSample.from_dict(item) for item in payload["scale_samples"]),
            perturbation_trials=int(payload["perturbation_trials"]),
            perturbation_successes=int(payload["perturbation_successes"]),
            gravity_directions_passed=int(payload["gravity_directions_passed"]),
        )


def _support_reasons(metrics: PregraspMetrics, gate: PregraspGate) -> tuple[str, ...]:
    'Compute stable physical rejection reasons for a support point.'

    reasons: list[str] = []  # Fixed order supports failure-histogram comparison across runs.
    if not metrics.finite:
        reasons.append("non_finite_state")
    if metrics.dropped:
        reasons.append("object_dropped")
    if metrics.penetration_depth_max_m > gate.max_penetration_depth_m:
        reasons.append("invalid_penetration")
    if metrics.object_anchor_distance_max_m > gate.max_anchor_distance_m:
        reasons.append("object_anchor_drift")
    if metrics.object_linear_velocity_rms_m_s > gate.max_linear_velocity_rms_m_s:
        reasons.append("object_linear_motion")
    if metrics.object_angular_velocity_rms_rad_s > gate.max_angular_velocity_rms_rad_s:
        reasons.append("object_angular_motion")
    if metrics.object_orientation_drift_max_rad > gate.max_object_orientation_drift_rad:
        reasons.append("object_orientation_drift")
    if metrics.joint_limit_margin_min_rad < gate.min_joint_limit_margin_rad:
        reasons.append("joint_limit_margin")
    if metrics.target_tracking_error_rms_rad > gate.max_target_tracking_error_rms_rad:
        reasons.append("target_tracking_error")
    if metrics.joint_effort_rms_N_m > gate.max_joint_effort_rms_N_m:
        reasons.append("joint_effort")
    return tuple(reasons)


def _infer_tier(metrics: PregraspMetrics, gate: PregraspGate, certificate: ScaleCertificate | None) -> tuple[PregraspTier, tuple[str, ...]]:
    'Derive the highest contact tier from point metrics and optional strong stress.'

    support_reasons = _support_reasons(metrics, gate)
    if support_reasons:
        return PregraspTier.REJECTED, support_reasons
    contact_reasons: list[str] = []
    if metrics.tip_ge_2_fraction < gate.min_tip_ge_2_fraction:
        contact_reasons.append("insufficient_tip_persistence")
    if metrics.finger_non_tip_occupancy_fraction > gate.max_finger_non_tip_fraction:
        contact_reasons.append("finger_non_tip_contact")
    if contact_reasons:
        return PregraspTier.SUPPORT_BASIN, tuple(contact_reasons)
    gravity_reasons: list[str] = []
    if metrics.tip_ge_3_fraction < gate.min_tip_ge_3_fraction:
        gravity_reasons.append("insufficient_three_tip_persistence")
    if certificate is None or certificate.gravity_directions_passed < gate.required_gravity_directions:
        gravity_reasons.append("gravity_stress_incomplete")
    if gravity_reasons:
        return PregraspTier.CONTACT_BASIN, tuple(gravity_reasons)
    return PregraspTier.GRAVITY_ROBUST, ()


@dataclass(frozen=True)
class PregraspRecord:
    'Self-validating schema-2 point/basin certification artifact for one candidate.'

    lookup_key: PregraspLookupKey  # Cache domain excluding scale interval.
    candidate: PregraspCandidate  # Nominal q and T_ho.
    metrics: PregraspMetrics  # Nominal-point physical metrics.
    gate: PregraspGate  # Admission thresholds; digest must match lookup.
    tier: PregraspTier  # Derived from metrics/certificate, never declared freely by writer.
    coverage: PregraspCoverage  # Point or basin coverage.
    scale_certificate: ScaleCertificate | None  # Required only for basin coverage.
    reason_codes: tuple[str, ...]  # Rejected or reason for not reaching a higher tier.

    def __post_init__(self) -> None:
        'Recompute identity and admission conclusions; reject inconsistent artifacts.'

        tier = PregraspTier(self.tier)
        coverage = PregraspCoverage(self.coverage)
        reasons = tuple(str(reason) for reason in self.reason_codes)
        if self.lookup_key.gate_digest != self.gate.digest:
            raise ValueError("lookup gate digest disagrees with embedded gate")
        if self.lookup_key.routing_digest != active_mask_digest(self.candidate.active_joint_mask):
            raise ValueError("candidate active mask disagrees with routing digest")
        certificate = self.scale_certificate
        if coverage == PregraspCoverage.BASIN:
            if certificate is None:
                raise ValueError("basin coverage requires a scale certificate")
            if not certificate.contains(self.candidate.object_scale):
                raise ValueError("candidate scale must lie inside its scale certificate")
            if certificate.basin_success_fraction < self.gate.min_basin_success_fraction:
                raise ValueError("basin certificate fails the configured perturbation success fraction")
        elif certificate is not None:
            raise ValueError("point/rejected coverage cannot carry a basin scale certificate")
        inferred_tier, inferred_reasons = _infer_tier(self.metrics, self.gate, certificate)
        inferred_coverage = PregraspCoverage.REJECTED if inferred_tier == PregraspTier.REJECTED else coverage
        if tier != inferred_tier or coverage != inferred_coverage or reasons != inferred_reasons:
            raise ValueError("record certification disagrees with metrics, gate, coverage, or certificate")
        object.__setattr__(self, "tier", tier)
        object.__setattr__(self, "coverage", coverage)
        object.__setattr__(self, "reason_codes", reasons)

    def payload_dict(self) -> dict[str, Any]:
        'Return canonical payload without its own digest.'

        return {
            "artifact_type": PREGRASP_RECORD_ARTIFACT_TYPE,
            "schema_version": PREGRASP_SCHEMA_VERSION,
            "lookup_key": self.lookup_key.to_dict(),
            "lookup_digest": self.lookup_key.digest,
            "candidate": self.candidate.to_dict(),
            "metrics": self.metrics.to_dict(),
            "gate": self.gate.to_dict(),
            "tier": self.tier.value,
            "coverage": self.coverage.value,
            "scale_certificate": self.scale_certificate.to_dict() if self.scale_certificate is not None else None,
            "reason_codes": list(self.reason_codes),
        }

    @property
    def digest(self) -> str:
        'Return content SHA-256 of the full certification payload.'

        return stable_digest(self.payload_dict())

    def to_dict(self) -> dict[str, Any]:
        'Return full artifact with record digest.'

        return {**self.payload_dict(), "record_digest": self.digest}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PregraspRecord:
        'Strictly restore record and recheck lookup/content digests.'

        if payload.get("artifact_type") != PREGRASP_RECORD_ARTIFACT_TYPE:
            raise ValueError("unsupported pregrasp artifact_type")
        if payload.get("schema_version") != PREGRASP_SCHEMA_VERSION:
            raise ValueError("unsupported pregrasp schema_version")
        lookup_key = PregraspLookupKey.from_dict(payload["lookup_key"])
        if payload.get("lookup_digest") != lookup_key.digest:
            raise ValueError("pregrasp lookup digest mismatch")
        certificate_payload = payload.get("scale_certificate")
        record = cls(
            lookup_key=lookup_key,
            candidate=PregraspCandidate.from_dict(payload["candidate"]),
            metrics=PregraspMetrics.from_dict(payload["metrics"]),
            gate=PregraspGate.from_dict(payload["gate"]),
            tier=PregraspTier(str(payload["tier"])),
            coverage=PregraspCoverage(str(payload["coverage"])),
            scale_certificate=(
                ScaleCertificate.from_dict(certificate_payload) if isinstance(certificate_payload, Mapping) else None
            ),
            reason_codes=tuple(str(reason) for reason in payload["reason_codes"]),
        )
        if payload.get("record_digest") != record.digest:
            raise ValueError("pregrasp record digest mismatch")
        return record


def certify_pregrasp(
    *,
    lookup_key: PregraspLookupKey,
    candidate: PregraspCandidate,
    metrics: PregraspMetrics,
    gate: PregraspGate,
    coverage: PregraspCoverage,
    scale_certificate: ScaleCertificate | None,
) -> PregraspRecord:
    (
        'Derive the highest tier from point/basin evidence and build a '
        'self-validating record. Inputs are exact lookup key, nominal candidate, '
        'point metrics, explicit gate, observed point/basin coverage, and optional '
        'basin scale certificate. The record tier/reason comes from evidence.'
    )

    requested_coverage = PregraspCoverage(coverage)
    inferred_tier, reasons = _infer_tier(metrics, gate, scale_certificate)
    actual_coverage = PregraspCoverage.REJECTED if inferred_tier == PregraspTier.REJECTED else requested_coverage
    retained_certificate = None if inferred_tier == PregraspTier.REJECTED else scale_certificate
    return PregraspRecord(
        lookup_key=lookup_key,
        candidate=candidate,
        metrics=metrics,
        gate=gate,
        tier=inferred_tier,
        coverage=actual_coverage,
        scale_certificate=retained_certificate,
        reason_codes=reasons,
    )


__all__ = [
    "PREGRASP_INDEX_ARTIFACT_TYPE",
    "PREGRASP_RECORD_ARTIFACT_TYPE",
    "PREGRASP_SCHEMA_VERSION",
    "SCALE_ANCHORS",
    "PregraspCandidate",
    "PregraspCoverage",
    "PregraspGate",
    "PregraspLookupKey",
    "PregraspMetrics",
    "PregraspRecord",
    "PregraspTier",
    "ScaleCertificate",
    "ScaleStressSample",
    "active_mask_digest",
    "canonical_json_bytes",
    "certify_pregrasp",
    "stable_digest",
    "tier_satisfies",
]
