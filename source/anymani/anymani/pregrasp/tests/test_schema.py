"""Pregrasp schema-2 identity、tier、point/basin与scale certificate合同。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

import pytest

from anymani.pregrasp.schema import (
    PREGRASP_SCHEMA_VERSION,
    PregraspCandidate,
    PregraspCoverage,
    PregraspGate,
    PregraspLookupKey,
    PregraspMetrics,
    PregraspRecord,
    PregraspTier,
    ScaleCertificate,
    ScaleStressSample,
    active_mask_digest,
    certify_pregrasp,
    tier_satisfies,
)

_SOURCE_HASH = "1" * 64
_PHYSICAL_HASH = "2" * 64
_SCHEMA_HASH = "3" * 64
_CUBE_HASH = "4" * 64


def _gate() -> PregraspGate:
    return PregraspGate(
        min_tip_ge_2_fraction=0.8,
        min_tip_ge_3_fraction=0.8,
        max_finger_non_tip_fraction=0.0,
        max_penetration_depth_m=0.001,
        max_anchor_distance_m=0.025,
        max_linear_velocity_rms_m_s=0.05,
        max_angular_velocity_rms_rad_s=0.5,
        max_object_orientation_drift_rad=0.5,
        min_joint_limit_margin_rad=0.0,
        max_target_tracking_error_rms_rad=0.1,
        max_joint_effort_rms_N_m=2.0,
        min_basin_success_fraction=0.8,
        required_gravity_directions=6,
    )


def _mask() -> tuple[bool, ...]:
    return (True,) * 7 + (False,) * 9


def _lookup_key(*, gate: PregraspGate | None = None, physical_hash: str = _PHYSICAL_HASH) -> PregraspLookupKey:
    resolved_gate = gate or _gate()
    return PregraspLookupKey(
        asset_id="asset-provenance-only",
        source_content_hash=_SOURCE_HASH,
        physical_geometry_hash=physical_hash,
        canonical_schema_digest=_SCHEMA_HASH,
        routing_digest=active_mask_digest(_mask()),
        cube_asset_id="DexCube",
        cube_asset_sha256=_CUBE_HASH,
        support_mode="palm_supported",
        gate_digest=resolved_gate.digest,
        physics_identity={"isaac_sim": "5.1.0", "gravity_m_s2": 9.81, "solver": {"position": 8}},
        search_identity={"algorithm": "hetero-pregrasp-v2", "seed": 7},
    )


def _candidate(*, scale: float = 1.2, q0: float = 0.1) -> PregraspCandidate:
    return PregraspCandidate(
        q_state_rad=(q0,) * 7 + (0.0,) * 9,
        q_target_rad=(q0 + 0.01,) * 7 + (0.0,) * 9,
        active_joint_mask=_mask(),
        object_position_h_m=(0.0, 0.08, 0.06),
        object_orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        object_scale=scale,
        seed_source="n000-template",
    )


def _metrics(**changes: Any) -> PregraspMetrics:
    baseline = PregraspMetrics(
        finite=True,
        dropped=False,
        penetration_depth_max_m=0.0,
        tip_ge_2_fraction=0.95,
        tip_ge_3_fraction=0.2,
        tip_active_count_mean=2.6,
        palm_occupancy_fraction=0.8,
        finger_non_tip_occupancy_fraction=0.0,
        tip_object_center_distance_mean_m=0.02,
        object_anchor_distance_max_m=0.01,
        object_linear_velocity_rms_m_s=0.01,
        object_angular_velocity_rms_rad_s=0.1,
        object_orientation_drift_max_rad=0.1,
        joint_limit_margin_min_rad=0.05,
        target_tracking_error_rms_rad=0.01,
        joint_effort_rms_N_m=0.2,
    )
    return replace(baseline, **changes)


def _certificate(
    *,
    anchor: Literal["1.1", "1.2", "1.25"] = "1.2",
    scale_min: float = 1.15,
    scale_max: float = 1.22,
    perturbation_trials: int = 10,
    perturbation_successes: int = 9,
    gravity_directions_passed: int = 0,
) -> ScaleCertificate:
    return ScaleCertificate(
        anchor=anchor,
        scale_min=scale_min,
        scale_max=scale_max,
        scale_samples=(
            ScaleStressSample(
                scale=scale_min,
                passed=True,
                reason_codes=(),
                physics_snapshot={"mass_kg": 0.216, "inertia_xx_kg_m2": 0.00018},
            ),
            ScaleStressSample(
                scale=float(anchor),
                passed=True,
                reason_codes=(),
                physics_snapshot={"mass_kg": 0.216, "inertia_xx_kg_m2": 0.00019},
            ),
            ScaleStressSample(
                scale=scale_max,
                passed=True,
                reason_codes=(),
                physics_snapshot={"mass_kg": 0.216, "inertia_xx_kg_m2": 0.00020},
            ),
        ),
        perturbation_trials=perturbation_trials,
        perturbation_successes=perturbation_successes,
        gravity_directions_passed=gravity_directions_passed,
    )


def _record(
    *,
    coverage: PregraspCoverage = PregraspCoverage.BASIN,
    metrics: PregraspMetrics | None = None,
    certificate: ScaleCertificate | None = None,
    candidate: PregraspCandidate | None = None,
) -> PregraspRecord:
    gate = _gate()
    return certify_pregrasp(
        lookup_key=_lookup_key(gate=gate),
        candidate=candidate or _candidate(),
        metrics=metrics or _metrics(),
        gate=gate,
        coverage=coverage,
        scale_certificate=certificate if certificate is not None else (_certificate() if coverage == PregraspCoverage.BASIN else None),
    )


def test_schema_version_and_tier_order_are_explicit() -> None:
    assert PREGRASP_SCHEMA_VERSION == "2.1.0"
    assert tier_satisfies(PregraspTier.GRAVITY_ROBUST, PregraspTier.CONTACT_BASIN)
    assert tier_satisfies(PregraspTier.CONTACT_BASIN, PregraspTier.SUPPORT_BASIN)
    assert not tier_satisfies(PregraspTier.SUPPORT_BASIN, PregraspTier.CONTACT_BASIN)
    assert not tier_satisfies(PregraspTier.REJECTED, PregraspTier.SUPPORT_BASIN)


def test_lookup_digest_is_strict_immutable_and_excludes_asset_row() -> None:
    physics = {"isaac_sim": "5.1.0", "nested": {"values": [1.0, 2.0]}}
    gate = _gate()
    key = PregraspLookupKey(
        asset_id="asset",
        source_content_hash=_SOURCE_HASH,
        physical_geometry_hash=_PHYSICAL_HASH,
        canonical_schema_digest=_SCHEMA_HASH,
        routing_digest=active_mask_digest(_mask()),
        cube_asset_id="DexCube",
        cube_asset_sha256=_CUBE_HASH,
        support_mode="palm_supported",
        gate_digest=gate.digest,
        physics_identity=physics,
        search_identity={"algorithm": "v2"},
    )
    before = key.digest
    physics["nested"]["values"].append(3.0)  # type: ignore[index,union-attr]
    assert key.digest == before
    assert "asset_row" not in key.to_dict()
    assert replace(key, asset_id="renamed-provenance").digest == key.digest
    assert replace(key, physical_geometry_hash="5" * 64).digest != key.digest


@pytest.mark.parametrize("field", ("source_content_hash", "physical_geometry_hash", "canonical_schema_digest", "routing_digest", "cube_asset_sha256", "gate_digest"))
def test_lookup_rejects_non_sha256_identity_fields(field: str) -> None:
    key = _lookup_key()
    with pytest.raises(ValueError, match="SHA-256"):
        replace(key, **{field: "not-a-digest"})


def test_identity_rejects_nested_non_finite_json() -> None:
    key = _lookup_key()
    with pytest.raises(ValueError, match="finite JSON"):
        replace(key, physics_identity={"bad": float("nan")})


def test_candidate_rejects_nonzero_ghost_and_routing_mismatch() -> None:
    with pytest.raises(ValueError, match="ghost"):
        replace(_candidate(), q_state_rad=(0.1,) * 16)
    with pytest.raises(ValueError, match="ghost"):
        replace(_candidate(), q_target_rad=(0.1,) * 16)
    with pytest.raises(ValueError, match="routing"):
        certify_pregrasp(
            lookup_key=replace(_lookup_key(), routing_digest="6" * 64),
            candidate=_candidate(),
            metrics=_metrics(),
            gate=_gate(),
            coverage=PregraspCoverage.POINT,
            scale_certificate=None,
        )


def test_candidate_digest_preserves_pd_preload_target() -> None:
    unloaded = replace(_candidate(), q_target_rad=_candidate().q_state_rad)
    preloaded = _candidate()
    assert unloaded.q_state_rad == preloaded.q_state_rad
    assert unloaded.q_target_rad != preloaded.q_target_rad
    assert _record(candidate=unloaded).digest != _record(candidate=preloaded).digest


def test_contact_point_does_not_satisfy_basin_coverage() -> None:
    record = _record(coverage=PregraspCoverage.POINT)
    assert record.tier == PregraspTier.CONTACT_BASIN
    assert record.coverage == PregraspCoverage.POINT
    assert record.scale_certificate is None


def test_basin_requires_certificate_and_candidate_scale_inside_interval() -> None:
    gate = _gate()
    with pytest.raises(ValueError, match="certificate"):
        certify_pregrasp(
            lookup_key=_lookup_key(gate=gate),
            candidate=_candidate(),
            metrics=_metrics(),
            gate=gate,
            coverage=PregraspCoverage.BASIN,
            scale_certificate=None,
        )
    with pytest.raises(ValueError, match="candidate scale"):
        _record(candidate=_candidate(scale=1.25), certificate=_certificate(scale_max=1.22))


def test_contact_gate_is_strict_but_support_tier_remains_available() -> None:
    support = _record(metrics=_metrics(tip_ge_2_fraction=0.0, tip_active_count_mean=0.0))
    assert support.tier == PregraspTier.SUPPORT_BASIN
    contact = _record()
    assert contact.tier == PregraspTier.CONTACT_BASIN
    non_tip = _record(metrics=_metrics(finger_non_tip_occupancy_fraction=0.01))
    assert non_tip.tier == PregraspTier.SUPPORT_BASIN


def test_gravity_tier_requires_three_tip_and_all_directions() -> None:
    gravity = _record(
        metrics=_metrics(tip_ge_3_fraction=0.9, tip_active_count_mean=3.2),
        certificate=_certificate(gravity_directions_passed=6),
    )
    assert gravity.tier == PregraspTier.GRAVITY_ROBUST
    missing_directions = _record(metrics=_metrics(tip_ge_3_fraction=0.9), certificate=_certificate(gravity_directions_passed=5))
    assert missing_directions.tier == PregraspTier.CONTACT_BASIN


def test_rejected_basin_attempt_keeps_failure_but_not_false_certificate() -> None:
    rejected = _record(metrics=_metrics(dropped=True))
    assert rejected.tier == PregraspTier.REJECTED
    assert rejected.coverage == PregraspCoverage.REJECTED
    assert rejected.scale_certificate is None
    assert "object_dropped" in rejected.reason_codes


def test_record_roundtrip_revalidates_gate_identity_and_digest() -> None:
    record = _record()
    restored = PregraspRecord.from_dict(record.to_dict())
    assert restored == record
    assert restored.digest == record.digest
    document = record.to_dict()
    document["tier"] = PregraspTier.GRAVITY_ROBUST.value
    with pytest.raises(ValueError, match="certification"):
        PregraspRecord.from_dict(document)
