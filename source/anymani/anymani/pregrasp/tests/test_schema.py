"""自动 pregrasp artifact identity、scale interval 与数值接受门合同。"""

from __future__ import annotations

from anymani.pregrasp.schema import (
    PregraspAcceptanceCfg,
    PregraspCandidate,
    PregraspIdentity,
    PregraspMetrics,
    PregraspResult,
    evaluate_pregrasp,
)


def _identity(*, physical_hash: str = "physical", scale_min: float = 1.1) -> PregraspIdentity:
    r"""构造不含selection-local asset row的最小物理搜索身份。"""

    return PregraspIdentity(
        asset_id="asset",
        source_content_hash="source",
        physical_geometry_hash=physical_hash,
        canonical_schema_digest="canonical",
        cube_asset_id="DexCube",
        cube_asset_sha256="cube",
        scale_min=scale_min,
        scale_max=1.25,
        support_mode="palm_supported",
        physics_identity={"isaac_sim": "5.1.0", "gravity_m_s2": 9.81},
        search_identity={"algorithm": "bounded-settle-v1", "seed": 7},
    )


def _candidate() -> PregraspCandidate:
    r"""构造7-DOF active hand的canonical 16-slot reset候选。"""

    active = (True,) * 7 + (False,) * 9
    return PregraspCandidate(
        q_rad=(0.1,) * 7 + (0.0,) * 9,
        active_joint_mask=active,
        object_position_h_m=(0.0, 0.08, 0.06),
        object_orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        object_scale=1.2,
        seed_source="n000-template",
    )


def _accepted_metrics() -> PregraspMetrics:
    r"""构造满足palm-supported easy-tier门的纯数值轨迹统计。"""

    return PregraspMetrics(
        finite=True,
        dropped=False,
        penetrated=False,
        tip_ge_2_fraction=0.95,
        tip_active_count_mean=2.6,
        palm_occupancy_fraction=0.8,
        finger_non_tip_occupancy_fraction=0.0,
        tip_object_center_distance_mean_m=0.02,
        object_anchor_distance_max_m=0.01,
        object_linear_velocity_rms_m_s=0.01,
        object_angular_velocity_rms_rad_s=0.1,
        joint_limit_margin_min_rad=0.05,
        target_tracking_error_rms_rad=0.01,
    )


def test_identity_digest_tracks_physics_and_scale_but_has_no_asset_row() -> None:
    r"""Cache identity必须反映物理/scale变化，并排除dataset selection row。"""

    baseline = _identity()
    changed_physical = _identity(physical_hash="changed")
    changed_scale = _identity(scale_min=1.15)

    assert baseline.digest != changed_physical.digest
    assert baseline.digest != changed_scale.digest
    assert "asset_row" not in baseline.to_dict()


def test_candidate_rejects_nonzero_ghost_joint() -> None:
    r"""Canonical ghost slot只服务ABI，pregrasp q必须在inactive位置精确为零。"""

    active = (True,) * 7 + (False,) * 9
    try:
        PregraspCandidate(
            q_rad=(0.1,) * 16,
            active_joint_mask=active,
            object_position_h_m=(0.0, 0.08, 0.06),
            object_orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            object_scale=1.2,
            seed_source="invalid",
        )
    except ValueError as exc:
        assert "ghost" in str(exc)
    else:
        raise AssertionError("non-zero ghost pregrasp coordinates must be rejected")


def test_acceptance_gate_preserves_palm_support_and_rejects_bad_contact() -> None:
    r"""Palm occupancy不扣分；TIP不足或finger non-tip接触必须给出稳定reason code。"""

    config = PregraspAcceptanceCfg(min_tip_ge_2_fraction=0.8, max_finger_non_tip_fraction=0.0)
    accepted = evaluate_pregrasp(_accepted_metrics(), config)
    assert accepted == ()

    rejected_metrics = PregraspMetrics(
        **{
            **_accepted_metrics().to_dict(),
            "tip_ge_2_fraction": 0.2,
            "finger_non_tip_occupancy_fraction": 0.1,
        }
    )
    reasons = evaluate_pregrasp(rejected_metrics, config)
    assert "insufficient_tip_persistence" in reasons
    assert "finger_non_tip_contact" in reasons
    assert "palm_contact" not in reasons


def test_result_round_trip_keeps_identity_candidate_metrics_and_status() -> None:
    r"""JSON-safe result roundtrip不能改变科研身份或接受结论。"""

    result = PregraspResult(
        identity=_identity(),
        candidate=_candidate(),
        metrics=_accepted_metrics(),
        status="accepted",
        reason_codes=(),
    )

    restored = PregraspResult.from_dict(result.to_dict())

    assert restored == result
    assert restored.identity.digest == result.identity.digest
    assert restored.candidate.object_scale == 1.2
