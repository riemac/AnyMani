r"""Formal task-owned pregrasp gate/physics/search identity反例合同。"""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from anymani.pregrasp import PregraspLookupKey
from anymani.tasks.hetero.config.generated.pregrasp_identity import (
    DEX_CUBE_SHA256,
    FORMAL_PREGRASP_GATE,
    FORMAL_SEARCH_PROTOCOL,
    FormalPregraspCatalogIdentity,
    formal_physics_identity,
)


def _search_identity(*, trials: int = 64, successes: int = 64) -> dict[str, object]:
    r"""构造允许任意seed/lineage、但协议字段与formal contract相同的search identity。"""

    return {
        "algorithm": FORMAL_SEARCH_PROTOCOL["certifier"],
        "candidate_state_semantics": FORMAL_SEARCH_PROTOCOL["candidate_state"],
        "basin_artifact_sha256": "a" * 64,
        "nominal_artifact_sha256": "b" * 64,
        "nominal_record_digest": "c" * 64,
        "perturbation_trials": trials,
        "perturbation_successes": successes,
        "basin_protocol": {
            "algorithm": FORMAL_SEARCH_PROTOCOL["point_basin_algorithm"],
            "seed": 20260911,
            "q_proposals": ["verified-candidate"],
            "basin_trial_count": trials,
            "basin_center_control_trials_per_asset": FORMAL_SEARCH_PROTOCOL["center_control_trials_per_asset"],
            "refinement_q_offset_rad": FORMAL_SEARCH_PROTOCOL["q_perturbation_rad"],
            "refinement_object_offset_h_m": FORMAL_SEARCH_PROTOCOL["position_perturbation_h_m"],
            "basin_rotation_vector_h_rad": FORMAL_SEARCH_PROTOCOL["rotation_vector_perturbation_h_rad"],
            "basin_linear_velocity_h_m_s": FORMAL_SEARCH_PROTOCOL["linear_velocity_perturbation_h_m_s"],
            "basin_angular_velocity_h_rad_s": FORMAL_SEARCH_PROTOCOL["angular_velocity_perturbation_h_rad_s"],
            "settle_policy_steps": FORMAL_SEARCH_PROTOCOL["settle_policy_steps"],
            "settle_prefix_policy_steps": FORMAL_SEARCH_PROTOCOL["settle_prefix_policy_steps"],
            "certification_tail_policy_steps": FORMAL_SEARCH_PROTOCOL["certification_tail_policy_steps"],
            "physics_substeps_per_policy_step": FORMAL_SEARCH_PROTOCOL["physics_substeps_per_policy_step"],
        },
    }


def _lookup_key() -> PregraspLookupKey:
    r"""构造scale1.2 valid lookup；hand hashes只需满足独立runtime identity层。"""

    return PregraspLookupKey(
        asset_id="row16",
        source_content_hash="1" * 64,
        physical_geometry_hash="2" * 64,
        canonical_schema_digest="3" * 64,
        routing_digest="4" * 64,
        cube_asset_id="DexCube",
        cube_asset_sha256=DEX_CUBE_SHA256,
        support_mode="palm_supported",
        gate_digest=FORMAL_PREGRASP_GATE.digest,
        physics_identity=formal_physics_identity(object_scale=1.2, cube_sha256=DEX_CUBE_SHA256),
        search_identity=_search_identity(),
    )


def test_formal_identity_accepts_exact_scene_gate_and_search_protocol() -> None:
    r"""正确hand-independent catalog字段通过；hand identity由下一层独立验证。"""

    identity = FormalPregraspCatalogIdentity.build(object_scale=1.2, cube_sha256=DEX_CUBE_SHA256)
    identity.validate_lookup_key(_lookup_key())
    assert FORMAL_PREGRASP_GATE.min_tip_ge_2_fraction == 0.8
    assert FORMAL_PREGRASP_GATE.max_finger_non_tip_fraction == 0.0
    assert FORMAL_PREGRASP_GATE.required_gravity_directions == 6


@pytest.mark.parametrize("mutation", ("gate", "cube", "material", "solver", "search", "perturbation", "trials"))
def test_formal_identity_rejects_unique_but_semantically_drifted_record(mutation: str) -> None:
    r"""即便cache中只有一个record，宽松gate或任一scene/search漂移也必须在provider前被拒绝。"""

    key = _lookup_key()
    if mutation == "gate":
        key = replace(key, gate_digest="f" * 64)
    elif mutation == "cube":
        key = replace(key, cube_asset_sha256="e" * 64)
    elif mutation in {"material", "solver"}:
        physics = copy.deepcopy(key.to_dict()["physics_identity"])
        if mutation == "material":
            physics["material"]["static_friction"] = 0.5
        else:
            physics["solver"]["position_iterations"] = 4
        key = replace(key, physics_identity=physics)
    else:
        search = copy.deepcopy(key.to_dict()["search_identity"])
        if mutation == "search":
            search["algorithm"] = "different-certifier"
        elif mutation == "perturbation":
            search["basin_protocol"]["refinement_q_offset_rad"] = [-0.1, 0.1]
        else:
            search["perturbation_trials"] = 16
            search["perturbation_successes"] = 16
            search["basin_protocol"]["basin_trial_count"] = 16
        key = replace(key, search_identity=search)
    identity = FormalPregraspCatalogIdentity.build(object_scale=1.2, cube_sha256=DEX_CUBE_SHA256)
    with pytest.raises(ValueError):
        identity.validate_lookup_key(key)


def test_formal_physics_identity_rejects_wrong_resolved_cube_bytes_and_unmeasured_scale() -> None:
    r"""实际USD bytes与独立scale probe缺一不可，不允许hash常量或scale外推。"""

    with pytest.raises(ValueError, match="DexCube bytes"):
        formal_physics_identity(object_scale=1.2, cube_sha256="0" * 64)
    with pytest.raises(ValueError, match="scale anchors"):
        formal_physics_identity(object_scale=1.15, cube_sha256=DEX_CUBE_SHA256)
