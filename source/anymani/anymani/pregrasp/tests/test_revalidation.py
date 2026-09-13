r"""Allegro128 局部关节限位修订的纯 CPU 输入合同测试。

这些测试只验证候选文件的身份、数组形状与 canonical hand-object 初态语义；
它们不启动 Isaac Sim，也不把候选误标为已经通过 PhysX 的安全 reset。
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from anymani.pregrasp.revalidation import (
    REVALIDATION_CANDIDATE_SOURCE,
    REVALIDATION_GENERATION_ARTIFACT_TYPE,
    REVALIDATION_GENERATION_KIND,
    REVALIDATION_GENERATION_SCHEMA_VERSION,
    RevalidationCandidates,
    build_revalidation_generation_identity,
    canonical_json_sha256,
    load_revalidation_candidates,
    stable_digest,
    validate_revalidation_generation_identity,
)

_HASH_A = "a" * 64
_HASH_B = "b" * 64
_HASH_C = "c" * 64
_HASH_D = "d" * 64
_JOINT_NAMES = np.asarray(
    [
        "index_j0",
        "middle_j0",
        "ring_j0",
        "thumb_j0",
        "index_j1",
        "middle_j1",
        "ring_j1",
        "thumb_j1",
        "index_j2",
        "middle_j2",
        "ring_j2",
        "thumb_j2",
        "index_j3",
        "middle_j3",
        "ring_j3",
        "thumb_j3",
    ],
    dtype="U16",
)


_REFINEMENT = {
    "elite_physical_candidates": 16,
    "elite_proposal_counts_descending": [24, 20, 16, 12, 8, 8, 6, 6, 4, 4, 4, 4, 3, 3, 3, 3],
    "full_physics_candidates_per_round": 128,
    "joint_pca_dimensions": 4,
    "object_position_dimensions": 3,
    "per_asset_manual_parameters": False,
    "position_center_feedback": "minus_physx_contact_normal_times_1p10_depth_plus_0p25mm",
    "proposals_per_round_per_failed_asset": 128,
    "random_stream_key": "source_content_and_physical_geometry_sha256",
    "rounds_max": 3,
    "settle_height_feedback": "if_initial_penetration_le_0p5mm_lower_by_clamped_displacement_minus_4p5mm",
    "strict_mode_exploitation": {
        "activation": "one_to_seven_strict_or_normalized_gate_violation_le_0p35",
        "joint_std_rad": 0.0005,
        "max_proposals_per_round": 96,
        "position_std_m": [5e-05, 5e-05, 2.5e-05],
    },
    "type": "per_asset_elite_mixture_low_rank_gaussian_cem",
}


def _identity(
    candidate_sha: str = _HASH_A,
    *,
    refinement: dict[str, Any] | None = None,
) -> dict[str, Any]:
    r"""构造测试用的、可被 loader 复核的修订 generation identity。"""

    return build_revalidation_generation_identity(
        candidate_npz_sha256=candidate_sha,
        source_catalog_index_sha256=_HASH_B,
        strict_gate_digest=_HASH_C,
        physics_identity_digest=_HASH_D,
        refinement_identity=refinement,
    )


def _write_candidates(
    path: Path,
    *,
    duplicate_new_id: bool = False,
    extra: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    r"""写入两个资产、每资产八项的最小合法 fixture，并返回其数组。"""

    original = np.zeros((2, 8, 16), dtype=np.float64)
    original[0, :, 0] = 0.10
    original[1, :, 0] = 0.20
    original[1, :, 4] = 0.20
    projected = original.copy()
    projected[1, :, 4] = 0.21
    active = np.ones((2, 16), dtype=np.bool_)
    active[0, 14:] = False
    active[1, 15] = False
    original[0, :, 14:] = 0.0
    projected[0, :, 14:] = 0.0
    original[1, :, 15] = 0.0
    projected[1, :, 15] = 0.0
    position = np.zeros((2, 8, 3), dtype=np.float64)
    position[:, :, 1] = 0.10
    position[:, :, 2] = 0.06
    position[:, :, 0] = np.arange(8, dtype=np.float64) * 1.0e-3
    orientation = np.zeros((2, 8, 4), dtype=np.float64)
    orientation[:, :, 0] = 1.0
    parent_id = np.asarray(["parent-a", "parent-b"], dtype="U16")
    new_id = np.asarray(["new-a", "new-b"], dtype="U16")
    if duplicate_new_id:
        new_id[1] = new_id[0]
    parent_digest = np.asarray([_HASH_A, _HASH_B], dtype="U64")
    arrays: dict[str, np.ndarray] = {
        "original_q_rad": original,
        "projected_q_rad": projected,
        "object_position_h_m": position,
        "object_orientation_h_wxyz": orientation,
        "active_joint_mask": active,
        "parent_asset_id": parent_id,
        "new_asset_id": new_id,
        "parent_entry_digest": parent_digest,
        "canonical_joint_names": _JOINT_NAMES,
    }
    arrays.update(extra or {})
    np.savez(path, **arrays)
    return arrays


def test_generation_identity_is_independent_and_fixed() -> None:
    r"""generation identity 明确区分修订候选、旧 c82 搜索和已认证 catalog。"""

    identity = _identity()
    assert identity["artifact_type"] == REVALIDATION_GENERATION_ARTIFACT_TYPE
    assert identity["kind"] == REVALIDATION_GENERATION_KIND
    assert identity["schema_version"] == REVALIDATION_GENERATION_SCHEMA_VERSION
    assert identity["candidate_source"] == REVALIDATION_CANDIDATE_SOURCE
    assert identity["candidate_count"] == 8
    assert identity["projection_margin_fraction"] == pytest.approx(0.101)
    assert identity["strict_gate_digest"] == _HASH_C
    assert identity["physics_identity_digest"] == _HASH_D
    assert identity["strict_physics_gate"]["window_seconds"] == pytest.approx(1.0)
    assert identity["search"]["uses_sobol"] is False
    assert identity["search"]["uses_cem"] is False
    assert identity["protocol"] != "c82"
    assert validate_revalidation_generation_identity(identity) == identity


def test_canonical_json_sha_is_order_independent() -> None:
    r"""canonical JSON digest 不受 mapping 插入顺序影响，且采用 SHA-256。"""

    left = {"z": [1, 2], "a": {"finite": 1.5}}
    right = {"a": {"finite": 1.5}, "z": [1, 2]}
    expected = hashlib.sha256(b'{"a":{"finite":1.5},"z":[1,2]}').hexdigest()
    assert canonical_json_sha256(left) == expected
    assert canonical_json_sha256(left) == canonical_json_sha256(right)


def test_none_refinement_keeps_the_published_v1_identity_byte_for_byte() -> None:
    r"""refinement=None 不新增字段，也不漂移已发布 v1 canonical digest。"""

    first = _identity()
    repeated = _identity(refinement=None)
    assert repeated == first
    assert stable_digest(repeated) == stable_digest(first)
    assert "refinement" not in repeated
    assert "refinement_rounds" not in repeated


def test_none_refinement_matches_the_persisted_5c3a_v1_identity() -> None:
    r"""回读已发布 revalidation-v1 artifact，保护历史 5c3a digest 不被扩展改写。"""

    repo_root = Path(__file__).parents[5]
    artifact_path = (
        repo_root / "logs/benchmarks/family_teacher_distillation/allegro-tuning-20260911/"
        "mcp2-223-assets-v1/revalidation-v1/generation-identity.json"
    )
    if not artifact_path.is_file():
        pytest.skip("persisted revalidation-v1 artifact is not present in this checkout")
    published = json.loads(artifact_path.read_text(encoding="utf-8"))
    generated = build_revalidation_generation_identity(
        candidate_npz_sha256=published["candidate_npz_sha256"],
        source_catalog_index_sha256=published["source_catalog_index_sha256"],
        strict_gate_digest=published["strict_gate_digest"],
        physics_identity_digest=published["physics_identity_digest"],
    )
    assert generated == published
    assert stable_digest(generated) == "5c3a71e6c9900fcbf50785ebe6f74744aac70920e4112bbce4300161d0ac6458"


def test_refined_identity_binds_the_real_three_round_cem_profile() -> None:
    r"""非空 refinement 进入独立 protocol，并逐值绑定真实 c82 CEM payload。"""

    base = _identity()
    refined = _identity(refinement=copy.deepcopy(_REFINEMENT))
    assert refined["artifact_type"] != base["artifact_type"]
    assert refined["kind"] != base["kind"]
    assert refined["schema_version"] != base["schema_version"]
    assert refined["protocol"] != base["protocol"]
    assert refined["candidate_count"] == 8
    assert refined["seed"] == 20260902
    assert refined["refinement_rounds"] == 3
    assert refined["initial"]["candidate_count"] == 8
    assert refined["initial"]["uses_sobol"] is False
    assert refined["refinement"] == _REFINEMENT
    assert refined["refinement_scope"] == "assets-with-fewer-than-8-initial-strict-passes"
    assert refined["cem_continuation"]["initial_elite_count"] == 8
    assert refined["cem_continuation"]["initial_center_assignment"] == "round_robin"
    assert refined["cem_continuation"]["later_elite_count"] == 16
    assert refined["ranking"] == "preserve-parent-rank-when-all-initial-pass-else-physical-quality"
    assert refined["search"]["uses_sobol"] is False
    assert refined["search"]["uses_cem"] is True
    assert validate_revalidation_generation_identity(refined) == refined
    assert stable_digest(refined) != stable_digest(base)
    assert stable_digest(refined) != "5c3a71e6c9900fcbf50785ebe6f74744aac70920e4112bbce4300161d0ac6458"


@pytest.mark.parametrize(
    "mutation",
    ["rounds", "physics", "elite", "type", "strict_exploitation", "sobol", "old_v1"],
)
def test_refined_identity_rejects_profile_drift_or_legacy_upgrade(mutation: str) -> None:
    r"""续跑 profile 任一关键字段漂移，或给旧 v1 伪造 refinement，均 fail-closed。"""

    if mutation == "old_v1":
        identity = _identity()
        identity["refinement"] = copy.deepcopy(_REFINEMENT)
        with pytest.raises(ValueError, match="legacy|refinement|protocol"):
            validate_revalidation_generation_identity(identity)
        return

    identity = _identity(refinement=copy.deepcopy(_REFINEMENT))
    if mutation == "rounds":
        identity["refinement_rounds"] = 2
    elif mutation == "physics":
        identity["refinement"]["full_physics_candidates_per_round"] = 127
    elif mutation == "elite":
        identity["refinement"]["elite_physical_candidates"] = 8
    elif mutation == "type":
        identity["refinement"]["type"] = "some-other-cem"
    elif mutation == "strict_exploitation":
        identity["refinement"]["strict_mode_exploitation"]["joint_std_rad"] = 0.001
    elif mutation == "sobol":
        identity["search"]["uses_sobol"] = True
    with pytest.raises(ValueError, match="refinement|CEM|Sobol|candidate|profile|protocol"):
        validate_revalidation_generation_identity(identity)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"candidate_npz_sha256": "A" * 64}, "lowercase"),
        ({"candidate_count": 7}, "candidate_count"),
        ({"projection_margin_fraction": 0.1}, "projection_margin_fraction"),
    ],
)
def test_generation_identity_rejects_non_protocol_values(
    kwargs: dict[str, object],
    message: str,
) -> None:
    r"""固定协议字段不能借参数名伪装成另一套搜索或门限。"""

    base: dict[str, object] = {
        "candidate_npz_sha256": _HASH_A,
        "source_catalog_index_sha256": _HASH_B,
        "strict_gate_digest": _HASH_C,
        "physics_identity_digest": _HASH_D,
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        build_revalidation_generation_identity(**base)  # type: ignore[arg-type]


def test_loader_reorders_arbitrary_subset_and_returns_read_only_candidates(tmp_path: Path) -> None:
    r"""loader 按 new_asset_id 精确重排，并保留未经物理认证的候选语义。"""

    path = tmp_path / "candidates.npz"
    arrays = _write_candidates(path)
    identity = _identity(hashlib.sha256(path.read_bytes()).hexdigest())
    masks = {"new-a": arrays["active_joint_mask"][0], "new-b": arrays["active_joint_mask"][1]}
    loaded = load_revalidation_candidates(
        path,
        generation_identity=identity,
        asset_ids=["new-b", "new-a"],
        active_joint_masks={"new-a": masks["new-a"], "new-b": masks["new-b"]},
        joint_names=_JOINT_NAMES.tolist(),
    )

    assert isinstance(loaded, RevalidationCandidates)
    assert loaded.new_asset_id.tolist() == ["new-b", "new-a"]
    np.testing.assert_array_equal(loaded.projected_q_rad[0], arrays["projected_q_rad"][1])
    np.testing.assert_array_equal(loaded.original_q_rad[1], arrays["original_q_rad"][0])
    np.testing.assert_array_equal(loaded.active_joint_mask[0], arrays["active_joint_mask"][1])
    assert loaded.candidate_source == REVALIDATION_CANDIDATE_SOURCE
    assert loaded.physically_certified is False
    assert not loaded.projected_q_rad.flags.writeable
    assert not loaded.original_q_rad.flags.writeable
    assert not loaded.object_position_h_m.flags.writeable
    assert not loaded.object_orientation_h_wxyz.flags.writeable


def test_loader_accepts_aligned_mask_sequence_for_subset(tmp_path: Path) -> None:
    r"""active mask 也支持与 asset_ids 同序的 numpy/list 输入，避免强制 caller 建 dict。"""

    path = tmp_path / "candidates.npz"
    arrays = _write_candidates(path)
    identity = _identity(hashlib.sha256(path.read_bytes()).hexdigest())
    loaded = load_revalidation_candidates(
        path,
        generation_identity=identity,
        asset_ids=["new-b"],
        active_joint_masks=[arrays["active_joint_mask"][1]],
        joint_names=_JOINT_NAMES.tolist(),
    )
    assert loaded.new_asset_id.tolist() == ["new-b"]


@pytest.mark.parametrize(
    "mutation",
    ["sha", "missing", "duplicate", "names", "mask", "finite", "ghost", "orientation", "shape", "target"],
)
def test_loader_rejects_tampered_or_ambiguous_input(tmp_path: Path, mutation: str) -> None:
    r"""输入合同对文件身份、ID、shape、有限性、ghost与target语义 fail-closed。"""

    path = tmp_path / f"{mutation}.npz"
    arrays = _write_candidates(path, duplicate_new_id=mutation == "duplicate")
    identity = _identity(hashlib.sha256(path.read_bytes()).hexdigest())
    kwargs: dict[str, object] = {
        "generation_identity": identity,
        "asset_ids": ["new-a", "new-b"],
        "active_joint_masks": arrays["active_joint_mask"],
        "joint_names": _JOINT_NAMES.tolist(),
    }

    if mutation == "sha":
        kwargs["generation_identity"] = _identity(_HASH_D)
    elif mutation == "missing":
        kwargs["asset_ids"] = ["new-a", "not-in-file"]
        kwargs["active_joint_masks"] = [arrays["active_joint_mask"][0], arrays["active_joint_mask"][0]]
    elif mutation == "names":
        names = _JOINT_NAMES.copy()
        names[0] = "wrong_j0"
        kwargs["joint_names"] = names
    elif mutation == "mask":
        masks = arrays["active_joint_mask"].copy()
        masks[1, 0] = ~masks[1, 0]
        kwargs["active_joint_masks"] = masks
    elif mutation == "finite":
        arrays["projected_q_rad"][0, 0, 0] = np.nan
        _write_candidates(path, extra={"projected_q_rad": arrays["projected_q_rad"]})
        kwargs["generation_identity"] = _identity(hashlib.sha256(path.read_bytes()).hexdigest())
    elif mutation == "ghost":
        arrays["projected_q_rad"][0, 0, 14] = 0.2
        _write_candidates(path, extra={"projected_q_rad": arrays["projected_q_rad"]})
        kwargs["generation_identity"] = _identity(hashlib.sha256(path.read_bytes()).hexdigest())
    elif mutation == "orientation":
        arrays["object_orientation_h_wxyz"][0, 0] = [0.0, 1.0, 0.0, 0.0]
        _write_candidates(path, extra={"object_orientation_h_wxyz": arrays["object_orientation_h_wxyz"]})
        kwargs["generation_identity"] = _identity(hashlib.sha256(path.read_bytes()).hexdigest())
    elif mutation == "shape":
        arrays["object_position_h_m"] = np.zeros((2, 7, 3), dtype=np.float64)
        _write_candidates(path, extra={"object_position_h_m": arrays["object_position_h_m"]})
        kwargs["generation_identity"] = _identity(hashlib.sha256(path.read_bytes()).hexdigest())
    elif mutation == "target":
        arrays["q_target_rad"] = arrays["original_q_rad"].copy()
        _write_candidates(path, extra={"q_target_rad": arrays["q_target_rad"]})
        kwargs["generation_identity"] = _identity(hashlib.sha256(path.read_bytes()).hexdigest())

    with pytest.raises(
        (ValueError, KeyError),
        match="(?i)candidate|asset|joint|mask|finite|ghost|orientation|shape|target|digest|sha",
    ):
        load_revalidation_candidates(path, **kwargs)  # type: ignore[arg-type]
