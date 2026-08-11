"""几何语义 sidecar 的唯一归属、基准构型与锚点种子合同。"""

from __future__ import annotations

from pathlib import Path

import pytest
from anymani.assets.asset_base import (
    BoxGeometryCfg,
    CollisionGeometryCfg,
    FingerCfg,
    HandCfg,
    JointCfg,
    PalmCfg,
)
from anymani.assets.asset_schema_geometry import (
    derive_generated_geometry_semantics,
    geometry_semantics_from_dict,
    geometry_semantics_to_dict,
)
from anymani.assets.bank import HandContainer, HandContainerCfg
from anymani.assets.generator.runtime.restore import load_post_mutate_source

_MOTHER_ROOT = (
    Path(__file__).resolve().parents[2]
    / "generated"
    / "2026-06-10_11-30-08"
    / "single_palm_leap"
    / "right_t4_i4_m4_r4"
)
_requires_local_mother = pytest.mark.skipif(
    not _MOTHER_ROOT.is_dir(),
    reason="generated LEAP mother asset is a local research artifact",
)


def _box_collision(name: str) -> CollisionGeometryCfg:
    """构造最小合法碰撞片，用于证伪含混的 fixed descendant。"""

    return CollisionGeometryCfg(name=name, geometry=BoxGeometryCfg(size=(0.01, 0.01, 0.01)))


@_requires_local_mother
def test_mother_geometry_semantics_uniquely_cover_all_collision_pieces() -> None:
    """母体 24 个碰撞片必须唯一覆盖为 PALM 4、JOINT 16、TIP 4。"""

    source = load_post_mutate_source(_MOTHER_ROOT)
    semantics = derive_generated_geometry_semantics(
        source.hand_cfg,
        asset_id=source.origin_sample_id,
        topology_key=_MOTHER_ROOT.name,
    )

    assert len(semantics.components) == 24
    assert len({component.component_id for component in semantics.components}) == 24
    assert {component.owner_id for component in semantics.components} == {
        owner.owner_id for owner in semantics.owners
    }

    owner_role = {owner.owner_id: owner.role for owner in semantics.owners}
    piece_count_by_role = {
        role: sum(owner_role[component.owner_id] == role for component in semantics.components)
        for role in ("palm", "joint", "tip")
    }
    assert piece_count_by_role == {"palm": 4, "joint": 16, "tip": 4}
    assert sum(owner.role == "joint" for owner in semantics.owners) == 16
    assert sum(owner.role == "tip" for owner in semantics.owners) == 4

    assert semantics.active_joint_names == tuple(
        joint.name for joint in source.hand_cfg.iter_joints() if joint.joint_type == "revolute"
    )
    assert len(semantics.kinematic_joints) == 23
    assert sum(joint.joint_type == "revolute" for joint in semantics.kinematic_joints) == 16
    assert sum(joint.joint_type == "fixed" for joint in semantics.kinematic_joints) == 7
    assert tuple(
        joint.joint_name for joint in semantics.kinematic_joints if joint.active_joint_index is not None
    ) == semantics.active_joint_names
    assert semantics.q_home_rad == (0.0,) * 16
    assert semantics.units == {"length": "m", "angle": "rad"}


@_requires_local_mother
def test_mother_anchor_seeds_follow_exported_first_active_joint_frames() -> None:
    """seed 使用实际导出链中的首活动关节原点，而非裸 finger mount。"""

    source = load_post_mutate_source(_MOTHER_ROOT)
    semantics = derive_generated_geometry_semantics(source.hand_cfg, asset_id=source.origin_sample_id)
    seed_by_finger = {seed.finger_name: seed for seed in semantics.anchor_seeds}

    assert seed_by_finger["index"].position_a_m == pytest.approx((0.046, 0.093, 0.008))
    assert seed_by_finger["middle"].position_a_m == pytest.approx((0.0, 0.093, 0.008))
    assert seed_by_finger["ring"].position_a_m == pytest.approx((-0.046, 0.093, 0.008))
    assert seed_by_finger["thumb"].position_a_m == pytest.approx((0.037, 0.031, 0.01))
    assert {seed.support_owner_id for seed in semantics.anchor_seeds} == {"palm"}

    repeated = derive_generated_geometry_semantics(source.hand_cfg, asset_id=source.origin_sample_id)
    assert repeated.content_hash == semantics.content_hash


@_requires_local_mother
def test_geometry_semantics_sidecar_round_trip_revalidates_content_hash() -> None:
    """bank 反序列化必须恢复完全相同的静态事实，并拒绝哈希不匹配。"""

    source = load_post_mutate_source(_MOTHER_ROOT)
    semantics = derive_generated_geometry_semantics(source.hand_cfg, asset_id=source.origin_sample_id)
    document = geometry_semantics_to_dict(semantics)

    assert geometry_semantics_from_dict(document) == semantics

    document["q_home_rad"] = (0.1, *document["q_home_rad"][1:])
    with pytest.raises(ValueError, match="content_hash"):
        geometry_semantics_from_dict(document)


@_requires_local_mother
def test_hand_container_only_materializes_geometry_semantics_when_required() -> None:
    """tasks 默认保持轻量；distill 显式要求后迁移 legacy generated sidecar。"""

    lightweight = HandContainer.from_cfg(HandContainerCfg(path=_MOTHER_ROOT))
    assert lightweight.geometry_semantics is None

    distill_container = HandContainer.from_cfg(
        HandContainerCfg(path=_MOTHER_ROOT),
        require_geometry_semantics=True,
    )
    assert distill_container.geometry_semantics is not None
    assert len(distill_container.geometry_semantics.components) == 24
    assert distill_container.geometry_semantics.topology_key == "right_t4_i4_m4_r4"


@_requires_local_mother
def test_official_container_without_manual_geometry_semantics_fails_closed() -> None:
    """即使存在完整 hand_cfg，也不得把 generated 启发式套到 official 资产。"""

    cfg = HandContainerCfg(path=_MOTHER_ROOT, source_kind="official")
    with pytest.raises(ValueError, match="official.*explicit.*geometry_semantics"):
        HandContainer.from_cfg(cfg, require_geometry_semantics=True)


def test_generated_derivation_rejects_ambiguous_fixed_distal_collision() -> None:
    """未显式标为 TIP 的远端 fixed 碰撞片不得按位置启发式猜 owner。"""

    hand = HandCfg(
        name="ambiguous_fixed_descendant",
        palm=PalmCfg(collisions=[_box_collision("palm")]),
        fingers=[
            FingerCfg(
                name="finger",
                joints=[
                    JointCfg(
                        name="finger_j0",
                        parent="palm",
                        child="finger_l0",
                        joint_type="revolute",
                        collisions=[_box_collision("segment")],
                    ),
                    JointCfg(
                        name="finger_distal_fixed",
                        parent="finger_l0",
                        child="finger_tip",
                        joint_type="fixed",
                        collisions=[_box_collision("ambiguous")],
                        is_tip=False,
                    ),
                ],
            )
        ],
    )

    with pytest.raises(ValueError, match="fixed descendant.*is_tip"):
        derive_generated_geometry_semantics(hand, asset_id="ambiguous")


def test_generated_derivation_ignores_collision_free_distal_fixed_spacer() -> None:
    """无碰撞且未标 TIP 的末端 fixed spacer 只保留在运动学链，不得制造空 owner。"""

    hand = HandCfg(
        name="distal_spacer",
        palm=PalmCfg(collisions=[_box_collision("palm")]),
        fingers=[
            FingerCfg(
                name="finger",
                joints=[
                    JointCfg(
                        name="finger_j0",
                        parent="palm",
                        child="finger_l0",
                        joint_type="revolute",
                        collisions=[_box_collision("segment")],
                    ),
                    JointCfg(
                        name="finger_distal_spacer",
                        parent="finger_l0",
                        child="finger_terminal",
                        joint_type="fixed",
                        collisions=[],
                        is_tip=False,
                    ),
                ],
            )
        ],
    )

    semantics = derive_generated_geometry_semantics(hand, asset_id="distal-spacer")

    assert tuple(owner.owner_id for owner in semantics.owners) == ("palm", "joint/finger_j0")
    assert tuple(joint.joint_name for joint in semantics.kinematic_joints) == (
        "finger_j0",
        "finger_distal_spacer",
    )
