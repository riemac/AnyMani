from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from anymani.assets.asset_schema_core import InertialCfg, JointLimitCfg, PoseCfg
from anymani.assets.asset_schema_embodiment import FingerCfg, HandCfg, JointCfg, PalmCfg
from anymani.assets.canonical_runtime import (
    CANONICAL_HAND_SCHEMA_V1,
    CanonicalHandSchemaCfg,
    lower_hand_to_canonical,
)
from anymani.assets.exporter import UrdfWriter, UrdfWriterCfg


def _inertial(mass: float) -> InertialCfg:
    r"""构造测试 link 的正定惯量；数值只用于区分真实 body 与 ghost body。"""

    return InertialCfg(
        mass=mass,  # kg；不同真实 link 使用不同质量，便于核对 lowering 没有串 body
        inertia={"ixx": 1.0e-5, "iyy": 1.1e-5, "izz": 1.2e-5},  # kg m^2；严格正定对角惯量
    )


def _finger(slot: str, dof: int, *, root_fixed: bool, mount_x: float) -> FingerCfg:
    r"""构造一条 proximal-to-distal 紧凑活动链和真实 fixed tip。"""

    joints: list[JointCfg] = []  # 当前 source finger 的有序 joint-child 链
    parent = "palm"  # 所有 generated finger 都从 palm 挂载

    # LEAP non-thumb 的 root-fixed body 是真实质量/几何承载体，不能被 canonical ghost 替换。
    if root_fixed:
        root_child = f"{slot}_source_root"
        joints.append(
            JointCfg(
                name=f"{slot}_root_fixed",
                parent=parent,
                child=root_child,
                joint_type="fixed",
                limit=None,
                origin=PoseCfg(pos=(0.01, 0.0, 0.0)),  # m；与 finger mount 复合后应成为 canonical root origin
                inertial=_inertial(0.02),
            )
        )
        parent = root_child

    # delete 后的 source joint 已重新紧凑编号；canonical lowering 只能按当前链序补远端槽。
    for depth in range(dof):
        child = f"{slot}_source_link_{depth}"
        joints.append(
            JointCfg(
                name=f"{slot}_j{depth}",
                parent=parent,
                child=child,
                axis=(1.0, 0.0, 0.0),
                origin=PoseCfg(pos=(0.0, 0.02 + 0.01 * depth, 0.0)),  # m；每层不同以核对局部变换
                limit=JointLimitCfg(lower=-0.5, upper=1.0, effort=2.0, velocity=3.0),
                inertial=_inertial(0.03 + 0.01 * depth),
            )
        )
        parent = child

    # tip 固定在当前真实活动链末端；新增 ghost 必须插在该 tip 之前且保持零位姿 FK 不变。
    joints.append(
        JointCfg(
            name=f"{slot}_tip",
            parent=parent,
            child=f"{slot}_source_tip",
            joint_type="fixed",
            limit=None,
            origin=PoseCfg(pos=(0.0, 0.04, 0.0)),
            inertial=_inertial(0.01),
            is_tip=True,
        )
    )
    return FingerCfg(
        name=slot,
        parent_link="palm",
        mount=PoseCfg(pos=(mount_x, 0.1, 0.0)),  # m；source exporter 会把它折叠进首 joint
        joints=joints,
    )


def _source_hand() -> HandCfg:
    r"""构造 7-DOF、缺 ring 的 LEAP-like source hand。"""

    return HandCfg(
        name="canonical_contract_source",
        palm=PalmCfg(name="palm", inertial=_inertial(0.2)),
        fingers=[
            _finger("index", 1, root_fixed=True, mount_x=0.04),
            _finger("middle", 3, root_fixed=True, mount_x=0.0),
            _finger("thumb", 3, root_fixed=False, mount_x=-0.04),
        ],
        family="leap",
        handedness="right",
    )


def test_schema_v1_freezes_physx_names_and_dimensions() -> None:
    r"""v1 schema 必须固定 16 DOF、25 bodies，并显式记录 importer 的 depth-major 顺序。"""

    schema = CANONICAL_HAND_SCHEMA_V1

    assert schema.dof_count == 16  # $4\text{ fingers}\times4\text{ revolute/finger}$
    assert schema.body_count == 25  # palm + $4\times(root+4\ joint\ links+tip)$
    assert schema.joint_names[:4] == ("index_j0", "middle_j0", "ring_j0", "thumb_j0")
    assert schema.joint_names[-4:] == ("index_j3", "middle_j3", "ring_j3", "thumb_j3")
    assert len(schema.digest) == 64  # schema digest 是 artifact/runtime 同构性的 SHA-256 锚点


def test_lowering_preserves_active_chain_and_appends_distal_ghosts() -> None:
    r"""真实活动 joint 保持局部参数；缺失槽只在真实 tip 前补零位姿 ghost。"""

    canonical, routing = lower_hand_to_canonical(_source_hand(), asset_id="source-7dof")
    fingers = {finger.name: finger for finger in canonical.fingers}

    assert canonical.dof_count == 16
    assert tuple(finger.name for finger in canonical.fingers) == CANONICAL_HAND_SCHEMA_V1.semantic_finger_slots
    assert routing.active_dof_count == 7
    assert routing.active_joint_names == (
        "index_j0",
        "middle_j0",
        "thumb_j0",
        "middle_j1",
        "thumb_j1",
        "middle_j2",
        "thumb_j2",
    )

    # index 只有一个真实活动 joint，因此 j1--j3 必须是远端 ghost，随后才连接真实 tip。
    index_joints = fingers["index"].joints
    assert [joint.name for joint in index_joints] == [
        "index_root_fixed",
        "index_j0",
        "index_j1",
        "index_j2",
        "index_j3",
        "index_tip_fixed",
    ]
    assert index_joints[1].metadata["canonical_active"] is True
    assert all(joint.metadata["canonical_active"] is False for joint in index_joints[2:5])
    assert all(joint.origin == PoseCfg() for joint in index_joints[2:5])  # ghost 在 $q=0$ 不改变 tip FK
    assert index_joints[-1].origin == PoseCfg(pos=(0.0, 0.04, 0.0))  # 真实 tip local transform 原样保留

    # source 没有 thumb root body；canonical root 吸收 mount+首 joint origin，活动 j0 改为 identity。
    thumb_joints = fingers["thumb"].joints
    assert thumb_joints[0].metadata["canonical_ghost"] is True
    assert thumb_joints[0].origin.pos == pytest.approx((-0.04, 0.12, 0.0))
    assert thumb_joints[1].origin == PoseCfg()

    # 缺失 ring 不伪造成真实 owner；四个 revolute slots 和 fixed endpoints 全部标记为 ghost。
    ring_joints = fingers["ring"].joints
    assert all(joint.metadata["canonical_active"] is False for joint in ring_joints)
    assert routing.active_tip_mask == (True, True, False, True)  # PhysX finger order index/middle/ring/thumb


def test_ghost_visual_marker_is_invisible_and_has_no_collision(tmp_path: Path) -> None:
    r"""ghost marker 只闭合 USD visual scope，不能进入碰撞或学习 geometry owner。"""

    canonical, _ = lower_hand_to_canonical(_source_hand(), asset_id="source-7dof")
    result = UrdfWriter(UrdfWriterCfg()).export(canonical, tmp_path)
    assert not result.errors

    root = ET.parse(tmp_path / "hand.urdf").getroot()
    ghost_link = root.find("./link[@name='ring_link_j0']")
    assert ghost_link is not None
    assert ghost_link.find("./collision") is None  # ghost body 不产生接触、reward 或 representation surface
    sphere = ghost_link.find("./visual/geometry/sphere")
    color = ghost_link.find("./visual/material/color")
    assert sphere is not None and float(sphere.attrib["radius"]) == pytest.approx(1.0e-7)
    assert color is not None and color.attrib["rgba"] == "0 0 0 0"


def test_lowering_rejects_non_compact_active_joint_names() -> None:
    r"""source delete 若没有完成 proximal 紧凑重编号，canonical lowering 必须 fail closed。"""

    source = _source_hand()
    index = next(finger for finger in source.fingers if finger.name == "index")
    index.joints[1] = index.joints[1].replace(name="index_j2")  # 构造 j0 缺失、j2 悬空的错误链语义

    with pytest.raises(ValueError, match="compact proximal order"):
        lower_hand_to_canonical(source, asset_id="broken")


def test_schema_rejects_duplicate_or_mismatched_finger_order() -> None:
    r"""semantic slots 与 PhysX traversal slots 必须是同一集合，不能遗漏或重复 finger。"""

    with pytest.raises(ValueError, match="same unique finger set"):
        CanonicalHandSchemaCfg(
            semantic_finger_slots=("thumb", "index", "middle", "ring"),
            physx_finger_order=("index", "middle", "thumb", "thumb"),
        )
