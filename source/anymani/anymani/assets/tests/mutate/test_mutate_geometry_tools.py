"""后序 mutate 的几何/尺度工具回归测试。

这组测试覆盖当前已经实现的两类“非拓扑”工具：

1. `link_scale`：改 joint origin 的有效长度
2. `tip_replace`：改末端 tip 的主体几何或 mesh 局部位姿

它们都不改 finger 链的 parent/child 关系，因此适合作为 mutate 第二阶段里
风险较低的实现切片。
"""

from __future__ import annotations

import math

from assets.builder.hand_builders import HumanLikeHandBuilder, HumanLikeHandBuilderCfg
from assets.generator.mutate import LinkScaleCfg, LinkScaleMutator, ScalarDistributionCfg, TipReplaceCfg, TipReplaceMutator
from assets.presets import get_finger_builder_preset, make_human_like_builder_cfg


def _make_allegro_builder_cfg() -> HumanLikeHandBuilderCfg:
    """构造一份稳定的 Allegro pre-made hand recipe。"""

    return make_human_like_builder_cfg(
        name="allegro_mutate_geometry_demo",
        family="allegro",
        handedness="right",
        palm_cfg="com_allegro",
        finger_cfg="allegro_non_thumb_v1",
        thumb_cfg="allegro_thumb_v1",
    )


def _build_allegro_hand():
    """构造一份稳定的 Allegro `HandCfg`。"""

    return HumanLikeHandBuilder(_make_allegro_builder_cfg()).build()


def _build_custom_tip_hand():
    """构造一份 index/middle/ring 统一使用 custom round tip 的 LEAP 风格手。"""

    finger_cfg = get_finger_builder_preset("leap_non_thumb_v1").replace(
        tip={"type": "mesh", "tip_type": "round"},
    )
    return HumanLikeHandBuilder(
        make_human_like_builder_cfg(
            name="leap_custom_tip_mutate_demo",
            family="leap",
            handedness="right",
            palm_cfg="com_leap",
            finger_cfg=finger_cfg,
            thumb_cfg="leap_thumb_v1",
        )
    ).build()


def _joint_by_name(hand, joint_name: str):
    """按名字取 joint。"""

    for joint in hand.iter_joints():
        if joint.name == joint_name:
            return joint
    raise KeyError(joint_name)


def _finger_by_name(hand, finger_name: str):
    """按名字取 finger。"""

    for finger in hand.fingers:
        if finger.name == finger_name:
            return finger
    raise KeyError(finger_name)


def test_link_scale_mutator_rescales_link_length_and_advances_next_joint_origin():
    """`link_scale` 应改变自身 link 长度，并用新 $L_i+d_i$ 推进下游 joint。"""

    hand = _build_allegro_hand()
    before_index_origin = _joint_by_name(hand, "index_j1").origin.pos
    before_index_size = _joint_by_name(hand, "index_j1").collisions[0].geometry.size
    before_next_origin = _joint_by_name(hand, "index_j2").origin.pos
    before_middle = _joint_by_name(hand, "middle_j1").origin.pos

    mutated = LinkScaleMutator(
        LinkScaleCfg(
            target_joints=("index_j1",),
            scale_mode="relative",
            delta_distribution=ScalarDistributionCfg(kind="fixed", value=0.1),
            clip_ratio=0.1,
        )
    ).mutate(hand, sampled_params={"index_j1": 0.1})

    assert mutated is not None
    after_index_origin = _joint_by_name(mutated, "index_j1").origin.pos
    after_index_size = _joint_by_name(mutated, "index_j1").collisions[0].geometry.size
    after_next_origin = _joint_by_name(mutated, "index_j2").origin.pos
    after_middle = _joint_by_name(mutated, "middle_j1").origin.pos
    assert math.isclose(after_index_size[1], before_index_size[1] * 1.1, rel_tol=0.0, abs_tol=1e-12)
    assert after_next_origin[1] > before_next_origin[1]
    assert after_index_origin == before_index_origin
    assert after_middle == before_middle


def test_tip_replace_mutator_swaps_primitive_tip_body_geometry():
    """`geometry_swap` 模式应把 `cs` tip 的主体从 cylinder 换成 box。"""

    hand = _build_allegro_hand()
    before_tip_joint = _finger_by_name(hand, "index").tip_joint
    before_inertial = before_tip_joint.inertial
    mutated = TipReplaceMutator(
        TipReplaceCfg(
            target_fingers=("index",),
            mode="geometry_swap",
            target_geometry="box",
        )
    ).mutate(hand)

    assert mutated is not None
    tip_joint = _finger_by_name(mutated, "index").tip_joint
    assert {collision.geometry.kind for collision in tip_joint.collisions} == {"box", "sphere"}
    assert {visual.geometry.kind for visual in tip_joint.visuals} == {"box", "sphere"}
    assert tip_joint.inertial is not None
    assert tip_joint.inertial.mass > 0.0
    assert tip_joint.inertial.origin.pos[1] > tip_joint.collisions[0].origin.pos[1]
    assert tip_joint.inertial.inertia.ixx > 0.0
    assert tip_joint.inertial.inertia.iyy > 0.0
    assert tip_joint.inertial.inertia.izz > 0.0
    assert not math.isclose(tip_joint.inertial.mass, before_inertial.mass, rel_tol=0.0, abs_tol=1e-12)
    assert not math.isclose(tip_joint.inertial.inertia.ixx, 1e-7, rel_tol=0.0, abs_tol=1e-12)


def test_tip_replace_mutator_perturbs_custom_mesh_tip_origin():
    """`mesh_perturb` 模式应对 custom mesh tip 的局部原点做比例扰动。"""

    hand = _build_custom_tip_hand()
    before_origin = _finger_by_name(hand, "index").tip_joint.collisions[0].origin.pos

    mutated = TipReplaceMutator(
        TipReplaceCfg(
            target_fingers=("index",),
            mode="mesh_perturb",
            mesh_offset_distribution=ScalarDistributionCfg(kind="fixed", value=0.1),
        )
    ).mutate(
        hand,
        sampled_params={
            "index::collisions::0::0": 0.1,
            "index::collisions::0::2": 0.1,
            "index::visuals::0::0": 0.1,
            "index::visuals::0::2": 0.1,
        },
    )

    assert mutated is not None
    after_origin = _finger_by_name(mutated, "index").tip_joint.collisions[0].origin.pos
    assert after_origin != before_origin
    assert math.isclose(after_origin[1], before_origin[1], rel_tol=0.0, abs_tol=1e-12)  # 当前 y=0，规则应保持它贴在轴上
