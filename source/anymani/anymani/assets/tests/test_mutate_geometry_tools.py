"""后序 mutate 的几何/尺度工具回归测试。

这组测试覆盖当前已经实现的两类“非拓扑”工具：

1. `link_scale`：改 joint origin 的有效长度
2. `tip_replace`：改末端 tip 的主体几何或 mesh 局部位姿

它们都不改 finger 链的 parent/child 关系，因此适合作为 mutate 第二阶段里
风险较低的实现切片。
"""

from __future__ import annotations

import math
import random

from assets.builder.hand_builders import HumanLikeHandBuilder, HumanLikeHandBuilderCfg
from assets.generator.mutate import LinkScaleCfg, LinkScaleMutator, TipReplaceCfg, TipReplaceMutator
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


def test_link_scale_mutator_rescales_target_joint_origin_only():
    """`link_scale` 应只缩放目标 joint 的 `origin.pos`，并保持方向不变。"""

    random.seed(0)
    hand = _build_allegro_hand()
    before_index = _joint_by_name(hand, "index_j1").origin.pos
    before_middle = _joint_by_name(hand, "middle_j1").origin.pos
    before_length = math.sqrt(sum(value * value for value in before_index))

    mutated = LinkScaleMutator(
        LinkScaleCfg(
            target_joints=("index_j1",),
            scale_mode="relative",
            sigma=0.1,
            clip_ratio=0.1,
        )
    ).mutate(hand)

    assert mutated is not None
    after_index = _joint_by_name(mutated, "index_j1").origin.pos
    after_middle = _joint_by_name(mutated, "middle_j1").origin.pos
    after_length = math.sqrt(sum(value * value for value in after_index))
    assert not math.isclose(after_length, before_length, rel_tol=0.0, abs_tol=1e-12)
    assert after_middle == before_middle
    assert math.isclose(after_index[0], 0.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(after_index[2], 0.0, rel_tol=0.0, abs_tol=1e-12)


def test_tip_replace_mutator_swaps_primitive_tip_body_geometry():
    """`geometry_swap` 模式应把 `cs` tip 的主体从 cylinder 换成 box。"""

    hand = _build_allegro_hand()
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


def test_tip_replace_mutator_perturbs_custom_mesh_tip_origin():
    """`mesh_perturb` 模式应对 custom mesh tip 的局部原点做比例扰动。"""

    random.seed(0)
    hand = _build_custom_tip_hand()
    before_origin = _finger_by_name(hand, "index").tip_joint.collisions[0].origin.pos

    mutated = TipReplaceMutator(
        TipReplaceCfg(
            target_fingers=("index",),
            mode="mesh_perturb",
            mesh_perturb_ratio=0.1,
        )
    ).mutate(hand)

    assert mutated is not None
    after_origin = _finger_by_name(mutated, "index").tip_joint.collisions[0].origin.pos
    assert after_origin != before_origin
    assert math.isclose(after_origin[1], before_origin[1], rel_tol=0.0, abs_tol=1e-12)  # 当前 y=0，规则应保持它贴在轴上
