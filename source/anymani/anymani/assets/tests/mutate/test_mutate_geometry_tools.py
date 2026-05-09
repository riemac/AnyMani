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
from assets.builder._utils import _build_cylinder_mesh
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


def test_link_scale_mutator_rescales_link_length_and_advances_next_joint_origin():
    """`link_scale` 应改变自身 link 长度，并用新 $L_i+d_i$ 推进下游 joint。"""

    hand = _build_allegro_hand()
    before_index_origin = _joint_by_name(hand, "index_j1").origin.pos
    before_index_size = _joint_by_name(hand, "index_j1").collisions[0].geometry.size
    before_next_origin = _joint_by_name(hand, "index_j2").origin.pos
    before_middle = _joint_by_name(hand, "middle_j1").origin.pos

    mutated = LinkScaleMutator(
        LinkScaleCfg(
            scale_type="rel",
            link_scale=(1.1, 1.1),
            distrib="uniform",
        )
    ).mutate(hand, sampled_params={"index_j1": 1.1})

    assert mutated is not None
    after_index_origin = _joint_by_name(mutated, "index_j1").origin.pos
    after_index_size = _joint_by_name(mutated, "index_j1").collisions[0].geometry.size
    after_next_origin = _joint_by_name(mutated, "index_j2").origin.pos
    after_middle = _joint_by_name(mutated, "middle_j1").origin.pos
    assert math.isclose(after_index_size[1], before_index_size[1] * 1.1, rel_tol=0.0, abs_tol=1e-12)
    assert after_next_origin[1] > before_next_origin[1]
    assert after_index_origin == before_index_origin
    assert after_middle == before_middle


def test_link_scale_mutator_uses_shared_width_and_height_scale_for_box_links():
    r"""`Vector6` 模式下，宽度/高度应是全手共享随机变量，而不是 per-link 独立采样。"""

    hand = _build_allegro_hand()
    before_index = _joint_by_name(hand, "index_j1").collisions[0].geometry.size
    before_middle = _joint_by_name(hand, "middle_j1").collisions[0].geometry.size

    mutated = LinkScaleMutator(
        LinkScaleCfg(
            scale_type="rel",
            link_scale=(1.0, 1.0, 1.2, 1.2, 0.8, 0.8),
            distrib="uniform",
        )
    ).mutate(
        hand,
        sampled_params={
            "shared::width": 1.2,
            "shared::height": 0.8,
            "index_j1::length": 1.0,
            "middle_j1::length": 1.0,
        },
    )

    assert mutated is not None
    after_index = _joint_by_name(mutated, "index_j1").collisions[0].geometry.size
    after_middle = _joint_by_name(mutated, "middle_j1").collisions[0].geometry.size
    assert math.isclose(after_index[0], before_index[0] * 1.2, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(after_middle[0], before_middle[0] * 1.2, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(after_index[2], before_index[2] * 0.8, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(after_middle[2], before_middle[2] * 0.8, rel_tol=0.0, abs_tol=1e-12)


def test_link_scale_mutator_maps_thumb_semantic_width_height_to_swapped_local_axes():
    r"""thumb 的 shared 宽高应先按 semantic 解释，再映射到 local $(x,z)$。

    当前 contract 是：

    - non-thumb：local $x \leftarrow$ semantic width，local $z \leftarrow$ semantic height
    - thumb：local $x \leftarrow$ semantic height，local $z \leftarrow$ semantic width

    这里直接锁住 thumb 非 CMC1 段的一条最小断言，避免后续又把 shared sample
    机械地当作“所有 link 的 local x/z 缩放”。
    """

    hand = _build_allegro_hand()
    before_thumb = _joint_by_name(hand, "thumb_j1").collisions[0].geometry.size

    mutated = LinkScaleMutator(
        LinkScaleCfg(
            scale_type="rel",
            link_scale=(1.0, 1.0, 1.2, 1.2, 0.8, 0.8),
            distrib="uniform",
        )
    ).mutate(
        hand,
        sampled_params={
            "shared::width": 1.2,
            "shared::height": 0.8,
            "thumb_j1::length": 1.0,
        },
    )

    assert mutated is not None
    after_thumb = _joint_by_name(mutated, "thumb_j1").collisions[0].geometry.size
    assert math.isclose(after_thumb[0], before_thumb[0] * 0.8, rel_tol=0.0, abs_tol=1e-12)  # thumb local $x$ 吃 semantic 高度
    assert math.isclose(after_thumb[2], before_thumb[2] * 1.2, rel_tol=0.0, abs_tol=1e-12)  # thumb local $z$ 吃 semantic 宽度


def test_link_scale_mutator_recomputes_thumb_cmc2_origin_from_cmc1_length_width_height():
    r"""CMC1 的长宽高变化应按 thumb builder 同源公式重解算 CMC2 origin。

    注意这里的宽高不再是“直接写 CMC1 / CMC2 local x/z 的 shared sample”，而是：

    - 先采样全手 semantic 宽/高；
    - 再对 thumb 做 `local x \leftarrow semantic height`、
      `local z \leftarrow semantic width` 的轴语义映射。

    因此 `CMC1 -> CMC2` 的边界对齐，也必须使用“下游 CMC2 经过同一次 semantic
    宽高映射后的新截面”，不能再拿 mutate 前的 `before_cmc2_size` 直接代入。
    """

    hand = _build_allegro_hand()
    cmc1 = _joint_by_name(hand, "thumb_j0")
    cmc2 = _joint_by_name(hand, "thumb_j1")
    before_cmc2_origin = cmc2.origin.pos
    before_cmc1_size = cmc1.collisions[0].geometry.size
    before_cmc2_size = cmc2.collisions[0].geometry.size
    before_cmc1_origin = cmc1.collisions[0].origin.pos

    mutated = LinkScaleMutator(
        LinkScaleCfg(
            scale_type="rel",
            link_scale=(1.2, 1.2, 1.1, 1.1, 0.9, 0.9),
            distrib="uniform",
        )
    ).mutate(
        hand,
        sampled_params={
            "shared::width": 1.1,
            "shared::height": 0.9,
            "thumb_j0::length": 1.2,
        },
    )

    assert mutated is not None
    after_cmc2_origin = _joint_by_name(mutated, "thumb_j1").origin.pos
    expected_x = ((before_cmc1_size[0] * 0.9) - (before_cmc2_size[0] * 0.9)) / 2.0
    expected_y = before_cmc1_origin[1] + (before_cmc1_size[1] * 1.2) / 2.0
    expected_z = before_cmc1_origin[2] - ((before_cmc1_size[2] * 1.1) - (before_cmc2_size[2] * 1.1)) / 2.0
    assert after_cmc2_origin != before_cmc2_origin
    assert math.isclose(after_cmc2_origin[0], expected_x, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(after_cmc2_origin[1], expected_y, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(after_cmc2_origin[2], expected_z, rel_tol=0.0, abs_tol=1e-12)


def test_link_scale_mutator_promotes_cylinder_to_elliptic_cylinder_on_anisotropic_cross_section():
    r"""圆柱在非等径宽高缩放下应提升为 `elliptic_cylinder`。"""

    finger_cfg = get_finger_builder_preset("allegro_non_thumb_v1").replace(
        radius=0.01,
        width=None,
        height=None,
    )
    finger_cfg.mesh_shape = [
        _build_cylinder_mesh(length=shape["length"], radius=0.01, offset=shape["offset"])
        for shape in finger_cfg.mesh_shape
    ]  # 预设 copy 后仍保留旧 box mesh_shape；这里显式重建成 cylinder 链，才能测试 cylinder→elliptic_cylinder 互转
    hand = HumanLikeHandBuilder(
        make_human_like_builder_cfg(
            name="allegro_cylinder_mutate_demo",
            family="allegro",
            handedness="right",
            palm_cfg="com_allegro",
            finger_cfg=finger_cfg,
            thumb_cfg="allegro_thumb_v1",
        )
    ).build()

    mutated = LinkScaleMutator(
        LinkScaleCfg(
            scale_type="rel",
            link_scale=(1.0, 1.0, 1.3, 1.3, 0.7, 0.7),
            distrib="uniform",
        )
    ).mutate(
        hand,
        sampled_params={
            "shared::width": 1.3,
            "shared::height": 0.7,
            "index_j1::length": 1.0,
        },
    )

    assert mutated is not None
    geom = _joint_by_name(mutated, "index_j1").collisions[0].geometry
    assert geom.kind == "elliptic_cylinder"


def test_link_scale_mutator_preserves_cylinder_when_cross_section_stays_isotropic():
    r"""圆柱在等径宽高缩放下应继续保持 `cylinder`。"""

    finger_cfg = get_finger_builder_preset("allegro_non_thumb_v1").replace(
        radius=0.01,
        width=None,
        height=None,
    )
    finger_cfg.mesh_shape = [
        _build_cylinder_mesh(length=shape["length"], radius=0.01, offset=shape["offset"])
        for shape in finger_cfg.mesh_shape
    ]  # 等径缩放测试也需要先把 copy 后的非拇指链显式改成 cylinder mesh
    hand = HumanLikeHandBuilder(
        make_human_like_builder_cfg(
            name="allegro_cylinder_mutate_iso_demo",
            family="allegro",
            handedness="right",
            palm_cfg="com_allegro",
            finger_cfg=finger_cfg,
            thumb_cfg="allegro_thumb_v1",
        )
    ).build()

    mutated = LinkScaleMutator(
        LinkScaleCfg(
            scale_type="rel",
            link_scale=(1.0, 1.0, 1.1, 1.1, 1.1, 1.1),
            distrib="uniform",
        )
    ).mutate(
        hand,
        sampled_params={
            "shared::width": 1.1,
            "shared::height": 1.1,
            "index_j1::length": 1.0,
        },
    )

    assert mutated is not None
    geom = _joint_by_name(mutated, "index_j1").collisions[0].geometry
    assert geom.kind == "cylinder"


def test_tip_replace_mutator_swaps_primitive_tip_body_geometry():
    """`geometry_swap` 模式应把 `cs` tip 的主体从 cylinder 换成 box。"""

    hand = _build_allegro_hand()
    before_tip_joint = _finger_by_name(hand, "index").tip_joint
    before_inertial = before_tip_joint.inertial
    mutated = TipReplaceMutator(
        TipReplaceCfg(
            mode="geometry_swap",
            target_geometry="box",
            self_mode="general",
            scale=(1.0, 1.0),
        )
    ).mutate(hand, sampled_params={"index::scale": 1.0})

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
            mode="mesh_perturb",
            self_mode="general",
            scale=(1.1, 1.1),
        )
    ).mutate(
        hand,
        sampled_params={
            "index::scale": 1.1,
        },
    )

    assert mutated is not None
    after_origin = _finger_by_name(mutated, "index").tip_joint.collisions[0].origin.pos
    assert after_origin != before_origin
    assert math.isclose(after_origin[1], before_origin[1], rel_tol=0.0, abs_tol=1e-12)  # 当前 y=0，规则应保持它贴在轴上
