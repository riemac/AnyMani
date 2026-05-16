"""`link_scale` 几何回归测试。"""

from __future__ import annotations

import math

from assets.builder._utils import _build_cylinder_mesh
from assets.builder.hand_builders import HumanLikeHandBuilder, HumanLikeHandBuilderCfg
from assets.generator.mutate import LinkScaleCfg, LinkScaleMutator
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


def _joint_by_name(hand, joint_name: str):
    """按名字取 joint。"""

    for joint in hand.iter_joints():
        if joint.name == joint_name:
            return joint
    raise KeyError(joint_name)


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
    r"""thumb 的 shared 宽高应先按 semantic 解释，再映射到 local $(x,z)$。"""

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
    assert math.isclose(after_thumb[0], before_thumb[0] * 0.8, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(after_thumb[2], before_thumb[2] * 1.2, rel_tol=0.0, abs_tol=1e-12)


def test_link_scale_mutator_recomputes_thumb_cmc2_origin_from_cmc1_length_width_height():
    r"""CMC1 的长宽高变化应按 thumb builder 同源公式重解算 CMC2 origin。"""

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
    ]
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
    ]
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
