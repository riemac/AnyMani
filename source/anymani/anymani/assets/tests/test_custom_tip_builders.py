"""custom fingertip v1 回归测试。

这组测试锁住的不是“所有 mesh tip 都已彻底定稿”，而是当前首轮最关键的
入口契约：

1. custom tip 可以被 lowering 成合法的 `JointCfg`
2. regular finger 的 `fixed tip joint` 能切到 custom mesh 路线
3. URDF writer 能把 custom mesh tip 正常写成 `<mesh .../>`

也就是说，这里优先确保“入口与主链打通”，而不是在测试里抢先替你拍死
更高层的建模选择。
"""

from __future__ import annotations

import math

from assets.builder.finger_buiders import get_finger_builder_preset
from assets.builder.hand_builders import HumanLikeHandBuilder, HumanLikeHandBuilderCfg
from assets.builder.joint_builders_custom import CustomTipBuilderCfg
from assets.builder.palm_builders import ComPalmBuilderCfg
from assets.exporter.urdf_writer import UrdfWriter, UrdfWriterCfg


def test_custom_tip_builder_builds_round_mesh_tip_with_anchor_alignment():
    """round custom tip 应被 lower 成 mesh 几何，并保留锚点对齐后的 canonical 位姿。"""

    cfg = CustomTipBuilderCfg(
        name="round_tip",
        parent="finger_link_3",
        child="round_tip_link",
        origin=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        tip_type="round",
        mesh_offset=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )

    joint = cfg.class_type(cfg).build()

    assert joint.joint_type == "fixed"
    assert joint.is_tip is True
    assert len(joint.collisions) == 1
    assert joint.collisions[0].geometry.kind == "mesh"
    assert joint.collisions[0].geometry.file_path.endswith("round_finger_tip_soft.stl")
    assert joint.collisions[0].geometry.scale == (0.001, 0.001, 0.001)
    assert math.isclose(joint.collisions[0].origin.pos[0], -0.0164913187022, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(joint.collisions[0].origin.pos[1], 0.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(joint.collisions[0].origin.pos[2], -0.00950986387389, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(joint.collisions[0].origin.rpy[1], -math.pi / 2.0, rel_tol=0.0, abs_tol=1e-12)
    assert joint.inertial.mass > 0.0


def test_regular_finger_builder_supports_mesh_tip_recipe():
    """regular finger 的 tip recipe 现在应支持切到 custom mesh 路线。"""

    cfg = get_finger_builder_preset("leap_non_thumb_v1").replace(
        name="index",
        parent_link="palm",
        tip={"type": "mesh", "tip_type": "round"},
    )

    finger = cfg.class_type(cfg).build()
    tip_joint = finger.joints[-1]

    assert tip_joint.is_tip is True
    assert tip_joint.joint_type == "fixed"
    assert tip_joint.collisions[0].geometry.kind == "mesh"
    assert tip_joint.collisions[0].geometry.file_path.endswith("round_finger_tip_soft.stl")


def test_urdf_writer_serializes_custom_tip_mesh_for_human_like_hand():
    """整手导出时，custom mesh tip 应通过 URDF `<mesh>` 正常写出。"""

    finger_cfg = get_finger_builder_preset("leap_non_thumb_v1").replace(
        tip={"type": "mesh", "tip_type": "leap_cube"},
    )
    hand = HumanLikeHandBuilder(
        HumanLikeHandBuilderCfg(
            name="leap_custom_tip_demo",
            family="leap",
            handedness="right",
            palm_cfg=ComPalmBuilderCfg(preset="leap"),
            finger_cfg=finger_cfg,
            thumb_cfg="leap_thumb_v1",
        )
    ).build()

    urdf = UrdfWriter(UrdfWriterCfg()).to_urdf_string(hand)

    assert "<mesh " in urdf
    assert "finger_tip_soft.stl" in urdf
