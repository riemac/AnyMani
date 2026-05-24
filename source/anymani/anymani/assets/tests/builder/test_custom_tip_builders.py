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

from assets.builder.hand_builders import HumanLikeHandBuilder, HumanLikeHandBuilderCfg
from assets.builder.joint_builders_custom import CustomTipBuilderCfg
from assets.exporter.urdf_writer import UrdfWriter, UrdfWriterCfg
from assets.presets import get_finger_builder_preset, make_human_like_builder_cfg


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
    assert joint.inertial is None


def test_custom_tip_builder_builds_thinner_mesh_tip_with_calibrated_anchor():
    """thinner custom tip 应使用安装底面中心锚点，而不是整 mesh 外接盒中心。"""

    cfg = CustomTipBuilderCfg(
        name="thinner_tip",
        parent="finger_link_3",
        child="thinner_tip_link",
        origin=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        tip_type="thinner",
        mesh_offset=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )

    joint = cfg.class_type(cfg).build()

    assert joint.joint_type == "fixed"
    assert joint.is_tip is True
    assert len(joint.collisions) == 1
    assert joint.collisions[0].geometry.kind == "mesh"
    assert joint.collisions[0].geometry.file_path.endswith("thinner_finger_tip_soft.stl")
    assert joint.collisions[0].geometry.scale == (0.001, 0.001, 0.001)
    assert math.isclose(joint.collisions[0].origin.pos[0], -0.0165, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(joint.collisions[0].origin.pos[1], 0.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(joint.collisions[0].origin.pos[2], -0.0095, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(joint.collisions[0].origin.rpy[1], -math.pi / 2.0, rel_tol=0.0, abs_tol=1e-12)
    assert joint.metadata["custom_tip_type"] == "thinner"
    assert joint.inertial is None


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
    assert math.isclose(tip_joint.collisions[0].origin.rpy[1], -math.pi / 2.0, rel_tol=0.0, abs_tol=1e-12)


def test_regular_finger_builder_supports_thinner_mesh_tip_recipe():
    """regular finger 的 custom mesh 路线应能透传 thinner tip preset。"""

    cfg = get_finger_builder_preset("leap_non_thumb_v1").replace(
        name="index",
        parent_link="palm",
        tip={"type": "mesh", "tip_type": "thinner"},
    )

    finger = cfg.class_type(cfg).build()
    tip_joint = finger.joints[-1]

    assert tip_joint.is_tip is True
    assert tip_joint.joint_type == "fixed"
    assert tip_joint.collisions[0].geometry.kind == "mesh"
    assert tip_joint.collisions[0].geometry.file_path.endswith("thinner_finger_tip_soft.stl")
    assert tip_joint.metadata["custom_tip_type"] == "thinner"


def test_regular_thumb_builder_applies_functional_phase_to_mesh_tip_recipe():
    r"""thumb custom tip 应额外叠加 CMC2 功能相位，而不是复用 non-thumb 相位。"""

    cfg = get_finger_builder_preset("leap_thumb_v1").replace(
        name="thumb",
        parent_link="palm",
        tip={"type": "mesh", "tip_type": "leap_cube"},
    )

    finger = cfg.class_type(cfg).build()
    tip_joint = finger.joints[-1]

    assert tip_joint.is_tip is True
    assert tip_joint.joint_type == "fixed"
    assert tip_joint.collisions[0].geometry.kind == "mesh"
    assert tip_joint.collisions[0].geometry.file_path.endswith("finger_tip_soft.stl")
    assert math.isclose(tip_joint.collisions[0].origin.rpy[1], -math.pi, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(tip_joint.collisions[0].origin.pos[0], 0.00948570692492, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(tip_joint.collisions[0].origin.pos[2], -0.0164999999586, rel_tol=0.0, abs_tol=1e-12)
    assert tip_joint.metadata["mesh_origin_rpy"] == tip_joint.collisions[0].origin.rpy


def test_urdf_writer_serializes_custom_tip_mesh_for_human_like_hand(tmp_path):
    """整手导出时，custom mesh tip 应复制到本地 `meshes/` 并写相对路径。"""

    finger_cfg = get_finger_builder_preset("leap_non_thumb_v1").replace(
        tip={"type": "mesh", "tip_type": "leap_cube"},
    )
    hand = HumanLikeHandBuilder(
        make_human_like_builder_cfg(
            name="leap_custom_tip_demo",
            family="leap",
            handedness="right",
            palm_cfg="com_leap",
            finger_cfg=finger_cfg,
            thumb_cfg="leap_thumb_v1",
        )
    ).build()

    writer = UrdfWriter(UrdfWriterCfg())
    urdf = writer.to_urdf_string(hand)

    assert "<mesh " in urdf
    assert "finger_tip_soft.stl" in urdf

    result = writer.export(hand, tmp_path)
    assert result.ok
    assert (tmp_path / "meshes" / "finger_tip_soft.stl").is_file()
    urdf_text = (tmp_path / "hand.urdf").read_text(encoding="utf-8")
    assert 'filename="meshes/finger_tip_soft.stl"' in urdf_text
    assert "/home/hac/isaac/AnyMani/source/anymani/anymani/assets/custom/tips/" not in urdf_text
