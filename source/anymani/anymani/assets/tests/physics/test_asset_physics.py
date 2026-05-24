r"""`asset_physics.py` 回归测试。

这里锁住的是当前物理闭包 contract：

1. custom tip mesh 的最终 `mass / inertial` 应来自真实 mesh 体积分；
2. uniform scale 的 custom mesh 不应每次重跑体积分，数值上应满足 $m\propto s^3,\ I\propto s^5$；
3. `HandGenerator` 正式主链应在导出前自动执行 physics closure，而不是要求调用者手工补一步。
"""

from __future__ import annotations

import math

import trimesh

from assets.asset_physics import AssetPhysicsCfg, AssetPhysicsClosure
from assets.builder.hand_builders import HumanLikeHandBuilder
from assets.generator.hand_generator import HandGenerator, HandGeneratorCfg
from assets.presets import get_finger_builder_preset, make_human_like_builder_cfg


def _build_mesh_tip_hand(*, scale: float = 1.0):
    r"""构造一只 index tip 使用 custom mesh 的稳定 LEAP hand。"""

    finger_cfg = get_finger_builder_preset("leap_non_thumb_v1").replace(
        name="index",  # 这里显式命名为 index，便于后续按 finger name 取 tip joint
        parent_link="palm",  # regular finger 仍挂在 palm 上
        tip={"type": "mesh", "tip_type": "round", "scale": scale},  # custom mesh tip 只做 uniform scale
    )
    builder_cfg = make_human_like_builder_cfg(
        name=f"mesh_tip_scale_{scale:.3f}",
        family="leap",
        handedness="right",
        palm_cfg="com_leap",
        finger_cfg=finger_cfg,
        thumb_cfg="leap_thumb_v1",
    )
    return HumanLikeHandBuilder(builder_cfg).build()


def _tip_joint_by_finger_name(hand, finger_name: str):
    r"""按逻辑 finger 名返回末端 tip joint。"""

    for finger in hand.fingers:
        if finger.name == finger_name:
            return finger.tip_joint
    raise KeyError(finger_name)


def _rotation_matrix_from_rpy(rpy: tuple[float, float, float]) -> tuple[tuple[float, float, float], ...]:
    r"""返回与主代码一致的 URDF 固定轴 RPY 旋转矩阵。"""

    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _apply_rotation(matrix: tuple[tuple[float, float, float], ...], point: tuple[float, float, float]) -> tuple[float, float, float]:
    r"""计算 $R\mathbf{x}$。"""

    return (
        matrix[0][0] * point[0] + matrix[0][1] * point[1] + matrix[0][2] * point[2],
        matrix[1][0] * point[0] + matrix[1][1] * point[1] + matrix[1][2] * point[2],
        matrix[2][0] * point[0] + matrix[2][1] * point[1] + matrix[2][2] * point[2],
    )


def test_asset_physics_closure_uses_real_trimesh_mass_properties_for_custom_tip():
    r"""custom tip 的最终 inertial 应来自真实 mesh 体积分。"""

    hand = _build_mesh_tip_hand(scale=1.0)
    before_tip = _tip_joint_by_finger_name(hand, "index")

    closed = AssetPhysicsClosure(AssetPhysicsCfg()).close(hand, stage="unit_test")
    after_tip = _tip_joint_by_finger_name(closed, "index")

    mesh = trimesh.load(before_tip.collisions[0].geometry.file_path, force="mesh", process=True)
    mesh = mesh.copy()
    mesh.apply_scale(before_tip.collisions[0].geometry.scale)
    mesh.density = 650.0  # 对齐 `AssetPhysicsCfg()` 的默认密度锚点
    mass_properties = mesh.mass_properties

    rotation = _rotation_matrix_from_rpy(before_tip.collisions[0].origin.rpy)
    rotated_center = _apply_rotation(
        rotation,
        (
            float(mass_properties.center_mass[0]),
            float(mass_properties.center_mass[1]),
            float(mass_properties.center_mass[2]),
        ),
    )
    expected_origin = (
        before_tip.collisions[0].origin.pos[0] + rotated_center[0],
        before_tip.collisions[0].origin.pos[1] + rotated_center[1],
        before_tip.collisions[0].origin.pos[2] + rotated_center[2],
    )

    assert before_tip.inertial is None
    assert after_tip.metadata["inertial_source"] == "collision_closure_v1"
    assert after_tip.metadata["inertial_backend"] == "trimesh"
    assert after_tip.inertial is not None
    assert math.isclose(after_tip.inertial.mass, float(mass_properties.mass), rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(after_tip.inertial.origin.pos[0], expected_origin[0], rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(after_tip.inertial.origin.pos[1], expected_origin[1], rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(after_tip.inertial.origin.pos[2], expected_origin[2], rel_tol=0.0, abs_tol=1e-12)


def test_asset_physics_closure_respects_uniform_scale_laws_for_custom_tip():
    r"""uniform scale custom tip 应满足 $m\propto s^3,\ I\propto s^5$。"""

    closed_small = AssetPhysicsClosure(AssetPhysicsCfg()).close(_build_mesh_tip_hand(scale=1.0), stage="unit_test")
    closed_large = AssetPhysicsClosure(AssetPhysicsCfg()).close(_build_mesh_tip_hand(scale=1.5), stage="unit_test")

    small_tip = _tip_joint_by_finger_name(closed_small, "index")
    large_tip = _tip_joint_by_finger_name(closed_large, "index")

    scale_ratio = 1.5
    assert small_tip.inertial is not None
    assert large_tip.inertial is not None
    assert math.isclose(large_tip.inertial.mass, small_tip.inertial.mass * scale_ratio**3, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(large_tip.inertial.inertia.ixx, small_tip.inertial.inertia.ixx * scale_ratio**5, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(large_tip.inertial.inertia.iyy, small_tip.inertial.inertia.iyy * scale_ratio**5, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(large_tip.inertial.inertia.izz, small_tip.inertial.inertia.izz * scale_ratio**5, rel_tol=0.0, abs_tol=1e-12)


def test_hand_generator_runs_physics_closure_before_returning_hand_cfg(tmp_path):
    r"""`HandGenerator` 主链应自动执行 physics closure，而不是只在单元级 API 生效。"""

    hand = _build_mesh_tip_hand(scale=1.0)
    cfg = HandGeneratorCfg(
        mode="made",
        artifact_level="hand_cfg",
        output_dir=tmp_path,
        Made=make_human_like_builder_cfg(
            name=hand.name,
            family="leap",
            handedness="right",
            palm_cfg="com_leap",
            finger_cfg=get_finger_builder_preset("leap_non_thumb_v1").replace(
                name="index",
                parent_link="palm",
                tip={"type": "mesh", "tip_type": "round"},
            ),
            thumb_cfg="leap_thumb_v1",
        ),
        Physics=AssetPhysicsCfg(),
    )

    result = HandGenerator(cfg).generate()

    assert result is not None
    assert result.hand_cfg is not None
    tip_joint = _tip_joint_by_finger_name(result.hand_cfg, "index")
    assert tip_joint.metadata["inertial_source"] == "collision_closure_v1"
    assert tip_joint.metadata["inertial_backend"] == "trimesh"
