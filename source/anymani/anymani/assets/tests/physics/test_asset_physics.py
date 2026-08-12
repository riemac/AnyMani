r"""`asset_physics.py` 回归测试。

这里锁住的是当前物理闭包 contract：

1. custom tip mesh 的最终 `mass / inertial` 应来自真实 mesh 体积分；
2. uniform scale 的 custom mesh 不应每次重跑体积分，数值上应满足 $m\propto s^3,\ I\propto s^5$；
3. `HandGenerator` 正式主链应在导出前自动执行 physics closure，而不是要求调用者手工补一步。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import trimesh
from assets.asset_physics import AssetPhysicsCfg, AssetPhysicsClosure
from assets.asset_schema_core import CollisionGeometryCfg, VisualGeometryCfg
from assets.asset_schema_embodiment import FingerCfg, HandCfg, JointCfg, PalmCfg
from assets.builder.hand_builders import HumanLikeHandBuilder
from assets.generator.hand_generator import HandGenerator, HandGeneratorCfg
from assets.handedness import lower_hand_to_handedness
from assets.presets import get_finger_builder_preset, make_human_like_builder_cfg
from assets.procedural_meshes import materialize_hand_procedural_meshes


def _build_mesh_tip_hand(*, scale: float = 1.0):
    r"""构造一只所有 fingertip 都已是 custom mesh 的稳定 LEAP hand。

    这里是 `asset_physics.py` 的直接单元测试，因此输入必须已经是“最终 collision
    几何”。procedural `cs` 的 materialization 属于 generator / exporter 前置阶段，
    不应混进这个 helper，否则测试会把阶段边界错误转嫁给 physics closure。
    """

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
        thumb_cfg=get_finger_builder_preset("leap_thumb_v1").replace(
            tip={"type": "mesh", "tip_type": "round", "scale": scale},
        ),
    )
    return HumanLikeHandBuilder(builder_cfg).build()


def _tip_joint_by_finger_name(hand, finger_name: str):
    r"""按逻辑 finger 名返回末端 tip joint。"""

    for finger in hand.fingers:
        if finger.name == finger_name:
            return finger.tip_joint
    raise KeyError(finger_name)


def _build_asymmetric_tetrahedron_hand(mesh_path: Path) -> HandCfg:
    r"""构造携带非对称 watertight 四面体 mesh 的 canonical right hand。

    顶点刻意不关于局部 $y$-$z$ 平面对称，因此其质心满足 $c_x\ne0$，惯量也含
    非零 $I_{xy}/I_{xz}$。这使测试能同时证伪“只镜像 geometry origin”和
    “复制原 mesh 但未修改 triangle winding”两种错误实现。
    """

    vertices = np.asarray(
        (
            (0.001, 0.002, 0.003),
            (0.031, 0.004, 0.006),
            (0.007, 0.043, 0.009),
            (0.011, 0.013, 0.057),
        ),
        dtype=np.float64,
    )  # 四个顶点在三轴上均不对称，单位 m
    faces = np.asarray(
        (
            (0, 2, 1),
            (0, 1, 3),
            (0, 3, 2),
            (1, 2, 3),
        ),
        dtype=np.int64,
    )  # 外法向一致的封闭四面体 triangle winding
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0.0
    mesh.export(mesh_path)  # 测试源 mesh 写在 tmp_path，不污染项目资产目录

    geometry = {"type": "mesh", "file_path": str(mesh_path), "scale": (1.0, 1.0, 1.0)}
    tip_joint = JointCfg(
        name="index_tip",
        parent="palm",
        child="index_tip_link",
        joint_type="fixed",
        collisions=[CollisionGeometryCfg(name="tip_col", geometry=geometry)],
        visuals=[VisualGeometryCfg(name="tip_vis", geometry=geometry)],
        is_tip=True,
        metadata={"finger_name": "index", "custom_tip_type": "asymmetric_tetrahedron"},
    )  # fixed joint 只承载一个非对称 child-link mesh
    return HandCfg(
        name="asymmetric_mesh_right",
        family="unit_test",
        handedness="right",
        palm=PalmCfg(name="palm"),
        fingers=[FingerCfg(name="index", parent_link="palm", joints=[tip_joint])],
    )


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


def test_left_asymmetric_mesh_is_reflected_before_physics_with_consistent_winding(tmp_path: Path):
    r"""Left custom mesh 必须在 physics closure 前烘焙 $x\mapsto-x$ 与反向绕序。

    对 canonical mesh 的单位密度质量属性，严格镜像应满足：

    $$
    V_L=V_R,\qquad \mathbf c_L=S\mathbf c_R,\qquad I_L=SI_RS.
    $$

    反射会改变三角面手性，因此每个 face 的顶点顺序还必须从 $(i,j,k)$ 变为
    $(i,k,j)$，否则 signed volume 为负且外法向朝内。
    """

    source_path = tmp_path / "asymmetric_tetrahedron.stl"
    right = _build_asymmetric_tetrahedron_hand(source_path)
    source_bytes = source_path.read_bytes()  # canonical source 是只读真源，materializer 不得原地覆盖
    left = lower_hand_to_handedness(right, "left")  # mesh schema 先标记为待局部 YZ 反射
    left_geometry_before = left.fingers[0].tip_joint.collisions[0].geometry
    assert left_geometry_before.reflected_about_yz is True

    mesh_root = tmp_path / "meshes"
    materialized, written_paths = materialize_hand_procedural_meshes(left, mesh_root_dir=mesh_root)
    materialized_again, second_written_paths = materialize_hand_procedural_meshes(left, mesh_root_dir=mesh_root)
    collision_geometry = materialized.fingers[0].tip_joint.collisions[0].geometry
    visual_geometry = materialized.fingers[0].tip_joint.visuals[0].geometry

    assert len(written_paths) == 1  # collision/visual 共享同一镜像文件，只记一次候选期写入
    assert written_paths[0].is_file() and written_paths[0].parent == mesh_root
    assert second_written_paths == []  # 相同 canonical source 的镜像缓存应稳定复用
    assert materialized_again.fingers[0].tip_joint.collisions[0].geometry.file_path == str(written_paths[0])
    assert collision_geometry.file_path == visual_geometry.file_path == str(written_paths[0])
    assert collision_geometry.reflected_about_yz is False  # 文件已经烘焙反射，不允许 exporter/restore 二次镜像
    assert source_path.read_bytes() == source_bytes  # canonical source 不被原地修改

    right_mesh = trimesh.load(source_path, force="mesh", process=True)
    left_mesh = trimesh.load(written_paths[0], force="mesh", process=True)
    assert left_mesh.is_watertight and left_mesh.is_winding_consistent
    assert left_mesh.volume > 0.0  # face winding 修正后 signed volume 保持正值
    assert math.isclose(left_mesh.volume, right_mesh.volume, rel_tol=0.0, abs_tol=1e-12)
    assert np.allclose(
        left_mesh.center_mass,
        np.asarray((-right_mesh.center_mass[0], right_mesh.center_mass[1], right_mesh.center_mass[2])),
        rtol=0.0,
        atol=1e-9,
    )  # $\mathbf c_L=S\mathbf c_R$
    reflection = np.diag((-1.0, 1.0, 1.0))  # $S=\operatorname{diag}(-1,1,1)$
    assert np.allclose(
        left_mesh.moment_inertia,
        reflection @ right_mesh.moment_inertia @ reflection,
        rtol=0.0,
        atol=1e-12,
    )  # $I_L=SI_RS$

    closed = AssetPhysicsClosure(AssetPhysicsCfg()).close(materialized, stage="unit_test")
    closed_tip = closed.fingers[0].tip_joint
    assert closed_tip.inertial is not None
    assert math.isclose(closed_tip.inertial.mass, left_mesh.volume * 650.0, rel_tol=0.0, abs_tol=1e-12)


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


def test_hand_generator_materializes_cs_tip_before_physics_and_uses_fingertip_density(tmp_path):
    r"""generator 应在 physics closure 前把 procedural `cs` 物化为 OBJ。

    这里同时锁住密度通道：`cs` 虽然最终由 `trimesh` 计算质量属性，但它仍是
    procedural primitive fingertip，质量应使用 `density.fingertip`，不能误用
    `density.custom_tip`。
    """

    cfg = HandGeneratorCfg(
        mode="made",
        artifact_level="hand_cfg",
        output_dir=tmp_path,
        Made=make_human_like_builder_cfg(
            name="procedural_cs_density_demo",
            family="leap",
            handedness="right",
            palm_cfg="com_leap",
            finger_cfg=get_finger_builder_preset("leap_non_thumb_v1"),
            thumb_cfg="leap_thumb_v1",
        ),
        Physics=AssetPhysicsCfg(
            density={"default": 1.0, "fingertip": 100.0, "custom_tip": 1000.0},
        ),
    )

    result = HandGenerator(cfg).generate()

    assert result is not None
    tip_joint = _tip_joint_by_finger_name(result.hand_cfg, "index")
    mesh_path = Path(tip_joint.collisions[0].geometry.file_path)
    mesh = trimesh.load(mesh_path, force="mesh", process=True)
    expected_mass = float(mesh.volume) * 100.0

    assert mesh_path.is_file()
    assert "procedural://" not in tip_joint.collisions[0].geometry.file_path
    assert tip_joint.metadata["procedural_mesh_kind"] == "cs_tip"
    assert tip_joint.metadata["inertial_backend"] == "trimesh"
    assert tip_joint.inertial is not None
    assert math.isclose(tip_joint.inertial.mass, expected_mass, rel_tol=0.0, abs_tol=1e-9)


def test_left_procedural_cs_reuses_axisymmetric_mesh_without_reflected_duplicate(tmp_path: Path):
    r"""轴对称 ``cs`` fingertip 的 left/right 应复用同一参数化 mesh。

    ``cs`` 关于其局部 $y$ 轴旋转对称，也关于局部 $y$-$z$ 平面对称；严格整手
    镜像只需变换 geometry frame，不需要再生成 ``*_yz_reflect_*`` 文件。该断言
    防止 handedness cache 数量无意义翻倍。
    """

    right = HumanLikeHandBuilder(
        make_human_like_builder_cfg(
            name="axisymmetric_cs_right",
            family="leap",
            handedness="right",
            palm_cfg="single_box_leap",
            finger_cfg="leap_non_thumb_v1",
            thumb_cfg="leap_thumb_v1",
        )
    ).build()
    left = lower_hand_to_handedness(right, "left")
    mesh_root = tmp_path / "meshes"

    materialized_right, right_written = materialize_hand_procedural_meshes(right, mesh_root_dir=mesh_root)
    materialized_left, left_written = materialize_hand_procedural_meshes(left, mesh_root_dir=mesh_root)

    right_tip = _tip_joint_by_finger_name(materialized_right, "index")
    left_tip = _tip_joint_by_finger_name(materialized_left, "index")
    assert right_written  # 首次 right materialization 真实生成参数化 cs cache
    assert left_written == []  # left 复用同一 cache，不重复发布 handedness mesh
    assert right_tip.collisions[0].geometry.file_path == left_tip.collisions[0].geometry.file_path
    assert left_tip.collisions[0].geometry.reflected_about_yz is False  # 轴对称真值已完成 lowering
    assert not list(mesh_root.glob("*_yz_reflect_v1_*"))  # 不产生无意义的 left duplicate
