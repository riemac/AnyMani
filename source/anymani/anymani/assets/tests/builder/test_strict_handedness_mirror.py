r"""严格左右手镜像契约测试。

这组测试把 generated hand 的 handedness 定义为一个可证伪的几何命题，而不是
“看起来像左手”的视觉约定。右手是唯一 canonical 真源，左手由 palm 的
$y$-$z$ 平面反射生成：

$$
S=\operatorname{diag}(-1,1,1).
$$

同一个广义坐标 $q$ 必须表达镜像后的同一功能动作，因此平移、旋转、转轴和惯量满足：

$$
\mathbf p_L=S\mathbf p_R,\qquad
R_L=SR_RS,\qquad
\mathbf a_L=\det(S)S\mathbf a_R,\qquad
I_L=SI_RS.
$$

该合同覆盖 single/composite palm、全部 finger mounts 与全部 joint child-link
embodiment；joint 名称、顺序、limits 和 DOF 则必须保持不变。
"""

from __future__ import annotations

import math

from assets.asset_schema_core import InertiaTensorCfg, PoseCfg
from assets.builder.hand_builders import HumanLikeHandBuilder
from assets.handedness import (
    compose_poses,
    lower_hand_to_handedness,
    mirror_inertia_tensor_about_yz,
    mirror_pose_about_yz,
    mirror_revolute_axis_about_yz,
)
from assets.presets.hand_presets import make_human_like_builder_cfg_from_preset

_ABS_TOL = 1e-9  # 几何 contract 使用双精度纯数学路径，容差只吸收三角函数舍入误差


def _assert_vec_close(actual, expected, *, tol: float = _ABS_TOL) -> None:
    r"""逐分量比较定长向量，保留每一维的几何诊断。"""

    assert len(actual) == len(expected)  # 左右对象必须位于同一维向量空间
    for index, (actual_value, expected_value) in enumerate(zip(actual, expected, strict=True)):
        assert math.isclose(actual_value, expected_value, rel_tol=0.0, abs_tol=tol), (
            f"component {index}: actual={actual_value}, expected={expected_value}"
        )


def _rotation_matrix_from_rpy(rpy: tuple[float, float, float]) -> tuple[tuple[float, float, float], ...]:
    r"""按 URDF 固定轴约定构造 $R=R_z(\psi)R_y(\theta)R_x(\phi)$。"""

    roll, pitch, yaw = rpy  # $(\phi,\theta,\psi)$ 分别为 fixed-axis roll/pitch/yaw
    cr, sr = math.cos(roll), math.sin(roll)  # $\cos\phi,\sin\phi$
    cp, sp = math.cos(pitch), math.sin(pitch)  # $\cos\theta,\sin\theta$
    cy, sy = math.cos(yaw), math.sin(yaw)  # $\cos\psi,\sin\psi$
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _assert_matrix_close(actual, expected, *, tol: float = _ABS_TOL) -> None:
    r"""逐元素比较 $3\times3$ 矩阵，避免欧拉角多解干扰物理旋转比较。"""

    for actual_row, expected_row in zip(actual, expected, strict=True):
        _assert_vec_close(actual_row, expected_row, tol=tol)  # 每一行都在同一 link frame 下表达


def _mirror_matrix_about_yz(matrix):
    r"""计算测试真值 $SMS$，其中 $S=\operatorname{diag}(-1,1,1)$。"""

    signs = (-1.0, 1.0, 1.0)  # YZ 平面反射只翻转 palm-frame $x$
    return tuple(
        tuple(signs[row] * matrix[row][column] * signs[column] for column in range(3))
        for row in range(3)
    )


def _build_hand(preset_name: str, handedness: str):
    r"""从同一 canonical preset 构建指定 handedness 的完整手。"""

    cfg = make_human_like_builder_cfg_from_preset(
        preset_name,
        name=f"{preset_name}_{handedness}",
        handedness=handedness,
    )  # 左右手只允许 handedness 与名字不同，其余离散锚点完全共享
    return HumanLikeHandBuilder(cfg).build()  # builder 输出必须已经是目标物理 handedness


def _assert_pose_is_yz_mirror(left: PoseCfg, right: PoseCfg) -> None:
    r"""验证两个局部位姿满足 $\mathbf p_L=S\mathbf p_R, R_L=SR_RS$。"""

    _assert_vec_close(left.pos, (-right.pos[0], right.pos[1], right.pos[2]))  # 极向量按 $S$ 反射
    expected_rotation = _mirror_matrix_about_yz(_rotation_matrix_from_rpy(right.rpy))  # $SR_RS$
    _assert_matrix_close(_rotation_matrix_from_rpy(left.rpy), expected_rotation)  # 比较旋转矩阵而非欧拉角表象


def test_pose_axis_and_inertia_follow_yz_reflection_contract() -> None:
    r"""基础几何对象必须分别遵守极向量、伪向量与二阶张量变换律。"""

    pose = PoseCfg(pos=(0.31, -0.27, 0.19), rpy=(0.42, -0.37, 1.13))  # 使用三轴均非零姿态证伪“只翻 yaw”近似
    mirrored_pose = mirror_pose_about_yz(pose)  # $\mathbf p'=S\mathbf p, R'=SRS$
    _assert_pose_is_yz_mirror(mirrored_pose, pose)

    axis = (0.2, -0.3, 0.4)  # 非轴对齐向量可同时检查三个分量的伪向量规则
    mirrored_axis = mirror_revolute_axis_about_yz(axis)  # $\det(S)S\mathbf a=(a_x,-a_y,-a_z)$
    _assert_vec_close(mirrored_axis, (0.2, 0.3, -0.4))

    inertia = InertiaTensorCfg(
        ixx=1.0,
        iyy=2.0,
        izz=3.0,
        ixy=0.11,
        ixz=-0.17,
        iyz=0.23,
    )  # 非零交叉项用于检查 $I'=SIS$，对角项应保持不变
    mirrored_inertia = mirror_inertia_tensor_about_yz(inertia)
    _assert_vec_close(
        (
            mirrored_inertia.ixx,
            mirrored_inertia.iyy,
            mirrored_inertia.izz,
            mirrored_inertia.ixy,
            mirrored_inertia.ixz,
            mirrored_inertia.iyz,
        ),
        (1.0, 2.0, 3.0, -0.11, 0.17, 0.23),
    )  # 含一个 $x$ 指标的 $I_{xy},I_{xz}$ 翻号，$I_{yz}$ 不变


def test_pose_composition_uses_exact_se3_instead_of_component_addition() -> None:
    r"""mount 与 first-joint origin 必须按 $T_{PJ}=T_{PM}T_{MJ}$ 严格复合。"""

    palm_to_mount = PoseCfg(pos=(0.10, -0.20, 0.30), rpy=(0.0, 0.0, math.pi / 2.0))  # mount 绕 $z$ 旋转 $90^\circ$
    mount_to_joint = PoseCfg(pos=(0.04, 0.00, 0.00), rpy=(0.20, -0.10, 0.00))  # local $+x$ 平移应被旋到 palm $+y$
    palm_to_joint = compose_poses(palm_to_mount, mount_to_joint)  # $T_{PJ}=T_{PM}T_{MJ}$

    _assert_vec_close(palm_to_joint.pos, (0.10, -0.16, 0.30))  # $\mathbf p_{PJ}=\mathbf p_{PM}+R_{PM}\mathbf p_{MJ}$
    expected_rotation = (
        # 旋转真值直接由矩阵乘法构造，避免把 RPY 分量相加当成 SE(3)。
        tuple(
            sum(
                _rotation_matrix_from_rpy(palm_to_mount.rpy)[row][inner]
                * _rotation_matrix_from_rpy(mount_to_joint.rpy)[inner][column]
                for inner in range(3)
            )
            for column in range(3)
        )
        for row in range(3)
    )
    _assert_matrix_close(_rotation_matrix_from_rpy(palm_to_joint.rpy), expected_rotation)


def test_single_palm_hands_are_strict_mirrors_with_same_q_contract() -> None:
    r"""Single-palm 左右手应严格镜像，并保持相同 joint identity 与 $q$ 域。"""

    right = _build_hand("single_palm_allegro", "right")  # canonical 真源
    left = _build_hand("single_palm_allegro", "left")  # 由同一真源程序化反射

    assert [finger.name for finger in left.fingers] == [finger.name for finger in right.fingers]  # finger policy identity 不变
    assert [joint.name for joint in left.iter_joints()] == [joint.name for joint in right.iter_joints()]  # joint identity/order 不变
    assert left.dof_count == right.dof_count == 16  # 镜像不是 topology 变换

    _assert_pose_is_yz_mirror(left.palm.origin, right.palm.origin)  # palm root pose 也属于整手合同
    for left_element, right_element in zip(left.palm.collisions, right.palm.collisions, strict=True):
        _assert_pose_is_yz_mirror(left_element.origin, right_element.origin)  # single box 虽对称，实例位姿仍走统一合同

    for left_finger, right_finger in zip(left.fingers, right.fingers, strict=True):
        _assert_pose_is_yz_mirror(left_finger.mount, right_finger.mount)  # 不只 thumb，所有 finger mount 都镜像
        for left_joint, right_joint in zip(left_finger.joints, right_finger.joints, strict=True):
            _assert_pose_is_yz_mirror(left_joint.origin, right_joint.origin)  # 每级 parent-child local transform 都镜像
            if right_joint.joint_type == "revolute":
                _assert_vec_close(
                    left_joint.axis,
                    mirror_revolute_axis_about_yz(right_joint.axis),
                )  # same-$q$ 要求 revolute axis 按伪向量变换
                assert left_joint.limit == right_joint.limit  # 同一 $q$ 域不允许 limits 反号或换序
            for left_element, right_element in zip(left_joint.collisions, right_joint.collisions, strict=True):
                _assert_pose_is_yz_mirror(left_element.origin, right_element.origin)  # child-link collision frame 镜像
            for left_element, right_element in zip(left_joint.visuals, right_joint.visuals, strict=True):
                _assert_pose_is_yz_mirror(left_element.origin, right_element.origin)  # visual 与 collision 同步镜像


def test_composite_palm_boxes_are_mirrored_without_left_preset() -> None:
    r"""Composite palm 应镜像 canonical box recipe，而不是维护第二份 left 数据表。"""

    right = _build_hand("com_palm_leap", "right")  # 多 box 组合整体不关于 YZ 对称
    left = _build_hand("com_palm_leap", "left")  # 每个 box 尺寸不变，只镜像实例位姿

    assert len(left.palm.collisions) == len(right.palm.collisions) > 1  # 确认测试覆盖真实 composite recipe
    for left_box, right_box in zip(left.palm.collisions, right.palm.collisions, strict=True):
        assert left_box.geometry == right_box.geometry  # box 本体尺寸在反射下保持不变
        _assert_pose_is_yz_mirror(left_box.origin, right_box.origin)  # 非对称性只通过实例位姿反射


def test_handedness_lowering_is_an_involution_on_physical_fields() -> None:
    r"""连续执行两次 YZ 反射应恢复右手物理字段，防止 post-mutate 往返漂移。"""

    right = _build_hand("single_palm_leap", "right")  # 右手 canonical physical hand
    left = lower_hand_to_handedness(right, "left")  # 第一次反射：right -> left
    restored = lower_hand_to_handedness(left, "right")  # 第二次反射：left -> right

    assert restored.handedness == "right"  # 顶层 handedness 与目标一致
    assert [joint.name for joint in restored.iter_joints()] == [joint.name for joint in right.iter_joints()]  # identity 不漂移
    for restored_finger, right_finger in zip(restored.fingers, right.fingers, strict=True):
        _assert_vec_close(restored_finger.mount.pos, right_finger.mount.pos)  # 两次 $S$ 满足 $S^2=I$
        _assert_matrix_close(
            _rotation_matrix_from_rpy(restored_finger.mount.rpy),
            _rotation_matrix_from_rpy(right_finger.mount.rpy),
        )
        for restored_joint, right_joint in zip(restored_finger.joints, right_finger.joints, strict=True):
            _assert_vec_close(restored_joint.origin.pos, right_joint.origin.pos)
            _assert_matrix_close(
                _rotation_matrix_from_rpy(restored_joint.origin.rpy),
                _rotation_matrix_from_rpy(right_joint.origin.rpy),
            )
            _assert_vec_close(restored_joint.axis, right_joint.axis)  # 伪向量变换同样满足二次恢复
