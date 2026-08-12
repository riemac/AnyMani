r"""Generated hand 的严格左右手几何映射。

本模块把 handedness 定义为一个作用于完整 ``HandCfg`` 的空间反射，而不是
某根手指的局部数值特例。右手是唯一 canonical 真源；左手由 palm 的
$y$-$z$ 平面反射得到：

$$
S=\operatorname{diag}(-1,1,1),\qquad \det(S)=-1.
$$

由于 URDF 中的 link 与 joint frame 都必须保持右手坐标系，不能把 $S$ 直接
写成一个旋转。正确做法是同时反射 parent 与 child frame，使每个局部刚体位姿满足：

$$
\mathbf p'=S\mathbf p,\qquad R'=SRS.
$$

Revolute axis 是轴向量（伪向量），同一个广义坐标 $q$ 要表达镜像后的同一功能
动作时必须满足：

$$
\mathbf a'=\det(S)S\mathbf a=(a_x,-a_y,-a_z).
$$

惯量是二阶张量，镜像后满足 $I'=SIS$。这些变换共同保证任意运动学链在同一
$q$ 下满足 $T_L(q)=ST_R(q)S$，因此运行时不需要额外 action sign map。
"""

from __future__ import annotations

import math
from typing import Literal

from .asset_schema_core import (
    CollisionGeometryCfg,
    InertialCfg,
    InertiaTensorCfg,
    MeshGeometryCfg,
    PoseCfg,
    Vector3,
    VisualGeometryCfg,
)
from .asset_schema_embodiment import FingerCfg, HandCfg, JointCfg, PalmCfg

HandTarget = Literal["left", "right"]
"""Generated human-like hand 支持的两个物理 handedness。"""


HANDEDNESS_CONTRACT_VERSION = "1.0"
"""严格整手镜像 sidecar contract 的首个版本。"""


def rpy_rotation_matrix(rpy: Vector3) -> tuple[Vector3, Vector3, Vector3]:
    r"""构造 URDF fixed-axis RPY 对应的旋转矩阵。

    URDF 使用：

    $$
    R(\phi,\theta,\psi)=R_z(\psi)R_y(\theta)R_x(\phi).
    $$

    Args:
        rpy (Vector3): $(\phi,\theta,\psi)$，单位为 rad。

    Returns:
        tuple[Vector3, Vector3, Vector3]: 按行存储的 $3\times3$ 旋转矩阵。
    """

    roll, pitch, yaw = rpy  # $(\phi,\theta,\psi)$ 为 URDF fixed-axis 欧拉角
    cr, sr = math.cos(roll), math.sin(roll)  # $\cos\phi,\sin\phi$
    cp, sp = math.cos(pitch), math.sin(pitch)  # $\cos\theta,\sin\theta$
    cy, sy = math.cos(yaw), math.sin(yaw)  # $\cos\psi,\sin\psi$
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def matrix_to_rpy(matrix: tuple[Vector3, Vector3, Vector3]) -> Vector3:
    r"""把旋转矩阵恢复为与 URDF 一致的 fixed-axis RPY。

    常规分支采用：

    $$
    \theta=\operatorname{atan2}(-R_{20},\sqrt{R_{00}^2+R_{10}^2}),\quad
    \phi=\operatorname{atan2}(R_{21},R_{22}),\quad
    \psi=\operatorname{atan2}(R_{10},R_{00}).
    $$

    在 $|\cos\theta|\approx0$ 的 gimbal-lock 点，roll 固定取 $0$，把可辨识的
    合成转角收口到 yaw。该选择只规范化欧拉角表象，不改变旋转矩阵真值。

    Args:
        matrix: 按行存储的正交旋转矩阵。

    Returns:
        Vector3: 等价的 URDF RPY，单位为 rad。
    """

    horizontal_norm = math.hypot(matrix[0][0], matrix[1][0])  # $|\cos\theta|$
    pitch = math.atan2(-matrix[2][0], horizontal_norm)  # $\theta\in[-\pi/2,\pi/2]$
    if horizontal_norm > 1e-12:
        roll = math.atan2(matrix[2][1], matrix[2][2])  # 非奇异点的 $\phi$
        yaw = math.atan2(matrix[1][0], matrix[0][0])  # 非奇异点的 $\psi$
    else:
        roll = 0.0  # gimbal lock 下固定一个规范代表，不改变 $R$
        yaw = math.atan2(-matrix[0][1], matrix[1][1])  # 把可辨识合成角写入 $\psi$
    return (roll, pitch, yaw)


def _matrix_multiply(
    lhs: tuple[Vector3, Vector3, Vector3],
    rhs: tuple[Vector3, Vector3, Vector3],
) -> tuple[Vector3, Vector3, Vector3]:
    r"""计算两个 $3\times3$ 矩阵的乘积 $C=AB$。"""

    return tuple(
        tuple(sum(lhs[row][inner] * rhs[inner][column] for inner in range(3)) for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _apply_rotation(rotation: tuple[Vector3, Vector3, Vector3], point: Vector3) -> Vector3:
    r"""计算 $\mathbf y=R\mathbf x$。"""

    return tuple(
        sum(rotation[row][column] * point[column] for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def compose_poses(parent: PoseCfg, local: PoseCfg) -> PoseCfg:
    r"""严格复合两个局部刚体位姿。

    若 ``parent`` 表示 $T_{AB}$，``local`` 表示 $T_{BC}$，则输出 $T_{AC}$：

    $$
    R_{AC}=R_{AB}R_{BC},\qquad
    \mathbf p_{AC}=\mathbf p_{AB}+R_{AB}\mathbf p_{BC}.
    $$

    Args:
        parent (PoseCfg): parent frame 到中间 frame 的位姿。
        local (PoseCfg): 中间 frame 到 child frame 的位姿。

    Returns:
        PoseCfg: parent frame 到 child frame 的严格 SE(3) 复合位姿。
    """

    parent_rotation = rpy_rotation_matrix(parent.rpy)  # $R_{AB}$
    local_rotation = rpy_rotation_matrix(local.rpy)  # $R_{BC}$
    rotated_local_position = _apply_rotation(parent_rotation, local.pos)  # $R_{AB}\mathbf p_{BC}$
    return PoseCfg(
        pos=tuple(
            parent.pos[index] + rotated_local_position[index]
            for index in range(3)
        ),  # type: ignore[arg-type]  # $\mathbf p_{AC}=\mathbf p_{AB}+R_{AB}\mathbf p_{BC}$
        rpy=matrix_to_rpy(_matrix_multiply(parent_rotation, local_rotation)),  # $R_{AC}=R_{AB}R_{BC}$
    )


def mirror_pose_about_yz(pose: PoseCfg) -> PoseCfg:
    r"""关于 palm 的 $y$-$z$ 平面反射一个局部位姿。

    对 URDF fixed-axis RPY，$R'=SRS$ 可精确写成：

    $$
    (\phi',\theta',\psi')=(\phi,-\theta,-\psi).
    $$

    该闭式表达在任意 roll/pitch/yaw 下都成立，不是“只翻 yaw”的小角度近似。

    Args:
        pose (PoseCfg): canonical right-hand 局部位姿。

    Returns:
        PoseCfg: 严格镜像后的局部位姿。
    """

    return PoseCfg(
        pos=(-pose.pos[0], pose.pos[1], pose.pos[2]),  # $\mathbf p'=S\mathbf p$
        rpy=(pose.rpy[0], -pose.rpy[1], -pose.rpy[2]),  # $R'=SRS$
    )


def mirror_revolute_axis_about_yz(axis: Vector3) -> Vector3:
    r"""按伪向量规律镜像 revolute axis。

    Args:
        axis (Vector3): right-hand joint frame 中的转轴 $\mathbf a$。

    Returns:
        Vector3: $(a_x,-a_y,-a_z)=\det(S)S\mathbf a$。
    """

    return (axis[0], -axis[1], -axis[2])  # 同一 $q$ 下保持镜像功能动作


def mirror_inertia_tensor_about_yz(inertia: InertiaTensorCfg) -> InertiaTensorCfg:
    r"""按 $I'=SIS$ 镜像惯量张量。

    对 $S=\operatorname{diag}(-1,1,1)$，对角项不变；含一个 $x$ 指标的
    $I_{xy},I_{xz}$ 翻号，$I_{yz}$ 不变。

    Args:
        inertia (InertiaTensorCfg): right-hand link frame 下的质心惯量。

    Returns:
        InertiaTensorCfg: left-hand link frame 下的镜像惯量。
    """

    return InertiaTensorCfg(
        ixx=inertia.ixx,  # $s_x^2I_{xx}=I_{xx}$
        iyy=inertia.iyy,  # $s_y^2I_{yy}=I_{yy}$
        izz=inertia.izz,  # $s_z^2I_{zz}=I_{zz}$
        ixy=-inertia.ixy,  # $s_xs_yI_{xy}=-I_{xy}$
        ixz=-inertia.ixz,  # $s_xs_zI_{xz}=-I_{xz}$
        iyz=inertia.iyz,  # $s_ys_zI_{yz}=I_{yz}$
    )


def _mirror_inertial(inertial: InertialCfg | None) -> InertialCfg | None:
    r"""镜像 link inertial，同时保持质量与数值 padding 不变。"""

    if inertial is None:
        return None  # 未闭包的 link 不人工制造惯量占位
    mirrored = inertial.copy()  # 避免 dataclass replace 重新执行 padding 并把它重复加到对角惯量
    mirrored.origin = mirror_pose_about_yz(inertial.origin)  # 质心与惯性参考系位姿按极向量/旋转规则镜像
    mirrored.inertia = mirror_inertia_tensor_about_yz(inertial.inertia)  # $I'=SIS$
    return mirrored  # mass、inertia_padding 与其它闭包证书保持原值


def _mirror_mesh_geometry(geometry):
    r"""切换 mesh 的局部 YZ 反射标记；primitive 本体在自身 frame 下保持不变。"""

    if not isinstance(geometry, MeshGeometryCfg):
        return geometry.copy()  # box/cylinder/sphere 的局部形状关于自身 YZ 平面对称
    return geometry.replace(reflected_about_yz=not geometry.reflected_about_yz)  # 两次反射恢复 canonical mesh


def _mirror_collision(element: CollisionGeometryCfg) -> CollisionGeometryCfg:
    r"""镜像一个 collision instance 的局部位姿与非对称 mesh 本体。"""

    return element.replace(
        geometry=_mirror_mesh_geometry(element.geometry),  # 非对称 mesh 需要在 materialization 阶段烘焙反射
        origin=mirror_pose_about_yz(element.origin),  # geometry frame 同时满足 $G'=SGS$
    )


def _mirror_visual(element: VisualGeometryCfg) -> VisualGeometryCfg:
    r"""镜像一个 visual instance，并保持材质语义不变。"""

    return element.replace(
        geometry=_mirror_mesh_geometry(element.geometry),  # visual 与 collision 必须引用同一手性几何
        origin=mirror_pose_about_yz(element.origin),  # $G'=SGS$
    )


def _mirror_joint(joint: JointCfg) -> JointCfg:
    r"""镜像 joint transform、axis 与其 child-link embodiment。"""

    axis = (
        mirror_revolute_axis_about_yz(joint.axis)
        if joint.joint_type == "revolute"
        else joint.axis
    )  # fixed joint 的占位轴没有物理运动语义，不参与伪向量变换
    return joint.replace(
        origin=mirror_pose_about_yz(joint.origin),  # parent->joint 的局部刚体位姿
        axis=axis,  # revolute axis 保证同一个 $q$ 代表同一功能动作
        inertial=_mirror_inertial(joint.inertial),  # child-link 质量不变，COM/惯量镜像
        collisions=[_mirror_collision(element) for element in joint.collisions],  # 接触几何完整镜像
        visuals=[_mirror_visual(element) for element in joint.visuals],  # 可视几何完整镜像
    )


def _mirror_finger(finger: FingerCfg) -> FingerCfg:
    r"""镜像 finger mount 与完整串联链，保持名称和链顺序不变。"""

    return finger.replace(
        mount=mirror_pose_about_yz(finger.mount),  # palm->finger root frame
        joints=[_mirror_joint(joint) for joint in finger.joints],  # 每级局部变换都执行 $STS$
    )


def _mirror_palm(palm: PalmCfg) -> PalmCfg:
    r"""镜像 palm root、复合几何和惯量，不改变 primitive 尺寸。"""

    metadata = dict(palm.metadata)  # palm metadata 可能携带 hand builder 可读的 canonical mounts
    raw_mounts = metadata.get("finger_mounts")
    if isinstance(raw_mounts, dict):
        metadata["finger_mounts"] = {
            name: mirror_pose_about_yz(PoseCfg.from_value(pose))
            for name, pose in raw_mounts.items()
        }  # sidecar 中的 palm-level mount provenance 必须与物理 left hand 一致
    return palm.replace(
        origin=mirror_pose_about_yz(palm.origin),  # hand root->palm frame
        inertial=_mirror_inertial(palm.inertial),  # composite palm 的 COM 与交叉惯量同步镜像
        collisions=[_mirror_collision(element) for element in palm.collisions],  # 每个 box/mesh instance 分别镜像
        visuals=[_mirror_visual(element) for element in palm.visuals],  # visual 与 collision 保持一致
        metadata=metadata,  # handedness-sensitive provenance 与物理字段同步 lowering
    )


def handedness_contract(*, target: HandTarget) -> dict[str, object]:
    r"""生成可写入 sidecar 的严格 handedness contract 证书。"""

    return {
        "version": HANDEDNESS_CONTRACT_VERSION,
        "canonical_handedness": "right",
        "target_handedness": target,
        "reflection_plane": "palm_yz",
        "same_q": True,
        "physical_lowering_complete": True,
    }  # 字段直接描述当前稳定数学合同，不记录迁移历史


def validate_generated_handedness_contract(
    sidecar: dict[str, object],
    *,
    allow_legacy_left_handedness: bool = False,
) -> None:
    r"""拒绝缺少严格整手镜像证书的 generated left sidecar。

    该函数是 HandBank 与 independent post-mutate restore 的共享事实源。安全门只
    作用于顶层 ``handedness="left"``；generated right 是 canonical 真源，无需
    借新证书证明一次并未发生的反射。

    Args:
        sidecar: 已解析的 generated ``hand.yaml`` 顶层 mapping。
        allow_legacy_left_handedness: 历史审计用显式 override。

    Raises:
        ValueError: left sidecar 缺少、损坏或伪造严格合同，且未显式 override。
    """

    if str(sidecar.get("handedness", "")).lower() != "left":
        return  # canonical generated right 不受 legacy-left gate 影响
    if allow_legacy_left_handedness:
        return  # override 只放行当前调用，不给旧 sidecar 伪造新证书

    contract = sidecar.get("handedness_contract")
    expected = handedness_contract(target="left")  # 字段值与当前 exporter contract 共享同一构造函数
    if not isinstance(contract, dict) or any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "legacy generated left hand lacks a valid strict handedness_contract; "
            "regenerate it with the current asset pipeline or set "
            "allow_legacy_left_handedness=True only for historical audit"
        )


def lower_hand_to_handedness(hand: HandCfg, target: HandTarget) -> HandCfg:
    r"""把完整 ``HandCfg`` lowering 到目标物理 handedness。

    该映射是 involution：同侧输入返回深拷贝；异侧输入执行一次完整反射。
    因而 post-mutate 可以把 left 恢复到 canonical right 空间，使用统一公式完成
    几何派生，再反射回 left，而无需让每个 mutator 持有 handedness 分支。

    Args:
        hand (HandCfg): 已处于 ``hand.handedness`` 所声明物理空间的整手。
        target (HandTarget): 目标 ``"left"`` 或 ``"right"``。

    Returns:
        HandCfg: 目标 handedness 下的完整物理手。

    Raises:
        ValueError: 输入 handedness 未知或 target 非法时抛出。
    """

    if target not in {"left", "right"}:
        raise ValueError(f"unsupported handedness target: {target!r}")
    if hand.handedness not in {"left", "right"}:
        raise ValueError(f"strict handedness lowering requires known source handedness, got {hand.handedness!r}")

    if hand.handedness == target:
        lowered = hand.copy()  # 保持函数式边界，调用方不能借同侧 lowering 修改真源
    else:
        lowered = hand.replace(
            palm=_mirror_palm(hand.palm),  # palm 与全部 finger 在同一次空间反射中处理
            fingers=[_mirror_finger(finger) for finger in hand.fingers],
            handedness=target,
        )

    metadata = dict(lowered.metadata)  # handedness 证书与现有 builder/topology provenance 并存
    metadata["handedness_contract"] = handedness_contract(target=target)
    return lowered.replace(handedness=target, metadata=metadata)


__all__ = [
    "HANDEDNESS_CONTRACT_VERSION",
    "compose_poses",
    "handedness_contract",
    "lower_hand_to_handedness",
    "matrix_to_rpy",
    "mirror_inertia_tensor_about_yz",
    "mirror_pose_about_yz",
    "mirror_revolute_axis_about_yz",
    "rpy_rotation_matrix",
    "validate_generated_handedness_contract",
]
