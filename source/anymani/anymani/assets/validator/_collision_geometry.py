r"""post-mutate finger-finger SDF validator 的 collision geometry 抽取层。

本模块只做一件事：把 `HandCfg` 里 joint-centric 的 collision primitives
收敛成世界系下的 body records。它不判断 clearance 是否合格，也不决定
validator 的 pass / fail；这些留给 `_sdf_clearance.py` 和 `hand_rules.py`。

坐标与姿态约定
--------------

当前 AnyMani 的 URDF exporter 在 **finger mount 折叠进第一关节** 这一步，
已明确采用一个工程近似：

$$
{}^{palm}T_{j_0}
\approx
\operatorname{pose\_add}(T_{mount}, T_{j_0}),
$$

即 first joint 的 mount 与 joint origin 使用平移逐分量相加、RPY 逐分量相加。

但要特别注意：这个近似**只适用于 root mount folding 本身**。进入 URDF
关节树之后，后续 joint origin 与 collision origin 都会被仿真器按真实刚体变换
解释，而不是简单分量相加。

因此本模块采用一个混合策略：

1. 第一段 child-link pose：复用 exporter 的 `pose_add(mount, first_joint.origin)`；
2. 第二段及以后 joint 链：使用真实刚体复合；
3. 每个 child-link 下的 `collision.origin`：使用真实刚体复合。

否则像 LEAP thumb 这类带明显姿态的链，会被摆到错误 world pose，从而制造出
假的 finger-finger penetration。

验证域边界
----------

- 只抽取 finger child-link 的 collision geometry；
- 不抽取 palm collision；
- 不检查同一 finger 内不同 body 之间的互碰；
- home pose 下所有 joint angle 取 $q=0$，因此 joint origin 只作为固定链式位姿累积。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal

from ..asset_base import HandCfg
from ..asset_schema_core import (
    BoxGeometryCfg,
    CollisionGeometryCfg,
    CylinderGeometryCfg,
    EllipticCylinderGeometryCfg,
    MeshGeometryCfg,
    PoseCfg,
    SphereGeometryCfg,
    Vector3,
)


UnsupportedGeometryPolicy = Literal["fail", "warn_skip"]
"""unsupported collision geometry 的处理策略。"""


SUPPORTED_PRIMITIVE_KINDS = ("box", "cylinder", "elliptic_cylinder", "sphere")
"""v1 SDF backend 明确支持的 primitive 类型。"""


@dataclass(frozen=True)
class CollisionBodyRecord:
    r"""一块 finger collision primitive 的世界系记录。

    Attributes:
        finger_name: 逻辑 finger 名称，只用于跨 finger 分组与诊断。
        joint_name: 携带该 child-link collision 的 joint 名称。
        link_name: child link 名称。
        body_name: collision 元素名；缺省时使用稳定 fallback。
        body_path: 面向错误消息和 certificate 的稳定路径。
        geometry_kind: primitive 类型。
        geometry: 原始几何 cfg，保留解析参数。
        world_pose: collision frame 在 palm/world frame 下的姿态。
    """

    finger_name: str
    joint_name: str
    link_name: str
    body_name: str
    body_path: str
    geometry_kind: str
    geometry: BoxGeometryCfg | CylinderGeometryCfg | EllipticCylinderGeometryCfg | SphereGeometryCfg
    world_pose: PoseCfg


@dataclass(frozen=True)
class SkippedCollisionBody:
    r"""被 exploratory `warn_skip` 跳过的一块 collision geometry。"""

    finger_name: str
    joint_name: str
    link_name: str
    body_name: str
    body_path: str
    geometry_kind: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        r"""把 skip 记录转成 certificate 可序列化字典。"""

        return {
            "finger_name": self.finger_name,
            "joint_name": self.joint_name,
            "link_name": self.link_name,
            "body_name": self.body_name,
            "body_path": self.body_path,
            "geometry_kind": self.geometry_kind,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CollisionExtractionResult:
    r"""collision geometry 抽取结果。"""

    bodies_by_finger: dict[str, list[CollisionBodyRecord]]
    skipped_bodies: list[SkippedCollisionBody]

    @property
    def complete(self) -> bool:
        r"""没有跳过 body 时，抽取证书才是完整的。"""

        return not self.skipped_bodies


def extract_finger_collision_bodies(
    hand: HandCfg,
    *,
    unsupported_policy: UnsupportedGeometryPolicy = "fail",
) -> CollisionExtractionResult:
    r"""抽取 post-mutate home pose 下所有 finger collision primitive。

    Args:
        hand: 已经完成 post-mutate 的整手 schema。
        unsupported_policy: 遇到 mesh / 未知几何时是硬失败，还是记录 skip。

    Returns:
        CollisionExtractionResult: 按 finger 分组的 world-space body records。

    Raises:
        ValueError: 当 unsupported_policy="fail" 且遇到不支持几何时抛出。
    """

    if unsupported_policy not in {"fail", "warn_skip"}:
        raise ValueError(f"unsupported collision geometry policy: {unsupported_policy!r}")

    bodies_by_finger: dict[str, list[CollisionBodyRecord]] = {}
    skipped_bodies: list[SkippedCollisionBody] = []

    for finger in hand.fingers:
        bodies_by_finger.setdefault(finger.name, [])
        parent_link_pose = PoseCfg()  # 当前 joint.parent 这根 link 在 palm/world 下的位姿

        for joint_index, joint in enumerate(finger.joints):
            if joint_index == 0:
                # 第一段必须镜像 exporter 的 mount folding 近似。
                link_pose = _pose_add(finger.mount, joint.origin)
            else:
                # 进入关节树后，joint.origin 是相对 parent link frame 的刚体变换。
                link_pose = _compose_pose(parent_link_pose, joint.origin)

            for collision_index, collision in enumerate(joint.collisions):
                body_name = collision.name or f"{joint.child}_collision_{collision_index}"
                body_path = f"{finger.name}/{joint.name}/{joint.child}/{body_name}"
                geometry = collision.geometry
                if not isinstance(geometry, (BoxGeometryCfg, CylinderGeometryCfg, EllipticCylinderGeometryCfg, SphereGeometryCfg)):
                    skipped = _make_skipped_body(
                        finger_name=finger.name,
                        joint_name=joint.name,
                        link_name=str(joint.child),
                        body_name=body_name,
                        body_path=body_path,
                        geometry=geometry,
                    )
                    if unsupported_policy == "fail":
                        raise ValueError(
                            f"unsupported collision geometry for SDF clearance: {body_path} kind={skipped.geometry_kind!r}"
                        )
                    skipped_bodies.append(skipped)
                    continue

                # `collision.origin` 是相对当前 child-link frame 的局部位姿。
                # 这里若继续用分量相加，会忽略 link 姿态对局部偏移的旋转作用。
                world_pose = _compose_pose(link_pose, collision.origin)
                bodies_by_finger[finger.name].append(
                    CollisionBodyRecord(
                        finger_name=finger.name,
                        joint_name=joint.name,
                        link_name=str(joint.child),
                        body_name=body_name,
                        body_path=body_path,
                        geometry_kind=geometry.kind,
                        geometry=geometry,
                        world_pose=world_pose,
                    )
                )

            parent_link_pose = link_pose

    return CollisionExtractionResult(bodies_by_finger=bodies_by_finger, skipped_bodies=skipped_bodies)


def _make_skipped_body(
    *,
    finger_name: str,
    joint_name: str,
    link_name: str,
    body_name: str,
    body_path: str,
    geometry: Any,
) -> SkippedCollisionBody:
    r"""构造 unsupported geometry 的结构化 skip 记录。"""

    geometry_kind = getattr(geometry, "kind", type(geometry).__name__)
    if isinstance(geometry, MeshGeometryCfg):
        reason = "mesh geometry is not certified by sampled primitive SDF v1"
    else:
        reason = "geometry kind is not supported by sampled primitive SDF v1"
    return SkippedCollisionBody(
        finger_name=finger_name,
        joint_name=joint_name,
        link_name=link_name,
        body_name=body_name,
        body_path=body_path,
        geometry_kind=str(geometry_kind),
        reason=reason,
    )


def _pose_add(lhs: PoseCfg, rhs: PoseCfg) -> PoseCfg:
    r"""复用当前 URDF exporter 的 pose 逐分量叠加近似。"""

    return PoseCfg(
        pos=(
            lhs.pos[0] + rhs.pos[0],
            lhs.pos[1] + rhs.pos[1],
            lhs.pos[2] + rhs.pos[2],
        ),
        rpy=(
            lhs.rpy[0] + rhs.rpy[0],
            lhs.rpy[1] + rhs.rpy[1],
            lhs.rpy[2] + rhs.rpy[2],
        ),
    )


def _compose_pose(parent: PoseCfg, local: PoseCfg) -> PoseCfg:
    r"""按真实刚体复合计算 $T_{world,child}=T_{world,parent}T_{parent,child}$。

    数学上：
    $$
    \mathbf{p}_{wc} = \mathbf{p}_{wp} + R_{wp}\mathbf{p}_{pc},
    \qquad
    R_{wc} = R_{wp}R_{pc}.
    $$

    # NOTE:
    这不是为了替代 root mount folding 的历史近似；它只用于进入 joint tree
    之后的真实 kinematic / collision 组合。
    """

    parent_rotation = rpy_rotation_matrix(parent.rpy)
    local_rotation = rpy_rotation_matrix(local.rpy)
    local_pos_in_world = apply_rotation(parent_rotation, local.pos)
    world_rotation = _matrix_multiply(parent_rotation, local_rotation)
    return PoseCfg(
        pos=(
            parent.pos[0] + local_pos_in_world[0],
            parent.pos[1] + local_pos_in_world[1],
            parent.pos[2] + local_pos_in_world[2],
        ),
        rpy=_matrix_to_rpy(world_rotation),
    )


def rpy_rotation_matrix(rpy: Vector3) -> tuple[Vector3, Vector3, Vector3]:
    r"""构造 URDF 风格固定轴 RPY 旋转矩阵 $R_z R_y R_x$。"""

    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def apply_rotation(matrix: tuple[Vector3, Vector3, Vector3], point: Vector3) -> Vector3:
    r"""计算 $R\mathbf{x}$。"""

    return (
        matrix[0][0] * point[0] + matrix[0][1] * point[1] + matrix[0][2] * point[2],
        matrix[1][0] * point[0] + matrix[1][1] * point[1] + matrix[1][2] * point[2],
        matrix[2][0] * point[0] + matrix[2][1] * point[1] + matrix[2][2] * point[2],
    )


def apply_inverse_pose(pose: PoseCfg, point_world: Vector3) -> Vector3:
    r"""把 world 点变换到 body local frame。

    对刚体变换 $x_w=R x_l+t$，逆变换为：
    $$
    x_l=R^\top(x_w-t).
    $$
    """

    dx = point_world[0] - pose.pos[0]
    dy = point_world[1] - pose.pos[1]
    dz = point_world[2] - pose.pos[2]
    rotation = rpy_rotation_matrix(pose.rpy)
    return (
        rotation[0][0] * dx + rotation[1][0] * dy + rotation[2][0] * dz,
        rotation[0][1] * dx + rotation[1][1] * dy + rotation[2][1] * dz,
        rotation[0][2] * dx + rotation[1][2] * dy + rotation[2][2] * dz,
    )


def apply_pose(pose: PoseCfg, point_local: Vector3) -> Vector3:
    r"""把 body local 点变换到 world frame。"""

    rotated = apply_rotation(rpy_rotation_matrix(pose.rpy), point_local)
    return (
        pose.pos[0] + rotated[0],
        pose.pos[1] + rotated[1],
        pose.pos[2] + rotated[2],
    )


def _matrix_multiply(
    lhs: tuple[Vector3, Vector3, Vector3],
    rhs: tuple[Vector3, Vector3, Vector3],
) -> tuple[Vector3, Vector3, Vector3]:
    r"""计算旋转矩阵乘积 $R=R_1R_2$。"""

    rhs_cols = (
        (rhs[0][0], rhs[1][0], rhs[2][0]),
        (rhs[0][1], rhs[1][1], rhs[2][1]),
        (rhs[0][2], rhs[1][2], rhs[2][2]),
    )
    rows: list[Vector3] = []
    for row in lhs:
        rows.append(
            (
                row[0] * rhs_cols[0][0] + row[1] * rhs_cols[0][1] + row[2] * rhs_cols[0][2],
                row[0] * rhs_cols[1][0] + row[1] * rhs_cols[1][1] + row[2] * rhs_cols[1][2],
                row[0] * rhs_cols[2][0] + row[1] * rhs_cols[2][1] + row[2] * rhs_cols[2][2],
            )
        )
    return (rows[0], rows[1], rows[2])


def _matrix_to_rpy(matrix: tuple[Vector3, Vector3, Vector3]) -> Vector3:
    r"""把旋转矩阵反解回 URDF 固定轴 RPY。"""

    pitch = math.asin(-max(-1.0, min(1.0, matrix[2][0])))
    cp = math.cos(pitch)

    if abs(cp) > 1e-12:
        roll = math.atan2(matrix[2][1], matrix[2][2])
        yaw = math.atan2(matrix[1][0], matrix[0][0])
    else:
        roll = math.atan2(-matrix[0][1], matrix[1][1])
        yaw = 0.0

    return (roll, pitch, yaw)


__all__ = [
    "CollisionBodyRecord",
    "CollisionExtractionResult",
    "SkippedCollisionBody",
    "UnsupportedGeometryPolicy",
    "extract_finger_collision_bodies",
    "apply_inverse_pose",
    "apply_pose",
]
