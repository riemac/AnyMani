r"""post-mutate finger axial length validator：沿 nominal distal axis 的真实几何长度。

本模块服务一个比 finger-finger clearance 更简单、但同样必须诚实建模的后验规则：

1. `link_scale` 会改变 link 的实际几何长度；
2. `tip_replace` 可能把 primitive tip 换成 custom mesh tip；
3. 我们真正想限制的是 home pose 下“从指根到指尖顶部”的**轴向真实长度**，
   而不是 joint-origin 近似链长，也不是三维最大欧氏直径。

因此这里采用用户已经确认的长度语义：

$$
L(F;\,\mathbf{a})
=
\max_{\mathbf{p}\in\mathcal{S}_F}\mathbf{a}^{\top}\mathbf{p}
\;-\;
\min_{\mathbf{p}\in\mathcal{S}_F}\mathbf{a}^{\top}\mathbf{p},
$$

其中：

- $F$：一根 finger 的 collision geometry union；
- $\mathcal{S}_F$：该 finger 在 home pose 下所有 collision body 的表面 / 顶点极值集合；
- $\mathbf{a}$：该 finger 的 nominal distal axis，单位向量。

与 SDF clearance 的关系
----------------------

SDF clearance 关心的是 finger-finger 的最小 signed distance，因此需要 surface sampling
和 union SDF 查询；本模块只关心**单个 finger 沿某一轴的一维投影宽度**，因此不需要
GPU、SDF 或两两点对搜索：

- primitive：直接解析 support / corner；
- mesh：直接对 `trimesh` 顶点做投影；
- union：对所有 body 的投影区间取全局 `max-min`。

这不是“偷懒近似”，而是线性泛函在多面体 / 三角网格上的极值本就出现在顶点或解析
support 点上，因此对当前问题它反而比 sampled surface 更直接、更高效。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..asset_base import HandCfg
from ..asset_schema_core import (
    BoxGeometryCfg,
    CylinderGeometryCfg,
    EllipticCylinderGeometryCfg,
    MeshGeometryCfg,
    PoseCfg,
    SphereGeometryCfg,
    Vector3,
)
from ._collision_geometry import (
    CollisionBodyRecord,
    UnsupportedGeometryPolicy,
    apply_rotation,
    extract_finger_collision_bodies,
    extract_finger_link_poses,
    rpy_rotation_matrix,
)


FingerRole = Literal["thumb", "non_thumb"]
"""手指长度规则当前只区分 thumb / non-thumb 两类角色。"""


NOT_CERTIFIED = (
    "all_pose_length",
    "bent_chain_geodesic_length",
    "physics_runtime_safety",
)
"""长度证书明确拒绝冒领的强 claim。"""


@dataclass(frozen=True)
class FingerLengthConfig:
    r"""finger axial length 评估配置。

    Attributes:
        max_thumb_length: thumb 允许的最大轴向真实长度，单位 meter；`None` 表示
            只测量、不做 hard gate。
        max_non_thumb_length: non-thumb 允许的最大轴向真实长度，单位 meter；
            `None` 表示只测量、不做 hard gate。
        tolerance: 阈值比较容差，只用于抵消浮点误差。
        unsupported_policy: 若遇到当前不支持的 collision geometry，是否 fail-hard
            还是记录成 incomplete certificate。
    """

    max_thumb_length: float | None = None
    max_non_thumb_length: float | None = None
    tolerance: float = 1e-9
    unsupported_policy: UnsupportedGeometryPolicy = "fail"


@dataclass(frozen=True)
class FingerLengthMeasurement:
    r"""单根 finger 的轴向长度测量结果。"""

    finger_name: str
    role: FingerRole
    axis: Vector3
    axis_source: str
    min_projection: float
    max_projection: float
    axial_length: float
    threshold: float | None

    def to_dict(self) -> dict[str, Any]:
        r"""转成 certificate / sidecar 可直接序列化的字典。"""

        return {
            "finger_name": self.finger_name,
            "role": self.role,
            "axis": tuple(float(component) for component in self.axis),
            "axis_source": self.axis_source,
            "min_projection": float(self.min_projection),
            "max_projection": float(self.max_projection),
            "axial_length": float(self.axial_length),
            "threshold": None if self.threshold is None else float(self.threshold),
        }


@dataclass
class FingerLengthCertificate:
    r"""finger axial length 的结构化证书。"""

    pose_scope: str = "post_mutate_home_pose"
    geometry_scope: str = "collision_geometry_only"
    length_kind: str = "axial_projection_extent"
    complete: bool = True
    skipped_bodies: list[dict[str, str]] = field(default_factory=list)
    not_certified: list[str] = field(default_factory=lambda: list(NOT_CERTIFIED))
    thresholds: dict[str, float | None] = field(default_factory=dict)
    measurements: list[dict[str, Any]] = field(default_factory=list)
    violations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        r"""转成 `ValidationResult.metadata` / sidecar 可直接保存的 dict。"""

        return {
            "pose_scope": self.pose_scope,
            "geometry_scope": self.geometry_scope,
            "length_kind": self.length_kind,
            "complete": self.complete,
            "skipped_bodies": list(self.skipped_bodies),
            "not_certified": list(self.not_certified),
            "thresholds": dict(self.thresholds),
            "measurements": list(self.measurements),
            "violations": list(self.violations),
        }


@dataclass(frozen=True)
class FingerLengthResult:
    r"""finger axial length 评估结果。"""

    passed: bool
    certificate: FingerLengthCertificate
    measurements: list[FingerLengthMeasurement]
    violations: list[FingerLengthMeasurement]


def evaluate_finger_axial_length(hand: HandCfg, cfg: FingerLengthConfig) -> FingerLengthResult:
    r"""评估一只 hand 在 home pose 下每根 finger 的轴向真实长度。

    Args:
        hand (HandCfg): 已完成 pre-made 或 post-mutate 的整手 schema。
        cfg (FingerLengthConfig): 轴向长度阈值与 unsupported geometry 策略。

    Returns:
        FingerLengthResult: 每根 finger 的长度测量、证书与违规列表。
    """

    extraction = extract_finger_collision_bodies(hand, unsupported_policy=cfg.unsupported_policy)  # 统一使用 collision 几何的 home-pose 事实
    link_poses_by_finger = extract_finger_link_poses(hand)  # nominal distal axis 的定义依赖 link frame 轨迹，而不是 collision pose
    measurements: list[FingerLengthMeasurement] = []  # 所有 finger 的长度测量结果，保持输入 hand 的 finger 顺序
    violations: list[FingerLengthMeasurement] = []  # 超过阈值的 finger 子集

    for finger in hand.fingers:
        finger_bodies = extraction.bodies_by_finger.get(finger.name, [])  # 当前 finger 下所有参与长度包络的 collision body
        if not finger_bodies:
            continue  # 没有任何 collision body 时无法定义真实几何长度；这里选择跳过而不是伪装成 0

        axis, axis_source = _nominal_distal_axis(
            finger_name=finger.name,
            joints=finger.joints,
            link_poses=link_poses_by_finger.get(finger.name, []),
        )  # 当前 finger 的 nominal distal axis 由建模约定决定
        min_projection, max_projection = _projection_interval_for_finger(finger_bodies, axis=axis)  # union 在轴向上的全局投影区间
        axial_length = max_projection - min_projection  # 从最靠近掌的一端到最远端的真实轴向长度
        role: FingerRole = "thumb" if finger.name == "thumb" else "non_thumb"
        threshold = cfg.max_thumb_length if role == "thumb" else cfg.max_non_thumb_length

        measurement = FingerLengthMeasurement(
            finger_name=finger.name,
            role=role,
            axis=axis,
            axis_source=axis_source,
            min_projection=min_projection,
            max_projection=max_projection,
            axial_length=axial_length,
            threshold=threshold,
        )
        measurements.append(measurement)

        if threshold is not None and axial_length > threshold + cfg.tolerance:
            violations.append(measurement)  # 只有显式声明阈值时才升级成 hard gate 违规

    certificate = FingerLengthCertificate(
        complete=extraction.complete,
        skipped_bodies=[body.to_dict() for body in extraction.skipped_bodies],
        thresholds={
            "thumb": None if cfg.max_thumb_length is None else float(cfg.max_thumb_length),
            "non_thumb": None if cfg.max_non_thumb_length is None else float(cfg.max_non_thumb_length),
        },
        measurements=[measurement.to_dict() for measurement in measurements],
        violations=[measurement.to_dict() for measurement in violations],
    )
    return FingerLengthResult(
        passed=certificate.complete and not violations,
        certificate=certificate,
        measurements=measurements,
        violations=violations,
    )


def measure_finger_axial_lengths(hand: HandCfg) -> list[FingerLengthMeasurement]:
    r"""只测量、不做阈值判定地返回每根 finger 的轴向真实长度。

    该 helper 主要服务 sidecar 摘要。它和 validator 共用同一套几何定义，
    避免再次出现“summary 一个算法、validator 另一个算法”的双语义漂移。
    """

    return evaluate_finger_axial_length(hand, FingerLengthConfig()).measurements


def _nominal_distal_axis(
    *,
    finger_name: str,
    joints,
    link_poses: list[PoseCfg],
) -> tuple[Vector3, str]:
    r"""根据当前资产建模约定恢复一根 finger 的 nominal distal axis。

    规则已经在讨论阶段与用户对齐：

    - non-thumb：直接采用 root-most link 的 local $+y$ 方向；
    - thumb：axis 的定义忽略 CMC1 对远端功能指段的污染，优先使用
      `CMC2 -> 最远 non-tip 指段` 的连线；但 CMC1 geometry 本身仍会参与
      投影长度的 min/max 点集。
    """

    if not joints or not link_poses:
        raise ValueError(f"finger '{finger_name}' has no joint/link pose to define nominal distal axis")

    if finger_name != "thumb":
        root_rotation = rpy_rotation_matrix(link_poses[0].rpy)  # non-thumb 的规范 distal 方向由 root link 的 local +y 给定
        axis = _normalize(apply_rotation(root_rotation, (0.0, 1.0, 0.0)))
        return axis, "root_link_local_+y"

    # thumb 的 axis 先跳过 CMC1：我们只借用 CMC2 以后那条功能性远端指段来定方向。
    start_index = next(
        (index for index, joint in enumerate(joints) if "cmc1" not in str(joint.child).lower()),
        0,
    )  # 若当前 thumb 拓扑恰好没有显式 CMC1，退回首段
    distal_indices = [index for index, joint in enumerate(joints) if not bool(joint.is_tip)]  # tip fixed joint 不参与 axis 的末端锚点定义
    end_index = distal_indices[-1] if distal_indices else len(joints) - 1

    if end_index > start_index:
        start_pose = link_poses[start_index]  # 通常对应 thumb_cmc2
        end_pose = link_poses[end_index]  # 通常对应 thumb_dip 或当前最远非 tip 指段
        axis_candidate = (
            end_pose.pos[0] - start_pose.pos[0],
            end_pose.pos[1] - start_pose.pos[1],
            end_pose.pos[2] - start_pose.pos[2],
        )
        if _norm(axis_candidate) > 1e-12:
            return _normalize(axis_candidate), f"{joints[start_index].child}_to_{joints[end_index].child}"

    fallback_rotation = rpy_rotation_matrix(link_poses[start_index].rpy)  # 极端退化时退回 thumb 功能远端段的 local +y
    fallback_axis = _normalize(apply_rotation(fallback_rotation, (0.0, 1.0, 0.0)))
    return fallback_axis, f"{joints[start_index].child}_local_+y_fallback"


def _projection_interval_for_finger(
    bodies: list[CollisionBodyRecord],
    *,
    axis: Vector3,
) -> tuple[float, float]:
    r"""汇总一根 finger 的 collision union 在某条轴上的投影区间。"""

    min_projection = math.inf  # 当前 finger 在轴向上最靠近掌的一端
    max_projection = -math.inf  # 当前 finger 在轴向上最远离掌的一端

    for body in bodies:
        body_min, body_max = _projection_interval_for_body(body, axis=axis)  # 单个 collision body 在轴向上的投影区间
        min_projection = min(min_projection, body_min)
        max_projection = max(max_projection, body_max)

    if not math.isfinite(min_projection) or not math.isfinite(max_projection):
        raise ValueError("finger axial length requires at least one valid collision body projection interval")
    return min_projection, max_projection


def _projection_interval_for_body(body: CollisionBodyRecord, *, axis: Vector3) -> tuple[float, float]:
    r"""计算单个 collision body 在给定 world axis 上的投影区间。"""

    geometry = body.geometry
    center_projection = _dot(body.world_pose.pos, axis)  # collision local 原点在目标轴上的投影中心

    if isinstance(geometry, BoxGeometryCfg):
        return _box_projection_interval(center_projection, geometry.size, body.world_pose, axis=axis)
    if isinstance(geometry, SphereGeometryCfg):
        return center_projection - float(geometry.radius), center_projection + float(geometry.radius)
    if isinstance(geometry, CylinderGeometryCfg):
        return _cylinder_projection_interval(
            center_projection,
            radius=float(geometry.radius),
            length=float(geometry.length),
            pose=body.world_pose,
            axis=axis,
        )
    if isinstance(geometry, EllipticCylinderGeometryCfg):
        return _elliptic_cylinder_projection_interval(
            center_projection,
            radius_x=float(geometry.radius_x),
            radius_z=float(geometry.radius_z),
            length=float(geometry.length),
            pose=body.world_pose,
            axis=axis,
        )
    if isinstance(geometry, MeshGeometryCfg):
        return _mesh_projection_interval(center_projection, geometry, body.world_pose, axis=axis)
    raise ValueError(f"unsupported geometry for finger axial length: {type(geometry).__name__}")


def _box_projection_interval(
    center_projection: float,
    size: Vector3,
    pose: PoseCfg,
    *,
    axis: Vector3,
) -> tuple[float, float]:
    r"""box 在某条轴上的投影区间。

    对中心在原点、半尺寸为 $(h_x,h_y,h_z)$ 的盒子，沿方向 $\mathbf{a}$ 的 support 半径是：
    $$
    \rho = |a_x|h_x + |a_y|h_y + |a_z|h_z.
    $$
    """

    local_axis = _world_axis_to_local_axis(axis, pose=pose)  # 把 world axis 拉回 body local frame，便于直接使用解析 support 公式
    half_x = float(size[0]) * 0.5
    half_y = float(size[1]) * 0.5
    half_z = float(size[2]) * 0.5
    half_extent = (
        abs(local_axis[0]) * half_x
        + abs(local_axis[1]) * half_y
        + abs(local_axis[2]) * half_z
    )
    return center_projection - half_extent, center_projection + half_extent


def _cylinder_projection_interval(
    center_projection: float,
    *,
    radius: float,
    length: float,
    pose: PoseCfg,
    axis: Vector3,
) -> tuple[float, float]:
    r"""y-axis cylinder 在某条轴上的投影区间。"""

    local_axis = _world_axis_to_local_axis(axis, pose=pose)  # cylinder 的解析 support 在 local y-axis 表达下最简单
    axial_half = abs(local_axis[1]) * (length * 0.5)  # 沿 cylinder 主轴分量贡献的端面长度
    radial_half = radius * math.sqrt(max(0.0, local_axis[0] ** 2 + local_axis[2] ** 2))  # 正交于主轴的圆盘半径贡献
    half_extent = axial_half + radial_half
    return center_projection - half_extent, center_projection + half_extent


def _elliptic_cylinder_projection_interval(
    center_projection: float,
    *,
    radius_x: float,
    radius_z: float,
    length: float,
    pose: PoseCfg,
    axis: Vector3,
) -> tuple[float, float]:
    r"""y-axis elliptic cylinder 在某条轴上的投影区间。"""

    local_axis = _world_axis_to_local_axis(axis, pose=pose)  # local $(x,z)$ 平面上的椭圆 support 由解析公式给出
    axial_half = abs(local_axis[1]) * (length * 0.5)
    radial_half = math.sqrt((radius_x * local_axis[0]) ** 2 + (radius_z * local_axis[2]) ** 2)
    half_extent = axial_half + radial_half
    return center_projection - half_extent, center_projection + half_extent


def _mesh_projection_interval(
    center_projection: float,
    geometry: MeshGeometryCfg,
    pose: PoseCfg,
    *,
    axis: Vector3,
) -> tuple[float, float]:
    r"""triangle mesh 在某条轴上的投影区间。

    由于 $\mathbf{a}^{\top}(R\mathbf{v}+\mathbf{t}) = \mathbf{a}^{\top}\mathbf{t} + (R^\top\mathbf{a})^{\top}\mathbf{v}$，
    因此只要把 world axis 拉回 mesh local frame，再对局部顶点做一维投影即可。
    这一步不需要 SDF，也不需要 surface sampling；对线性投影极值来说顶点就是精确支撑集。
    """

    local_vertices = _load_mesh_vertices_local(geometry.file_path, _scale_tuple(geometry.scale))  # 已经烘焙 scale 的 mesh local 顶点集
    if local_vertices.size == 0:
        raise ValueError(f"finger axial length got empty mesh vertices: {geometry.file_path}")
    local_axis = np.asarray(_world_axis_to_local_axis(axis, pose=pose), dtype=np.float64)  # world axis 拉回 local 后即可直接做 `v @ a`
    projections = local_vertices @ local_axis  # 所有局部顶点在 local_axis 上的一维投影
    return center_projection + float(np.min(projections)), center_projection + float(np.max(projections))


def _world_axis_to_local_axis(axis_world: Vector3, *, pose: PoseCfg) -> Vector3:
    r"""把一条 world-space 单位轴拉回 body local frame。"""

    rotation = rpy_rotation_matrix(pose.rpy)  # `rotation` 的行向量表达 world <- local
    return (
        rotation[0][0] * axis_world[0] + rotation[1][0] * axis_world[1] + rotation[2][0] * axis_world[2],
        rotation[0][1] * axis_world[0] + rotation[1][1] * axis_world[1] + rotation[2][1] * axis_world[2],
        rotation[0][2] * axis_world[0] + rotation[1][2] * axis_world[1] + rotation[2][2] * axis_world[2],
    )


def _scale_tuple(scale: Vector3) -> tuple[float, float, float]:
    r"""把 mesh scale 规约成稳定三元组，便于缓存键统一。"""

    return float(scale[0]), float(scale[1]), float(scale[2])


def _resolve_mesh_path(file_path: str | Path) -> Path:
    r"""把 mesh 路径规范化成绝对路径。"""

    path = Path(file_path).expanduser()
    return path if path.is_absolute() else path.resolve()


def _load_mesh_vertices_local(file_path: str, scale: tuple[float, float, float]) -> np.ndarray:
    r"""加载、缩放并缓存 mesh 顶点。

    对 finger axial length 而言，我们只需要顶点集合的投影极值，不要求 mesh watertight
    或 winding consistent。因此这层 loader 比 SDF 的 `_load_checked_trimesh(...)` 更轻。
    """

    return _load_mesh_vertices_local_cached(str(_resolve_mesh_path(file_path)), scale)


@lru_cache(maxsize=128)
def _load_mesh_vertices_local_cached(file_path: str, scale: tuple[float, float, float]) -> np.ndarray:
    r"""缓存已经完成 `process=True` 与 scale 烘焙的 mesh 顶点数组。"""

    import trimesh

    mesh = trimesh.load(file_path, force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"finger axial length expects a triangle mesh, got {type(mesh).__name__}: {file_path}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"finger axial length got empty mesh: {file_path}")

    mesh = mesh.copy()
    mesh.apply_scale(scale)  # 用户 `scale` 与 `unit_scale` 已经被 builder/exporter 路线 lower 到这里
    return np.asarray(mesh.vertices, dtype=np.float64)


def _dot(lhs: Vector3, rhs: Vector3) -> float:
    r"""计算三维点积 $\mathbf{x}^{\top}\mathbf{y}$。"""

    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]


def _norm(vector: Vector3) -> float:
    r"""计算三维向量的欧氏范数 $\|\mathbf{x}\|_2$。"""

    return math.sqrt(_dot(vector, vector))


def _normalize(vector: Vector3) -> Vector3:
    r"""把三维向量归一化成单位向量。"""

    length = _norm(vector)
    if length <= 1e-12:
        raise ValueError(f"cannot normalize near-zero axis vector: {vector!r}")
    return (vector[0] / length, vector[1] / length, vector[2] / length)


__all__ = [
    "FingerLengthCertificate",
    "FingerLengthConfig",
    "FingerLengthMeasurement",
    "FingerLengthResult",
    "evaluate_finger_axial_length",
    "measure_finger_axial_lengths",
]
