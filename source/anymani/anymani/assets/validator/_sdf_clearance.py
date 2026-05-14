r"""finger-finger sampled-surface SDF clearance 后端。

本模块实现用户最终确认的 validator 路线：

1. 每根 finger 的所有 collision primitives 组成一个 union SDF；
2. 在 post-mutate home pose 下，只检查不同 fingers 之间；
3. 用 sampled surface points 近似真实 mesh surface；
4. 以 symmetric clearance 作为 `min_finger_spacing` 的真实几何语义。

核心公式
--------

对 finger $F_i$ 的 union SDF：

$$
\operatorname{SDF}_{F_i}(\mathbf{x})
=
\min_{b\in F_i}\operatorname{SDF}_b(\mathbf{x}).
$$

两根 finger 的对称 clearance 定义为：

$$
c(F_i,F_j)
=
\min\left(
\min_{\mathbf{x}\in S_{F_i}}\operatorname{SDF}_{F_j}(\mathbf{x}),
\min_{\mathbf{y}\in S_{F_j}}\operatorname{SDF}_{F_i}(\mathbf{y})
\right).
$$

reject 条件：

$$
c(F_i,F_j) < m,
\qquad m=\texttt{min\_finger\_spacing}.
$$

近似边界
--------

这是 sampled-surface SDF approximation，不是 mesh-exact clearance，也不是
all-pose collision-free 证明。certificate 必须显式保留这些 non-goals。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Literal

from ..asset_base import HandCfg
from ..asset_schema_core import BoxGeometryCfg, CylinderGeometryCfg, EllipticCylinderGeometryCfg, SphereGeometryCfg, Vector3
from ._collision_geometry import (
    CollisionBodyRecord,
    SkippedCollisionBody,
    UnsupportedGeometryPolicy,
    apply_inverse_pose,
    apply_pose,
    extract_finger_collision_bodies,
)


NOT_CERTIFIED = (
    "all_pose_collision_free",
    "mesh_exact_clearance",
    "trajectory_safety",
    "physics_runtime_safety",
)
"""SDF clearance certificate 必须拒绝冒领的强 claim。"""


@dataclass(frozen=True)
class SdfClearanceConfig:
    r"""sampled SDF clearance 的数值配置。

    Attributes:
        min_clearance: finger-finger surface clearance margin，单位 meter。
        surface_samples_per_axis: box / cylinder / sphere 每个方向的采样密度。
        unsupported_policy: unsupported geometry 是 fail-hard 还是 warn_skip。
        tolerance: 阈值比较容差；默认只抵消浮点误差，不改变规则语义。
    """

    min_clearance: float
    surface_samples_per_axis: int = 5
    unsupported_policy: UnsupportedGeometryPolicy = "fail"
    tolerance: float = 1e-9
    device: Literal["auto", "cuda", "cpu"] = "auto"
    """SDF 计算设备策略。

    v1 的几何抽取仍是 Python object 层；真正适合 GPU 的部分是 surface samples
    对 target body 的批量 SDF 查询。默认 ``"auto"`` 会优先尝试 CUDA，若当前
    环境无 PyTorch/CUDA 或遇到不支持路径，则自动回退 CPU，并把实际设备写入证书。
    """


@dataclass(frozen=True)
class FingerPairClearance:
    r"""一对 finger 的对称 clearance 诊断。"""

    finger_i: str
    finger_j: str
    clearance: float
    direction_i_to_j: float
    direction_j_to_i: float

    def to_dict(self) -> dict[str, float | str]:
        r"""转成 certificate 可序列化字段。"""

        return {
            "finger_i": self.finger_i,
            "finger_j": self.finger_j,
            "clearance": self.clearance,
            "direction_i_to_j": self.direction_i_to_j,
            "direction_j_to_i": self.direction_j_to_i,
        }


@dataclass
class SdfClearanceCertificate:
    r"""post-mutate SDF clearance 的结构化证书。"""

    pose_scope: str = "post_mutate_home_pose"
    geometry_scope: str = "collision_geometry_only"
    sdf_kind: str = "sampled_surface_sdf_approx"
    complete: bool = True
    skipped_bodies: list[dict[str, str]] = field(default_factory=list)
    not_certified: list[str] = field(default_factory=lambda: list(NOT_CERTIFIED))
    min_clearance: float = 0.0
    device: str = "cpu"
    pair_clearances: list[dict[str, float | str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        r"""转成 `ValidationResult.metadata` / sidecar 可直接保存的 dict。"""

        return {
            "pose_scope": self.pose_scope,
            "geometry_scope": self.geometry_scope,
            "sdf_kind": self.sdf_kind,
            "complete": self.complete,
            "skipped_bodies": list(self.skipped_bodies),
            "not_certified": list(self.not_certified),
            "min_clearance": self.min_clearance,
            "device": self.device,
            "pair_clearances": list(self.pair_clearances),
        }


@dataclass(frozen=True)
class SdfClearanceResult:
    r"""SDF clearance 检测结果。"""

    passed: bool
    certificate: SdfClearanceCertificate
    violations: list[FingerPairClearance]


def evaluate_finger_sdf_clearance(hand: HandCfg, cfg: SdfClearanceConfig) -> SdfClearanceResult:
    r"""评估一只 hand 的 post-mutate home-pose finger clearance。

    Args:
        hand: 待检测的 hand schema。
        cfg: SDF 数值与 unsupported geometry 策略。

    Returns:
        SdfClearanceResult: pass/fail、certificate 与违规 pair。
    """

    extraction = extract_finger_collision_bodies(hand, unsupported_policy=cfg.unsupported_policy)
    device = _resolve_sdf_device(cfg.device)
    pair_clearances: list[FingerPairClearance] = []
    violations: list[FingerPairClearance] = []
    finger_names = [finger.name for finger in hand.fingers]

    for left_index, finger_i in enumerate(finger_names):
        for finger_j in finger_names[left_index + 1 :]:
            bodies_i = extraction.bodies_by_finger.get(finger_i, [])
            bodies_j = extraction.bodies_by_finger.get(finger_j, [])
            if not bodies_i or not bodies_j:
                continue

            try:
                clearance_i_to_j = _surface_to_union_sdf_min(
                    source_bodies=bodies_i,
                    target_bodies=bodies_j,
                    samples_per_axis=cfg.surface_samples_per_axis,
                    device=device,
                )
                clearance_j_to_i = _surface_to_union_sdf_min(
                    source_bodies=bodies_j,
                    target_bodies=bodies_i,
                    samples_per_axis=cfg.surface_samples_per_axis,
                    device=device,
                )
            except RuntimeError:
                if cfg.device == "cuda":
                    raise
                device = "cpu"
                clearance_i_to_j = _surface_to_union_sdf_min(
                    source_bodies=bodies_i,
                    target_bodies=bodies_j,
                    samples_per_axis=cfg.surface_samples_per_axis,
                    device=device,
                )
                clearance_j_to_i = _surface_to_union_sdf_min(
                    source_bodies=bodies_j,
                    target_bodies=bodies_i,
                    samples_per_axis=cfg.surface_samples_per_axis,
                    device=device,
                )
            clearance = min(clearance_i_to_j, clearance_j_to_i)
            pair = FingerPairClearance(
                finger_i=finger_i,
                finger_j=finger_j,
                clearance=clearance,
                direction_i_to_j=clearance_i_to_j,
                direction_j_to_i=clearance_j_to_i,
            )
            pair_clearances.append(pair)
            if clearance < cfg.min_clearance - cfg.tolerance:
                violations.append(pair)

    skipped = [body.to_dict() for body in extraction.skipped_bodies]
    certificate = SdfClearanceCertificate(
        complete=extraction.complete,
        skipped_bodies=skipped,
        min_clearance=cfg.min_clearance,
        device=device,
        pair_clearances=[pair.to_dict() for pair in pair_clearances],
    )
    return SdfClearanceResult(
        passed=not violations and certificate.complete,
        certificate=certificate,
        violations=violations,
    )


def signed_distance_to_body(point_world: Vector3, body: CollisionBodyRecord) -> float:
    r"""计算 world 点到单个 primitive body 的 signed distance。

    sign convention:

    - outside: positive；
    - surface: zero；
    - inside / penetration: negative。
    """

    point = apply_inverse_pose(body.world_pose, point_world)
    geometry = body.geometry
    if isinstance(geometry, BoxGeometryCfg):
        half_size = (geometry.size[0] / 2.0, geometry.size[1] / 2.0, geometry.size[2] / 2.0)
        return _sdf_box(point, half_size)
    if isinstance(geometry, CylinderGeometryCfg):
        return _sdf_cylinder_z(point, radius=geometry.radius, half_length=geometry.length / 2.0)
    if isinstance(geometry, EllipticCylinderGeometryCfg):
        return _sdf_elliptic_cylinder_y(
            point,
            radius_x=geometry.radius_x,
            radius_z=geometry.radius_z,
            half_length=geometry.length / 2.0,
        )
    if isinstance(geometry, SphereGeometryCfg):
        return _norm(point) - geometry.radius
    raise TypeError(f"unsupported SDF body geometry: {type(geometry).__name__}")


def union_signed_distance(point_world: Vector3, bodies: list[CollisionBodyRecord]) -> float:
    r"""计算一个 finger union body 的 signed distance。"""

    if not bodies:
        return math.inf
    return min(signed_distance_to_body(point_world, body) for body in bodies)


def sample_body_surface(body: CollisionBodyRecord, *, samples_per_axis: int) -> list[Vector3]:
    r"""为一个 primitive body 生成 world-space surface samples。"""

    geometry = body.geometry
    density = max(int(samples_per_axis), 2)
    if isinstance(geometry, BoxGeometryCfg):
        local_points = _sample_box_surface(geometry.size, density=density)
    elif isinstance(geometry, CylinderGeometryCfg):
        local_points = _sample_cylinder_z_surface(geometry.radius, geometry.length, density=density)
    elif isinstance(geometry, EllipticCylinderGeometryCfg):
        local_points = _sample_elliptic_cylinder_y_surface(
            geometry.radius_x,
            geometry.radius_z,
            geometry.length,
            density=density,
        )
    elif isinstance(geometry, SphereGeometryCfg):
        local_points = _sample_sphere_surface(geometry.radius, density=density)
    else:
        raise TypeError(f"unsupported surface sampling geometry: {type(geometry).__name__}")
    return [apply_pose(body.world_pose, point) for point in local_points]


def _surface_to_union_sdf_min(
    *,
    source_bodies: list[CollisionBodyRecord],
    target_bodies: list[CollisionBodyRecord],
    samples_per_axis: int,
    device: str,
) -> float:
    r"""计算 $\min_{x\in S_{source}}\operatorname{SDF}_{target}(x)$ 的采样近似。

    # NOTE:
    `source_bodies` 是一根 finger 的 union body。若 finger 内有复合 tip
    （例如 cylinder + sphere cap），单个 primitive 的部分表面可能埋在同一
    finger 的另一个 primitive 内部。那些点不是 union surface $S_F$，不能
    参与 inter-finger clearance，否则会把“同一 finger 内部复合重叠”误投射成
    finger-finger 约束。这里用 source union SDF 做一次轻量过滤：

    $$
    \operatorname{SDF}_{source}(\mathbf{x}) \ge -\epsilon
    $$

    才把该采样点视作 source union 的外表面近似点。
    """

    points: list[Vector3] = []
    for body in source_bodies:
        for point in sample_body_surface(body, samples_per_axis=samples_per_axis):
            if union_signed_distance(point, source_bodies) < -1e-7:
                continue
            points.append(point)
    if not points:
        return math.inf
    if device == "cuda":
        try:
            return _surface_to_union_sdf_min_torch(points, target_bodies)
        except Exception as exc:
            raise RuntimeError("CUDA SDF evaluation failed") from exc
    return min(union_signed_distance(point, target_bodies) for point in points)


def _resolve_sdf_device(device: str) -> str:
    r"""解析 SDF 计算设备，默认 auto 优先 CUDA、失败回 CPU。"""

    if device == "cpu":
        return "cpu"
    if device not in {"auto", "cuda"}:
        raise ValueError(f"unsupported SDF device: {device!r}")
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    if device == "cuda":
        raise RuntimeError("SDF device 'cuda' requested but PyTorch CUDA is unavailable")
    return "cpu"


def _surface_to_union_sdf_min_torch(points: list[Vector3], target_bodies: list[CollisionBodyRecord]) -> float:
    r"""用 PyTorch/CUDA 批量查询 surface points 到 target union SDF。

    当前覆盖 v1 validator 支持的四类 primitive：box / sphere / cylinder /
    elliptic_cylinder。公式与 CPU path 保持同一近似边界。
    """

    import torch

    tensor = torch.tensor(points, dtype=torch.float32, device="cuda")
    distances = []
    for body in target_bodies:
        rotation = torch.tensor(_rotation_matrix_rows(body.world_pose.rpy), dtype=torch.float32, device="cuda")
        translation = torch.tensor(body.world_pose.pos, dtype=torch.float32, device="cuda")
        local = (tensor - translation) @ rotation
        geometry = body.geometry
        if isinstance(geometry, BoxGeometryCfg):
            half_size = torch.tensor(
                (geometry.size[0] / 2.0, geometry.size[1] / 2.0, geometry.size[2] / 2.0),
                dtype=torch.float32,
                device="cuda",
            )
            q = torch.abs(local) - half_size
            outside = torch.linalg.norm(torch.clamp(q, min=0.0), dim=1)
            inside = torch.minimum(torch.amax(q, dim=1), torch.zeros_like(outside))
            distances.append(outside + inside)
        elif isinstance(geometry, SphereGeometryCfg):
            distances.append(torch.linalg.norm(local, dim=1) - float(geometry.radius))
        elif isinstance(geometry, CylinderGeometryCfg):
            radial = torch.sqrt(local[:, 0] ** 2 + local[:, 1] ** 2) - float(geometry.radius)
            axial = torch.abs(local[:, 2]) - float(geometry.length / 2.0)
            outside = torch.sqrt(torch.clamp(radial, min=0.0) ** 2 + torch.clamp(axial, min=0.0) ** 2)
            inside = torch.minimum(torch.maximum(radial, axial), torch.zeros_like(outside))
            distances.append(outside + inside)
        elif isinstance(geometry, EllipticCylinderGeometryCfg):
            x = local[:, 0]
            y = local[:, 1]
            z = local[:, 2]
            radius_x = float(geometry.radius_x)
            radius_z = float(geometry.radius_z)
            scaled_radius = torch.sqrt((x / radius_x) ** 2 + (z / radius_z) ** 2)
            radial_norm = torch.sqrt(x * x + z * z)
            safe_norm = torch.clamp(radial_norm, min=1e-12)
            ux = x / safe_norm
            uz = z / safe_norm
            directional_boundary = 1.0 / torch.sqrt((ux / radius_x) ** 2 + (uz / radius_z) ** 2)
            center_boundary = torch.full_like(directional_boundary, min(radius_x, radius_z))
            boundary_radius = torch.where(radial_norm <= 1e-12, center_boundary, directional_boundary)
            radial = (scaled_radius - 1.0) * boundary_radius
            axial = torch.abs(y) - float(geometry.length / 2.0)
            outside = torch.sqrt(torch.clamp(radial, min=0.0) ** 2 + torch.clamp(axial, min=0.0) ** 2)
            inside = torch.minimum(torch.maximum(radial, axial), torch.zeros_like(outside))
            distances.append(outside + inside)
        else:
            raise TypeError(f"CUDA SDF path does not yet support {geometry.kind!r}")
    union = torch.stack(distances, dim=0).amin(dim=0)
    return float(union.amin().detach().cpu().item())


def _rotation_matrix_rows(rpy: Vector3) -> tuple[Vector3, Vector3, Vector3]:
    r"""局部 helper：返回与 `_collision_geometry.rpy_rotation_matrix` 一致的矩阵行。"""

    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _sdf_box(point: Vector3, half_size: Vector3) -> float:
    r"""axis-aligned box signed distance。"""

    q = (abs(point[0]) - half_size[0], abs(point[1]) - half_size[1], abs(point[2]) - half_size[2])
    outside = (max(q[0], 0.0), max(q[1], 0.0), max(q[2], 0.0))
    outside_dist = _norm(outside)
    inside_dist = min(max(q[0], q[1], q[2]), 0.0)
    return outside_dist + inside_dist


def _sdf_cylinder_z(point: Vector3, *, radius: float, half_length: float) -> float:
    r"""URDF primitive cylinder 的 z-axis signed distance。"""

    radial = math.sqrt(point[0] * point[0] + point[1] * point[1]) - radius
    axial = abs(point[2]) - half_length
    outside = math.sqrt(max(radial, 0.0) ** 2 + max(axial, 0.0) ** 2)
    inside = min(max(radial, axial), 0.0)
    return outside + inside


def _sdf_elliptic_cylinder_y(
    point: Vector3,
    *,
    radius_x: float,
    radius_z: float,
    half_length: float,
) -> float:
    r"""y-axis elliptic cylinder 的保守近似 signed distance。

    对椭圆截面 exact SDF 需要迭代求最近点。v1 采用径向归一化近似：
    $$
    \rho=\sqrt{(x/a)^2+(z/b)^2}.
    $$
    横截面距离使用局部方向上的边界半径换算。该近似对 validator 是可解释的，
    但 certificate 仍必须标注 sampled approximation。
    """

    x, y, z = point
    scaled_radius = math.sqrt((x / radius_x) ** 2 + (z / radius_z) ** 2)
    radial_norm = math.sqrt(x * x + z * z)
    if radial_norm <= 1e-12:
        boundary_radius = min(radius_x, radius_z)
    else:
        ux, uz = x / radial_norm, z / radial_norm
        boundary_radius = 1.0 / math.sqrt((ux / radius_x) ** 2 + (uz / radius_z) ** 2)
    radial = (scaled_radius - 1.0) * boundary_radius
    axial = abs(y) - half_length
    outside = math.sqrt(max(radial, 0.0) ** 2 + max(axial, 0.0) ** 2)
    inside = min(max(radial, axial), 0.0)
    return outside + inside


def _sample_box_surface(size: Vector3, *, density: int) -> list[Vector3]:
    r"""采样 box 六个面的规则网格点。"""

    hx, hy, hz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
    xs = _linspace(-hx, hx, density)
    ys = _linspace(-hy, hy, density)
    zs = _linspace(-hz, hz, density)
    points: list[Vector3] = []
    for x in xs:
        for y in ys:
            points.append((x, y, -hz))
            points.append((x, y, hz))
    for x in xs:
        for z in zs:
            points.append((x, -hy, z))
            points.append((x, hy, z))
    for y in ys:
        for z in zs:
            points.append((-hx, y, z))
            points.append((hx, y, z))
    return points


def _sample_cylinder_z_surface(radius: float, length: float, *, density: int) -> list[Vector3]:
    r"""采样 z-axis cylinder 的侧面与端盖。"""

    angles = _angles(density)
    zs = _linspace(-length / 2.0, length / 2.0, density)
    points: list[Vector3] = []
    for z in zs:
        for angle in angles:
            points.append((radius * math.cos(angle), radius * math.sin(angle), z))
    for z in (-length / 2.0, length / 2.0):
        for radial in _linspace(0.0, radius, density):
            for angle in angles:
                points.append((radial * math.cos(angle), radial * math.sin(angle), z))
    return points


def _sample_elliptic_cylinder_y_surface(
    radius_x: float,
    radius_z: float,
    length: float,
    *,
    density: int,
) -> list[Vector3]:
    r"""采样 y-axis elliptic cylinder 的侧面与端盖。"""

    angles = _angles(density)
    ys = _linspace(-length / 2.0, length / 2.0, density)
    points: list[Vector3] = []
    for y in ys:
        for angle in angles:
            points.append((radius_x * math.cos(angle), y, radius_z * math.sin(angle)))
    for y in (-length / 2.0, length / 2.0):
        for radial in _linspace(0.0, 1.0, density):
            for angle in angles:
                points.append((radial * radius_x * math.cos(angle), y, radial * radius_z * math.sin(angle)))
    return points


def _sample_sphere_surface(radius: float, *, density: int) -> list[Vector3]:
    r"""采样 sphere surface 的经纬网格。"""

    points: list[Vector3] = []
    latitudes = _linspace(0.0, math.pi, density + 1)
    longitudes = _angles(density)
    for theta in latitudes:
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        for phi in longitudes:
            points.append(
                (
                    radius * sin_theta * math.cos(phi),
                    radius * sin_theta * math.sin(phi),
                    radius * cos_theta,
                )
            )
    return points


def _linspace(start: float, stop: float, count: int) -> list[float]:
    r"""不依赖 numpy 的闭区间 linspace。"""

    if count <= 1:
        return [(start + stop) / 2.0]
    step = (stop - start) / float(count - 1)
    return [start + step * index for index in range(count)]


def _angles(density: int) -> list[float]:
    r"""按采样密度生成一圈角度。"""

    count = max(density * 4, 8)
    return [2.0 * math.pi * index / float(count) for index in range(count)]


def _norm(vector: Vector3) -> float:
    r"""三维欧氏范数。"""

    return math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)


__all__ = [
    "FingerPairClearance",
    "SdfClearanceCertificate",
    "SdfClearanceConfig",
    "SdfClearanceResult",
    "evaluate_finger_sdf_clearance",
    "sample_body_surface",
    "signed_distance_to_body",
    "union_signed_distance",
]
