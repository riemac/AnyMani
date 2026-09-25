"Checks collision clearance using distances in meters and the declared pair ownership."

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Literal

from ..asset_base import HandCfg
from ..asset_schema_core import BoxGeometryCfg, CylinderGeometryCfg, EllipticCylinderGeometryCfg, MeshGeometryCfg, SphereGeometryCfg, Vector3
from ._collision_geometry import (
    CollisionBodyRecord,
    UnsupportedGeometryPolicy,
    apply_inverse_pose,
    apply_pose,
    extract_finger_collision_bodies,
)
from ._mesh_sdf import (
    MeshSdfBackend,
    MeshSdfQueryStats,
    sample_mesh_surface,
    signed_distance_to_mesh_body,
    signed_distance_to_mesh_body_batch,
)


NOT_CERTIFIED = (
    "all_pose_collision_free",
    "mesh_exact_clearance",
    "trajectory_safety",
    "physics_runtime_safety",
)
"Marker indicating that the geometry has no complete signed-distance certificate."


@dataclass(frozen=True)
class SdfClearanceConfig:
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    min_clearance: float
    surface_samples_per_axis: int = 5
    unsupported_policy: UnsupportedGeometryPolicy = "fail"
    tolerance: float = 1e-9
    device: Literal["auto", "cuda", "cpu"] = "auto"
    "Compute device selected for signed-distance validation."

    mesh_backend: MeshSdfBackend = "auto"
    "Implementation used to compute signed distances to mesh geometry."

    mesh_surface_samples: int = 4096
    "Number of surface samples used by mesh-based validation."


@dataclass(frozen=True)
class FingerPairClearance:
    "Symmetric clearance diagnostics for one ordered finger pair."

    finger_i: str
    finger_j: str
    clearance: float
    direction_i_to_j: float
    direction_j_to_i: float

    def to_dict(self) -> dict[str, float | str]:
        'Serializes the typed object as a dictionary.'

        return {
            "finger_i": self.finger_i,
            "finger_j": self.finger_j,
            "clearance": self.clearance,
            "direction_i_to_j": self.direction_i_to_j,
            "direction_j_to_i": self.direction_j_to_i,
        }


@dataclass
class SdfClearanceCertificate:
    "Source IDs, query statistics, and measured surface-clearance evidence."

    pose_scope: str = "post_mutate_home_pose"
    geometry_scope: str = "collision_geometry_only"
    sdf_kind: str = "sampled_surface_sdf_approx"
    complete: bool = True
    skipped_bodies: list[dict[str, str]] = field(default_factory=list)
    not_certified: list[str] = field(default_factory=lambda: list(NOT_CERTIFIED))
    min_clearance: float = 0.0
    device: str = "cpu"
    mesh_sdf: dict[str, object] = field(default_factory=dict)
    pair_clearances: list[dict[str, float | str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        'Serializes the typed object as a dictionary.'

        return {
            "pose_scope": self.pose_scope,
            "geometry_scope": self.geometry_scope,
            "sdf_kind": self.sdf_kind,
            "complete": self.complete,
            "skipped_bodies": list(self.skipped_bodies),
            "not_certified": list(self.not_certified),
            "min_clearance": self.min_clearance,
            "device": self.device,
            "mesh_sdf": dict(self.mesh_sdf),
            "pair_clearances": list(self.pair_clearances),
        }


@dataclass(frozen=True)
class SdfClearanceResult:
    "Result record with ordered geometry, provenance, and validation status."

    passed: bool
    certificate: SdfClearanceCertificate
    violations: list[FingerPairClearance]


def evaluate_finger_sdf_clearance(hand: HandCfg, cfg: SdfClearanceConfig) -> SdfClearanceResult:
    "Measures symmetric surface clearance for every configured finger pair."

    extraction = extract_finger_collision_bodies(hand, unsupported_policy=cfg.unsupported_policy)
    device = _resolve_sdf_device(cfg.device)
    mesh_stats = MeshSdfQueryStats(requested_backend=cfg.mesh_backend)
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
                    mesh_surface_samples=cfg.mesh_surface_samples,
                    device=device,
                    mesh_backend=cfg.mesh_backend,
                    mesh_stats=mesh_stats,
                )
                clearance_j_to_i = _surface_to_union_sdf_min(
                    source_bodies=bodies_j,
                    target_bodies=bodies_i,
                    samples_per_axis=cfg.surface_samples_per_axis,
                    mesh_surface_samples=cfg.mesh_surface_samples,
                    device=device,
                    mesh_backend=cfg.mesh_backend,
                    mesh_stats=mesh_stats,
                )
            except RuntimeError:
                if cfg.device == "cuda" or cfg.mesh_backend == "warp":
                    raise
                device = "cpu"
                clearance_i_to_j = _surface_to_union_sdf_min(
                    source_bodies=bodies_i,
                    target_bodies=bodies_j,
                    samples_per_axis=cfg.surface_samples_per_axis,
                    mesh_surface_samples=cfg.mesh_surface_samples,
                    device=device,
                    mesh_backend="trimesh",
                    mesh_stats=mesh_stats,
                )
                clearance_j_to_i = _surface_to_union_sdf_min(
                    source_bodies=bodies_j,
                    target_bodies=bodies_i,
                    samples_per_axis=cfg.surface_samples_per_axis,
                    mesh_surface_samples=cfg.mesh_surface_samples,
                    device=device,
                    mesh_backend="trimesh",
                    mesh_stats=mesh_stats,
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
        mesh_sdf=mesh_stats.to_dict(),
        pair_clearances=[pair.to_dict() for pair in pair_clearances],
    )
    return SdfClearanceResult(
        passed=not violations and certificate.complete,
        certificate=certificate,
        violations=violations,
    )


def signed_distance_to_body(
    point_world: Vector3,
    body: CollisionBodyRecord,
    *,
    mesh_backend: MeshSdfBackend = "trimesh",
    device: str = "cpu",
    mesh_stats: MeshSdfQueryStats | None = None,
) -> float:
    "Returns signed distance to the declared primitive or mesh body."

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
    if isinstance(geometry, MeshGeometryCfg):
        return signed_distance_to_mesh_body(
            point_world,
            body,
            backend=mesh_backend,
            device=device,
            stats=mesh_stats,
        )
    raise TypeError(f"unsupported SDF body geometry: {type(geometry).__name__}")


def union_signed_distance(
    point_world: Vector3,
    bodies: list[CollisionBodyRecord],
    *,
    mesh_backend: MeshSdfBackend = "trimesh",
    device: str = "cpu",
    mesh_stats: MeshSdfQueryStats | None = None,
) -> float:
    "Returns minimum signed distance to the union of a finger collision body set."

    if not bodies:
        return math.inf
    return min(
        signed_distance_to_body(
            point_world,
            body,
            mesh_backend=mesh_backend,
            device=device,
            mesh_stats=mesh_stats,
        )
        for body in bodies
    )


def sample_body_surface(
    body: CollisionBodyRecord,
    *,
    samples_per_axis: int,
    mesh_surface_samples: int = 4096,
) -> list[Vector3]:
    'Selects surface points for the selected collision body.'

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
    elif isinstance(geometry, MeshGeometryCfg):
        return sample_mesh_surface(body, sample_count=max(int(mesh_surface_samples), 4))
    else:
        raise TypeError(f"unsupported surface sampling geometry: {type(geometry).__name__}")
    return [apply_pose(body.world_pose, point) for point in local_points]


def _surface_to_union_sdf_min(
    *,
    source_bodies: list[CollisionBodyRecord],
    target_bodies: list[CollisionBodyRecord],
    samples_per_axis: int,
    mesh_surface_samples: int,
    device: str,
    mesh_backend: MeshSdfBackend,
    mesh_stats: MeshSdfQueryStats,
) -> float:

    candidate_points: list[Vector3] = []
    for body in source_bodies:
        if isinstance(body.geometry, MeshGeometryCfg):
            mesh_stats.mesh_sample_count += 1
        for point in sample_body_surface(
            body,
            samples_per_axis=samples_per_axis,
            mesh_surface_samples=mesh_surface_samples,
        ):
            candidate_points.append(point)
    points = _filter_union_surface_points(
        candidate_points,
        source_bodies,
        mesh_backend=mesh_backend,
        device=device,
        mesh_stats=mesh_stats,
    )
    if not points:
        return math.inf
    if device == "cuda":
        try:
            return _surface_to_union_sdf_min_torch(
                points,
                target_bodies,
                mesh_backend=mesh_backend,
                mesh_stats=mesh_stats,
            )
        except Exception as exc:
            if isinstance(exc, ValueError):
                raise
            raise RuntimeError("CUDA SDF evaluation failed") from exc
    return min(
        union_signed_distance(
            point,
            target_bodies,
            mesh_backend="trimesh",
            device="cpu",
            mesh_stats=mesh_stats,
        )
        for point in points
    )


def _filter_union_surface_points(
    points: list[Vector3],
    source_bodies: list[CollisionBodyRecord],
    *,
    mesh_backend: MeshSdfBackend,
    device: str,
    mesh_stats: MeshSdfQueryStats,
) -> list[Vector3]:
    """Keeps only samples on the source finger's union surface.

    Points buried inside another primitive of the same finger do not represent inter-finger contact and must not constrain pair clearance.
    """

    if not points:
        return []

    union_distances = [math.inf for _ in points]
    for body in source_bodies:
        geometry = body.geometry
        if isinstance(geometry, MeshGeometryCfg):
            distances = signed_distance_to_mesh_body_batch(
                points,
                body,
                backend=mesh_backend,
                device=device,
                stats=mesh_stats,
            )
            for index, distance in enumerate(distances):
                union_distances[index] = min(union_distances[index], float(distance))
        else:
            for index, point in enumerate(points):
                union_distances[index] = min(union_distances[index], signed_distance_to_body(point, body))
    return [point for point, distance in zip(points, union_distances) if distance >= -1e-7]


def _resolve_sdf_device(device: str) -> str:

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


def _surface_to_union_sdf_min_torch(
    points: list[Vector3],
    target_bodies: list[CollisionBodyRecord],
    *,
    mesh_backend: MeshSdfBackend,
    mesh_stats: MeshSdfQueryStats,
) -> float:

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
        elif isinstance(geometry, MeshGeometryCfg):
            mesh_distances = signed_distance_to_mesh_body_batch(
                points,
                body,
                backend=mesh_backend,
                device="cuda",
                stats=mesh_stats,
            )
            distances.append(torch.tensor(mesh_distances, dtype=torch.float32, device="cuda"))
        else:
            raise TypeError(f"CUDA SDF path does not yet support {geometry.kind!r}")
    union = torch.stack(distances, dim=0).amin(dim=0)
    return float(union.amin().detach().cpu().item())


def _rotation_matrix_rows(rpy: Vector3) -> tuple[Vector3, Vector3, Vector3]:

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

    q = (abs(point[0]) - half_size[0], abs(point[1]) - half_size[1], abs(point[2]) - half_size[2])
    outside = (max(q[0], 0.0), max(q[1], 0.0), max(q[2], 0.0))
    outside_dist = _norm(outside)
    inside_dist = min(max(q[0], q[1], q[2]), 0.0)
    return outside_dist + inside_dist


def _sdf_cylinder_z(point: Vector3, *, radius: float, half_length: float) -> float:

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

    if count <= 1:
        return [(start + stop) / 2.0]
    step = (stop - start) / float(count - 1)
    return [start + step * index for index in range(count)]


def _angles(density: int) -> list[float]:

    count = max(density * 4, 8)
    return [2.0 * math.pi * index / float(count) for index in range(count)]


def _norm(vector: Vector3) -> float:

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
