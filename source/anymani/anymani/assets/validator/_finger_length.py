"Measures link-chain and fingertip length in the declared palm frame."

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
"Anatomical role for a thumb or non-thumb finger."


NOT_CERTIFIED = (
    "all_pose_length",
    "bent_chain_geodesic_length",
    "physics_runtime_safety",
)
"Marker indicating that the geometry has no complete signed-distance certificate."


@dataclass(frozen=True)
class FingerLengthConfig:
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    max_thumb_length: float | None = None
    max_non_thumb_length: float | None = None
    tolerance: float = 1e-9
    unsupported_policy: UnsupportedGeometryPolicy = "fail"


@dataclass(frozen=True)
class FingerLengthMeasurement:
    "Measured palm-relative proximal-to-distal length for one finger chain."

    finger_name: str
    role: FingerRole
    axis: Vector3
    axis_source: str
    min_projection: float
    max_projection: float
    axial_length: float
    threshold: float | None

    def to_dict(self) -> dict[str, Any]:
        'Serializes the typed object as a dictionary.'

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
    "Length result and threshold evidence for one generated candidate."

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
        'Serializes the typed object as a dictionary.'

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
    "Result record with ordered geometry, provenance, and validation status."

    passed: bool
    certificate: FingerLengthCertificate
    measurements: list[FingerLengthMeasurement]
    violations: list[FingerLengthMeasurement]


def evaluate_finger_axial_length(hand: HandCfg, cfg: FingerLengthConfig) -> FingerLengthResult:
    "Measures each finger along its nominal distal axis and checks configured meter thresholds."

    extraction = extract_finger_collision_bodies(hand, unsupported_policy=cfg.unsupported_policy)
    link_poses_by_finger = extract_finger_link_poses(hand)
    measurements: list[FingerLengthMeasurement] = []
    violations: list[FingerLengthMeasurement] = []

    for finger in hand.fingers:
        finger_bodies = extraction.bodies_by_finger.get(finger.name, [])
        if not finger_bodies:
            continue

        axis, axis_source = _nominal_distal_axis(
            finger_name=finger.name,
            joints=finger.joints,
            link_poses=link_poses_by_finger.get(finger.name, []),
        )
        min_projection, max_projection = _projection_interval_for_finger(finger_bodies, axis=axis)
        axial_length = max_projection - min_projection
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
            violations.append(measurement)

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
    'Computes finger axial lengths.'

    return evaluate_finger_axial_length(hand, FingerLengthConfig()).measurements


def _nominal_distal_axis(
    *,
    finger_name: str,
    joints,
    link_poses: list[PoseCfg],
) -> tuple[Vector3, str]:

    if not joints or not link_poses:
        raise ValueError(f"finger '{finger_name}' has no joint/link pose to define nominal distal axis")

    if finger_name != "thumb":
        root_rotation = rpy_rotation_matrix(link_poses[0].rpy)
        axis = _normalize(apply_rotation(root_rotation, (0.0, 1.0, 0.0)))
        return axis, "root_link_local_+y"


    start_index = next(
        (index for index, joint in enumerate(joints) if "cmc1" not in str(joint.child).lower()),
        0,
    )
    distal_indices = [index for index, joint in enumerate(joints) if not bool(joint.is_tip)]
    end_index = distal_indices[-1] if distal_indices else len(joints) - 1

    if end_index > start_index:
        start_pose = link_poses[start_index]
        end_pose = link_poses[end_index]
        axis_candidate = (
            end_pose.pos[0] - start_pose.pos[0],
            end_pose.pos[1] - start_pose.pos[1],
            end_pose.pos[2] - start_pose.pos[2],
        )
        if _norm(axis_candidate) > 1e-12:
            return _normalize(axis_candidate), f"{joints[start_index].child}_to_{joints[end_index].child}"

    fallback_rotation = rpy_rotation_matrix(link_poses[start_index].rpy)
    fallback_axis = _normalize(apply_rotation(fallback_rotation, (0.0, 1.0, 0.0)))
    return fallback_axis, f"{joints[start_index].child}_local_+y_fallback"


def _projection_interval_for_finger(
    bodies: list[CollisionBodyRecord],
    *,
    axis: Vector3,
) -> tuple[float, float]:

    min_projection = math.inf
    max_projection = -math.inf

    for body in bodies:
        body_min, body_max = _projection_interval_for_body(body, axis=axis)
        min_projection = min(min_projection, body_min)
        max_projection = max(max_projection, body_max)

    if not math.isfinite(min_projection) or not math.isfinite(max_projection):
        raise ValueError("finger axial length requires at least one valid collision body projection interval")
    return min_projection, max_projection


def _projection_interval_for_body(body: CollisionBodyRecord, *, axis: Vector3) -> tuple[float, float]:

    geometry = body.geometry
    center_projection = _dot(body.world_pose.pos, axis)

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

    local_axis = _world_axis_to_local_axis(axis, pose=pose)
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

    local_axis = _world_axis_to_local_axis(axis, pose=pose)
    axial_half = abs(local_axis[1]) * (length * 0.5)
    radial_half = radius * math.sqrt(max(0.0, local_axis[0] ** 2 + local_axis[2] ** 2))
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

    local_axis = _world_axis_to_local_axis(axis, pose=pose)
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

    local_vertices = _load_mesh_vertices_local(geometry.file_path, _scale_tuple(geometry.scale))
    if local_vertices.size == 0:
        raise ValueError(f"finger axial length got empty mesh vertices: {geometry.file_path}")
    local_axis = np.asarray(_world_axis_to_local_axis(axis, pose=pose), dtype=np.float64)
    projections = local_vertices @ local_axis
    return center_projection + float(np.min(projections)), center_projection + float(np.max(projections))


def _world_axis_to_local_axis(axis_world: Vector3, *, pose: PoseCfg) -> Vector3:

    rotation = rpy_rotation_matrix(pose.rpy)
    return (
        rotation[0][0] * axis_world[0] + rotation[1][0] * axis_world[1] + rotation[2][0] * axis_world[2],
        rotation[0][1] * axis_world[0] + rotation[1][1] * axis_world[1] + rotation[2][1] * axis_world[2],
        rotation[0][2] * axis_world[0] + rotation[1][2] * axis_world[1] + rotation[2][2] * axis_world[2],
    )


def _scale_tuple(scale: Vector3) -> tuple[float, float, float]:

    return float(scale[0]), float(scale[1]), float(scale[2])


def _resolve_mesh_path(file_path: str | Path) -> Path:

    path = Path(file_path).expanduser()
    return path if path.is_absolute() else path.resolve()


def _load_mesh_vertices_local(file_path: str, scale: tuple[float, float, float]) -> np.ndarray:

    return _load_mesh_vertices_local_cached(str(_resolve_mesh_path(file_path)), scale)


@lru_cache(maxsize=128)
def _load_mesh_vertices_local_cached(file_path: str, scale: tuple[float, float, float]) -> np.ndarray:

    import trimesh

    mesh = trimesh.load(file_path, force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"finger axial length expects a triangle mesh, got {type(mesh).__name__}: {file_path}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"finger axial length got empty mesh: {file_path}")

    mesh = mesh.copy()
    mesh.apply_scale(scale)
    return np.asarray(mesh.vertices, dtype=np.float64)


def _dot(lhs: Vector3, rhs: Vector3) -> float:

    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]


def _norm(vector: Vector3) -> float:

    return math.sqrt(_dot(vector, vector))


def _normalize(vector: Vector3) -> Vector3:

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
