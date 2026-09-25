"Closes mass and inertia from final collision geometry and part-specific density profiles. Mass is in kilograms and inertia in kilogram square meters."

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .asset_base import AssetCfgBase, HandCfg, InertialCfg, JointCfg, PalmCfg, PoseCfg
from .asset_schema_core import CollisionGeometryCfg, InertiaTensorCfg, MeshGeometryCfg

_FLOAT_TOLERANCE = 1e-12
"Absolute tolerance for comparing floating-point geometry values."


_ASSETS_ROOT = Path(__file__).resolve().parent
"Root used to resolve bundled hand references and custom geometry."


@dataclass
class DensityProfileCfg(AssetCfgBase):
    "Part-specific mass density values converted to kilograms per cubic meter."

    default: float = 650.0
    "Default value used when no more specific part density is declared."

    palm: float | None = None
    "Palm body density or configuration, depending on the enclosing schema."

    finger_link: float | None = None
    "Density or geometry settings for non-tip finger links."

    fingertip: float | None = None
    "Density or geometry settings for the primitive tip body."

    custom_tip: float | None = None
    "Density or geometry settings for mesh-based tip bodies."

    def __post_init__(self) -> None:
        self.default = _coerce_positive_density(self.default, field_name="density.default")
        self.palm = _coerce_optional_density(self.palm, field_name="density.palm")
        self.finger_link = _coerce_optional_density(self.finger_link, field_name="density.finger_link")
        self.fingertip = _coerce_optional_density(self.fingertip, field_name="density.fingertip")
        self.custom_tip = _coerce_optional_density(self.custom_tip, field_name="density.custom_tip")

    def for_palm(self) -> float:
        "Returns density and closure settings for the palm body."

        return self.palm if self.palm is not None else self.default

    def for_joint(self, joint: JointCfg) -> float:
        "Returns density and closure settings for one child link."

        has_mesh_collision = any(collision.geometry.kind == "mesh" for collision in joint.collisions)
        if joint.is_tip and has_mesh_collision and _is_procedural_cs_tip_joint(joint):
            return self.fingertip if self.fingertip is not None else self.default
        if joint.is_tip and has_mesh_collision:
            if self.custom_tip is not None:
                return self.custom_tip
            if self.fingertip is not None:
                return self.fingertip
            return self.default
        if joint.is_tip:
            return self.fingertip if self.fingertip is not None else self.default
        return self.finger_link if self.finger_link is not None else self.default


@dataclass
class AssetPhysicsCfg(AssetCfgBase):
    "Density and closure settings used to compute final mass and inertia from exported collision geometry."

    class_type: type[AssetPhysicsClosure] | None = None
    "Associated runtime implementation for this configuration class."

    enabled: bool = True
    "Whether this generation or validation condition is active."

    density: DensityProfileCfg | dict[str, Any] = field(default_factory=DensityProfileCfg)
    "Part-specific mass density in the unit represented by its constructor."

    min_mass: float = 1e-6
    "Minimum accepted rigid-body mass in kilograms."

    inertia_padding: float = 0.0
    "Diagonal regularization added to inertia eigenvalues, in kg m^2."

    mesh_backend: Literal["trimesh"] = "trimesh"
    "Implementation used to compute signed distances to mesh geometry."

    nonuniform_mesh_scale_policy: Literal["fail"] = "fail"
    "Policy for mesh assets with nonuniform scale during inertia closure."

    def __post_init__(self) -> None:
        if self.class_type is None:
            self.class_type = AssetPhysicsClosure
        if not isinstance(self.density, DensityProfileCfg):
            self.density = DensityProfileCfg(**dict(self.density))
        self.min_mass = float(self.min_mass)
        if self.min_mass <= 0.0:
            raise ValueError("AssetPhysicsCfg.min_mass must be positive")
        self.inertia_padding = float(self.inertia_padding)
        if self.inertia_padding < 0.0:
            raise ValueError("AssetPhysicsCfg.inertia_padding must be >= 0")
        if self.mesh_backend != "trimesh":
            raise ValueError(f"Unsupported mesh_backend: {self.mesh_backend!r}")
        if self.nonuniform_mesh_scale_policy != "fail":
            raise ValueError(
                "Unsupported nonuniform_mesh_scale_policy: "
                f"{self.nonuniform_mesh_scale_policy!r}"
            )


@dataclass
class _MassContribution:
    "Intermediate mass and first-moment contribution from one collision component."

    mass: float
    "Mass contribution from this collision component in kilograms."

    center_of_mass: tuple[float, float, float]
    "Rigid-body center of mass in the link frame, in meters."

    inertia_about_com: np.ndarray
    "Rigid-body inertia tensor about the center of mass, in kg m^2."

    backend: Literal["analytic", "trimesh"]
    "Backend selected for this geometry or signed-distance operation."


@dataclass
class _CanonicalMeshMassProperties:
    "Canonical mesh volume, centroid, and inertia used during physics closure."

    volume: float
    "Collision-mesh volume in cubic meters."

    center_of_mass: tuple[float, float, float]
    "Rigid-body center of mass in the link frame, in meters."

    inertia_about_com: np.ndarray
    "Rigid-body inertia tensor about the center of mass, in kg m^2."


class AssetPhysicsClosure:
    "Computes final body mass and inertia from geometry and the configured density profile."

    cfg: AssetPhysicsCfg

    def __init__(self, cfg: AssetPhysicsCfg):
        self.cfg = cfg

    def close(self, target: HandCfg, *, stage: str | None = None) -> HandCfg:
        "Closes mass and inertia for the declared collision geometry."

        if not self.cfg.enabled:
            return target.copy()

        closed_hand = target.copy()


        closed_hand.palm = self._close_palm(closed_hand.palm, stage=stage)



        closed_fingers = []
        for finger in closed_hand.fingers:
            closed_joints = [self._close_joint_child_link(joint, stage=stage) for joint in finger.joints]
            closed_fingers.append(finger.replace(joints=closed_joints))
        closed_hand.fingers = closed_fingers


        hand_metadata = dict(closed_hand.metadata)
        hand_metadata["physics_closure"] = {
            "enabled": True,
            "stage": stage or "unspecified",
            "mesh_backend": self.cfg.mesh_backend,
            "nonuniform_mesh_scale_policy": self.cfg.nonuniform_mesh_scale_policy,
        }
        closed_hand.metadata = hand_metadata
        return closed_hand

    def _close_palm(self, target: PalmCfg, *, stage: str | None) -> PalmCfg:

        closed_inertial, backend = _aggregate_collision_inertial(
            target.collisions,
            density=self.cfg.density.for_palm(),
            min_mass=self.cfg.min_mass,
            inertia_padding=self.cfg.inertia_padding,
            mesh_backend=self.cfg.mesh_backend,
            nonuniform_mesh_scale_policy=self.cfg.nonuniform_mesh_scale_policy,
        )
        if closed_inertial is None:
            return target

        metadata = dict(target.metadata)
        metadata["inertial_source"] = "collision_closure_v1"
        metadata["inertial_backend"] = backend
        metadata["inertial_stage"] = stage or "unspecified"
        return target.replace(inertial=closed_inertial, metadata=metadata)

    def _close_joint_child_link(self, target: JointCfg, *, stage: str | None) -> JointCfg:

        closed_inertial, backend = _aggregate_collision_inertial(
            target.collisions,
            density=self.cfg.density.for_joint(target),
            min_mass=self.cfg.min_mass,
            inertia_padding=self.cfg.inertia_padding,
            mesh_backend=self.cfg.mesh_backend,
            nonuniform_mesh_scale_policy=self.cfg.nonuniform_mesh_scale_policy,
        )
        if closed_inertial is None:
            return target

        metadata = dict(target.metadata)
        metadata["inertial_source"] = "collision_closure_v1"
        metadata["inertial_backend"] = backend
        metadata["inertial_stage"] = stage or "unspecified"
        return target.replace(inertial=closed_inertial, metadata=metadata)


def close_hand_physics(target: HandCfg, cfg: AssetPhysicsCfg | None, *, stage: str | None = None) -> HandCfg:
    "Computes final mass and inertia from collision geometry and part-specific density."

    if cfg is None:
        return target.copy()
    return AssetPhysicsClosure(cfg).close(target, stage=stage)


def _is_procedural_cs_tip_joint(joint: JointCfg) -> bool:

    metadata = dict(joint.metadata)
    return (
        metadata.get("tip_type") == "cs"
        or metadata.get("procedural_tip_type") == "cs"
        or metadata.get("procedural_mesh_kind") == "cs_tip"
    )


def _aggregate_collision_inertial(
    collisions: list[CollisionGeometryCfg],
    *,
    density: float,
    min_mass: float,
    inertia_padding: float,
    mesh_backend: Literal["trimesh"],
    nonuniform_mesh_scale_policy: Literal["fail"],
) -> tuple[InertialCfg | None, str]:

    if not collisions:
        return None, "none"

    contributions = [
        _mass_contribution_from_collision(
            collision,
            density=density,
            min_mass=min_mass,
            mesh_backend=mesh_backend,
            nonuniform_mesh_scale_policy=nonuniform_mesh_scale_policy,
        )
        for collision in collisions
    ]

    total_mass = sum(item.mass for item in contributions)
    if total_mass <= 0.0:
        raise ValueError("physics closure got non-positive total mass from collision geometry")



    # The combined center of mass is the mass-weighted mean of component centers.

    total_com = np.zeros(3, dtype=np.float64)
    for item in contributions:
        total_com += item.mass * np.asarray(item.center_of_mass, dtype=np.float64)
    total_com /= total_mass



    # Shift each component inertia to the combined center of mass with the parallel-axis theorem.

    total_inertia = np.zeros((3, 3), dtype=np.float64)
    for item in contributions:
        delta = np.asarray(item.center_of_mass, dtype=np.float64) - total_com
        total_inertia += item.inertia_about_com + item.mass * _parallel_axis_matrix(delta)

    backend_names = {item.backend for item in contributions}
    backend = "mixed" if len(backend_names) > 1 else next(iter(backend_names))

    inertial = InertialCfg(
        mass=total_mass,
        origin=PoseCfg(pos=(float(total_com[0]), float(total_com[1]), float(total_com[2]))),
        inertia=InertiaTensorCfg(
            ixx=float(total_inertia[0, 0]),
            iyy=float(total_inertia[1, 1]),
            izz=float(total_inertia[2, 2]),
            ixy=float(total_inertia[0, 1]),
            ixz=float(total_inertia[0, 2]),
            iyz=float(total_inertia[1, 2]),
        ),
        inertia_padding=inertia_padding,
    )
    return inertial, backend


def _mass_contribution_from_collision(
    collision: CollisionGeometryCfg,
    *,
    density: float,
    min_mass: float,
    mesh_backend: Literal["trimesh"],
    nonuniform_mesh_scale_policy: Literal["fail"],
) -> _MassContribution:

    geometry = collision.geometry
    rotation = _rotation_matrix(collision.origin.rpy)

    if geometry.kind == "box":
        sx, sy, sz = (float(geometry.size[0]), float(geometry.size[1]), float(geometry.size[2]))
        volume = sx * sy * sz
        mass = max(density * volume, min_mass)
        local_inertia = InertialCfg.from_box((sx, sy, sz), density=mass / volume, min_mass=mass).inertia
        inertia_about_com = _rotate_inertia(
            _inertia_cfg_to_matrix(local_inertia),
            rotation,
        )
        return _MassContribution(
            mass=mass,
            center_of_mass=collision.origin.pos,
            inertia_about_com=inertia_about_com,
            backend="analytic",
        )

    if geometry.kind == "cylinder":
        radius = float(geometry.radius)
        length = float(geometry.length)
        volume = math.pi * radius * radius * length
        mass = max(density * volume, min_mass)
        local_inertia = InertialCfg.from_cylinder(
            radius,
            length,
            density=mass / volume,
            principal_axis="z",
            min_mass=mass,
        ).inertia
        inertia_about_com = _rotate_inertia(
            _inertia_cfg_to_matrix(local_inertia),
            rotation,
        )
        return _MassContribution(
            mass=mass,
            center_of_mass=collision.origin.pos,
            inertia_about_com=inertia_about_com,
            backend="analytic",
        )

    if geometry.kind == "elliptic_cylinder":
        radius_x = float(geometry.radius_x)
        radius_z = float(geometry.radius_z)
        length = float(geometry.length)
        volume = math.pi * radius_x * radius_z * length
        mass = max(density * volume, min_mass)
        local_inertia = InertialCfg.from_elliptic_cylinder(
            radius_x,
            radius_z,
            length,
            density=mass / volume,
            principal_axis="y",
            min_mass=mass,
        ).inertia
        inertia_about_com = _rotate_inertia(
            _inertia_cfg_to_matrix(local_inertia),
            rotation,
        )
        return _MassContribution(
            mass=mass,
            center_of_mass=collision.origin.pos,
            inertia_about_com=inertia_about_com,
            backend="analytic",
        )

    if geometry.kind == "sphere":
        radius = float(geometry.radius)
        volume = 4.0 * math.pi * radius**3 / 3.0
        mass = max(density * volume, min_mass)
        local_inertia = InertialCfg.from_sphere(
            radius,
            density=mass / volume,
            min_mass=mass,
        ).inertia
        inertia_about_com = _rotate_inertia(
            _inertia_cfg_to_matrix(local_inertia),
            rotation,
        )
        return _MassContribution(
            mass=mass,
            center_of_mass=collision.origin.pos,
            inertia_about_com=inertia_about_com,
            backend="analytic",
        )

    if geometry.kind != "mesh":
        raise ValueError(f"Unsupported collision geometry for physics closure: {geometry.kind!r}")

    if mesh_backend != "trimesh":
        raise ValueError(f"Unsupported mesh backend for physics closure: {mesh_backend!r}")

    uniform_scale = _extract_uniform_scale(
        geometry,
        nonuniform_mesh_scale_policy=nonuniform_mesh_scale_policy,
    )
    canonical = _canonical_mesh_mass_properties(_mesh_cache_key(geometry.file_path))


    scaled_volume = canonical.volume * uniform_scale**3
    mass = max(density * scaled_volume, min_mass)
    scaled_center = tuple(component * uniform_scale for component in canonical.center_of_mass)
    rotated_center = _apply_rotation(rotation, scaled_center)
    center_of_mass = (
        collision.origin.pos[0] + rotated_center[0],
        collision.origin.pos[1] + rotated_center[1],
        collision.origin.pos[2] + rotated_center[2],
    )
    scaled_inertia = canonical.inertia_about_com * (density * uniform_scale**5)
    inertia_about_com = _rotate_inertia(scaled_inertia, rotation)
    return _MassContribution(
        mass=mass,
        center_of_mass=center_of_mass,
        inertia_about_com=inertia_about_com,
        backend="trimesh",
    )


def _coerce_positive_density(value: Any, *, field_name: str) -> float:

    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{field_name} must be positive, got {value}")
    return value


def _coerce_optional_density(value: Any, *, field_name: str) -> float | None:

    if value is None:
        return None
    return _coerce_positive_density(value, field_name=field_name)


def _rotation_matrix(rpy: tuple[float, float, float]) -> np.ndarray:

    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ),
        dtype=np.float64,
    )


def _apply_rotation(rotation: np.ndarray, point: tuple[float, float, float]) -> tuple[float, float, float]:

    rotated = rotation @ np.asarray(point, dtype=np.float64)
    return (float(rotated[0]), float(rotated[1]), float(rotated[2]))


def _rotate_inertia(inertia: np.ndarray, rotation: np.ndarray) -> np.ndarray:

    return rotation @ inertia @ rotation.T


def _parallel_axis_matrix(delta: np.ndarray) -> np.ndarray:

    distance_sq = float(delta @ delta)
    return distance_sq * np.eye(3, dtype=np.float64) - np.outer(delta, delta)


def _inertia_cfg_to_matrix(inertia: InertiaTensorCfg) -> np.ndarray:

    return np.asarray(
        (
            (inertia.ixx, inertia.ixy, inertia.ixz),
            (inertia.ixy, inertia.iyy, inertia.iyz),
            (inertia.ixz, inertia.iyz, inertia.izz),
        ),
        dtype=np.float64,
    )


def _extract_uniform_scale(
    geometry: MeshGeometryCfg,
    *,
    nonuniform_mesh_scale_policy: Literal["fail"],
) -> float:

    sx, sy, sz = (float(geometry.scale[0]), float(geometry.scale[1]), float(geometry.scale[2]))
    if math.isclose(sx, sy, rel_tol=0.0, abs_tol=_FLOAT_TOLERANCE) and math.isclose(
        sx,
        sz,
        rel_tol=0.0,
        abs_tol=_FLOAT_TOLERANCE,
    ):
        return sx
    if nonuniform_mesh_scale_policy == "fail":
        raise ValueError(
            "physics closure only supports uniform mesh scale in v1, "
            f"got {geometry.scale!r} for {geometry.file_path!r}"
        )
    raise ValueError(f"Unsupported nonuniform_mesh_scale_policy: {nonuniform_mesh_scale_policy!r}")


def _mesh_cache_key(file_path: str) -> tuple[str, int, int]:

    resolved = _resolve_mesh_path(file_path)
    stat = resolved.stat()
    return (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))


@lru_cache(maxsize=128)
def _canonical_mesh_mass_properties(cache_key: tuple[str, int, int]) -> _CanonicalMeshMassProperties:



    path = Path(cache_key[0])
    mesh = _load_checked_trimesh(path)


    mass_properties = mesh.mass_properties
    return _CanonicalMeshMassProperties(
        volume=float(mass_properties.volume),
        center_of_mass=(
            float(mass_properties.center_mass[0]),
            float(mass_properties.center_mass[1]),
            float(mass_properties.center_mass[2]),
        ),
        inertia_about_com=np.asarray(mass_properties.inertia, dtype=np.float64),
    )


def _resolve_mesh_path(file_path: str) -> Path:

    if file_path.startswith("package://"):
        raise ValueError(
            "physics closure expects local mesh paths before export, "
            f"got package path {file_path!r}"
        )
    raw_path = Path(os.path.expanduser(file_path))
    if raw_path.is_absolute():
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)
        return raw_path
    for candidate in (Path.cwd() / raw_path, _ASSETS_ROOT / raw_path):
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Unable to resolve mesh path for physics closure: {file_path!r}")


def _load_checked_trimesh(path: Path):

    import trimesh

    mesh = trimesh.load(path, force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"physics closure expects triangle mesh, got {type(mesh).__name__}: {path}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"physics closure got empty mesh: {path}")
    if not mesh.is_volume:
        raise ValueError(f"physics closure requires watertight positive-volume mesh: {path}")
    return mesh


__all__ = [
    "DensityProfileCfg",
    "AssetPhysicsCfg",
    "AssetPhysicsClosure",
    "close_hand_physics",
]
