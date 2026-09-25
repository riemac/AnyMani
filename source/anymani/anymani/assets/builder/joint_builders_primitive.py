"Builds box and cylinder joint links from typed dimensions and local poses."

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ..asset_base import JointCfg
from ..asset_builders import JointBuilder, JointBuilderCfg
from ..asset_schema_core import (
    CollisionGeometryCfg,
    InertialCfg,
    JointLimitCfg,
    JointPropertiesCfg,
    PoseCfg,
    Vector3,
    VisualGeometryCfg,
    _ensure_tuple,
)
from ..procedural_meshes import make_procedural_cs_tip_uri

_DEFAULT_DENSITY = 650.0
"Default mass density used when no part-specific override is set."


def _pose_from_value(value: PoseCfg | Sequence[float] | Mapping[str, Any] | None) -> PoseCfg:

    return PoseCfg.from_value(value)


def _add_rpy(lhs: Vector3, rhs: Vector3) -> Vector3:

    return (lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2])


def _make_geometry_pose(
    *,
    offset: PoseCfg,
    default_pos: Vector3,
    default_rpy: Vector3 = (0.0, 0.0, 0.0),
    center_on_joint: bool = False,
) -> PoseCfg:


    base = offset.pos if center_on_joint else default_pos
    return PoseCfg(pos=base, rpy=_add_rpy(default_rpy, offset.rpy))


def _box_inertia(size: Vector3, mass: float) -> dict[str, float]:
    sx, sy, sz = size
    return {
        "ixx": mass * (sy * sy + sz * sz) / 12.0,
        "iyy": mass * (sx * sx + sz * sz) / 12.0,
        "izz": mass * (sx * sx + sy * sy) / 12.0,
    }


def _cylinder_inertia(radius: float, length: float, mass: float) -> dict[str, float]:
    return {
        "ixx": mass * (3.0 * radius * radius + length * length) / 12.0,
        "iyy": mass * radius * radius / 2.0,
        "izz": mass * (3.0 * radius * radius + length * length) / 12.0,
    }


def _elliptic_cylinder_inertia(radius_x: float, radius_z: float, length: float, mass: float) -> dict[str, float]:

    return {
        "ixx": mass * (3.0 * radius_z * radius_z + length * length) / 12.0,
        "iyy": mass * (radius_x * radius_x + radius_z * radius_z) / 4.0,
        "izz": mass * (3.0 * radius_x * radius_x + length * length) / 12.0,
    }


def _sphere_inertia(radius: float, mass: float) -> dict[str, float]:
    moment = 2.0 * mass * radius * radius / 5.0
    return {"ixx": moment, "iyy": moment, "izz": moment}


def _estimate_mass(*, volume: float, cfg_mass: float | None, density: float) -> float:

    return float(cfg_mass) if cfg_mass is not None else max(volume * density, 1e-6)


@dataclass
class PrimJointBuilderCfg(JointBuilderCfg):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    class_type: type[PrimJointBuilder] | type[ComPrimJointBuilder] | None = None
    "Associated runtime implementation for this configuration class."

    name: str = "joint"
    "Stable semantic identifier preserved in the output metadata."

    parent: str = "palm"
    "Parent link or owner identifier in the source topology."

    child: str | None = None
    "Child link name in the source kinematic tree."

    mesh: dict[str, Any] = field(default_factory=dict)
    "Mesh source and per-axis scale for one collision or visual component."

    joint_type: Literal["revolute", "fixed"] = "revolute"
    "Whether the link connection is fixed or actuated by a revolute coordinate."

    origin: PoseCfg | Sequence[float] | Mapping[str, Any] | None = field(default_factory=PoseCfg)
    "Rigid transform with translation in meters and rotation in radians."

    axis: Vector3 = (0.0, 0.0, 1.0)
    "Unit direction expressed in the owning local frame."

    limit: JointLimitCfg | Sequence[float] | Mapping[str, Any] | None = (-math.pi, math.pi)
    "Lower and upper joint-coordinate bounds in radians."

    joint_properties: JointPropertiesCfg | Mapping[str, Any] | None = None
    "Optional effort, velocity, and friction values aligned to the source chain."

    density: float = _DEFAULT_DENSITY
    "Part-specific mass density in the unit represented by its constructor."

    mass: float | None = None
    "Optional explicit mass in kilograms; final closure may replace it."

    is_tip: bool = False
    "Whether this generation or validation condition is active."

    metadata: dict[str, Any] = field(default_factory=dict)
    "Geometry and provenance fields written with the joint sidecar."

    def __post_init__(self):
        super().__post_init__()
        self.origin = _pose_from_value(self.origin)
        self.axis = _ensure_tuple(self.axis, length=3, field_name="prim_joint.axis")
        self.density = float(self.density)
        if self.density <= 0.0:
            raise ValueError("density must be positive")
        if self.mass is not None:
            self.mass = float(self.mass)
            if self.mass <= 0.0:
                raise ValueError("mass must be positive")
        if self.class_type in {None, JointBuilder}:
            mesh_kind = str(self.mesh.get("type", self.mesh.get("kind", "box"))).lower()
            self.class_type = ComPrimJointBuilder if mesh_kind in {"cs", "bs"} else PrimJointBuilder


class PrimJointBuilder(JointBuilder):
    "Builds a configured hand component from typed geometry and local frames."

    cfg: PrimJointBuilderCfg

    def __init__(self, cfg: PrimJointBuilderCfg):
        super().__init__(cfg)
        self.cfg = cfg

    def build(self) -> JointCfg:
        "Builds the configured geometry component from typed dimensions and local frames."

        geom_kind = str(self.cfg.mesh.get("type", self.cfg.mesh.get("kind", "box"))).lower()
        if geom_kind == "box":
            collisions, visuals, inertial = self._build_box()
        elif geom_kind == "cylinder":
            collisions, visuals, inertial = self._build_cylinder()
        elif geom_kind == "elliptic_cylinder":
            collisions, visuals, inertial = self._build_elliptic_cylinder()
        elif geom_kind == "sphere":
            collisions, visuals, inertial = self._build_sphere()
        else:
            raise ValueError(f"Unsupported primitive joint mesh type: {geom_kind}")

        return JointCfg(
            name=self.cfg.name,
            parent=self.cfg.parent,
            child=self.cfg.child,
            joint_type=self.cfg.joint_type,  # revolute / fixed
            axis=self.cfg.axis,
            limit=self.cfg.limit,
            joint_properties=self.cfg.joint_properties,
            origin=self.cfg.origin,
            inertial=inertial,
            collisions=collisions,
            visuals=visuals,
            is_tip=self.cfg.is_tip,
            metadata=self.cfg.metadata.copy(),
        )

    def _build_box(self) -> tuple[list[CollisionGeometryCfg], list[VisualGeometryCfg], InertialCfg]:




        mesh = self.cfg.mesh
        if "size" in mesh:
            size = _ensure_tuple(mesh["size"], length=3, field_name="box.size")
        else:
            size = (
                float(mesh["width"]),
                float(mesh["length"]),
                float(mesh["height"]),
            )

        offset = _pose_from_value(mesh.get("offset", mesh.get("origin")))
        center_on_joint = bool(mesh.get("center_on_joint", False))
        origin = _make_geometry_pose(
            offset=offset,
            default_pos=(offset.pos[0], size[1] / 2.0 + offset.pos[1], offset.pos[2]),
            center_on_joint=center_on_joint,
        )
        mass = _estimate_mass(volume=size[0] * size[1] * size[2], cfg_mass=self.cfg.mass, density=self.cfg.density)
        inertial = InertialCfg(mass=mass, origin=origin, inertia=_box_inertia(size, mass))
        collision = CollisionGeometryCfg(
            name=f"{self.cfg.name}_col",
            geometry={"type": "box", "size": size},
            origin=origin,  # box mesh frame
        )
        visual = VisualGeometryCfg(
            name=f"{self.cfg.name}_vis",
            geometry={"type": "box", "size": size},
            origin=origin,  # visual frame
        )
        return [collision], [visual], inertial

    def _build_cylinder(self) -> tuple[list[CollisionGeometryCfg], list[VisualGeometryCfg], InertialCfg]:




        mesh = self.cfg.mesh
        radius = float(mesh["radius"])
        length = float(mesh["length"])
        offset = _pose_from_value(mesh.get("offset", mesh.get("origin")))
        center_on_joint = bool(mesh.get("center_on_joint", False))

        origin = _make_geometry_pose(
            offset=offset,
            default_pos=(offset.pos[0], length / 2.0 + offset.pos[1], offset.pos[2]),
            default_rpy=(-math.pi / 2.0, 0.0, 0.0),
            center_on_joint=center_on_joint,
        )
        mass = _estimate_mass(
            volume=math.pi * radius * radius * length,
            cfg_mass=self.cfg.mass,
            density=self.cfg.density,
        )
        inertial = InertialCfg(mass=mass, origin=origin, inertia=_cylinder_inertia(radius, length, mass))
        geometry = {"type": "cylinder", "radius": radius, "length": length}
        collision = CollisionGeometryCfg(name=f"{self.cfg.name}_col", geometry=geometry, origin=origin)
        visual = VisualGeometryCfg(name=f"{self.cfg.name}_vis", geometry=geometry, origin=origin)
        return [collision], [visual], inertial

    def _build_elliptic_cylinder(self) -> tuple[list[CollisionGeometryCfg], list[VisualGeometryCfg], InertialCfg]:

        mesh = self.cfg.mesh
        radius_x = float(mesh["radius_x"])
        radius_z = float(mesh["radius_z"])
        length = float(mesh["length"])
        offset = _pose_from_value(mesh.get("offset", mesh.get("origin")))
        center_on_joint = bool(mesh.get("center_on_joint", False))




        origin = _make_geometry_pose(
            offset=offset,
            default_pos=(offset.pos[0], length / 2.0 + offset.pos[1], offset.pos[2]),
            center_on_joint=center_on_joint,
        )

        volume = math.pi * radius_x * radius_z * length
        mass = _estimate_mass(volume=volume, cfg_mass=self.cfg.mass, density=self.cfg.density)
        inertial = InertialCfg(
            mass=mass,
            origin=origin,
            inertia=_elliptic_cylinder_inertia(radius_x, radius_z, length, mass),
        )
        geometry = {
            "type": "elliptic_cylinder",
            "radius_x": radius_x,
            "radius_z": radius_z,
            "length": length,
        }
        collision = CollisionGeometryCfg(name=f"{self.cfg.name}_col", geometry=geometry, origin=origin)
        visual = VisualGeometryCfg(name=f"{self.cfg.name}_vis", geometry=geometry, origin=origin)
        return [collision], [visual], inertial

    def _build_sphere(self) -> tuple[list[CollisionGeometryCfg], list[VisualGeometryCfg], InertialCfg]:

        mesh = self.cfg.mesh
        radius = float(mesh["radius"])
        offset = _pose_from_value(mesh.get("offset", mesh.get("origin")))
        center_on_joint = bool(mesh.get("center_on_joint", False))
        origin = _make_geometry_pose(
            offset=offset,
            default_pos=(offset.pos[0], radius + offset.pos[1], offset.pos[2]),
            center_on_joint=center_on_joint,
        )
        mass = _estimate_mass(
            volume=4.0 * math.pi * radius**3 / 3.0,
            cfg_mass=self.cfg.mass,
            density=self.cfg.density,
        )
        inertial = InertialCfg(mass=mass, origin=origin, inertia=_sphere_inertia(radius, mass))
        geometry = {"type": "sphere", "radius": radius}
        collision = CollisionGeometryCfg(name=f"{self.cfg.name}_col", geometry=geometry, origin=origin)
        visual = VisualGeometryCfg(name=f"{self.cfg.name}_vis", geometry=geometry, origin=origin)
        return [collision], [visual], inertial


class ComPrimJointBuilder(JointBuilder):
    "Builds a configured hand component from typed geometry and local frames."

    cfg: PrimJointBuilderCfg

    def __init__(self, cfg: PrimJointBuilderCfg):
        super().__init__(cfg)
        self.cfg = cfg

    def build(self) -> JointCfg:
        "Builds the configured geometry component from typed dimensions and local frames."

        mesh_kind = str(self.cfg.mesh.get("type", self.cfg.mesh.get("kind"))).lower()
        if mesh_kind == "cs":
            collisions, visuals, inertial, metadata = self._build_cylinder_sphere_tip()
        elif mesh_kind == "bs":
            collisions, visuals, inertial = self._build_box_sphere_tip()
            metadata = self.cfg.metadata.copy()
        else:
            raise ValueError(f"Unsupported composite primitive mesh type: {mesh_kind}")

        return JointCfg(
            name=self.cfg.name,
            parent=self.cfg.parent,
            child=self.cfg.child,
            joint_type=self.cfg.joint_type,  # fixed / revolute
            axis=self.cfg.axis,
            limit=self.cfg.limit,
            origin=self.cfg.origin,
            inertial=inertial,
            collisions=collisions,
            visuals=visuals,
            is_tip=self.cfg.is_tip,
            metadata=metadata,
        )

    def _build_cylinder_sphere_tip(self) -> tuple[list[CollisionGeometryCfg], list[VisualGeometryCfg], None, dict[str, Any]]:

        mesh = self.cfg.mesh
        radius = float(mesh["radius"])
        length = float(mesh["height"])
        offset = _pose_from_value(mesh.get("offset", mesh.get("origin")))
        mesh_uri = make_procedural_cs_tip_uri(radius=radius, height=length)
        geometry = {"type": "mesh", "file_path": mesh_uri, "scale": (1.0, 1.0, 1.0)}


        collisions = [
            CollisionGeometryCfg(
                name=f"{self.cfg.name}_mesh_col",
                geometry=geometry,
                origin=offset,
            )
        ]
        visuals = [
            VisualGeometryCfg(
                name=f"{self.cfg.name}_mesh_vis",
                geometry=geometry,
                origin=offset,
            )
        ]

        metadata = {
            **self.cfg.metadata.copy(),
            "tip_type": "cs",
            "procedural_tip_type": "cs",
            "procedural_mesh_kind": "cs_tip",
            "procedural_mesh_schema": "flat_base_cylinder_upper_hemisphere_v1",
            "cs_radius": radius,
            "cs_height": length,
            "cs_ratio": length / radius,
        }
        return collisions, visuals, None, metadata

    def _build_box_sphere_tip(self) -> tuple[list[CollisionGeometryCfg], list[VisualGeometryCfg], InertialCfg]:


        mesh = self.cfg.mesh
        radius = float(mesh["radius"])
        height = float(mesh["height"])
        width = float(mesh["width"])
        depth = float(mesh.get("depth", width))
        offset = _pose_from_value(mesh.get("offset", mesh.get("origin")))

        box_origin = PoseCfg(pos=(offset.pos[0], height / 2.0 + offset.pos[1], offset.pos[2]), rpy=offset.rpy)
        sph_origin = PoseCfg(pos=(offset.pos[0], height + offset.pos[1], offset.pos[2]), rpy=offset.rpy)

        box_mass = _estimate_mass(
            volume=width * height * depth,
            cfg_mass=None if self.cfg.mass is None else self.cfg.mass * 0.55,
            density=self.cfg.density,
        )
        sph_mass = _estimate_mass(
            volume=4.0 * math.pi * radius**3 / 3.0,
            cfg_mass=None if self.cfg.mass is None else self.cfg.mass * 0.45,
            density=self.cfg.density,
        )
        total_mass = box_mass + sph_mass
        com_y = (box_mass * box_origin.pos[1] + sph_mass * sph_origin.pos[1]) / total_mass

        inertial = InertialCfg(
            mass=total_mass,
            origin=PoseCfg(pos=(offset.pos[0], com_y, offset.pos[2])),
            inertia=_box_inertia((width, height + 2.0 * radius, depth), total_mass),
        )
        collisions = [
            CollisionGeometryCfg(
                name=f"{self.cfg.name}_body_col",
                geometry={"type": "box", "size": (width, height, depth)},
                origin=box_origin,
            ),
            CollisionGeometryCfg(
                name=f"{self.cfg.name}_cap_col",
                geometry={"type": "sphere", "radius": radius},
                origin=sph_origin,
            ),
        ]
        visuals = [
            VisualGeometryCfg(
                name=f"{self.cfg.name}_body_vis",
                geometry={"type": "box", "size": (width, height, depth)},
                origin=box_origin,
            ),
            VisualGeometryCfg(
                name=f"{self.cfg.name}_cap_vis",
                geometry={"type": "sphere", "radius": radius},
                origin=sph_origin,
            ),
        ]
        return collisions, visuals, inertial


__all__ = ["PrimJointBuilderCfg", "PrimJointBuilder", "ComPrimJointBuilder"]
