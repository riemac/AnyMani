"""Primitive joint builders for joint-centric hand asset generation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Literal, Mapping, Sequence

from ..asset_base import JointCfg
from ..asset_builders import JointBuilder, JointBuilderCfg
from ..asset_schema_core import (
    CollisionGeometryCfg,
    InertialCfg,
    JointLimitCfg,
    PoseCfg,
    Vector3,
    Vector6,
    VisualGeometryCfg,
    _ensure_tuple,
)


_DEFAULT_DENSITY = 650.0
"""A light-weight default density used to synthesize inertial values."""


def _pose_from_value(value: PoseCfg | Sequence[float] | Mapping[str, Any] | None) -> PoseCfg:
    r"""Normalize a pose-like input into ``PoseCfg``."""

    return PoseCfg.from_value(value)


def _add_rpy(lhs: Vector3, rhs: Vector3) -> Vector3:
    r"""Compose two tiny Euler adjustments with direct component-wise addition.

    The primitive builders only need simple axis-aligned rotations, so a full SO(3)
    composition utility would be unnecessary complexity at this stage.
    """

    return (lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2])


def _make_geometry_pose(
    *,
    offset: PoseCfg,
    default_pos: Vector3,
    default_rpy: Vector3 = (0.0, 0.0, 0.0),
    center_on_joint: bool = False,
) -> PoseCfg:
    r"""Construct a primitive geometry pose under the current joint convention.

    For regular links, when the offset is zero, the geometry grows from the joint
    frame's ``x-z`` plane toward ``+y``. For ``CMC1``-like cases, callers may set
    ``center_on_joint=True`` to keep the mesh frame coincident with the joint frame.
    """

    if center_on_joint:
        base = offset.pos
    else:
        base = default_pos
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


def _sphere_inertia(radius: float, mass: float) -> dict[str, float]:
    moment = 2.0 * mass * radius * radius / 5.0
    return {"ixx": moment, "iyy": moment, "izz": moment}


def _estimate_mass(*, volume: float, cfg_mass: float | None, density: float) -> float:
    r"""Estimate mass when the cfg does not pin a specific value."""

    return float(cfg_mass) if cfg_mass is not None else max(volume * density, 1e-6)


@dataclass
class PrimJointBuilderCfg(JointBuilderCfg):
    r"""Primitive joint builder configuration.

    The cfg intentionally exposes the joint-centric fields that later builder stages
    need to wire into a ``JointCfg`` directly, so that finger builders can delegate
    geometry construction to this layer without reimplementing primitive handling.
    """

    class_type: type["PrimJointBuilder"] | type["ComPrimJointBuilder"] | None = None
    """Associated runtime builder type."""

    name: str = "joint"
    """Joint name used in the emitted ``JointCfg``."""

    parent: str = "palm"
    """Parent link name in the emitted ``JointCfg``."""

    child: str | None = None
    """Optional explicit child link name."""

    mesh: dict[str, Any] = field(default_factory=dict)
    """Primitive mesh recipe.

    Supported ``type`` values:
    - ``box``: expects ``length``/``width``/``height`` or ``size``
    - ``cylinder``: expects ``length`` and ``radius``
    - ``sphere``: expects ``radius``
    - ``cs``: cylinder + sphere composite tip
    - ``bs``: box + sphere composite tip
    """

    joint_type: Literal["revolute", "fixed"] = "revolute"
    """Joint type emitted into the resulting ``JointCfg``."""

    origin: PoseCfg | Sequence[float] | Mapping[str, Any] | None = field(default_factory=PoseCfg)
    """Joint frame pose relative to the parent link."""

    axis: Vector3 = (0.0, 0.0, 1.0)
    """Joint axis used for ``revolute`` joints."""

    limit: JointLimitCfg | Sequence[float] | Mapping[str, Any] | None = (-math.pi, math.pi)
    """Optional joint limit information."""

    density: float = _DEFAULT_DENSITY
    """Fallback density used to derive inertial mass from primitive volume."""

    mass: float | None = None
    """Optional explicit mass override for the synthesized child link."""

    is_tip: bool = False
    """Whether the emitted joint/link should be marked as fingertip related."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata forwarded to the resulting ``JointCfg``."""

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
    r"""Builder that turns one primitive recipe into one ``JointCfg``."""

    cfg: PrimJointBuilderCfg

    def __init__(self, cfg: PrimJointBuilderCfg):
        super().__init__(cfg)
        self.cfg = cfg

    def build(self) -> JointCfg:
        r"""Build a single-primitive ``JointCfg``.

        The emitted collision and visual geometry share the same primitive recipe so
        that later URDF export remains deterministic and easy to inspect.
        """

        geom_kind = str(self.cfg.mesh.get("type", self.cfg.mesh.get("kind", "box"))).lower()
        if geom_kind == "box":
            collisions, visuals, inertial = self._build_box()
        elif geom_kind == "cylinder":
            collisions, visuals, inertial = self._build_cylinder()
        elif geom_kind == "sphere":
            collisions, visuals, inertial = self._build_sphere()
        else:
            raise ValueError(f"Unsupported primitive joint mesh type: {geom_kind}")

        return JointCfg(
            name=self.cfg.name,
            parent=self.cfg.parent,
            child=self.cfg.child,
            joint_type=self.cfg.joint_type,
            axis=self.cfg.axis,
            limit=self.cfg.limit,
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
            origin=origin,
        )
        visual = VisualGeometryCfg(
            name=f"{self.cfg.name}_vis",
            geometry={"type": "box", "size": size},
            origin=origin,
        )
        return [collision], [visual], inertial

    def _build_cylinder(self) -> tuple[list[CollisionGeometryCfg], list[VisualGeometryCfg], InertialCfg]:
        mesh = self.cfg.mesh
        radius = float(mesh["radius"])
        length = float(mesh["length"])
        offset = _pose_from_value(mesh.get("offset", mesh.get("origin")))
        center_on_joint = bool(mesh.get("center_on_joint", False))
        # URDF cylinders are aligned with +z by default, so we rotate them into +y.
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
    r"""Builder for composite primitive tips such as cylinder+sphere."""

    cfg: PrimJointBuilderCfg

    def __init__(self, cfg: PrimJointBuilderCfg):
        super().__init__(cfg)
        self.cfg = cfg

    def build(self) -> JointCfg:
        r"""Build a composite-tip ``JointCfg``."""

        mesh_kind = str(self.cfg.mesh.get("type", self.cfg.mesh.get("kind"))).lower()
        if mesh_kind == "cs":
            collisions, visuals, inertial = self._build_cylinder_sphere_tip()
        elif mesh_kind == "bs":
            collisions, visuals, inertial = self._build_box_sphere_tip()
        else:
            raise ValueError(f"Unsupported composite primitive mesh type: {mesh_kind}")

        return JointCfg(
            name=self.cfg.name,
            parent=self.cfg.parent,
            child=self.cfg.child,
            joint_type=self.cfg.joint_type,
            axis=self.cfg.axis,
            limit=self.cfg.limit,
            origin=self.cfg.origin,
            inertial=inertial,
            collisions=collisions,
            visuals=visuals,
            is_tip=self.cfg.is_tip,
            metadata=self.cfg.metadata.copy(),
        )

    def _build_cylinder_sphere_tip(self) -> tuple[list[CollisionGeometryCfg], list[VisualGeometryCfg], InertialCfg]:
        mesh = self.cfg.mesh
        radius = float(mesh["radius"])
        length = float(mesh["height"])
        offset = _pose_from_value(mesh.get("offset", mesh.get("origin")))

        cyl_origin = PoseCfg(
            pos=(offset.pos[0], length / 2.0 + offset.pos[1], offset.pos[2]),
            rpy=_add_rpy((-math.pi / 2.0, 0.0, 0.0), offset.rpy),
        )
        sph_origin = PoseCfg(pos=(offset.pos[0], length + offset.pos[1], offset.pos[2]), rpy=offset.rpy)

        cyl_mass = _estimate_mass(
            volume=math.pi * radius * radius * length,
            cfg_mass=None if self.cfg.mass is None else self.cfg.mass * 0.55,
            density=self.cfg.density,
        )
        sph_mass = _estimate_mass(
            volume=4.0 * math.pi * radius**3 / 3.0,
            cfg_mass=None if self.cfg.mass is None else self.cfg.mass * 0.45,
            density=self.cfg.density,
        )
        total_mass = cyl_mass + sph_mass
        com_y = (cyl_mass * cyl_origin.pos[1] + sph_mass * sph_origin.pos[1]) / total_mass

        equivalent_length = length + 2.0 * radius
        inertial = InertialCfg(
            mass=total_mass,
            origin=PoseCfg(pos=(offset.pos[0], com_y, offset.pos[2])),
            inertia=_cylinder_inertia(radius, equivalent_length, total_mass),
        )
        collisions = [
            CollisionGeometryCfg(
                name=f"{self.cfg.name}_body_col",
                geometry={"type": "cylinder", "radius": radius, "length": length},
                origin=cyl_origin,
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
                geometry={"type": "cylinder", "radius": radius, "length": length},
                origin=cyl_origin,
            ),
            VisualGeometryCfg(
                name=f"{self.cfg.name}_cap_vis",
                geometry={"type": "sphere", "radius": radius},
                origin=sph_origin,
            ),
        ]
        return collisions, visuals, inertial

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
