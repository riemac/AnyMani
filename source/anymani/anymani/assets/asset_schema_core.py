"Defines hand anatomy, primitive geometry, joint limits, and rigid poses. Lengths use meters and angles use radians."

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, ClassVar, Literal, cast, overload


def _class_to_dict(value: Any) -> Any:

    if is_dataclass(value):
        return {obj_field.name: _class_to_dict(getattr(value, obj_field.name)) for obj_field in fields(value)}
    if isinstance(value, list):
        return [_class_to_dict(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_class_to_dict(item) for item in value)
    if isinstance(value, dict):
        return {key: _class_to_dict(item) for key, item in value.items()}
    return value


def _update_from_dict(obj: Any, data: dict[str, Any]) -> None:

    for key, value in data.items():
        if not hasattr(obj, key):
            raise KeyError(f"Unknown config field: {key}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, Mapping):
            _update_from_dict(current, dict(value))
        else:
            setattr(obj, key, value)
    if hasattr(obj, "__post_init__"):
        obj.__post_init__()


def _validate_missing(obj: Any, prefix: str = "") -> list[str]:

    missing: list[str] = []
    for obj_field in fields(obj):
        value = getattr(obj, obj_field.name)
        key = f"{prefix}.{obj_field.name}" if prefix else obj_field.name
        if value is ...:
            missing.append(key)
        elif is_dataclass(value):
            missing.extend(_validate_missing(value, key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if is_dataclass(item):
                    missing.extend(_validate_missing(item, f"{key}[{index}]"))
    return missing


class AssetCfgBase:
    "Base typed record with explicit replacement and normalization behavior for hand geometry configuration."

    def to_dict(self) -> dict[str, Any]:
        'Serializes the typed object as a dictionary.'

        return _class_to_dict(self)

    def from_dict(self, data: dict[str, Any]) -> None:
        'Updates the typed object from a dictionary.'

        _update_from_dict(self, data)

    def copy(self):
        "Returns an independent typed configuration copy with the same geometry values."

        return deepcopy(self)

    def replace(self, **kwargs):
        "Returns an independent configuration with only the named fields replaced."

        return replace(cast(Any, self), **kwargs)

    def validate(self) -> list[str]:
        "Runs the configured acceptance gates and returns explicit candidate evidence."

        return _validate_missing(self)


Vector2 = tuple[float, float]
"Two authored dimensions, with meaning declared by the consuming geometry field."

Vector3 = tuple[float, float, float]
"Three-value vector; the owning field declares whether it stores meters, radians, or a unit axis."

Vector4 = tuple[float, float, float, float]
"Four-value typed vector used by the owning schema."

Vector6 = tuple[float, float, float, float, float, float]
"Six length, width, and height scaling ranges in the declared order."

JointType = Literal["revolute", "fixed"]
"Fixed or revolute joint type supported by the source schema."

Handedness = Literal["left", "right", "unknown"]
"Declared right or left hand side."

PrimitiveGeometryType = Literal["box", "cylinder", "elliptic_cylinder", "sphere"]
"Supported box, cylinder, elliptic-cylinder, or sphere geometry type."

_FLOAT_TOLERANCE = 1e-12
"Absolute tolerance for comparing floating-point geometry values."


def _sanitize_identifier(name: str, *, field_name: str) -> str:

    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    name = name.strip()
    if name[0].isdigit():
        name = f"a_{name}"
    return name


@overload
def _ensure_tuple(value: Any, *, length: Literal[2], field_name: str) -> Vector2: ...


@overload
def _ensure_tuple(value: Any, *, length: Literal[3], field_name: str) -> Vector3: ...


@overload
def _ensure_tuple(value: Any, *, length: Literal[4], field_name: str) -> Vector4: ...


@overload
def _ensure_tuple(value: Any, *, length: Literal[6], field_name: str) -> Vector6: ...


@overload
def _ensure_tuple(value: Any, *, length: int, field_name: str) -> tuple[float, ...]: ...


def _ensure_tuple(value: Any, *, length: int, field_name: str) -> tuple[float, ...]:

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence with {length} floats, got {value!r}")
    if len(value) != length:
        raise ValueError(f"{field_name} must have length {length}, got {len(value)}")
    return tuple(float(item) for item in value)


def _normalize_axis(axis: Vector3) -> Vector3:

    x, y, z = _ensure_tuple(axis, length=3, field_name="axis")
    norm = math.sqrt(x * x + y * y + z * z)
    if norm <= _FLOAT_TOLERANCE:
        raise ValueError("axis cannot be zero vector")
    return (x / norm, y / norm, z / norm)


def _ensure_list(value: Any, *, field_name: str) -> list[Any]:

    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


@dataclass
class PoseCfg(AssetCfgBase):
    "Rigid pose value with translation in meters and rotation in radians."

    pos: Vector3 = (0.0, 0.0, 0.0)
    "Translation vector in meters."

    rpy: Vector3 = (0.0, 0.0, 0.0)
    "Roll, pitch, and yaw angles in radians."

    def __post_init__(self):
        self.pos = _ensure_tuple(self.pos, length=3, field_name="pos")
        self.rpy = _ensure_tuple(self.rpy, length=3, field_name="rpy")

    @classmethod
    def from_value(cls, value: PoseCfg | Sequence[float] | Mapping[str, Any] | None) -> PoseCfg:
        'Converts the input value to the typed pose.'

        if value is None:
            return cls()
        if isinstance(value, cls):
            return value.copy()
        if isinstance(value, Mapping):
            pos = value.get("pos", value.get("xyz", value.get("position", (0.0, 0.0, 0.0))))
            rpy = value.get("rpy", value.get("rot", value.get("rotation", (0.0, 0.0, 0.0))))
            return cls(pos=pos, rpy=rpy)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) == 3:
                return cls(pos=_ensure_tuple(value, length=3, field_name="pose.pos"))
            if len(value) == 6:
                packed = _ensure_tuple(value, length=6, field_name="pose")
                x, y, z, roll, pitch, yaw = packed
                return cls(pos=(x, y, z), rpy=(roll, pitch, yaw))
        raise TypeError(f"Unsupported pose value: {value!r}")

    @property
    def packed(self) -> Vector6:
        "Returns dimensions in the stable serialized vector order."

        return (*self.pos, *self.rpy)


@dataclass
class MaterialCfg(AssetCfgBase):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    name: str | None = None
    "Stable semantic identifier preserved in the output metadata."

    rgba: Vector4 = (0.7, 0.7, 0.7, 1.0)
    "Visual red, green, blue, and alpha values in the normalized interval [0, 1]."

    def __post_init__(self):
        self.rgba = _ensure_tuple(self.rgba, length=4, field_name="rgba")
        if self.name is not None:
            self.name = _sanitize_identifier(self.name, field_name="material.name")


@dataclass
class GeometryCfg(AssetCfgBase):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    geometry_type: ClassVar[str] = "geometry"
    "Primitive kind used to interpret the size or mesh fields."

    @property
    def kind(self) -> str:
        "Returns the primitive or mesh geometry kind carried by this record."

        return self.geometry_type

    @property
    def is_primitive(self) -> bool:
        "Reports whether the geometry uses a supported analytic primitive rather than a mesh."

        return self.geometry_type in {"box", "cylinder", "elliptic_cylinder", "sphere"}


@dataclass
class BoxGeometryCfg(GeometryCfg):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    geometry_type: ClassVar[str] = "box"
    size: Vector3
    "Primitive side lengths in meters."

    def __post_init__(self):
        self.size = _ensure_tuple(self.size, length=3, field_name="box.size")
        if any(edge <= 0.0 for edge in self.size):
            raise ValueError(f"box.size must be positive, got {self.size}")


@dataclass
class CylinderGeometryCfg(GeometryCfg):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    geometry_type: ClassVar[str] = "cylinder"
    radius: float
    "Primitive radius in meters."

    length: float
    "Primary link span along the declared anatomical or joint-local axis, in meters."

    def __post_init__(self):
        self.radius = float(self.radius)
        self.length = float(self.length)
        if self.radius <= 0.0 or self.length <= 0.0:
            raise ValueError(f"cylinder radius/length must be positive, got {(self.radius, self.length)}")


@dataclass
class EllipticCylinderGeometryCfg(GeometryCfg):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    geometry_type: ClassVar[str] = "elliptic_cylinder"
    radius_x: float
    "Elliptic-cylinder radius along the local x axis, in meters."

    radius_z: float
    "Elliptic-cylinder radius along the local z axis, in meters."

    length: float
    "Primary link span along the declared anatomical or joint-local axis, in meters."

    def __post_init__(self):
        self.radius_x = float(self.radius_x)
        self.radius_z = float(self.radius_z)
        self.length = float(self.length)
        if self.radius_x <= 0.0 or self.radius_z <= 0.0 or self.length <= 0.0:
            raise ValueError(
                f"elliptic_cylinder radii/length must be positive, got {(self.radius_x, self.radius_z, self.length)}"
            )

    @property
    def equivalent_cylinder_radius(self) -> float:
        "Computes the radius of a circular cylinder with the same cross-sectional area as the source shape."

        return math.sqrt(self.radius_x * self.radius_z)


@dataclass
class SphereGeometryCfg(GeometryCfg):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    geometry_type: ClassVar[str] = "sphere"
    radius: float
    "Primitive radius in meters."

    def __post_init__(self):
        self.radius = float(self.radius)
        if self.radius <= 0.0:
            raise ValueError(f"sphere.radius must be positive, got {self.radius}")


@dataclass
class MeshGeometryCfg(GeometryCfg):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    geometry_type: ClassVar[str] = "mesh"
    file_path: str
    "Path to the mesh or artifact used by this geometry component."

    scale: Vector3 = (1.0, 1.0, 1.0)
    "Dimensionless scale applied after source mesh unit conversion."

    reflected_about_yz: bool = False
    "Whether this geometry is certified as mirrored across the palm YZ plane."

    def __post_init__(self):
        if not isinstance(self.file_path, str) or not self.file_path.strip():
            raise ValueError("mesh.file_path must be a non-empty string")
        self.file_path = self.file_path.strip()
        self.scale = _ensure_tuple(self.scale, length=3, field_name="mesh.scale")
        if any(scale <= 0.0 for scale in self.scale):
            raise ValueError(f"mesh.scale must be positive, got {self.scale}")
        self.reflected_about_yz = bool(self.reflected_about_yz)

    @property
    def suffix(self) -> str:
        return Path(self.file_path).suffix.lower()


GeometryValue = GeometryCfg | str | Mapping[str, Any]
"Supported primitive or mesh geometry configuration."


def make_geometry_cfg(value: GeometryValue) -> GeometryCfg:
    'Converts a primitive or mesh specification to a geometry config.'

    if isinstance(value, GeometryCfg):
        return value.copy()
    if isinstance(value, str):
        return MeshGeometryCfg(file_path=value)
    if not isinstance(value, Mapping):
        raise TypeError(f"Unsupported geometry value: {value!r}")

    geometry_type = value.get("type", value.get("kind"))
    if geometry_type is None:
        raise KeyError("Geometry dict must contain 'type' or 'kind'")

    geometry_type = str(geometry_type).lower()
    if geometry_type == "box":
        return BoxGeometryCfg(size=value["size"])
    if geometry_type == "cylinder":
        return CylinderGeometryCfg(radius=value["radius"], length=value["length"])
    if geometry_type == "elliptic_cylinder":
        return EllipticCylinderGeometryCfg(
            radius_x=value["radius_x"],
            radius_z=value["radius_z"],
            length=value["length"],
        )
    if geometry_type == "sphere":
        return SphereGeometryCfg(radius=value["radius"])
    if geometry_type == "mesh":
        file_path = value.get("file_path", value.get("path", value.get("mesh")))
        return MeshGeometryCfg(
            file_path=file_path,
            scale=value.get("scale", (1.0, 1.0, 1.0)),
            reflected_about_yz=value.get("reflected_about_yz", False),
        )

    raise ValueError(f"Unsupported geometry type: {geometry_type}")


@dataclass
class GeometryElementCfg(AssetCfgBase):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    geometry: GeometryCfg
    "Typed collision or visual geometry attached to a link."

    name: str | None = None
    "Stable semantic identifier preserved in the output metadata."

    origin: PoseCfg | Sequence[float] | Mapping[str, Any] | None = field(default_factory=PoseCfg)
    "Rigid transform with translation in meters and rotation in radians."

    material: MaterialCfg | Mapping[str, Any] | None = None
    "Visual material attached to a rendered geometry element; it does not define collision behavior."

    def __post_init__(self):
        if self.name is not None:
            self.name = _sanitize_identifier(self.name, field_name="geometry_element.name")
        self.geometry = make_geometry_cfg(self.geometry)
        self.origin = PoseCfg.from_value(self.origin)
        if self.material is not None and not isinstance(self.material, MaterialCfg):
            if not isinstance(self.material, Mapping):
                raise TypeError(f"material must be MaterialCfg or mapping, got {self.material!r}")
            self.material = MaterialCfg(**self.material)


@dataclass
class CollisionGeometryCfg(GeometryElementCfg):
    "Declares unit-aware inputs and validation policy for the asset pipeline."


@dataclass
class VisualGeometryCfg(GeometryElementCfg):
    "Declares unit-aware inputs and validation policy for the asset pipeline."


@dataclass
class InertiaTensorCfg(AssetCfgBase):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    ixx: float
    "Diagonal inertia component about the local x axis, in kg m^2."

    iyy: float
    "Diagonal inertia component about the local y axis, in kg m^2."

    izz: float
    "Diagonal inertia component about the local z axis, in kg m^2."

    ixy: float = 0.0
    "Off-diagonal inertia component in the local frame, in kg m^2."

    ixz: float = 0.0
    "Off-diagonal inertia component in the local frame, in kg m^2."

    iyz: float = 0.0
    "Off-diagonal inertia component in the local frame, in kg m^2."

    def __post_init__(self):
        self.ixx = float(self.ixx)
        self.iyy = float(self.iyy)
        self.izz = float(self.izz)
        self.ixy = float(self.ixy)
        self.ixz = float(self.ixz)
        self.iyz = float(self.iyz)
        if self.ixx <= 0.0 or self.iyy <= 0.0 or self.izz <= 0.0:
            raise ValueError("Inertia diagonal entries must be positive")


@dataclass
class InertialCfg(AssetCfgBase):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    mass: float
    "Rigid-body mass in kilograms."

    inertia: InertiaTensorCfg | Mapping[str, Any]
    "Symmetric rigid-body inertia tensor about the center of mass, in kg m^2."

    origin: PoseCfg | Sequence[float] | Mapping[str, Any] | None = field(default_factory=PoseCfg)
    "Rigid transform with translation in meters and rotation in radians."

    inertia_padding: float = 0.0
    "Diagonal regularization added to inertia eigenvalues, in kg m^2."

    def __post_init__(self):
        self.mass = float(self.mass)
        if self.mass <= 0.0:
            raise ValueError(f"mass must be positive, got {self.mass}")
        self.origin = PoseCfg.from_value(self.origin)
        if not isinstance(self.inertia, InertiaTensorCfg):
            if not isinstance(self.inertia, Mapping):
                raise TypeError(f"inertia must be InertiaTensorCfg or mapping, got {self.inertia!r}")
            self.inertia = InertiaTensorCfg(**self.inertia)
        self.inertia_padding = float(self.inertia_padding)
        if self.inertia_padding < 0.0:
            raise ValueError("inertia_padding must be >= 0")
        if self.inertia_padding > 0.0:

            self.inertia = InertiaTensorCfg(
                ixx=self.inertia.ixx + self.inertia_padding,
                iyy=self.inertia.iyy + self.inertia_padding,
                izz=self.inertia.izz + self.inertia_padding,
                ixy=self.inertia.ixy,
                ixz=self.inertia.ixz,
                iyz=self.inertia.iyz,
            )

    @classmethod
    def from_box(
        cls,
        size: Vector3,
        density: float,
        *,
        origin: PoseCfg | Sequence[float] | Mapping[str, Any] | None = None,
        min_mass: float = 1e-4,
        inertia_padding: float = 1e-8,
    ) -> InertialCfg:
        'Computes box inertia from dimensions and density.'

        sx, sy, sz = _ensure_tuple(size, length=3, field_name="size")
        density = float(density)
        if density <= 0.0:
            raise ValueError("density must be positive")

        mass = max(density * sx * sy * sz, min_mass)

        ixx = mass * (sy * sy + sz * sz) / 12.0
        iyy = mass * (sx * sx + sz * sz) / 12.0
        izz = mass * (sx * sx + sy * sy) / 12.0
        return cls(
            mass=mass,
            origin=origin,
            inertia=InertiaTensorCfg(ixx=ixx, iyy=iyy, izz=izz),
            inertia_padding=inertia_padding,
        )

    @classmethod
    def from_cylinder(
        cls,
        radius: float,
        length: float,
        density: float,
        *,
        origin: PoseCfg | Sequence[float] | Mapping[str, Any] | None = None,
        principal_axis: Literal["x", "y", "z"] = "z",
        min_mass: float = 1e-4,
        inertia_padding: float = 1e-8,
    ) -> InertialCfg:
        'Computes cylinder inertia from radius, length, and density.'

        radius = float(radius)
        length = float(length)
        density = float(density)
        if radius <= 0.0 or length <= 0.0 or density <= 0.0:
            raise ValueError("radius, length and density must be positive")

        volume = math.pi * radius * radius * length
        mass = max(density * volume, min_mass)

        i_parallel = 0.5 * mass * radius * radius

        i_perp = mass * (3.0 * radius * radius + length * length) / 12.0

        if principal_axis == "x":
            ixx, iyy, izz = i_parallel, i_perp, i_perp
        elif principal_axis == "y":
            ixx, iyy, izz = i_perp, i_parallel, i_perp
        else:
            ixx, iyy, izz = i_perp, i_perp, i_parallel
        return cls(
            mass=mass,
            origin=origin,
            inertia=InertiaTensorCfg(ixx=ixx, iyy=iyy, izz=izz),
            inertia_padding=inertia_padding,
        )

    @classmethod
    def from_elliptic_cylinder(
        cls,
        radius_x: float,
        radius_z: float,
        length: float,
        density: float,
        *,
        origin: PoseCfg | Sequence[float] | Mapping[str, Any] | None = None,
        principal_axis: Literal["x", "y", "z"] = "z",
        min_mass: float = 1e-4,
        inertia_padding: float = 1e-8,
    ) -> InertialCfg:
        'Computes elliptic-cylinder inertia from its dimensions and density.'

        radius_x = float(radius_x)
        radius_z = float(radius_z)
        length = float(length)
        density = float(density)
        if radius_x <= 0.0 or radius_z <= 0.0 or length <= 0.0 or density <= 0.0:
            raise ValueError("radius_x, radius_z, length and density must be positive")

        volume = math.pi * radius_x * radius_z * length
        mass = max(density * volume, min_mass)




        i_parallel = 0.25 * mass * (radius_x * radius_x + radius_z * radius_z)
        i_perp_x = mass * (3.0 * radius_z * radius_z + length * length) / 12.0
        i_perp_z = mass * (3.0 * radius_x * radius_x + length * length) / 12.0

        if principal_axis == "x":
            ixx = i_parallel
            iyy = i_perp_x
            izz = i_perp_z
        elif principal_axis == "y":
            ixx = i_perp_x
            iyy = i_parallel
            izz = i_perp_z
        else:
            ixx = i_perp_x
            iyy = i_perp_z
            izz = i_parallel

        return cls(
            mass=mass,
            origin=origin,
            inertia=InertiaTensorCfg(ixx=ixx, iyy=iyy, izz=izz),
            inertia_padding=inertia_padding,
        )

    @classmethod
    def from_sphere(
        cls,
        radius: float,
        density: float,
        *,
        origin: PoseCfg | Sequence[float] | Mapping[str, Any] | None = None,
        min_mass: float = 1e-4,
        inertia_padding: float = 1e-8,
    ) -> InertialCfg:
        'Computes sphere inertia from radius and density.'

        radius = float(radius)
        density = float(density)
        if radius <= 0.0 or density <= 0.0:
            raise ValueError("radius and density must be positive")

        volume = 4.0 / 3.0 * math.pi * radius**3
        mass = max(density * volume, min_mass)

        diagonal = 0.4 * mass * radius * radius
        return cls(
            mass=mass,
            origin=origin,
            inertia=InertiaTensorCfg(ixx=diagonal, iyy=diagonal, izz=diagonal),
            inertia_padding=inertia_padding,
        )


@dataclass
class JointLimitCfg(AssetCfgBase):
    "Legal revolute-joint interval and optional effort and speed limits."

    lower: float
    "Lower legal or sampling bound in the units of the associated field."

    upper: float
    "Upper legal or sampling bound in the units of the associated field."

    effort: float | None = None
    "Joint effort limit in newton meters."

    velocity: float | None = None
    "Joint velocity limit in radians per second."

    def __post_init__(self):
        self.lower = float(self.lower)
        self.upper = float(self.upper)
        if self.upper < self.lower:
            raise ValueError(f"upper limit must be >= lower limit, got {(self.lower, self.upper)}")
        if self.effort is not None:
            self.effort = float(self.effort)
        if self.velocity is not None:
            self.velocity = float(self.velocity)


@dataclass
class JointPropertiesCfg(AssetCfgBase):
    """Stores joint properties copied from the source family profile.

    The exporter writes LEAP friction with the LEAP joint_properties tag. It does not also emit a standard dynamics tag, avoiding duplicate friction interpretation.
    """

    friction: float | None = None
    "Joint friction coefficient copied from the reviewed family profile."

    def __post_init__(self):
        if self.friction is not None:
            self.friction = float(self.friction)


@dataclass
class MimicCfg(AssetCfgBase):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    joint: str
    "Typed source joint or joint-level physical profile entry."

    multiplier: float = 1.0
    "Dimensionless mimic-joint multiplier applied to the source joint coordinate."

    offset: float = 0.0
    "Translation in meters or joint-coordinate offset in radians as declared by the owning field."

    def __post_init__(self):
        self.joint = _sanitize_identifier(self.joint, field_name="mimic.joint")
        self.multiplier = float(self.multiplier)
        self.offset = float(self.offset)


def _make_collision_cfg(value: Any) -> CollisionGeometryCfg:

    if isinstance(value, CollisionGeometryCfg):
        return value.copy()
    if isinstance(value, GeometryCfg | str):
        return CollisionGeometryCfg(geometry=make_geometry_cfg(value))
    if not isinstance(value, Mapping):
        raise TypeError(f"Unsupported collision geometry value: {value!r}")
    if "geometry" in value:
        return CollisionGeometryCfg(**dict(value))
    geometry_keys = {"type", "kind", "size", "radius", "length", "file_path", "path", "mesh", "scale"}
    if geometry_keys.intersection(value.keys()):
        element_kwargs = {key: value[key] for key in ("name", "origin", "material") if key in value}
        element_kwargs["geometry"] = value
        return CollisionGeometryCfg(**element_kwargs)
    raise TypeError(f"Unsupported collision geometry mapping: {value!r}")


def _make_visual_cfg(value: Any) -> VisualGeometryCfg:

    if isinstance(value, VisualGeometryCfg):
        return value.copy()
    if isinstance(value, GeometryCfg | str):
        return VisualGeometryCfg(geometry=make_geometry_cfg(value))
    if not isinstance(value, Mapping):
        raise TypeError(f"Unsupported visual geometry value: {value!r}")
    if "geometry" in value:
        return VisualGeometryCfg(**dict(value))
    geometry_keys = {"type", "kind", "size", "radius", "length", "file_path", "path", "mesh", "scale"}
    if geometry_keys.intersection(value.keys()):
        element_kwargs = {key: value[key] for key in ("name", "origin", "material") if key in value}
        element_kwargs["geometry"] = value
        return VisualGeometryCfg(**element_kwargs)
    raise TypeError(f"Unsupported visual geometry mapping: {value!r}")


@dataclass
class WristJointSpec(AssetCfgBase):
    "Optional wrist articulation stored before the palm body in a complete hand configuration."

    axis: Vector3
    "Unit direction expressed in the owning local frame."

    position: Vector3 = (0.0, 0.0, 0.0)
    "Position in the owning frame, in meters."

    limits: JointLimitCfg | None = None
    "Lower and upper joint-coordinate bounds in radians."

    def __post_init__(self):
        self.axis = _normalize_axis(self.axis)
        self.position = _ensure_tuple(self.position, length=3, field_name="wrist_joint.position")


__all__ = [
    "AssetCfgBase",
    "Vector2",
    "Vector3",
    "Vector4",
    "Vector6",
    "JointType",
    "Handedness",
    "PrimitiveGeometryType",
    "PoseCfg",
    "MaterialCfg",
    "GeometryCfg",
    "BoxGeometryCfg",
    "CylinderGeometryCfg",
    "EllipticCylinderGeometryCfg",
    "SphereGeometryCfg",
    "MeshGeometryCfg",
    "GeometryValue",
    "GeometryElementCfg",
    "CollisionGeometryCfg",
    "VisualGeometryCfg",
    "InertiaTensorCfg",
    "InertialCfg",
    "JointLimitCfg",
    "JointPropertiesCfg",
    "MimicCfg",
    "make_geometry_cfg",
    "_FLOAT_TOLERANCE",
    "_sanitize_identifier",
    "_ensure_tuple",
    "_normalize_axis",
    "_ensure_list",
    "_make_collision_cfg",
    "_make_visual_cfg",
    "WristJointSpec",
]
