"""Core schema and helper utilities for hand-asset declaration.

This module contains the lower-level schema objects that are shared by the
embodiment-level asset description:

- generic dataclass helpers
- pose/material primitives
- geometry schema
- inertial schema
- low-level normalization helpers

It deliberately does not define `JointCfg` / `FingerCfg` / `PalmCfg` /
`HandCfg`. Those live in `asset_schema_embodiment.py`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass, replace
import math
from pathlib import Path
from typing import Any, ClassVar, Literal, cast, overload


def _class_to_dict(value: Any) -> Any:
    r"""Recursively convert dataclass-based asset configs into Python containers.

    Args:
        value (Any): The object to convert.

    Returns:
        Any: A recursively converted Python container.
    """

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
    r"""Update dataclass fields in-place and rerun normalization.

    Args:
        obj (Any): The dataclass instance to update.
        data (dict[str, Any]): The new field values.

    Raises:
        KeyError: If `data` contains unknown fields.
    """

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
    r"""Collect all unresolved required-field paths.

    Args:
        obj (Any): The object to inspect.
        prefix (str): Current recursive prefix.

    Returns:
        list[str]: All missing-field paths.
    """

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
    r"""Shared helper mixin for asset declaration dataclasses."""

    def to_dict(self) -> dict[str, Any]:
        r"""Serialize the config into native Python containers.

        Returns:
            dict[str, Any]: Recursive dictionary representation.
        """

        return _class_to_dict(self)

    def from_dict(self, data: dict[str, Any]) -> None:
        r"""Update the config in-place from a dictionary.

        Args:
            data (dict[str, Any]): Input mapping.
        """

        _update_from_dict(self, data)

    def copy(self):
        r"""Create a deep copy.

        Returns:
            Any: Deep-copied instance.
        """

        return deepcopy(self)

    def replace(self, **kwargs):
        r"""Return a new config with selected fields replaced.

        Args:
            **kwargs: Replacement fields.

        Returns:
            Any: Replaced copy.
        """

        return replace(cast(Any, self), **kwargs)

    def validate(self) -> list[str]:
        r"""Return unresolved required-field paths.

        Returns:
            list[str]: Missing required-field paths.
        """

        return _validate_missing(self)


Vector2 = tuple[float, float]
"""Two-dimensional float tuple."""

Vector3 = tuple[float, float, float]
"""Three-dimensional float tuple."""

Vector4 = tuple[float, float, float, float]
"""Four-dimensional float tuple."""

Vector6 = tuple[float, float, float, float, float, float]
"""Six-dimensional float tuple."""

JointType = Literal["revolute", "fixed"]
"""Supported URDF joint types in the current project scope."""

Handedness = Literal["left", "right", "unknown"]
"""Handedness tag; `unknown` is reserved for non-typical or undecided embodiments."""

PrimitiveGeometryType = Literal["box", "cylinder", "sphere"]
"""Supported primitive geometry kinds."""

_FLOAT_TOLERANCE = 1e-12
"""Unified near-zero tolerance for geometry and axis checks."""


def _sanitize_identifier(name: str, *, field_name: str) -> str:
    r"""Normalize string identifiers used in schema objects.

    Args:
        name (str): Raw identifier.
        field_name (str): Logical field name for error messages.

    Returns:
        str: Normalized identifier.

    Raises:
        ValueError: If the identifier is empty.
    """

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
    r"""Convert sequence-like input into a fixed-length float tuple.

    Args:
        value (Any): Input object.
        length (int): Required tuple length.
        field_name (str): Field name used in diagnostics.

    Returns:
        tuple[float, ...]: Float tuple of fixed length.

    Raises:
        TypeError: If the input is not a valid sequence.
        ValueError: If the input length mismatches the requested length.
    """

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence with {length} floats, got {value!r}")
    if len(value) != length:
        raise ValueError(f"{field_name} must have length {length}, got {len(value)}")
    return tuple(float(item) for item in value)


def _normalize_axis(axis: Vector3) -> Vector3:
    r"""Normalize a joint axis into a unit vector.

    Args:
        axis (Vector3): Raw axis vector.

    Returns:
        Vector3: Unit axis.

    Raises:
        ValueError: If the vector norm is zero.
    """

    x, y, z = _ensure_tuple(axis, length=3, field_name="axis")
    norm = math.sqrt(x * x + y * y + z * z)
    if norm <= _FLOAT_TOLERANCE:
        raise ValueError("axis cannot be zero vector")
    return (x / norm, y / norm, z / norm)


def _ensure_list(value: Any, *, field_name: str) -> list[Any]:
    r"""Normalize single object / tuple input into a list.

    Args:
        value (Any): Object, tuple, list or `None`.
        field_name (str): Reserved for future diagnostics.

    Returns:
        list[Any]: Normalized list.
    """

    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


@dataclass
class PoseCfg(AssetCfgBase):
    r"""Local pose represented by `pos` + `rpy`.

    This mirrors the URDF `<origin xyz="" rpy="">` semantics directly.
    """

    pos: Vector3 = (0.0, 0.0, 0.0)
    """Local translation $(x, y, z)$."""

    rpy: Vector3 = (0.0, 0.0, 0.0)
    """Local Euler angles $(roll, pitch, yaw)$."""

    def __post_init__(self):
        self.pos = _ensure_tuple(self.pos, length=3, field_name="pos")
        self.rpy = _ensure_tuple(self.rpy, length=3, field_name="rpy")

    @classmethod
    def from_value(cls, value: PoseCfg | Sequence[float] | Mapping[str, Any] | None) -> PoseCfg:
        r"""Construct a `PoseCfg` from common input forms.

        Args:
            value (PoseCfg | Sequence[float] | Mapping[str, Any] | None): Input pose.

        Returns:
            PoseCfg: Normalized pose.

        Raises:
            TypeError: If the input form is unsupported.
        """

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
        r"""Pack translation and Euler angles into a 6D tuple.

        Returns:
            Vector6: `(*pos, *rpy)`.
        """

        return (*self.pos, *self.rpy)


@dataclass
class MaterialCfg(AssetCfgBase):
    r"""Optional material/color descriptor."""

    name: str | None = None
    """Material name, mainly for visual/recolored export."""

    rgba: Vector4 = (0.7, 0.7, 0.7, 1.0)
    """RGBA color and alpha."""

    def __post_init__(self):
        self.rgba = _ensure_tuple(self.rgba, length=4, field_name="rgba")
        if self.name is not None:
            self.name = _sanitize_identifier(self.name, field_name="material.name")


@dataclass
class GeometryCfg(AssetCfgBase):
    r"""Base class for geometry description.

    Geometry and geometry instances are deliberately split:
    `GeometryCfg` describes shape parameters, while `GeometryElementCfg`
    adds placement and optional material.
    """

    geometry_type: ClassVar[str] = "geometry"
    """Geometry type tag for derived-class dispatch."""

    @property
    def kind(self) -> str:
        return self.geometry_type

    @property
    def is_primitive(self) -> bool:
        return self.geometry_type in {"box", "cylinder", "sphere"}


@dataclass
class BoxGeometryCfg(GeometryCfg):
    geometry_type: ClassVar[str] = "box"
    size: Vector3
    """Box side lengths $(s_x, s_y, s_z)$."""

    def __post_init__(self):
        self.size = _ensure_tuple(self.size, length=3, field_name="box.size")
        if any(edge <= 0.0 for edge in self.size):
            raise ValueError(f"box.size must be positive, got {self.size}")


@dataclass
class CylinderGeometryCfg(GeometryCfg):
    geometry_type: ClassVar[str] = "cylinder"
    radius: float
    """Cylinder radius $r$."""

    length: float
    """Cylinder length $l$."""

    def __post_init__(self):
        self.radius = float(self.radius)
        self.length = float(self.length)
        if self.radius <= 0.0 or self.length <= 0.0:
            raise ValueError(f"cylinder radius/length must be positive, got {(self.radius, self.length)}")


@dataclass
class SphereGeometryCfg(GeometryCfg):
    geometry_type: ClassVar[str] = "sphere"
    radius: float
    """Sphere radius $r$."""

    def __post_init__(self):
        self.radius = float(self.radius)
        if self.radius <= 0.0:
            raise ValueError(f"sphere.radius must be positive, got {self.radius}")


@dataclass
class MeshGeometryCfg(GeometryCfg):
    geometry_type: ClassVar[str] = "mesh"
    file_path: str
    """Mesh file path; exporter decides relative vs absolute interpretation."""

    scale: Vector3 = (1.0, 1.0, 1.0)
    """Mesh local scale $(s_x, s_y, s_z)$."""

    def __post_init__(self):
        if not isinstance(self.file_path, str) or not self.file_path.strip():
            raise ValueError("mesh.file_path must be a non-empty string")
        self.file_path = self.file_path.strip()
        self.scale = _ensure_tuple(self.scale, length=3, field_name="mesh.scale")
        if any(scale <= 0.0 for scale in self.scale):
            raise ValueError(f"mesh.scale must be positive, got {self.scale}")

    @property
    def suffix(self) -> str:
        return Path(self.file_path).suffix.lower()


GeometryValue = GeometryCfg | str | Mapping[str, Any]
"""Loose geometry input accepted by schema normalization."""


def make_geometry_cfg(value: GeometryValue) -> GeometryCfg:
    r"""Normalize loose geometry input into a `GeometryCfg`.

    Args:
        value (GeometryValue): Loose geometry input.

    Returns:
        GeometryCfg: Normalized geometry object.

    Raises:
        TypeError: If the input type is unsupported.
        KeyError: If a dictionary input misses the geometry-type key.
        ValueError: If the geometry type value is unsupported.
    """

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
    if geometry_type == "sphere":
        return SphereGeometryCfg(radius=value["radius"])
    if geometry_type == "mesh":
        file_path = value.get("file_path", value.get("path", value.get("mesh")))
        return MeshGeometryCfg(file_path=file_path, scale=value.get("scale", (1.0, 1.0, 1.0)))

    raise ValueError(f"Unsupported geometry type: {geometry_type}")


@dataclass
class GeometryElementCfg(AssetCfgBase):
    r"""Concrete geometry instance with local pose and optional material."""

    geometry: GeometryCfg
    """Underlying geometry descriptor."""

    name: str | None = None
    """Optional instance name for debugging/export."""

    origin: PoseCfg | Sequence[float] | Mapping[str, Any] | None = field(default_factory=PoseCfg)
    """Local pose relative to the owning joint/child-link frame."""

    material: MaterialCfg | Mapping[str, Any] | None = None
    """Optional material, only consumed by visual/recolored flows."""

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
    r"""Collision geometry instance."""


@dataclass
class VisualGeometryCfg(GeometryElementCfg):
    r"""Visual geometry instance."""


@dataclass
class InertiaTensorCfg(AssetCfgBase):
    r"""URDF-style symmetric inertia tensor.

    $$
    \mathbf{I} =
    \begin{bmatrix}
    i_{xx} & i_{xy} & i_{xz} \\
    i_{xy} & i_{yy} & i_{yz} \\
    i_{xz} & i_{yz} & i_{zz}
    \end{bmatrix}.
    $$
    """

    ixx: float
    """Diagonal entry $i_{xx}$."""

    iyy: float
    """Diagonal entry $i_{yy}$."""

    izz: float
    """Diagonal entry $i_{zz}$."""

    ixy: float = 0.0
    """Off-diagonal entry $i_{xy}$."""

    ixz: float = 0.0
    """Off-diagonal entry $i_{xz}$."""

    iyz: float = 0.0
    """Off-diagonal entry $i_{yz}$."""

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
    r"""Inertial descriptor for one link-level rigid body."""

    mass: float
    """Link mass $m$."""

    inertia: InertiaTensorCfg | Mapping[str, Any]
    """Inertia tensor descriptor."""

    origin: PoseCfg | Sequence[float] | Mapping[str, Any] | None = field(default_factory=PoseCfg)
    """Pose of the inertial frame relative to the link frame."""

    inertia_padding: float = 0.0
    """Engineering padding applied to tensor diagonal for numerical robustness."""

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
        r"""Construct inertial parameters from a uniform box primitive.

        Args:
            size (Vector3): Box side lengths $(s_x, s_y, s_z)$.
            density (float): Density $\rho$.
            origin (PoseCfg | Sequence[float] | Mapping[str, Any] | None): Inertial-frame pose.
            min_mass (float): Lower bound for mass stabilization.
            inertia_padding (float): Diagonal padding for numerical robustness.

        Returns:
            InertialCfg: Box-derived inertial descriptor.
        """

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
        r"""Construct inertial parameters from a uniform cylinder primitive.

        Args:
            radius (float): Cylinder radius $r$.
            length (float): Cylinder length $l$.
            density (float): Density $\rho$.
            origin (PoseCfg | Sequence[float] | Mapping[str, Any] | None): Inertial-frame pose.
            principal_axis (Literal["x", "y", "z"]): Local cylinder longitudinal axis.
            min_mass (float): Lower bound for mass stabilization.
            inertia_padding (float): Diagonal padding for numerical robustness.

        Returns:
            InertialCfg: Cylinder-derived inertial descriptor.
        """

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
    def from_sphere(
        cls,
        radius: float,
        density: float,
        *,
        origin: PoseCfg | Sequence[float] | Mapping[str, Any] | None = None,
        min_mass: float = 1e-4,
        inertia_padding: float = 1e-8,
    ) -> InertialCfg:
        r"""Construct inertial parameters from a uniform sphere primitive.

        Args:
            radius (float): Sphere radius $r$.
            density (float): Density $\rho$.
            origin (PoseCfg | Sequence[float] | Mapping[str, Any] | None): Inertial-frame pose.
            min_mass (float): Lower bound for mass stabilization.
            inertia_padding (float): Diagonal padding for numerical robustness.

        Returns:
            InertialCfg: Sphere-derived inertial descriptor.
        """

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
    r"""Joint limits and optional drive bounds."""

    lower: float
    r"""Joint lower bound $q_{\min}$."""

    upper: float
    r"""Joint upper bound $q_{\max}$."""

    effort: float | None = None
    """Optional torque/force upper bound."""

    velocity: float | None = None
    """Optional velocity upper bound."""

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
class MimicCfg(AssetCfgBase):
    r"""URDF mimic-joint schema."""

    joint: str
    """The parent joint referenced by the mimic relation."""

    multiplier: float = 1.0
    """Linear multiplier $\alpha$."""

    offset: float = 0.0
    """Linear offset $\beta$."""

    def __post_init__(self):
        self.joint = _sanitize_identifier(self.joint, field_name="mimic.joint")
        self.multiplier = float(self.multiplier)
        self.offset = float(self.offset)


def _make_collision_cfg(value: Any) -> CollisionGeometryCfg:
    r"""Normalize loose input into `CollisionGeometryCfg`.

    Args:
        value (Any): Loose collision-geometry input.

    Returns:
        CollisionGeometryCfg: Normalized collision geometry.

    Raises:
        TypeError: If the input cannot be interpreted as collision geometry.
    """

    if isinstance(value, CollisionGeometryCfg):
        return value.copy()
    if isinstance(value, GeometryCfg) or isinstance(value, str):
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
    r"""Normalize loose input into `VisualGeometryCfg`.

    Args:
        value (Any): Loose visual-geometry input.

    Returns:
        VisualGeometryCfg: Normalized visual geometry.

    Raises:
        TypeError: If the input cannot be interpreted as visual geometry.
    """

    if isinstance(value, VisualGeometryCfg):
        return value.copy()
    if isinstance(value, GeometryCfg) or isinstance(value, str):
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
    "SphereGeometryCfg",
    "MeshGeometryCfg",
    "GeometryValue",
    "GeometryElementCfg",
    "CollisionGeometryCfg",
    "VisualGeometryCfg",
    "InertiaTensorCfg",
    "InertialCfg",
    "JointLimitCfg",
    "MimicCfg",
    "make_geometry_cfg",
    "_FLOAT_TOLERANCE",
    "_sanitize_identifier",
    "_ensure_tuple",
    "_normalize_axis",
    "_ensure_list",
    "_make_collision_cfg",
    "_make_visual_cfg",
]
