r"""Regular finger builders for the first pre-made asset slice."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from ..asset_base import FingerCfg
from ..asset_builders import FingerBuilder, FingerBuilderCfg
from ..asset_schema_core import JointLimitCfg, PoseCfg, Vector2, Vector3, Vector6, _ensure_tuple, _normalize_axis
from .joint_builders_primitive import PrimJointBuilderCfg


def _to_si(value: float | int) -> float:
    r"""Convert likely-centimeter inputs into meters while preserving SI inputs."""

    value = float(value)
    return value / 100.0 if abs(value) > 0.5 else value


def _normalize_pose_value(value: float | Sequence[float] | None, *, field_name: str) -> Vector6:
    r"""Normalize float / xyz / xyzrpy into one packed 6D pose."""

    if value is None:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if isinstance(value, (int, float)):
        return (0.0, _to_si(value), 0.0, 0.0, 0.0, 0.0)
    packed = _ensure_tuple(value, length=len(value), field_name=field_name)
    if len(packed) == 2:
        return (0.0, _to_si(packed[0]), _to_si(packed[1]), 0.0, 0.0, 0.0)
    if len(packed) == 3:
        return (_to_si(packed[0]), _to_si(packed[1]), _to_si(packed[2]), 0.0, 0.0, 0.0)
    if len(packed) == 6:
        return (
            _to_si(packed[0]),
            _to_si(packed[1]),
            _to_si(packed[2]),
            float(packed[3]),
            float(packed[4]),
            float(packed[5]),
        )
    raise ValueError(f"{field_name} must be scalar / xyz / xyzrpy, got {value!r}")


def _normalize_pose_list(values: Sequence[Any], *, count: int, field_name: str) -> list[Vector6]:
    r"""Normalize a pose list into a fixed-length list of packed 6D poses."""

    if not values:
        return [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0) for _ in range(count)]
    if len(values) != count:
        raise ValueError(f"{field_name} length must be {count}, got {len(values)}")
    return [_normalize_pose_value(value, field_name=f"{field_name}[{idx}]") for idx, value in enumerate(values)]


def _normalize_joint_limits(values: Sequence[Any] | None, *, count: int) -> list[JointLimitCfg | None]:
    r"""Normalize optional joint limit overrides."""

    if not values:
        return [(-3.141592653589793, 3.141592653589793) for _ in range(count)]
    if len(values) != count:
        raise ValueError(f"joint_limits length must be {count}, got {len(values)}")
    limits: list[JointLimitCfg | None] = []
    for value in values:
        if value is None:
            limits.append(None)
        elif isinstance(value, JointLimitCfg):
            limits.append(value.copy())
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            low, high = _ensure_tuple(value, length=2, field_name="joint_limits")
            limits.append(JointLimitCfg(lower=float(low), upper=float(high)))
        else:
            raise TypeError(f"Unsupported joint limit value: {value!r}")
    return limits


def _normalize_tip_dict(tip: dict[str, Any] | None) -> dict[str, Any]:
    r"""Normalize tip recipe values and convert likely-cm lengths into meters."""

    tip = dict(tip or {"type": "cs", "radius": 0.012, "height": 0.01})
    tip_type = str(tip.get("type", tip.get("kind", "cs"))).lower()
    normalized: dict[str, Any] = {"type": tip_type}
    if tip_type == "cs":
        normalized["radius"] = _to_si(tip.get("radius", 0.012))
        normalized["height"] = _to_si(tip.get("height", 0.01))
    elif tip_type == "bs":
        normalized["radius"] = _to_si(tip.get("radius", 0.012))
        normalized["height"] = _to_si(tip.get("height", 0.01))
        normalized["width"] = _to_si(tip.get("width", tip.get("depth", 0.02)))
        normalized["depth"] = _to_si(tip.get("depth", tip.get("width", 0.02)))
    else:
        raise ValueError(f"Only cs/bs tip recipes are supported in v1, got {tip_type!r}")
    return normalized


def _mesh_length(mesh: dict[str, Any]) -> float:
    if mesh["type"] == "box":
        return float(mesh["length"])
    if mesh["type"] == "cylinder":
        return float(mesh["length"])
    raise ValueError(f"Unsupported mesh type for length inference: {mesh['type']}")


def _mesh_cross_section(mesh: dict[str, Any]) -> tuple[float, float]:
    if mesh["type"] == "box":
        return float(mesh["width"]), float(mesh["height"])
    radius = float(mesh["radius"])
    diameter = radius * 2.0
    return diameter, diameter


def _build_box_mesh(*, length: float, width: float, height: float, offset: Vector6, center_on_joint: bool = False) -> dict[str, Any]:
    return {
        "type": "box",
        "length": length,
        "width": width,
        "height": height,
        "offset": offset,
        "center_on_joint": center_on_joint,
    }


def _build_cylinder_mesh(*, length: float, radius: float, offset: Vector6, center_on_joint: bool = False) -> dict[str, Any]:
    return {
        "type": "cylinder",
        "length": length,
        "radius": radius,
        "offset": offset,
        "center_on_joint": center_on_joint,
    }


@dataclass
class RegularFingerBuilderCfg(FingerBuilderCfg):
    r"""Regular, non-spherical finger builder configuration.

    The regular builder targets the first embodiment slice only: Allegro-like
    non-thumbs, LEAP-like non-thumbs, and the shared thumb path with a special
    ``CMC1`` frame convention.
    """

    class_type: type["RegularFingerBuilder"] | None = None
    """Associated runtime builder type."""

    name: str = "finger"
    """Logical finger name used to derive joint/link identifiers."""

    parent_link: str = "palm"
    """Parent link to which the finger root attaches inside ``FingerCfg``."""

    num_joints: int = 4
    """Number of actuated joints, excluding the fixed tip joint added in v1."""

    mesh_shape: list[dict[str, Any]] = field(default_factory=list)
    """Per-joint primitive mesh recipes after normalization."""

    mesh_offsets: list[Any] = field(default_factory=list)
    """Per-joint mesh offsets supplied as scalar / xyz / xyzrpy."""

    _mesh_offsets_6d: list[Vector6] = field(init=False, default_factory=list)
    """Canonical per-joint 6D mesh offsets."""

    tip: dict[str, Any] = field(default_factory=dict)
    """Tip recipe. Only ``cs`` and ``bs`` are supported in v1."""

    tip_offset: Any = None
    """Tip mesh offset relative to the fixed tip joint frame."""

    _tip_offset_6d: Vector6 = field(init=False, default=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    """Canonical 6D tip offset."""

    axes: list[Vector3] = field(default_factory=list)
    """Per-joint axes expressed in the unified finger frame."""

    joint_limits: list[Any] = field(default_factory=list)
    """Optional per-joint limit overrides."""

    def __post_init__(self):
        super().__post_init__()
        if self.num_joints < 1:
            raise ValueError("num_joints must be >= 1")
        self._mesh_offsets_6d = _normalize_pose_list(self.mesh_offsets, count=self.num_joints, field_name="mesh_offsets")
        self._tip_offset_6d = _normalize_pose_value(self.tip_offset, field_name="tip_offset")
        self.tip = _normalize_tip_dict(self.tip)
        if not self.axes:
            self.axes = [(0.0, 0.0, 1.0) for _ in range(self.num_joints)]
        if len(self.axes) != self.num_joints:
            raise ValueError(f"axes length must equal num_joints={self.num_joints}")
        self.axes = [_normalize_axis(_ensure_tuple(axis, length=3, field_name="axes")) for axis in self.axes]
        self.joint_limits = _normalize_joint_limits(self.joint_limits, count=self.num_joints)
        if len(self.mesh_shape) != self.num_joints:
            raise ValueError(f"mesh_shape length must equal num_joints={self.num_joints}")
        self.class_type = RegularFingerBuilder


@dataclass
class AllegroFingerBuilderCfg(RegularFingerBuilderCfg):
    r"""Allegro non-thumb builder configuration."""

    width: float | None = None
    height: float | None = None
    radius: float | None = None
    length: list[float] = field(default_factory=lambda: [1.8, 5.4, 3.8, 2.2])

    def __post_init__(self):
        lengths = [_to_si(value) for value in self.length[: self.num_joints]]
        width = _to_si(self.width or 2.7)
        height = _to_si(self.height or 2.0)
        radius = _to_si(self.radius) if self.radius is not None else None
        defaults = _normalize_pose_list([0.0, 0.0, -0.6, 0.0][: self.num_joints], count=self.num_joints, field_name="allegro_default_offsets")
        merged_offsets = self.mesh_offsets or defaults
        self.mesh_offsets = merged_offsets
        if not self.axes:
            self.axes = [(0.0, 1.0, 0.0)] + [(1.0, 0.0, 0.0)] * max(self.num_joints - 1, 0)
        if not self.tip:
            self.tip = {"type": "cs", "radius": 1.2, "height": 1.0}
        if not self.mesh_shape:
            builder = _build_cylinder_mesh if radius is not None and self.width is None and self.height is None else _build_box_mesh
            self.mesh_shape = [
                builder(length=length, radius=radius, offset=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                if builder is _build_cylinder_mesh
                else _build_box_mesh(length=length, width=width, height=height, offset=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                for length in lengths
            ]
        super().__post_init__()
        for idx, offset in enumerate(self._mesh_offsets_6d):
            self.mesh_shape[idx]["offset"] = offset


@dataclass
class LeapFingerBuilderCfg(RegularFingerBuilderCfg):
    r"""LEAP non-thumb builder configuration."""

    width: float | None = None
    height: float | None = None
    radius: float | None = None
    length: list[float] = field(default_factory=lambda: [3.9, 1.5, 3.6, 2.0])
    fixed_part: float | None = None
    """A palm-side fixed segment length inserted before the first revolute joint."""

    def __post_init__(self):
        lengths = [_to_si(value) for value in self.length[: self.num_joints]]
        width = _to_si(self.width or 3.4)
        height = _to_si(self.height or 2.05)
        radius = _to_si(self.radius) if self.radius is not None else None
        self.fixed_part = _to_si(self.fixed_part or 1.3)
        if not self.axes:
            defaults = [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
            self.axes = defaults[: self.num_joints]
        if not self.tip:
            # User confirmed that the first testing path may use cylinder+sphere.
            self.tip = {"type": "cs", "radius": 1.2, "height": 1.0}
        if not self.mesh_shape:
            builder = _build_cylinder_mesh if radius is not None and self.width is None and self.height is None else _build_box_mesh
            self.mesh_shape = [
                builder(length=length, radius=radius, offset=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                if builder is _build_cylinder_mesh
                else _build_box_mesh(length=length, width=width, height=height, offset=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                for length in lengths
            ]
        super().__post_init__()
        for idx, offset in enumerate(self._mesh_offsets_6d):
            self.mesh_shape[idx]["offset"] = offset


@dataclass
class RegularThumbBuilderCfg(RegularFingerBuilderCfg):
    r"""Shared thumb builder configuration with a special ``CMC1`` rule."""

    cmc1_width: float | None = None
    cmc1_height: float | None = None
    width: float | None = None
    height: float | None = None
    lengths: list[float] = field(default_factory=lambda: [4.5, 1.7, 4.3, 4.0])
    cmc1_offset: float | Vector2 | Vector3 = (0.9, 1.45)
    non_cmc1_offset: list[Any] = field(default_factory=lambda: [-0.2, 0.0, -0.9])

    def __post_init__(self):
        self.num_joints = len(self.lengths)
        lengths = [_to_si(value) for value in self.lengths]
        cmc1_width = _to_si(self.cmc1_width or 3.5)
        cmc1_height = _to_si(self.cmc1_height or 3.4)
        width = _to_si(self.width or 1.9)
        height = _to_si(self.height or 2.7)

        cmc1_pose = _normalize_pose_value(self.cmc1_offset, field_name="cmc1_offset")
        other_offsets = _normalize_pose_list(self.non_cmc1_offset, count=self.num_joints - 1, field_name="non_cmc1_offset")
        self.mesh_offsets = [cmc1_pose] + other_offsets
        if not self.axes:
            # Question:
            # Thumb axis semantics across Allegro / LEAP families are still a research
            # choice. For the first slice we use one stable canonical convention.
            self.axes = [
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            ]
        if not self.tip:
            self.tip = {"type": "cs", "radius": 1.2, "height": 1.0}
        if not self.mesh_shape:
            self.mesh_shape = [
                _build_box_mesh(length=lengths[0], width=cmc1_width, height=cmc1_height, offset=cmc1_pose, center_on_joint=True),
                *[
                    _build_box_mesh(length=lengths[idx], width=width, height=height, offset=other_offsets[idx - 1])
                    for idx in range(1, self.num_joints)
                ],
            ]
        super().__post_init__()
        self.mesh_shape[0]["center_on_joint"] = True
        for idx, offset in enumerate(self._mesh_offsets_6d):
            self.mesh_shape[idx]["offset"] = offset


@dataclass
class SphericalFingerBuilderCfg(FingerBuilderCfg):
    r"""Placeholder for future spherical-joint finger builders."""


class RegularFingerBuilder(FingerBuilder):
    r"""Builder for regular fingers with primitive-link geometry."""

    cfg: RegularFingerBuilderCfg

    def __init__(self, cfg: RegularFingerBuilderCfg):
        super().__init__(cfg)
        self.cfg = cfg

    def build(self) -> FingerCfg:
        r"""Build a regular finger into canonical ``FingerCfg`` form."""

        if isinstance(self.cfg, RegularThumbBuilderCfg):
            joints = self._build_thumb_chain()
        else:
            first_gap = self.cfg.fixed_part if isinstance(self.cfg, LeapFingerBuilderCfg) else 0.0
            joints = self._build_serial_chain(first_gap=first_gap)
        return FingerCfg(
            name=self.cfg.name,
            parent_link=self.cfg.parent_link,
            mount=PoseCfg(),
            joints=joints,
            metadata={"builder": self.cfg.__class__.__name__},
        )

    def _build_serial_chain(self, *, first_gap: float) -> list[Any]:
        joints = []
        parent_link = self.cfg.parent_link
        previous_valid_length = first_gap
        for index in range(self.cfg.num_joints):
            origin = PoseCfg(pos=(0.0, previous_valid_length, 0.0)) if index > 0 or first_gap > 0.0 else PoseCfg()
            joint = self._build_joint(index=index, parent_link=parent_link, origin=origin)
            joints.append(joint)
            parent_link = joint.child
            previous_valid_length = _mesh_length(self.cfg.mesh_shape[index]) + self.cfg._mesh_offsets_6d[index][1]

        joints.append(self._build_tip_joint(parent_link=parent_link, tip_origin_y=previous_valid_length))
        return joints

    def _build_thumb_chain(self) -> list[Any]:
        cfg = self.cfg
        assert isinstance(cfg, RegularThumbBuilderCfg)

        joints = [self._build_joint(index=0, parent_link=cfg.parent_link, origin=PoseCfg())]
        parent_link = joints[0].child

        cmc1_offset = cfg._mesh_offsets_6d[0]
        cmc1_length = _mesh_length(cfg.mesh_shape[0])
        cmc1_width, cmc1_height = _mesh_cross_section(cfg.mesh_shape[0])
        next_width, next_height = _mesh_cross_section(cfg.mesh_shape[1])
        origin_1 = PoseCfg(
            pos=(
                (cmc1_width - next_width) / 2.0,
                cmc1_offset[1] + cmc1_length / 2.0,
                cmc1_offset[2] - (cmc1_height - next_height) / 2.0,
            )
        )
        joint_1 = self._build_joint(index=1, parent_link=parent_link, origin=origin_1)
        joints.append(joint_1)
        parent_link = joint_1.child

        previous_valid_length = _mesh_length(cfg.mesh_shape[1]) + cfg._mesh_offsets_6d[1][1]
        for index in range(2, cfg.num_joints):
            origin = PoseCfg(pos=(0.0, previous_valid_length, 0.0))
            joint = self._build_joint(index=index, parent_link=parent_link, origin=origin)
            joints.append(joint)
            parent_link = joint.child
            previous_valid_length = _mesh_length(cfg.mesh_shape[index]) + cfg._mesh_offsets_6d[index][1]

        joints.append(self._build_tip_joint(parent_link=parent_link, tip_origin_y=previous_valid_length))
        return joints

    def _build_joint(self, *, index: int, parent_link: str, origin: PoseCfg):
        mesh = dict(self.cfg.mesh_shape[index])
        builder_cfg = PrimJointBuilderCfg(
            name=f"{self.cfg.name}_j{index}",
            parent=parent_link,
            child=f"{self.cfg.name}_link_{index}",
            joint_type="revolute",
            origin=origin,
            axis=self.cfg.axes[index],
            limit=self.cfg.joint_limits[index],
            mesh=mesh,
            metadata={
                "finger_name": self.cfg.name,
                "joint_index": index,
                "allow_zero_origin": index == 0 and origin.pos == (0.0, 0.0, 0.0),
            },
        )
        builder = builder_cfg.class_type(builder_cfg)
        return builder.build()

    def _build_tip_joint(self, *, parent_link: str, tip_origin_y: float):
        tip_recipe = dict(self.cfg.tip)
        tip_recipe["offset"] = self.cfg._tip_offset_6d
        builder_cfg = PrimJointBuilderCfg(
            name=f"{self.cfg.name}_tip",
            parent=parent_link,
            child=f"{self.cfg.name}_tip_link",
            joint_type="fixed",
            origin=PoseCfg(pos=(0.0, tip_origin_y, 0.0)),
            axis=(0.0, 0.0, 0.0),
            limit=None,
            mesh=tip_recipe,
            is_tip=True,
            metadata={"finger_name": self.cfg.name, "joint_index": "tip"},
        )
        builder = builder_cfg.class_type(builder_cfg)
        return builder.build()


__all__ = [
    "RegularFingerBuilderCfg",
    "AllegroFingerBuilderCfg",
    "LeapFingerBuilderCfg",
    "RegularThumbBuilderCfg",
    "SphericalFingerBuilderCfg",
    "RegularFingerBuilder",
]
