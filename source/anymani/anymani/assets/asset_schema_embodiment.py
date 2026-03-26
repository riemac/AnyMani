"""Embodiment-level schema for hand-asset declaration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from .asset_schema_core import (
    AssetCfgBase,
    CollisionGeometryCfg,
    Handedness,
    InertialCfg,
    JointLimitCfg,
    JointType,
    MimicCfg,
    PoseCfg,
    Vector3,
    _FLOAT_TOLERANCE,
    _ensure_list,
    _ensure_tuple,
    _make_collision_cfg,
    _make_visual_cfg,
    _normalize_axis,
    _sanitize_identifier,
)
from .asset_schema_core import VisualGeometryCfg


@dataclass
class JointCfg(AssetCfgBase):
    r"""Joint-centric descriptor for one revolute/fixed joint and its child link."""

    name: str
    """Current joint name and downstream indexing key."""

    parent: str
    """Parent link name."""

    joint_type: JointType = "revolute"
    """Supported joint type in the current project scope."""

    child: str | None = None
    """Child link name; auto-derived if omitted."""

    axis: Vector3 = (0.0, 0.0, 1.0)
    """Joint axis direction $\vec{a}$."""

    limit: JointLimitCfg | Mapping[str, Any] | Sequence[float] | None = (-3.141592653589793, 3.141592653589793)
    """Joint limits, as object / dict / `(lower, upper)` shorthand."""

    origin: PoseCfg | Sequence[float] | Mapping[str, Any] | None = field(default_factory=PoseCfg)
    """Joint-frame pose relative to the parent-link frame."""

    inertial: InertialCfg | Mapping[str, Any] | None = None
    """Child-link inertial descriptor."""

    collisions: list[CollisionGeometryCfg] = field(default_factory=list)
    """Child-link collision geometry list."""

    visuals: list[VisualGeometryCfg] = field(default_factory=list)
    """Child-link visual geometry list."""

    mimic: MimicCfg | Mapping[str, Any] | None = None
    """Optional mimic relation; schema only, not used by generator v1."""

    is_tip: bool = False
    """Whether this joint/link pair is treated as fingertip-related in v1."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Reserved extension metadata."""

    def __post_init__(self):
        self.name = _sanitize_identifier(self.name, field_name="joint.name")
        if self.joint_type not in {"revolute", "fixed"}:
            raise ValueError(f"invalid joint_type: {self.joint_type}, must be 'revolute' or 'fixed'")
        self.parent = _sanitize_identifier(self.parent, field_name="joint.parent")
        self.child = _sanitize_identifier(self.child or f"{self.name}_link", field_name="joint.child")
        self.origin = PoseCfg.from_value(self.origin)

        axis_tuple = _ensure_tuple(self.axis, length=3, field_name="joint.axis")
        if self.joint_type == "fixed" and all(abs(value) <= _FLOAT_TOLERANCE for value in axis_tuple):
            self.axis = (0.0, 0.0, 1.0)
        else:
            self.axis = _normalize_axis(axis_tuple)

        if self.limit is None:
            if self.joint_type != "fixed":
                raise ValueError("Non-fixed joint must provide limit")
        elif isinstance(self.limit, JointLimitCfg):
            self.limit = self.limit.copy()
        elif isinstance(self.limit, Mapping):
            self.limit = JointLimitCfg(**self.limit)
        elif isinstance(self.limit, Sequence) and not isinstance(self.limit, (str, bytes)):
            packed = _ensure_tuple(self.limit, length=2, field_name="joint.limit")
            self.limit = JointLimitCfg(lower=packed[0], upper=packed[1])
        else:
            raise TypeError(f"Unsupported joint limit: {self.limit!r}")

        if self.inertial is not None and not isinstance(self.inertial, InertialCfg):
            if not isinstance(self.inertial, Mapping):
                raise TypeError(f"inertial must be InertialCfg or mapping, got {self.inertial!r}")
            self.inertial = InertialCfg(**self.inertial)

        self.collisions = [_make_collision_cfg(item) for item in _ensure_list(self.collisions, field_name="collisions")]
        self.visuals = [_make_visual_cfg(item) for item in _ensure_list(self.visuals, field_name="visuals")]

        if self.mimic is not None and not isinstance(self.mimic, MimicCfg):
            if not isinstance(self.mimic, Mapping):
                raise TypeError(f"mimic must be MimicCfg or mapping, got {self.mimic!r}")
            self.mimic = MimicCfg(**self.mimic)

    @property
    def dof_count(self) -> int:
        return 0 if self.joint_type == "fixed" else 1

    @property
    def uses_only_primitive_collision(self) -> bool:
        return all(collision.geometry.is_primitive for collision in self.collisions)


@dataclass
class FingerCfg(AssetCfgBase):
    r"""Logical finger descriptor made of a serial joint chain."""

    name: str
    """Logical finger name."""

    parent_link: str = "palm"
    """Link to which the finger is mounted; defaults to palm."""

    mount: PoseCfg | Sequence[float] | Mapping[str, Any] | None = field(default_factory=PoseCfg)
    """Finger-level mounting pose entry."""

    joints: list[JointCfg] = field(default_factory=list)
    """Serial joint list that constitutes the finger."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Reserved extension metadata."""

    def __post_init__(self):
        self.name = _sanitize_identifier(self.name, field_name="finger.name")
        self.parent_link = _sanitize_identifier(self.parent_link, field_name="finger.parent_link")
        self.mount = PoseCfg.from_value(self.mount)
        self.joints = [joint if isinstance(joint, JointCfg) else JointCfg(**joint) for joint in self.joints]

        if not self.joints:
            raise ValueError(f"finger '{self.name}' must contain at least one joint")

        first_parent = self.joints[0].parent
        if first_parent != self.parent_link:
            raise ValueError(
                f"finger '{self.name}' first joint parent must be '{self.parent_link}', got '{first_parent}'"
            )

        for previous, current in zip(self.joints[:-1], self.joints[1:]):
            if current.parent != previous.child:
                raise ValueError(
                    f"finger '{self.name}' chain broken: joint '{current.name}' parent is "
                    f"'{current.parent}', expected '{previous.child}'"
                )

        joint_names = [joint.name for joint in self.joints]
        if len(joint_names) != len(set(joint_names)):
            raise ValueError(f"finger '{self.name}' contains duplicated joint names: {joint_names}")

    @property
    def joint_names(self) -> list[str]:
        return [joint.name for joint in self.joints]

    @property
    def tip_joint(self) -> JointCfg:
        return self.joints[-1]

    @property
    def tip_link(self) -> str:
        return cast(str, self.tip_joint.child)

    @property
    def dof_count(self) -> int:
        return sum(joint.dof_count for joint in self.joints)


@dataclass
class PalmCfg(AssetCfgBase):
    r"""Palm / root-link descriptor."""

    name: str = "palm"
    """Palm/root link name."""

    origin: PoseCfg | Sequence[float] | Mapping[str, Any] | None = field(default_factory=PoseCfg)
    """Palm-frame pose relative to the hand root reference."""

    inertial: InertialCfg | Mapping[str, Any] | None = None
    """Palm inertial descriptor."""

    collisions: list[CollisionGeometryCfg] = field(default_factory=list)
    """Palm collision geometry list."""

    visuals: list[VisualGeometryCfg] = field(default_factory=list)
    """Palm visual geometry list."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Reserved extension metadata."""

    def __post_init__(self):
        self.name = _sanitize_identifier(self.name, field_name="palm.name")
        self.origin = PoseCfg.from_value(self.origin)
        if self.inertial is not None and not isinstance(self.inertial, InertialCfg):
            if not isinstance(self.inertial, Mapping):
                raise TypeError(f"inertial must be InertialCfg or mapping, got {self.inertial!r}")
            self.inertial = InertialCfg(**self.inertial)
        self.collisions = [_make_collision_cfg(item) for item in _ensure_list(self.collisions, field_name="collisions")]
        self.visuals = [_make_visual_cfg(item) for item in _ensure_list(self.visuals, field_name="visuals")]


@dataclass
class HandCfg(AssetCfgBase):
    r"""Canonical top-level hand asset descriptor."""

    name: str
    """Hand asset name."""

    palm: PalmCfg | Mapping[str, Any] = field(default_factory=PalmCfg)
    """Palm/root-link config."""

    fingers: list[FingerCfg] = field(default_factory=list)
    """All fingers in the hand."""

    family: str = "generic"
    """Hand-family tag, such as `leap`, `allegro`, or `generic`."""

    handedness: Handedness = "unknown"
    """Handedness tag; `unknown` is reserved for non-typical embodiments."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Reserved extension metadata."""

    def __post_init__(self):
        self.name = _sanitize_identifier(self.name, field_name="hand.name")
        self.family = _sanitize_identifier(self.family, field_name="hand.family")
        if self.handedness not in {"left", "right", "unknown"}:
            raise ValueError(f"invalid handedness: {self.handedness}")

        if not isinstance(self.palm, PalmCfg):
            if not isinstance(self.palm, Mapping):
                raise TypeError(f"palm must be PalmCfg or mapping, got {self.palm!r}")
            self.palm = PalmCfg(**self.palm)

        self.fingers = [finger if isinstance(finger, FingerCfg) else FingerCfg(**finger) for finger in self.fingers]
        if not self.fingers:
            raise ValueError("hand must contain at least one finger")

        finger_names = [finger.name for finger in self.fingers]
        if len(finger_names) != len(set(finger_names)):
            raise ValueError(f"hand contains duplicated finger names: {finger_names}")

        all_joint_names = [joint.name for joint in self.iter_joints()]
        if len(all_joint_names) != len(set(all_joint_names)):
            raise ValueError(f"hand contains duplicated joint names: {all_joint_names}")

        all_link_names = [self.palm.name] + [joint.child for joint in self.iter_joints()]
        if len(all_link_names) != len(set(all_link_names)):
            raise ValueError(f"hand contains duplicated link names: {all_link_names}")

        for finger in self.fingers:
            if finger.parent_link != self.palm.name:
                raise ValueError(
                    f"finger '{finger.name}' is mounted on '{finger.parent_link}', expected palm link '{self.palm.name}'"
                )

    def iter_joints(self) -> list[JointCfg]:
        r"""Flatten the hand joints in finger order.

        Returns:
            list[JointCfg]: Flattened joint list.
        """

        return [joint for finger in self.fingers for joint in finger.joints]

    @property
    def joint_name_to_index(self) -> dict[str, int]:
        r"""Convenience joint-name to index mapping for debug/export only.

        Returns:
            dict[str, int]: Finger-order joint index mapping.
        """

        return {joint.name: index for index, joint in enumerate(self.iter_joints())}

    @property
    def dof_count(self) -> int:
        return sum(joint.dof_count for joint in self.iter_joints())

    @property
    def fingertip_links(self) -> list[str]:
        return [finger.tip_link for finger in self.fingers]


joint = JointCfg
finger = FingerCfg
palm = PalmCfg
hand = HandCfg


__all__ = [
    "JointCfg",
    "FingerCfg",
    "PalmCfg",
    "HandCfg",
    "joint",
    "finger",
    "palm",
    "hand",
]
