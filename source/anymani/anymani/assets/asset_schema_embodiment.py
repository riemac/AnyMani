"Defines topology and family-composition fields for a generated hand embodiment."

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
    JointPropertiesCfg,
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
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    name: str
    "Stable semantic identifier preserved in the output metadata."

    parent: str
    "Parent link or owner identifier in the source topology."

    joint_type: JointType = "revolute"
    "Whether the link connection is fixed or actuated by a revolute coordinate."

    child: str | None = None
    "Child link name in the source kinematic tree."

    axis: Vector3 = (0.0, 0.0, 1.0)
    "Unit direction expressed in the owning local frame."

    limit: JointLimitCfg | Mapping[str, Any] | Sequence[float] | None = (-3.141592653589793, 3.141592653589793)
    "Lower and upper joint-coordinate bounds in radians."

    origin: PoseCfg | Sequence[float] | Mapping[str, Any] | None = field(default_factory=PoseCfg)
    "Rigid transform with translation in meters and rotation in radians."

    inertial: InertialCfg | Mapping[str, Any] | None = None
    "Mass, center of mass, and inertia for a rigid body."

    joint_properties: JointPropertiesCfg | Mapping[str, Any] | None = None
    "Optional effort, velocity, and friction values aligned to the source chain."

    collisions: list[CollisionGeometryCfg] = field(default_factory=list)
    "Collision shapes that determine physical contact and physics closure."

    visuals: list[VisualGeometryCfg] = field(default_factory=list)
    "Presentation geometry and materials; excluded from collision and physical identity."

    mimic: MimicCfg | Mapping[str, Any] | None = None
    "Optional relation to another joint coordinate."

    is_tip: bool = False
    "Whether this generation or validation condition is active."

    metadata: dict[str, Any] = field(default_factory=dict)
    "JSON-safe provenance and extension fields, separate from physical geometry."

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

        if self.joint_properties is not None and not isinstance(self.joint_properties, JointPropertiesCfg):
            if not isinstance(self.joint_properties, Mapping):
                raise TypeError(
                    f"joint_properties must be JointPropertiesCfg or mapping, got {self.joint_properties!r}"
                )
            self.joint_properties = JointPropertiesCfg(**self.joint_properties)

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
        "Returns the number of active revolute joints in canonical order."

        return 0 if self.joint_type == "fixed" else 1

    @property
    def uses_only_primitive_collision(self) -> bool:
        "Reports whether every collision component uses a supported primitive."

        return all(collision.geometry.is_primitive for collision in self.collisions)


@dataclass
class FingerCfg(AssetCfgBase):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    name: str
    "Stable semantic identifier preserved in the output metadata."

    parent_link: str = "palm"
    "Source parent link for this joint or finger root."

    mount: PoseCfg | Sequence[float] | Mapping[str, Any] | None = field(default_factory=PoseCfg)
    "Finger-root pose in the palm frame, with translation in meters and rotation in radians."

    joints: list[JointCfg] = field(default_factory=list)
    "Joint chain in the declared proximal-to-distal source order."

    metadata: dict[str, Any] = field(default_factory=dict)
    "Finger-level provenance and extension fields, separate from joint geometry."

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
        "Returns active joint names in stable canonical depth-major order."

        return [joint.name for joint in self.joints]

    @property
    def tip_joint(self) -> JointCfg:
        "Returns the terminal active joint for this finger."

        return self.joints[-1]

    @property
    def tip_link(self) -> str:
        "Returns the terminal link for this finger."

        return cast(str, self.tip_joint.child)

    @property
    def dof_count(self) -> int:
        "Returns the number of active revolute joints in canonical order."

        return sum(joint.dof_count for joint in self.joints)


@dataclass
class PalmCfg(AssetCfgBase):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    name: str = "palm"
    "Stable semantic identifier preserved in the output metadata."

    origin: PoseCfg | Sequence[float] | Mapping[str, Any] | None = field(default_factory=PoseCfg)
    "Rigid transform with translation in meters and rotation in radians."

    inertial: InertialCfg | Mapping[str, Any] | None = None
    "Mass, center of mass, and inertia for a rigid body."

    collisions: list[CollisionGeometryCfg] = field(default_factory=list)
    "Collision shapes that determine physical contact and physics closure."

    visuals: list[VisualGeometryCfg] = field(default_factory=list)
    "Presentation geometry and materials; excluded from collision and physical identity."

    metadata: dict[str, Any] = field(default_factory=dict)
    "Palm-level provenance and extension fields, separate from collision geometry."

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
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    name: str
    "Stable semantic identifier preserved in the output metadata."

    palm: PalmCfg | Mapping[str, Any] = field(default_factory=PalmCfg)
    "Palm body density or configuration, depending on the enclosing schema."

    fingers: list[FingerCfg] = field(default_factory=list)
    "Finger configurations in semantic slot order."

    family: str = "generic"
    "Base-palm family; surviving slot families are stored separately for mixed hands."

    handedness: Handedness = "unknown"
    "Requested side; generated left hands follow the strict mirror contract."

    metadata: dict[str, Any] = field(default_factory=dict)
    "Hand-level provenance and extension fields, separate from physical geometry."

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
        'Returns the joints in their declared order.'

        return [joint for finger in self.fingers for joint in finger.joints]

    @property
    def joint_name_to_index(self) -> dict[str, int]:
        "Maps active joint names to canonical action indices."

        return {joint.name: index for index, joint in enumerate(self.iter_joints())}

    @property
    def dof_count(self) -> int:
        "Returns the number of active revolute joints in canonical order."

        return sum(joint.dof_count for joint in self.iter_joints())

    @property
    def fingertip_links(self) -> list[str]:
        "Returns the terminal body link for each surviving finger."

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
