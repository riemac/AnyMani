"Stores reviewed official joint limits, effort, velocity, and friction by anatomical child-link slot."

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..asset_schema_core import AssetCfgBase, JointLimitCfg, JointPropertiesCfg, _ensure_tuple


_NON_THUMB_SUFFIXES = ("mcp1", "mcp2", "pip", "dip")
"Canonical non-thumb child-link stages in proximal-to-distal order."


_THUMB_SUFFIXES = ("cmc1", "cmc2", "mcp", "dip")
"Canonical thumb child-link stages in proximal-to-distal order."


@dataclass
class JointPhysicalPreset(AssetCfgBase):
    "Reviewed limits, effort, velocity, and friction keyed by anatomical child-link slot."

    child_suffix: str
    "Anatomical child-link stage such as mcp1, pip, or dip."

    source_joints: tuple[str, ...] | str
    "Original URDF joint names used to audit physical presets."

    limit: JointLimitCfg | Mapping[str, Any] | Sequence[float]
    "Lower and upper joint-coordinate bounds in radians."

    friction: float | None = None
    "Joint friction coefficient copied from the reviewed family profile."

    def __post_init__(self):

        if isinstance(self.source_joints, str):
            self.source_joints = (self.source_joints,)
        else:
            self.source_joints = tuple(str(item) for item in self.source_joints)


        if isinstance(self.limit, JointLimitCfg):
            self.limit = self.limit.copy()
        elif isinstance(self.limit, Mapping):
            self.limit = JointLimitCfg(**dict(self.limit))
        elif isinstance(self.limit, Sequence) and not isinstance(self.limit, (str, bytes)):
            lower, upper = _ensure_tuple(self.limit, length=2, field_name="physical.limit")
            self.limit = JointLimitCfg(lower=lower, upper=upper)
        else:
            raise TypeError(f"Unsupported physical limit: {self.limit!r}")


        if self.friction is not None:
            self.friction = float(self.friction)

    def joint_properties(self) -> JointPropertiesCfg | None:
        "Returns the optional friction properties associated with this joint profile."

        if self.friction is None:
            return None
        return JointPropertiesCfg(friction=self.friction)


def _limit(lower: float, upper: float, effort: float, velocity: float) -> JointLimitCfg:

    return JointLimitCfg(lower=lower, upper=upper, effort=effort, velocity=velocity)


LEAP_NON_THUMB_PHYSICAL_PROFILE: tuple[JointPhysicalPreset, ...] = (
    JointPhysicalPreset("mcp1", ("1", "5", "9"), _limit(-0.314, 2.23, 0.95, 8.48), friction=0.0),
    JointPhysicalPreset("mcp2", ("0", "4", "8"), _limit(-1.047, 1.047, 0.95, 8.48), friction=0.0),
    JointPhysicalPreset("pip", ("2", "6", "10"), _limit(-0.506, 1.885, 0.95, 8.48), friction=0.0),
    JointPhysicalPreset("dip", ("3", "7", "11"), _limit(-0.366, 2.042, 0.95, 8.48), friction=0.0),
)
"Official LEAP non-thumb limits, actuator values, and friction in anatomical order."


LEAP_THUMB_PHYSICAL_PROFILE: tuple[JointPhysicalPreset, ...] = (
    JointPhysicalPreset("cmc1", "12", _limit(-0.349, 2.094, 0.95, 8.48), friction=0.0),
    JointPhysicalPreset("cmc2", "13", _limit(-0.47, 2.443, 0.95, 8.48), friction=0.0),
    JointPhysicalPreset("mcp", "14", _limit(-1.20, 1.90, 0.95, 8.48), friction=0.0),
    JointPhysicalPreset("dip", "15", _limit(-1.34, 1.88, 0.95, 8.48), friction=0.0),
)
"Official LEAP thumb limits, actuator values, and friction in anatomical order."


ALLEGRO_NON_THUMB_PHYSICAL_PROFILE: tuple[JointPhysicalPreset, ...] = (
    JointPhysicalPreset("mcp1", ("joint_0.0", "joint_4.0", "joint_8.0"), _limit(-0.47, 0.47, 10.0, 3.14)),
    JointPhysicalPreset("mcp2", ("joint_1.0", "joint_5.0", "joint_9.0"), _limit(-0.196, 1.61, 10.0, 3.14)),
    JointPhysicalPreset("pip", ("joint_2.0", "joint_6.0", "joint_10.0"), _limit(-0.174, 1.709, 10.0, 3.14)),
    JointPhysicalPreset("dip", ("joint_3.0", "joint_7.0", "joint_11.0"), _limit(-0.227, 1.618, 10.0, 3.14)),
)
"Official Allegro non-thumb limits and actuator values in anatomical order."


ALLEGRO_THUMB_PHYSICAL_PROFILE: tuple[JointPhysicalPreset, ...] = (
    JointPhysicalPreset("cmc1", "joint_12.0", _limit(0.263, 1.396, 10.0, 3.14)),
    JointPhysicalPreset("cmc2", "joint_13.0", _limit(-0.105, 1.163, 10.0, 3.14)),
    JointPhysicalPreset("mcp", "joint_14.0", _limit(-0.189, 1.644, 10.0, 3.14)),
    JointPhysicalPreset("dip", "joint_15.0", _limit(-0.162, 1.719, 10.0, 3.14)),
)
"Official Allegro thumb limits and actuator values in anatomical order."


FINGER_PHYSICAL_PROFILE_REGISTRY: dict[str, tuple[JointPhysicalPreset, ...]] = {
    "leap_non_thumb_v1": LEAP_NON_THUMB_PHYSICAL_PROFILE,
    "leap_thumb_v1": LEAP_THUMB_PHYSICAL_PROFILE,
    "allegro_non_thumb_v1": ALLEGRO_NON_THUMB_PHYSICAL_PROFILE,
    "allegro_thumb_v1": ALLEGRO_THUMB_PHYSICAL_PROFILE,
}
"Registry of reviewed official joint-limit profiles."


def get_finger_physical_profile(preset_name: str) -> tuple[JointPhysicalPreset, ...]:
    'Returns finger physical profile.'

    try:
        return tuple(item.copy() for item in FINGER_PHYSICAL_PROFILE_REGISTRY[preset_name])
    except KeyError as exc:
        raise KeyError(f"Unknown finger physical profile: {preset_name!r}") from exc


def apply_physical_profile_to_finger_cfg(preset_name: str, cfg):
    'Applies physical profile to finger cfg.'

    profile = get_finger_physical_profile(preset_name)
    if len(profile) != cfg.num_joints:
        raise ValueError(f"physical profile length must be {cfg.num_joints}, got {len(profile)} for {preset_name!r}")


    return cfg.replace(
        joint_limits=[item.limit.copy() for item in profile],
        joint_properties=[item.joint_properties() for item in profile],
    )


__all__ = [
    "JointPhysicalPreset",
    "LEAP_NON_THUMB_PHYSICAL_PROFILE",
    "LEAP_THUMB_PHYSICAL_PROFILE",
    "ALLEGRO_NON_THUMB_PHYSICAL_PROFILE",
    "ALLEGRO_THUMB_PHYSICAL_PROFILE",
    "FINGER_PHYSICAL_PROFILE_REGISTRY",
    "get_finger_physical_profile",
    "apply_physical_profile_to_finger_cfg",
]
