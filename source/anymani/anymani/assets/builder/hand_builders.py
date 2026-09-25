"Assembles canonical right-hand palms and fingers, then mirrors the complete HandCfg for left-hand output."

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..asset_base import HandCfg
from ..asset_builders import FingerBuilderCfg, HandBuilder, HandBuilderCfg
from ..asset_schema_core import PoseCfg
from ..handedness import lower_hand_to_handedness
from .finger_buiders import RegularFingerBuilderCfg
from .palm_builders import SinglePalmBuilderCfg

NON_THUMB_FINGER_NAMES: tuple[str, ...] = ("index", "middle", "ring", "little")


def _to_pose_dict(values: dict[str, PoseCfg]) -> dict[str, PoseCfg]:
    return {name: PoseCfg.from_value(value) for name, value in values.items()}


def _ensure_resolved_finger_cfg(slot_name: str, cfg: FingerBuilderCfg | str | None) -> FingerBuilderCfg | None:

    if cfg is None:
        return None
    if isinstance(cfg, str):
        raise TypeError(
            f"{slot_name} must be a resolved FingerBuilderCfg, got preset string {cfg!r}. "
            "Resolve preset names in `assets.presets` or `RecipeLoader` before constructing HumanLikeHandBuilderCfg."
        )
    return cfg


@dataclass
class HumanLikeHandBuilderCfg(HandBuilderCfg):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    class_type: type[HumanLikeHandBuilder] | None = None
    handedness: Literal["left", "right"] = "right"
    finger_cfg: FingerBuilderCfg | dict[str, FingerBuilderCfg] | None = None
    thumb_cfg: FingerBuilderCfg | None = None
    num_non_thumb: int = 3
    mounts: dict[str, PoseCfg] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self.mounts = _to_pose_dict(self.mounts)
        if isinstance(self.finger_cfg, dict):
            self.finger_cfg = {
                name: _ensure_resolved_finger_cfg(f"finger_cfg[{name!r}]", cfg)
                for name, cfg in self.finger_cfg.items()
            }
            invalid = set(self.finger_cfg) - set(NON_THUMB_FINGER_NAMES)
            if invalid:
                raise ValueError(f"finger_cfg dict keys must be drawn from {NON_THUMB_FINGER_NAMES}, got {invalid}")
            self.num_non_thumb = len(self.finger_cfg)
        elif self.finger_cfg is not None and not 1 <= self.num_non_thumb <= len(NON_THUMB_FINGER_NAMES):
            raise ValueError(f"num_non_thumb must be in [1, {len(NON_THUMB_FINGER_NAMES)}]")
        else:
            self.finger_cfg = _ensure_resolved_finger_cfg("finger_cfg", self.finger_cfg)
        self.thumb_cfg = _ensure_resolved_finger_cfg("thumb_cfg", self.thumb_cfg)
        self.class_type = HumanLikeHandBuilder


@dataclass
class GripperLikeHandBuilderCfg(HandBuilderCfg):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    class_type: type[GripperLikeHandBuilder] | None = None
    finger_cfg: FingerBuilderCfg | dict[str, FingerBuilderCfg] | None = None
    num_fingers: int = 3
    mounts: dict[str, PoseCfg] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self.mounts = _to_pose_dict(self.mounts)
        self.class_type = GripperLikeHandBuilder


class HumanLikeHandBuilder(HandBuilder):
    "Builds a configured hand component from typed geometry and local frames."

    cfg: HumanLikeHandBuilderCfg

    def __init__(self, cfg: HumanLikeHandBuilderCfg):
        super().__init__(cfg)
        self.cfg = cfg

    def build(self) -> HandCfg:
        "Builds the configured geometry component from typed dimensions and local frames."

        if self.cfg.palm_cfg is None:
            raise ValueError("HumanLikeHandBuilder requires palm_cfg")
        if self.cfg.finger_cfg is None:
            raise ValueError("HumanLikeHandBuilder requires finger_cfg")

        palm_builder = self.cfg.palm_cfg.class_type(self.cfg.palm_cfg)
        palm = palm_builder.build()

        metadata_mounts = self._metadata_mounts(palm)
        explicit_mounts = {name: pose.copy() for name, pose in self.cfg.mounts.items()}

        mounts = {**self._fallback_mounts(palm), **metadata_mounts, **explicit_mounts}

        fingers = []
        if isinstance(self.cfg.finger_cfg, dict):
            items = list(self.cfg.finger_cfg.items())
        else:
            items = [(name, self.cfg.finger_cfg) for name in NON_THUMB_FINGER_NAMES[: self.cfg.num_non_thumb]]

        for finger_name, finger_cfg in items:
            built = self._build_named_finger(finger_cfg, finger_name, mounts.get(finger_name, PoseCfg()))
            fingers.append(built)

        if self.cfg.thumb_cfg is not None:
            thumb_mount = mounts.get("thumb", PoseCfg())
            fingers.append(self._build_named_finger(self.cfg.thumb_cfg, "thumb", thumb_mount))

        metadata = {"builder": "HumanLikeHandBuilder"}
        if self.cfg.palm_cfg.wrist_joints:
            # Question:



            metadata["wrist_joints"] = [joint.to_dict() for joint in self.cfg.palm_cfg.wrist_joints]

        canonical_hand = HandCfg(
            name=self.cfg.name,
            family=self.cfg.family,
            handedness="right",
            palm=palm,
            fingers=fingers,
            metadata=metadata,
        )
        return lower_hand_to_handedness(canonical_hand, self.cfg.handedness)

    def _build_named_finger(self, finger_cfg: FingerBuilderCfg, finger_name: str, mount: PoseCfg):

        if not hasattr(finger_cfg, "replace"):
            raise TypeError(f"Finger cfg {finger_cfg!r} is not a dataclass-backed config")

        updates = {"name": finger_name}
        if isinstance(finger_cfg, RegularFingerBuilderCfg):
            updates["parent_link"] = "palm"
        built_cfg = finger_cfg.replace(**updates)
        finger_builder = built_cfg.class_type(built_cfg)
        finger = finger_builder.build()
        return finger.replace(name=finger_name, mount=mount, parent_link="palm")

    def _metadata_mounts(self, palm) -> dict[str, PoseCfg]:

        if not isinstance(palm.metadata, dict):
            return {}
        finger_mounts = palm.metadata.get("finger_mounts")
        if not isinstance(finger_mounts, dict):
            return {}
        return _to_pose_dict(finger_mounts)

    def _fallback_mounts(self, palm) -> dict[str, PoseCfg]:

        # Fallback anchors follow the right-hand convention; handedness lowering reflects the complete hand later.

        if isinstance(self.cfg.palm_cfg, SinglePalmBuilderCfg) and self.cfg.palm_cfg.shape == "box":
            width = float(self.cfg.palm_cfg.width)
            length = float(self.cfg.palm_cfg.length)
            height = float(self.cfg.palm_cfg.height)
            names = NON_THUMB_FINGER_NAMES[: self.cfg.num_non_thumb]
            if len(names) == 1:
                xs = [0.0]
            else:

                half_span = width * 0.35
                step = 2.0 * half_span / max(len(names) - 1, 1)
                xs = [half_span - idx * step for idx in range(len(names))]
            mounts = {
                name: PoseCfg(pos=(x, length, height / 2.0))
                for name, x in zip(names, xs)
            }

            thumb_x = width * 0.22
            mounts["thumb"] = PoseCfg(
                pos=(thumb_x, length * 0.33, -height * 0.15),
                rpy=(0.0, 0.0, -1.5707963267948966),  # canonical right-hand thumb yaw
            )
            return mounts
        return {name: PoseCfg() for name in (*NON_THUMB_FINGER_NAMES[: self.cfg.num_non_thumb], "thumb")}


class GripperLikeHandBuilder(HandBuilder):
    "Builds a configured hand component from typed geometry and local frames."

    cfg: GripperLikeHandBuilderCfg

    def __init__(self, cfg: GripperLikeHandBuilderCfg):
        super().__init__(cfg)
        self.cfg = cfg

    def build(self) -> HandCfg:
        raise NotImplementedError("GripperLikeHandBuilder is intentionally out of scope for the first pre-made slice.")


__all__ = [
    "NON_THUMB_FINGER_NAMES",
    "HumanLikeHandBuilderCfg",
    "GripperLikeHandBuilderCfg",
    "HumanLikeHandBuilder",
    "GripperLikeHandBuilder",
]
