r"""Hand-level builders for human-like and gripper-like assemblies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..asset_base import HandCfg
from ..asset_builders import FingerBuilderCfg, HandBuilder, HandBuilderCfg
from ..asset_schema_core import PoseCfg
from .finger_buiders import RegularFingerBuilderCfg
from .palm_builders import SinglePalmBuilderCfg


NON_THUMB_FINGER_NAMES: tuple[str, ...] = ("index", "middle", "ring", "little")


def _to_pose_dict(values: dict[str, PoseCfg]) -> dict[str, PoseCfg]:
    return {name: PoseCfg.from_value(value) for name, value in values.items()}


@dataclass
class HumanLikeHandBuilderCfg(HandBuilderCfg):
    r"""Builder cfg for human-like dexterous hands."""

    class_type: type["HumanLikeHandBuilder"] | None = None
    handedness: Literal["left", "right"] = "right"
    finger_cfg: FingerBuilderCfg | dict[str, FingerBuilderCfg] | None = None
    thumb_cfg: FingerBuilderCfg | None = None
    num_non_thumb: int = 3
    mounts: dict[str, PoseCfg] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self.mounts = _to_pose_dict(self.mounts)
        if isinstance(self.finger_cfg, dict):
            invalid = set(self.finger_cfg) - set(NON_THUMB_FINGER_NAMES)
            if invalid:
                raise ValueError(f"finger_cfg dict keys must be drawn from {NON_THUMB_FINGER_NAMES}, got {invalid}")
            self.num_non_thumb = len(self.finger_cfg)
        elif self.finger_cfg is not None and not 1 <= self.num_non_thumb <= len(NON_THUMB_FINGER_NAMES):
            raise ValueError(f"num_non_thumb must be in [1, {len(NON_THUMB_FINGER_NAMES)}]")
        self.class_type = HumanLikeHandBuilder


@dataclass
class GripperLikeHandBuilderCfg(HandBuilderCfg):
    r"""Placeholder cfg for future gripper-like hand builders."""

    class_type: type["GripperLikeHandBuilder"] | None = None
    finger_cfg: FingerBuilderCfg | dict[str, FingerBuilderCfg] | None = None
    num_fingers: int = 3
    mounts: dict[str, PoseCfg] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self.mounts = _to_pose_dict(self.mounts)
        self.class_type = GripperLikeHandBuilder


class HumanLikeHandBuilder(HandBuilder):
    r"""Assemble a human-like hand from one palm and multiple finger builders."""

    cfg: HumanLikeHandBuilderCfg

    def __init__(self, cfg: HumanLikeHandBuilderCfg):
        super().__init__(cfg)
        self.cfg = cfg

    def build(self) -> HandCfg:
        r"""Build a human-like hand into canonical ``HandCfg`` form."""

        if self.cfg.palm_cfg is None:
            raise ValueError("HumanLikeHandBuilder requires palm_cfg")
        if self.cfg.finger_cfg is None:
            raise ValueError("HumanLikeHandBuilder requires finger_cfg")

        palm_builder = self.cfg.palm_cfg.class_type(self.cfg.palm_cfg)
        palm = palm_builder.build()

        preset_mounts = {
            name: PoseCfg.from_value(value)
            for name, value in palm.metadata.get("finger_mounts", {}).items()
        }
        mounts = {**self._fallback_mounts(palm), **preset_mounts, **self.cfg.mounts}

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
            # ``wrist_joints`` is preserved at the configuration boundary, but the
            # current ``HandCfg`` canonical structure has no wrist chain slot yet.
            # We therefore keep the declaration in metadata for later lowering.
            metadata["wrist_joints"] = [joint.to_dict() for joint in self.cfg.palm_cfg.wrist_joints]

        return HandCfg(
            name=self.cfg.name,
            family=self.cfg.family,
            handedness=self.cfg.handedness,
            palm=palm,
            fingers=fingers,
            metadata=metadata,
        )

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

    def _fallback_mounts(self, palm) -> dict[str, PoseCfg]:
        r"""Approximate human-like mounts when neither explicit nor preset mounts exist."""

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
            thumb_x = width * 0.22 if self.cfg.handedness == "right" else -width * 0.22
            mounts["thumb"] = PoseCfg(
                pos=(thumb_x, length * 0.33, -height * 0.15),
                rpy=(0.0, 0.0, -1.5707963267948966 if self.cfg.handedness == "right" else 1.5707963267948966),
            )
            return mounts
        return {name: PoseCfg() for name in (*NON_THUMB_FINGER_NAMES[: self.cfg.num_non_thumb], "thumb")}


class GripperLikeHandBuilder(HandBuilder):
    r"""Placeholder runtime for future gripper-like embodiments."""

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
