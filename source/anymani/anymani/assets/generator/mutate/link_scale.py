"""Link length post-mutation.

The scientific contract follows `doc/长度变异示意.jpg`: changing a link length
changes the link's own effective length, but does not scale its mesh offset.
The next joint/tip origin is then advanced by the new effective length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ...asset_base import HandCfg
from ...asset_schema_core import PoseCfg
from ._base import HandPatch, MutatorBase, MutatorBaseCfg
from ._distribution import ScalarDistributionCfg, normalize_distribution


@dataclass
class LinkScaleCfg(MutatorBaseCfg):
    """Configuration for continuous link length mutation."""

    class_type: type["LinkScaleMutator"] | None = None
    target_joints: tuple[str, ...] | None = None
    scale_mode: Literal["relative", "absolute"] = "relative"
    delta_distribution: Any = None
    clip_ratio: float | None = None

    # Legacy research-facing aliases kept for recipe compatibility.
    link_type: str = "box"
    scale_type: Literal["abs", "rel"] | None = None
    link_scale: Any = None
    clip: Any = None
    distrib: Any = "uniform"
    boundary_policy: Literal["none", "clip", "truncate", "resample"] | None = None

    def __post_init__(self) -> None:
        self.class_type = LinkScaleMutator
        if isinstance(self.target_joints, list):
            self.target_joints = tuple(self.target_joints)
        if self.scale_type is not None:
            self.scale_mode = "absolute" if self.scale_type == "abs" else "relative"
        if self.delta_distribution is None:
            if isinstance(self.link_scale, tuple) and len(self.link_scale) == 2:
                low, high = self.link_scale
                self.delta_distribution = ScalarDistributionCfg(kind="uniform", low=float(low), high=float(high))
            else:
                self.delta_distribution = ScalarDistributionCfg(kind="normal", mean=0.0, sigma=0.03)
        else:
            self.delta_distribution = normalize_distribution(self.delta_distribution)


class LinkScaleMutator(MutatorBase):
    """Generate/apply patches that scale selected child-link effective lengths."""

    cfg: LinkScaleCfg

    def __init__(self, cfg: LinkScaleCfg):
        self.cfg = cfg

    def describe_sampling(self, target: HandCfg) -> dict[str, Any]:
        return {
            joint.name: self.cfg.delta_distribution
            for _, _, joint in _iter_target_joints(target, self.cfg.target_joints)
        }

    def plan_patch(self, target: HandCfg, sampled_params: dict[str, Any] | None = None) -> HandPatch:
        sampled_params = sampled_params or {}
        patch = HandPatch()

        for finger_index, joint_index, joint in _iter_target_joints(target, self.cfg.target_joints):
            delta = float(sampled_params.get(joint.name, 0.0))
            old_length = _joint_primary_length(joint)
            if old_length is None:
                continue
            new_length = _mutated_length(old_length, delta, self.cfg)
            if new_length <= 1e-6:
                continue
            length_delta = new_length - old_length
            is_cmc1 = str(joint.child).endswith("_cmc1")

            def apply_link(hand: HandCfg, *, fi=finger_index, ji=joint_index, old=old_length, new=new_length, cmc1=is_cmc1) -> None:
                mutated_joint = hand.fingers[fi].joints[ji]
                _set_joint_primary_length(mutated_joint, old_length=old, new_length=new, keep_center=cmc1)

            patch.add(("finger", finger_index, "joint", joint_index, "link_length"), apply_link)

            next_index = joint_index + 1
            if next_index < len(target.fingers[finger_index].joints):
                advance_delta = length_delta * 0.5 if is_cmc1 else length_delta

                def apply_next_origin(hand: HandCfg, *, fi=finger_index, ni=next_index, dy=advance_delta) -> None:
                    next_joint = hand.fingers[fi].joints[ni]
                    pos = next_joint.origin.pos
                    next_joint.origin = PoseCfg(pos=(pos[0], pos[1] + dy, pos[2]), rpy=next_joint.origin.rpy)

                patch.add(("finger", finger_index, "joint", next_index, "origin_from_link_scale", joint.name), apply_next_origin)

        return patch


def _iter_target_joints(hand: HandCfg, target_joints: tuple[str, ...] | None):
    target_set = set(target_joints or ())
    for finger_index, finger in enumerate(hand.fingers):
        for joint_index, joint in enumerate(finger.joints):
            if joint.joint_type != "revolute" or joint.is_tip:
                continue
            if target_set and joint.name not in target_set:
                continue
            if _joint_primary_length(joint) is None:
                continue
            yield finger_index, joint_index, joint


def _joint_primary_length(joint) -> float | None:
    geometry = None
    if joint.collisions:
        geometry = joint.collisions[0].geometry
    elif joint.visuals:
        geometry = joint.visuals[0].geometry
    if geometry is None:
        return None
    if geometry.kind == "box":
        return float(geometry.size[1])
    if geometry.kind == "cylinder":
        return float(geometry.length)
    return None


def _mutated_length(old_length: float, delta: float, cfg: LinkScaleCfg) -> float:
    if cfg.scale_mode == "relative":
        if cfg.clip_ratio is not None:
            delta = max(-float(cfg.clip_ratio), min(float(cfg.clip_ratio), delta))
        return old_length * (1.0 + delta)
    if cfg.clip_ratio is not None:
        delta = max(-old_length * float(cfg.clip_ratio), min(old_length * float(cfg.clip_ratio), delta))
    return old_length + delta


def _set_joint_primary_length(joint, *, old_length: float, new_length: float, keep_center: bool) -> None:
    for collection_name in ("collisions", "visuals"):
        collection = getattr(joint, collection_name)
        for index, element in enumerate(collection):
            geometry = element.geometry
            if geometry.kind == "box":
                size = geometry.size
                geometry = geometry.replace(size=(size[0], new_length, size[2]))
            elif geometry.kind == "cylinder":
                geometry = geometry.replace(length=new_length)
            else:
                continue

            origin = element.origin
            if keep_center:
                new_origin = origin.copy()
            else:
                offset_y = origin.pos[1] - old_length / 2.0
                new_origin = PoseCfg(
                    pos=(origin.pos[0], new_length / 2.0 + offset_y, origin.pos[2]),
                    rpy=origin.rpy,
                )
            collection[index] = element.replace(geometry=geometry, origin=new_origin)
            if joint.inertial is not None and index == 0:
                joint.inertial = joint.inertial.replace(origin=new_origin)


__all__ = ["LinkScaleCfg", "LinkScaleMutator"]
