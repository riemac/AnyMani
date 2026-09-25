"Perturbs joint limits within the configured domain while retaining the source joint identity."

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import math
import random
from typing import Any, Literal

from ...asset_base import HandCfg
from ...asset_schema_core import JointLimitCfg, Vector2
from .base import HandPatch, MutatorBase, MutatorBaseCfg, _make_range_sampler


_MODE_IDENTITY = "identity"
_MODE_DISTURB = "disturb"
_MODE_HOMOLOGOUS_NON_THUMB = "homologous_non_thumb"

_ALL_SELF_MODES = (
    _MODE_IDENTITY,
    _MODE_DISTURB,
    _MODE_HOMOLOGOUS_NON_THUMB,
)
"Allowed whole-hand proposal modes for this operator."

_MODE_TOLERANCE = 1e-9
"Numerical tolerance for proposal probability sums."


@dataclass
class LimitTweakCfg(MutatorBaseCfg):
    "Proposal settings for lower and upper revolute limits in radians."

    class_type: type["LimitTweakMutator"] | None = field(init=False, default=None, repr=False)
    "Associated runtime implementation for this configuration class."

    disturb_object: Literal["independent", "shared"] = "independent"
    "Sampling relationship for lower and upper limits within one joint."

    self_mode: Literal[
        "identity",
        "disturb",
        "homologous_non_thumb",
    ] | dict[str, float] | None = _MODE_DISTURB
    "Identity, shared, or per-finger mutation proposal mode."

    disturb_type: Literal["add", "scale"] = "add"
    "Additive or relative limit perturbation mode."

    joint_range: Vector2 | None = None
    "Perturbation interval for joint coordinates, in radians."

    clip: dict[str, float] | None = None
    "Optional legal interval for sampled geometry parameters."

    distrib: Literal["uniform", "normal"] | dict[str, Any] = "uniform"
    "Proposal distribution used inside the declared parameter interval."

    boundary_policy: Literal["none", "clip", "truncate", "resample"] | None = None
    "Action for proposals outside their permitted domain."

    _active_modes: tuple[str, ...] = field(init=False, default=(), repr=False)
    "Resolved proposal modes with positive probability."

    def __post_init__(self) -> None:
        "Normalizes typed inputs and enforces the class invariants."

        self.class_type = LimitTweakMutator
        self._active_modes = _resolve_active_modes(self.self_mode)
        if any(mode != _MODE_IDENTITY for mode in self._active_modes) and self.joint_range is None:
            raise ValueError("limit_tweak requires joint_range when any non-identity self_mode is active")


class LimitTweakMutator(MutatorBase):
    "Applies a typed geometry proposal to a hand copy and records sampled values."

    cfg: LimitTweakCfg

    def __init__(self, cfg: LimitTweakCfg):
        self.cfg = cfg

    def describe_sampling(self, target: HandCfg) -> dict[str, Any]:
        'Summarizes proposal settings and sample outcomes.'

        if self.cfg.self_mode == _MODE_IDENTITY and self.cfg.joint_range is None:
            return {"sample": lambda: {"resolved_self_mode": _MODE_IDENTITY}}
        return {"sample": lambda: self._sample_one(target)}

    def plan_patch(self, target: HandCfg, sampled_params: dict[str, Any] | None = None) -> HandPatch:
        "Plans lower and upper joint-limit edits in radians while retaining source topology."

        sample = _normalize_sample_payload(sampled_params, self.cfg, target=target)
        resolved_mode = str(sample["resolved_self_mode"])

        patch = HandPatch()
        patch.metadata.setdefault("post_mutate_samples", {})
        patch.metadata["post_mutate_samples"]["limit_tweak"] = sample
        patch.metadata["post_mutate_limit_tweak"] = sample

        if resolved_mode == _MODE_IDENTITY:
            return patch

        joint_deltas = dict(sample.get("joint_deltas", {}))
        for finger_index, joint_index, joint in _iter_target_joints(target, None):
            payload = dict(joint_deltas.get(joint.name, {}))
            lower_delta = _clip_delta(float(payload.get("lower", 0.0)), self.cfg.clip)
            upper_delta = _clip_delta(float(payload.get("upper", 0.0)), self.cfg.clip)

            def apply_limit(
                hand: HandCfg,
                *,
                fi=finger_index,
                ji=joint_index,
                dl=lower_delta,
                du=upper_delta,
            ) -> None:
                current = hand.fingers[fi].joints[ji].limit
                if current is None:
                    return



                if self.cfg.disturb_type == "scale":
                    lower = current.lower * (1.0 + dl)
                    upper = current.upper * (1.0 + (dl if self.cfg.disturb_object == "shared" else du))
                elif self.cfg.disturb_object == "shared":
                    lower = current.lower + dl
                    upper = current.upper + dl
                else:
                    lower = current.lower + dl
                    upper = current.upper + du


                if lower >= upper:
                    center = 0.5 * (lower + upper)
                    lower, upper = center - 1e-4, center + 1e-4

                hand.fingers[fi].joints[ji].limit = JointLimitCfg(
                    lower=lower,
                    upper=upper,
                    effort=current.effort,
                    velocity=current.velocity,
                )

            patch.add(("finger", finger_index, "joint", joint_index, "limit"), apply_limit)
        return patch

    def _sample_one(self, target: HandCfg) -> dict[str, Any]:

        resolved_mode = _draw_resolved_mode(self.cfg)
        return self.sample_one_for_mode(target, resolved_mode=resolved_mode)

    def sample_one_for_mode(self, target: HandCfg, *, resolved_mode: str) -> dict[str, Any]:
        'Samples parameters for the resolved mutation mode.'

        if resolved_mode not in _ALL_SELF_MODES:
            raise ValueError(f"unsupported limit_tweak resolved mode: {resolved_mode!r}")
        if resolved_mode == _MODE_IDENTITY:
            return {"resolved_self_mode": _MODE_IDENTITY}

        sampler = _make_range_sampler(
            self.cfg.joint_range,
            distrib=self.cfg.distrib,
            boundary_policy=self.cfg.boundary_policy,
        )

        if resolved_mode == _MODE_DISTURB:
            joint_deltas = {
                joint.name: _sample_delta_pair(sampler, disturb_object=self.cfg.disturb_object)
                for _, _, joint in _iter_target_joints(target, None)
            }
            return {
                "resolved_self_mode": _MODE_DISTURB,
                "joint_deltas": joint_deltas,
            }

        groups = _resolve_homologous_non_thumb_groups(target)
        joint_deltas: dict[str, dict[str, float]] = {}
        homologous_groups: OrderedDict[str, dict[str, Any]] = OrderedDict()
        for group_key, joint_names in groups.items():
            sampled = _sample_delta_pair(sampler, disturb_object=self.cfg.disturb_object)
            homologous_groups[group_key] = {
                "joint_names": list(joint_names),
                "lower": sampled["lower"],
                "upper": sampled["upper"],
            }
            for joint_name in joint_names:
                joint_deltas[joint_name] = dict(sampled)


        thumb_joint_deltas: OrderedDict[str, dict[str, float]] = OrderedDict()
        for _, _, joint in _iter_target_joints(target, None):
            if not joint.name.startswith("thumb_"):
                continue
            sampled = _sample_delta_pair(sampler, disturb_object=self.cfg.disturb_object)
            thumb_joint_deltas[joint.name] = sampled
            joint_deltas[joint.name] = dict(sampled)

        return {
            "resolved_self_mode": _MODE_HOMOLOGOUS_NON_THUMB,
            "joint_deltas": joint_deltas,
            "homologous_groups": dict(homologous_groups),
            "thumb_joint_deltas": dict(thumb_joint_deltas),
        }


def _resolve_active_modes(self_mode: str | dict[str, float] | None) -> tuple[str, ...]:

    if self_mode is None:
        return (_MODE_DISTURB,)
    if isinstance(self_mode, str):
        if self_mode not in _ALL_SELF_MODES:
            raise ValueError(f"unsupported limit_tweak self_mode: {self_mode!r}")
        return (self_mode,)

    positive_modes: list[str] = []
    total = 0.0
    for mode_name, probability in self_mode.items():
        if mode_name not in _ALL_SELF_MODES:
            raise ValueError(f"unsupported limit_tweak self_mode key: {mode_name!r}")
        prob = float(probability)
        if prob < 0.0:
            raise ValueError(f"limit_tweak.self_mode probability must be non-negative, got {mode_name!r}={prob!r}")
        total += prob
        if prob > _MODE_TOLERANCE:
            positive_modes.append(mode_name)

    if not positive_modes:
        raise ValueError("limit_tweak.self_mode dict must contain at least one positive-probability mode")
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=_MODE_TOLERANCE):
        raise ValueError(f"limit_tweak.self_mode probabilities must sum to 1, got {total!r}")
    return tuple(positive_modes)


def _draw_resolved_mode(cfg: LimitTweakCfg) -> str:

    if cfg.self_mode is None:
        return _MODE_DISTURB
    if isinstance(cfg.self_mode, str):
        return cfg.self_mode

    threshold = random.random()
    cumulative = 0.0
    last_mode = _MODE_DISTURB
    for mode_name, probability in cfg.self_mode.items():
        prob = float(probability)
        if prob <= _MODE_TOLERANCE:
            continue
        cumulative += prob
        last_mode = mode_name
        if threshold <= cumulative + _MODE_TOLERANCE:
            return mode_name
    return last_mode


def _normalize_sample_payload(
    sampled_params: dict[str, Any] | None,
    cfg: LimitTweakCfg,
    *,
    target: HandCfg,
) -> dict[str, Any]:

    sampled = dict(sampled_params or {})
    sample = sampled.get("sample")
    if isinstance(sample, dict):
        return dict(sample)

    if cfg.self_mode == _MODE_IDENTITY:
        return {"resolved_self_mode": _MODE_IDENTITY}

    joint_deltas: dict[str, dict[str, float]] = {}
    for _, _, joint in _iter_target_joints(target, None):
        lower = float(sampled.get(f"{joint.name}::lower", sampled.get(joint.name, 0.0)))
        if cfg.disturb_object == "shared":
            upper = lower
        else:
            upper = float(sampled.get(f"{joint.name}::upper", 0.0))
        joint_deltas[joint.name] = {"lower": lower, "upper": upper}
    return {
        "resolved_self_mode": _MODE_DISTURB,
        "joint_deltas": joint_deltas,
    }


def _sample_delta_pair(sampler, *, disturb_object: str) -> dict[str, float]:

    lower = float(sampler())
    if disturb_object == "shared":
        return {"lower": lower, "upper": lower}
    return {"lower": lower, "upper": float(sampler())}


def _resolve_homologous_non_thumb_groups(target: HandCfg) -> OrderedDict[str, list[str]]:

    slot_family_map = _resolve_slot_family_map(target)
    groups: OrderedDict[str, list[str]] = OrderedDict()
    for finger in target.fingers:
        if finger.name == "thumb":
            continue
        family = slot_family_map.get(finger.name)
        if not isinstance(family, str) or not family:
            raise ValueError(
                "limit_tweak.homologous_non_thumb requires premade slot_family_map; "
                f"missing family for finger {finger.name!r}"
            )
        for joint in finger.joints:
            if joint.joint_type != "revolute" or joint.limit is None:
                continue
            semantic = _resolve_joint_semantic(finger_name=finger.name, child_link=str(joint.child))
            group_key = f"{family}:{semantic}"
            groups.setdefault(group_key, []).append(joint.name)
    return groups


def _resolve_slot_family_map(target: HandCfg) -> dict[str, str]:

    metadata = dict(target.metadata or {})
    premade_topology = metadata.get("premade_topology")
    if isinstance(premade_topology, dict):
        slot_family_map = premade_topology.get("slot_family_map")
        if isinstance(slot_family_map, dict):
            return {str(slot): str(family) for slot, family in slot_family_map.items()}
    premade_connectivity = metadata.get("premade_connectivity")
    if isinstance(premade_connectivity, dict):
        slot_family_map = premade_connectivity.get("slot_family_map")
        if isinstance(slot_family_map, dict):
            return {str(slot): str(family) for slot, family in slot_family_map.items()}
    raise ValueError("limit_tweak.homologous_non_thumb requires hand metadata with premade slot_family_map")


def _resolve_joint_semantic(*, finger_name: str, child_link: str) -> str:

    prefix = f"{finger_name}_"
    if not child_link.startswith(prefix):
        raise ValueError(
            "limit_tweak.homologous_non_thumb expects child link names to preserve anatomy suffix; "
            f"got finger={finger_name!r}, child_link={child_link!r}"
        )
    semantic = child_link[len(prefix) :]
    if not semantic:
        raise ValueError(
            "limit_tweak.homologous_non_thumb could not parse non-empty joint semantic suffix from "
            f"child_link={child_link!r}"
        )
    return semantic


def _iter_target_joints(hand: HandCfg, target_joints: tuple[str, ...] | None):

    target_set = set(target_joints or ())
    for finger_index, finger in enumerate(hand.fingers):
        for joint_index, joint in enumerate(finger.joints):
            if joint.joint_type != "revolute" or joint.limit is None:
                continue
            if target_set and joint.name not in target_set:
                continue
            yield finger_index, joint_index, joint


def _clip_delta(delta: float, clip: dict[str, float] | None) -> float:

    if clip is None:
        return delta
    if "abs" in clip:
        bound = abs(float(clip["abs"]))
    elif "rel" in clip:
        bound = abs(float(clip["rel"]))
    else:
        return delta
    return max(-bound, min(bound, delta))


__all__ = ["LimitTweakCfg", "LimitTweakMutator"]
