"Samples per-link length scales and shared width and height scales. Vector6 sampling and both transverse-scaling paths are implemented; thumb CMC1 uses a separate CMC2-origin update. Splitting the deferred patch into three layers remains a follow-up."

from __future__ import annotations

import math
import random
from dataclasses import MISSING, dataclass, field
from typing import Any, Literal

from ...asset_base import HandCfg
from ...asset_schema_core import PoseCfg, Vector2, Vector6
from .axial_geometry import (
    AxialGeometryEdit,
    make_axial_geometry_patch_op,
)
from .axial_geometry import (
    joint_cross_section as _joint_cross_section,
)
from .axial_geometry import (
    joint_primary_length as _joint_primary_length,
)
from .base import HandPatch, MutatorBase, MutatorBaseCfg, _make_range_sampler

_MODE_IDENTITY = "identity"
_MODE_GENERAL = "general"
_MODE_ONLY_LENGTH = "only_length"

_ALL_SELF_MODES = (
    _MODE_IDENTITY,
    _MODE_GENERAL,
    _MODE_ONLY_LENGTH,
)
"Allowed whole-hand proposal modes for this operator."

_MODE_TOLERANCE = 1e-9
"Numerical tolerance for proposal probability sums."


@dataclass
class LinkScaleCfg(MutatorBaseCfg):
    "Proposal settings for per-link length scales and candidate-shared width and height scales. CMC1 thumb geometry has a separate CMC2-origin rule."

    # DONE(link-scale-vector6): Samples independent per-link lengths and candidate-shared width and height scales.
    # DONE(link-scale-nonthumb-width-height): Scales non-thumb cross-sections without shifting joint or mesh origins.
    # DONE(link-scale-thumb-cmc1): Recomputes the CMC2 origin from the revised CMC1 dimensions.

    class_type: type[LinkScaleMutator] | None = field(init=False, default=None, repr=False)
    "Associated runtime implementation for this configuration class."

    link_scale: Vector2 | Vector6 = field(default=MISSING)
    "Dimensionless length and optional cross-section scale ranges."

    self_mode: Literal["identity", "general", "only_length"] | dict[str, float] | None = _MODE_GENERAL
    "Identity, shared, or per-finger mutation proposal mode."

    link_type: str = "box"
    "Primitive or custom mesh geometry used for the child link."

    scale_type: Literal["abs", "rel"] = "rel"
    "Whether declared scales are relative factors or absolute dimensions."

    clip: Vector2 | Vector6 | None = None
    "Optional legal interval for sampled geometry parameters."

    distrib: Literal["uniform", "normal"] | dict[str, Any] = "uniform"
    "Proposal distribution used inside the declared parameter interval."

    boundary_policy: Literal["none", "clip", "truncate", "resample"] | None = None
    "Action for proposals outside their permitted domain."

    _link_meshes: list[Any] = field(default_factory=list)
    "Source mesh records used to transform link-local geometry."

    _distribution: Any = field(init=False, repr=False)
    "Prepared random sampler created from the typed proposal config."

    def __post_init__(self) -> None:
        "Normalizes typed inputs and enforces the class invariants."

        self.class_type = LinkScaleMutator
        _resolve_active_modes(self.self_mode)
        if self.link_scale is MISSING:
            raise ValueError("LinkScaleCfg.link_scale must be set explicitly")
        if isinstance(self.link_scale, dict):
            raise TypeError("LinkScaleCfg.link_scale no longer supports dict[str, ...]; use Vector2 or Vector6 only")
        if isinstance(self.clip, dict):
            raise TypeError("LinkScaleCfg.clip no longer supports dict[str, ...]; use Vector2, Vector6 or None only")


class LinkScaleMutator(MutatorBase):
    "Applies a typed geometry proposal to a hand copy and records sampled values."

    cfg: LinkScaleCfg

    def __init__(self, cfg: LinkScaleCfg):
        "Stores the per-link scale config and prepares its proposal sampler."

        self.cfg = cfg

    def describe_sampling(self, target: HandCfg) -> dict[str, Any]:
        'Summarizes proposal settings and sample outcomes.'

        return {"sample": lambda: self._sample_one(target)}

    def plan_patch(self, target: HandCfg, sampled_params: dict[str, Any] | None = None) -> HandPatch:
        "Plans one deferred patch from the original hand. Keep local size updates, ordinary downstream shifts, and the thumb CMC1-to-CMC2 origin update separate."

        # TODO(link-scale-vector6): Keep local dimensions, regular downstream shifts, and CMC1 re-resolution separate while reading one original HandCfg.

        sample = self._sample_one(target) if sampled_params is None else _normalize_sample_payload(sampled_params, self.cfg, target=target)
        resolved_mode = str(sample["resolved_self_mode"])

        patch = HandPatch()
        patch.metadata.setdefault("post_mutate_samples", {})
        patch.metadata["post_mutate_samples"]["link_scale"] = sample
        patch.metadata["post_mutate_link_scale"] = sample

        if resolved_mode == _MODE_IDENTITY:
            return patch

        shared_width = sample.get("width_scale")
        shared_height = sample.get("height_scale")
        child_length_scales = dict(sample.get("length_scale", {}))




        for finger_index, joint_index, joint in _iter_target_joints(target):
            delta_or_ratio = _length_sample_for_joint(sample, joint, default=0.0)
            old_length = _joint_primary_length(joint)
            if old_length is None:
                continue
            new_length = _mutated_length(old_length, delta_or_ratio, self.cfg)
            if new_length <= 1e-6:
                continue
            child_length_scales[joint.child] = delta_or_ratio

            old_cross_section = _joint_cross_section(joint)
            new_cross_section = _mutated_cross_section(
                old_cross_section,
                width_scale=_semantic_width_scale_for_joint(joint, semantic_width_scale=shared_width, semantic_height_scale=shared_height),
                height_scale=_semantic_height_scale_for_joint(joint, semantic_width_scale=shared_width, semantic_height_scale=shared_height),
            )
            is_cmc1 = str(joint.child).endswith("_cmc1")



            patch.add_op(
                make_axial_geometry_patch_op(
                    AxialGeometryEdit(
                        finger_index=finger_index,
                        joint_index=joint_index,
                        joint_name=joint.name,
                        child_link=str(joint.child),
                        source_length=old_length,
                        source_cross_section=old_cross_section,
                        keep_center=is_cmc1,
                        scaled_length=new_length,
                        scaled_cross_section=new_cross_section,
                    )
                )
            )

            next_index = joint_index + 1
            if next_index < len(target.fingers[finger_index].joints):
                next_joint = target.fingers[finger_index].joints[next_index]
                next_old_cross_section = _joint_cross_section(next_joint)
                next_new_cross_section = _mutated_cross_section(
                    next_old_cross_section,
                    width_scale=_semantic_width_scale_for_joint(
                        next_joint,
                        semantic_width_scale=shared_width,
                        semantic_height_scale=shared_height,
                    ),
                    height_scale=_semantic_height_scale_for_joint(
                        next_joint,
                        semantic_width_scale=shared_width,
                        semantic_height_scale=shared_height,
                    ),
                )
                next_origin = _next_origin_from_link_scale(
                    current_joint=joint,
                    next_joint=next_joint,
                    old_length=old_length,
                    new_length=new_length,
                    old_cross_section=old_cross_section,
                    new_cross_section=new_cross_section,
                    next_old_cross_section=next_old_cross_section,
                    next_new_cross_section=next_new_cross_section,
                )

                def apply_next_origin(hand: HandCfg, *, fi=finger_index, ni=next_index, origin=next_origin) -> None:
                    r"""Recomputes downstream joint origins from the updated geometry bounds."""

                    hand.fingers[fi].joints[ni].origin = origin

                patch.add(("finger", finger_index, "joint", next_index, "origin_from_link_scale", joint.name), apply_next_origin)

        sample["length_scale"] = child_length_scales
        patch.metadata["post_mutate_samples"]["link_scale"] = sample
        patch.metadata["post_mutate_link_scale"] = sample
        return patch

    def _sample_one(self, target: HandCfg) -> dict[str, Any]:

        resolved_mode = _draw_resolved_mode(self.cfg)
        return self.sample_one_for_mode(target, resolved_mode=resolved_mode)

    def sample_one_for_mode(self, target: HandCfg, *, resolved_mode: str) -> dict[str, Any]:
        'Samples parameters for the resolved mutation mode.'

        if resolved_mode not in _ALL_SELF_MODES:
            raise ValueError(f"unsupported link_scale resolved mode: {resolved_mode!r}")
        if resolved_mode == _MODE_IDENTITY:
            return {
                "resolved_self_mode": _MODE_IDENTITY,
                "joint_length_scale": {},
                "length_scale": {},
                "width_scale": None,
                "height_scale": None,
            }

        joint_length_scale = _sample_joint_length_scales(self.cfg, target)
        width_scale = None
        height_scale = None
        if resolved_mode == _MODE_GENERAL:
            shared_ranges = _shared_cross_section_ranges(self.cfg.link_scale, self.cfg.clip, target=target)
            if shared_ranges is not None:
                width_range, height_range = shared_ranges
                width_scale = _make_range_sampler(
                    width_range,
                    distrib=self.cfg.distrib,
                    boundary_policy=self.cfg.boundary_policy,
                )()
                height_scale = _make_range_sampler(
                    height_range,
                    distrib=self.cfg.distrib,
                    boundary_policy=self.cfg.boundary_policy,
                )()

        return {
            "resolved_self_mode": resolved_mode,
            "joint_length_scale": joint_length_scale,
            "length_scale": {},
            "width_scale": width_scale,
            "height_scale": height_scale,
        }


def _iter_target_joints(hand: HandCfg):



    for finger_index, finger in enumerate(hand.fingers):
        for joint_index, joint in enumerate(finger.joints):
            if joint.joint_type != "revolute" or joint.is_tip:
                continue
            if _joint_primary_length(joint) is None:
                continue
            yield finger_index, joint_index, joint


def _resolve_active_modes(self_mode: Any) -> tuple[str, ...]:

    if self_mode is None:
        return (_MODE_GENERAL,)
    if isinstance(self_mode, str):
        if self_mode not in _ALL_SELF_MODES:
            raise ValueError(f"unsupported link_scale self_mode: {self_mode!r}")
        return (self_mode,)
    if not isinstance(self_mode, dict):
        raise TypeError(f"link_scale.self_mode must be str | dict[str, float] | None, got {type(self_mode).__name__}")

    active: list[str] = []
    total = 0.0
    for mode_name, probability in self_mode.items():
        if mode_name not in _ALL_SELF_MODES:
            raise ValueError(f"unsupported link_scale self_mode key: {mode_name!r}")
        prob = float(probability)
        if prob < 0.0:
            raise ValueError(f"link_scale.self_mode probability must be non-negative, got {mode_name!r}={prob!r}")
        total += prob
        if prob > _MODE_TOLERANCE:
            active.append(str(mode_name))
    if not active:
        raise ValueError("link_scale.self_mode dict must contain at least one positive-probability mode")
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=_MODE_TOLERANCE):
        raise ValueError(f"link_scale.self_mode probabilities must sum to 1.0, got {total!r}")
    return tuple(active)


def _draw_resolved_mode(cfg: LinkScaleCfg) -> str:

    self_mode = cfg.self_mode
    if self_mode is None:
        return _MODE_GENERAL
    if isinstance(self_mode, str):
        return self_mode

    threshold = random.random()
    cumulative = 0.0
    last_positive = _MODE_GENERAL
    for mode in _ALL_SELF_MODES:
        probability = float(self_mode.get(mode, 0.0))
        if probability <= _MODE_TOLERANCE:
            continue
        cumulative += probability
        last_positive = mode
        if threshold <= cumulative:
            return mode
    return last_positive


def _sample_joint_length_scales(cfg: LinkScaleCfg, target: HandCfg) -> dict[str, float]:

    joint_length_scale: dict[str, float] = {}
    for _, _, joint in _iter_target_joints(target):
        value_range = _range_for_joint(cfg.link_scale, joint.child)
        clip_range = _range_for_joint(cfg.clip, joint.child) if cfg.clip is not None else None
        joint_length_scale[joint.name] = _make_range_sampler(
            _length_range(value_range, clip_range),
            distrib=cfg.distrib,
            boundary_policy=cfg.boundary_policy,
        )()
    return joint_length_scale


def _normalize_sample_payload(
    sampled_params: dict[str, Any] | None,
    cfg: LinkScaleCfg,
    *,
    target: HandCfg,
) -> dict[str, Any]:

    params = dict(sampled_params or {})
    raw_sample = params.get("sample")
    if isinstance(raw_sample, dict):
        sample = dict(raw_sample)
    elif "resolved_self_mode" in params:
        sample = dict(params)
    else:

        joint_length_scale = {
            key.removesuffix("::length"): float(value)
            for key, value in params.items()
            if isinstance(key, str) and key.endswith("::length")
        }
        for _, _, joint in _iter_target_joints(target):
            if joint.name in params:
                joint_length_scale[joint.name] = float(params[joint.name])
        sample = {
            "resolved_self_mode": _MODE_GENERAL,
            "joint_length_scale": joint_length_scale,
            "length_scale": {},
            "width_scale": float(params["shared::width"]) if "shared::width" in params else None,
            "height_scale": float(params["shared::height"]) if "shared::height" in params else None,
        }

    default_mode = cfg.self_mode if isinstance(cfg.self_mode, str) else _MODE_GENERAL
    resolved_mode = str(sample.get("resolved_self_mode", default_mode))
    if resolved_mode not in _ALL_SELF_MODES:
        raise ValueError(f"unsupported link_scale resolved mode: {resolved_mode!r}")
    sample["resolved_self_mode"] = resolved_mode
    sample.setdefault("joint_length_scale", {})
    sample.setdefault("length_scale", {})
    if resolved_mode == _MODE_IDENTITY:
        sample["joint_length_scale"] = {}
        sample["length_scale"] = {}
        sample["width_scale"] = None
        sample["height_scale"] = None
    elif resolved_mode == _MODE_ONLY_LENGTH:
        sample["width_scale"] = None
        sample["height_scale"] = None
    else:
        sample.setdefault("width_scale", None)
        sample.setdefault("height_scale", None)
    return sample


def _length_sample_for_joint(sample: dict[str, Any], joint, *, default: float) -> float:

    joint_length_scale = sample.get("joint_length_scale")
    if isinstance(joint_length_scale, dict):
        if joint.name in joint_length_scale:
            return float(joint_length_scale[joint.name])
        if joint.child in joint_length_scale:
            return float(joint_length_scale[joint.child])
    return float(default)


def _range_for_joint(config: Vector2 | Vector6 | None, child_name: str) -> Vector2 | Vector6 | None:

    _ = child_name
    return config


def _length_range(value_range: Vector2 | Vector6, clip_range: Vector2 | Vector6 | None) -> Vector2:

    low = float(value_range[0])
    high = float(value_range[1])
    if clip_range is not None:
        clip_low = float(clip_range[0])
        clip_high = float(clip_range[1])
        low = max(low, clip_low)
        high = min(high, clip_high)
    return (low, high)


def _shared_cross_section_ranges(
    value_range: Vector2 | Vector6,
    clip_range: Vector2 | Vector6 | None,
    *,
    target: HandCfg,
) -> tuple[Vector2, Vector2] | None:

    if len(value_range) == 2:
        return None
    if not any(True for _ in _iter_target_joints(target)):
        return None

    width_range = (float(value_range[2]), float(value_range[3]))
    height_range = (float(value_range[4]), float(value_range[5]))
    if clip_range is not None and len(clip_range) == 6:
        width_range = (
            max(width_range[0], float(clip_range[2])),
            min(width_range[1], float(clip_range[3])),
        )
        height_range = (
            max(height_range[0], float(clip_range[4])),
            min(height_range[1], float(clip_range[5])),
        )
    return width_range, height_range


def _is_thumb_joint(joint) -> bool:

    child_name = str(getattr(joint, "child", ""))
    return child_name.startswith("thumb_")


def _semantic_width_scale_for_joint(
    joint,
    *,
    semantic_width_scale: float | None,
    semantic_height_scale: float | None,
) -> float | None:

    if not _is_thumb_joint(joint):
        return semantic_width_scale
    return semantic_height_scale


def _semantic_height_scale_for_joint(
    joint,
    *,
    semantic_width_scale: float | None,
    semantic_height_scale: float | None,
) -> float | None:

    if not _is_thumb_joint(joint):
        return semantic_height_scale
    return semantic_width_scale


def _mutated_length(old_length: float, delta_or_ratio: float, cfg: LinkScaleCfg) -> float:



    if cfg.scale_type == "rel":
        return old_length * float(delta_or_ratio)
    return old_length + float(delta_or_ratio)


def _mutated_cross_section(
    old_cross_section: tuple[float, float] | None,
    *,
    width_scale: float | None,
    height_scale: float | None,
) -> tuple[float, float] | None:

    if old_cross_section is None:
        return None
    width, height = old_cross_section
    new_width = width if width_scale is None else width * float(width_scale)
    new_height = height if height_scale is None else height * float(height_scale)
    if new_width <= 1e-6 or new_height <= 1e-6:
        return old_cross_section
    return new_width, new_height


def _next_origin_from_link_scale(
    *,
    current_joint,
    next_joint,
    old_length: float,
    new_length: float,
    old_cross_section: tuple[float, float] | None,
    new_cross_section: tuple[float, float] | None,
    next_old_cross_section: tuple[float, float] | None,
    next_new_cross_section: tuple[float, float] | None,
) -> PoseCfg:

    if not str(current_joint.child).endswith("_cmc1"):
        delta_y = new_length - old_length
        return PoseCfg(
            pos=(next_joint.origin.pos[0], next_joint.origin.pos[1] + delta_y, next_joint.origin.pos[2]),
            rpy=next_joint.origin.rpy,
        )

    old_width = old_cross_section[0] if old_cross_section is not None else 0.0
    old_height = old_cross_section[1] if old_cross_section is not None else 0.0
    new_width = new_cross_section[0] if new_cross_section is not None else old_width
    new_height = new_cross_section[1] if new_cross_section is not None else old_height
    next_width = (
        next_new_cross_section[0] if next_new_cross_section is not None else (
            next_old_cross_section[0] if next_old_cross_section is not None else old_width
        )
    )
    next_height = (
        next_new_cross_section[1] if next_new_cross_section is not None else (
            next_old_cross_section[1] if next_old_cross_section is not None else old_height
        )
    )
    current_origin = current_joint.collisions[0].origin if current_joint.collisions else current_joint.visuals[0].origin
    return PoseCfg(
        pos=(
            (new_width - next_width) / 2.0,
            current_origin.pos[1] + new_length / 2.0,
            current_origin.pos[2] - (new_height - next_height) / 2.0,
        ),
        rpy=next_joint.origin.rpy,
    )


__all__ = ["LinkScaleCfg", "LinkScaleMutator"]
