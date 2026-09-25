"Replaces terminal collision, visual, and contact metadata while preserving the finger chain. Proposal and accepted tip distributions can differ after validation; physics closure computes final mass and inertia."

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Literal

from ...asset_base import HandCfg, JointCfg
from ...asset_schema_core import PoseCfg, Vector2
from ...builder.joint_builders_custom import CustomTipBuilderCfg, apply_thumb_functional_tip_phase
from ...builder.joint_builders_primitive import PrimJointBuilderCfg
from ...procedural_meshes import is_procedural_cs_tip_uri, parse_procedural_cs_tip_uri
from .base import HandPatch, MutatorBase, MutatorBaseCfg, _make_range_sampler

_MODE_IDENTITY = "identity"
_MODE_SAME = "same"
_MODE_GENERAL = "general"

_ALL_SELF_MODES = (
    _MODE_IDENTITY,
    _MODE_SAME,
    _MODE_GENERAL,
)
"Allowed whole-hand proposal modes for this operator."

_CUSTOM_TIP_TYPES = (
    "leap_cube",
    "round",
    "wedge",
    "thinner",
)
"Supported custom mesh tip names."

_PRIMITIVE_TIP_TYPES = ("cs",)
"Primitive tip names supported by the tip replacement operator."

_DEFAULT_TIP_TYPES = _PRIMITIVE_TIP_TYPES + _CUSTOM_TIP_TYPES
"Default proposal candidates for tip replacement."

_MODE_TOLERANCE = 1e-9
"Numerical tolerance for proposal probability sums."

_MIN_POSITIVE = 1e-6
"Smallest positive scale accepted for non-degenerate mesh geometry."


@dataclass
class TipReplaceCfg(MutatorBaseCfg):
    """Proposal settings for terminal contact geometry.

    tip_type names a local tip shape, not the base hand family. self_mode couples choices across a hand, while tip_range proposes a recipe per tip; validator rejection can change accepted counts. Final mass and inertia come from physics closure.
    """

    class_type: type[TipReplaceMutator] | None = field(init=False, default=None, repr=False)
    "Associated runtime implementation for this configuration class."

    target_fingers: tuple[str, ...] | None = None
    "Semantic finger slots included by this mutation."

    self_mode: Literal["identity", "same", "general"] | dict[str, float] | None = _MODE_SAME
    "Identity, shared, or per-finger mutation proposal mode."

    tip_range: list[str] | dict[str, float] | None = None
    "Proposal distribution over terminal shape recipes; it is distinct from the LEAP or Allegro base family. Validation can change accepted counts."

    scale: Vector2 | dict[str, Vector2] = (1.0, 1.0)
    "Dimensionless scale applied after source mesh unit conversion."

    cs_ratio: Vector2 | dict[Literal["add", "abs"], Vector2] | None = None
    "Height-to-radius ratio for the cylindrical-segment tip geometry."

    distrib: Literal["uniform", "normal"] | dict[str, Any] = "uniform"
    "Proposal distribution used inside the declared parameter interval."

    boundary_policy: Literal["none", "clip", "truncate", "resample"] | None = None
    "Action for proposals outside their permitted domain."

    _active_modes: tuple[str, ...] = field(init=False, default=(), repr=False)
    "Resolved proposal modes with positive probability."

    def __post_init__(self) -> None:
        "Normalizes typed inputs and enforces the class invariants."

        self.class_type = TipReplaceMutator
        if isinstance(self.target_fingers, list):
            self.target_fingers = tuple(str(name) for name in self.target_fingers)
        self._active_modes = _resolve_active_modes(self.self_mode)
        _resolve_tip_distribution(self.tip_range)
        _validate_scale_ranges(self.scale)
        _validate_cs_ratio(self.cs_ratio)


class TipReplaceMutator(MutatorBase):
    "Applies a typed geometry proposal to a hand copy and records sampled values."

    cfg: TipReplaceCfg

    def __init__(self, cfg: TipReplaceCfg):
        "Stores the tip geometry proposal config for later per-hand sampling."

        self.cfg = cfg

    def describe_sampling(self, target: HandCfg) -> dict[str, Any]:
        'Summarizes proposal settings and sample outcomes.'

        return {"sample": lambda: self._sample_one(target)}

    def plan_patch(self, target: HandCfg, sampled_params: dict[str, Any] | None = None) -> HandPatch:
        "Plans terminal collision, visual, and contact metadata edits; final inertial properties come from physics closure."

        sample = _normalize_sample_payload(sampled_params, self.cfg, target=target)
        resolved_mode = str(sample["resolved_self_mode"])

        patch = HandPatch()
        patch.metadata.setdefault("post_mutate_samples", {})
        patch.metadata["post_mutate_samples"]["tip_replace"] = sample
        patch.metadata["post_mutate_tip_replace"] = sample

        if resolved_mode == _MODE_IDENTITY:
            return patch

        finger_specs = dict(sample.get("finger_specs", {}))
        for finger_index, finger in _iter_target_fingers(target, self.cfg.target_fingers):
            spec = dict(finger_specs.get(finger.name, {}))
            if not spec:
                continue
            replacement = _build_replacement_tip_joint(finger.tip_joint, spec)
            patch.add(
                ("finger", finger_index, "tip"),
                _tip_joint_replacer(finger_index=finger_index, replacement=replacement),
            )
        return patch

    def _sample_one(self, target: HandCfg) -> dict[str, Any]:

        resolved_mode = _draw_resolved_mode(self.cfg)
        return self.sample_one_for_mode(target, resolved_mode=resolved_mode)

    def sample_one_for_mode(self, target: HandCfg, *, resolved_mode: str) -> dict[str, Any]:
        'Samples parameters for the resolved mutation mode.'

        if resolved_mode not in _ALL_SELF_MODES:
            raise ValueError(f"unsupported tip_replace resolved mode: {resolved_mode!r}")
        if resolved_mode == _MODE_IDENTITY:
            return {"resolved_self_mode": _MODE_IDENTITY, "finger_specs": {}}

        target_fingers = list(_iter_target_fingers(target, self.cfg.target_fingers))
        if resolved_mode == _MODE_SAME:
            shared_spec = _sample_tip_spec(self.cfg, target_fingers[0][1].tip_joint if target_fingers else None)
            return {
                "resolved_self_mode": _MODE_SAME,
                "finger_specs": {finger.name: dict(shared_spec) for _, finger in target_fingers},
            }

        return {
            "resolved_self_mode": _MODE_GENERAL,
            "finger_specs": {
                finger.name: _sample_tip_spec(self.cfg, finger.tip_joint)
                for _, finger in target_fingers
            },
        }


def _iter_target_fingers(hand: HandCfg, target_fingers: tuple[str, ...] | None):

    target_set = set(target_fingers or ())
    for finger_index, finger in enumerate(hand.fingers):
        if target_set and finger.name not in target_set:
            continue
        yield finger_index, finger


def _tip_joint_replacer(*, finger_index: int, replacement: JointCfg):

    def _apply(hand: HandCfg) -> None:
        joints = list(hand.fingers[finger_index].joints)
        current_tip = joints[-1]
        # Keep the current joint origin; earlier link-scale edits may have moved the terminal kinematic shell.
        joints[-1] = current_tip.replace(
            inertial=replacement.inertial.copy() if replacement.inertial is not None else None,
            collisions=[collision.copy() for collision in replacement.collisions],
            visuals=[visual.copy() for visual in replacement.visuals],
            is_tip=replacement.is_tip,
            metadata=dict(replacement.metadata),
        )
        hand.fingers[finger_index] = hand.fingers[finger_index].replace(joints=joints)

    return _apply


def _build_replacement_tip_joint(original: JointCfg, spec: dict[str, Any]) -> JointCfg:

    tip_type = str(spec["tip_type"])
    common_kwargs = {
        "name": original.name,
        "parent": original.parent,
        "child": original.child,
        "joint_type": original.joint_type,
        "origin": original.origin,
        "axis": (0.0, 0.0, 0.0) if original.joint_type == "fixed" else original.axis,
        "limit": original.limit,
        "is_tip": True,
        "metadata": _replacement_metadata(original, spec),
    }

    if tip_type == "cs":
        radius = max(_MIN_POSITIVE, float(spec["radius"]))
        height = max(_MIN_POSITIVE, float(spec["height"]))
        builder_cfg = PrimJointBuilderCfg(
            mesh={"type": "cs", "radius": radius, "height": height, "offset": _tip_offset_from_original(original)},
            **common_kwargs,
        )
    else:
        mesh_offset = _tip_offset_from_original(original)
        if _is_thumb_tip_joint(original):
            mesh_offset = apply_thumb_functional_tip_phase(mesh_offset)
        builder_cfg = CustomTipBuilderCfg(
            tip_type=tip_type,
            mesh_offset=mesh_offset,
            scale=float(spec.get("scale", 1.0)),
            **common_kwargs,
        )
    builder = builder_cfg.class_type(builder_cfg)
    return builder.build()


def _replacement_metadata(original: JointCfg, spec: dict[str, Any]) -> dict[str, Any]:

    metadata = {
        **dict(original.metadata),
        "post_mutate_tip_mode": "tip_replace",
        "tip_type": str(spec["tip_type"]),
        "post_mutate_tip_type": str(spec["tip_type"]),
        "post_mutate_tip_scale": float(spec.get("scale", 1.0)),
        "post_mutate_tip_spec": dict(spec),
    }
    if str(spec["tip_type"]) != "cs" and _is_thumb_tip_joint(original):
        metadata["thumb_functional_tip_phase_rpy"] = apply_thumb_functional_tip_phase(PoseCfg()).rpy
    return metadata


def _is_thumb_tip_joint(joint: JointCfg) -> bool:

    finger_name = str(joint.metadata.get("finger_name", "")).lower()
    if finger_name == "thumb":
        return True
    name_fields = (joint.name, joint.parent, joint.child)
    return any(str(value).lower().startswith("thumb_") or str(value).lower() == "thumb" for value in name_fields)


def _tip_offset_from_original(original: JointCfg) -> PoseCfg:

    if original.collisions:
        first = original.collisions[0]
        if first.geometry.kind == "mesh" and _is_cs_tip_metadata(original.metadata):
            return first.origin.copy()
        if first.geometry.kind == "cylinder":
            length = float(first.geometry.length)
            cap_rpy = original.collisions[1].origin.rpy if len(original.collisions) > 1 else (0.0, 0.0, 0.0)
            return PoseCfg(
                pos=(first.origin.pos[0], first.origin.pos[1] - length / 2.0, first.origin.pos[2]),
                rpy=cap_rpy,
            )
        if first.geometry.kind == "box":
            size_y = float(first.geometry.size[1])
            cap_rpy = original.collisions[1].origin.rpy if len(original.collisions) > 1 else first.origin.rpy
            return PoseCfg(
                pos=(first.origin.pos[0], first.origin.pos[1] - size_y / 2.0, first.origin.pos[2]),
                rpy=cap_rpy,
            )
    return PoseCfg()


def _sample_tip_spec(cfg: TipReplaceCfg, current_tip: JointCfg | None) -> dict[str, Any]:

    tip_type = _draw_tip_type(cfg.tip_range)
    scale = _sample_scale(cfg, tip_type=tip_type)
    spec: dict[str, Any] = {"tip_type": tip_type, "scale": scale}
    if tip_type == "cs":
        radius, base_ratio = _current_cs_radius_and_ratio(current_tip)
        scaled_radius = max(_MIN_POSITIVE, radius * scale)
        ratio = _sample_cs_ratio(cfg, current_ratio=base_ratio)
        spec.update(
            {
                "radius": scaled_radius,
                "height": max(_MIN_POSITIVE, scaled_radius * ratio),
                "cs_ratio": ratio,
            }
        )
    return spec


def _current_cs_radius_and_ratio(current_tip: JointCfg | None) -> tuple[float, float]:

    if current_tip is None:
        return 0.012, 1.0

    metadata_radius, metadata_ratio = _cs_radius_and_ratio_from_metadata(current_tip.metadata)
    if metadata_radius is not None:
        return metadata_radius, metadata_ratio

    radius: float | None = None
    height: float | None = None
    for element in current_tip.collisions:
        geometry = element.geometry
        if geometry.kind == "mesh" and is_procedural_cs_tip_uri(geometry.file_path):
            spec = parse_procedural_cs_tip_uri(geometry.file_path)
            return max(_MIN_POSITIVE, float(spec.radius)), max(_MIN_POSITIVE, float(spec.ratio))
        if geometry.kind == "cylinder":
            radius = float(geometry.radius)
            height = float(geometry.length)
            break
    if radius is None:
        for element in current_tip.collisions:
            geometry = element.geometry
            if geometry.kind == "sphere":
                radius = float(geometry.radius)
                break
    radius = max(_MIN_POSITIVE, float(radius if radius is not None else 0.012))
    ratio = max(_MIN_POSITIVE, float(height) / radius) if height is not None else 1.0
    return radius, ratio


def _cs_radius_and_ratio_from_metadata(metadata: dict[str, Any]) -> tuple[float | None, float]:

    if not _is_cs_tip_metadata(metadata):
        return None, 1.0
    radius_raw = metadata.get("cs_radius")
    height_raw = metadata.get("cs_height")
    ratio_raw = metadata.get("cs_ratio")
    if radius_raw is None:
        return None, 1.0
    radius = max(_MIN_POSITIVE, float(radius_raw))
    if ratio_raw is not None:
        return radius, max(_MIN_POSITIVE, float(ratio_raw))
    if height_raw is not None:
        return radius, max(_MIN_POSITIVE, float(height_raw) / radius)
    return radius, 1.0


def _is_cs_tip_metadata(metadata: dict[str, Any]) -> bool:

    return (
        metadata.get("tip_type") == "cs"
        or metadata.get("procedural_tip_type") == "cs"
        or metadata.get("procedural_mesh_kind") == "cs_tip"
    )


def _sample_scale(cfg: TipReplaceCfg, *, tip_type: str) -> float:

    low, high = _scale_range_for_tip(cfg.scale, tip_type)
    sampler = _make_range_sampler(
        (low, high),
        distrib=cfg.distrib,
        boundary_policy=cfg.boundary_policy,
    )
    return max(_MIN_POSITIVE, float(sampler()))


def _sample_cs_ratio(cfg: TipReplaceCfg, *, current_ratio: float) -> float:

    if cfg.cs_ratio is None:
        return max(_MIN_POSITIVE, float(current_ratio))

    mode, ratio_range = _resolve_cs_ratio_range(cfg.cs_ratio)
    sampler = _make_range_sampler(
        ratio_range,
        distrib=cfg.distrib,
        boundary_policy=cfg.boundary_policy,
    )
    sampled = float(sampler())
    if mode == "add":
        return max(_MIN_POSITIVE, float(current_ratio) + sampled)
    return max(_MIN_POSITIVE, sampled)


def _resolve_active_modes(self_mode: Any) -> tuple[str, ...]:

    if self_mode is None:
        return (_MODE_SAME,)
    if isinstance(self_mode, str):
        if self_mode not in _ALL_SELF_MODES:
            raise ValueError(f"unsupported tip_replace self_mode: {self_mode!r}")
        return (self_mode,)
    if not isinstance(self_mode, dict):
        raise TypeError(f"tip_replace.self_mode must be str | dict[str, float] | None, got {type(self_mode).__name__}")

    positive_modes: list[str] = []
    total = 0.0
    for mode_name, probability in self_mode.items():
        if mode_name not in _ALL_SELF_MODES:
            raise ValueError(f"unsupported tip_replace self_mode key: {mode_name!r}")
        prob = float(probability)
        if prob < 0.0:
            raise ValueError(f"tip_replace.self_mode probability must be non-negative, got {mode_name!r}={prob!r}")
        total += prob
        if prob > _MODE_TOLERANCE:
            positive_modes.append(mode_name)

    if not positive_modes:
        raise ValueError("tip_replace.self_mode dict must contain at least one positive-probability mode")
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=_MODE_TOLERANCE):
        raise ValueError(f"tip_replace.self_mode probabilities must sum to 1, got {total!r}")
    return tuple(positive_modes)


def _draw_resolved_mode(cfg: TipReplaceCfg) -> str:

    if cfg.self_mode is None:
        return _MODE_SAME
    if isinstance(cfg.self_mode, str):
        return cfg.self_mode

    threshold = random.random()
    cumulative = 0.0
    last_mode = _MODE_SAME
    for mode_name, probability in cfg.self_mode.items():
        prob = float(probability)
        if prob <= _MODE_TOLERANCE:
            continue
        cumulative += prob
        last_mode = mode_name
        if threshold <= cumulative + _MODE_TOLERANCE:
            return mode_name
    return last_mode


def _resolve_tip_distribution(tip_range: list[str] | dict[str, float] | None) -> dict[str, float]:

    if tip_range is None:
        probability = 1.0 / len(_DEFAULT_TIP_TYPES)
        return {tip_type: probability for tip_type in _DEFAULT_TIP_TYPES}
    if isinstance(tip_range, list):
        if not tip_range:
            raise ValueError("tip_replace.tip_range list must not be empty")
        normalized = tuple(_normalize_tip_type(tip_type) for tip_type in tip_range)
        probability = 1.0 / len(normalized)
        return {tip_type: probability for tip_type in normalized}
    if not isinstance(tip_range, dict):
        raise TypeError(f"tip_replace.tip_range must be list[str] | dict[str, float] | None, got {type(tip_range).__name__}")

    distribution: dict[str, float] = {}
    total = 0.0
    for tip_type, probability in tip_range.items():
        normalized = _normalize_tip_type(tip_type)
        prob = float(probability)
        if prob < 0.0:
            raise ValueError(f"tip_replace.tip_range probability must be non-negative, got {tip_type!r}={prob!r}")
        total += prob
        if prob > _MODE_TOLERANCE:
            distribution[normalized] = prob
    if not distribution:
        raise ValueError("tip_replace.tip_range dict must contain at least one positive-probability tip_type")
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=_MODE_TOLERANCE):
        raise ValueError(f"tip_replace.tip_range probabilities must sum to 1, got {total!r}")
    return distribution


def _draw_tip_type(tip_range: list[str] | dict[str, float] | None) -> str:

    distribution = _resolve_tip_distribution(tip_range)
    threshold = random.random()
    cumulative = 0.0
    last_tip_type = next(iter(distribution))
    for tip_type, probability in distribution.items():
        cumulative += float(probability)
        last_tip_type = tip_type
        if threshold <= cumulative + _MODE_TOLERANCE:
            return tip_type
    return last_tip_type


def _normalize_tip_type(tip_type: Any) -> str:

    normalized = str(tip_type).lower()
    if normalized not in set(_DEFAULT_TIP_TYPES):
        raise ValueError(f"unsupported tip_replace tip_type: {tip_type!r}")
    return normalized


def _scale_range_for_tip(scale: Vector2 | dict[str, Vector2], tip_type: str) -> Vector2:

    if isinstance(scale, dict):
        return scale.get(tip_type, scale.get("shared", (1.0, 1.0)))
    return scale


def _validate_scale_ranges(scale: Vector2 | dict[str, Vector2]) -> None:

    ranges = scale.values() if isinstance(scale, dict) else (scale,)
    for value_range in ranges:
        low, high = float(value_range[0]), float(value_range[1])
        if low <= 0.0 or high <= 0.0:
            raise ValueError(f"tip_replace.scale range must be positive, got {value_range!r}")


def _validate_cs_ratio(cs_ratio: Vector2 | dict[Literal["add", "abs"], Vector2] | None) -> None:

    if cs_ratio is None:
        return
    _resolve_cs_ratio_range(cs_ratio)


def _resolve_cs_ratio_range(cs_ratio: Vector2 | dict[Literal["add", "abs"], Vector2]) -> tuple[str, Vector2]:

    if isinstance(cs_ratio, dict):
        if set(cs_ratio) == {"add"}:
            return "add", _checked_ratio_range(cs_ratio["add"], allow_negative=True)
        if set(cs_ratio) == {"abs"}:
            return "abs", _checked_ratio_range(cs_ratio["abs"], allow_negative=False)
        raise ValueError("tip_replace.cs_ratio dict must contain exactly one key: 'add' or 'abs'")
    return "abs", _checked_ratio_range(cs_ratio, allow_negative=False)


def _checked_ratio_range(value_range: Vector2, *, allow_negative: bool) -> Vector2:

    low, high = float(value_range[0]), float(value_range[1])
    if not allow_negative and (low <= 0.0 or high <= 0.0):
        raise ValueError(f"tip_replace.cs_ratio abs range must be positive, got {value_range!r}")
    return (low, high)


def _normalize_sample_payload(
    sampled_params: dict[str, Any] | None,
    cfg: TipReplaceCfg,
    *,
    target: HandCfg,
) -> dict[str, Any]:

    sampled = dict(sampled_params or {})
    sample = sampled.get("sample")
    if isinstance(sample, dict):
        return dict(sample)
    if "resolved_self_mode" in sampled:
        return sampled
    if cfg.self_mode == _MODE_IDENTITY:
        return {"resolved_self_mode": _MODE_IDENTITY, "finger_specs": {}}
    return TipReplaceMutator(cfg).sample_one_for_mode(target, resolved_mode=_draw_resolved_mode(cfg))


def iter_tip_types_from_sample(sample: dict[str, Any] | None) -> list[str]:
    'Returns tip types from sample.'

    if not isinstance(sample, dict):
        return []
    payload = sample.get("sample") if "sample" in sample else sample
    if not isinstance(payload, dict):
        return []
    finger_specs = payload.get("finger_specs")
    if not isinstance(finger_specs, dict):
        return []
    tip_types: list[str] = []
    for spec in finger_specs.values():
        if isinstance(spec, dict) and "tip_type" in spec:
            tip_types.append(str(spec["tip_type"]))
    return tip_types


__all__ = ["TipReplaceCfg", "TipReplaceMutator", "iter_tip_types_from_sample"]
