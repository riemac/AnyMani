"Applies named visual material palettes without changing collision geometry."

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias, cast

from ...asset_base import HandCfg
from ...asset_schema_core import MaterialCfg, _ensure_tuple
from ...presets.color_presets import COLOR_PRESETS, DEFAULT_COLOR_PRESET_NAME

RgbaTuple: TypeAlias = tuple[float, float, float, float]
RecolorSpec: TypeAlias = str | dict[str, RgbaTuple] | bool | None


def normalize_recolor_spec(recolored: Any) -> RecolorSpec:
    'Normalizes recolor spec.'

    if recolored is None or recolored is False:
        return None
    if recolored is True:
        return DEFAULT_COLOR_PRESET_NAME
    if isinstance(recolored, str):
        normalized_name = recolored.strip()
        if not normalized_name:
            raise ValueError("recolored preset name cannot be empty")
        return normalized_name
    if isinstance(recolored, Mapping):
        normalized: dict[str, RgbaTuple] = {}
        for link_name, rgba in recolored.items():
            normalized_name = str(link_name).strip()
            if not normalized_name:
                raise ValueError("recolored override contains an empty child-link name")
            packed = _ensure_tuple(rgba, length=4, field_name=f"recolored[{normalized_name!r}]")
            normalized[normalized_name] = cast(RgbaTuple, tuple(float(value) for value in packed))
        return normalized
    raise TypeError(
        "recolored must be None/False, a palette name string, or a dict[child_link_name, rgba]; "
        f"got {type(recolored).__name__}"
    )


def describe_recolor_spec(recolored: RecolorSpec) -> dict[str, Any] | None:
    'Summarizes recolor spec.'

    normalized = normalize_recolor_spec(recolored)
    if normalized is None:
        return None
    if isinstance(normalized, str):
        return {"mode": "preset", "preset": normalized}
    if isinstance(normalized, dict):
        return {"mode": "overrides", "links": sorted(normalized)}
    raise TypeError(f"Unexpected normalized recolor payload: {normalized!r}")


def resolve_visual_recolor_materials(hand_cfg: HandCfg, recolored: RecolorSpec) -> dict[str, MaterialCfg]:
    'Resolves visual recolor materials.'

    normalized = normalize_recolor_spec(recolored)
    if normalized is None:
        return {}
    if isinstance(normalized, str):
        palette = _resolve_named_palette(normalized)
        return {
            link_name: _make_material(link_name=link_name, rgba=palette_rgba)
            for link_name, palette_rgba in _resolve_named_palette_targets(hand_cfg, palette).items()
        }
    if isinstance(normalized, dict):
        return {
            link_name: _make_material(link_name=link_name, rgba=rgba)
            for link_name, rgba in normalized.items()
        }
    raise TypeError(f"Unexpected normalized recolor payload: {normalized!r}")


def _resolve_named_palette(preset_name: str) -> dict[str, RgbaTuple]:

    if preset_name not in COLOR_PRESETS:
        raise ValueError(
            f"Unknown recolored palette {preset_name!r}; available presets are {sorted(COLOR_PRESETS)!r}"
        )
    return dict(COLOR_PRESETS[preset_name])


def _resolve_named_palette_targets(hand_cfg: HandCfg, palette: Mapping[str, RgbaTuple]) -> dict[str, RgbaTuple]:

    resolved: dict[str, RgbaTuple] = {}
    if "palm" in palette:
        resolved[hand_cfg.palm.name] = palette["palm"]

    for finger in hand_cfg.fingers:
        for joint in finger.joints:
            semantic_key = _infer_semantic_color_key(str(joint.child))
            if semantic_key is None or semantic_key not in palette:
                continue
            resolved[str(joint.child)] = palette[semantic_key]
    return resolved


def _infer_semantic_color_key(link_name: str) -> str | None:

    if link_name.endswith("_root_fixed_link"):
        return "root_fixed"

    for semantic_key in ("cmc1", "cmc2", "mcp1", "mcp2", "mcp", "pip", "dip", "tip"):
        if link_name.endswith(f"_{semantic_key}"):
            return semantic_key
    return None


def _make_material(*, link_name: str, rgba: RgbaTuple) -> MaterialCfg:

    return MaterialCfg(
        name=f"{link_name}_recolor",
        rgba=rgba,
    )


__all__ = [
    "RgbaTuple",
    "RecolorSpec",
    "normalize_recolor_spec",
    "describe_recolor_spec",
    "resolve_visual_recolor_materials",
]
