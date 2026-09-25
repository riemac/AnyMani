"Serializes hand identity, generation provenance, and typed geometry semantics beside each URDF."

from __future__ import annotations

from typing import Any

from .asset_schema_embodiment import HandCfg


def restore_hand_cfg_snapshot(hand_cfg_raw: dict[str, Any]) -> HandCfg:
    'Applies hand cfg snapshot.'

    if not isinstance(hand_cfg_raw, dict):
        raise TypeError(f"'hand_cfg' must be a mapping, got {type(hand_cfg_raw).__name__}")
    normalized = _rehydrate_geometry_mappings(hand_cfg_raw)
    return HandCfg(**normalized)


def _rehydrate_geometry_mappings(value: Any) -> Any:

    if isinstance(value, list):
        return [_rehydrate_geometry_mappings(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_rehydrate_geometry_mappings(item) for item in value)
    if not isinstance(value, dict):
        return value

    normalized = {key: _rehydrate_geometry_mappings(item) for key, item in value.items()}
    if "geometry" in normalized and isinstance(normalized["geometry"], dict):
        normalized["geometry"] = _inject_geometry_type(normalized["geometry"])
    return _inject_geometry_type(normalized)


def _inject_geometry_type(geometry_doc: dict[str, Any]) -> dict[str, Any]:

    if "type" in geometry_doc or "kind" in geometry_doc:
        return geometry_doc

    normalized = dict(geometry_doc)
    if any(key in normalized for key in ("file_path", "path", "mesh")):
        normalized["type"] = "mesh"
    elif {"radius_x", "radius_z", "length"} <= normalized.keys():
        normalized["type"] = "elliptic_cylinder"
    elif "size" in normalized:
        normalized["type"] = "box"
    elif "radius" in normalized and "length" in normalized:
        normalized["type"] = "cylinder"
    elif "radius" in normalized:
        normalized["type"] = "sphere"
    return normalized


__all__ = ["restore_hand_cfg_snapshot"]
