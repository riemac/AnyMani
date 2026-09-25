"Loads and validates sidecar geometry semantics while preserving coordinate frames, units, and source identity."

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypeAlias

from ..asset_schema_geometry import (
    HandGeometrySemanticsCfg,
    derive_generated_geometry_semantics,
    geometry_semantics_from_dict,
)
from ..asset_sidecar import restore_hand_cfg_snapshot

GEOMETRY_SEMANTICS_SIDECAR_KEY = "geometry_semantics"
"hand.yaml key containing versioned typed geometry semantics."

HandAssetSourceKind: TypeAlias = Literal["generated", "official"]
"Sidecar contract used to resolve generated or official hands."


def resolve_hand_geometry_semantics(
    sidecar: Mapping[str, Any],
    *,
    source_kind: HandAssetSourceKind,
    asset_id: str,
    topology_key: str | None = None,
) -> HandGeometrySemanticsCfg:
    'Resolves hand geometry semantics.'

    raw_semantics = sidecar.get(GEOMETRY_SEMANTICS_SIDECAR_KEY)
    if raw_semantics is not None:
        if not isinstance(raw_semantics, Mapping):
            raise TypeError(f"{GEOMETRY_SEMANTICS_SIDECAR_KEY} must be a mapping")
        semantics = geometry_semantics_from_dict(raw_semantics)
        if semantics.asset_id != asset_id:
            raise ValueError(
                f"geometry semantics asset_id={semantics.asset_id!r} does not match container asset_id={asset_id!r}"
            )
        if semantics.source_kind != source_kind:
            raise ValueError(
                f"geometry semantics source_kind={semantics.source_kind!r} does not match "
                f"container source_kind={source_kind!r}"
            )
        return semantics

    if source_kind == "official":
        raise ValueError(
            "official hand assets require an explicit, manually verified geometry_semantics sidecar field"
        )

    hand_cfg_raw = sidecar.get("hand_cfg")
    if not isinstance(hand_cfg_raw, dict):
        raise ValueError(
            "legacy generated hand sidecar is missing top-level 'hand_cfg'; "
            "geometry semantics cannot be migrated"
        )
    hand = restore_hand_cfg_snapshot(hand_cfg_raw)
    resolved_topology_key = topology_key
    if resolved_topology_key is None and sidecar.get("topology_name") is not None:
        resolved_topology_key = str(sidecar["topology_name"])
    return derive_generated_geometry_semantics(
        hand,
        asset_id=asset_id,
        topology_key=resolved_topology_key,
    )


__all__ = [
    "GEOMETRY_SEMANTICS_SIDECAR_KEY",
    "HandAssetSourceKind",
    "resolve_hand_geometry_semantics",
]
