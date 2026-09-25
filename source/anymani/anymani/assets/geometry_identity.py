"Computes stable geometry identities from typed hand geometry, excluding presentation-only details."

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .asset_base import HandCfg
from .asset_schema_geometry import derive_generated_geometry_semantics, geometry_semantics_to_dict


def geometry_fingerprint_from_hand(hand: HandCfg) -> str:
    "Hashes physical fields from a typed HandCfg for uniqueness checks."

    semantics = derive_generated_geometry_semantics(hand, asset_id="__geometry_identity__")
    return geometry_fingerprint_from_semantics(geometry_semantics_to_dict(semantics))


def geometry_fingerprint_from_sidecar(sidecar_path: str | Path) -> str:
    "Returns the physical geometry fingerprint recorded in a hand sidecar."

    resolved_path = Path(sidecar_path).resolve()
    document = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    if not isinstance(document, dict):
        raise TypeError(f"hand sidecar must be a mapping: {resolved_path}")
    semantics = document.get("geometry_semantics")
    if not isinstance(semantics, dict):
        raise ValueError(f"hand sidecar lacks geometry_semantics: {resolved_path}")
    return geometry_fingerprint_from_semantics(semantics, sidecar_dir=resolved_path.parent)


def geometry_fingerprint_from_semantics(
    semantics: dict[str, Any],
    *,
    sidecar_dir: Path | None = None,
) -> str:
    "Hashes canonical kinematics, limits, owners, and collision geometry."

    payload = deepcopy(semantics)
    for field_name in (
        "content_hash",
        "migration_version",
        "source_kind",
        "asset_id",
        "asset_name",
        "topology_key",
        "family",
        "joint_limits_rad",
        "anchor_seeds",
    ):
        payload.pop(field_name, None)


    components = payload.get("components", ())
    if not isinstance(components, (list, tuple)):
        raise TypeError("geometry_semantics.components must be a sequence")
    for component in components:
        if not isinstance(component, dict):
            raise TypeError("geometry semantics component must be a mapping")
        geometry = component.get("geometry_payload")
        if not isinstance(geometry, dict) or "file_path" not in geometry:
            continue
        raw_path = Path(str(geometry.pop("file_path")))
        mesh_path = raw_path if raw_path.is_absolute() or sidecar_dir is None else sidecar_dir / raw_path
        if not mesh_path.is_file():
            raise FileNotFoundError(f"geometry fingerprint mesh does not exist: {mesh_path}")
        geometry["mesh_sha256"] = hashlib.sha256(mesh_path.read_bytes()).hexdigest()

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    digest = hashlib.sha256()
    digest.update(b"anymani-dataset-build-geometry-v1\0")
    digest.update(encoded)
    return digest.hexdigest()


__all__ = [
    "geometry_fingerprint_from_hand",
    "geometry_fingerprint_from_semantics",
    "geometry_fingerprint_from_sidecar",
]
