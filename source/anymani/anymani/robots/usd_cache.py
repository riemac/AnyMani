(
    'Stable cache identity for AnyMani URDF-to-USD conversion. Isaac Lab 2.3.2 '
    'hashes the main URDF and converter config but not referenced mesh bytes, so '
    'updated meshes may reuse stale USD. Build the directory key from URDF hash, '
    'ordered URI/mesh hashes, converter config, Isaac Lab/Sim versions, converter '
    'source hash, and canonical schema identity. Exclude selection-local '
    'asset_row because it does not change physics. Return a path only: do not '
    'create directories or import Isaac Sim/Kit; UrdfFileCfg and Isaac Lab own '
    'lazy hit/miss.'
)

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from anymani.assets.bank.urdf_utils import parse_urdf_mesh_refs

_ROUTING_ONLY_CANONICAL_FIELDS = frozenset({"asset_row", "asset_row_start"})
'Selection-local routing field; excluded from physical USD identity.'


def _sha256_file(path: Path) -> str:
    'Stream file SHA-256 instead of buffering all mesh bytes.'

    digest = hashlib.sha256()  # Identify files by content, not absolute path.
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)  # Read in 1 MiB chunks so peak memory is independent of file size.
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    'Convert converter config to deterministic JSON containers without object addresses.'

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        module = getattr(value, "__module__", "unknown")  # Stable implementation namespace for callables.
        qualname = getattr(value, "__qualname__", getattr(value, "__name__", type(value).__qualname__))
        return f"{module}.{qualname}"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"USD cache identity cannot serialize value of type {type(value).__qualname__}")


def build_urdf_usd_cache_dir(
    *,
    urdf_path: Path,
    converter_config: Mapping[str, Any],
    isaaclab_version: str,
    isaac_sim_version: str,
    converter_implementation_sha256: str,
    canonical_identity: Mapping[str, Any] | None = None,
    cache_root: Path | None = None,
) -> Path:
    (
        'Compute a mesh-aware, versioned USD cache directory independent of selection '
        'row. Include only importer settings that change output. Defaults use the '
        'current Isaac Lab/Sim and converter-source identities; canonical_identity '
        'adds schema/artifact identity but excludes routing row. cache_root may '
        'override the default cache root. Return '
        '<root>/isaaclab/usd/<sim-version>/<sha256-key> without creating it.'
    )

    resolved_urdf = urdf_path.expanduser().resolve(strict=True)  # physical importer input
    mesh_refs = parse_urdf_mesh_refs(resolved_urdf, require_existing=True)  # URI-to-real-path mapping in XML order.
    mesh_dependencies = [
        {
            "raw_uri": ref.raw_uri,
            "sha256": _sha256_file(ref.real_path),
        }
        for ref in mesh_refs
    ]  # Include URI and bytes in the key; equal basenames in different directories cannot collide.
    canonical = {
        str(key): _json_safe(value)
        for key, value in sorted((canonical_identity or {}).items())
        if key not in _ROUTING_ONLY_CANONICAL_FIELDS
    }  # asset_row affects policy routing only, not physical USD identity.
    payload = {
        "schema": "anymani.urdf_usd_cache.v1",
        "urdf_sha256": _sha256_file(resolved_urdf),
        "mesh_dependencies": mesh_dependencies,
        "converter_config": _json_safe(converter_config),
        "isaaclab_version": str(isaaclab_version),
        "isaac_sim_version": str(isaac_sim_version),
        "converter_implementation_sha256": str(converter_implementation_sha256),
        "canonical_identity": canonical,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    cache_key = hashlib.sha256(encoded).hexdigest()  # Stable 64-hex directory name for the full input identity.
    root = cache_root or Path(os.environ.get("ANYMANI_CACHE_DIR", "~/.cache/anymani"))
    safe_sim_version = str(isaac_sim_version).replace("/", "_")  # Prevent version strings from escaping the directory hierarchy.
    return root.expanduser() / "isaaclab" / "usd" / safe_sim_version / cache_key


__all__ = ["build_urdf_usd_cache_dir"]
