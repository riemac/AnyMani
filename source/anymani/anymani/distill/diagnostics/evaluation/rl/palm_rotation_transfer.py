r"""Validate frozen-policy evaluation on a different morphology population.

Only the selected asset population may change. Policy/task/observation semantics,
retained geometry encoder, precision and implementation remain checked. Shared
physical assets retain their exact ranked pregrasps. Catalog growth is allowed
only after checking the archived source index and current selected payloads.
These rules authorize evaluation, not full PPO-state resume or a capability pass.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _catalog_entries(pregrasp: Mapping[str, Any], root: Path) -> dict[str, dict[str, Any]]:
    r"""Read the exact catalog revision referenced by an evaluation or checkpoint."""
    catalog = root / pregrasp["catalog_root"]
    expected = pregrasp["index_sha256"]
    payload = (catalog / "index.json").read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        payload = (catalog / "index_history" / f"{expected}.json").read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise RuntimeError("pregrasp catalog revision does not match its recorded contents")
    entries = json.loads(payload)["entries"]
    by_key = {entry["key_digest"]: entry for entry in entries}
    if len(by_key) != len(entries):
        raise RuntimeError("pregrasp catalog contains duplicate keys")
    return by_key


def validate_transfer_evaluation(
    runtime: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    root: Path,
    implementation_certificate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    r"""Validate shared semantics and report the changed evaluation population.

    The `training` block still describes the source checkpoint. New assets are
    declared by the runtime manifest/provider/pregrasp fields, never represented
    as newly trained assets. Geometry population digests may vary; the retained
    artifact, feature definitions and precision may not.
    """
    for identity in (runtime, checkpoint):
        if identity.get("identity_schema_version") not in {"3.0.0", "4.0.0"}:
            raise RuntimeError("unsupported transfer evaluation identity schema")
        payload = {key: value for key, value in identity.items() if key != "identity_digest"}
        actual = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        ).hexdigest()
        if actual != identity.get("identity_digest"):
            raise RuntimeError("transfer evaluation identity does not match its payload")

    for field in ("task_id", "task_contract", "policy", "transport_abi", "training"):
        if field not in runtime or runtime[field] != checkpoint.get(field):
            raise RuntimeError(f"transfer evaluation changed shared semantics: {field}")
    population_fields = {"asset_ids", "physical_geometry_hashes", "base_provider_identity_digest", "identity_digest"}
    geometry = [identity["geometry_provider"] for identity in (checkpoint, runtime)]
    shared_geometry = [{key: value for key, value in item.items() if key not in population_fields} for item in geometry]
    if shared_geometry[0] != shared_geometry[1] or not shared_geometry[0].get("retained_artifact"):
        raise RuntimeError("transfer evaluation changed geometry encoder or precision semantics")

    source_files = checkpoint.get("implementation", {}).get("files")
    target_files = runtime.get("implementation", {}).get("files")
    if not source_files or not target_files:
        raise RuntimeError("transfer evaluation requires implementation identities")
    if source_files != target_files:
        certificate = implementation_certificate or {}
        if not (
            certificate.get("artifact_type") == "anymani.palm_rotation.refactor_equivalence"
            and certificate.get("schema_version") == "1.0.0"
            and certificate.get("passed") is True
            and certificate.get("reference_implementation_files") == source_files
            and certificate.get("current_implementation_files") == target_files
        ):
            raise RuntimeError("transfer evaluation implementation changed without an exact equivalence certificate")

    # Pair reset identities by physical geometry, not by selection-local row number.
    physical_keys = []
    assets = []
    for identity, provider in zip((checkpoint, runtime), geometry, strict=True):
        physical = provider["physical_geometry_hashes"]
        ids = provider["asset_ids"]
        keys = identity["pregrasp"]["ordered_key_digests"]
        count = identity["manifest"]["support_asset_count"]
        if not (len(physical) == len(ids) == len(keys) == count) or len(set(physical)) != count:
            raise RuntimeError("transfer population/pregrasp axes disagree")
        physical_keys.append(dict(zip(physical, keys, strict=True)))
        assets.append(dict(zip(ids, physical, strict=True)))
    for asset in assets[0].keys() & assets[1].keys():
        if assets[0][asset] != assets[1][asset]:
            raise RuntimeError("shared asset physical geometry changed")

    target_entries = _catalog_entries(runtime["pregrasp"], root)
    if any(key not in target_entries for key in physical_keys[1].values()):
        raise RuntimeError("target pregrasp catalog misses selected assets")
    shared = physical_keys[0].keys() & physical_keys[1].keys()
    if shared:
        source_entries = _catalog_entries(checkpoint["pregrasp"], root)
        for physical in shared:
            key = physical_keys[0][physical]
            if key != physical_keys[1][physical] or key not in source_entries:
                raise RuntimeError("shared asset pregrasp protocol changed")
            if source_entries[key]["entry_digest"] != target_entries[key]["entry_digest"]:
                raise RuntimeError("shared asset pregrasp payload changed")
    return {
        "source_asset_count": len(physical_keys[0]),
        "target_asset_count": len(physical_keys[1]),
        "shared_asset_count": len(shared),
        "new_asset_count": len(physical_keys[1]) - len(shared),
        "support_changed": runtime["manifest"] != checkpoint["manifest"],
        "shared_pregrasp_records_equal": True,
        "source_cohort_id": checkpoint["training"].get("cohort_id"),
    }
