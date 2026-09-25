"Restores typed hand configurations from exported sidecars for mutate-only runs."

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ...asset_base import HandCfg
from ...asset_sidecar import restore_hand_cfg_snapshot
from ...handedness import validate_generated_handedness_contract

_SIDECAR_FILENAME = "hand.yaml"
_SIDECAR_SUMMARY_KEYS = {
    "id",
    "timestamp",
    "name",
    "family",
    "handedness",
    "dof",
    "finger_count",
    "fingers",
    "provenance",
    "hand_cfg",
    "warnings",
}


@dataclass(frozen=True)
class PostMutateSource:
    "Restored source hand and run context loaded from one mother bundle."

    topology_dir: Path
    origin_sidecar_path: Path
    origin_sample_id: str
    hand_cfg: HandCfg
    metadata: dict[str, Any]


def load_post_mutate_source(
    topology_dir: Path | str,
    *,
    allow_legacy_left_handedness: bool = False,
) -> PostMutateSource:
    'Loads and validates post mutate source.'

    resolved_topology_dir = Path(topology_dir)
    if not resolved_topology_dir.is_dir():
        raise FileNotFoundError(f"Post-mutate source topology directory does not exist: {resolved_topology_dir}")


    sidecar_path = resolved_topology_dir / _SIDECAR_FILENAME
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            "Independent post-mutate now requires a topology-root sidecar; "
            f"missing {sidecar_path}"
        )

    sidecar_doc = yaml.safe_load(sidecar_path.read_text(encoding="utf-8")) or {}
    if not isinstance(sidecar_doc, dict):
        raise ValueError(f"Sidecar must be a mapping, got {type(sidecar_doc).__name__}: {sidecar_path}")
    validate_generated_handedness_contract(
        sidecar_doc,
        allow_legacy_left_handedness=allow_legacy_left_handedness,
    )

    hand_cfg_raw = sidecar_doc.get("hand_cfg")
    if not isinstance(hand_cfg_raw, dict):
        raise ValueError(
            f"Sidecar {sidecar_path} is missing top-level 'hand_cfg'; cannot restore independent post-mutate source."
        )

    origin_sample_id = str(sidecar_doc.get("id") or "")
    if not origin_sample_id:
        raise ValueError(
            f"Topology-root sidecar {sidecar_path} is missing top-level 'id'; "
            "independent post-mutate requires a stable pre-made sample identifier."
        )

    hand_cfg = restore_hand_cfg_snapshot(hand_cfg_raw)
    metadata = {
        key: value
        for key, value in sidecar_doc.items()
        if key not in _SIDECAR_SUMMARY_KEYS
    }
    metadata["source_origin_sample_id"] = origin_sample_id
    metadata["source_origin_topology_dir"] = str(resolved_topology_dir)
    metadata["source_topology_dir"] = str(resolved_topology_dir)

    return PostMutateSource(
        topology_dir=resolved_topology_dir,
        origin_sidecar_path=sidecar_path,
        origin_sample_id=origin_sample_id,
        hand_cfg=hand_cfg,
        metadata=metadata,
    )


__all__ = ["PostMutateSource", "load_post_mutate_source"]
