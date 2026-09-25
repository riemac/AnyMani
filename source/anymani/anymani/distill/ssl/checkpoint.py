"""Full training checkpoint and standalone retained-artifact I/O."""


from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

CHECKPOINT_SCHEMA_VERSION = "9.0.0"


@dataclass(frozen=True)
class PretrainCheckpointMetadata:


    code_revision: str
    package_version: str
    geometry_semantics_schema: str
    dataset_identity: Mapping[str, Any]
    resolved_config: Mapping[str, Any]
    declared_objective: Mapping[str, float]
    objective_formula: Mapping[str, str] = field(default_factory=dict)
    fairgrad_formula: Mapping[str, Any] = field(default_factory=dict)
    parameter_partition: Mapping[str, Any] = field(default_factory=dict)
    source_artifact: Mapping[str, Any] = field(default_factory=dict)
    worktree_dirty: bool = False
    worktree_fingerprint: str = ""


def save_pretrain_checkpoint(
    path: Path,
    *,
    method_state: Mapping[str, Any],
    optimizer_state: Mapping[str, Any],
    epoch: int,
    optimizer_update: int,
    metadata: PretrainCheckpointMetadata,
    trainer_state: Mapping[str, Any],
) -> None:


    if epoch < 0 or optimizer_update < 0:
        raise ValueError("checkpoint epoch and optimizer_update must be non-negative")
    if not method_state:
        raise ValueError("checkpoint method_state must be non-empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "epoch": int(epoch),
            "optimizer_update": int(optimizer_update),
            "method_state": dict(method_state),
            "optimizer_state": dict(optimizer_state),
            "metadata": asdict(metadata),
            "trainer_state": dict(trainer_state),
        },
        temporary,
    )
    temporary.replace(path)


def load_pretrain_checkpoint(
    path: Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:


    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("pretraining checkpoint payload must be a mapping")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported pretraining checkpoint schema={payload.get('schema_version')!r}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION!r}"
        )
    required = {"epoch", "optimizer_update", "method_state", "optimizer_state", "metadata", "trainer_state"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"pretraining checkpoint is missing fields: {sorted(missing)}")
    for name in ("method_state", "optimizer_state", "metadata", "trainer_state"):
        if not isinstance(payload[name], Mapping):
            raise ValueError(f"pretraining checkpoint {name} must be a mapping")
    return payload


def save_retained_artifact(path: Path, payload: Mapping[str, Any]) -> None:


    if not payload:
        raise ValueError("retained artifact payload must be non-empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "PretrainCheckpointMetadata",
    "load_pretrain_checkpoint",
    "save_pretrain_checkpoint",
    "save_retained_artifact",
]
