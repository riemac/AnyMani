r"""Embodiment pretraining 的通用 full-checkpoint 与 standalone artifact 容器。

本模块只保证原子写入、schema 和基础 mapping 完整性。具体 Method 的模型 namespace、frame/unit、
retained/disposable 边界和下游 loader 均由 concrete Method 拥有。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

CHECKPOINT_SCHEMA_VERSION = "7.0.0"


@dataclass(frozen=True)
class PretrainCheckpointMetadata:
    r"""随 full checkpoint 保存、与具体 Method 张量结构无关的实验 lineage。"""

    code_revision: str
    package_version: str
    geometry_semantics_schema: str
    dataset_identity: Mapping[str, Any]
    resolved_config: Mapping[str, Any]
    declared_objective: Mapping[str, float]
    calibration_artifact_hash: str = ""
    teacher_baselines: Mapping[str, float] = field(default_factory=dict)
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
    r"""原子保存 Method、optimizer 与 Trainer 可恢复状态。

    Args:
        path (Path): 正式 ``.pt`` 路径。
        method_state (Mapping[str, Any]): 由 concrete Method 定义的完整训练状态。
        optimizer_state (Mapping[str, Any]): optimizer moments 与 param groups。
        epoch (int): 已完整完成、可恢复的训练 epoch 数。
        optimizer_update (int): 已执行的参数更新总数。
        metadata (PretrainCheckpointMetadata): 实验配置、数据和代码 lineage。
        trainer_state (Mapping[str, Any]): schedule/session/RNG 与训练预算状态。
    """

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
    r"""读取并验证 full-checkpoint 顶层 schema，不解释 Method 内部 state。"""

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
    r"""原子写出 concrete Method 已经闭合语义的 standalone artifact payload。"""

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
