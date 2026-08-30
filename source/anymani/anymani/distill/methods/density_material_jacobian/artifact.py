r"""Density + Gamma method 的 schema-5 encoder-only retained artifact。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch


def build_retained_artifact(
    method: Any,
    *,
    metadata: Mapping[str, Any],
    source_checkpoint: Path,
) -> dict[str, Any]:
    r"""发布只含 unified encoder 的 schema-5 artifact，不泄漏 density/Gamma readers。"""

    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"retained artifact source checkpoint does not exist: {source_checkpoint}")
    raw = method.retained_state_dict()
    if not raw or any(not name.startswith("encoder.") for name in raw):
        raise ValueError("retained artifact requires non-empty encoder-only state")
    if any(value.dtype != torch.float32 for value in raw.values()):
        raise ValueError("retained artifact requires FP32 encoder master parameters")
    retained = {
        name: value.detach().to(device="cpu", dtype=torch.float32).clone()
        for name, value in raw.items()
    }
    resolved = metadata.get("resolved_config", {})
    trainer = resolved.get("trainer", {}) if isinstance(resolved, Mapping) else {}
    precision = trainer.get("execution", {}) if isinstance(trainer, Mapping) else {}
    source_artifact = metadata.get("source_artifact", {})
    if not isinstance(precision, Mapping) or not isinstance(source_artifact, Mapping):
        raise ValueError("retained artifact lineage lacks precision or source identity")
    return {
        "schema_version": "5.0.0",
        "artifact_type": "retained_geometry_encoder",
        "retained_state": retained,
        "retained_model_config": {"encoder": asdict(method.config.model.encoder)},
        "feature_spec": asdict(method.feature_spec()),
        "input_contract": {
            "frame": "physical geometry in hand frame {h}; in-plane SO(2) gauge",
            "units": "length=m,joint=rad,density=dimensionless,Gamma=rad^-1",
            "retained_inputs": "physical q + static geometry evidence",
            "discarded_ssl_readers": "density,material_jacobian",
        },
        "lineage": {
            "source_checkpoint": str(source_checkpoint),
            "checkpoint_schema_version": "9.0.0",
            "code_revision": metadata.get("code_revision", "unknown"),
            "package_version": metadata.get("package_version", "unknown"),
            "asset_manifest": dict(metadata.get("asset_manifest", {})),
            "dataset_identity": dict(metadata.get("dataset_identity", {})),
            "execution_precision": dict(precision),
            "source_artifact": dict(source_artifact),
            "parameter_partition": dict(metadata.get("parameter_partition", {})),
            "worktree_dirty": bool(metadata.get("worktree_dirty", False)),
            "worktree_fingerprint": str(metadata.get("worktree_fingerprint", "")),
        },
    }


__all__ = ["build_retained_artifact"]
