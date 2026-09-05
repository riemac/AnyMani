r"""多锚点 Gaussian Method 的 standalone retained artifact loader。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from anymani.distill.models.input_adapters.geometry import ImplicitGeometryEncoder

RETAINED_ARTIFACT_SCHEMA_VERSION = "5.0.0"


@dataclass(frozen=True)
class RetainedLoadReport:
    r"""PPO/IL 初始化时必须记录的 missing/unexpected encoder keys。"""

    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def build_retained_geometry_artifact(
    method: Any,
    *,
    metadata: Mapping[str, Any],
    source_checkpoint: Path,
) -> dict[str, Any]:
    r"""构造 schema-5 encoder-only transfer artifact，不泄漏 SSL readers/optimizer。"""

    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"retained artifact source checkpoint does not exist: {source_checkpoint}")
    raw_retained = method.retained_state_dict()
    if not raw_retained or any(not key.startswith("encoder.") for key in raw_retained):
        raise ValueError("retained artifact requires a non-empty encoder-only state")
    if any(value.dtype != torch.float32 for value in raw_retained.values()):
        raise ValueError("retained artifact requires FP32 encoder master parameters")
    retained = {
        str(key): value.detach().to(device="cpu", dtype=torch.float32).clone()
        for key, value in raw_retained.items()
    }
    resolved_config = metadata.get("resolved_config", {})
    trainer_config = resolved_config.get("trainer", {}) if isinstance(resolved_config, Mapping) else {}
    precision = trainer_config.get("execution", {}) if isinstance(trainer_config, Mapping) else {}
    source_artifact = metadata.get("source_artifact", {})
    if not isinstance(precision, Mapping) or not isinstance(source_artifact, Mapping):
        raise ValueError("retained artifact lineage lacks execution precision or source identity")
    required_precision = {
        "teacher_dtype",
        "parameter_dtype",
        "model_autocast_dtype",
        "loss_dtype",
        "fairgrad_accumulation_dtype",
        "allow_tf32",
        "compile_enabled",
        "compile_mode",
    }
    if required_precision - precision.keys():
        raise ValueError("retained artifact lineage lacks the complete execution precision profile")
    return {
        "schema_version": RETAINED_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "retained_geometry_encoder",
        "retained_state": retained,
        "retained_model_config": {"encoder": asdict(method.config.model.encoder)},
        "feature_spec": asdict(method.feature_spec()),
        "input_contract": {
            "frame": "query/closest/surface in hand frame {h}",
            "units": "length=m,joint=rad,density=dimensionless,kappa=m/rad",
            "retained_inputs": "physical q + static geometry evidence",
        },
        "lineage": {
            "source_checkpoint": str(source_checkpoint),
            "checkpoint_schema_version": "9.0.0",
            "code_revision": metadata.get("code_revision", "unknown"),
            "package_version": metadata.get("package_version", "unknown"),
            "geometry_semantics_schema": metadata.get("geometry_semantics_schema", "unknown"),
            "asset_manifest": dict(metadata.get("asset_manifest", {})),
            "dataset_identity": dict(metadata.get("dataset_identity", {})),
            "execution_precision": dict(precision),
            "source_artifact": dict(source_artifact),
            "parameter_partition": dict(metadata.get("parameter_partition", {})),
            "worktree_dirty": bool(metadata.get("worktree_dirty", False)),
            "worktree_fingerprint": str(metadata.get("worktree_fingerprint", "")),
        },
    }


def load_retained_geometry_artifact(
    path: Path,
    *,
    encoder: ImplicitGeometryEncoder,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> RetainedLoadReport:
    r"""严格加载 Method-owned retained encoder artifact。"""

    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != RETAINED_ARTIFACT_SCHEMA_VERSION:
        actual = payload.get("schema_version") if isinstance(payload, dict) else None
        raise ValueError(
            f"unsupported retained artifact schema={actual!r}; expected {RETAINED_ARTIFACT_SCHEMA_VERSION!r}"
        )
    if payload.get("artifact_type") != "retained_geometry_encoder":
        raise ValueError("retained artifact type is not retained_geometry_encoder")
    required = {"retained_state", "retained_model_config", "feature_spec", "input_contract", "lineage"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"retained artifact is missing fields: {sorted(missing)}")
    forbidden = ("optimizer_state", "trainer_state", "method_state", "query_backend", "target_backend", "objective")
    leaked = tuple(name for name in forbidden if name in payload)
    if leaked:
        raise ValueError(f"retained artifact contains disposable fields: {leaked}")
    feature_spec = payload.get("feature_spec")
    if not isinstance(feature_spec, Mapping) or set(feature_spec) != {
        "entity_width",
        "entity_axis",
        "joint_view",
        "frame_contract",
        "coordinate_rewrite_contract",
    }:
        raise ValueError("retained artifact feature_spec is not the unified entity/JOINT-view contract")
    retained = payload.get("retained_state")
    if not isinstance(retained, Mapping):
        raise ValueError("retained artifact retained_state must be a mapping")
    unexpected_namespace = tuple(str(key) for key in retained if not str(key).startswith("encoder."))
    if unexpected_namespace:
        raise ValueError(f"retained artifact contains non-encoder namespaces: {unexpected_namespace}")
    encoder_state = {str(key)[len("encoder.") :]: value for key, value in retained.items()}
    incompatible = encoder.load_state_dict(encoder_state, strict=False)
    report = RetainedLoadReport(tuple(incompatible.missing_keys), tuple(incompatible.unexpected_keys))
    if strict and (report.missing_keys or report.unexpected_keys):
        raise RuntimeError(
            f"retained encoder key mismatch: missing={report.missing_keys}, unexpected={report.unexpected_keys}"
        )
    return report


__all__ = [
    "RETAINED_ARTIFACT_SCHEMA_VERSION",
    "RetainedLoadReport",
    "build_retained_geometry_artifact",
    "load_retained_geometry_artifact",
]
