"""Build and validate the encoder-only N040 artifact."""


from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from anymani.distill.methods.multi_anchor_gaussian_implicit_field.artifact import (
    RETAINED_ARTIFACT_SCHEMA_VERSION,
    RetainedLoadReport,
)
from anymani.distill.models.backbones.geometry_transformer import GraphBiasedTransformerCfg
from anymani.distill.models.input_adapters.se3_invariant_encoder import (
    SE3InvariantAnchorFrontendCfg,
    SE3InvariantGeometryEncoder,
    SE3InvariantGeometryEncoderCfg,
)


@dataclass(frozen=True)
class SE3RetainedEncoderArtifact:


    encoder: SE3InvariantGeometryEncoder
    load_report: RetainedLoadReport
    artifact_sha256: str
    path: Path
    feature_spec: Mapping[str, Any]
    input_contract: Mapping[str, Any]
    lineage: Mapping[str, Any]


def _file_sha256(path: Path) -> str:


    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_se3_retained_encoder_artifact(
    path: Path,
    *,
    expected_sha256: str,
    map_location: str | torch.device = "cpu",
) -> SE3RetainedEncoderArtifact:


    resolved_path = path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"retained artifact does not exist: {resolved_path}")
    if len(expected_sha256) != 64:
        raise ValueError("expected retained artifact SHA-256 must contain 64 hexadecimal characters")
    actual_sha256 = _file_sha256(resolved_path)
    if actual_sha256 != expected_sha256.lower():
        raise ValueError(
            f"retained artifact SHA-256 mismatch: expected={expected_sha256.lower()}, actual={actual_sha256}"
        )


    payload = torch.load(resolved_path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != RETAINED_ARTIFACT_SCHEMA_VERSION:
        actual_schema = payload.get("schema_version") if isinstance(payload, dict) else None
        raise ValueError(
            f"unsupported retained artifact schema={actual_schema!r}; expected {RETAINED_ARTIFACT_SCHEMA_VERSION!r}"
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


    model_config = payload.get("retained_model_config")
    if not isinstance(model_config, Mapping) or set(model_config) != {"encoder_type", "encoder"}:
        raise ValueError("retained_model_config must contain exactly encoder_type and encoder")
    if model_config.get("encoder_type") != "se3_invariant":
        raise ValueError("retained artifact encoder_type is not se3_invariant")
    encoder_payload = model_config.get("encoder")
    if not isinstance(encoder_payload, Mapping) or set(encoder_payload) != {"frontend", "backbone"}:
        raise ValueError("N040 encoder config must contain exactly frontend and backbone")
    frontend_payload = encoder_payload.get("frontend")
    backbone_payload = encoder_payload.get("backbone")
    if not isinstance(frontend_payload, Mapping) or not isinstance(backbone_payload, Mapping):
        raise ValueError("N040 frontend/backbone configs must be mappings")
    encoder_config = SE3InvariantGeometryEncoderCfg(
        frontend=SE3InvariantAnchorFrontendCfg(**dict(frontend_payload)),
        backbone=GraphBiasedTransformerCfg(**dict(backbone_payload)),  # 4-layer graph-biased retained trunk
    )
    if asdict(encoder_config) != dict(encoder_payload):
        raise ValueError("N040 encoder config cannot be reconstructed without drift")
    encoder = SE3InvariantGeometryEncoder(encoder_config)


    retained_state = payload.get("retained_state")
    if not isinstance(retained_state, Mapping) or not retained_state:
        raise ValueError("retained artifact retained_state must be a non-empty mapping")
    unexpected_namespace = tuple(str(key) for key in retained_state if not str(key).startswith("encoder."))
    if unexpected_namespace:
        raise ValueError(f"retained artifact contains non-encoder namespaces: {unexpected_namespace}")
    if any(not isinstance(value, torch.Tensor) or value.dtype != torch.float32 for value in retained_state.values()):
        raise ValueError("retained artifact requires FP32 tensor encoder state")
    encoder_state = {str(key)[len("encoder.") :]: value for key, value in retained_state.items()}
    incompatible = encoder.load_state_dict(encoder_state, strict=False)
    report = RetainedLoadReport(tuple(incompatible.missing_keys), tuple(incompatible.unexpected_keys))
    if report.missing_keys or report.unexpected_keys:
        raise RuntimeError(
            f"retained N040 encoder key mismatch: missing={report.missing_keys}, unexpected={report.unexpected_keys}"
        )


    feature_spec = payload.get("feature_spec")
    input_contract = payload.get("input_contract")
    lineage = payload.get("lineage")
    if not isinstance(feature_spec, Mapping):
        raise ValueError("retained artifact feature_spec must be a mapping")
    if not isinstance(input_contract, Mapping):
        raise ValueError("retained artifact input_contract must be a mapping")
    if not isinstance(lineage, Mapping):
        raise ValueError("retained artifact lineage must be a mapping")
    return SE3RetainedEncoderArtifact(
        encoder=encoder,
        load_report=report,
        artifact_sha256=actual_sha256,
        path=resolved_path,
        feature_spec={str(key): value for key, value in feature_spec.items()},
        input_contract={str(key): value for key, value in input_contract.items()},
        lineage={str(key): value for key, value in lineage.items()},
    )


def build_retained_artifact(
    method: Any,
    *,
    metadata: Mapping[str, Any],
    source_checkpoint: Path,
) -> dict[str, Any]:


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
    encoder_config = method.config.model.encoder
    encoder_type = (
        "se3_invariant" if isinstance(encoder_config, SE3InvariantGeometryEncoderCfg) else "legacy_so2"
    )
    feature_spec = method.feature_spec()
    return {
        "schema_version": "5.0.0",
        "artifact_type": "retained_geometry_encoder",
        "retained_state": retained,
        "retained_model_config": {
            "encoder_type": encoder_type,
            "encoder": asdict(encoder_config),
        },
        "feature_spec": asdict(feature_spec),
        "input_contract": {
            "frame": feature_spec.frame_contract,
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


__all__ = [
    "SE3RetainedEncoderArtifact",
    "build_retained_artifact",
    "load_se3_retained_encoder_artifact",
]
