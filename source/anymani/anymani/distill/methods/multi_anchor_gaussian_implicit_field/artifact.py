r"""多锚点 Gaussian Method 的 standalone retained artifact loader。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from anymani.distill.models.input_adapters.geometry import ImplicitGeometryEncoder

RETAINED_ARTIFACT_SCHEMA_VERSION = "5.0.0"


@dataclass(frozen=True)
class RetainedLoadReport:
    r"""PPO/IL 初始化时必须记录的 missing/unexpected encoder keys。"""

    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


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


__all__ = ["RETAINED_ARTIFACT_SCHEMA_VERSION", "RetainedLoadReport", "load_retained_geometry_artifact"]
