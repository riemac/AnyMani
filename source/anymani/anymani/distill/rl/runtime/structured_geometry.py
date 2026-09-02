r"""新`tasks/hetero` asset binding到严格N040 retained geometry provider的装配。

本模块不import旧任务族。Asset binding以Protocol交付ordered source/canonical axes；selection-local
``prototype_index``作为opaque side-channel路由provider，不进入actor/critic连续特征。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

import torch

from anymani.assets.bank.hand_container import HandContainer
from anymani.assets.bank.path_utils import resolve_anymani_root
from anymani.assets.canonical_runtime import (
    CANONICAL_HAND_SCHEMA_V1,
    CanonicalHandArtifact,
    CanonicalHandGroupManifest,
)
from anymani.distill.methods.density_material_jacobian.artifact import load_se3_retained_encoder_artifact
from anymani.distill.models.structured_heterogeneous import GeometryTokenBatch, StructuredActorObservation
from anymani.distill.rl.canonical_evidence import build_canonical_evidence_bank
from anymani.distill.rl.runtime.retained_geometry import RetainedGeometryBatch, RetainedGeometryProvider
from anymani.distill.rl.runtime.source_config import N040_PPO_SOURCE_CFG
from anymani.robots.hand_spawn import HandSpawnCfg

N040_RETAINED_ARTIFACT_PATH = Path(
    "logs/ssl/geometry_ssl_density_material_jacobian_se3_v0_8_1_extended512_matched/"
    "20260830T164445Z/retained_encoder.pt"
)
N040_RETAINED_ARTIFACT_SHA256 = "cda44cc9eae5ca28a1a735176ef4764805559d13e235c52477b6ac438b20ddea"


class StructuredGeometryAssetBinding(Protocol):
    r"""N040 provider所需的task-independent ordered asset surface。"""

    @property
    def source_assets(self) -> tuple[HandContainer, ...]: ...

    @property
    def hand_spawn_cfg(self) -> HandSpawnCfg: ...

    @property
    def canonical_artifacts(self) -> tuple[CanonicalHandArtifact, ...]: ...

    @property
    def dataset_sha256(self) -> str: ...


def canonical_group_manifest_digest(artifacts: tuple[CanonicalHandArtifact, ...]) -> str:
    r"""计算不落盘的ordered canonical group manifest SHA-256。"""

    manifest = CanonicalHandGroupManifest(
        schema_version=CANONICAL_HAND_SCHEMA_V1.version,
        schema_digest=CANONICAL_HAND_SCHEMA_V1.digest,
        artifacts=artifacts,
    )
    payload = json.dumps(manifest.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_structured_artifact_contract(
    feature_spec: Mapping[str, object], input_contract: Mapping[str, object]
) -> None:
    r"""补足schema-5 loader尚未逐值核对的structured downstream metadata。"""

    expected_feature = {
        "entity_width": 128,
        "entity_axis": "PALM/JOINT/TIP owner sequence",
        "joint_view": "gather entities with joint_entity_index",
        "frame_contract": "proper-SE(3)-invariant hand-coordinate representation; reflection-sensitive chirality",
    }
    for key, expected in expected_feature.items():
        if feature_spec.get(key) != expected:
            raise ValueError(f"N040 feature_spec {key!r} disagrees with structured contract")
    if input_contract.get("retained_inputs") != "physical q + static geometry evidence":
        raise ValueError("N040 retained input contract must be physical q + static geometry evidence")
    if input_contract.get("discarded_ssl_readers") != "density,material_jacobian":
        raise ValueError("N040 artifact must discard density/material_jacobian readers")


def build_structured_retained_geometry_provider(
    binding: StructuredGeometryAssetBinding,
    *,
    artifact_path: Path = N040_RETAINED_ARTIFACT_PATH,
    artifact_sha256: str = N040_RETAINED_ARTIFACT_SHA256,
    device: torch.device | str = "cpu",
) -> RetainedGeometryProvider:
    r"""严格加载N040、构造canonical evidence并返回冻结q-dependent provider。"""

    resolved_path = artifact_path if artifact_path.is_absolute() else resolve_anymani_root() / artifact_path
    artifact = load_se3_retained_encoder_artifact(
        resolved_path.resolve(), expected_sha256=artifact_sha256, map_location="cpu"
    )
    _validate_structured_artifact_contract(artifact.feature_spec, artifact.input_contract)
    evidence_bank = build_canonical_evidence_bank(
        binding.hand_spawn_cfg,
        binding.canonical_artifacts,
        source_assets=binding.source_assets,
        source_cfg=N040_PPO_SOURCE_CFG,
        device="cpu",
    )
    provider = RetainedGeometryProvider(
        artifact=artifact,
        evidence_bank=evidence_bank,
        dataset_digest=binding.dataset_sha256,
        manifest_digest=canonical_group_manifest_digest(binding.canonical_artifacts),
        canonical_schema_digest=CANONICAL_HAND_SCHEMA_V1.digest,
        evidence_source_config=asdict(N040_PPO_SOURCE_CFG),
    )
    provider.to(device)
    return provider


def resolve_structured_geometry(
    provider: RetainedGeometryProvider,
    prototype_index: torch.Tensor,
    actor_observation: StructuredActorObservation,
) -> tuple[GeometryTokenBatch, RetainedGeometryBatch]:
    r"""从structured current-q与opaque local rows计算一次共享N040$Z^e$。

    返回精简模型batch与完整graph batch；actor/critic共享前者，未来graph-aware ablation可消费后者。
    """

    q_rad = actor_observation.jnt_current[..., 0] * torch.pi
    retained = provider.resolve(prototype_index, q_rad)
    torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
        torch.all(retained.owner_valid_mask == actor_observation.owner_valid),
        "N040 owner routing disagrees with structured task",
    )
    torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
        torch.all(retained.joint_valid_mask == actor_observation.jnt_valid),
        "N040 joint routing disagrees with structured task",
    )
    return GeometryTokenBatch(retained.geometry_entities, retained.owner_valid_mask), retained


__all__ = [
    "N040_RETAINED_ARTIFACT_PATH",
    "N040_RETAINED_ARTIFACT_SHA256",
    "StructuredGeometryAssetBinding",
    "build_structured_retained_geometry_provider",
    "canonical_group_manifest_digest",
    "resolve_structured_geometry",
]
