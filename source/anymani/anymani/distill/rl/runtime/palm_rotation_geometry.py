'Run the frozen N040 encoder with scoped BF16 CUDA inference and return FP32 owner tokens.'

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from anymani.distill.models.palm_rotation_policy import PalmRotationActorObservation, PalmRotationGeometry

from .retained_geometry import RetainedGeometryProvider
from .structured_geometry import (
    N040_RETAINED_ARTIFACT_SHA256,
    StructuredGeometryAssetBinding,
    build_structured_retained_geometry_provider,
)


def _identity_digest(payload: dict[str, Any]) -> str:
    'Handle identity digest.'

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class PalmRotationBf16GeometryProvider(nn.Module):
    'Contract for PALM rotation bf16 geometry provider.'

    def __init__(self, provider: RetainedGeometryProvider) -> None:
        'Initialize the instance.'

        super().__init__()
        self.provider = provider
        if any(parameter.requires_grad for parameter in provider.encoder.parameters()):
            raise ValueError("palm-rotation rollout cache requires a strictly frozen N040 encoder")
        if any(parameter.dtype != torch.float32 for parameter in provider.encoder.parameters()):
            raise ValueError("N040 retained master weights must remain FP32")
        self.resolve_call_count = 0
        base_identity = provider.identity
        precision = {
            "master_weight_dtype": "float32",
            "encoder_compute": "cuda_bfloat16_autocast",
            "output_dtype": "float32",
            "actor_critic_compute": "float32",
            "tf32": False,
        }
        identity = {
            "identity_schema_version": "2.0.0",
            "provider_type": "retained_se3_geometry_encoder_scoped_precision",
            "base_provider_identity_digest": base_identity["identity_digest"],
            "retained_artifact": base_identity["retained_artifact"],
            "asset_ids": base_identity["asset_ids"],
            "physical_geometry_hashes": base_identity["physical_geometry_hashes"],
            "precision": precision,
        }
        self._identity = {**identity, "identity_digest": _identity_digest(identity)}

    @property
    def identity(self) -> dict[str, Any]:
        'Handle identity.'

        return json.loads(json.dumps(self._identity))

    def train(self, mode: bool = True) -> PalmRotationBf16GeometryProvider:
        'Train the declared contract.'

        super().train(mode)
        self.provider.eval()
        return self

    def resolve(
        self,
        prototype_index: torch.Tensor,
        actor_observation: PalmRotationActorObservation,
    ) -> PalmRotationGeometry:
        'Resolve the declared contract.'

        if any(parameter.requires_grad for parameter in self.provider.encoder.parameters()):
            raise RuntimeError("rollout-cached N040 cannot be unfrozen")
        q_rad = actor_observation.jnt_current[..., 0] * torch.pi  # shapes [B,16]; units rad
        enabled = q_rad.is_cuda
        self.resolve_call_count += 1
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled):
            retained = self.provider.resolve(prototype_index, q_rad)
        torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
            torch.all(retained.owner_valid_mask == actor_observation.owner_valid),
            "BF16 N040 owner routing disagrees with actor observation",
        )
        torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
            torch.all(retained.joint_valid_mask == actor_observation.jnt_valid),
            "BF16 N040 joint routing disagrees with actor observation",
        )
        return PalmRotationGeometry(
            tokens=retained.geometry_entities.float(),
            owner_valid=retained.owner_valid_mask,
            shortest_path=retained.shortest_path,
            parent_direction=retained.parent_direction,
            child_direction=retained.child_direction,
        )


def build_palm_rotation_bf16_geometry_provider(
    binding: StructuredGeometryAssetBinding,
    *,
    artifact_path: Path | None = None,
    artifact_sha256: str = N040_RETAINED_ARTIFACT_SHA256,
    device: torch.device | str = "cpu",
) -> PalmRotationBf16GeometryProvider:
    'Build PALM rotation bf16 geometry provider.'

    provider = build_structured_retained_geometry_provider(
        binding,
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha256,
        device=device,
    )
    return PalmRotationBf16GeometryProvider(provider).to(device)


__all__ = [
    "PalmRotationBf16GeometryProvider",
    "build_palm_rotation_bf16_geometry_provider",
]
