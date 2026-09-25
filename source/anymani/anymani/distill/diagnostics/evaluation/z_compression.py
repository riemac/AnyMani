'Fit a shared token-space PCA basis from valid training entities and evaluate fixed validation tokens.'

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch

from anymani.distill.models.geometry_ssl import GeometrySSLForward, GeometrySSLModel
from anymani.distill.models.input_adapters.geometry import GeometryLatents


@dataclass(frozen=True)
class UnifiedPCABasis:
    'Contract for unified pcabasis.'

    mean: torch.Tensor  # `[D]`
    components: torch.Tensor  # shapes [D,D]
    eigenvalues: torch.Tensor  # `[D]`
    sample_count: int

    def __post_init__(self) -> None:
        width = self.mean.numel()
        if self.mean.shape != (width,) or self.components.shape != (width, width):
            raise ValueError("unified PCA mean/components must have shapes [D] and [D,D]")
        if self.eigenvalues.shape != (width,) or self.sample_count < 2:
            raise ValueError("unified PCA requires D eigenvalues and at least two valid tokens")


class UnifiedPCAAccumulator:
    'Contract for unified pcaaccumulator.'

    def __init__(self, width: int) -> None:
        if width < 1:
            raise ValueError("unified PCA width must be positive")
        self.width = width
        self.count = 0
        self.sum = torch.zeros(width, dtype=torch.float64)
        self.second = torch.zeros(width, width, dtype=torch.float64)

    def update(self, entities: torch.Tensor, valid_mask: torch.Tensor) -> None:
        'Update the declared contract; shapes [B,G,D].'

        if entities.ndim != 3 or entities.shape[-1] != self.width:
            raise ValueError("PCA entities must have shape [B,G,D]")
        if valid_mask.shape != entities.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError("PCA valid_mask must have bool shape [B,G]")
        selected = entities.detach()[valid_mask].to(device="cpu", dtype=torch.float64)
        if selected.numel() == 0:
            return
        self.count += selected.shape[0]
        self.sum += selected.sum(dim=0)
        self.second += selected.transpose(0, 1) @ selected

    def state_dict(self) -> dict[str, Any]:
        'Handle state dict.'

        return {"width": self.width, "count": self.count, "sum": self.sum, "second": self.second}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        'Load state dict.'

        if state.get("width") != self.width or not isinstance(state.get("count"), int):
            raise ValueError("unified PCA state width/count mismatch")
        total = state.get("sum")
        second = state.get("second")
        if not isinstance(total, torch.Tensor) or not isinstance(second, torch.Tensor):
            raise ValueError("unified PCA state lacks tensor sufficient statistics")
        if total.shape != self.sum.shape or second.shape != self.second.shape:
            raise ValueError("unified PCA state tensor shapes mismatch")
        self.count = int(state["count"])
        self.sum.copy_(total.to(torch.float64))
        self.second.copy_(second.to(torch.float64))

    def finalize(self) -> UnifiedPCABasis:
        'Handle finalize.'

        if self.count < 2:
            raise ValueError("unified PCA requires at least two valid entity tokens")
        mean = self.sum / self.count
        covariance = (self.second - self.count * torch.outer(mean, mean)) / (self.count - 1)
        covariance = 0.5 * (covariance + covariance.transpose(0, 1))
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        order = torch.arange(self.width - 1, -1, -1)
        return UnifiedPCABasis(
            mean=mean,
            components=eigenvectors[:, order].transpose(0, 1).contiguous(),
            eigenvalues=eigenvalues[order].clamp_min(0.0),
            sample_count=self.count,
        )


def reconstruct_unified_entities(
    entities: torch.Tensor,
    valid_mask: torch.Tensor,
    basis: UnifiedPCABasis,
    *,
    rank: int,
) -> torch.Tensor:
    'Handle reconstruct unified entities.'

    width = entities.shape[-1]
    if entities.ndim != 3 or valid_mask.shape != entities.shape[:2]:
        raise ValueError("unified PCA reconstruction requires [B,G,D] entities and [B,G] mask")
    if basis.mean.shape != (width,) or not 1 <= rank < width:
        raise ValueError("compression rank must lie in [1,D); D-dimensional reference must use original Z")
    mean = basis.mean.to(device=entities.device, dtype=entities.dtype)
    components = basis.components[:rank].to(device=entities.device, dtype=entities.dtype)
    centered = entities - mean
    reconstruction = (centered @ components.transpose(0, 1)) @ components + mean
    return reconstruction * valid_mask.unsqueeze(-1)


def unified_pca_basis_digest(basis: UnifiedPCABasis) -> str:
    'Handle unified pca basis digest.'

    digest = hashlib.sha256(b"anymani-unified-pca-basis-v1\0")
    for name, tensor in (("mean", basis.mean), ("components", basis.components), ("eigenvalues", basis.eigenvalues)):
        array = tensor.detach().to(device="cpu", dtype=torch.float64).contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(tuple(array.shape)).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def decode_z_compression_ranks(
    model: GeometrySSLModel,
    reference: GeometrySSLForward,
    *,
    basis: UnifiedPCABasis,
    ranks: tuple[int, ...],
    entity_valid_mask: torch.Tensor,
    bandwidths: torch.Tensor,
    joint_entity_index: torch.Tensor,
    owner_index: torch.Tensor,
    query_index: torch.Tensor,
    joint_index: torch.Tensor,
) -> dict[int, GeometrySSLForward]:
    'Handle decode z compression ranks.'

    width = reference.latents.entities.shape[-1]
    outputs: dict[int, GeometrySSLForward] = {width: reference}
    for rank in ranks:
        if rank == width:
            continue
        reconstructed = reconstruct_unified_entities(
            reference.latents.entities,
            entity_valid_mask,
            basis,
            rank=rank,
        )
        outputs[rank] = model.decode_latents(
            GeometryLatents(reconstructed),
            reference.query_features,
            bandwidths=bandwidths,
            entity_valid_mask=entity_valid_mask,
            joint_entity_index=joint_entity_index,
            owner_index=owner_index,
            query_index=query_index,
            joint_index=joint_index,
        )
    return outputs


__all__ = [
    "UnifiedPCAAccumulator",
    "UnifiedPCABasis",
    "decode_z_compression_ranks",
    "reconstruct_unified_entities",
    "unified_pca_basis_digest",
]
