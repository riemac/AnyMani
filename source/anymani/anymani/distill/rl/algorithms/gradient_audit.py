'Audit per-asset Actor gradients without changing optimizer updates.'

from __future__ import annotations

import torch
from torch import nn


def _flatten_parameter_gradients(
    parameters: tuple[nn.Parameter, ...], gradients: tuple[torch.Tensor | None, ...]
) -> torch.Tensor:
    'Handle flatten parameter gradients; shapes [P].'

    return torch.cat(
        tuple(
            torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.reshape(-1)
            for parameter, gradient in zip(parameters, gradients, strict=True)
        )
    )


def per_asset_replica_half_gradients(
    objective: torch.Tensor,
    labels: torch.Tensor,
    replica_halves: torch.Tensor,
    parameters: tuple[nn.Parameter, ...],
    *,
    asset_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    'Handle per asset replica half gradients; shapes [M], [A,2,P], [A,2]; units M.'

    scalar_objective = objective.reshape(-1)  # shapes [M]; units M
    asset_labels = labels.reshape(-1).long()  # shapes [M]; units M
    half_labels = replica_halves.reshape(-1).long()  # shapes [M]; units M
    if not parameters or scalar_objective.shape != asset_labels.shape or scalar_objective.shape != half_labels.shape:
        raise ValueError("gradient objective, asset labels, replica halves and parameters must align")
    if asset_count < 1 or bool(((asset_labels < 0) | (asset_labels >= asset_count)).any().item()):
        raise ValueError("gradient audit asset labels lie outside the declared axis")
    if bool(((half_labels < 0) | (half_labels > 1)).any().item()):
        raise ValueError("gradient audit replica halves must contain only 0 or 1")

    rows: list[torch.Tensor] = []  # shapes [2,P]
    counts = torch.zeros(asset_count, 2, dtype=torch.long, device=scalar_objective.device)  # `[A,2]`
    for asset_index in range(asset_count):
        half_rows: list[torch.Tensor] = []
        for half_index in range(2):
            member = (asset_labels == asset_index) & (half_labels == half_index)
            member_count = int(member.sum().item())
            if member_count < 1:
                raise RuntimeError(f"gradient audit lacks asset={asset_index}, replica_half={half_index}")
            counts[asset_index, half_index] = member_count
            gradients = torch.autograd.grad(
                scalar_objective[member].mean(),
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            half_rows.append(_flatten_parameter_gradients(parameters, gradients))
        rows.append(torch.stack(half_rows, dim=0))
    return torch.stack(rows, dim=0), counts  # `[A,2,P]`, `[A,2]`


def _cosine_rows(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    'Handle cosine rows.'

    numerator = (left * right).sum(dim=-1)  # shapes [A]
    denominator = torch.linalg.vector_norm(left, dim=-1) * torch.linalg.vector_norm(right, dim=-1)
    return torch.where(denominator > 1.0e-20, numerator / denominator.clamp_min(1.0e-20), torch.zeros_like(numerator))


def compute_actor_gradient_scope_audit(
    half_gradients: torch.Tensor,
    half_counts: torch.Tensor,
) -> dict[str, torch.Tensor]:
    'Compute Actor gradient scope audit; shapes [A,2,P].'

    if half_gradients.ndim != 3 or half_gradients.shape[1] != 2:
        raise ValueError("half gradients must have shape [asset,2,parameter]")
    if half_counts.shape != half_gradients.shape[:2] or bool((half_counts <= 0).any().item()):
        raise ValueError("half counts must be positive and align with [asset,2]")
    weights = half_counts.to(dtype=half_gradients.dtype).unsqueeze(-1)  # `[A,2,1]`
    asset_gradients = (half_gradients * weights).sum(dim=1) / weights.sum(dim=1)  # `[A,P]`
    asset_norms = torch.linalg.vector_norm(asset_gradients, dim=-1)  # `[A]`
    half_norms = torch.linalg.vector_norm(half_gradients, dim=-1)  # `[A,2]`
    half_self_cosine = _cosine_rows(half_gradients[:, 0], half_gradients[:, 1])  # `[A]`
    gram = asset_gradients @ asset_gradients.T  # shapes [A,A]
    norm_outer = asset_norms[:, None] * asset_norms[None, :]
    cosine = torch.where(norm_outer > 1.0e-20, gram / norm_outer.clamp_min(1.0e-20), torch.zeros_like(gram))
    off_diagonal = ~torch.eye(asset_gradients.shape[0], dtype=torch.bool, device=asset_gradients.device)
    pair_values = cosine[torch.triu(off_diagonal, diagonal=1)]
    positive = asset_norms > 1.0e-12
    if bool(positive.any().item()):
        positive_min = asset_norms[positive].min()
        norm_span = asset_norms.max() / positive_min
        if not bool(positive.all().item()):
            norm_span = torch.full_like(norm_span, torch.inf)
    else:
        norm_span = asset_norms.new_tensor(1.0)
    norm_sum = asset_norms.sum().clamp_min(1.0e-20)
    aggregate_ratio = torch.linalg.vector_norm(asset_gradients.sum(dim=0)) / norm_sum
    return {
        "half_counts": half_counts,
        "half_norms": half_norms,
        "half_self_cosine": half_self_cosine,
        "reliable_asset_fraction": (half_self_cosine > 0.0).float().mean(),
        "asset_gradients": asset_gradients,
        "asset_norms": asset_norms,
        "asset_gram": gram,
        "asset_cosine": cosine,
        "negative_pair_fraction": (
            (pair_values < 0.0).float().mean() if pair_values.numel() else asset_norms.new_tensor(0.0)
        ),
        "top1_norm_fraction": asset_norms.max() / norm_sum,
        "norm_span": norm_span,
        "aggregate_to_individual_norm_ratio": aggregate_ratio,
    }


__all__ = ["compute_actor_gradient_scope_audit", "per_asset_replica_half_gradients"]
