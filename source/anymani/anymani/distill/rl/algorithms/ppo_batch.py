'Build asset-balanced PPO batches and the declared adaptive learning-rate schedule.'

from __future__ import annotations

import torch


def bounded_adaptive_learning_rate(requested_lr: float, reference_lr: float) -> float:
    'Handle bounded adaptive learning rate.'

    if not (requested_lr > 0.0 and reference_lr > 0.0):
        raise ValueError("adaptive and reference learning rates must be positive")
    return min(float(requested_lr), float(reference_lr))


def stratified_asset_permutation(
    prototype_index: torch.Tensor,
    *,
    asset_count: int,
    minibatch_count: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    'Handle stratified asset permutation; shapes [B], [B,1]; units M.'

    labels = prototype_index.reshape(-1).long()  # shapes [B]
    if asset_count < 1 or minibatch_count < 1:
        raise ValueError("asset_count and minibatch_count must be positive")
    if labels.numel() < asset_count or bool(((labels < 0) | (labels >= asset_count)).any().item()):
        raise ValueError("prototype_index contains an invalid or incomplete asset axis")


    counts = torch.bincount(labels, minlength=asset_count)  # `[A]`
    if bool((counts != counts[0]).any().item()) or int(counts[0].item()) % minibatch_count != 0:
        raise ValueError("stratified PPO requires equal per-asset counts divisible by minibatch_count")
    per_asset_per_minibatch = int(counts[0].item()) // minibatch_count

    # One row list per minibatch, partitioned by minibatch and local sample indices.
    parts: list[list[torch.Tensor]] = [[] for _ in range(minibatch_count)]
    for asset_index in range(asset_count):
        members = torch.nonzero(labels == asset_index, as_tuple=False).squeeze(-1)
        order = torch.randperm(members.numel(), device=members.device, generator=generator)
        members = members[order]
        for minibatch_index in range(minibatch_count):
            start = minibatch_index * per_asset_per_minibatch  # Offset in this asset's shuffled rows.
            stop = start + per_asset_per_minibatch
            parts[minibatch_index].append(members[start:stop])


    minibatches: list[torch.Tensor] = []
    for asset_parts in parts:
        indices = torch.cat(asset_parts, dim=0)  # shapes [B/M]; units M
        random_order = torch.randperm(indices.numel(), device=indices.device, generator=generator)
        minibatches.append(indices[random_order])
    return torch.cat(minibatches, dim=0)  # shapes [B]


def normalize_advantages_per_asset(
    advantages: torch.Tensor,
    prototype_index: torch.Tensor,
    *,
    asset_count: int,
    epsilon: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    'Normalize advantages per asset; shapes [B], [B,1], [A].'

    flat = advantages.reshape(-1)  # shapes [B]
    labels = prototype_index.reshape(-1).long()  # shapes [B]
    if flat.numel() != labels.numel():
        raise ValueError("advantage and prototype labels must align sample-by-sample")
    if asset_count < 1 or epsilon <= 0.0:
        raise ValueError("asset_count and normalization epsilon must be positive")
    if not torch.is_floating_point(flat) or not bool(torch.isfinite(flat).all().item()):
        raise ValueError("advantages must be finite floating-point values")
    if bool(((labels < 0) | (labels >= asset_count)).any().item()):
        raise ValueError("prototype labels lie outside the declared asset axis")

    counts_int = torch.bincount(labels, minlength=asset_count)  # shapes [A]
    if bool((counts_int < 2).any().item()):
        raise ValueError("per-asset advantage normalization requires at least two samples per asset")
    counts = counts_int.to(dtype=flat.dtype)
    sums = torch.zeros(asset_count, dtype=flat.dtype, device=flat.device)
    sums.scatter_add_(0, labels, flat)
    means = sums / counts  # shapes [A]
    centered = flat - means[labels]  # shapes [B]
    squared_sums = torch.zeros_like(sums)
    squared_sums.scatter_add_(0, labels, centered.square())
    variances = squared_sums / (counts - 1.0)
    standard_deviations = torch.sqrt(torch.clamp_min(variances, 0.0))
    normalized = centered / (standard_deviations[labels] + float(epsilon))
    return normalized.reshape_as(advantages), means, standard_deviations
