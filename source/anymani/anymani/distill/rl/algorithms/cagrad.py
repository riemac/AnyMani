'Combine per-asset PPO task gradients while retaining the mean gradient as the anchor.'

from __future__ import annotations

import torch


@torch.no_grad()
def cagrad_coefficients(
    gram: torch.Tensor,
    *,
    c: float = 0.4,
    iterations: int = 64,
    tolerance: float = 1e-3,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    'Handle cagrad coefficients; shapes [K,K].'
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or gram.shape[0] < 1:
        raise ValueError("CAGrad requires a nonempty square task Gram")
    if not 0 <= c < 1 or iterations < 1 or tolerance <= 0:
        raise ValueError("CAGrad requires 0<=c<1 and positive solver controls")
    torch._assert_async(torch.isfinite(gram).all(), "CAGrad Gram must be finite")  # pyright: ignore[reportPrivateImportUsage]

    q = (gram.detach().double() + gram.detach().double().T) * 0.5
    scale = q.diagonal().mean().clamp_min(1e-30)
    q = q / scale
    k = q.shape[0]
    uniform = torch.full((k,), 1.0 / k, dtype=q.dtype, device=q.device)
    mean_products = q @ uniform
    mean_norm_sq = (uniform @ mean_products).clamp_min(0)
    radius = c * mean_norm_sq.sqrt()
    weights = uniform.clone()
    eps = 1e-12
    used = 0
    for used in range(1, iterations + 1):
        products = q @ weights
        norm_sq = (weights @ products).clamp_min(0)
        norm = (norm_sq + eps).sqrt()
        local_improvements = mean_products + radius * products / norm
        dual = weights @ mean_products + radius * norm_sq.sqrt()
        gap = (dual - local_improvements.min()).clamp_min(0)
        if used % 4 == 1 and float(gap) <= tolerance * max(float(mean_norm_sq), 1e-12):
            break
        index = local_improvements.argmin(keepdim=True)
        qi = q.diagonal().gather(0, index).squeeze(0)
        vi = products.gather(0, index).squeeze(0)
        linear = mean_products.gather(0, index).squeeze(0) - weights @ mean_products

        a = (qi - 2 * vi + norm_sq).clamp_min(0)
        b = vi - norm_sq
        discriminant = (a * (norm_sq + eps) - b.square()).clamp_min(0)
        denominator = radius.square() * a - linear.square()
        root = (-linear * (discriminant / denominator.clamp_min(eps)).sqrt() - b) / a.clamp_min(eps)
        root = root.clamp(0, 1)
        candidates = torch.stack((torch.zeros_like(root), torch.ones_like(root), root))
        costs = (
            linear * candidates
            + radius * (norm_sq + 2 * b * candidates + a * candidates.square() + eps).clamp_min(eps).sqrt()
        )
        step = candidates[costs.argmin()]
        weights.mul_(1 - step)
        weights.scatter_add_(0, index, step.reshape(1))

    products = q @ weights
    norm_sq = (weights @ products).clamp_min(0)
    norm = (norm_sq + eps).sqrt()
    unscaled = uniform + radius * weights / norm
    primal = (q @ unscaled).min()
    dual = weights @ mean_products + radius * norm_sq.sqrt()
    diagnostics = {
        "primal_dual_gap": (dual - primal).clamp_min(0) * scale,
        "relative_gap": (dual - primal).clamp_min(0) / mean_norm_sq.clamp_min(1e-12),
        "worst_projection": primal * scale,
        "mean_gradient_norm_sq": mean_norm_sq * scale,
        "iterations": torch.tensor(used, device=q.device),
    }
    return (unscaled / (1 + c * c)).to(gram.dtype), diagnostics


@torch.no_grad()
def combine_task_gradients(
    gradients: dict[str, torch.Tensor],
    *,
    c: float = 0.4,
    iterations: int = 64,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    'Handle combine task gradients; shapes [K,*parameter_shape].'
    if not gradients:
        raise ValueError("CAGrad needs task gradients")
    first = next(iter(gradients.values()))
    tasks = first.shape[0]
    gram = torch.zeros(tasks, tasks, dtype=first.dtype, device=first.device)

    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        for value in gradients.values():
            if value.shape[0] != tasks:
                raise ValueError("task gradient axes disagree")
            flat = value.reshape(tasks, -1)
            gram.addmm_(flat, flat.T)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    coefficients, diagnostics = cagrad_coefficients(gram, c=c, iterations=iterations)
    combined = {
        name: torch.einsum("k,kp->p", coefficients, value.reshape(tasks, -1)).reshape(value.shape[1:])
        for name, value in gradients.items()
    }
    return combined, diagnostics
