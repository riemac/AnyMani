r"""异构PPO的完整Actor梯度可靠性审计，不定义或修改optimizer更新。

每个资产$i$的replicas被确定性分成两半$h\in\{0,1\}$，先计算各half的mean-objective gradient
$g_{ih}$，再由样本数恢复完整资产方向$g_i$。同资产余弦用于区分可重复任务梯度与小样本符号噪声；只有前者
才允许后续把跨资产负余弦解释成可供PCGrad/FairGrad处理的稳定冲突。
"""

from __future__ import annotations

import torch
from torch import nn


def _flatten_parameter_gradients(
    parameters: tuple[nn.Parameter, ...], gradients: tuple[torch.Tensor | None, ...]
) -> torch.Tensor:
    r"""把一个objective对完整参数集的梯度拼成$[P]$，unused参数按零处理。"""

    return torch.cat(
        tuple(
            torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.reshape(-1)
            for parameter, gradient in zip(parameters, gradients, strict=True)
        )
    )  # 参数声明顺序固定，因此不同资产/half的坐标逐值对应


def per_asset_replica_half_gradients(
    objective: torch.Tensor,
    labels: torch.Tensor,
    replica_halves: torch.Tensor,
    parameters: tuple[nn.Parameter, ...],
    *,
    asset_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""计算每资产、每replica half的完整mean-objective gradient。

    Args:
        objective (torch.Tensor): per-sample Actor objective，形状$[M]$。
        labels (torch.Tensor): selection-local asset index，形状$[M]$。
        replica_halves (torch.Tensor): 每样本所属replica half，值域$\{0,1\}$，形状$[M]$。
        parameters (tuple[nn.Parameter, ...]): 完整共享Actor参数，坐标顺序必须冻结。
        asset_count (int): 当前支持资产数$A$。

    Returns:
        tuple: gradients为$[A,2,P]$，counts为$[A,2]$；每行使用该subset自己的样本均值。

    Raises:
        ValueError: 输入轴、标签、half或参数集合非法。
        RuntimeError: 当前minibatch缺少任一资产的任一replica half，无法估计自一致性。
    """

    scalar_objective = objective.reshape(-1)  # `[M]`，保留当前autograd graph
    asset_labels = labels.reshape(-1).long()  # `[M]`，只参与分组
    half_labels = replica_halves.reshape(-1).long()  # `[M]`，0/1定义两份独立replica集合
    if not parameters or scalar_objective.shape != asset_labels.shape or scalar_objective.shape != half_labels.shape:
        raise ValueError("gradient objective, asset labels, replica halves and parameters must align")
    if asset_count < 1 or bool(((asset_labels < 0) | (asset_labels >= asset_count)).any().item()):
        raise ValueError("gradient audit asset labels lie outside the declared axis")
    if bool(((half_labels < 0) | (half_labels > 1)).any().item()):
        raise ValueError("gradient audit replica halves must contain only 0 or 1")

    rows: list[torch.Tensor] = []  # asset-major，每项最终为`[2,P]`
    counts = torch.zeros(asset_count, 2, dtype=torch.long, device=scalar_objective.device)  # `[A,2]`
    for asset_index in range(asset_count):
        half_rows: list[torch.Tensor] = []
        for half_index in range(2):
            member = (asset_labels == asset_index) & (half_labels == half_index)  # 当前$(i,h)$样本集合
            member_count = int(member.sum().item())
            if member_count < 1:
                raise RuntimeError(f"gradient audit lacks asset={asset_index}, replica_half={half_index}")
            counts[asset_index, half_index] = member_count
            gradients = torch.autograd.grad(
                scalar_objective[member].mean(),
                parameters,
                retain_graph=True,
                allow_unused=True,
            )  # 只读梯度，不写parameter.grad或执行optimizer
            half_rows.append(_flatten_parameter_gradients(parameters, gradients))
        rows.append(torch.stack(half_rows, dim=0))
    return torch.stack(rows, dim=0), counts  # `[A,2,P]`, `[A,2]`


def _cosine_rows(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    r"""逐行计算余弦；任一零范数行返回0，避免伪造同向或反向证据。"""

    numerator = (left * right).sum(dim=-1)  # `[A]`逐资产dot
    denominator = torch.linalg.vector_norm(left, dim=-1) * torch.linalg.vector_norm(right, dim=-1)
    return torch.where(denominator > 1.0e-20, numerator / denominator.clamp_min(1.0e-20), torch.zeros_like(numerator))


def compute_actor_gradient_scope_audit(
    half_gradients: torch.Tensor,
    half_counts: torch.Tensor,
) -> dict[str, torch.Tensor]:
    r"""由$[A,2,P]$ half gradients形成自一致性、跨资产Gram与合成抵消统计。

    完整资产gradient按subset样本数加权：

    $$
    g_i=\frac{n_{i0}g_{i0}+n_{i1}g_{i1}}{n_{i0}+n_{i1}}.
    $$

    ``aggregate_to_individual_norm_ratio``定义为$\|\sum_i g_i\|/\sum_i\|g_i\|$；1表示完全同向，接近0表示
    强抵消。它不自动区分真实任务冲突与共同噪声，必须结合half self-cosine解释。
    """

    if half_gradients.ndim != 3 or half_gradients.shape[1] != 2:
        raise ValueError("half gradients must have shape [asset,2,parameter]")
    if half_counts.shape != half_gradients.shape[:2] or bool((half_counts <= 0).any().item()):
        raise ValueError("half counts must be positive and align with [asset,2]")
    weights = half_counts.to(dtype=half_gradients.dtype).unsqueeze(-1)  # `[A,2,1]`
    asset_gradients = (half_gradients * weights).sum(dim=1) / weights.sum(dim=1)  # `[A,P]`
    asset_norms = torch.linalg.vector_norm(asset_gradients, dim=-1)  # `[A]`
    half_norms = torch.linalg.vector_norm(half_gradients, dim=-1)  # `[A,2]`
    half_self_cosine = _cosine_rows(half_gradients[:, 0], half_gradients[:, 1])  # `[A]`
    gram = asset_gradients @ asset_gradients.T  # `[A,A]`，完整Actor逐资产Gram
    norm_outer = asset_norms[:, None] * asset_norms[None, :]
    cosine = torch.where(norm_outer > 1.0e-20, gram / norm_outer.clamp_min(1.0e-20), torch.zeros_like(gram))
    off_diagonal = ~torch.eye(asset_gradients.shape[0], dtype=torch.bool, device=asset_gradients.device)
    pair_values = cosine[torch.triu(off_diagonal, diagonal=1)]  # 每个无序资产pair恰一次
    positive = asset_norms > 1.0e-12
    if bool(positive.any().item()):
        positive_min = asset_norms[positive].min()
        norm_span = asset_norms.max() / positive_min
        if not bool(positive.all().item()):
            norm_span = torch.full_like(norm_span, torch.inf)  # 部分零gradient表示无限尺度跨度
    else:
        norm_span = asset_norms.new_tensor(1.0)  # 全零表示probe无信息，不伪造无限冲突
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
