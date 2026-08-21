r"""固定 query/sigma 上密度预测对物理 $q$ 的抽样边导数。

五项 method objective 需要同一密度预测器的构型自导数

$$
\hat g_{e,\ell}^{auto}
=
\frac{\partial\hat\rho_{g,r,\ell}}{\partial q_i},
$$

单位 $\mathrm{rad}^{-1}$。本模块只提供这条自动微分原语；density、$\kappa$、derived-field、
Sobolev 与 chain 的比较公式由 `distill.methods.multi_anchor_gaussian_implicit_field` 拥有。
"""

from __future__ import annotations

import torch


def selected_density_coordinate_derivative(
    density_prediction: torch.Tensor,
    q: torch.Tensor,
    owner_index: torch.Tensor,
    query_index: torch.Tensor,
    joint_index: torch.Tensor,
    *,
    create_graph: bool,
) -> torch.Tensor:
    r"""计算同一密度预测器在抽样边上的逐带宽构型导数。

    对边 $e=(g,r,i)$ 与带宽 $\ell$：

    $$
    \hat g_{e,\ell}^{auto}
    =
    \frac{\partial\hat\rho_{g,r,\ell}}{\partial q_i}.
    $$

    模型不沿批次轴混合样本，因此对批次输出求和后对 $q$ 求梯度，得到的每一行仍是
    该样本自身的导数。实现只循环抽样的 $E\times L$，不构造完整 Jacobian。

    Args:
        density_prediction (torch.Tensor): ``[B,G,N_Q,N_sigma]`` 的同一密度解码器输出。
        q (torch.Tensor): ``[B,N_J]`` 的物理关节角，单位 rad，必须 ``requires_grad=True``。
        owner_index (torch.Tensor): ``[E]`` 或跨结构 padding 后 ``[B,E]`` 抽样归属体索引。
        query_index (torch.Tensor): 与 owner selector 同形状的查询点索引。
        joint_index (torch.Tensor): 与 owner selector 同形状的 JOINT 索引。
        create_graph (bool): 是否保留导数图；训练 Sobolev/chain 时必须为 ``True``。

    Returns:
        torch.Tensor: ``[B,E,L]`` 的 $\hat g^{auto}$，单位 $\mathrm{rad}^{-1}$。

    Raises:
        ValueError: 当物理 $q$ 未启用梯度，或 selector 形状不闭合时抛出。
    """

    if not q.requires_grad:
        raise ValueError("q.requires_grad must be True for Sobolev/JVP supervision")
    if owner_index.ndim not in {1, 2} or query_index.shape != owner_index.shape or joint_index.shape != owner_index.shape:
        raise ValueError("owner/query/joint selectors must share [E] or [B,E] shape")
    if owner_index.ndim == 2 and owner_index.shape[0] != density_prediction.shape[0]:
        raise ValueError("batched selectors must share B with density_prediction")
    edge_count = owner_index.shape[-1]  # 每个样本的抽样边存储预算 $E$
    bandwidth_count = density_prediction.shape[-1]  # 当前显式 sigma 数据轴 $N_\sigma$
    batch_index = torch.arange(density_prediction.shape[0], device=density_prediction.device)
    per_edge: list[torch.Tensor] = []  # 每项最终为 `[B,L]`
    for edge in range(edge_count):
        per_band: list[torch.Tensor] = []  # 当前归属体/查询点边的全部带宽导数
        for band in range(bandwidth_count):
            if owner_index.ndim == 1:
                selected = density_prediction[:, owner_index[edge], query_index[edge], band]  # `[B]`
            else:
                selected = density_prediction[
                    batch_index, owner_index[:, edge], query_index[:, edge], band
                ]  # 每个样本自己的 sampled edge
            gradient = torch.autograd.grad(
                selected.sum(),
                q,
                create_graph=create_graph,
                retain_graph=True,
                allow_unused=False,
            )[0]  # `[B,N_J]`，物理 q 求导自动包含 q/pi 的 $1/\pi$
            if joint_index.ndim == 1:
                per_band.append(gradient[:, joint_index[edge]])  # 同结构共享 JOINT selector
            else:
                per_band.append(gradient.gather(1, joint_index[:, edge : edge + 1]).squeeze(1))
        per_edge.append(torch.stack(per_band, dim=-1))  # `[B,L]`
    return torch.stack(per_edge, dim=1)  # `[B,E,L]`


__all__ = ["selected_density_coordinate_derivative"]
