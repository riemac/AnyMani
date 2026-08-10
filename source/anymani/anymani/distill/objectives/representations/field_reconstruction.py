r"""多锚点条件隐式场的联合密度、一阶与 Sobolev objective。

第一版同时约束函数值、显式距离灵敏度、链式派生场灵敏度、同一密度预测器的构型
自导数和两条一阶路径的一致性。objective 只消费 prediction/target/mask，不生成查询点、
最近点或运动学教师信号。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ...representations.targets.field_samples import FieldTargetBatch, SensitivityTargetBatch


@dataclass(frozen=True)
class GeometrySSLWeights:
    r"""第一版强默认联合损失的无量纲权重。"""

    density: float = 1.0
    kappa: float = 1.0
    derived_field: float = 1.0
    sobolev: float = 1.0
    chain: float = 1.0

    def __post_init__(self) -> None:
        r"""所有项必须非负；正式配置中的强默认会在 calibration 后写入正值。"""

        values = (self.density, self.kappa, self.derived_field, self.sobolev, self.chain)
        if any(value < 0.0 for value in values):
            raise ValueError("Geometry SSL loss weights must be non-negative")


@dataclass(frozen=True)
class GeometrySSLTerms:
    r"""联合损失及两条场灵敏度预测路径的审计输出。"""

    total: torch.Tensor  # 加权总损失，标量
    density: torch.Tensor  # $\mathcal L_{density}$
    kappa: torch.Tensor  # $\mathcal L_\kappa$
    derived_field: torch.Tensor  # $\mathcal L_g^{(\kappa)}$
    sobolev: torch.Tensor  # $\mathcal L_{Sob}$
    chain: torch.Tensor  # $\mathcal L_{chain}$
    derived_field_sensitivity: torch.Tensor  # `[B,E,L]` 的 $\hat g^{(\kappa)}$
    auto_field_sensitivity: torch.Tensor  # `[B,E,L]` 的 $\hat g^{auto}$


def _masked_mean_square(error: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    r"""按有效物理标量数归一化平方误差。"""

    while mask.ndim < error.ndim:
        mask = mask.unsqueeze(-1)  # 把 `[B,...]` mask 广播到 bandwidth 等尾轴
    weight = mask.to(error.dtype).expand_as(error)  # 每个有效标量通道权重为 1
    denominator = weight.sum()  # 有效标量 target 总数
    if int(denominator.detach().item()) == 0:
        raise ValueError("masked objective received no valid targets")
    return torch.sum(weight * error.square()) / denominator  # 对有效标量取均值


def selected_density_coordinate_derivative(
    density_prediction: torch.Tensor,
    q: torch.Tensor,
    owner_index: torch.Tensor,
    query_index: torch.Tensor,
    joint_index: torch.Tensor,
    *,
    create_graph: bool,
) -> torch.Tensor:
    r"""计算同一密度预测器在 sampled edges 上的逐带宽构型导数。

    对 edge $e=(g,r,i)$ 与带宽 $\ell$：

    $$
    \hat g_{e,\ell}^{auto}
    =
    \frac{\partial\hat\rho_{g,r,\ell}}{\partial q_i}.
    $$

    模型不沿 batch 轴混合样本，因此对 batch 输出求和后对 q 求梯度，得到的每一行仍是
    该样本自身的导数。实现只循环 sampled $E\times L$，不构造完整 Jacobian。
    """

    if not q.requires_grad:
        raise ValueError("q.requires_grad must be True for Sobolev/JVP supervision")
    edge_count = owner_index.numel()  # sampled edge 数 $E$
    bandwidth_count = density_prediction.shape[-1]  # Gaussian 通道数 $L$
    per_edge: list[torch.Tensor] = []  # 每项最终为 `[B,L]`
    for edge in range(edge_count):
        per_band: list[torch.Tensor] = []  # 当前 owner/query edge 的全部带宽导数
        for band in range(bandwidth_count):
            selected = density_prediction[:, owner_index[edge], query_index[edge], band]  # `[B]`
            gradient = torch.autograd.grad(
                selected.sum(),
                q,
                create_graph=create_graph,
                retain_graph=True,
                allow_unused=False,
            )[0]  # `[B,N_J]`，物理 q 求导自动包含 q/pi 的 $1/\pi$
            per_band.append(gradient[:, joint_index[edge]])  # `[B]`，sampled JOINT 坐标
        per_edge.append(torch.stack(per_band, dim=-1))  # `[B,L]`
    return torch.stack(per_edge, dim=1)  # `[B,E,L]`


class GeometrySSLObjective(nn.Module):
    r"""协同计算密度、显式 κ、派生 g、Sobolev 与链式一致性。"""

    def __init__(self, weights: GeometrySSLWeights) -> None:
        super().__init__()
        self.weights = weights

    def forward(
        self,
        *,
        q: torch.Tensor,
        density_prediction: torch.Tensor,
        kappa_prediction: torch.Tensor,
        field_targets: FieldTargetBatch,
        sensitivity_targets: SensitivityTargetBatch,
    ) -> GeometrySSLTerms:
        r"""计算第一版完整强默认 objective，并返回可分别记录的各项。"""

        if density_prediction.shape != field_targets.density.shape:
            raise ValueError("density prediction and target must share shape [B,G,N_Q,L]")
        if kappa_prediction.shape != sensitivity_targets.kappa.shape:
            raise ValueError("kappa prediction and target must share shape [B,E]")

        density_loss = _masked_mean_square(
            density_prediction - field_targets.density,
            field_targets.valid_mask,
        )  # $\mathcal L_{density}$，全部 query strata/bands 都受监督
        kappa_loss = _masked_mean_square(
            kappa_prediction - sensitivity_targets.kappa,
            sensitivity_targets.valid_mask,
        )  # $\mathcal L_\kappa$，包含 non-ancestor structural zeros

        owner_index = sensitivity_targets.owner_index  # `[E]` sampled owner selectors
        query_index = sensitivity_targets.query_index  # `[E]` sampled query selectors
        selected_density = density_prediction[:, owner_index, query_index]  # `[B,E,L]`
        selected_distance = field_targets.distance[:, owner_index, query_index]  # `[B,E]`，m
        inverse_sigma_squared = field_targets.bandwidths.square().reciprocal()  # `[L]`，m⁻²
        derived_field = (
            -selected_distance.unsqueeze(-1)
            * inverse_sigma_squared
            * selected_density
            * kappa_prediction.unsqueeze(-1)
        )  # $\hat g^{(\kappa)}=-(d/\sigma^2)\hat\rho\hat\kappa$，`[B,E,L]`
        derived_field_loss = _masked_mean_square(
            derived_field - sensitivity_targets.field_sensitivity,
            sensitivity_targets.valid_mask,
        )  # $\mathcal L_g^{(\kappa)}$

        auto_field = selected_density_coordinate_derivative(
            density_prediction,
            q,
            owner_index,
            query_index,
            sensitivity_targets.joint_index,
            create_graph=True,
        )  # `[B,E,L]`，同一 density predictor 对固定 `{h}` query 的自导数
        sobolev_loss = _masked_mean_square(
            auto_field - sensitivity_targets.field_sensitivity,
            sensitivity_targets.valid_mask,
        )  # $\mathcal L_{Sob}$
        chain_loss = _masked_mean_square(
            derived_field - auto_field,
            sensitivity_targets.valid_mask,
        )  # $\mathcal L_{chain}$，连接显式一阶 head 与密度函数切向

        total = (
            self.weights.density * density_loss
            + self.weights.kappa * kappa_loss
            + self.weights.derived_field * derived_field_loss
            + self.weights.sobolev * sobolev_loss
            + self.weights.chain * chain_loss
        )  # 第一版联合标量 objective
        return GeometrySSLTerms(
            total=total,
            density=density_loss,
            kappa=kappa_loss,
            derived_field=derived_field_loss,
            sobolev=sobolev_loss,
            chain=chain_loss,
            derived_field_sensitivity=derived_field,
            auto_field_sensitivity=auto_field,
        )


__all__ = [
    "GeometrySSLObjective",
    "GeometrySSLTerms",
    "GeometrySSLWeights",
    "selected_density_coordinate_derivative",
]
