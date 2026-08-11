r"""多锚点整手条件隐式场的联合物理目标。

本模块位于 ``distill.objectives``，只负责把模型预测、物理真值、掩码与权重归约为标量损失。
查询点采样、最近点、归属体表面、点 Jacobian 与解析教师信号由
``distill.representations`` 生成；可学习编码器与解码器由 ``distill.models`` 拥有。
这种拆分保证每个物理项可以独立关闭、替换或审计，而不把目标生成器和网络绑成不可消融单体。

对批大小 $B$、表面归属体 $G=N_E$、每归属体查询数 $N_Q$、Gaussian 带宽数 $L$，
零阶预测与真值形状为 ``[B,G,N_Q,L]``：

$$
\rho_{\sigma_\ell,g}(x;q)
=
\exp\left(-\frac{d_g(x;q)^2}{2\sigma_\ell^2}\right),
\qquad
\mathcal L_{density}
=
\mathbb E\left[\left|\hat\rho-\rho\right|^2\right].
$$

一阶监督只实际生成 $E$ 条抽样归属体—查询点—JOINT 边，不构造完整
``[B,G,N_Q,L,N_J]`` Jacobian。距离灵敏度和场灵敏度分别为：

$$
\kappa_{g,i}(x;q)
=
\frac{\partial d_g(x;q)}{\partial q_i}
\quad[\mathrm{m/rad}],
$$

$$
g_{\sigma_\ell,g,i}(x;q)
=
\frac{\partial\rho_{\sigma_\ell,g}(x;q)}{\partial q_i}
=
-\frac{d_g}{\sigma_\ell^2}\rho_{\sigma_\ell,g}\kappa_{g,i}
\quad[\mathrm{rad}^{-1}].
$$

第一版同时训练五项：

$$
\mathcal L_{SSL}
=
\lambda_\rho\mathcal L_{density}
+\lambda_\kappa\mathcal L_\kappa
+\lambda_g\mathcal L_g^{(\kappa)}
+\lambda_{Sob}\mathcal L_{Sob}
+\lambda_{chain}\mathcal L_{chain}.
$$

其中 $\hat g^{(\kappa)}$ 由 $\hat\rho$、$\hat\kappa$ 与链式法则产生；
$\hat g^{auto}=\partial\hat\rho/\partial q_i$ 来自同一个密度预测器的自动微分。
$\mathcal L_{Sob}$ 将 $\hat g^{auto}$ 对齐解析教师 $g$，$\mathcal L_{chain}$ 再把
$\hat g^{(\kappa)}$ 与 $\hat g^{auto}$ 对齐，形成：

$$
\hat g^{(\kappa)}
\approx
g
\approx
\hat g^{auto}.
$$

求构型导数时查询点必须固定于手部语义坐标系 ``{h}``；上游模型调用应对查询点坐标和采样路径
停止梯度。$q$ 是物理 rad 坐标，即使编码器内部读取 $q/\pi$，计算图也必须保留
$\partial/\partial q=(1/\pi)\partial/\partial(q/\pi)$。本模块及全部解码器在 SSL 完成后删除，
不得进入 PPO 前向。

监督张量轴：

```text
density target / prediction : [B, G, N_Q, L]
kappa target / prediction   : [B, E]
g target                    : [B, E, L]
valid query mask            : [B, G, N_Q]
valid edge mask             : [B, E]
```

这里 $E$ 只包含本次实际抽样的一阶边。有效边掩码同时承载最近点唯一性、精确表面退化与后端
能力边界；被降权或屏蔽的样本比例必须由诊断系统另行报告，不能因为不进入损失就从科研结果消失。

非祖先边的教师 $\kappa/g$ 必须精确为零。目标函数对祖先边和非祖先边使用同一形状与公式，
从而显式惩罚虚假的跨指灵敏度。若未来在解码器结构中强制非祖先输出为零，也仍应保留合同测试，
确认拓扑选择器没有错误地屏蔽真实祖先。

本模块使用均方误差作为第一版统一读数；距离、密度与灵敏度各自保留独立物理单位和损失项，
不在公共函数中做匿名标准化。不同量纲之间的相对权重由 generated 校准批次上的共享编码器
梯度范数决定，并写入完整解析配置。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ...representations.targets.field_samples import FieldTargetBatch, SensitivityTargetBatch


@dataclass(frozen=True)
class GeometrySSLWeights:
    r"""第一版联合物理目标的无量纲权重。

    五项在正式首版中均为正；默认 ``1.0`` 只提供可运行的解析/集成测试锚点，不表示不同单位与
    数值尺度的目标天然等权。正式训练应在固定 generated 校准批次上比较各项写入共享编码器的
    梯度范数，再把校准后的常数保存到完整解析的实验配置。官方 LEAP/Allegro
    留出数据不能参与权重校准。

    Attributes:
        density (float): $\lambda_\rho$，无量纲。
        kappa (float): $\lambda_\kappa$，无量纲。
        derived_field (float): $\lambda_g$，约束 $\hat g^{(\kappa)}\approx g$。
        sobolev (float): $\lambda_{Sob}$，约束同一密度函数的构型自导数。
        chain (float): $\lambda_{chain}$，约束两条预测斜率彼此一致。
    """

    density: float = 1.0
    kappa: float = 1.0
    derived_field: float = 1.0
    sobolev: float = 1.0
    chain: float = 1.0

    def __post_init__(self) -> None:
        r"""验证所有权重非负。

        权重为零是合法的受控消融，因此配置层不强制严格为正；“第一版强默认均开启”由完整
        实验配置与测试记录负责，而不是在通用目标函数类中阻止消融。

        Raises:
            ValueError: 任一权重小于零时抛出。
        """

        values = (self.density, self.kappa, self.derived_field, self.sobolev, self.chain)
        if any(value < 0.0 for value in values):
            raise ValueError("Geometry SSL loss weights must be non-negative")


@dataclass(frozen=True)
class GeometrySSLTerms:
    r"""联合损失及两条场灵敏度预测路径的审计输出。

    各标量项保持独立字段，训练记录器可以分别报告优化动态、梯度量级与关闭消融；不能只记录
    ``total`` 后把某一物理路径未被使用的问题隐藏在合并损失中。两个稠密抽样边张量用于
    按归属体、JOINT、带宽、距离壳层与最近点唯一性分层分析。

    Attributes:
        total (torch.Tensor): 加权总损失，标量。
        density (torch.Tensor): 多带宽零阶密度均方误差。
        kappa (torch.Tensor): 抽样边上的距离灵敏度均方误差。
        derived_field (torch.Tensor): $\hat g^{(\kappa)}$ 对解析教师 $g$ 的误差。
        sobolev (torch.Tensor): $\hat g^{auto}$ 对解析教师 $g$ 的误差。
        chain (torch.Tensor): $\hat g^{(\kappa)}$ 与 $\hat g^{auto}$ 的一致性误差。
        derived_field_sensitivity (torch.Tensor): ``[B,E,L]``，单位 $\mathrm{rad}^{-1}$。
        auto_field_sensitivity (torch.Tensor): ``[B,E,L]``，单位 $\mathrm{rad}^{-1}$。
    """

    total: torch.Tensor  # 加权总损失，标量
    density: torch.Tensor  # $\mathcal L_{density}$
    kappa: torch.Tensor  # $\mathcal L_\kappa$
    derived_field: torch.Tensor  # $\mathcal L_g^{(\kappa)}$
    sobolev: torch.Tensor  # $\mathcal L_{Sob}$
    chain: torch.Tensor  # $\mathcal L_{chain}$
    derived_field_sensitivity: torch.Tensor  # `[B,E,L]` 的 $\hat g^{(\kappa)}$
    auto_field_sensitivity: torch.Tensor  # `[B,E,L]` 的 $\hat g^{auto}$


def _masked_mean_square(error: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    r"""按有效物理标量数归一化平方误差。

    对误差张量 $e$ 与可广播布尔掩码 $m$：

    $$
    \operatorname{MSE}_{mask}(e,m)
    =
    \frac{\sum_j m_j e_j^2}{\sum_j m_j}.
    $$

    归一化分母是有效标量通道数，而不是归属体或查询点总数，避免不同无效比例改变梯度尺度。
    """

    while mask.ndim < error.ndim:
        mask = mask.unsqueeze(-1)  # 把 `[B,...]` 掩码广播到带宽等尾轴
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
        density_prediction (torch.Tensor): ``[B,G,N_Q,L]`` 的同一密度解码器输出。
        q (torch.Tensor): ``[B,N_J]`` 的物理关节角，单位 rad，必须 ``requires_grad=True``。
        owner_index (torch.Tensor): ``[E]`` 或跨结构 padding 后 ``[B,E]`` 抽样归属体索引。
        query_index (torch.Tensor): 与 owner selector 同形状的查询点索引。
        joint_index (torch.Tensor): 与 owner selector 同形状的 JOINT 索引。
        create_graph (bool): 是否保留导数图；训练 Sobolev loss 时必须为 ``True``。

    Returns:
        torch.Tensor: ``[B,E,L]`` 的 $\hat g^{auto}$，单位 $\mathrm{rad}^{-1}$。

    Raises:
        ValueError: 当物理 $q$ 未启用梯度时抛出。
    """

    if not q.requires_grad:
        raise ValueError("q.requires_grad must be True for Sobolev/JVP supervision")
    if owner_index.ndim not in {1, 2} or query_index.shape != owner_index.shape or joint_index.shape != owner_index.shape:
        raise ValueError("owner/query/joint selectors must share [E] or [B,E] shape")
    if owner_index.ndim == 2 and owner_index.shape[0] != density_prediction.shape[0]:
        raise ValueError("batched selectors must share B with density_prediction")
    edge_count = owner_index.shape[-1]  # 每个样本的抽样边存储预算 $E$
    bandwidth_count = density_prediction.shape[-1]  # Gaussian 通道数 $L$
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


def _select_owner_queries(
    values: torch.Tensor,
    owner_index: torch.Tensor,
    query_index: torch.Tensor,
) -> torch.Tensor:
    """从 `[B,G,N_Q,...]` 读取共享 `[E]` 或逐样本 `[B,E]` selectors。"""

    if owner_index.ndim == 1:
        return values[:, owner_index, query_index]
    batch_index = torch.arange(values.shape[0], device=values.device).unsqueeze(1)
    return values[batch_index, owner_index, query_index]


class GeometrySSLObjective(nn.Module):
    r"""协同计算密度、显式 κ、派生 g、Sobolev 与链式一致性。

    本类不持有带宽、查询点采样器或监督后端；带宽来自 ``FieldTargetBatch``，保证预测与
    真值使用同一米制测量尺度。它也不拥有优化器或自动调权逻辑，使各损失的物理公式可以在
    合同测试中独立验证。
    """

    def __init__(self, weights: GeometrySSLWeights) -> None:
        r"""保存已经解析并冻结的联合损失权重。

        Args:
            weights (GeometrySSLWeights): 本次实验的无量纲损失权重。
        """

        super().__init__()
        self.weights = weights  # 运行期间保持不变；动态校准在训练器启动前完成

    def forward(
        self,
        *,
        q: torch.Tensor,
        density_prediction: torch.Tensor,
        kappa_prediction: torch.Tensor,
        field_targets: FieldTargetBatch,
        sensitivity_targets: SensitivityTargetBatch,
    ) -> GeometrySSLTerms:
        r"""计算第一版完整联合目标，并返回可分别记录的各项。

        Args:
            q (torch.Tensor): ``[B,N_J]`` 物理关节构型，单位 rad，且保留编码器计算图。
            density_prediction (torch.Tensor): ``[B,G,N_Q,L]`` 的 $\hat\rho$。
            kappa_prediction (torch.Tensor): ``[B,E]`` 的 $\hat\kappa$，单位 m/rad。
            field_targets (FieldTargetBatch): 零阶距离/密度真值、带宽和查询点有效掩码。
            sensitivity_targets (SensitivityTargetBatch): 抽样边上的 $\kappa/g$ 与非光滑掩码。

        Returns:
            GeometrySSLTerms: 五个标量损失、总损失和两条 ``[B,E,L]`` 场灵敏度路径。

        Raises:
            ValueError: 预测/目标形状不一致，或 Sobolev 路径无法从物理 $q$ 求导时抛出。
        """

        if density_prediction.shape != field_targets.density.shape:
            raise ValueError("density prediction and target must share shape [B,G,N_Q,L]")
        if kappa_prediction.shape != sensitivity_targets.kappa.shape:
            raise ValueError("kappa prediction and target must share shape [B,E]")

        density_loss = _masked_mean_square(
            density_prediction - field_targets.density,
            field_targets.valid_mask,
        )  # $\mathcal L_{density}$，全部查询点来源与带宽都受监督
        kappa_loss = _masked_mean_square(
            kappa_prediction - sensitivity_targets.kappa,
            sensitivity_targets.valid_mask,
        )  # $\mathcal L_\kappa$，包含非祖先结构零

        owner_index = sensitivity_targets.owner_index  # `[E]` 抽样归属体索引
        query_index = sensitivity_targets.query_index  # `[E]` 抽样查询点索引
        selected_density = _select_owner_queries(density_prediction, owner_index, query_index)  # `[B,E,L]`
        selected_distance = _select_owner_queries(field_targets.distance, owner_index, query_index)  # `[B,E]`，m
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
        )  # `[B,E,L]`，同一密度预测器对固定 `{h}` 查询点的自导数
        sobolev_loss = _masked_mean_square(
            auto_field - sensitivity_targets.field_sensitivity,
            sensitivity_targets.valid_mask,
        )  # $\mathcal L_{Sob}$
        chain_loss = _masked_mean_square(
            derived_field - auto_field,
            sensitivity_targets.valid_mask,
        )  # $\mathcal L_{chain}$，连接显式一阶输出头与密度函数切向

        total = (
            self.weights.density * density_loss
            + self.weights.kappa * kappa_loss
            + self.weights.derived_field * derived_field_loss
            + self.weights.sobolev * sobolev_loss
            + self.weights.chain * chain_loss
        )  # 第一版联合标量目标
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
