r"""连续邻近场的物理定义与纯张量实现。

本模块保留“连续密度团”这一研究直觉，但不把所有平滑标量场混成同一语义。

1. 表面核密度估计：在面积加权表面样本 $y_m^h(q)$ 上放置核并求和。它表示表面测度的
   平滑分布，结果会受核带宽、表面积归一化与采样质量影响。
2. Gaussian distance shell：由 UDF 构造

   $$\large
   \rho_g^{shell}(x^h;q)
   =
   \exp\!\left(-\frac{u_g(x^h;q)^2}{2\sigma^2}\right)
   \in(0,1],
   $$

   其中 $\sigma>0$ 是米制表面带宽。它在表面上取 1，向内外两侧衰减，是表面壳层，不是填满实体内部的占据场。
3. Gaussian 平滑占据场：对实体体积指示函数做 Gaussian 卷积，保留内外侧与绝对体积语义。
   它需要可靠闭合体，不能由无符号点云标签直接冒充。

当前阶段固定第 2 类逐归属体 Gaussian 邻近场，并同时使用多个物理带宽：

$$\large
\left(
\rho_{\sigma_1,g}(x^h;q),\ldots,\rho_{\sigma_L,g}(x^h;q)
\right),
\qquad
0<\sigma_1<\cdots<\sigma_L.
$$

小 $\sigma$ 监督表面/接触边界，中等 $\sigma$ 表达几何包络，大 $\sigma$ 为较远工作空间查询点
保留连续信号。PALM/JOINT/TIP 归属体轴固定为 $G=N_E$，并与实体/解码器轴同索引；当前不训练
独立并集输出头。$L$、具体米制带宽、查询分层与损失权重由实验裁定。

三类场都可接固定 BPS 或随机查询点，但当前主线使用抽样式条件隐式查询。早期核密度/体积基线
是否归一化表面或体积测度仍待裁定；物理尺寸对接触物理有意义，因此不能无说明地把每个手型或
归属体归一化到积分为 1。
"""

from __future__ import annotations

import torch


def gaussian_density_from_distance(distance: torch.Tensor, bandwidths: torch.Tensor) -> torch.Tensor:
    r"""把逐归属体无符号距离映射为显式 sigma 条件的 Gaussian 邻近场。

    核心公式为：

    $$\large
    \rho_{\sigma_\ell,g}(x;q)
    =
    \exp\left(-\frac{d_g(x;q)^2}{2\sigma_\ell^2}\right).
    $$

    ``bandwidths`` 可以是跨 batch 共享的 ``[L]``，也可以是每个样本实际采样的 ``[B,L]``；后者
    沿 distance 的 batch 后续 owner/query 轴广播。这里不截断距离或带宽，因为截断会静默改变
    $d=0$ 和小带宽附近的解析导数。

    Args:
        distance (torch.Tensor): 非负无符号距离，任意形状，单位 m。
        bandwidths (torch.Tensor): 严格正带宽，形状 ``[L]`` 或 ``[B,L]``，单位 m。

    Returns:
        torch.Tensor: 多带宽密度，形状 `[*distance.shape, L]`，无量纲。

    Raises:
        ValueError: 当距离含负值、带宽轴不匹配 batch 或带宽不严格为正时抛出。
    """

    sigma = _broadcast_bandwidths(distance, bandwidths)  # `[*distance.shape,L]`，m
    if torch.any(bandwidths <= 0):
        raise ValueError("Gaussian bandwidths must be strictly positive")
    if torch.any(distance < 0):
        raise ValueError("unsigned distance cannot contain negative values")

    sigma_squared = sigma.square()  # $\sigma_{b,\ell}^2$，形状 `[*distance.shape,L]`，单位 $\mathrm{m}^2$
    squared_distance = distance.unsqueeze(-1).square()  # $d_g^2$，形状 `[...,L]` 广播前为 `[...,1]`
    return torch.exp(-squared_distance / (2.0 * sigma_squared))  # $\rho\in(0,1]$，无量纲


def field_sensitivity_from_distance(
    distance: torch.Tensor,
    density: torch.Tensor,
    bandwidths: torch.Tensor,
    kappa: torch.Tensor,
) -> torch.Tensor:
    r"""由距离灵敏度解析得到多带宽场灵敏度。

    对物理构型坐标 $q_i$：

    $$\large
    g_{\sigma_\ell,g,i}
    =
    -\frac{d_g}{\sigma_\ell^2}\rho_{\sigma_\ell,g}\kappa_{g,i}.
    $$

    Args:
        distance (torch.Tensor): 形状 `[..., N_Q]` 的无符号距离，单位 m。
        density (torch.Tensor): 形状 `[*distance.shape, L]` 的密度，无量纲。
        bandwidths (torch.Tensor): 形状 ``[L]`` 或 ``[B,L]`` 的米制带宽。
        kappa (torch.Tensor): 形状 `[*distance.shape, N_J]` 的距离灵敏度，单位 m/rad。

    Returns:
        torch.Tensor: 形状 `[*distance.shape, L, N_J]`，单位 `1/rad`。

    Raises:
        ValueError: 当张量轴与物理合同不一致时抛出。
    """

    sigma = _broadcast_bandwidths(distance, bandwidths)  # `[*distance.shape,L]`，m
    if density.shape != sigma.shape:
        raise ValueError(
            "density must have shape [*distance.shape, L], "
            f"got distance={tuple(distance.shape)}, density={tuple(density.shape)}, sigma={tuple(sigma.shape)}"
        )
    if kappa.shape[:-1] != distance.shape:
        raise ValueError(
            "kappa must have shape [*distance.shape, N_J], "
            f"got distance={tuple(distance.shape)}, kappa={tuple(kappa.shape)}"
        )
    if torch.any(bandwidths <= 0):
        raise ValueError("Gaussian bandwidths must be strictly positive")

    inverse_sigma_squared = sigma.square().reciprocal()  # $1/\sigma_{b,\ell}^2$，单位 $\mathrm{m}^{-2}$
    radial_factor = -distance.unsqueeze(-1) * inverse_sigma_squared  # $-d_g/\sigma_\ell^2$，形状 `[...,L]`
    return radial_factor.unsqueeze(-1) * density.unsqueeze(-1) * kappa.unsqueeze(-2)  # 链式结果 `[...,L,N_J]`


def _broadcast_bandwidths(distance: torch.Tensor, bandwidths: torch.Tensor) -> torch.Tensor:
    r"""把共享 ``[L]`` 或逐样本 ``[B,L]`` sigma 广播到 distance 的完整前导轴。

    sigma 只沿 batch 轴允许变化；owner/query/edge 都读取同一物理测量尺度 realization。该约束使同资产
    q 子批次可以共享 sigma，同时避免把 owner 或 query identity 偷渡进带宽采样。
    """

    if bandwidths.ndim == 1:  # 纯公式/固定 validation grid 跨全部样本共享 $\sigma_\ell$
        view_shape = (1,) * distance.ndim + (bandwidths.shape[0],)
        return bandwidths.reshape(view_shape).expand(*distance.shape, -1)
    if bandwidths.ndim == 2 and distance.ndim >= 1 and bandwidths.shape[0] == distance.shape[0]:
        view_shape = (bandwidths.shape[0],) + (1,) * (distance.ndim - 1) + (bandwidths.shape[1],)
        return bandwidths.reshape(view_shape).expand(*distance.shape, -1)
    raise ValueError(
        f"bandwidths must have shape [L] or [B,L] matching distance batch, got {tuple(bandwidths.shape)}"
    )


__all__ = ["field_sensitivity_from_distance", "gaussian_density_from_distance"]
