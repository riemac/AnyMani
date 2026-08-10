r"""Continuous density field 的物理定义与纯张量实现。

本模块保留“连续密度团”这一研究直觉，但不把所有平滑标量场混成同一语义。

1. Surface KDE：在面积加权 surface samples $y_m^h(q)$ 上放置 kernel 并求和。它表示
   surface mass 的平滑分布，结果会受 kernel bandwidth、surface area normalization 与
   sampling quality 影响。
2. Gaussian distance shell：由 UDF 构造

   $$
   \rho_g^{shell}(x^h;q)
   =
   \exp\!\left(-\frac{u_g(x^h;q)^2}{2\sigma^2}\right)
   \in(0,1],
   $$

   其中 $\sigma>0$ 是 m 制表面带宽。它在 surface 上取 1，向内外两侧衰减，是
   surface shell，不是填满实体内部的 occupancy。
3. Gaussian-smoothed occupancy：对实体 volume indicator 做 Gaussian convolution，
   保留 inside/outside 与绝对 volume 语义。它需要可靠闭合体，不能由无符号点云标签
   直接冒充。

当前阶段固定第 2 类逐归属体 Gaussian 邻近场，并同时使用多个物理带宽：

$$
\left(
\rho_{\sigma_1,g}(x^h;q),\ldots,\rho_{\sigma_L,g}(x^h;q)
\right),
\qquad
0<\sigma_1<\cdots<\sigma_L.
$$

小 $\sigma$ 监督 surface/contact boundary，中等 $\sigma$ 表达 geometry envelope，大
$\sigma$ 为较远 workspace query 保留连续信号。PALM/JOINT/TIP owner 轴已经固定为
$G=N_E$ 且与 entity/decoder axis 同索引；当前不训练独立 union head。$L$、带宽米制数值、
query 分层与 loss weighting 尚未裁定。

三类 field 都可接 fixed BPS 或 sampled queries，但当前主线使用 sampled conditional-implicit
queries。是否归一化早期 KDE/volume baseline 的 surface/volume mass 尚未裁定；physical size 对 contact physics 有意义，
因此默认不能无说明地把每个 embodiment/group 归一化到积分为 1。
"""

from __future__ import annotations

import torch


def gaussian_density_from_distance(distance: torch.Tensor, bandwidths: torch.Tensor) -> torch.Tensor:
    r"""把逐归属体无符号距离映射为多带宽 Gaussian 邻近场。

    核心公式为：

    $$
    \rho_{\sigma_\ell,g}(x;q)
    =
    \exp\left(-\frac{d_g(x;q)^2}{2\sigma_\ell^2}\right).
    $$

    `distance` 的最后若干轴可自由表示 batch、owner 与 query；本函数只在末尾追加
    bandwidth 轴。这里不 clamp 距离或带宽，因为 clamp 会静默改变 $d=0$ 和小带宽
    附近的解析导数。

    Args:
        distance (torch.Tensor): 非负无符号距离，任意形状，单位 m。
        bandwidths (torch.Tensor): 一维严格正带宽，形状 `[L]`，单位 m。

    Returns:
        torch.Tensor: 多带宽密度，形状 `[*distance.shape, L]`，无量纲。

    Raises:
        ValueError: 当距离含负值、带宽不是一维或带宽不严格为正时抛出。
    """

    if bandwidths.ndim != 1:
        raise ValueError(f"bandwidths must have shape [L], got {tuple(bandwidths.shape)}")
    if torch.any(bandwidths <= 0):
        raise ValueError("Gaussian bandwidths must be strictly positive")
    if torch.any(distance < 0):
        raise ValueError("unsigned distance cannot contain negative values")

    sigma_squared = bandwidths.square()  # $\sigma_\ell^2$，形状 `[L]`，单位 $\mathrm{m}^2$
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

    $$
    g_{\sigma_\ell,g,i}
    =
    -\frac{d_g}{\sigma_\ell^2}\rho_{\sigma_\ell,g}\kappa_{g,i}.
    $$

    Args:
        distance (torch.Tensor): 形状 `[..., N_Q]` 的无符号距离，单位 m。
        density (torch.Tensor): 形状 `[*distance.shape, L]` 的密度，无量纲。
        bandwidths (torch.Tensor): 形状 `[L]` 的米制带宽。
        kappa (torch.Tensor): 形状 `[*distance.shape, N_J]` 的距离灵敏度，单位 m/rad。

    Returns:
        torch.Tensor: 形状 `[*distance.shape, L, N_J]`，单位 `1/rad`。

    Raises:
        ValueError: 当张量轴与物理合同不一致时抛出。
    """

    if density.shape != (*distance.shape, bandwidths.numel()):
        raise ValueError(
            "density must have shape [*distance.shape, L], "
            f"got distance={tuple(distance.shape)}, density={tuple(density.shape)}, L={bandwidths.numel()}"
        )
    if kappa.shape[:-1] != distance.shape:
        raise ValueError(
            "kappa must have shape [*distance.shape, N_J], "
            f"got distance={tuple(distance.shape)}, kappa={tuple(kappa.shape)}"
        )
    if torch.any(bandwidths <= 0):
        raise ValueError("Gaussian bandwidths must be strictly positive")

    inverse_sigma_squared = bandwidths.square().reciprocal()  # $1/\sigma_\ell^2$，单位 $\mathrm{m}^{-2}$
    radial_factor = -distance.unsqueeze(-1) * inverse_sigma_squared  # $-d_g/\sigma_\ell^2$，形状 `[...,L]`
    return radial_factor.unsqueeze(-1) * density.unsqueeze(-1) * kappa.unsqueeze(-2)  # 链式结果 `[...,L,N_J]`


__all__ = ["field_sensitivity_from_distance", "gaussian_density_from_distance"]
