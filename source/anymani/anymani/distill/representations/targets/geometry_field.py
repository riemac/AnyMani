r"""Warp 最近面、解析点 Jacobian 与 Gaussian 公式组成的在线 SSL 教师。

本模块闭合一份同构结构微批次的监督路径：

```text
q + typed robot spec
  -> current owner transforms
  -> fixed/workspace + current shell/adjacent queries
  -> Warp owner-local nearest face
  -> d and multi-band rho
  -> sampled owner-query-JOINT point Jacobian
  -> kappa and chain-derived g
```

对固定 `{h}` query $x$、唯一最近物质点 $y_g^*(q)$ 与单位方向
$n=(x-y^*)/d$：

$$
\kappa_{g,i}
=
\frac{\partial d_g}{\partial q_i}
=
-n^\top J_{g,i}^h(y_g^*),
\qquad
g_{\sigma,g,i}
=
-\frac{d_g}{\sigma^2}\rho_{\sigma,g}\kappa_{g,i}.
$$

一阶监督只在 owner-shell query 上抽样少量边；非祖先边由 source 拓扑掩码产生精确零。当前 Warp
后端提供最近 face、barycentric 和三角面内物理 margin。该 margin 是局部光滑保守证据，不冒充
全局第二近点间隔；provenance 明确记录这一能力边界，后续 CPU/Kaolin oracle 可独立加强。

全部 teacher 输出停止梯度。训练只对模型参数建立普通一阶梯度图，绝不穿过 query 采样、Warp
最近面或解析教师。

零阶和一阶张量轴刻意分离：

```text
distance / density  [B, G, N_Q] / [B, G, N_Q, N_sigma]
sampled selectors   owner/query/joint: [E]
closest point       [B, E, 3]
kappa               [B, E]
g                    [B, E, N_sigma]
```

完整 $[B,G,N_Q,N_J]$ Jacobian 会把大量非祖先结构零和未使用 query 物化到显存，因此默认不构造。
edge selector 在每个 owner 的 shell query 中交替选择祖先与非祖先 JOINT：祖先提供真实局部运动，
非祖先显式监督跨指/掌部结构零。PALM 没有祖先，只选择非祖先；TIP 与 distal JOINT 可选择整条
finger chain 的祖先。

距离导数的符号来自固定 query。令 surface point 速度为 $J_i$，则 query-to-surface 向量为
$r=x-y^*$，$n=r/\|r\|$。由于 query 对 q 停止梯度，$\partial r/\partial q_i=-J_i$，故
$\kappa_i=-n^T J_i$。若 query 跟着 owner 共动，这个公式会多出 query velocity 并退化；因此 workspace
realization 固定于同资产 q 子批次的 `{h}`，target 计算显式使用 detached query coordinates。

最近点先由 Warp 以 owner-local 坐标返回，再变回 `{h}`。为了调用 source 的解析 point Jacobian，
本模块把 selected closest point 反变换回对应 owner reference link。这里使用当前 owner transform，
而不是 home transform；否则 q 不为 home 时物质点会映射到错误局部位置。

$d\approx0$ 时 unsigned distance 的径向单位向量未定义，即使 Gaussian density 本身在表面取 1，
也不能给 $\kappa$ 伪造方向。``distance_epsilon_m`` 只屏蔽一阶样本，不改变零阶 $\rho(d=0)=1$。
同理，triangle feature margin 只控制一阶 valid mask，不能删除零阶 surface query。

closest source 把 owner index 放在高 32 位、union face 放在低 32 位。这个整数只有和
asset content hash、Manifold backend 与 owner axis 一起才构成稳定 provenance；它不是跨资产 face ID。
后续若加入 component-level remap，应扩展 provenance schema，不得重解释已保存整数。

当前 smoothness mask 是：有效 Warp face、owner-shell stratum、$d$ 大于 epsilon、triangle feature
margin 大于阈值。provenance 明确写 ``global_second_nearest_margin=not_materialized``。因此正式诊断要
分别报告 mask 覆盖率、按 owner/距离壳层的有效比例，以及 CPU/Kaolin reference 上的最近源切换；
不能只报告被 mask 后的低误差。

sigma 使用统一 SI 米制。训练中心为 4/16/64 mm 并施加 log-space 有界 jitter；validation 关闭 jitter，固定使用同一组 4/16/64 mm 中心。stratum 不决定 sigma，owner 大小也不做独立归一化。共同尺度实验必须同时缩放 geometry、query、anchor 与
bandwidth；只缩放 geometry 而保持米制 bandwidth 是有意改变物理尺度，不是 invariance test。

生成的 ``FieldTargetBatch`` 与 ``SensitivityTargetBatch`` 是 privileged data package。模型 decoder
只读取 query feature 和 latent，不读取 target distance、closest source、mask provenance 或 stratum。
这些 teacher package 不保存到 retained-only checkpoint，PPO 迁移时整个生成器和 Warp cache 删除。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from time import perf_counter
from typing import TypedDict

import torch

from anymani.distill.representations.sources.collision_geometry import OwnerGeometryCache, WarpOwnerGeometryCache
from anymani.distill.representations.sources.kinematics import (
    EmbodimentGeometrySpec,
    forward_owner_transforms,
    selected_point_jacobian,
)

from ..fields.density import field_sensitivity_from_distance, gaussian_density_from_distance
from ..queries.spatial_sampling import SpatialQueryBatch
from .field_samples import (
    FieldTargetBatch,
    QueryStratum,
    SensitivityOwnerCategory,
    SensitivitySamplingRole,
    SensitivityTargetBatch,
)
from .warp_surface import query_owner_surfaces_warp


class _CentralDifferenceAudit(TypedDict):
    """中心差分审计字段的精确 kwargs 类型，避免 Tensor/float union 污染 target dataclass。"""

    central_difference: torch.Tensor
    central_difference_valid_mask: torch.Tensor
    central_difference_plus_face: torch.Tensor
    central_difference_minus_face: torch.Tensor
    central_difference_elapsed_seconds: float


@dataclass(frozen=True)
class GaussianProximityFieldCfg:
    r"""多尺度 unsigned Gaussian proximity field 的显式 sigma 测量配置。

    三个中心带宽 $\bar\sigma=(4,16,64)$ mm 定义近/中/远测量尺度。训练时在 log-space 有界均匀
    采样，默认相对范围为 $[0.9\bar\sigma,1.1\bar\sigma]$；同一资产 q 子批次共享一次 realization。
    decoder 读取每个实际 sigma，而不是把中心编号当作固定输出通道身份。
    """

    bandwidth_centers_m: tuple[float, ...] = (0.004, 0.016, 0.064)  # 4/16/64 mm 训练中心
    bandwidth_jitter_relative: float = 0.10  # log-space 有界采样的相对半宽，默认 ±10%
    validation_bandwidths_m: tuple[float, ...] = (0.004, 0.016, 0.064)  # 固定 4/16/64 mm，与训练中心一致
    def __post_init__(self) -> None:
        """拒绝无带宽、非递增带宽与越界的 log-space jitter。"""

        if not self.bandwidth_centers_m or any(value <= 0.0 for value in self.bandwidth_centers_m):
            raise ValueError("bandwidth_centers_m must contain strictly positive values")
        if any(left >= right for left, right in zip(self.bandwidth_centers_m[:-1], self.bandwidth_centers_m[1:])):
            raise ValueError("bandwidth_centers_m must be strictly increasing")
        if not self.validation_bandwidths_m or any(value <= 0.0 for value in self.validation_bandwidths_m):
            raise ValueError("validation_bandwidths_m must contain strictly positive values")
        if any(
            left >= right for left, right in zip(self.validation_bandwidths_m[:-1], self.validation_bandwidths_m[1:])
        ):
            raise ValueError("validation_bandwidths_m must be strictly increasing")
        if not 0.0 <= self.bandwidth_jitter_relative < 1.0:
            raise ValueError("bandwidth_jitter_relative must lie in [0,1)")


@dataclass(frozen=True)
class GeometryFieldTargetCfg:
    r"""sampled $\kappa$ edge 数与 UDF 非光滑区域有效性阈值。"""

    train_active_per_joint: int = 1  # 每个有效 JOINT、每个 q 的 descendant/active shell 边数
    train_zero_per_joint: int = 1  # 每个有效 JOINT、每个 q 的 structure-zero shell 边数
    validation_active_per_joint: int = 4  # 固定 validation bank 的 descendant 边数
    validation_zero_per_joint: int = 4  # 固定 validation bank 的 structure-zero 边数
    distance_epsilon_m: float = 1.0e-6  # $d\approx0$ 时 UDF 方向未定义
    feature_margin_min_m: float = 1.0e-5  # 最近投影点远离当前三角面边界的阈值

    def __post_init__(self) -> None:
        r"""拒绝失去 sampled-edge 监督或不合法的米制 mask 阈值。"""

        counts = (
            self.train_active_per_joint,
            self.train_zero_per_joint,
            self.validation_active_per_joint,
            self.validation_zero_per_joint,
        )
        if min(counts) < 1:
            raise ValueError("joint-first active/zero edge budgets must be positive")
        if self.distance_epsilon_m <= 0.0 or self.feature_margin_min_m < 0.0:
            raise ValueError("distance epsilon must be positive and feature margin non-negative")


def sample_geometry_bandwidths(
    config: GaussianProximityFieldCfg,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    sampling_seed: int,
) -> torch.Tensor:
    r"""为一个同资产 q 子批次采一次显式 sigma realization。

    设中心为 $\bar\sigma_\ell$、相对半宽为 $\delta_\sigma$，则：

    $$
    \epsilon_\ell\sim\mathcal U(\log(1-\delta_\sigma),\log(1+\delta_\sigma)),
    \qquad
    \sigma_\ell=\bar\sigma_\ell\exp(\epsilon_\ell).
    $$

    同一 realization 沿 q 子批次 batch 轴广播，使不同 q 在相同测量尺度下可比较；下一 q 子批次由
    新 seed 重采。返回 tensor 不启用梯度，sigma 只作为 decoder/teacher 的固定条件。
    """

    centers = torch.tensor(config.bandwidth_centers_m, device=device, dtype=dtype)  # `[N_σ]`，m
    generator = torch.Generator(device=device)  # 与 query/edge seed 分离但由同一在线 step 确定
    generator.manual_seed(int(sampling_seed))
    relative = config.bandwidth_jitter_relative  # $\delta_\sigma$，无量纲
    lower = math.log1p(-relative)  # 精确对应 $(1-\delta_\sigma)\bar\sigma$
    upper = math.log1p(relative)  # 精确对应 $(1+\delta_\sigma)\bar\sigma$
    epsilon = lower + (upper - lower) * torch.rand(centers.shape, generator=generator, device=device, dtype=dtype)
    realization = centers * torch.exp(epsilon)  # `[N_σ]`，m，严格正
    return realization.unsqueeze(0).expand(batch_size, -1)  # `[B,N_σ]` shared q-subbatch view


def fixed_validation_gaussian_field_config(config: GaussianProximityFieldCfg) -> GaussianProximityFieldCfg:
    r"""返回使用固定 sigma 网格、完全关闭 jitter 的 validation teacher 配置。"""

    return replace(
        config,
        bandwidth_centers_m=config.validation_bandwidths_m,
        bandwidth_jitter_relative=0.0,
    )


def generate_geometry_field_targets(
    q: torch.Tensor,
    spec: EmbodimentGeometrySpec,
    geometry_cache: OwnerGeometryCache,
    warp_cache: WarpOwnerGeometryCache,
    queries: SpatialQueryBatch,
    *,
    field_config: GaussianProximityFieldCfg = GaussianProximityFieldCfg(),
    target_config: GeometryFieldTargetCfg = GeometryFieldTargetCfg(),
    edge_sampling_seed: int = 0,
    supervision_split: str = "train",
    owner_transforms: torch.Tensor | None = None,
    current_spatial_screws: torch.Tensor | None = None,
    q_index: torch.Tensor | None = None,
) -> tuple[FieldTargetBatch, SensitivityTargetBatch]:
    r"""生成多带宽零阶目标与 sampled-edge 一阶目标。

    Args:
        q (torch.Tensor): ``[B,N_J]`` 当前物理关节角，rad，CUDA float32。
        spec (EmbodimentGeometrySpec): 同结构模式的 robots 动态规格。
        geometry_cache (OwnerGeometryCache): owner role 与静态 cache provenance。
        warp_cache (WarpOwnerGeometryCache): 同资产 GPU BVH。
        queries (SpatialQueryBatch): ``[B,G,N_Q,3]`` 当前 `{h}` queries。
        field_config (GaussianProximityFieldCfg): train sigma realization 的测量尺度。
        target_config (GeometryFieldTargetCfg): edge 数与局部光滑 mask 阈值。
        edge_sampling_seed (int): 可复现 edge/query/JOINT 选择种子。
        supervision_split (str): ``train`` 使用每 JOINT 1+1 边；``eval`` 使用 4+4。

    Returns:
        tuple[FieldTargetBatch, SensitivityTargetBatch]: 零阶完整 query 轴与一阶抽样边轴。
    """

    if queries.query_points_h.device != q.device or queries.query_points_h.dtype != q.dtype:
        raise ValueError("q and query points must share CUDA device and float dtype")
    if owner_transforms is None:
        owner_transforms = forward_owner_transforms(spec, q.detach())  # 独立 target 调用保留旧行为
    expected_transform_shape = (q.shape[0], spec.owner_home_transforms.shape[0], 4, 4)
    if (
        owner_transforms.shape != expected_transform_shape
        or owner_transforms.device != q.device
        or owner_transforms.dtype != q.dtype
        or owner_transforms.requires_grad
    ):
        raise ValueError("owner_transforms must be detached [B,G,4,4] matching q/spec")
    if current_spatial_screws is not None:
        expected_screw_shape = (q.shape[0], spec.space_screws.shape[0], 6)
        if (
            current_spatial_screws.shape != expected_screw_shape
            or current_spatial_screws.device != q.device
            or current_spatial_screws.dtype != q.dtype
            or current_spatial_screws.requires_grad
        ):
            raise ValueError("current_spatial_screws must be detached [B,N_J,6] matching q/spec")
    surface = query_owner_surfaces_warp(queries.query_points_h, owner_transforms, warp_cache)
    bandwidths = sample_geometry_bandwidths(  # `[B,N_σ]` 实际 sigma，不是固定输出 channel identity
        field_config,
        batch_size=q.shape[0],
        device=q.device,
        dtype=q.dtype,
        sampling_seed=edge_sampling_seed + 104_729,
    )
    density = gaussian_density_from_distance(surface.distance_m, bandwidths)
    field_valid = torch.isfinite(surface.distance_m) & (surface.face_index >= 0)
    role_index = {"palm": 0, "joint": 1, "tip": 2}
    owner_role = torch.tensor(
        [role_index[record.role] for record in geometry_cache.records],
        device=q.device,
        dtype=torch.long,
    )
    field_targets = FieldTargetBatch(
        query_points=queries.query_points_h.detach(),
        query_stratum=queries.query_stratum,
        distance=surface.distance_m.detach(),
        density=density.detach(),
        valid_mask=field_valid.detach(),
        owner_role=owner_role,
        bandwidths=bandwidths,
        provenance={
            "frame": "h",
            "length_unit": "m",
            "backend": "warp_mesh_query_point",
            "asset_content_hash": geometry_cache.asset_content_hash,
            "query_mixture": "workspace=0.50,owner_shell=0.25,adjacent=0.25",
        },
    )

    if supervision_split == "eval":
        active_per_joint = target_config.validation_active_per_joint
        zero_per_joint = target_config.validation_zero_per_joint
    elif supervision_split == "train":
        active_per_joint = target_config.train_active_per_joint
        zero_per_joint = target_config.train_zero_per_joint
    else:
        raise ValueError(f"unknown supervision_split={supervision_split!r}")
    (
        owner_index,
        query_index,
        joint_index,
        active_mask,
        owner_category,
        selected_query_stratum,
        fallback_category,
        sampling_role,
    ) = _sample_sensitivity_edges(
        spec,
        queries,
        active_per_joint=active_per_joint,
        zero_per_joint=zero_per_joint,
        sampling_seed=edge_sampling_seed,
        q_index=q_index,
    )
    if owner_index.ndim == 1:
        closest_h = surface.closest_point_h_m[:, owner_index, query_index]
        selected_query_h = queries.query_points_h[:, owner_index, query_index]
        selected_distance = surface.distance_m[:, owner_index, query_index]
        selected_feature_margin = surface.feature_margin_m[:, owner_index, query_index]
        selected_face = surface.face_index[:, owner_index, query_index]
        selected_transform = owner_transforms.index_select(1, owner_index)
    else:
        batch_index = torch.arange(q.shape[0], device=q.device).unsqueeze(1)
        closest_h = surface.closest_point_h_m[batch_index, owner_index, query_index]
        selected_query_h = queries.query_points_h[batch_index, owner_index, query_index]
        selected_distance = surface.distance_m[batch_index, owner_index, query_index]
        selected_feature_margin = surface.feature_margin_m[batch_index, owner_index, query_index]
        selected_face = surface.face_index[batch_index, owner_index, query_index]
        selected_transform = owner_transforms[batch_index, owner_index]
    closest_local = torch.matmul(
        selected_transform[..., :3, :3].transpose(-1, -2),
        (closest_h - selected_transform[..., :3, 3]).unsqueeze(-1),
    ).squeeze(-1)  # `[B,E,3]`，最近物质点在 owner reference link 中的局部坐标
    point_jacobian = selected_point_jacobian(
        spec,
        q.detach(),
        owner_index,
        joint_index,
        closest_local,
        owner_transforms=owner_transforms,
        current_spatial_screws=current_spatial_screws,
    )  # `[B,E,3]`，m/rad；非祖先严格为零
    radial_direction = (selected_query_h - closest_h) / selected_distance.clamp_min(
        target_config.distance_epsilon_m
    ).unsqueeze(-1)
    kappa = -(radial_direction * point_jacobian).sum(dim=-1)  # $-n^TJ$，m/rad
    ancestor_mask = spec.owner_ancestor_mask[owner_index, joint_index]
    ancestor_for_batch = ancestor_mask if ancestor_mask.ndim == 2 else ancestor_mask.unsqueeze(0)
    kappa = torch.where(ancestor_for_batch, kappa, torch.zeros_like(kappa))
    selected_density = (
        density[:, owner_index, query_index]
        if owner_index.ndim == 1
        else density[torch.arange(q.shape[0], device=q.device).unsqueeze(1), owner_index, query_index]
    )  # 从完整 $\rho[B,G,N_Q,L]$ gather，避免重复同一 Gaussian 公式
    field_sensitivity = field_sensitivity_from_distance(
        selected_distance,
        selected_density,
        bandwidths,
        kappa.unsqueeze(-1),
    ).squeeze(-1)  # `[B,E,L]`，1/rad
    if owner_index.ndim == 1:
        selected_stratum = queries.query_stratum[:, owner_index, query_index]
        selected_face_valid = field_valid[:, owner_index, query_index]
        active_for_batch = active_mask.unsqueeze(0)
        closest_owner = owner_index.to(torch.int64).view(1, -1)
    else:
        batch_index = torch.arange(q.shape[0], device=q.device).unsqueeze(1)
        selected_stratum = queries.query_stratum[batch_index, owner_index, query_index]
        selected_face_valid = field_valid[batch_index, owner_index, query_index]
        active_for_batch = active_mask
        closest_owner = owner_index.to(torch.int64)
    if not torch.equal(selected_stratum, selected_query_stratum):
        raise RuntimeError("sensitivity selector query stratum disagrees with sampled query provenance")
    # Active-context 的 adjacent/workspace query 仍要求有效最近面；局部 feature margin 继续控制 UDF 可微性。
    selected_face_valid = selected_face_valid & torch.isfinite(selected_distance)
    active_smooth = (
        selected_face_valid
        & (selected_distance > target_config.distance_epsilon_m)
        & (selected_feature_margin >= target_config.feature_margin_min_m)
    )
    selected_valid = torch.where(active_for_batch, active_smooth, selected_face_valid)
    closest_source = (
        closest_owner.bitwise_left_shift(32)
        | selected_face.to(torch.int64)
    )  # 高 32 位 owner、低 32 位 union face，随 asset hash 一起解释
    sensitivity_targets = SensitivityTargetBatch(
        owner_index=owner_index,
        query_index=query_index,
        joint_index=joint_index,
        ancestor_mask=ancestor_mask,
        active_mask=active_mask,
        closest_point=closest_h.detach(),
        closest_source=closest_source.detach(),
        uniqueness_margin=selected_feature_margin.detach(),
        kappa=kappa.detach(),
        field_sensitivity=field_sensitivity.detach(),
        valid_mask=selected_valid.detach(),
        owner_category=owner_category,
        query_stratum=selected_query_stratum,
        fallback_category=fallback_category,
        sampling_role=sampling_role,
        **_central_difference_source_audit(
            asset_id=geometry_cache.asset_id,
            q=q,
            q_index=q_index,
            spec=spec,
            warp_cache=warp_cache,
            queries=queries,
            owner_index=owner_index,
            query_index=query_index,
            joint_index=joint_index,
            active_mask=active_mask,
        ),
        provenance={
            "frame": "h",
            "distance_unit": "m",
            "joint_unit": "rad",
            "backend": "warp_mesh_query_point",
            "smoothness_mask": "owner_shell_and_triangle_feature_margin",
            "global_second_nearest_margin": "not_materialized",
        },
    )
    return field_targets, sensitivity_targets


def _central_difference_source_audit(
    *,
    asset_id: str,
    q: torch.Tensor,
    q_index: torch.Tensor | None,
    spec: EmbodimentGeometrySpec,
    warp_cache: WarpOwnerGeometryCache,
    queries: SpatialQueryBatch,
    owner_index: torch.Tensor,
    query_index: torch.Tensor,
    joint_index: torch.Tensor,
    active_mask: torch.Tensor,
    delta_rad: float = 1.0e-3,
) -> _CentralDifferenceAudit:
    r"""对稳定选中的约 1% q 行审计全部合法 active edges，不改变正式 teacher。"""

    batch_size, edge_count = owner_index.shape if owner_index.ndim == 2 else (q.shape[0], owner_index.shape[0])
    difference = torch.zeros(batch_size, edge_count, device=q.device, dtype=q.dtype)
    valid = torch.zeros(batch_size, edge_count, device=q.device, dtype=torch.bool)
    plus_face = torch.full((batch_size, edge_count), -1, device=q.device, dtype=torch.long)
    minus_face = torch.full_like(plus_face, -1)
    elapsed_seconds = 0.0  # 未命中稳定 1% q-row 时保持严格零开销证据
    if q_index is None or spec.joint_limits is None:
        return {
            "central_difference": difference,
            "central_difference_valid_mask": valid,
            "central_difference_plus_face": plus_face,
            "central_difference_minus_face": minus_face,
            "central_difference_elapsed_seconds": elapsed_seconds,
        }
    for batch_index, absolute_q_index in enumerate(q_index.detach().cpu().tolist()):
        digest = hashlib.sha256(f"source-audit-v1\0{asset_id}\0{int(absolute_q_index)}".encode()).digest()
        if int.from_bytes(digest[:8], "little") % 100 != 0:
            continue
        row_owner = owner_index[batch_index] if owner_index.ndim == 2 else owner_index
        row_query = query_index[batch_index] if query_index.ndim == 2 else query_index
        row_joint = joint_index[batch_index] if joint_index.ndim == 2 else joint_index
        row_active = active_mask[batch_index] if active_mask.ndim == 2 else active_mask
        limits = spec.joint_limits.index_select(0, row_joint)
        current = q[batch_index].index_select(0, row_joint)
        legal = row_active & (current - delta_rad >= limits[:, 0]) & (current + delta_rad <= limits[:, 1])
        edge_slots = torch.where(legal)[0]
        if edge_slots.numel() == 0:
            continue
        selected_joint = row_joint.index_select(0, edge_slots)
        q_plus = q[batch_index].unsqueeze(0).expand(edge_slots.numel(), -1).clone()
        q_minus = q_plus.clone()
        row_axis = torch.arange(edge_slots.numel(), device=q.device)
        q_plus[row_axis, selected_joint] += delta_rad
        q_minus[row_axis, selected_joint] -= delta_rad
        fixed_queries = queries.query_points_h[batch_index].unsqueeze(0).expand(edge_slots.numel(), -1, -1, -1)
        audit_started = perf_counter()  # 只覆盖该 q 行的两次 perturbed FK 与 Warp surface query
        plus = query_owner_surfaces_warp(
            fixed_queries,
            forward_owner_transforms(spec, q_plus),
            warp_cache,
        )
        minus = query_owner_surfaces_warp(
            fixed_queries,
            forward_owner_transforms(spec, q_minus),
            warp_cache,
        )
        elapsed_seconds += perf_counter() - audit_started
        selected_owner = row_owner.index_select(0, edge_slots)
        selected_query = row_query.index_select(0, edge_slots)
        plus_distance = plus.distance_m[row_axis, selected_owner, selected_query]
        minus_distance = minus.distance_m[row_axis, selected_owner, selected_query]
        difference[batch_index, edge_slots] = (plus_distance - minus_distance) / (2.0 * delta_rad)
        plus_face[batch_index, edge_slots] = plus.face_index[row_axis, selected_owner, selected_query].to(torch.long)
        minus_face[batch_index, edge_slots] = minus.face_index[row_axis, selected_owner, selected_query].to(torch.long)
        valid[batch_index, edge_slots] = torch.isfinite(plus_distance) & torch.isfinite(minus_distance)
    return {
        "central_difference": difference.detach(),
        "central_difference_valid_mask": valid.detach(),
        "central_difference_plus_face": plus_face.detach(),
        "central_difference_minus_face": minus_face.detach(),
        "central_difference_elapsed_seconds": elapsed_seconds,
    }


def _sample_sensitivity_edges(
    spec: EmbodimentGeometrySpec,
    queries: SpatialQueryBatch,
    *,
    active_per_joint: int,
    zero_per_joint: int,
    sampling_seed: int,
    q_index: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    r"""逐 q、逐 JOINT 采 2+1 edges，并显式记录 owner/query/fallback 类别。

    绝对 ``q_index`` 决定类别轮换；随机 generator 只在同一类别候选与同一 stratum query 槽内部抽样。
    因而改变 microbatch 切法不会改变 selector，resume 后同一 `(asset,q)` 也可逐元素复现。
    """

    device = queries.query_stratum.device
    batch_size = queries.query_stratum.shape[0]
    if q_index is None:
        q_identities = tuple(range(batch_size))
    else:
        if q_index.shape != (batch_size,):
            raise ValueError("q_index must have shape [B] for q-specific sensitivity sampling")
        q_identities = tuple(int(value) for value in q_index.detach().cpu().tolist())
    rows: list[tuple[list[int], list[int], list[int], list[bool], list[int], list[int], list[int], list[int]]] = []
    owner_count = spec.owner_ancestor_mask.shape[0]
    joint_count = spec.owner_ancestor_mask.shape[1]
    roles = spec.owner_roles or tuple("joint" for _ in range(owner_count))
    fingers = spec.owner_finger_names or tuple(None for _ in range(owner_count))
    owner_joint_indices = spec.owner_joint_indices or tuple(-1 for _ in range(owner_count))
    active_cycle = (
        SensitivityOwnerCategory.SELF,
        SensitivityOwnerCategory.SAME_FINGER_TIP,
        SensitivityOwnerCategory.OTHER_DESCENDANT,
        SensitivityOwnerCategory.OTHER_DESCENDANT,
    )
    zero_cycle = (
        SensitivityOwnerCategory.PALM,
        SensitivityOwnerCategory.SAME_FINGER_UPSTREAM,
        SensitivityOwnerCategory.OTHER_FINGER_JOINT,
        SensitivityOwnerCategory.OTHER_FINGER_TIP,
    )
    zero_strata = (QueryStratum.OWNER_SHELL, QueryStratum.ADJACENT, QueryStratum.WORKSPACE)
    for batch_index, q_identity in enumerate(q_identities):
        generator = torch.Generator(device=device)
        generator.manual_seed((int(sampling_seed) + q_identity * 1_000_003) % (2**63 - 1))
        owner_axis: list[int] = []
        query_axis: list[int] = []
        joint_axis: list[int] = []
        active_axis: list[bool] = []
        category_axis: list[int] = []
        stratum_axis: list[int] = []
        fallback_axis: list[int] = []
        role_axis: list[int] = []
        for joint_index in range(joint_count):
            descendant_owners = [int(value) for value in torch.where(spec.owner_ancestor_mask[:, joint_index])[0].tolist()]
            zero_owners = [int(value) for value in torch.where(~spec.owner_ancestor_mask[:, joint_index])[0].tolist()]
            self_owners = [
                owner for owner, mapped_joint in enumerate(owner_joint_indices) if mapped_joint == joint_index
            ]
            self_finger = fingers[self_owners[0]] if self_owners else None
            tip_owners = [
                owner
                for owner in descendant_owners
                if roles[owner] == "tip" and fingers[owner] == self_finger
            ]
            other_descendants = [
                owner for owner in descendant_owners if owner not in self_owners and owner not in tip_owners
            ]
            active_candidates = {
                SensitivityOwnerCategory.SELF: self_owners,
                SensitivityOwnerCategory.SAME_FINGER_TIP: tip_owners,
                SensitivityOwnerCategory.OTHER_DESCENDANT: other_descendants,
            }
            zero_candidates = {
                SensitivityOwnerCategory.PALM: [owner for owner in zero_owners if roles[owner] == "palm"],
                SensitivityOwnerCategory.SAME_FINGER_UPSTREAM: [
                    owner for owner in zero_owners if fingers[owner] == self_finger and roles[owner] != "palm"
                ],
                SensitivityOwnerCategory.OTHER_FINGER_JOINT: [
                    owner for owner in zero_owners if roles[owner] == "joint" and fingers[owner] != self_finger
                ],
                SensitivityOwnerCategory.OTHER_FINGER_TIP: [
                    owner for owner in zero_owners if roles[owner] == "tip" and fingers[owner] != self_finger
                ],
            }
            for edge_offset in range(active_per_joint):
                requested = active_cycle[(q_identity + joint_index + edge_offset) % len(active_cycle)]
                candidates = active_candidates[requested]
                fallback = -1
                if not candidates:
                    candidates = descendant_owners
                    fallback = int(requested)
                owner_choice = _choose_candidate(candidates, generator=generator, device=device)
                stratum = (
                    QueryStratum.OWNER_SHELL
                    if edge_offset == 0
                    else (QueryStratum.ADJACENT if (q_identity + joint_index) % 2 == 0 else QueryStratum.WORKSPACE)
                )
                query_choice = _choose_query(
                    queries,
                    batch_index,
                    owner_choice,
                    stratum,
                    generator=generator,
                )
                owner_axis.append(owner_choice)
                query_axis.append(query_choice)
                joint_axis.append(joint_index)
                active_axis.append(True)
                category_axis.append(int(_actual_owner_category(
                    owner_choice,
                    joint_index=joint_index,
                    self_owners=self_owners,
                    tip_owners=tip_owners,
                    descendant_owners=descendant_owners,
                    zero_owners=zero_owners,
                    roles=roles,
                    fingers=fingers,
                    self_finger=self_finger,
                )))
                stratum_axis.append(int(stratum))
                fallback_axis.append(fallback)
                role_axis.append(int(
                    SensitivitySamplingRole.ACTIVE_OWNER_SHELL
                    if edge_offset == 0
                    else SensitivitySamplingRole.ACTIVE_CONTEXT
                ))
            for edge_offset in range(zero_per_joint):
                requested = zero_cycle[(q_identity + joint_index + edge_offset) % len(zero_cycle)]
                candidates = zero_candidates[requested]
                fallback = -1
                if not candidates:
                    candidates = zero_owners
                    fallback = int(requested)
                owner_choice = _choose_candidate(candidates, generator=generator, device=device)
                stratum = zero_strata[(q_identity + joint_index + edge_offset) % len(zero_strata)]
                query_choice = _choose_query(
                    queries,
                    batch_index,
                    owner_choice,
                    stratum,
                    generator=generator,
                )
                owner_axis.append(owner_choice)
                query_axis.append(query_choice)
                joint_axis.append(joint_index)
                active_axis.append(False)
                category_axis.append(int(_actual_owner_category(
                    owner_choice,
                    joint_index=joint_index,
                    self_owners=self_owners,
                    tip_owners=tip_owners,
                    descendant_owners=descendant_owners,
                    zero_owners=zero_owners,
                    roles=roles,
                    fingers=fingers,
                    self_finger=self_finger,
                )))
                stratum_axis.append(int(stratum))
                fallback_axis.append(fallback)
                role_axis.append(int(SensitivitySamplingRole.STRUCTURAL_ZERO))
        rows.append((owner_axis, query_axis, joint_axis, active_axis, category_axis, stratum_axis, fallback_axis, role_axis))
    columns = tuple(zip(*rows, strict=True))
    return (
        torch.tensor(columns[0], device=device, dtype=torch.long),
        torch.tensor(columns[1], device=device, dtype=torch.long),
        torch.tensor(columns[2], device=device, dtype=torch.long),
        torch.tensor(columns[3], device=device, dtype=torch.bool),
        torch.tensor(columns[4], device=device, dtype=torch.long),
        torch.tensor(columns[5], device=device, dtype=torch.long),
        torch.tensor(columns[6], device=device, dtype=torch.long),
        torch.tensor(columns[7], device=device, dtype=torch.long),
    )


def _cycle_owner_pool(
    self_owners: list[int],
    tip_owners: list[int],
    other_descendants: list[int],
    all_descendants: list[int],
) -> list[int]:
    r"""按 self / tip / other-descendant / other-descendant 的长期 25/25/50 轮换构造 active owner 池。"""

    ordered: list[int] = []
    for pool in (self_owners, tip_owners, other_descendants, other_descendants):
        ordered.extend(pool if pool else all_descendants)
    if not ordered:
        raise ValueError("joint-first active sampling requires at least one descendant owner")
    return ordered


def _cycle_zero_owner_pool(
    zero_owners: list[int],
    roles: tuple[str, ...],
    fingers: tuple[str | None, ...],
    self_finger: str | None,
) -> list[int]:
    r"""按 PALM / same-finger upstream / other-finger JOINT / other-finger TIP 轮换构造 structure-zero owner 池。"""

    if not zero_owners:
        raise ValueError("joint-first zero sampling requires at least one non-descendant owner")
    palm = [owner for owner in zero_owners if roles[owner] == "palm"]
    same_finger_upstream = [
        owner for owner in zero_owners if fingers[owner] == self_finger and roles[owner] != "palm"
    ]
    other_joint = [
        owner for owner in zero_owners if roles[owner] == "joint" and fingers[owner] != self_finger
    ]
    other_tip = [
        owner for owner in zero_owners if roles[owner] == "tip" and fingers[owner] != self_finger
    ]
    ordered: list[int] = []
    for pool in (palm, same_finger_upstream, other_joint, other_tip):
        ordered.extend(pool if pool else zero_owners)
    return ordered


def _choose_candidate(
    candidates: list[int],
    *,
    generator: torch.Generator,
    device: torch.device,
) -> int:
    """从稳定有序候选集合中抽一项；调用方负责记录类别 fallback。"""

    if not candidates:
        raise ValueError("sensitivity edge owner candidate set must be non-empty")
    cursor = torch.randint(len(candidates), (), generator=generator, device=device)
    return int(candidates[int(cursor)])


def _actual_owner_category(
    owner_index: int,
    *,
    joint_index: int,
    self_owners: list[int],
    tip_owners: list[int],
    descendant_owners: list[int],
    zero_owners: list[int],
    roles: tuple[str, ...],
    fingers: tuple[str | None, ...],
    self_finger: str | None,
) -> SensitivityOwnerCategory:
    """按实际被选 owner 与当前 JOINT 的关系恢复可审计类别。"""

    del joint_index  # 关系集合已经由当前 JOINT 构造，保留具名参数便于调用点核对语义
    if owner_index in self_owners:
        return SensitivityOwnerCategory.SELF
    if owner_index in tip_owners:
        return SensitivityOwnerCategory.SAME_FINGER_TIP
    if owner_index in descendant_owners:
        return SensitivityOwnerCategory.OTHER_DESCENDANT
    if owner_index not in zero_owners:
        return SensitivityOwnerCategory.FALLBACK
    if roles[owner_index] == "palm":
        return SensitivityOwnerCategory.PALM
    if fingers[owner_index] == self_finger:
        return SensitivityOwnerCategory.SAME_FINGER_UPSTREAM
    if roles[owner_index] == "joint":
        return SensitivityOwnerCategory.OTHER_FINGER_JOINT
    if roles[owner_index] == "tip":
        return SensitivityOwnerCategory.OTHER_FINGER_TIP
    return SensitivityOwnerCategory.FALLBACK


def _choose_query(
    queries: SpatialQueryBatch,
    batch_index: int,
    owner_index: int,
    stratum: QueryStratum,
    *,
    generator: torch.Generator,
) -> int:
    r"""从当前 q/owner 的指定物理 stratum 中确定性抽取一个 query 槽。"""

    candidates = torch.where(queries.query_stratum[batch_index, owner_index] == int(stratum))[0]
    if len(candidates) == 0:
        raise ValueError(f"owner {owner_index} has no {stratum.name} query for first-order edges")
    choice = candidates[torch.randint(len(candidates), (), generator=generator, device=candidates.device)]
    return int(choice)


__all__ = [
    "GaussianProximityFieldCfg",
    "GeometryFieldTargetCfg",
    "fixed_validation_gaussian_field_config",
    "generate_geometry_field_targets",
    "sample_geometry_bandwidths",
]
