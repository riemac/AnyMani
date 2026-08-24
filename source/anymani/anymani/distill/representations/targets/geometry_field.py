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

import math
from dataclasses import dataclass, replace

import torch

from anymani.distill.representations.sources.collision_geometry import OwnerGeometryCache, WarpOwnerGeometryCache
from anymani.distill.representations.sources.kinematics import (
    EmbodimentGeometrySpec,
    forward_owner_transforms,
    selected_point_jacobian,
)

from ..fields.density import field_sensitivity_from_distance, gaussian_density_from_distance
from ..queries.spatial_sampling import SpatialQueryBatch
from .field_samples import FieldTargetBatch, QueryStratum, SensitivityTargetBatch
from .warp_surface import query_owner_surfaces_warp


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
    owner_transforms = forward_owner_transforms(spec, q.detach())  # teacher/query 路径停止 q 梯度
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
    owner_index, query_index, joint_index, active_mask = _sample_sensitivity_edges(
        spec,
        queries,
        active_per_joint=active_per_joint,
        zero_per_joint=zero_per_joint,
        sampling_seed=edge_sampling_seed,
    )
    closest_h = surface.closest_point_h_m[:, owner_index, query_index]
    selected_query_h = queries.query_points_h[:, owner_index, query_index]
    selected_distance = surface.distance_m[:, owner_index, query_index]
    selected_feature_margin = surface.feature_margin_m[:, owner_index, query_index]
    selected_face = surface.face_index[:, owner_index, query_index]
    selected_transform = owner_transforms.index_select(1, owner_index)
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
    )  # `[B,E,3]`，m/rad；非祖先严格为零
    radial_direction = (selected_query_h - closest_h) / selected_distance.clamp_min(
        target_config.distance_epsilon_m
    ).unsqueeze(-1)
    kappa = -(radial_direction * point_jacobian).sum(dim=-1)  # $-n^TJ$，m/rad
    ancestor_mask = spec.owner_ancestor_mask[owner_index, joint_index]
    kappa = torch.where(ancestor_mask.unsqueeze(0), kappa, torch.zeros_like(kappa))
    selected_density = gaussian_density_from_distance(selected_distance, bandwidths)
    field_sensitivity = field_sensitivity_from_distance(
        selected_distance,
        selected_density,
        bandwidths,
        kappa.unsqueeze(-1),
    ).squeeze(-1)  # `[B,E,L]`，1/rad
    selected_shell = queries.query_stratum[:, owner_index, query_index] == int(QueryStratum.OWNER_SHELL)
    selected_face_valid = field_valid[:, owner_index, query_index] & selected_shell
    active_smooth = (
        selected_face_valid
        & (selected_distance > target_config.distance_epsilon_m)
        & (selected_feature_margin >= target_config.feature_margin_min_m)
    )
    selected_valid = torch.where(active_mask.unsqueeze(0), active_smooth, selected_face_valid)
    closest_source = (
        owner_index.to(torch.int64).view(1, -1).bitwise_left_shift(32)
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


def _sample_sensitivity_edges(
    spec: EmbodimentGeometrySpec,
    queries: SpatialQueryBatch,
    *,
    active_per_joint: int,
    zero_per_joint: int,
    sampling_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""按 JOINT 列覆盖从 owner-shell query 抽取 active descendant 与 structure-zero 边。"""

    device = queries.query_stratum.device
    generator = torch.Generator(device=device)
    generator.manual_seed(int(sampling_seed))
    owner_axis: list[int] = []
    query_axis: list[int] = []
    joint_axis: list[int] = []
    active_axis: list[bool] = []
    owner_count = spec.owner_ancestor_mask.shape[0]
    joint_count = spec.owner_ancestor_mask.shape[1]
    roles = spec.owner_roles or tuple("joint" for _ in range(owner_count))
    fingers = spec.owner_finger_names or tuple(None for _ in range(owner_count))
    owner_joint_indices = spec.owner_joint_indices or tuple(-1 for _ in range(owner_count))
    for joint_index in range(joint_count):
        descendant_owners = torch.where(spec.owner_ancestor_mask[:, joint_index])[0]
        zero_owners = torch.where(~spec.owner_ancestor_mask[:, joint_index])[0]
        self_owners = [
            owner_index
            for owner_index, mapped_joint in enumerate(owner_joint_indices)
            if mapped_joint == joint_index
        ]
        self_finger = fingers[self_owners[0]] if self_owners else None
        tip_owners = [
            int(owner_index)
            for owner_index in descendant_owners.tolist()
            if roles[int(owner_index)] == "tip" and fingers[int(owner_index)] == self_finger
        ]
        other_descendants = [
            int(owner_index)
            for owner_index in descendant_owners.tolist()
            if int(owner_index) not in self_owners and int(owner_index) not in tip_owners
        ]
        active_pool = _cycle_owner_pool(self_owners, tip_owners, other_descendants, descendant_owners.tolist())
        zero_pool = _cycle_zero_owner_pool(zero_owners.tolist(), roles, fingers, self_finger)
        for edge_offset in range(active_per_joint):
            owner_choice = active_pool[edge_offset % len(active_pool)]
            query_choice = _choose_shell_query(queries, owner_choice, generator=generator)
            owner_axis.append(owner_choice)
            query_axis.append(query_choice)
            joint_axis.append(joint_index)
            active_axis.append(True)
        for edge_offset in range(zero_per_joint):
            owner_choice = zero_pool[edge_offset % len(zero_pool)]
            query_choice = _choose_shell_query(queries, owner_choice, generator=generator)
            owner_axis.append(owner_choice)
            query_axis.append(query_choice)
            joint_axis.append(joint_index)
            active_axis.append(False)
    return (
        torch.tensor(owner_axis, device=device, dtype=torch.long),
        torch.tensor(query_axis, device=device, dtype=torch.long),
        torch.tensor(joint_axis, device=device, dtype=torch.long),
        torch.tensor(active_axis, device=device, dtype=torch.bool),
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


def _choose_shell_query(
    queries: SpatialQueryBatch,
    owner_index: int,
    *,
    generator: torch.Generator,
) -> int:
    r"""从指定 owner 的 owner-shell query 中确定性抽取一个槽。"""

    shell_queries = torch.where(queries.query_stratum[0, owner_index] == int(QueryStratum.OWNER_SHELL))[0]
    if len(shell_queries) == 0:
        raise ValueError(f"owner {owner_index} has no owner-shell queries for first-order edges")
    choice = shell_queries[torch.randint(len(shell_queries), (), generator=generator, device=shell_queries.device)]
    return int(choice)


__all__ = [
    "GaussianProximityFieldCfg",
    "GeometryFieldTargetCfg",
    "fixed_validation_gaussian_field_config",
    "generate_geometry_field_targets",
    "sample_geometry_bandwidths",
]
