r"""只生成 Gaussian density 的在线表面场教师。

给定当前构型 $q$、owner surface $\partial\Omega_g(q)$ 与 hand-frame query $x$，先计算 unsigned
Euclidean distance：

$$
d_g(x;q)=\min_{y\in\partial\Omega_g(q)}\|x-y\|_2,
$$

再按显式米制带宽形成：

$$
\rho_{\sigma,g}(x;q)=\exp\left[-\frac{d_g(x;q)^2}{2\sigma^2}\right].
$$

本模块不采样 owner/query/JOINT edges，不计算 closest material-point Jacobian，也不构造 κ/g。它供
Gaussian density + anchor-relational Material-point Jacobian method 复用，使零阶 teacher 保持与 v0.7.5
相同的 Warp surface、query 和 bandwidth 语义，同时从运行时彻底删除旧一阶最近点链路。
"""

from __future__ import annotations

import torch

from anymani.distill.representations.queries.spatial_sampling import SpatialQueryBatch
from anymani.distill.representations.sources.collision_geometry import OwnerGeometryCache, WarpOwnerGeometryCache
from anymani.distill.representations.sources.kinematics import EmbodimentGeometrySpec, forward_owner_transforms
from anymani.distill.representations.targets.field_samples import FieldTargetBatch
from anymani.distill.representations.targets.geometry_field import GaussianProximityFieldCfg, sample_geometry_bandwidths
from anymani.distill.representations.targets.warp_surface import query_owner_surfaces_warp

from ..fields.density import gaussian_density_from_distance


def generate_density_field_targets(
    q: torch.Tensor,
    spec: EmbodimentGeometrySpec,
    geometry_cache: OwnerGeometryCache,
    warp_cache: WarpOwnerGeometryCache,
    queries: SpatialQueryBatch,
    *,
    field_config: GaussianProximityFieldCfg = GaussianProximityFieldCfg(),
    sampling_seed: int = 0,
    owner_transforms: torch.Tensor | None = None,
) -> FieldTargetBatch:
    r"""生成完整 owner/query/sigma Gaussian density truth，不触发任何一阶 teacher。

    Args:
        q (torch.Tensor): 当前构型，形状 `[B,N_J]`，rad，CUDA floating tensor。
        spec (EmbodimentGeometrySpec): 与 q 同 device/dtype 的 POE 与 owner 语义。
        geometry_cache (OwnerGeometryCache): owner role 与 physical identity provenance。
        warp_cache (WarpOwnerGeometryCache): 当前资产 owner-local Warp BVHs。
        queries (SpatialQueryBatch): `[B,G,N_Q,3]` hand-frame query realization。
        field_config (GaussianProximityFieldCfg): train jitter 或 fixed evaluation bandwidth measure。
        sampling_seed (int): 当前 q-block 的 bandwidth realization seed。
        owner_transforms (torch.Tensor | None): 可复用 `[B,G,4,4]` 当前 owner poses。

    Returns:
        FieldTargetBatch: distance `[B,G,N_Q]`、density `[B,G,N_Q,N_sigma]` 与 valid mask。
    """

    if queries.query_points_h.device != q.device or queries.query_points_h.dtype != q.dtype:
        raise ValueError("q and query points must share device and dtype")
    if owner_transforms is None:
        owner_transforms = forward_owner_transforms(spec, q.detach())  # `[B,G,4,4]`，只在未复用时计算 POE
    expected_transform_shape = (q.shape[0], spec.owner_home_transforms.shape[0], 4, 4)  # `[B,G,4,4]`
    if (
        owner_transforms.shape != expected_transform_shape
        or owner_transforms.device != q.device
        or owner_transforms.dtype != q.dtype
        or owner_transforms.requires_grad
    ):
        raise ValueError("owner_transforms must be detached [B,G,4,4] matching q/spec")

    # Warp 只返回 zero-order nearest-surface distance；closest face/point 不进入 retained 输入或新一阶目标。
    surface = query_owner_surfaces_warp(
        queries.query_points_h,
        owner_transforms,
        warp_cache,
    )  # `[B,G,N_Q]` unsigned distance 与 face provenance
    bandwidths = sample_geometry_bandwidths(
        field_config,
        batch_size=q.shape[0],
        device=q.device,
        dtype=q.dtype,
        sampling_seed=sampling_seed + 104_729,
    )  # `[B,N_sigma]`，同资产 q-block 共享实际 sigma realization
    density = gaussian_density_from_distance(surface.distance_m, bandwidths)  # `[B,G,N_Q,N_sigma]`
    valid = torch.isfinite(surface.distance_m) & (surface.face_index >= 0)  # `[B,G,N_Q]`，zero-order 有效域
    role_index = {"palm": 0, "joint": 1, "tip": 2}  # 与 unified entity role contract 一致
    owner_role = torch.tensor(
        [role_index[record.role] for record in geometry_cache.records],
        device=q.device,
        dtype=torch.long,
    )  # `[G]`，physical owner role
    return FieldTargetBatch(
        query_points=queries.query_points_h.detach(),
        query_stratum=queries.query_stratum,
        distance=surface.distance_m.detach(),
        density=density.detach(),
        valid_mask=valid.detach(),
        owner_role=owner_role,
        bandwidths=bandwidths,
        provenance={
            "frame": "h",
            "length_unit": "m",
            "backend": "warp_mesh_query_point_density_only",
            "asset_content_hash": geometry_cache.asset_content_hash,
            "query_mixture": "workspace=0.50,owner_shell=0.25,adjacent=0.25",
            "first_order_teacher": "absent",
        },
    )


__all__ = ["generate_density_field_targets"]
