"""Evaluate unsigned distance and Gaussian density targets."""


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


    if queries.query_points_h.device != q.device or queries.query_points_h.dtype != q.dtype:
        raise ValueError("q and query points must share device and dtype")
    if owner_transforms is None:
        owner_transforms = forward_owner_transforms(spec, q.detach())
    expected_transform_shape = (q.shape[0], spec.owner_home_transforms.shape[0], 4, 4)  # `[B,G,4,4]`
    if (
        owner_transforms.shape != expected_transform_shape
        or owner_transforms.device != q.device
        or owner_transforms.dtype != q.dtype
        or owner_transforms.requires_grad
    ):
        raise ValueError("owner_transforms must be detached [B,G,4,4] matching q/spec")


    surface = query_owner_surfaces_warp(
        queries.query_points_h,
        owner_transforms,
        warp_cache,
    )
    bandwidths = sample_geometry_bandwidths(
        field_config,
        batch_size=q.shape[0],
        device=q.device,
        dtype=q.dtype,
        sampling_seed=sampling_seed + 104_729,
    )
    density = gaussian_density_from_distance(surface.distance_m, bandwidths)  # `[B,G,N_Q,N_sigma]`
    valid = torch.isfinite(surface.distance_m) & (surface.face_index >= 0)
    role_index = {"palm": 0, "joint": 1, "tip": 2}
    owner_role = torch.tensor(
        [role_index[record.role] for record in geometry_cache.records],
        device=q.device,
        dtype=torch.long,
    )
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
