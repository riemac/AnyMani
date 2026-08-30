r"""N040 q-block 级随机 proper-SE(3) coordinate augmentation。"""

from __future__ import annotations

from dataclasses import replace

import torch

from anymani.distill.models.input_adapters.se3_gauge import rewrite_static_geometry_evidence_se3
from anymani.distill.representations.queries.spatial_sampling import SpatialQueryBatch

from .batch import PaddedDensityGammaBatch
from .se3_config import SE3CoordinateRewriteCfg


def _random_so3(count: int, *, generator: torch.Generator, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    r"""由 normalized Gaussian quaternion 采样 Haar-SO(3) rotation。"""

    quaternion = torch.randn(count, 4, generator=generator, dtype=torch.float64)
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    w, x, y, z = quaternion.unbind(dim=-1)
    rotation = torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(count, 3, 3)
    return rotation.to(device=device, dtype=dtype)


def sample_se3_coordinate_rewrite(
    count: int,
    *,
    config: SE3CoordinateRewriteCfg,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""为 unique evidence rows 返回 `[A,3,3]` rotations 与 `[A,3]` translations。"""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(config.seed_offset))
    rotation = _random_so3(count, generator=generator, device=device, dtype=dtype)
    translation = (
        (2.0 * torch.rand(count, 3, generator=generator, dtype=torch.float64) - 1.0)
        * config.translation_half_extent_m
    ).to(device=device, dtype=dtype)
    selected = torch.rand(count, generator=generator) < config.probability
    identity = torch.eye(3, device=device, dtype=dtype).expand(count, -1, -1)
    rotation = torch.where(selected[:, None, None].to(device), rotation, identity)
    translation = torch.where(selected[:, None].to(device), translation, torch.zeros_like(translation))
    return rotation, translation


def _transform_points(points: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    r"""按 q row 执行 $p'=Rp+t$。"""

    rotated = torch.einsum("b...j,bij->b...i", points, rotation)
    shape = (translation.shape[0],) + (1,) * (points.ndim - 2) + (3,)
    return rotated + translation.view(shape)


def maybe_rewrite_density_gamma_batch_se3(
    batch: PaddedDensityGammaBatch,
    *,
    config: SE3CoordinateRewriteCfg,
    seed: int,
) -> PaddedDensityGammaBatch:
    r"""同步改写 evidence/query/material vectors，并保持 density/Gamma scalar truth 不变。"""

    if config.probability <= 0.0:
        return batch
    asset_count = batch.evidence.anchors.shape[0]
    rotation, translation = sample_se3_coordinate_rewrite(
        asset_count,
        config=config,
        seed=seed,
        device=batch.q.device,
        dtype=batch.q.dtype,
    )
    evidence = rewrite_static_geometry_evidence_se3(
        batch.evidence,
        rotation=rotation,
        translation=translation,
    )
    row_rotation = rotation[batch.evidence_row_index]
    row_translation = translation[batch.evidence_row_index]
    query_points = _transform_points(batch.queries.query_points_h, row_rotation, row_translation)
    queries = SpatialQueryBatch(
        query_points,
        batch.queries.query_stratum,
        batch.queries.adjacent_owner_index,
        batch.queries.workspace_anchor_index,
    )
    field = replace(batch.field_targets, query_points=query_points)
    target = batch.material_targets
    material_points = _transform_points(target.material_points_h_m, row_rotation, row_translation)
    point_jacobian = torch.einsum("bej,bij->bei", target.point_jacobian_h_m_per_rad, row_rotation)
    material_target = replace(
        target,
        material_points_h_m=material_points,
        point_jacobian_h_m_per_rad=point_jacobian,
    )
    return replace(
        batch,
        evidence=evidence,
        queries=queries,
        field_targets=field,
        material_targets=material_target,
    )


__all__ = [
    "maybe_rewrite_density_gamma_batch_se3",
    "sample_se3_coordinate_rewrite",
]
