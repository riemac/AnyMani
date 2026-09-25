"""Sample 50 percent workspace, 25 percent owner-shell, and 25 percent adjacent queries in hand frame {h}."""


from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from anymani.distill.representations.sources.collision_geometry import (
    OwnerGeometryCache,
    OwnerSurfaceSamplingArrays,
    prepare_owner_surface_sampling_arrays,
)
from anymani.distill.representations.sources.kinematics import EmbodimentGeometrySpec, forward_owner_transforms

from ..targets.field_samples import QueryStratum

SURFACE_QUERY_SAMPLING_VERSION = "owner-triangle-area-barycentric-v1"


@dataclass(frozen=True)
class SpatialQuerySamplerCfg:


    query_count: int = 64
    workspace_fraction: float = 0.50
    owner_shell_fraction: float = 0.25
    adjacent_fraction: float = 0.25
    workspace_radius_m: float = 0.05
    shell_offset_min_m: float = 0.0005
    shell_offset_max_m: float = 0.004
    adjacent_candidate_count: int = 4

    def __post_init__(self) -> None:


        if self.query_count < 4:
            raise ValueError("query_count must leave at least one query in each stratum")
        fractions = (self.workspace_fraction, self.owner_shell_fraction, self.adjacent_fraction)
        if any(fraction < 0.0 for fraction in fractions) or not np.isclose(sum(fractions), 1.0):
            raise ValueError("query stratum fractions must be non-negative and sum to one")
        counts = tuple(round(self.query_count * fraction) for fraction in fractions)
        if sum(counts) != self.query_count or any(count < 1 for count in counts):
            raise ValueError(
                f"query_count={self.query_count} cannot represent configured 50/25/25 mixture as integers"
            )
        if counts[1] % 2:
            raise ValueError("owner-shell query count must be even for an exact 50/50 inside/outside split")
        if self.workspace_radius_m <= 0.0:
            raise ValueError("workspace anchor-cloud radius must be positive")
        if not 0.0 < self.shell_offset_min_m <= self.shell_offset_max_m:
            raise ValueError("shell offsets must be positive and ordered")
        if self.adjacent_candidate_count < 1:
            raise ValueError("adjacent_candidate_count must be positive")

    @property
    def stratum_counts(self) -> tuple[int, int, int]:


        return (
            round(self.query_count * self.workspace_fraction),
            round(self.query_count * self.owner_shell_fraction),
            round(self.query_count * self.adjacent_fraction),
        )


@dataclass(frozen=True)
class SpatialQueryBatch:


    query_points_h: torch.Tensor
    query_stratum: torch.Tensor
    adjacent_owner_index: torch.Tensor
    workspace_anchor_index: torch.Tensor


@dataclass(frozen=True)
class OwnerSurfaceSamplingCache:


    vertices_owner_local_m: tuple[torch.Tensor, ...]
    faces: tuple[torch.Tensor, ...]
    face_normals_owner_local: tuple[torch.Tensor, ...]
    face_area_cdf: tuple[torch.Tensor, ...]


def materialize_owner_surface_sampling_cache(
    cache: OwnerGeometryCache,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    arrays: OwnerSurfaceSamplingArrays | None = None,
) -> OwnerSurfaceSamplingCache:


    source_arrays = arrays if arrays is not None else prepare_owner_surface_sampling_arrays(cache)
    if len(source_arrays.vertices_owner_local_m) != len(cache.records):
        raise ValueError("owner surface sampling array count must match geometry cache")
    vertices: list[torch.Tensor] = []
    faces: list[torch.Tensor] = []
    normals: list[torch.Tensor] = []
    cdfs: list[torch.Tensor] = []
    for owner_index, record in enumerate(cache.records):
        vertex = torch.tensor(source_arrays.vertices_owner_local_m[owner_index], device=device, dtype=dtype)
        face = torch.tensor(source_arrays.faces[owner_index], device=device, dtype=torch.long)
        normal = torch.tensor(source_arrays.face_normals_owner_local[owner_index], device=device, dtype=dtype)
        cdf = torch.tensor(source_arrays.face_area_cdf[owner_index], device=device, dtype=dtype)
        if vertex.ndim != 2 or vertex.shape[1:] != (3,) or face.ndim != 2 or face.shape[1:] != (3,) or cdf.ndim != 1:
            raise ValueError(f"owner {record.owner_id!r} surface sampling cache requires positive triangles")
        cdf[-1] = 1.0
        vertices.append(vertex.contiguous())
        faces.append(face.contiguous())
        normals.append(normal.contiguous())
        cdfs.append(cdf.contiguous())
    return OwnerSurfaceSamplingCache(tuple(vertices), tuple(faces), tuple(normals), tuple(cdfs))


def sample_spatial_queries(
    q: torch.Tensor,
    spec: EmbodimentGeometrySpec,
    surface_sampling: OwnerSurfaceSamplingCache,
    anchors_hand_m: torch.Tensor,
    *,
    config: SpatialQuerySamplerCfg = SpatialQuerySamplerCfg(),
    sampling_seed: int = 0,
    owner_transforms: torch.Tensor | None = None,
) -> SpatialQueryBatch:


    if q.ndim != 2 or q.shape[1] != spec.space_screws.shape[0]:
        raise ValueError("q must have shape [B,N_J]")
    if anchors_hand_m.ndim != 2 or anchors_hand_m.shape[-1] != 3 or anchors_hand_m.shape[0] < 1:
        raise ValueError("anchors_hand_m must have non-empty shape [K,3]")
    if len(surface_sampling.vertices_owner_local_m) != spec.owner_home_transforms.shape[0]:
        raise ValueError("surface sampling cache/spec owner axes must match")

    batch_size = q.shape[0]
    owner_count = spec.owner_home_transforms.shape[0]
    workspace_count, shell_count, adjacent_count = config.stratum_counts
    generator = torch.Generator(device=q.device)
    generator.manual_seed(int(sampling_seed))
    if owner_transforms is None:
        owner_transforms = forward_owner_transforms(spec, q.detach())
    expected_transform_shape = (batch_size, owner_count, 4, 4)
    if (
        owner_transforms.shape != expected_transform_shape
        or owner_transforms.device != q.device
        or owner_transforms.dtype != q.dtype
        or owner_transforms.requires_grad
    ):
        raise ValueError(
            "owner_transforms must be detached [B,G,4,4] on the same device/dtype as q"
        )
    workspace, workspace_anchor = _sample_anchor_workspace_queries(
        anchors_hand_m.to(device=q.device, dtype=q.dtype),
        batch_size=batch_size,
        owner_count=owner_count,
        workspace_count=workspace_count,
        radius_m=config.workspace_radius_m,
        generator=generator,
    )
    shell = _sample_owner_shell_queries(
        owner_transforms,
        surface_sampling,
        shell_count=shell_count,
        config=config,
        generator=generator,
    )
    adjacent, adjacent_owner_index = _sample_adjacent_queries(
        owner_transforms,
        surface_sampling,
        spec,
        adjacent_count=adjacent_count,
        candidate_count=config.adjacent_candidate_count,
        generator=generator,
    )
    query_points = torch.cat((workspace, shell, adjacent), dim=2).detach()
    query_stratum = torch.cat(
        (
            torch.full((batch_size, owner_count, workspace_count), int(QueryStratum.WORKSPACE), device=q.device),
            torch.full((batch_size, owner_count, shell_count), int(QueryStratum.OWNER_SHELL), device=q.device),
            torch.full((batch_size, owner_count, adjacent_count), int(QueryStratum.ADJACENT), device=q.device),
        ),
        dim=2,
    )
    adjacent_index = torch.cat(
        (
            torch.full((batch_size, owner_count, workspace_count + shell_count), -1, device=q.device),
            adjacent_owner_index,
        ),
        dim=2,
    )
    workspace_index = torch.cat(
        (
            workspace_anchor,
            torch.full((batch_size, owner_count, shell_count + adjacent_count), -1, device=q.device),
        ),
        dim=2,
    )
    return SpatialQueryBatch(query_points, query_stratum, adjacent_index, workspace_index)


def _sample_anchor_workspace_queries(
    anchors_hand_m: torch.Tensor,
    *,
    batch_size: int,
    owner_count: int,
    workspace_count: int,
    radius_m: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:


    device = anchors_hand_m.device
    anchor_index = torch.randint(
        anchors_hand_m.shape[0],
        (workspace_count,),
        generator=generator,
        device=device,
    )
    direction = torch.randn((workspace_count, 3), generator=generator, device=device)
    direction = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min(1.0e-12)
    radial = radius_m * torch.rand(
        (workspace_count, 1), generator=generator, device=device
    ).pow(1.0 / 3.0)
    realization = anchors_hand_m.index_select(0, anchor_index) + radial * direction
    workspace = realization.view(1, 1, workspace_count, 3).expand(batch_size, owner_count, -1, -1)
    provenance = anchor_index.view(1, 1, workspace_count).expand(batch_size, owner_count, -1)
    return workspace, provenance


def _sample_owner_shell_queries(
    owner_transforms: torch.Tensor,
    surface_sampling: OwnerSurfaceSamplingCache,
    *,
    shell_count: int,
    config: SpatialQuerySamplerCfg,
    generator: torch.Generator,
) -> torch.Tensor:


    points, normals = _sample_current_owner_surface(
        owner_transforms,
        surface_sampling,
        sample_count=shell_count,
        generator=generator,
    )
    batch_size, owner_count = owner_transforms.shape[:2]
    signs = torch.ones((batch_size, owner_count, shell_count, 1), device=owner_transforms.device)
    signs[..., : shell_count // 2, :] = -1.0
    offsets = torch.rand(
        (batch_size, owner_count, shell_count, 1), generator=generator, device=owner_transforms.device
    )
    offsets = config.shell_offset_min_m + offsets * (config.shell_offset_max_m - config.shell_offset_min_m)
    return points + signs * offsets * normals


def _sample_adjacent_queries(
    owner_transforms: torch.Tensor,
    surface_sampling: OwnerSurfaceSamplingCache,
    spec: EmbodimentGeometrySpec,
    *,
    adjacent_count: int,
    candidate_count: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:


    batch_size, owner_count = owner_transforms.shape[:2]
    if spec.owner_graph_shortest is None:
        raise ValueError("spec must contain owner graph distances for adjacent query sampling")
    neighbors = [
        torch.where(spec.owner_graph_shortest[owner_index] == 1)[0]
        for owner_index in range(owner_count)
    ]
    if any(len(values) == 0 for values in neighbors):
        raise ValueError("every owner must have at least one graph neighbor for adjacent queries")
    selected_neighbor = torch.empty(
        (batch_size, owner_count, adjacent_count), device=owner_transforms.device, dtype=torch.long
    )
    for owner_index, values in enumerate(neighbors):
        choice = torch.randint(
            len(values),
            (batch_size, adjacent_count),
            generator=generator,
            device=owner_transforms.device,
        )
        selected_neighbor[:, owner_index] = values.to(device=owner_transforms.device).index_select(0, choice.reshape(-1)).reshape(
            batch_size, adjacent_count
        )

    sample_count = adjacent_count * candidate_count
    left, _ = _sample_current_owner_surface(
        owner_transforms, surface_sampling, sample_count=sample_count, generator=generator
    )
    right_all, _ = _sample_current_owner_surface(
        owner_transforms, surface_sampling, sample_count=sample_count, generator=generator
    )
    left = left.reshape(batch_size, owner_count, adjacent_count, candidate_count, 3)
    right_all = right_all.reshape(batch_size, owner_count, adjacent_count, candidate_count, 3)
    neighbor_gather = selected_neighbor.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, candidate_count, 3)
    right = torch.gather(right_all, 1, neighbor_gather)  # `[B,G,N_A,C_A,3]` selected neighbor candidates
    distances = torch.linalg.vector_norm(left - right, dim=-1)
    best = torch.argmin(distances, dim=-1)
    best_index = best.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, 3)
    left_best = torch.gather(left, 3, best_index).squeeze(3)
    right_best = torch.gather(right, 3, best_index).squeeze(3)
    interpolation = 0.25 + 0.5 * torch.rand(
        (batch_size, owner_count, adjacent_count, 1), generator=generator, device=owner_transforms.device
    )
    return (1.0 - interpolation) * left_best + interpolation * right_best, selected_neighbor


def _sample_current_owner_surface(
    owner_transforms: torch.Tensor,
    surface_sampling: OwnerSurfaceSamplingCache,
    *,
    sample_count: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:


    batch_size, owner_count = owner_transforms.shape[:2]
    points = torch.empty(
        (batch_size, owner_count, sample_count, 3), device=owner_transforms.device, dtype=owner_transforms.dtype
    )
    normals = torch.empty_like(points)  # `[B,G,N,3]` current face unit normals
    for owner_index in range(owner_count):
        cdf = surface_sampling.face_area_cdf[owner_index]
        face_uniform = torch.rand(
            (batch_size, sample_count), generator=generator, device=owner_transforms.device
        )
        face_index = torch.searchsorted(cdf, face_uniform.contiguous())
        face_vertices = surface_sampling.faces[owner_index][face_index]  # `[B,N,3]` selected vertex indices
        triangles = surface_sampling.vertices_owner_local_m[owner_index][face_vertices]  # `[B,N,3,3]`
        root = torch.sqrt(torch.rand(
            (batch_size, sample_count, 1), generator=generator, device=owner_transforms.device
        ))
        second = torch.rand(
            (batch_size, sample_count, 1), generator=generator, device=owner_transforms.device
        )
        barycentric = torch.cat((1.0 - root, root * (1.0 - second), root * second), dim=-1)
        local_point = torch.sum(triangles * barycentric.unsqueeze(-1), dim=-2)
        local_normal = surface_sampling.face_normals_owner_local[owner_index][face_index]  # `[B,N,3]`
        rotation = owner_transforms[:, owner_index, :3, :3]
        translation = owner_transforms[:, owner_index, :3, 3]
        points[:, owner_index] = torch.einsum("bij,bnj->bni", rotation, local_point) + translation.unsqueeze(1)
        normals[:, owner_index] = torch.einsum("bij,bnj->bni", rotation, local_normal)
    return points, normals


__all__ = [
    "OwnerSurfaceSamplingCache",
    "SURFACE_QUERY_SAMPLING_VERSION",
    "SpatialQueryBatch",
    "SpatialQuerySamplerCfg",
    "materialize_owner_surface_sampling_cache",
    "sample_spatial_queries",
]
