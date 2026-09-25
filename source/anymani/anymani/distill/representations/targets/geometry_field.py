"""Gaussian density supervision and active or structural-zero query edges. Distances and bandwidths use metres; density is unitless."""


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


    central_difference: torch.Tensor
    central_difference_valid_mask: torch.Tensor
    central_difference_plus_face: torch.Tensor
    central_difference_minus_face: torch.Tensor
    central_difference_elapsed_seconds: float


@dataclass(frozen=True)
class GaussianProximityFieldCfg:


    bandwidth_centers_m: tuple[float, ...] = (0.004, 0.016, 0.064)
    bandwidth_jitter_relative: float = 0.10
    fixed_bandwidths_m: tuple[float, ...] = (0.004, 0.016, 0.064)
    def __post_init__(self) -> None:


        if not self.bandwidth_centers_m or any(value <= 0.0 for value in self.bandwidth_centers_m):
            raise ValueError("bandwidth_centers_m must contain strictly positive values")
        if any(left >= right for left, right in zip(self.bandwidth_centers_m[:-1], self.bandwidth_centers_m[1:])):
            raise ValueError("bandwidth_centers_m must be strictly increasing")
        if not self.fixed_bandwidths_m or any(value <= 0.0 for value in self.fixed_bandwidths_m):
            raise ValueError("fixed_bandwidths_m must contain strictly positive values")
        if any(left >= right for left, right in zip(self.fixed_bandwidths_m[:-1], self.fixed_bandwidths_m[1:])):
            raise ValueError("fixed_bandwidths_m must be strictly increasing")
        if not 0.0 <= self.bandwidth_jitter_relative < 1.0:
            raise ValueError("bandwidth_jitter_relative must lie in [0,1)")


@dataclass(frozen=True)
class GeometryFieldTargetCfg:


    train_active_per_joint: int = 1
    train_zero_per_joint: int = 1
    fixed_active_per_joint: int = 4
    fixed_zero_per_joint: int = 4
    distance_epsilon_m: float = 1.0e-6
    feature_margin_min_m: float = 1.0e-5

    def __post_init__(self) -> None:


        counts = (
            self.train_active_per_joint,
            self.train_zero_per_joint,
            self.fixed_active_per_joint,
            self.fixed_zero_per_joint,
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


    centers = torch.tensor(config.bandwidth_centers_m, device=device, dtype=dtype)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(sampling_seed))
    relative = config.bandwidth_jitter_relative
    lower = math.log1p(-relative)
    upper = math.log1p(relative)
    epsilon = lower + (upper - lower) * torch.rand(centers.shape, generator=generator, device=device, dtype=dtype)
    realization = centers * torch.exp(epsilon)
    return realization.unsqueeze(0).expand(batch_size, -1)


def fixed_gaussian_field_config(config: GaussianProximityFieldCfg) -> GaussianProximityFieldCfg:


    return replace(
        config,
        bandwidth_centers_m=config.fixed_bandwidths_m,
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


    if queries.query_points_h.device != q.device or queries.query_points_h.dtype != q.dtype:
        raise ValueError("q and query points must share CUDA device and float dtype")
    if owner_transforms is None:
        owner_transforms = forward_owner_transforms(spec, q.detach())
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
    bandwidths = sample_geometry_bandwidths(
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
        active_per_joint = target_config.fixed_active_per_joint
        zero_per_joint = target_config.fixed_zero_per_joint
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
    ).squeeze(-1)
    point_jacobian = selected_point_jacobian(
        spec,
        q.detach(),
        owner_index,
        joint_index,
        closest_local,
        owner_transforms=owner_transforms,
        current_spatial_screws=current_spatial_screws,
    )
    radial_direction = (selected_query_h - closest_h) / selected_distance.clamp_min(
        target_config.distance_epsilon_m
    ).unsqueeze(-1)
    kappa = -(radial_direction * point_jacobian).sum(dim=-1)
    ancestor_mask = spec.owner_ancestor_mask[owner_index, joint_index]
    ancestor_for_batch = ancestor_mask if ancestor_mask.ndim == 2 else ancestor_mask.unsqueeze(0)
    kappa = torch.where(ancestor_for_batch, kappa, torch.zeros_like(kappa))
    selected_density = (
        density[:, owner_index, query_index]
        if owner_index.ndim == 1
        else density[torch.arange(q.shape[0], device=q.device).unsqueeze(1), owner_index, query_index]
    )
    field_sensitivity = field_sensitivity_from_distance(
        selected_distance,
        selected_density,
        bandwidths,
        kappa.unsqueeze(-1),
    ).squeeze(-1)
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
    )
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


    batch_size, edge_count = owner_index.shape if owner_index.ndim == 2 else (q.shape[0], owner_index.shape[0])
    difference = torch.zeros(batch_size, edge_count, device=q.device, dtype=q.dtype)
    valid = torch.zeros(batch_size, edge_count, device=q.device, dtype=torch.bool)
    plus_face = torch.full((batch_size, edge_count), -1, device=q.device, dtype=torch.long)
    minus_face = torch.full_like(plus_face, -1)
    elapsed_seconds = 0.0
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
        audit_started = perf_counter()
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


    del joint_index
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


    candidates = torch.where(queries.query_stratum[batch_index, owner_index] == int(stratum))[0]
    if len(candidates) == 0:
        raise ValueError(f"owner {owner_index} has no {stratum.name} query for first-order edges")
    choice = candidates[torch.randint(len(candidates), (), generator=generator, device=candidates.device)]
    return int(choice)


__all__ = [
    "GaussianProximityFieldCfg",
    "GeometryFieldTargetCfg",
    "fixed_gaussian_field_config",
    "generate_geometry_field_targets",
    "sample_geometry_bandwidths",
]
