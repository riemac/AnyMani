r"""Density + relational Material Jacobian 的物理 sample、identity sampling 与跨结构 padding。"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, TypeVar, cast

import torch

from anymani.distill.models.input_adapters.geometry import (
    GeometryPaddingCfg,
    StaticGeometryEvidence,
    build_static_geometry_evidence,
    pad_static_geometry_evidence,
)
from anymani.distill.representations.geometry import GeometryRepresentationState
from anymani.distill.representations.queries.spatial_sampling import SpatialQueryBatch, sample_spatial_queries
from anymani.distill.representations.sources.kinematics import forward_owner_transforms_and_spatial_screws
from anymani.distill.representations.targets.density_field import generate_density_field_targets
from anymani.distill.representations.targets.field_samples import FieldTargetBatch
from anymani.distill.representations.targets.material_point_jacobian import (
    MaterialPointRelationJacobianTarget,
    generate_material_point_relation_jacobian_targets,
)

from .config import DensityMaterialJacobianMethodCfg

_DataclassT = TypeVar("_DataclassT")


@dataclass(frozen=True)
class DensityGammaOnlineSample:
    r"""一项资产完整 q-block 的 zero/first-order physical truth 与 retained evidence。"""

    asset_id: str
    q: torch.Tensor  # `[Q,N_J]`，rad
    evidence: StaticGeometryEvidence  # 单资产、当前 anchor realization
    queries: SpatialQueryBatch  # `[Q,G,N_Q,...]`
    field_targets: FieldTargetBatch  # distance/density zero-order truth
    material_targets: MaterialPointRelationJacobianTarget  # `[Q,E,K,4]` Gamma truth
    material_point_index: torch.Tensor  # `[Q,E]`，home-surface identity；同 q-block 行相同
    edge_valid_mask: torch.Tensor  # `[Q,E]`，source-local 全 True，padding 后区分无效槽
    anchor_index: int = 0  # 当前 $A^{(k)}$
    q_index: torch.Tensor | None = None  # `[Q]`，资产本地 Sobol cursor


@dataclass(frozen=True)
class PaddedDensityGammaBatch:
    r"""跨 morphology 的 dense storage；所有 padding 都由显式 masks 删除。"""

    asset_ids: tuple[str, ...]
    q: torch.Tensor  # `[B,N_J^max]`
    evidence: StaticGeometryEvidence  # `[A_unique,G^max,K^max,...]`
    evidence_row_index: torch.Tensor  # `[B]`
    queries: SpatialQueryBatch  # `[B,G^max,N_Q,...]`
    field_targets: FieldTargetBatch
    material_targets: MaterialPointRelationJacobianTarget
    material_point_index: torch.Tensor  # `[B,E^max]`
    edge_valid_mask: torch.Tensor  # `[B,E^max]`
    anchor_index: torch.Tensor  # `[B]`
    q_index: torch.Tensor  # `[B]`
    joint_coordinate_sign: torch.Tensor | None = None  # `[B,N_J^max]`


def _select_material_edges(
    state: GeometryRepresentationState,
    config: DensityMaterialJacobianMethodCfg,
    *,
    sampling_seed: int,
    supervision_split: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""按 joint-first 预算选择 owner、JOINT、home-point identity 与 active truth。

    同一 q-block 只采样一次 `[E]` identities，随后沿 Q 轴共享。跨 block seed 改变会覆盖 64-point
    home bank；有限差分与 replay 始终追踪同一 owner-local material identity。
    """

    sampling = config.material_sampling
    if supervision_split == "train":
        active_count = sampling.train_active_per_joint
        zero_count = sampling.train_zero_per_joint
    elif supervision_split == "eval":
        active_count = sampling.fixed_active_per_joint
        zero_count = sampling.fixed_zero_per_joint
    else:
        raise ValueError(f"unknown material supervision split={supervision_split!r}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(sampling_seed + sampling.seed_offset))
    ancestor = state.spec.owner_ancestor_mask.detach().cpu()  # `[G,N_J]`，离散 topology truth
    home_count = state.source.home_surface.points_owner_local_m.shape[1]  # 当前正式 source 为 64
    owners: list[int] = []
    joints: list[int] = []
    points: list[int] = []
    active_flags: list[bool] = []

    # 每个 JOINT 独立在 descendant/non-descendant owner 集合中采样；不足预算时确定性循环复用 owner，
    # 但 material point identity 仍由独立随机索引提供不同 surface measurement。
    for joint in range(state.spec.space_screws.shape[0]):
        active_owners = torch.where(ancestor[:, joint])[0]
        zero_owners = torch.where(~ancestor[:, joint])[0]
        if active_owners.numel() == 0 or zero_owners.numel() == 0:
            raise ValueError(f"joint {joint} lacks active or structural-zero owner support")
        active_order = active_owners[torch.randperm(active_owners.numel(), generator=generator)]
        zero_order = zero_owners[torch.randperm(zero_owners.numel(), generator=generator)]
        for is_active, count, candidates in (
            (True, active_count, active_order),
            (False, zero_count, zero_order),
        ):
            for edge_slot in range(count):
                owner = int(candidates[edge_slot % candidates.numel()])
                for _point_slot in range(sampling.points_per_edge):
                    owners.append(owner)
                    joints.append(joint)
                    points.append(int(torch.randint(home_count, (1,), generator=generator)))
                    active_flags.append(is_active)
    device = state.spec.space_screws.device
    return (
        torch.tensor(owners, device=device, dtype=torch.long),
        torch.tensor(joints, device=device, dtype=torch.long),
        torch.tensor(points, device=device, dtype=torch.long),
        torch.tensor(active_flags, device=device, dtype=torch.bool),
    )


def sample_density_gamma_block(
    state: GeometryRepresentationState,
    q: torch.Tensor,
    config: DensityMaterialJacobianMethodCfg,
    *,
    sampling_seed: int,
    q_index: torch.Tensor | None,
    anchor_index: int,
    supervision_split: str,
) -> DensityGammaOnlineSample:
    r"""一次 FK 后联合生成 density-only field 与 fixed-material Gamma truth。"""

    if q.ndim != 2 or q.shape[1] != state.spec.space_screws.shape[0]:
        raise ValueError("density/Gamma sample requires q shape [Q,N_J]")
    realization = state.source.anchor_realization
    if realization is None or realization.bank_index != int(anchor_index):
        raise ValueError("resident source anchor realization does not match requested block bank")
    anchors_h = torch.as_tensor(
        realization.samples.anchors_hand_m,
        device=q.device,
        dtype=q.dtype,
    )  # `[K,3]`，encoder/density/Gamma 共用同一 $A^{(k)}$
    owner_transforms, current_screws = forward_owner_transforms_and_spatial_screws(
        state.spec,
        q.detach(),
    )  # 当前 q-block 只计算一次 POE/current screws
    queries = sample_spatial_queries(
        q,
        state.spec,
        state.surface_sampling,
        anchors_h,
        config=config.representation.query,
        sampling_seed=sampling_seed,
        owner_transforms=owner_transforms,
    )
    field_targets = generate_density_field_targets(
        q,
        state.spec,
        state.source.geometry_cache,
        state.warp_cache,
        queries,
        field_config=config.representation.field,
        sampling_seed=sampling_seed,
        owner_transforms=owner_transforms,
    )
    owner_index, joint_index, point_index, sampled_active = _select_material_edges(
        state,
        config,
        sampling_seed=sampling_seed,
        supervision_split=supervision_split,
    )
    local_home = torch.as_tensor(
        state.source.home_surface.points_owner_local_m,
        device=q.device,
        dtype=q.dtype,
    )  # `[G,M,3]` owner-local fixed material bank
    local_points = local_home[owner_index, point_index]  # `[E,3]`，同一 q-block 共享 identity
    material_targets = generate_material_point_relation_jacobian_targets(
        state.spec,
        q,
        owner_index,
        joint_index,
        local_points,
        anchors_h,
        torch.tensor((0.0, 0.0, 1.0), device=q.device, dtype=q.dtype),
        config.material_target,
        owner_transforms=owner_transforms,
        current_spatial_screws=current_screws,
    )
    if not torch.equal(material_targets.ancestor_mask[0], sampled_active):
        raise RuntimeError("sampled active flags disagree with kinematic ancestor truth")
    q_count, edge_count = material_targets.owner_index.shape
    point_index_batch = point_index.unsqueeze(0).expand(q_count, edge_count)  # `[Q,E]` identity replay contract
    edge_valid = torch.ones(q_count, edge_count, device=q.device, dtype=torch.bool)
    semantics = state.source.container.geometry_semantics
    if semantics is None:
        raise ValueError("geometry source lost typed semantics")
    evidence = build_static_geometry_evidence(
        semantics,
        state.spec,
        state.source.home_surface,
        realization.samples,
        device=q.device,
        dtype=q.dtype,
    )
    return DensityGammaOnlineSample(
        asset_id=state.source.asset_id,
        q=q,
        evidence=evidence,
        queries=queries,
        field_targets=field_targets,
        material_targets=material_targets,
        material_point_index=point_index_batch,
        edge_valid_mask=edge_valid,
        anchor_index=int(anchor_index),
        q_index=q_index.detach().cpu() if q_index is not None else None,
    )


def pad_density_gamma_blocks(
    blocks: list[DensityGammaOnlineSample],
    *,
    padding: GeometryPaddingCfg,
) -> PaddedDensityGammaBatch:
    r"""把多资产 q-block pad 到统一 JOINT/owner/edge/anchor 轴。"""

    if not blocks:
        raise ValueError("density/Gamma padding requires at least one asset block")
    device = blocks[0].q.device
    dtype = blocks[0].q.dtype
    q_counts = [block.q.shape[0] for block in blocks]
    batch_size = sum(q_counts)
    max_owner = padding.max_owner_count
    max_joint = padding.max_joint_count
    query_count = blocks[0].queries.query_points_h.shape[2]
    sigma_count = blocks[0].field_targets.density.shape[-1]
    max_edge = max(block.material_targets.owner_index.shape[1] for block in blocks)
    evidence_keys: dict[tuple[str, int], int] = {}
    unique_evidence: list[StaticGeometryEvidence] = []
    evidence_rows: list[int] = []
    asset_ids: list[str] = []
    for block, q_count in zip(blocks, q_counts, strict=True):
        key = (block.asset_id, block.anchor_index)
        row = evidence_keys.get(key)
        if row is None:
            row = len(unique_evidence)
            evidence_keys[key] = row
            unique_evidence.append(block.evidence)
        evidence_rows.extend([row] * q_count)
        asset_ids.extend([block.asset_id] * q_count)
    evidence = pad_static_geometry_evidence(unique_evidence, config=padding)
    max_anchor = evidence.anchors.shape[1]  # padded $K^{max}$ 与 model reader 输出一致

    # Zero-initialized dense storage；所有物理有效位置随后按 source-local 前缀写入。
    q = torch.zeros(batch_size, max_joint, device=device, dtype=dtype)
    query_points = torch.zeros(batch_size, max_owner, query_count, 3, device=device, dtype=dtype)
    query_stratum = torch.zeros(batch_size, max_owner, query_count, device=device, dtype=torch.long)
    adjacent_owner = torch.full_like(query_stratum, -1)
    workspace_anchor = torch.full_like(query_stratum, -1)
    distance = torch.zeros(batch_size, max_owner, query_count, device=device, dtype=dtype)
    density = torch.zeros(batch_size, max_owner, query_count, sigma_count, device=device, dtype=dtype)
    field_valid = torch.zeros(batch_size, max_owner, query_count, device=device, dtype=torch.bool)
    owner_role = torch.zeros(batch_size, max_owner, device=device, dtype=torch.long)
    bandwidths = torch.zeros(batch_size, sigma_count, device=device, dtype=dtype)
    owner_index = torch.zeros(batch_size, max_edge, device=device, dtype=torch.long)
    joint_index = torch.zeros_like(owner_index)
    material_point_index = torch.zeros_like(owner_index)
    ancestor_mask = torch.zeros(batch_size, max_edge, device=device, dtype=torch.bool)
    edge_valid = torch.zeros_like(ancestor_mask)
    material_points = torch.zeros(batch_size, max_edge, 3, device=device, dtype=dtype)
    point_jacobian = torch.zeros_like(material_points)
    gamma_distance = torch.zeros(batch_size, max_edge, max_anchor, device=device, dtype=dtype)
    gamma_distance_sensitivity = torch.zeros_like(gamma_distance)
    gamma_relation = torch.zeros(batch_size, max_edge, max_anchor, 4, device=device, dtype=dtype)
    gamma_sensitivity = torch.zeros_like(gamma_relation)
    distance_valid = torch.zeros(batch_size, max_edge, max_anchor, device=device, dtype=torch.bool)
    radius_valid = torch.zeros_like(distance_valid)
    anchor_index = torch.zeros(batch_size, device=device, dtype=torch.long)
    q_index = torch.full((batch_size,), -1, device=device, dtype=torch.long)

    start = 0
    for block, q_count in zip(blocks, q_counts, strict=True):
        rows = slice(start, start + q_count)
        joint_count = block.q.shape[1]
        owner_count = block.queries.query_points_h.shape[1]
        edge_count = block.material_targets.owner_index.shape[1]
        anchor_count = block.material_targets.distance_m.shape[2]
        q[rows, :joint_count] = block.q
        query_points[rows, :owner_count] = block.queries.query_points_h
        query_stratum[rows, :owner_count] = block.queries.query_stratum
        adjacent_owner[rows, :owner_count] = block.queries.adjacent_owner_index
        workspace_anchor[rows, :owner_count] = block.queries.workspace_anchor_index
        distance[rows, :owner_count] = block.field_targets.distance
        density[rows, :owner_count] = block.field_targets.density
        field_valid[rows, :owner_count] = block.field_targets.valid_mask
        role = block.field_targets.owner_role
        owner_role[rows, :owner_count] = role.unsqueeze(0).expand(q_count, -1) if role.ndim == 1 else role
        bandwidths[rows] = block.field_targets.bandwidths
        owner_index[rows, :edge_count] = block.material_targets.owner_index
        joint_index[rows, :edge_count] = block.material_targets.joint_index
        material_point_index[rows, :edge_count] = block.material_point_index
        ancestor_mask[rows, :edge_count] = block.material_targets.ancestor_mask
        edge_valid[rows, :edge_count] = block.edge_valid_mask
        material_points[rows, :edge_count] = block.material_targets.material_points_h_m
        point_jacobian[rows, :edge_count] = block.material_targets.point_jacobian_h_m_per_rad
        gamma_distance[rows, :edge_count, :anchor_count] = block.material_targets.distance_m
        gamma_distance_sensitivity[rows, :edge_count, :anchor_count] = (
            block.material_targets.distance_sensitivity_m_per_rad
        )
        gamma_relation[rows, :edge_count, :anchor_count] = block.material_targets.relation_values
        gamma_sensitivity[rows, :edge_count, :anchor_count] = block.material_targets.relation_sensitivity_per_rad
        distance_valid[rows, :edge_count, :anchor_count] = block.material_targets.distance_valid_mask
        radius_valid[rows, :edge_count, :anchor_count] = block.material_targets.radius_valid_mask
        anchor_index[rows] = block.anchor_index
        if block.q_index is not None:
            q_index[rows] = block.q_index.to(device=device)
        start += q_count

    queries = SpatialQueryBatch(query_points, query_stratum, adjacent_owner, workspace_anchor)
    field = FieldTargetBatch(
        query_points=query_points,
        query_stratum=query_stratum,
        distance=distance,
        density=density,
        valid_mask=field_valid,
        owner_role=owner_role,
        bandwidths=bandwidths,
        provenance={"frame": "h", "length_unit": "m", "backend": "density_only_padded"},
    )
    material = MaterialPointRelationJacobianTarget(
        distance_m=gamma_distance,
        distance_sensitivity_m_per_rad=gamma_distance_sensitivity,
        relation_values=gamma_relation,
        relation_sensitivity_per_rad=gamma_sensitivity,
        distance_valid_mask=distance_valid,
        radius_valid_mask=radius_valid,
        material_points_h_m=material_points,
        point_jacobian_h_m_per_rad=point_jacobian,
        owner_index=owner_index,
        joint_index=joint_index,
        ancestor_mask=ancestor_mask,
        provenance={
            "frame": "h",
            "distance_unit": "m",
            "joint_unit": "rad",
            "relation_unit": "dimensionless",
            "relation_sensitivity_unit": "rad^-1",
            "relation_channels": "height,radius,dot,chirality",
            "material_identity": "fixed_owner_local_home_surface_point",
            "anchor_motion": "fixed_palm_support",
        },
    )
    return PaddedDensityGammaBatch(
        asset_ids=tuple(asset_ids),
        q=q,
        evidence=evidence,
        evidence_row_index=torch.tensor(evidence_rows, device=device, dtype=torch.long),
        queries=queries,
        field_targets=field,
        material_targets=material,
        material_point_index=material_point_index,
        edge_valid_mask=edge_valid,
        anchor_index=anchor_index,
        q_index=q_index,
    )


def _map_dataclass(value: _DataclassT, transform: Any) -> _DataclassT:
    r"""保持 concrete dataclass，只迁移其直接 tensor fields。"""

    updates = {
        field_info.name: transform(field_value) if isinstance(field_value := getattr(value, field_info.name), torch.Tensor) else field_value
        for field_info in fields(cast(Any, value))
    }
    return cast(_DataclassT, replace(cast(Any, value), **updates))


def map_padded_batch(batch: PaddedDensityGammaBatch, transform: Any) -> PaddedDensityGammaBatch:
    r"""统一迁移 model inputs、density truth 与 Gamma truth。"""

    return PaddedDensityGammaBatch(
        asset_ids=batch.asset_ids,
        q=transform(batch.q),
        evidence=_map_dataclass(batch.evidence, transform),
        evidence_row_index=transform(batch.evidence_row_index),
        queries=_map_dataclass(batch.queries, transform),
        field_targets=_map_dataclass(batch.field_targets, transform),
        material_targets=_map_dataclass(batch.material_targets, transform),
        material_point_index=transform(batch.material_point_index),
        edge_valid_mask=transform(batch.edge_valid_mask),
        anchor_index=transform(batch.anchor_index),
        q_index=transform(batch.q_index),
        joint_coordinate_sign=None if batch.joint_coordinate_sign is None else transform(batch.joint_coordinate_sign),
    )


def stage_padded_batch_for_replay(batch: PaddedDensityGammaBatch) -> PaddedDensityGammaBatch:
    r"""把 detached teacher batch 放入 pinned CPU，供有限 mini-epoch replay。"""

    def stage(tensor: torch.Tensor) -> torch.Tensor:
        source = tensor.detach()
        target = torch.empty_like(source, device="cpu", pin_memory=torch.cuda.is_available())
        target.copy_(source, non_blocking=False)
        return target

    return map_padded_batch(batch, stage)


def restore_padded_batch_from_replay(
    batch: PaddedDensityGammaBatch,
    *,
    device: torch.device | str,
) -> PaddedDensityGammaBatch:
    r"""把当前 opaque replay unit 恢复到训练 device。"""

    target = torch.device(device)
    return map_padded_batch(batch, lambda tensor: tensor.to(device=target, non_blocking=tensor.is_pinned()))


__all__ = [
    "DensityGammaOnlineSample",
    "PaddedDensityGammaBatch",
    "pad_density_gamma_blocks",
    "restore_padded_batch_from_replay",
    "sample_density_gamma_block",
    "stage_padded_batch_for_replay",
]
