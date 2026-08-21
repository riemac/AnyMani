r"""方法专属 batch 适配：选 $A^{(k)}$、构造 retained evidence、跨结构 padding。

representation 只交付物理 teacher：query、$d/\rho/\kappa/g$ 与有效性。本模块再把当前锚点
realization 编成 encoder 输入，并把异构 $N_J/G/E$ 填进稠密容器。一次 method batch 在逻辑上
分成三块，模型不得读取 truth：

- `model_input`：$q$、anchors、home、screws、graph、masks；
- `readout_condition`：query、sigma、edge selectors；
- `truth`：distance/density/$\kappa/g$、物理有效、active/zero 类别与 provenance。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from anymani.distill.models.input_adapters.geometry import (
    GeometryPaddingCfg,
    StaticGeometryEvidence,
    build_static_geometry_evidence,
    pad_static_geometry_evidence,
)
from anymani.distill.representations.geometry import PhysicalOnlineGeometrySample
from anymani.distill.representations.queries.spatial_sampling import SpatialQueryBatch
from anymani.distill.representations.sources.collision_geometry import AnchorSamples
from anymani.distill.representations.sources.geometry_source import GeometrySource
from anymani.distill.representations.sources.kinematics import EmbodimentGeometrySpec
from anymani.distill.representations.targets.field_samples import FieldTargetBatch, SensitivityTargetBatch


@dataclass(frozen=True)
class OnlineGeometrySample:
    r"""物理 teacher 加上当前 $A^{(k)}$ 的 retained encoder 输入。"""

    asset_id: str
    q: torch.Tensor  # `[1,N_J]`，rad
    evidence: StaticGeometryEvidence  # 单资产，无 batch 静态轴
    queries: SpatialQueryBatch  # `[1,G,N_Q,...]`
    field_targets: FieldTargetBatch
    sensitivity_targets: SensitivityTargetBatch
    q_index: torch.Tensor | None = None  # `[1]`


@dataclass(frozen=True)
class PaddedOnlineGeometryBatch:
    r"""异构结构可共同进入一次模型前向的稠密 batch。

    padding 只改变存储：真实轴写入前缀，其余槽由 entity/joint/field/edge mask 屏蔽。
    """

    asset_ids: tuple[str, ...]
    q: torch.Tensor  # `[B,N_J^{max}]`，padding JOINT 为 0
    evidence: StaticGeometryEvidence  # `[B,G^{max},...]` + masks
    queries: SpatialQueryBatch  # `[B,G^{max},N_Q,...]`
    field_targets: FieldTargetBatch
    sensitivity_targets: SensitivityTargetBatch
    q_index: torch.Tensor | None = None  # `[B]`


@dataclass(frozen=True)
class MethodBatchViews:
    r"""把一份 padded batch 拆成模型输入、读出条件与物理真值。"""

    model_input: tuple[torch.Tensor, StaticGeometryEvidence]  # $(q,\mathrm{evidence})$
    readout_condition: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    truth: tuple[FieldTargetBatch, SensitivityTargetBatch]


def attach_static_evidence(
    sample: PhysicalOnlineGeometrySample,
    *,
    source: GeometrySource,
    spec: EmbodimentGeometrySpec,
    anchors: AnchorSamples,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[OnlineGeometrySample, ...]:
    r"""为同资产 $q$-block 构造一次 $A^{(k)}$ evidence，再切成 padding 所需的 $[1,N_J]$ 样本。"""

    semantics = source.container.geometry_semantics
    if semantics is None:
        raise ValueError("geometry source lost its typed semantics")
    evidence = build_static_geometry_evidence(
        semantics,
        spec,
        source.home_surface,
        anchors,
        device=device,
        dtype=dtype,
    )
    pieces = split_physical_online_geometry_sample(sample)
    return tuple(
        OnlineGeometrySample(
            asset_id=piece.asset_id,
            q=piece.q,
            evidence=evidence,
            queries=piece.queries,
            field_targets=piece.field_targets,
            sensitivity_targets=piece.sensitivity_targets,
            q_index=piece.q_index,
        )
        for piece in pieces
    )


def split_online_geometry_sample(sample: OnlineGeometrySample) -> tuple[OnlineGeometrySample, ...]:
    r"""把一次已附着 evidence 的 `[Q,N_J]` 样本展开为 `[1,N_J]`。"""

    q_count = sample.q.shape[0]
    return tuple(
        OnlineGeometrySample(
            asset_id=sample.asset_id,
            q=sample.q[index : index + 1],
            evidence=sample.evidence,
            queries=SpatialQueryBatch(
                sample.queries.query_points_h[index : index + 1],
                sample.queries.query_stratum[index : index + 1],
                sample.queries.adjacent_owner_index[index : index + 1],
                sample.queries.workspace_anchor_index[index : index + 1],
            ),
            field_targets=FieldTargetBatch(
                query_points=sample.field_targets.query_points[index : index + 1],
                query_stratum=sample.field_targets.query_stratum[index : index + 1],
                distance=sample.field_targets.distance[index : index + 1],
                density=sample.field_targets.density[index : index + 1],
                valid_mask=sample.field_targets.valid_mask[index : index + 1],
                owner_role=sample.field_targets.owner_role,
                bandwidths=(
                    sample.field_targets.bandwidths
                    if sample.field_targets.bandwidths.ndim == 1
                    else sample.field_targets.bandwidths[index : index + 1]
                ),
                provenance=sample.field_targets.provenance,
            ),
            sensitivity_targets=SensitivityTargetBatch(
                owner_index=sample.sensitivity_targets.owner_index,
                query_index=sample.sensitivity_targets.query_index,
                joint_index=sample.sensitivity_targets.joint_index,
                ancestor_mask=sample.sensitivity_targets.ancestor_mask,
                active_mask=sample.sensitivity_targets.active_mask,
                closest_point=sample.sensitivity_targets.closest_point[index : index + 1],
                closest_source=sample.sensitivity_targets.closest_source[index : index + 1],
                uniqueness_margin=sample.sensitivity_targets.uniqueness_margin[index : index + 1],
                kappa=sample.sensitivity_targets.kappa[index : index + 1],
                field_sensitivity=sample.sensitivity_targets.field_sensitivity[index : index + 1],
                valid_mask=sample.sensitivity_targets.valid_mask[index : index + 1],
                provenance=sample.sensitivity_targets.provenance,
            ),
            q_index=sample.q_index[index : index + 1] if sample.q_index is not None else None,
        )
        for index in range(q_count)
    )


def split_physical_online_geometry_sample(
    sample: PhysicalOnlineGeometrySample,
) -> tuple[PhysicalOnlineGeometrySample, ...]:
    r"""把一次同资产 `[Q,N_J]` teacher block 展开为 `[1,N_J]` 物理样本。"""

    q_count = sample.q.shape[0]
    return tuple(
        PhysicalOnlineGeometrySample(
            asset_id=sample.asset_id,
            q=sample.q[index : index + 1],
            queries=SpatialQueryBatch(
                sample.queries.query_points_h[index : index + 1],
                sample.queries.query_stratum[index : index + 1],
                sample.queries.adjacent_owner_index[index : index + 1],
                sample.queries.workspace_anchor_index[index : index + 1],
            ),
            field_targets=FieldTargetBatch(
                query_points=sample.field_targets.query_points[index : index + 1],
                query_stratum=sample.field_targets.query_stratum[index : index + 1],
                distance=sample.field_targets.distance[index : index + 1],
                density=sample.field_targets.density[index : index + 1],
                valid_mask=sample.field_targets.valid_mask[index : index + 1],
                owner_role=sample.field_targets.owner_role,
                bandwidths=(
                    sample.field_targets.bandwidths
                    if sample.field_targets.bandwidths.ndim == 1
                    else sample.field_targets.bandwidths[index : index + 1]
                ),
                provenance=sample.field_targets.provenance,
            ),
            sensitivity_targets=SensitivityTargetBatch(
                owner_index=sample.sensitivity_targets.owner_index,
                query_index=sample.sensitivity_targets.query_index,
                joint_index=sample.sensitivity_targets.joint_index,
                ancestor_mask=sample.sensitivity_targets.ancestor_mask,
                active_mask=sample.sensitivity_targets.active_mask,
                closest_point=sample.sensitivity_targets.closest_point[index : index + 1],
                closest_source=sample.sensitivity_targets.closest_source[index : index + 1],
                uniqueness_margin=sample.sensitivity_targets.uniqueness_margin[index : index + 1],
                kappa=sample.sensitivity_targets.kappa[index : index + 1],
                field_sensitivity=sample.sensitivity_targets.field_sensitivity[index : index + 1],
                valid_mask=sample.sensitivity_targets.valid_mask[index : index + 1],
                provenance=sample.sensitivity_targets.provenance,
            ),
            anchor_index=sample.anchor_index,
            q_index=sample.q_index[index : index + 1] if sample.q_index is not None else None,
        )
        for index in range(q_count)
    )


def pad_online_geometry_samples(
    samples: list[OnlineGeometrySample],
    *,
    padding: GeometryPaddingCfg,
) -> PaddedOnlineGeometryBatch:
    r"""把不同 $N_J/G/E$ 的在线样本填充为统一训练 batch。

    每个样本的真实轴写入前缀；selector padding 使用合法 0 索引，但 `edge_valid=False`。
    """

    if not samples:
        raise ValueError("at least one OnlineGeometrySample is required")
    device = samples[0].q.device
    dtype = samples[0].q.dtype
    query_count = samples[0].queries.query_points_h.shape[2]
    bandwidth_count = samples[0].field_targets.bandwidths.shape[-1]
    if any(sample.q.device != device or sample.q.dtype != dtype for sample in samples):
        raise ValueError("all online samples must share device and dtype")
    if any(sample.queries.query_points_h.shape[2] != query_count for sample in samples):
        raise ValueError("all samples must share N_Q")
    if any(sample.field_targets.bandwidths.shape[-1] != bandwidth_count for sample in samples):
        raise ValueError("all samples in one dense batch must share N_sigma")

    batch_size = len(samples)
    max_owner_count = padding.max_owner_count
    max_joint_count = padding.max_joint_count
    max_edge_count = max(sample.sensitivity_targets.kappa.shape[1] for sample in samples)
    evidence = pad_static_geometry_evidence([sample.evidence for sample in samples], config=padding)
    q = torch.zeros(batch_size, max_joint_count, device=device, dtype=dtype)
    query_points = torch.zeros(batch_size, max_owner_count, query_count, 3, device=device, dtype=dtype)
    query_stratum = torch.zeros(batch_size, max_owner_count, query_count, device=device, dtype=torch.long)
    adjacent_owner = torch.full_like(query_stratum, -1)
    workspace_anchor = torch.full_like(query_stratum, -1)
    bandwidths = torch.zeros(batch_size, bandwidth_count, device=device, dtype=dtype)
    distance = torch.zeros(batch_size, max_owner_count, query_count, device=device, dtype=dtype)
    density = torch.zeros(batch_size, max_owner_count, query_count, bandwidth_count, device=device, dtype=dtype)
    field_valid = torch.zeros(batch_size, max_owner_count, query_count, device=device, dtype=torch.bool)
    owner_role = torch.zeros(batch_size, max_owner_count, device=device, dtype=torch.long)
    owner_index = torch.zeros(batch_size, max_edge_count, device=device, dtype=torch.long)
    edge_query_index = torch.zeros_like(owner_index)
    joint_index = torch.zeros_like(owner_index)
    ancestor_mask = torch.zeros(batch_size, max_edge_count, device=device, dtype=torch.bool)
    active_mask = torch.zeros(batch_size, max_edge_count, device=device, dtype=torch.bool)
    closest_point = torch.zeros(batch_size, max_edge_count, 3, device=device, dtype=dtype)
    closest_source = torch.zeros(batch_size, max_edge_count, device=device, dtype=torch.long)
    uniqueness_margin = torch.zeros(batch_size, max_edge_count, device=device, dtype=dtype)
    kappa = torch.zeros(batch_size, max_edge_count, device=device, dtype=dtype)
    field_sensitivity = torch.zeros(batch_size, max_edge_count, bandwidth_count, device=device, dtype=dtype)
    edge_valid = torch.zeros(batch_size, max_edge_count, device=device, dtype=torch.bool)
    q_index = torch.full((batch_size,), -1, device=device, dtype=torch.long)

    for batch_index, sample in enumerate(samples):
        joint_count = sample.q.shape[1]
        owner_count = sample.queries.query_points_h.shape[1]
        edge_count = sample.sensitivity_targets.kappa.shape[1]
        q[batch_index, :joint_count] = sample.q[0]
        query_points[batch_index, :owner_count] = sample.queries.query_points_h[0]
        query_stratum[batch_index, :owner_count] = sample.queries.query_stratum[0]
        adjacent_owner[batch_index, :owner_count] = sample.queries.adjacent_owner_index[0]
        workspace_anchor[batch_index, :owner_count] = sample.queries.workspace_anchor_index[0]
        sample_bandwidths = sample.field_targets.bandwidths
        bandwidths[batch_index] = sample_bandwidths if sample_bandwidths.ndim == 1 else sample_bandwidths[0]
        distance[batch_index, :owner_count] = sample.field_targets.distance[0]
        density[batch_index, :owner_count] = sample.field_targets.density[0]
        field_valid[batch_index, :owner_count] = sample.field_targets.valid_mask[0]
        role = sample.field_targets.owner_role
        owner_role[batch_index, :owner_count] = role if role.ndim == 1 else role[0]
        sensitivity = sample.sensitivity_targets
        owner_index[batch_index, :edge_count] = sensitivity.owner_index
        edge_query_index[batch_index, :edge_count] = sensitivity.query_index
        joint_index[batch_index, :edge_count] = sensitivity.joint_index
        ancestor_mask[batch_index, :edge_count] = sensitivity.ancestor_mask
        active_mask[batch_index, :edge_count] = sensitivity.active_mask
        closest_point[batch_index, :edge_count] = sensitivity.closest_point[0]
        closest_source[batch_index, :edge_count] = sensitivity.closest_source[0]
        uniqueness_margin[batch_index, :edge_count] = sensitivity.uniqueness_margin[0]
        kappa[batch_index, :edge_count] = sensitivity.kappa[0]
        field_sensitivity[batch_index, :edge_count] = sensitivity.field_sensitivity[0]
        edge_valid[batch_index, :edge_count] = sensitivity.valid_mask[0]
        if sample.q_index is not None:
            if sample.q_index.numel() != 1:
                raise ValueError("pad_online_geometry_samples expects split samples with one q_index")
            q_index[batch_index] = sample.q_index.reshape(-1)[0].to(device=device)

    queries = SpatialQueryBatch(query_points, query_stratum, adjacent_owner, workspace_anchor)
    field_targets = FieldTargetBatch(
        query_points=query_points,
        query_stratum=query_stratum,
        distance=distance,
        density=density,
        valid_mask=field_valid,
        owner_role=owner_role,
        bandwidths=bandwidths,
        provenance={
            "frame": "h",
            "length_unit": "m",
            "backend": "warp_mesh_query_point",
            "padding": f"joint={max_joint_count},owner={max_owner_count}",
        },
    )
    sensitivity_targets = SensitivityTargetBatch(
        owner_index=owner_index,
        query_index=edge_query_index,
        joint_index=joint_index,
        ancestor_mask=ancestor_mask,
        active_mask=active_mask,
        closest_point=closest_point,
        closest_source=closest_source,
        uniqueness_margin=uniqueness_margin,
        kappa=kappa,
        field_sensitivity=field_sensitivity,
        valid_mask=edge_valid,
        provenance={
            "frame": "h",
            "distance_unit": "m",
            "joint_unit": "rad",
            "padding": f"edge={max_edge_count}",
        },
    )
    return PaddedOnlineGeometryBatch(
        asset_ids=tuple(sample.asset_id for sample in samples),
        q=q,
        evidence=evidence,
        queries=queries,
        field_targets=field_targets,
        sensitivity_targets=sensitivity_targets,
        q_index=q_index,
    )


def method_batch_views(batch: PaddedOnlineGeometryBatch) -> MethodBatchViews:
    r"""按模型/读出/真值三块切开；模型路径不得消费 `truth`。"""

    targets = batch.sensitivity_targets
    return MethodBatchViews(
        model_input=(batch.q, batch.evidence),
        readout_condition=(
            batch.queries.query_points_h,
            batch.field_targets.bandwidths,
            targets.owner_index,
            targets.query_index,
            targets.joint_index,
        ),
        truth=(batch.field_targets, batch.sensitivity_targets),
    )


__all__ = [
    "GeometryPaddingCfg",
    "MethodBatchViews",
    "OnlineGeometrySample",
    "PaddedOnlineGeometryBatch",
    "PhysicalOnlineGeometrySample",
    "attach_static_evidence",
    "method_batch_views",
    "pad_online_geometry_samples",
    "split_online_geometry_sample",
    "split_physical_online_geometry_sample",
]
