r"""方法专属 batch 适配：选 $A^{(k)}$、构造 retained evidence、跨结构 padding。

representation 只交付物理 teacher：query、$d/\rho/\kappa/g$ 与有效性。本模块再把当前锚点
realization 编成 encoder 输入，并把异构 $N_J/G/E$ 填进稠密容器。一次 method batch 在逻辑上
分成三块，模型不得读取 truth：

- `model_input`：$q$、anchors、home、screws、graph、masks；
- `readout_condition`：query、sigma、edge selectors；
- `truth`：distance/density/$\kappa/g$、物理有效、active/zero 类别与 provenance。
"""

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
from anymani.distill.representations.geometry import PhysicalOnlineGeometrySample
from anymani.distill.representations.queries.spatial_sampling import SpatialQueryBatch
from anymani.distill.representations.sources.anchor_sampling import AnchorSamples
from anymani.distill.representations.sources.geometry_source import GeometrySource
from anymani.distill.representations.sources.kinematics import EmbodimentGeometrySpec
from anymani.distill.representations.targets.field_samples import FieldTargetBatch, SensitivityTargetBatch

_DataclassT = TypeVar("_DataclassT")


@dataclass(frozen=True)
class OnlineGeometrySample:
    r"""物理 teacher 加上当前 $A^{(k)}$ 的 retained encoder 输入。"""

    asset_id: str
    q: torch.Tensor  # `[1,N_J]`，rad
    evidence: StaticGeometryEvidence  # 单资产，无 batch 静态轴
    queries: SpatialQueryBatch  # `[1,G,N_Q,...]`
    field_targets: FieldTargetBatch
    sensitivity_targets: SensitivityTargetBatch
    anchor_index: int = 0  # 当前 evidence 对应的 $A^{(k)}$
    q_index: torch.Tensor | None = None  # `[1]`


@dataclass(frozen=True)
class PaddedOnlineGeometryBatch:
    r"""异构结构可共同进入一次模型前向的稠密 batch。

    padding 只改变存储：真实轴写入前缀，其余槽由 entity/joint/field/edge mask 屏蔽。
    """

    asset_ids: tuple[str, ...]
    q: torch.Tensor  # `[B,N_J^{max}]`，padding JOINT 为 0
    evidence: StaticGeometryEvidence  # `[A_unique,G^{max},...]` + masks
    queries: SpatialQueryBatch  # `[B,G^{max},N_Q,...]`
    field_targets: FieldTargetBatch
    sensitivity_targets: SensitivityTargetBatch
    evidence_row_index: torch.Tensor | None = None  # `[B]`，q 行 -> unique static evidence 行
    anchor_index: torch.Tensor | None = None  # `[B]`，每行 q 使用的 anchor realization
    q_index: torch.Tensor | None = None  # `[B]`


@dataclass(frozen=True)
class MethodBatchViews:
    r"""把一份 padded batch 拆成模型输入、读出条件与物理真值。"""

    model_input: tuple[torch.Tensor, StaticGeometryEvidence, torch.Tensor | None]
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
    entity_permutation: torch.Tensor | None = None,
) -> tuple[OnlineGeometrySample, ...]:
    r"""为同资产 $q$-block 构造一次 $A^{(k)}$ evidence，再切成 padding 所需的 $[1,N_J]$ 样本。"""

    return split_online_geometry_sample(
        attach_static_evidence_block(
            sample,
            source=source,
            spec=spec,
            anchors=anchors,
            device=device,
            dtype=dtype,
            entity_permutation=entity_permutation,
        )
    )


def attach_static_evidence_block(
    sample: PhysicalOnlineGeometrySample,
    *,
    source: GeometrySource,
    spec: EmbodimentGeometrySpec,
    anchors: AnchorSamples,
    device: torch.device | str,
    dtype: torch.dtype,
    entity_permutation: torch.Tensor | None = None,
) -> OnlineGeometrySample:
    r"""为完整同资产 q-block 附着一次静态 evidence，不制造逐 q Python 对象。"""

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
    block = OnlineGeometrySample(
        asset_id=sample.asset_id,
        q=sample.q,
        evidence=evidence,
        queries=sample.queries,
        field_targets=sample.field_targets,
        sensitivity_targets=sample.sensitivity_targets,
        anchor_index=sample.anchor_index,
        q_index=sample.q_index,
    )
    if entity_permutation is not None:
        from .augmentation import permute_online_geometry_sample

        block = permute_online_geometry_sample(block, entity_permutation)
    return block


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
                owner_index=_slice_selector(sample.sensitivity_targets.owner_index, index),
                query_index=_slice_selector(sample.sensitivity_targets.query_index, index),
                joint_index=_slice_selector(sample.sensitivity_targets.joint_index, index),
                ancestor_mask=_slice_selector(sample.sensitivity_targets.ancestor_mask, index),
                active_mask=_slice_selector(sample.sensitivity_targets.active_mask, index),
                closest_point=sample.sensitivity_targets.closest_point[index : index + 1],
                closest_source=sample.sensitivity_targets.closest_source[index : index + 1],
                uniqueness_margin=sample.sensitivity_targets.uniqueness_margin[index : index + 1],
                kappa=sample.sensitivity_targets.kappa[index : index + 1],
                field_sensitivity=sample.sensitivity_targets.field_sensitivity[index : index + 1],
                valid_mask=sample.sensitivity_targets.valid_mask[index : index + 1],
                owner_category=_slice_optional_selector(sample.sensitivity_targets.owner_category, index),
                query_stratum=_slice_optional_selector(sample.sensitivity_targets.query_stratum, index),
                fallback_category=_slice_optional_selector(sample.sensitivity_targets.fallback_category, index),
                sampling_role=_slice_optional_selector(sample.sensitivity_targets.sampling_role, index),
                central_difference=(
                    sample.sensitivity_targets.central_difference[index : index + 1]
                    if sample.sensitivity_targets.central_difference is not None
                    else None
                ),
                central_difference_valid_mask=(
                    sample.sensitivity_targets.central_difference_valid_mask[index : index + 1]
                    if sample.sensitivity_targets.central_difference_valid_mask is not None
                    else None
                ),
                central_difference_plus_face=(
                    sample.sensitivity_targets.central_difference_plus_face[index : index + 1]
                    if sample.sensitivity_targets.central_difference_plus_face is not None
                    else None
                ),
                central_difference_minus_face=(
                    sample.sensitivity_targets.central_difference_minus_face[index : index + 1]
                    if sample.sensitivity_targets.central_difference_minus_face is not None
                    else None
                ),
                central_difference_elapsed_seconds=(
                    sample.sensitivity_targets.central_difference_elapsed_seconds if index == 0 else 0.0
                ),
                provenance=sample.sensitivity_targets.provenance,
            ),
            anchor_index=sample.anchor_index,
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
                owner_index=_slice_selector(sample.sensitivity_targets.owner_index, index),
                query_index=_slice_selector(sample.sensitivity_targets.query_index, index),
                joint_index=_slice_selector(sample.sensitivity_targets.joint_index, index),
                ancestor_mask=_slice_selector(sample.sensitivity_targets.ancestor_mask, index),
                active_mask=_slice_selector(sample.sensitivity_targets.active_mask, index),
                closest_point=sample.sensitivity_targets.closest_point[index : index + 1],
                closest_source=sample.sensitivity_targets.closest_source[index : index + 1],
                uniqueness_margin=sample.sensitivity_targets.uniqueness_margin[index : index + 1],
                kappa=sample.sensitivity_targets.kappa[index : index + 1],
                field_sensitivity=sample.sensitivity_targets.field_sensitivity[index : index + 1],
                valid_mask=sample.sensitivity_targets.valid_mask[index : index + 1],
                owner_category=_slice_optional_selector(sample.sensitivity_targets.owner_category, index),
                query_stratum=_slice_optional_selector(sample.sensitivity_targets.query_stratum, index),
                fallback_category=_slice_optional_selector(sample.sensitivity_targets.fallback_category, index),
                sampling_role=_slice_optional_selector(sample.sensitivity_targets.sampling_role, index),
                central_difference=(
                    sample.sensitivity_targets.central_difference[index : index + 1]
                    if sample.sensitivity_targets.central_difference is not None
                    else None
                ),
                central_difference_valid_mask=(
                    sample.sensitivity_targets.central_difference_valid_mask[index : index + 1]
                    if sample.sensitivity_targets.central_difference_valid_mask is not None
                    else None
                ),
                central_difference_plus_face=(
                    sample.sensitivity_targets.central_difference_plus_face[index : index + 1]
                    if sample.sensitivity_targets.central_difference_plus_face is not None
                    else None
                ),
                central_difference_minus_face=(
                    sample.sensitivity_targets.central_difference_minus_face[index : index + 1]
                    if sample.sensitivity_targets.central_difference_minus_face is not None
                    else None
                ),
                central_difference_elapsed_seconds=(
                    sample.sensitivity_targets.central_difference_elapsed_seconds if index == 0 else 0.0
                ),
                provenance=sample.sensitivity_targets.provenance,
            ),
            anchor_index=sample.anchor_index,
            q_index=sample.q_index[index : index + 1] if sample.q_index is not None else None,
        )
        for index in range(q_count)
    )


def _slice_selector(selector: torch.Tensor, index: int) -> torch.Tensor:
    """把逐 q `[Q,E]` selector 切为 `[1,E]`；历史共享 `[E]` selector 保持原样。"""

    return selector[index : index + 1] if selector.ndim == 2 else selector


def _slice_optional_selector(selector: torch.Tensor | None, index: int) -> torch.Tensor | None:
    """对可选 edge provenance 复用 selector 的 q-row 切分语义。"""

    return None if selector is None else _slice_selector(selector, index)


def _selector_row(selector: torch.Tensor) -> torch.Tensor:
    """把单 q 样本的 `[1,E]` selector 还原为 padding 写入使用的 `[E]`。"""

    if selector.ndim == 1:
        return selector
    if selector.ndim != 2 or selector.shape[0] != 1:
        raise ValueError("split OnlineGeometrySample selector must have shape [E] or [1,E]")
    return selector[0]


def pad_online_geometry_samples(
    samples: list[OnlineGeometrySample],
    *,
    padding: GeometryPaddingCfg,
) -> PaddedOnlineGeometryBatch:
    r"""把不同 $N_J/G/E$ 的在线样本或同资产 q-block 填充为统一训练 batch。

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

    q_counts = [int(sample.q.shape[0]) for sample in samples]
    batch_size = sum(q_counts)
    max_owner_count = padding.max_owner_count
    max_joint_count = padding.max_joint_count
    max_edge_count = max(sample.sensitivity_targets.kappa.shape[1] for sample in samples)
    evidence_keys: dict[tuple[str, int], int] = {}
    unique_evidence: list[StaticGeometryEvidence] = []
    evidence_rows: list[int] = []
    expanded_asset_ids: list[str] = []
    for sample, q_count in zip(samples, q_counts):
        key = (sample.asset_id, int(sample.anchor_index))
        row = evidence_keys.get(key)
        if row is None:
            row = len(unique_evidence)
            evidence_keys[key] = row
            unique_evidence.append(sample.evidence)
        evidence_rows.extend([row] * q_count)
        expanded_asset_ids.extend([sample.asset_id] * q_count)
    evidence = pad_static_geometry_evidence(unique_evidence, config=padding)
    evidence_row_index = torch.tensor(evidence_rows, device=device, dtype=torch.long)
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
    edge_owner_category = torch.full((batch_size, max_edge_count), -1, device=device, dtype=torch.long)
    edge_query_stratum = torch.full_like(edge_owner_category, -1)
    edge_fallback_category = torch.full_like(edge_owner_category, -1)
    edge_sampling_role = torch.full_like(edge_owner_category, -1)
    central_difference = torch.zeros(batch_size, max_edge_count, device=device, dtype=dtype)
    central_difference_valid = torch.zeros(batch_size, max_edge_count, device=device, dtype=torch.bool)
    central_plus_face = torch.full((batch_size, max_edge_count), -1, device=device, dtype=torch.long)
    central_minus_face = torch.full_like(central_plus_face, -1)
    central_difference_elapsed_seconds = 0.0  # 对输入 asset blocks 求和；逐 q reference 只在首行携带耗时
    anchor_index = torch.zeros(batch_size, device=device, dtype=torch.long)
    q_index = torch.full((batch_size,), -1, device=device, dtype=torch.long)

    batch_start = 0
    for sample, q_count in zip(samples, q_counts):
        batch_slice = slice(batch_start, batch_start + q_count)
        joint_count = sample.q.shape[1]
        owner_count = sample.queries.query_points_h.shape[1]
        edge_count = sample.sensitivity_targets.kappa.shape[1]
        q[batch_slice, :joint_count] = sample.q
        query_points[batch_slice, :owner_count] = sample.queries.query_points_h
        query_stratum[batch_slice, :owner_count] = sample.queries.query_stratum
        adjacent_owner[batch_slice, :owner_count] = sample.queries.adjacent_owner_index
        workspace_anchor[batch_slice, :owner_count] = sample.queries.workspace_anchor_index
        sample_bandwidths = sample.field_targets.bandwidths
        bandwidths[batch_slice] = (
            sample_bandwidths.unsqueeze(0).expand(q_count, -1)
            if sample_bandwidths.ndim == 1
            else sample_bandwidths
        )
        distance[batch_slice, :owner_count] = sample.field_targets.distance
        density[batch_slice, :owner_count] = sample.field_targets.density
        field_valid[batch_slice, :owner_count] = sample.field_targets.valid_mask
        role = sample.field_targets.owner_role
        owner_role[batch_slice, :owner_count] = role.unsqueeze(0).expand(q_count, -1) if role.ndim == 1 else role
        sensitivity = sample.sensitivity_targets
        owner_index[batch_slice, :edge_count] = _selector_block(sensitivity.owner_index, q_count)
        edge_query_index[batch_slice, :edge_count] = _selector_block(sensitivity.query_index, q_count)
        joint_index[batch_slice, :edge_count] = _selector_block(sensitivity.joint_index, q_count)
        ancestor_mask[batch_slice, :edge_count] = _selector_block(sensitivity.ancestor_mask, q_count)
        active_mask[batch_slice, :edge_count] = _selector_block(sensitivity.active_mask, q_count)
        closest_point[batch_slice, :edge_count] = sensitivity.closest_point
        closest_source[batch_slice, :edge_count] = sensitivity.closest_source
        uniqueness_margin[batch_slice, :edge_count] = sensitivity.uniqueness_margin
        kappa[batch_slice, :edge_count] = sensitivity.kappa
        field_sensitivity[batch_slice, :edge_count] = sensitivity.field_sensitivity
        edge_valid[batch_slice, :edge_count] = sensitivity.valid_mask
        for source, target in (
            (sensitivity.owner_category, edge_owner_category),
            (sensitivity.query_stratum, edge_query_stratum),
            (sensitivity.fallback_category, edge_fallback_category),
            (sensitivity.sampling_role, edge_sampling_role),
        ):
            if source is not None:
                target[batch_slice, :edge_count] = _selector_block(source, q_count)
        if sensitivity.central_difference is not None:
            central_difference[batch_slice, :edge_count] = sensitivity.central_difference
        if sensitivity.central_difference_valid_mask is not None:
            central_difference_valid[batch_slice, :edge_count] = sensitivity.central_difference_valid_mask
        if sensitivity.central_difference_plus_face is not None:
            central_plus_face[batch_slice, :edge_count] = sensitivity.central_difference_plus_face
        if sensitivity.central_difference_minus_face is not None:
            central_minus_face[batch_slice, :edge_count] = sensitivity.central_difference_minus_face
        central_difference_elapsed_seconds += sensitivity.central_difference_elapsed_seconds
        anchor_index[batch_slice] = int(sample.anchor_index)
        if sample.q_index is not None:
            if sample.q_index.numel() != q_count:
                raise ValueError("online geometry q_index count must match its q-block")
            q_index[batch_slice] = sample.q_index.reshape(-1).to(device=device)
        batch_start += q_count

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
        owner_category=edge_owner_category,
        query_stratum=edge_query_stratum,
        fallback_category=edge_fallback_category,
        sampling_role=edge_sampling_role,
        central_difference=central_difference,
        central_difference_valid_mask=central_difference_valid,
        central_difference_plus_face=central_plus_face,
        central_difference_minus_face=central_minus_face,
        central_difference_elapsed_seconds=central_difference_elapsed_seconds,
        provenance={
            "frame": "h",
            "distance_unit": "m",
            "joint_unit": "rad",
            "padding": f"edge={max_edge_count}",
        },
    )
    return PaddedOnlineGeometryBatch(
        asset_ids=tuple(expanded_asset_ids),
        q=q,
        evidence=evidence,
        evidence_row_index=evidence_row_index,
        queries=queries,
        field_targets=field_targets,
        sensitivity_targets=sensitivity_targets,
        anchor_index=anchor_index,
        q_index=q_index,
    )


def pad_online_geometry_blocks(
    blocks: list[OnlineGeometrySample],
    *,
    padding: GeometryPaddingCfg,
) -> PaddedOnlineGeometryBatch:
    r"""显式 hot-path 名称：按资产 block 顺序连续写入每项资产的 q 轴。"""

    return pad_online_geometry_samples(blocks, padding=padding)


def _selector_block(selector: torch.Tensor, q_count: int) -> torch.Tensor:
    """把共享 `[E]` selector 广播为 `[Q,E]`，逐 q selector 原样返回。"""

    if selector.ndim == 1:
        return selector.unsqueeze(0).expand(q_count, -1)
    if selector.ndim != 2 or selector.shape[0] != q_count:
        raise ValueError("q-block selector must have shape [E] or [Q,E]")
    return selector


def method_batch_views(batch: PaddedOnlineGeometryBatch) -> MethodBatchViews:
    r"""按模型/读出/真值三块切开；模型路径不得消费 `truth`。"""

    targets = batch.sensitivity_targets
    return MethodBatchViews(
        model_input=(batch.q, batch.evidence, batch.evidence_row_index),
        readout_condition=(
            batch.queries.query_points_h,
            batch.field_targets.bandwidths,
            targets.owner_index,
            targets.query_index,
            targets.joint_index,
        ),
        truth=(batch.field_targets, batch.sensitivity_targets),
    )


def split_padded_online_geometry_batch(
    batch: PaddedOnlineGeometryBatch,
    *,
    microbatch_size: int,
) -> tuple[PaddedOnlineGeometryBatch, ...]:
    r"""沿 `(asset,q)` 样本轴切分 padded batch，不改变物理值或采样身份。

    外部 schedule 仍生成一个完整 logical minibatch；本函数只控制 GPU activation 的
    瞬时存活规模。每个切片保留原始 `asset_ids` 与 `q_index`，因此 joint-sign rewrite
    应在切分前完成，避免同一逻辑 batch 的随机选择依赖切片位置。
    """

    if microbatch_size < 1:
        raise ValueError("microbatch_size must be positive")
    batch_size = int(batch.q.shape[0])
    if batch_size < 1:
        raise ValueError("padded geometry batch must contain at least one sample")
    if microbatch_size >= batch_size:
        return (batch,)
    return tuple(
        _slice_padded_batch(batch, start=start, stop=min(start + microbatch_size, batch_size))
        for start in range(0, batch_size, microbatch_size)
    )


def _slice_padded_batch(
    batch: PaddedOnlineGeometryBatch,
    *,
    start: int,
    stop: int,
) -> PaddedOnlineGeometryBatch:
    r"""切分一个 batch 及其三类嵌套 typed tensors。"""

    batch_size = int(batch.q.shape[0])

    def slice_dataclass(value: _DataclassT) -> _DataclassT:
        r"""沿样本轴切片任意 typed batch，同时保留调用点的具体 dataclass 类型。"""

        updates: dict[str, Any] = {}
        for field_info in fields(cast(Any, value)):
            field_value = getattr(value, field_info.name)
            if isinstance(field_value, torch.Tensor) and field_value.ndim > 0 and field_value.shape[0] == batch_size:
                field_value = field_value[start:stop]
            updates[field_info.name] = field_value
        return cast(_DataclassT, replace(cast(Any, value), **updates))

    return PaddedOnlineGeometryBatch(
        asset_ids=batch.asset_ids[start:stop],
        q=batch.q[start:stop],
        evidence=batch.evidence,
        evidence_row_index=(
            batch.evidence_row_index[start:stop] if batch.evidence_row_index is not None else None
        ),
        queries=slice_dataclass(batch.queries),
        field_targets=slice_dataclass(batch.field_targets),
        sensitivity_targets=slice_dataclass(batch.sensitivity_targets),
        anchor_index=batch.anchor_index[start:stop] if batch.anchor_index is not None else None,
        q_index=batch.q_index[start:stop] if batch.q_index is not None else None,
    )


__all__ = [
    "GeometryPaddingCfg",
    "MethodBatchViews",
    "OnlineGeometrySample",
    "PaddedOnlineGeometryBatch",
    "PhysicalOnlineGeometrySample",
    "attach_static_evidence",
    "attach_static_evidence_block",
    "method_batch_views",
    "pad_online_geometry_samples",
    "pad_online_geometry_blocks",
    "split_padded_online_geometry_batch",
    "split_online_geometry_sample",
    "split_physical_online_geometry_sample",
]
