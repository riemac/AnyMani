"""Construct shared geometry evidence and padded owner-token batches."""


from __future__ import annotations

from collections.abc import Callable
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


    asset_id: str
    q: torch.Tensor
    evidence: StaticGeometryEvidence
    queries: SpatialQueryBatch  # `[1,G,N_Q,...]`
    field_targets: FieldTargetBatch
    sensitivity_targets: SensitivityTargetBatch
    anchor_index: int = 0
    q_index: torch.Tensor | None = None  # `[1]`


@dataclass(frozen=True)
class PaddedOnlineGeometryBatch:


    asset_ids: tuple[str, ...]
    q: torch.Tensor
    evidence: StaticGeometryEvidence  # `[A_unique,G^{max},...]` + masks
    queries: SpatialQueryBatch  # `[B,G^{max},N_Q,...]`
    field_targets: FieldTargetBatch
    sensitivity_targets: SensitivityTargetBatch
    evidence_row_index: torch.Tensor | None = None
    anchor_index: torch.Tensor | None = None
    q_index: torch.Tensor | None = None  # `[B]`
    joint_coordinate_sign: torch.Tensor | None = None


@dataclass(frozen=True)
class MethodBatchViews:


    model_input: tuple[torch.Tensor, StaticGeometryEvidence, torch.Tensor | None, torch.Tensor | None]
    readout_condition: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    truth: tuple[FieldTargetBatch, SensitivityTargetBatch]


def _map_dataclass_tensors(value: _DataclassT, transform: Callable[[torch.Tensor], torch.Tensor]) -> _DataclassT:


    updates: dict[str, Any] = {}
    for field_info in fields(cast(Any, value)):
        field_value = getattr(value, field_info.name)
        updates[field_info.name] = transform(field_value) if isinstance(field_value, torch.Tensor) else field_value
    return cast(_DataclassT, replace(cast(Any, value), **updates))


def _map_padded_batch_tensors(
    batch: PaddedOnlineGeometryBatch,
    transform: Callable[[torch.Tensor], torch.Tensor],
) -> PaddedOnlineGeometryBatch:


    def optional(value: torch.Tensor | None) -> torch.Tensor | None:
        return None if value is None else transform(value)

    return PaddedOnlineGeometryBatch(
        asset_ids=batch.asset_ids,
        q=transform(batch.q),
        evidence=_map_dataclass_tensors(batch.evidence, transform),
        queries=_map_dataclass_tensors(batch.queries, transform),
        field_targets=_map_dataclass_tensors(batch.field_targets, transform),
        sensitivity_targets=_map_dataclass_tensors(batch.sensitivity_targets, transform),
        evidence_row_index=optional(batch.evidence_row_index),
        anchor_index=optional(batch.anchor_index),
        q_index=optional(batch.q_index),
        joint_coordinate_sign=optional(batch.joint_coordinate_sign),
    )


def stage_padded_batch_for_replay(batch: PaddedOnlineGeometryBatch) -> PaddedOnlineGeometryBatch:


    def stage(tensor: torch.Tensor) -> torch.Tensor:
        source = tensor.detach()
        if torch.cuda.is_available():
            target = torch.empty_like(source, device="cpu", pin_memory=True)
            target.copy_(source, non_blocking=False)
            return target
        return source.cpu().clone()

    return _map_padded_batch_tensors(batch, stage)


def restore_padded_batch_from_replay(
    batch: PaddedOnlineGeometryBatch,
    *,
    device: torch.device | str,
) -> PaddedOnlineGeometryBatch:


    target = torch.device(device)
    if batch.q.device.type != "cpu":
        raise ValueError("replay unit must be staged on CPU before device restore")
    return _map_padded_batch_tensors(
        batch,
        lambda tensor: tensor.to(device=target, non_blocking=tensor.is_pinned()),
    )


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


    return selector[index : index + 1] if selector.ndim == 2 else selector


def _slice_optional_selector(selector: torch.Tensor | None, index: int) -> torch.Tensor | None:


    return None if selector is None else _slice_selector(selector, index)


def _selector_row(selector: torch.Tensor) -> torch.Tensor:


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
    central_difference_elapsed_seconds = 0.0
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
        joint_coordinate_sign=None,
    )


def pad_online_geometry_blocks(
    blocks: list[OnlineGeometrySample],
    *,
    padding: GeometryPaddingCfg,
) -> PaddedOnlineGeometryBatch:


    return pad_online_geometry_samples(blocks, padding=padding)


def _selector_block(selector: torch.Tensor, q_count: int) -> torch.Tensor:


    if selector.ndim == 1:
        return selector.unsqueeze(0).expand(q_count, -1)
    if selector.ndim != 2 or selector.shape[0] != q_count:
        raise ValueError("q-block selector must have shape [E] or [Q,E]")
    return selector


def method_batch_views(batch: PaddedOnlineGeometryBatch) -> MethodBatchViews:


    targets = batch.sensitivity_targets
    return MethodBatchViews(
        model_input=(batch.q, batch.evidence, batch.evidence_row_index, batch.joint_coordinate_sign),
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


    batch_size = int(batch.q.shape[0])

    def slice_dataclass(value: _DataclassT) -> _DataclassT:


        updates: dict[str, Any] = {}
        for field_info in fields(cast(Any, value)):
            field_value = getattr(value, field_info.name)
            if isinstance(field_value, torch.Tensor) and field_value.ndim > 0 and field_value.shape[0] == batch_size:
                field_value = field_value[start:stop]
            updates[field_info.name] = field_value
        return cast(_DataclassT, replace(cast(Any, value), **updates))


    evidence = batch.evidence
    compact_row_index = batch.evidence_row_index[start:stop] if batch.evidence_row_index is not None else None
    if compact_row_index is not None:
        selected_rows, inverse = torch.unique_consecutive(compact_row_index, return_inverse=True)
        if torch.unique(selected_rows).numel() != selected_rows.numel():
            raise ValueError("microbatch evidence rows must remain asset-major contiguous blocks")
        evidence = _select_static_evidence_rows(evidence, selected_rows)
        compact_row_index = inverse.to(dtype=torch.long)
    return PaddedOnlineGeometryBatch(
        asset_ids=batch.asset_ids[start:stop],
        q=batch.q[start:stop],
        evidence=evidence,
        evidence_row_index=compact_row_index,
        queries=slice_dataclass(batch.queries),
        field_targets=slice_dataclass(batch.field_targets),
        sensitivity_targets=slice_dataclass(batch.sensitivity_targets),
        anchor_index=batch.anchor_index[start:stop] if batch.anchor_index is not None else None,
        q_index=batch.q_index[start:stop] if batch.q_index is not None else None,
        joint_coordinate_sign=(
            batch.joint_coordinate_sign[start:stop] if batch.joint_coordinate_sign is not None else None
        ),
    )


def _select_static_evidence_rows(
    evidence: StaticGeometryEvidence,
    row_index: torch.Tensor,
) -> StaticGeometryEvidence:


    if evidence.anchors.ndim != 3 or row_index.ndim != 1 or row_index.dtype != torch.long:
        raise ValueError("static evidence compaction requires batched evidence and long row_index")
    source_rows = evidence.anchors.shape[0]

    def select(value: torch.Tensor) -> torch.Tensor:
        return value.index_select(0, row_index) if value.ndim > 0 and value.shape[0] == source_rows else value

    def select_optional(value: torch.Tensor | None) -> torch.Tensor | None:
        return None if value is None else select(value)

    return StaticGeometryEvidence(
        anchors=select(evidence.anchors),
        home_surface_points=select(evidence.home_surface_points),
        home_surface_mask=select(evidence.home_surface_mask),
        palm_normal=select(evidence.palm_normal),
        space_screws=select(evidence.space_screws),
        q_home=select(evidence.q_home),
        entity_role=select(evidence.entity_role),
        entity_joint_index=select(evidence.entity_joint_index),
        joint_entity_index=select(evidence.joint_entity_index),
        shortest_path=select(evidence.shortest_path),
        parent_direction=select(evidence.parent_direction),
        child_direction=select(evidence.child_direction),
        entity_valid_mask=select_optional(evidence.entity_valid_mask),
        joint_valid_mask=select_optional(evidence.joint_valid_mask),
        anchor_valid_mask=select_optional(evidence.anchor_valid_mask),
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
    "restore_padded_batch_from_replay",
    "split_padded_online_geometry_batch",
    "split_online_geometry_sample",
    "split_physical_online_geometry_sample",
    "stage_padded_batch_for_replay",
]
