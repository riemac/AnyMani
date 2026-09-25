"""Physical geometry teacher assembly. Joint angles use radians, lengths metres, and all positions and queries use hand frame {h}. Model inputs exclude teacher labels."""


from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field

import torch

from anymani.distill.representations.queries.spatial_sampling import (
    OwnerSurfaceSamplingCache,
    SpatialQueryBatch,
    SpatialQuerySamplerCfg,
    materialize_owner_surface_sampling_cache,
    sample_spatial_queries,
)
from anymani.distill.representations.sources.anchor_sampling import AnchorClassificationStats
from anymani.distill.representations.sources.geometry_source import (
    DeviceGeometrySource,
    GeometrySource,
    GeometrySourceCfg,
    GeometrySourceCore,
)
from anymani.distill.representations.sources.kinematics import (
    EmbodimentGeometrySpec,
    forward_owner_transforms_and_spatial_screws,
)
from anymani.distill.representations.targets.field_samples import (
    FieldTargetBatch,
    SensitivityTargetBatch,
)
from anymani.distill.representations.targets.geometry_field import (  # Warp teacher assembly
    GaussianProximityFieldCfg,  # train/validation sigma measure
    GeometryFieldTargetCfg,  # edges/mask thresholds
    generate_geometry_field_targets,
)


@dataclass(frozen=True)
class GeometryRepresentationCfg:


    source: GeometrySourceCfg = dataclass_field(default_factory=GeometrySourceCfg)  # q-independent physical oracle
    field: GaussianProximityFieldCfg = dataclass_field(default_factory=GaussianProximityFieldCfg)  # sigma measure
    query: SpatialQuerySamplerCfg = dataclass_field(default_factory=SpatialQuerySamplerCfg)  # query measure
    target: GeometryFieldTargetCfg = dataclass_field(default_factory=GeometryFieldTargetCfg)  # edge/mask teacher


@dataclass(frozen=True)
class GeometryRepresentationState:


    source: GeometrySource
    device_source: DeviceGeometrySource
    surface_sampling: OwnerSurfaceSamplingCache  # owner triangle/normal/area proposal tables
    anchor_classification: AnchorClassificationStats | None = None

    @property
    def anchor_realization(self):


        return self.source.anchor_realization

    @property
    def spec(self) -> EmbodimentGeometrySpec:


        return self.device_source.spec

    @property
    def warp_cache(self):


        return self.device_source.warp_cache


class GeometryRepresentation:


    def __init__(self, config: GeometryRepresentationCfg) -> None:


        self.config = config

    def materialize_source(self, container, *, anchor_device: str = "cpu") -> GeometrySource:


        return GeometrySource.materialize(container, config=self.config.source, anchor_device=anchor_device)

    def materialize_core(self, container) -> GeometrySourceCore:


        return GeometrySourceCore.materialize(container, config=self.config.source)

    def to_device(
        self,
        source: GeometrySource | GeometrySourceCore,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        bank_index: int = 0,
    ) -> GeometryRepresentationState:


        if isinstance(source, GeometrySourceCore):
            finalized, device_source, anchor_stats = source.finalize_selected_on_device(
                config=self.config.source,
                bank_index=bank_index,
                device=device,
                dtype=dtype,
            )
        else:
            finalized = source
            device_source = source.to_device(device=device, dtype=dtype)
            anchor_stats = None
        return self.assemble_device_state(
            finalized,
            device_source,
            anchor_stats=anchor_stats,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def assemble_device_state(
        source: GeometrySource,
        device_source: DeviceGeometrySource,
        *,
        anchor_stats: AnchorClassificationStats | None,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> GeometryRepresentationState:


        try:
            surface_sampling = materialize_owner_surface_sampling_cache(
                source.geometry_cache,
                device=torch.device(device),
                dtype=dtype,
                arrays=source.surface_sampling_arrays,
            )
            return GeometryRepresentationState(source, device_source, surface_sampling, anchor_stats)
        except Exception:
            device_source.release()
            raise

    def sample(
        self,
        state: GeometryRepresentationState,
        q: torch.Tensor,
        *,
        sampling_seed: int,
        q_index: torch.Tensor | None = None,
        anchor_index: int = 0,
        supervision_split: str = "train",
    ) -> PhysicalOnlineGeometrySample:


        return sample_online_geometry(
            state,
            q,
            field_config=self.config.field,
            query_config=self.config.query,
            target_config=self.config.target,
            sampling_seed=sampling_seed,
            q_index=q_index,
            anchor_index=anchor_index,
            supervision_split=supervision_split,
        )


@dataclass(frozen=True)
class PhysicalOnlineGeometrySample:


    asset_id: str
    q: torch.Tensor
    queries: SpatialQueryBatch
    field_targets: FieldTargetBatch
    sensitivity_targets: SensitivityTargetBatch
    anchor_index: int = 0
    q_index: torch.Tensor | None = None


def sample_online_geometry(
    state: GeometryRepresentationState,
    q: torch.Tensor,
    *,
    field_config: GaussianProximityFieldCfg = GaussianProximityFieldCfg(),
    query_config: SpatialQuerySamplerCfg = SpatialQuerySamplerCfg(),
    target_config: GeometryFieldTargetCfg = GeometryFieldTargetCfg(),
    sampling_seed: int = 0,
    q_index: torch.Tensor | None = None,
    anchor_index: int = 0,
    supervision_split: str = "train",
) -> PhysicalOnlineGeometrySample:


    if q.ndim != 2 or q.shape[1] != state.spec.space_screws.shape[0] or q.shape[0] < 1:
        raise ValueError("sample_online_geometry expects [Q,N_J] with the asset's true N_J")
    if q_index is not None and q_index.shape != (q.shape[0],):
        raise ValueError("q_index must have shape [Q] matching the asset q block")
    bank = state.source.anchor_bank
    if not bank:
        raise ValueError("geometry source is missing its physical anchor realization")
    realization = state.anchor_realization
    if realization is not None:
        if int(anchor_index) != realization.bank_index:
            raise IndexError(
                f"requested anchor_index={anchor_index} does not match resident bank={realization.bank_index}"
            )
        selected_anchors = realization.samples
    else:
        if not 0 <= int(anchor_index) < len(bank):
            raise IndexError(f"anchor_index={anchor_index} is outside bank size {len(bank)}")
        selected_anchors = bank[int(anchor_index)]
    anchors = torch.as_tensor(
        selected_anchors.anchors_hand_m,
        device=q.device,
        dtype=q.dtype,
    )
    owner_transforms, current_spatial_screws = forward_owner_transforms_and_spatial_screws(
        state.spec,
        q.detach(),
    )
    queries = sample_spatial_queries(
        q,
        state.spec,
        state.surface_sampling,
        anchors,
        config=query_config,
        sampling_seed=sampling_seed,
        owner_transforms=owner_transforms,
    )
    field_targets, sensitivity_targets = generate_geometry_field_targets(  # GPU Warp teacher
        q,
        state.spec,  # POE/ancestor masks
        state.source.geometry_cache,  # CPU face/component provenance
        state.warp_cache,  # GPU BVHs
        queries,
        field_config=field_config,
        target_config=target_config,  # sampled edges/margins
        edge_sampling_seed=sampling_seed,  # sampled `(g,r,i)` realization
        supervision_split=supervision_split,  # train 1+1 / validation 4+4
        owner_transforms=owner_transforms,
        current_spatial_screws=current_spatial_screws,
        q_index=q_index,
    )
    return PhysicalOnlineGeometrySample(
        asset_id=state.source.asset_id,
        q=q,
        queries=queries,
        field_targets=field_targets,
        sensitivity_targets=sensitivity_targets,
        anchor_index=int(anchor_index),
        q_index=q_index.detach().cpu() if q_index is not None else None,
    )


__all__ = [
    "GeometryRepresentation",
    "GeometryRepresentationCfg",
    "GeometryRepresentationState",
    "PhysicalOnlineGeometrySample",
    "sample_online_geometry",
]
