"""Lower typed hand geometry into device-ready source data."""


from __future__ import annotations

from dataclasses import dataclass, field

import torch

from anymani.assets.bank import HandContainer

from .anchor_sampling import (
    AnchorClassificationStats,
    AnchorRealization,
    AnchorSamples,
    sample_palm_anchor_bank_warp,
    sample_palm_anchor_realization_warp,
)
from .collision_geometry import (
    GeometryIdentity,
    HomeSurfaceSamples,
    OwnerGeometryCache,
    OwnerSurfaceSamplingArrays,
    WarpOwnerGeometryCache,
    WarpSurfaceView,
    geometry_identity,
    materialize_owner_geometry_cache,
    materialize_warp_owner_geometry_cache,
    prepare_owner_surface_sampling_arrays,
    prepare_warp_surface_view,
    release_warp_owner_geometry_cache,
    sample_owner_home_surfaces,
)
from .kinematics import EmbodimentGeometrySpec, lower_hand_geometry_semantics


@dataclass(frozen=True)
class AnchorBankCfg:


    bank_size: int = 8
    anchors_per_finger: int = 10
    radius_m: float = 0.05
    radial_decay_scale_m: float = 0.025
    surface_fraction: float = 0.5

    def __post_init__(self) -> None:


        if self.bank_size < 1 or self.anchors_per_finger < 1:
            raise ValueError("anchor bank size and anchors_per_finger must be positive")
        if not 0.0 < self.radial_decay_scale_m <= self.radius_m:
            raise ValueError("anchor radial decay scale must lie in (0, radius_m]")
        if not 0.0 <= self.surface_fraction <= 1.0:
            raise ValueError("anchor surface_fraction must lie in [0,1]")


@dataclass(frozen=True)
class GeometrySourceCfg:


    home_points_per_owner: int = 64
    home_surface_oversample_factor: int = 8
    static_sampling_seed: int = 0
    anchors: AnchorBankCfg = field(default_factory=AnchorBankCfg)

    def __post_init__(self) -> None:


        if self.home_points_per_owner < 1 or self.home_surface_oversample_factor < 1:
            raise ValueError("home-surface point and oversample budgets must be positive")

    @property
    def anchors_per_finger(self) -> int:


        return self.anchors.anchors_per_finger

    @property
    def anchor_bank_size(self) -> int:


        return self.anchors.bank_size

    @property
    def anchor_radius_m(self) -> float:
        return self.anchors.radius_m

    @property
    def anchor_radial_decay_scale_m(self) -> float:
        return self.anchors.radial_decay_scale_m

    @property
    def anchor_surface_fraction(self) -> float:
        return self.anchors.surface_fraction


@dataclass(frozen=True)
class GeometrySourceCore:


    container: HandContainer
    spec_cpu: EmbodimentGeometrySpec  # CPU float64 POE/graph/component transforms
    geometry_cache: OwnerGeometryCache  # owner-local strict surface/solid union
    home_surface: HomeSurfaceSamples  # `[G,M,3]` owner-local boundary realization
    identity: GeometryIdentity
    surface_sampling_arrays: OwnerSurfaceSamplingArrays | None = None  # query triangles/normals/CDF
    warp_surface_views: tuple[WarpSurfaceView, ...] | None = None

    @property
    def asset_id(self) -> str:


        return self.container.asset_id

    @classmethod
    def materialize(
        cls,
        container: HandContainer,
        *,
        config: GeometrySourceCfg = GeometrySourceCfg(),
    ) -> GeometrySourceCore:


        semantics = container.geometry_semantics
        if semantics is None:
            raise ValueError("container must be resolved with require_geometry_semantics=True")
        spec = lower_hand_geometry_semantics(semantics, dtype=torch.float64)
        geometry_cache = materialize_owner_geometry_cache(container, spec)
        identity = geometry_identity(semantics, spec, geometry_cache)
        home_surface = sample_owner_home_surfaces(
            geometry_cache,
            points_per_owner=config.home_points_per_owner,
            sampling_seed=config.static_sampling_seed,
            oversample_factor=config.home_surface_oversample_factor,
        )
        surface_sampling_arrays = prepare_owner_surface_sampling_arrays(geometry_cache)
        warp_surface_views = tuple(
            prepare_warp_surface_view(record.surface_mesh, owner_id=record.owner_id)
            for record in geometry_cache.records
        )
        return cls(
            container,
            spec,
            geometry_cache,
            home_surface,
            identity,
            surface_sampling_arrays,
            warp_surface_views,
        )

    def finalize_on_device(
        self,
        *,
        config: GeometrySourceCfg,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> tuple[GeometrySource, DeviceGeometrySource, AnchorClassificationStats]:


        target_device = torch.device(device)
        spec_device = self.spec_cpu.to(device=target_device, dtype=dtype)
        warp_cache = materialize_warp_owner_geometry_cache(
            self.geometry_cache,
            device=str(target_device),
            surface_views=self.warp_surface_views,
        )
        try:
            semantics = self.container.geometry_semantics
            if semantics is None:
                raise ValueError("geometry source core lost typed semantics before CUDA finalization")
            anchor_bank, stats = sample_palm_anchor_bank_warp(
                self.geometry_cache,
                semantics,
                self.spec_cpu,
                warp_cache,
                bank_size=config.anchors.bank_size,
                anchors_per_finger=config.anchors.anchors_per_finger,
                static_sampling_seed=config.static_sampling_seed,
                radial_support_radius_m=config.anchors.radius_m,
                radial_decay_scale_m=config.anchors.radial_decay_scale_m,
                surface_fraction=config.anchors.surface_fraction,
            )
            source = GeometrySource.from_core(self, anchor_bank=anchor_bank)
            return source, DeviceGeometrySource(source, spec_device, warp_cache), stats
        except Exception:
            release_warp_owner_geometry_cache(warp_cache)
            raise

    def finalize_selected_on_device(
        self,
        *,
        config: GeometrySourceCfg,
        bank_index: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> tuple[GeometrySource, DeviceGeometrySource, AnchorClassificationStats]:


        target_device = torch.device(device)
        spec_device = self.spec_cpu.to(device=target_device, dtype=dtype)
        warp_cache = materialize_warp_owner_geometry_cache(
            self.geometry_cache,
            device=str(target_device),
            surface_views=self.warp_surface_views,
        )
        try:
            semantics = self.container.geometry_semantics
            if semantics is None:
                raise ValueError("geometry source core lost typed semantics before selected anchor finalization")
            realization, stats = sample_palm_anchor_realization_warp(
                self.geometry_cache,
                semantics,
                self.spec_cpu,
                warp_cache,
                bank_index=bank_index,
                bank_size=config.anchors.bank_size,
                anchors_per_finger=config.anchors.anchors_per_finger,
                static_sampling_seed=config.static_sampling_seed,
                radial_support_radius_m=config.anchors.radius_m,
                radial_decay_scale_m=config.anchors.radial_decay_scale_m,
                surface_fraction=config.anchors.surface_fraction,
            )
            source = GeometrySource.from_core(
                self,
                anchor_bank=(realization.samples,),
                anchor_realization=realization,
            )
            return source, DeviceGeometrySource(source, spec_device, warp_cache), stats
        except Exception:
            release_warp_owner_geometry_cache(warp_cache)
            raise


@dataclass(frozen=True)
class GeometrySource:


    container: HandContainer
    spec_cpu: EmbodimentGeometrySpec  # CPU float64 POE/graph/component transforms
    geometry_cache: OwnerGeometryCache  # owner-local strict surface/solid union
    home_surface: HomeSurfaceSamples  # `[G,M,3]` owner-local boundary-only realization
    anchors: AnchorSamples
    anchor_bank: tuple[AnchorSamples, ...]
    identity: GeometryIdentity
    anchor_realization: AnchorRealization | None = None
    surface_sampling_arrays: OwnerSurfaceSamplingArrays | None = None
    warp_surface_views: tuple[WarpSurfaceView, ...] | None = None

    @property
    def asset_id(self) -> str:


        return self.container.asset_id

    @classmethod
    def materialize(
        cls,
        container: HandContainer,
        *,
        config: GeometrySourceCfg = GeometrySourceCfg(),
        anchor_device: str = "cpu",
    ) -> GeometrySource:


        core = GeometrySourceCore.materialize(container, config=config)
        semantics = core.container.geometry_semantics
        if semantics is None:
            raise ValueError("geometry source core is missing typed semantics")

        warp_cache = materialize_warp_owner_geometry_cache(
            core.geometry_cache,
            device=anchor_device,
            surface_views=core.warp_surface_views,
        )
        try:
            anchor_bank, _stats = sample_palm_anchor_bank_warp(
                core.geometry_cache,
                semantics,
                core.spec_cpu,
                warp_cache,
                bank_size=config.anchors.bank_size,
                anchors_per_finger=config.anchors.anchors_per_finger,
                static_sampling_seed=config.static_sampling_seed,
                radial_support_radius_m=config.anchors.radius_m,
                radial_decay_scale_m=config.anchors.radial_decay_scale_m,
                surface_fraction=config.anchors.surface_fraction,
            )
        finally:
            release_warp_owner_geometry_cache(warp_cache)
        return cls.from_core(core, anchor_bank=anchor_bank)

    @classmethod
    def from_core(
        cls,
        core: GeometrySourceCore,
        *,
        anchor_bank: tuple[AnchorSamples, ...],
        anchor_realization: AnchorRealization | None = None,
    ) -> GeometrySource:


        if not anchor_bank:
            raise ValueError("geometry source finalization requires a non-empty anchor bank")
        return cls(
            core.container,
            core.spec_cpu,
            core.geometry_cache,
            core.home_surface,
            anchor_bank[0],
            anchor_bank,
            core.identity,
            anchor_realization,
            core.surface_sampling_arrays,
            core.warp_surface_views,
        )

    def to_device(
        self,
        *,
        device: torch.device | str = "cuda:0",
        dtype: torch.dtype = torch.float32,
    ) -> DeviceGeometrySource:


        target_device = torch.device(device)
        spec = self.spec_cpu.to(device=target_device, dtype=dtype)
        warp_cache = materialize_warp_owner_geometry_cache(
            self.geometry_cache,
            device=str(target_device),
            surface_views=self.warp_surface_views,
        )
        return DeviceGeometrySource(source=self, spec=spec, warp_cache=warp_cache)


@dataclass(frozen=True)
class DeviceGeometrySource:


    source: GeometrySource
    spec: EmbodimentGeometrySpec  # GPU POE/graph tensors
    warp_cache: WarpOwnerGeometryCache

    def release(self) -> bool:


        return release_warp_owner_geometry_cache(self.warp_cache)


__all__ = ["AnchorBankCfg", "DeviceGeometrySource", "GeometrySource", "GeometrySourceCfg", "GeometrySourceCore"]
