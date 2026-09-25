"""Bounded resident geometry asset windows."""


from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from time import perf_counter

import torch

from anymani.distill.representations.geometry import GeometryRepresentationState
from anymani.distill.representations.sources.geometry_source import GeometrySource, GeometrySourceCore


class ResidentGeometryAssetWindow:


    def __init__(
        self,
        runtimes: Sequence[GeometrySource | GeometrySourceCore],
        *,
        device: str,
        dtype,
        max_resident_assets: int,
        loader: Callable[..., GeometryRepresentationState],
        releaser: Callable[[GeometryRepresentationState], bool] | None = None,
        catalog_ids: tuple[str, ...] | None = None,
        source_provider: object | None = None,
        resource_profile: bool = False,
    ) -> None:
        if not runtimes:
            raise ValueError("resident window requires a non-empty CPU catalog")
        if max_resident_assets < 1:
            raise ValueError("max_resident_assets must be positive")
        if max_resident_assets > len(runtimes):
            max_resident_assets = len(runtimes)
        self.source_provider = source_provider
        if catalog_ids is not None:
            if len(set(catalog_ids)) != len(catalog_ids):
                raise ValueError("CPU catalog asset IDs must be unique")
            if len(catalog_ids) != len(runtimes):
                raise ValueError("lazy source catalog IDs must match the source provider length")
            self.catalog = {asset_id: None for asset_id in catalog_ids}
        else:
            self.catalog = {runtime.asset_id: runtime for runtime in runtimes}
        if len(self.catalog) != len(runtimes):
            raise ValueError("CPU catalog asset IDs must be unique")
        self.device = device
        self.dtype = dtype
        self.max_resident_assets = max_resident_assets
        self.loader = loader
        self.releaser = releaser or (lambda state: state.device_source.release())
        self.resource_profile = bool(resource_profile)
        self._resident: dict[str, GeometryRepresentationState] = {}
        self._resident_bank_index: int | None = None
        self._telemetry_events: list[dict[str, object]] = []

    @property
    def resident_asset_ids(self) -> tuple[str, ...]:
        return tuple(self._resident)

    def ensure(
        self,
        asset_ids: tuple[str, ...] | list[str],
        *,
        prefetch_sources: bool = True,
        prepared_sources: Mapping[str, GeometrySource | GeometrySourceCore] | None = None,
        bank_index: int | None = None,
    ) -> tuple[GeometryRepresentationState, ...]:


        requested = tuple(asset_ids)
        if len(set(requested)) != len(requested):
            raise ValueError("one minibatch cannot request duplicate asset IDs")
        if len(requested) > self.max_resident_assets:
            raise ValueError("requested asset group exceeds max_resident_assets")
        for asset_id in requested:
            if asset_id not in self.catalog:
                raise KeyError(f"unknown geometry asset ID={asset_id!r}")
        if prepared_sources is not None and set(prepared_sources) != set(requested):
            raise ValueError("prepared source IDs must exactly match the requested device subwindow")
        if tuple(self._resident) == requested and self._resident_bank_index == bank_index:
            return tuple(self._resident[asset_id] for asset_id in requested)
        before_memory = self._memory_snapshot() if self.resource_profile else None
        started = perf_counter()
        released_asset_ids: list[str] = []
        loaded_asset_ids: list[str] = []
        release_started = perf_counter()
        for asset_id in tuple(self._resident):
            if asset_id not in requested or self._resident_bank_index != bank_index:
                self._evict(asset_id)
                released_asset_ids.append(asset_id)
        release_seconds = perf_counter() - release_started
        load_started = perf_counter()
        if prefetch_sources and self.source_provider is not None:
            prefetch_fn = getattr(self.source_provider, "prefetch", None)
            if prefetch_fn is not None:
                prefetch_fn(requested)
        for asset_id in requested:
            if asset_id not in self._resident:
                source = prepared_sources[asset_id] if prepared_sources is not None else self._source_for_asset(asset_id)
                loader_kwargs = {"device": self.device, "dtype": self.dtype}
                if bank_index is not None:
                    loader_kwargs["bank_index"] = bank_index
                self._resident[asset_id] = self.loader(source, **loader_kwargs)
                loaded_asset_ids.append(asset_id)
        self._resident_bank_index = bank_index
        if len(self._resident) > self.max_resident_assets:
            raise RuntimeError("resident asset window exceeded configured cap")
        load_seconds = perf_counter() - load_started
        if loaded_asset_ids or released_asset_ids:
            after_memory = self._memory_snapshot() if self.resource_profile else None
            self._telemetry_events.append(
                {
                    "event": "resident_window",
                    "requested_asset_ids": list(requested),
                    "loaded_asset_ids": loaded_asset_ids,
                    "released_asset_ids": released_asset_ids,
                    "resident_asset_ids": list(self.resident_asset_ids),
                    "resident_asset_count": len(self._resident),
                    "anchor_bank_index": bank_index,
                    "resident_owner_bvh_count": self._resident_owner_bvh_count(),
                    "resident_triangle_count": self._resident_triangle_count(),
                    "load_seconds": load_seconds,
                    "release_seconds": release_seconds,
                    "transition_seconds": perf_counter() - started,
                    "device_memory_before": before_memory,
                    "device_memory_after": after_memory,
                    "device_used_delta_bytes": _memory_used_delta(before_memory, after_memory),
                    "torch_allocated_delta_bytes": _memory_allocator_delta(
                        before_memory, after_memory, key="torch_allocated_bytes"
                    ),
                    "torch_reserved_delta_bytes": _memory_allocator_delta(
                        before_memory, after_memory, key="torch_reserved_bytes"
                    ),
                }
            )
        return tuple(self._resident[asset_id] for asset_id in requested)

    def evict(self, asset_id: str) -> None:


        if asset_id not in self._resident:
            raise KeyError(f"asset is not resident: {asset_id!r}")
        before_memory = self._memory_snapshot() if self.resource_profile else None
        started = perf_counter()
        self._evict(asset_id)
        after_memory = self._memory_snapshot() if self.resource_profile else None
        self._telemetry_events.append(
            {
                "event": "resident_eviction",
                "requested_asset_ids": [asset_id],
                "loaded_asset_ids": [],
                "released_asset_ids": [asset_id],
                "resident_asset_ids": list(self.resident_asset_ids),
                "resident_asset_count": len(self._resident),
                "resident_owner_bvh_count": self._resident_owner_bvh_count(),
                "resident_triangle_count": self._resident_triangle_count(),
                "load_seconds": 0.0,
                "release_seconds": perf_counter() - started,
                "transition_seconds": perf_counter() - started,
                "device_memory_before": before_memory,
                "device_memory_after": after_memory,
                "device_used_delta_bytes": _memory_used_delta(before_memory, after_memory),
                "torch_allocated_delta_bytes": _memory_allocator_delta(
                    before_memory, after_memory, key="torch_allocated_bytes"
                ),
                "torch_reserved_delta_bytes": _memory_allocator_delta(
                    before_memory, after_memory, key="torch_reserved_bytes"
                ),
            }
        )

    def release_all(self) -> None:


        if not self._resident:
            return
        before_memory = self._memory_snapshot() if self.resource_profile else None
        started = perf_counter()
        released_asset_ids = tuple(self._resident)
        for asset_id in released_asset_ids:
            self._evict(asset_id)
        self._resident_bank_index = None
        after_memory = self._memory_snapshot() if self.resource_profile else None
        self._telemetry_events.append(
            {
                "event": "resident_window_release_all",
                "requested_asset_ids": list(released_asset_ids),
                "loaded_asset_ids": [],
                "released_asset_ids": list(released_asset_ids),
                "resident_asset_ids": [],
                "resident_asset_count": 0,
                "resident_owner_bvh_count": 0,
                "resident_triangle_count": 0,
                "load_seconds": 0.0,
                "release_seconds": perf_counter() - started,
                "transition_seconds": perf_counter() - started,
                "device_memory_before": before_memory,
                "device_memory_after": after_memory,
                "device_used_delta_bytes": _memory_used_delta(before_memory, after_memory),
                "torch_allocated_delta_bytes": _memory_allocator_delta(
                    before_memory, after_memory, key="torch_allocated_bytes"
                ),
                "torch_reserved_delta_bytes": _memory_allocator_delta(
                    before_memory, after_memory, key="torch_reserved_bytes"
                ),
            }
        )

    def drain_telemetry_events(self) -> tuple[dict[str, object], ...]:


        events = tuple(self._telemetry_events)
        self._telemetry_events.clear()
        return events

    def state_dict(self) -> dict[str, object]:
        return {"resident_asset_ids": self.resident_asset_ids, "max_resident_assets": self.max_resident_assets}

    def _evict(self, asset_id: str) -> None:


        state = self._resident.pop(asset_id)
        self.releaser(state)

    def _source_for_asset(self, asset_id: str) -> GeometrySource | GeometrySourceCore:


        if self.source_provider is not None:
            getter = getattr(self.source_provider, "get", None)
            if getter is None:
                raise TypeError("source_provider must expose get(asset_id)")
            return getter(asset_id)
        source = self.catalog[asset_id]
        if source is None:
            raise RuntimeError(f"CPU source provider is missing for eager asset {asset_id!r}")
        return source

    def _resident_owner_bvh_count(self) -> int:


        return sum(len(getattr(state.warp_cache, "handles", ())) for state in self._resident.values())

    def _resident_triangle_count(self) -> int:


        return sum(
            handle.face_count
            for state in self._resident.values()
            for handle in getattr(state.warp_cache, "handles", ())
        )

    def _memory_snapshot(self) -> dict[str, int | None]:


        if not str(self.device).startswith("cuda") or not torch.cuda.is_available():
            return {
                "cuda_free_bytes": None,
                "cuda_total_bytes": None,
                "torch_allocated_bytes": None,
                "torch_reserved_bytes": None,
            }
        torch.cuda.synchronize(self.device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        return {
            "cuda_free_bytes": int(free_bytes),
            "cuda_total_bytes": int(total_bytes),
            "torch_allocated_bytes": int(torch.cuda.memory_allocated(self.device)),
            "torch_reserved_bytes": int(torch.cuda.memory_reserved(self.device)),
        }


def _memory_used_delta(
    before: dict[str, int | None] | None,
    after: dict[str, int | None] | None,
) -> int | None:


    if before is None or after is None:
        return None
    before_free = before["cuda_free_bytes"]
    after_free = after["cuda_free_bytes"]
    if before_free is None or after_free is None:
        return None
    return before_free - after_free


def _memory_allocator_delta(
    before: dict[str, int | None] | None,
    after: dict[str, int | None] | None,
    *,
    key: str,
) -> int | None:


    if before is None or after is None:
        return None
    before_value = before[key]
    after_value = after[key]
    if before_value is None or after_value is None:
        return None
    return after_value - before_value


__all__ = [
    "ResidentGeometryAssetWindow",
]
