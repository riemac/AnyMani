"""Lazy geometry sources, sessions, and source-artifact lifecycle."""


from __future__ import annotations

import math
import shutil
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event
from time import perf_counter
from typing import Any, overload

import torch

from anymani.assets.bank.hand_container import HandContainer
from anymani.distill.models.input_adapters.geometry import GeometryPaddingCfg
from anymani.distill.representations.geometry import GeometryRepresentation
from anymani.distill.representations.sources.artifacts import GeometrySourceArtifactStore
from anymani.distill.representations.sources.cache import GeometrySourceArena
from anymani.distill.representations.sources.geometry_source import GeometrySource, GeometrySourceCore
from anymani.distill.representations.sources.kinematics import lower_hand_geometry_semantics

from .batch import PaddedOnlineGeometryBatch
from .state_measure import SobolJointSampler


def _derive_padding(assets: Sequence[HandContainer], *, max_graph_distance: int) -> GeometryPaddingCfg:


    if not assets:
        raise ValueError("padding derivation requires at least one materialized source")
    max_joint = 0
    max_tip = 0
    for asset in assets:
        semantics = asset.geometry_semantics
        if semantics is None:
            raise ValueError(f"asset {asset.asset_id!r} is missing geometry semantics")
        max_joint = max(max_joint, len(semantics.active_joint_names))
        max_tip = max(max_tip, sum(owner.role == "tip" for owner in semantics.owners))
    if max_joint < 1 or max_tip < 1:
        raise ValueError("resolved dataset must contain at least one JOINT and one TIP owner")
    return GeometryPaddingCfg(
        max_joint_count=max_joint,
        max_tip_count=max_tip,
        max_graph_distance=max_graph_distance,
    )


@dataclass(frozen=True)
class SourcePrefetchHandle:


    asset_ids: tuple[str, ...]
    futures: tuple[Future[GeometrySourceCore], ...]
    started: float


class LazyGeometrySources(Sequence[GeometrySourceCore]):


    def __init__(
        self,
        assets: Sequence[HandContainer],
        *,
        cache: GeometrySourceArena,
        config: Any,
        materialize: Callable[[HandContainer], GeometrySourceCore],
    ) -> None:


        self.assets = tuple(assets)
        self.asset_ids = tuple(asset.asset_id for asset in self.assets)
        if len(set(self.asset_ids)) != len(self.asset_ids):
            raise ValueError("lazy geometry source asset IDs must be unique")
        self.cache = cache  # 16-entry/512 MiB CPU core arena
        self.config = config
        self.materialize = materialize
        self._index_by_id = {asset_id: index for index, asset_id in enumerate(self.asset_ids)}
        self._prefetch_executor: ThreadPoolExecutor | None = None
        self._prefetch_stats: dict[str, int | float] = {
            "subwindow_count": 0,
            "asset_count": 0,
            "ready_latency_seconds": 0.0,
            "blocked_wait_seconds": 0.0,
        }
        self._ready_latencies: list[float] = []
        self._blocked_waits: list[float] = []

    def __len__(self) -> int:


        return len(self.assets)

    @overload
    def __getitem__(self, index: int) -> GeometrySourceCore: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[GeometrySourceCore, ...]: ...

    def __getitem__(self, index: int | slice) -> GeometrySourceCore | tuple[GeometrySourceCore, ...]:


        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        asset = self.assets[index]
        core = self.cache.load_or_create(
            asset,
            config=self.config,
            materialize=lambda asset=asset: self.materialize(asset),
        )
        if not isinstance(core, GeometrySourceCore):
            raise TypeError("training source arena returned a finalized source instead of GeometrySourceCore")
        return core

    def get(self, asset_id: str) -> GeometrySourceCore:


        try:
            return self[self._index_by_id[asset_id]]
        except KeyError as exc:
            raise KeyError(f"unknown geometry asset ID={asset_id!r}") from exc

    def prefetch_async(self, asset_ids: Sequence[str]) -> SourcePrefetchHandle:


        requested = tuple(asset_ids)
        started = perf_counter()
        if not requested:
            return SourcePrefetchHandle(requested, (), started)
        if self._prefetch_executor is None:
            self._prefetch_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="source-prefetch")
        futures = tuple(self._prefetch_executor.submit(self.get, asset_id) for asset_id in requested)
        return SourcePrefetchHandle(requested, futures, started)

    def await_prefetch(self, handle: SourcePrefetchHandle) -> tuple[GeometrySourceCore, ...]:


        wait_started = perf_counter()
        cores = [future.result() for future in handle.futures]
        completed = perf_counter()
        ready_latency = completed - handle.started
        blocked_wait = completed - wait_started
        self._prefetch_stats["subwindow_count"] = int(self._prefetch_stats["subwindow_count"]) + 1
        self._prefetch_stats["asset_count"] = int(self._prefetch_stats["asset_count"]) + len(handle.asset_ids)
        self._prefetch_stats["ready_latency_seconds"] = (
            float(self._prefetch_stats["ready_latency_seconds"]) + ready_latency
        )
        self._prefetch_stats["blocked_wait_seconds"] = (
            float(self._prefetch_stats["blocked_wait_seconds"]) + blocked_wait
        )
        self._ready_latencies.append(ready_latency)
        self._blocked_waits.append(blocked_wait)
        return tuple(cores)

    def prefetch_stats(self) -> dict[str, int | float]:


        evidence = dict(self._prefetch_stats)
        if self._ready_latencies:
            rank = math.ceil(0.95 * len(self._ready_latencies)) - 1
            evidence["ready_latency_p95_seconds"] = sorted(self._ready_latencies)[rank]
            evidence["blocked_wait_p95_seconds"] = sorted(self._blocked_waits)[rank]
        return evidence

    def prefetch(self, asset_ids: Sequence[str]) -> None:


        self.await_prefetch(self.prefetch_async(asset_ids))

    def close(self) -> None:


        if self._prefetch_executor is not None:
            self._prefetch_executor.shutdown(wait=True, cancel_futures=True)
            self._prefetch_executor = None


class LazySobolSamplers(Sequence[SobolJointSampler]):


    def __init__(self, sources: LazyGeometrySources, *, seed: int) -> None:


        self.sources = sources
        self.seed = int(seed)
        self._samplers: dict[int, SobolJointSampler] = {}

    def __len__(self) -> int:


        return len(self.sources)

    @overload
    def __getitem__(self, index: int) -> SobolJointSampler: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[SobolJointSampler, ...]: ...

    def __getitem__(self, index: int | slice) -> SobolJointSampler | tuple[SobolJointSampler, ...]:


        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        sampler = self._samplers.get(index)
        if sampler is None:
            semantics = self.sources.assets[index].geometry_semantics
            if semantics is None:
                raise ValueError(f"asset {self.sources.asset_ids[index]!r} is missing geometry semantics")
            spec = lower_hand_geometry_semantics(semantics, dtype=torch.float64)
            sampler = SobolJointSampler(spec, seed=self.seed + index)
            self._samplers[index] = sampler
        return sampler

    def clear(self) -> None:


        self._samplers.clear()

    def state_dict(self) -> dict[str, dict[str, int]]:


        return {str(index): sampler.state_dict() for index, sampler in sorted(self._samplers.items())}

    def load_state_dict(self, states: Mapping[str, object]) -> None:


        for raw_index, state in states.items():
            if not isinstance(raw_index, str) or not raw_index.isdigit():
                raise ValueError("sparse Sobol sampler state keys must be decimal asset indices")
            index = int(raw_index)
            if not 0 <= index < len(self):
                raise ValueError("sparse Sobol sampler state contains out-of-range asset index")
            if not isinstance(state, Mapping):
                raise ValueError("lazy sampler state must be a mapping")
            if not all(isinstance(key, str) and isinstance(value, int) for key, value in state.items()):
                raise ValueError("lazy sampler state keys/values must be str/int")
            self[index].load_state_dict({str(key): int(value) for key, value in state.items()})


class PhysicalAuditHandle:


    def __init__(
        self,
        future: Future[dict[str, Any]],
        executor: ThreadPoolExecutor,
        cancel_event: Event,
    ) -> None:


        self._future = future
        self._executor = executor
        self._cancel_event = cancel_event
        self._result: dict[str, Any] | None = None
        self._closed = False

    def wait(self) -> dict[str, Any]:


        if self._result is None:
            try:
                self._result = self._future.result()
            finally:
                self._executor.shutdown(wait=True)
                self._closed = True
        if self._result is None:
            raise RuntimeError("physical asset audit completed without a manifest")
        return self._result

    def cancel(self) -> None:


        if self._closed:
            return
        self._cancel_event.set()
        self._future.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._closed = True


class MultiAnchorGaussianSession:


    def __init__(
        self,
        method: Any,
        *,
        role: str,
        suite: str,
        sources: LazyGeometrySources,
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
        max_resident_assets: int,
        window_factory: Any,
        resource_profile: bool = False,
    ) -> None:


        if not sources:
            raise ValueError(f"method session role={role!r} suite={suite!r} requires at least one asset")
        self.method = method  # concrete scientific aggregation root
        self.role = role  # train/evaluation
        self.suite = suite
        self.sources = sources
        self.seed = int(seed)
        self.samplers = method.make_independent_samplers(sources, seed=self.seed)
        loader = method.load_device_state if role == "train" else method.load_fixed_device_state
        self.window = window_factory(
            sources,
            device=str(device),
            dtype=dtype,
            max_resident_assets=min(max_resident_assets, len(sources)),
            loader=loader,
            catalog_ids=sources.asset_ids,
            source_provider=sources,
            resource_profile=resource_profile,
        )

    @property
    def asset_count(self) -> int:


        return len(self.sources)

    def realize(self, schedule_item: Any, *, schedule: Any, step: int) -> PaddedOnlineGeometryBatch:


        del step
        return self.method.realize_minibatch(
            schedule_item,
            sources=self.sources,
            samplers=self.samplers,
            window=self.window,
            seed=self.seed,
            schedule=schedule,
            mode="train" if self.role == "train" else "eval",
        )

    def realize_units(self, schedule_item: Any, *, schedule: Any, step: int):


        del step
        return self.method.realize_minibatch_units(
            schedule_item,
            sources=self.sources,
            samplers=self.samplers,
            window=self.window,
            seed=self.seed,
            schedule=schedule,
            mode="train" if self.role == "train" else "eval",
        )

    def state_dict(self) -> dict[str, object]:


        return {"asset_ids": self.sources.asset_ids, "samplers": self.samplers.state_dict()}

    def load_state_dict(self, state: Mapping[str, object]) -> None:


        raw_asset_ids = state.get("asset_ids")
        if not isinstance(raw_asset_ids, (tuple, list)) or tuple(raw_asset_ids) != self.sources.asset_ids:
            raise ValueError("method session checkpoint asset axis does not match current split")
        raw_samplers = state.get("samplers")
        if not isinstance(raw_samplers, Mapping):
            raise ValueError("checkpoint lacks method session sparse samplers")
        self.samplers.load_state_dict(raw_samplers)

    def close(self) -> None:


        self.window.release_all()
        self.samplers.clear()

    def drain_runtime_events(self) -> tuple[dict[str, object], ...]:


        return self.window.drain_telemetry_events()


def configure_source_artifacts(
    method: Any,
    *,
    root: str,
    mode: str,
    dataset_manifest_sha256: str,
    producer_device: str,
    role: str = "train",
) -> None:


    method.source_artifact_store = (
        None
        if mode == "off"
        else GeometrySourceArtifactStore(
            root,
            mode=mode,
            dataset_manifest_sha256=dataset_manifest_sha256,
            producer_device=producer_device,
            role=role,
        )
    )


def source_artifact_identity(method: Any) -> dict[str, object]:


    store = method.source_artifact_store
    return {"schema_version": "2.0.0", "mode": "off"} if store is None else store.identity()


def materialize_or_load_core(
    method: Any,
    container: HandContainer,
    representation: GeometryRepresentation,
) -> GeometrySourceCore:


    store = method.source_artifact_store
    if store is None:
        return representation.materialize_core(container)
    try:
        core, reference = store.load_base(container, method.config.representation.source)
    except (FileNotFoundError, ValueError):
        if store.mode != "read-write":
            raise
        built = representation.materialize_core(container)
        store.write_base(built, method.config.representation.source)
        core, reference = store.load_base(container, method.config.representation.source)
    with method._source_artifact_lock:
        method._base_artifact_refs[container.asset_id] = reference
    return core


def lazy_sources(
    method: Any,
    assets: Sequence[HandContainer],
    representation: GeometryRepresentation,
) -> LazyGeometrySources:


    return LazyGeometrySources(
        assets,
        cache=method.source_cache,
        config=method.config.representation.source,
        materialize=lambda container: materialize_or_load_core(method, container, representation),
    )


def source_partitions(method: Any) -> dict[str, tuple[LazyGeometrySources, int]]:


    if method.active_role == "train":
        return {"train": (require_train_sources(method), method.config.representation.source.anchors.bank_size)}
    if method.active_role == "evaluation":
        return {f"evaluation.{name}": (source, 1) for name, source in method.evaluation_sources.items()}
    raise RuntimeError("source partitions requested before method role preparation")


def prepare_source_artifacts(
    method: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
    partitions: tuple[str, ...] = (),
) -> dict[str, object]:


    store = method.source_artifact_store
    if store is None or store.mode != "read-write":
        raise RuntimeError("prepare_source_artifacts requires a configured read-write store")
    selected = set(partitions)
    available = source_partitions(method)
    unknown = selected - available.keys()
    if unknown:
        raise ValueError(f"unknown source preparation partitions: {sorted(unknown)}")
    providers = {name: value for name, value in available.items() if not selected or name in selected}
    started = perf_counter()
    base_count = 0
    shard_count = 0
    for _partition, (sources, bank_count) in providers.items():
        for asset_index in range(len(sources)):
            core = sources[asset_index]
            base_count += 1
            for bank_index in range(bank_count):
                state = method._load_device_state_with_artifact(
                    core,
                    representation=method.representation,
                    bank_index=bank_index,
                    device=device,
                    dtype=dtype,
                )
                state.device_source.release()
                shard_count += 1
    disk = shutil.disk_usage(store.root.parent if store.root.parent.exists() else Path.cwd())
    return {
        "schema_version": "2.0.0",
        "source_artifact_schema": "2.0.0",
        "root": str(store.root),
        "partitions": sorted(providers),
        "base_count": base_count,
        "anchor_shard_count": shard_count,
        "elapsed_seconds": perf_counter() - started,
        "disk_free_bytes": disk.free,
    }


def preflight_source_artifacts(method: Any) -> dict[str, int]:


    store = method.source_artifact_store
    if store is None:
        return {"base_count": 0, "anchor_shard_count": 0}
    base_count = 0
    shard_count = 0
    for _partition, (sources, bank_count) in source_partitions(method).items():
        for container in sources.assets:
            store.load_base(container, method.config.representation.source)
            base_count += 1
            for bank_index in range(bank_count):
                store.load_anchor(container, method.config.representation.source, bank_index)
                shard_count += 1
    return {"base_count": base_count, "anchor_shard_count": shard_count}


def split_names(method: Any, role: str) -> tuple[str, ...]:


    if role == "train":
        return ("",)
    if role == "evaluation":
        return tuple(method.evaluation_sources)
    raise ValueError(f"unknown method split role={role!r}")


def require_train_sources(method: Any) -> LazyGeometrySources:


    if method.train_sources is None:
        raise RuntimeError("multi-anchor method train sources have not been prepared")
    return method.train_sources


def split_asset_count(method: Any, role: str, *, suite: str = "") -> int:


    if role == "train":
        return len(require_train_sources(method))
    if role == "evaluation":
        return len(method.evaluation_sources.get(suite, ()))
    raise ValueError(f"unknown method split role={role!r}")


def asset_manifest(method: Any, catalog: Any, *, cancel_event: Event | None = None) -> dict[str, Any]:


    from .provenance import (
        anchor_realization_record,
        home_surface_realization_record,
        validate_asset_manifest_isolation,
    )

    def record(asset: Any, source: GeometrySource, *, partition: str, provenance: Any) -> dict[str, Any]:


        semantics = asset.geometry_semantics
        if semantics is None:
            raise ValueError(f"asset {asset.asset_id!r} is missing geometry semantics")
        identity = source.identity
        return {
            "asset_id": asset.asset_id,
            "content_hash": semantics.content_hash,
            "physical_geometry_hash": identity.physical_geometry_hash,
            "configuration_domain_hash": identity.configuration_domain_hash,
            "partition": partition,
            "source_kind": semantics.source_kind,
            "topology_key": semantics.topology_key or "",
            "family": semantics.family,
            "handedness": semantics.handedness,
            "joint_count": len(semantics.active_joint_names),
            "owner_count": len(semantics.owners),
            **anchor_realization_record(source.anchors),
            **home_surface_realization_record(source.home_surface, source.geometry_cache),
            **(asdict(provenance) if hasattr(provenance, "__dataclass_fields__") else dict(provenance)),
        }

    def record_item(item: Any, *, partition: str, representation: GeometryRepresentation) -> dict[str, Any]:


        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("physical asset audit cancelled before completion")
        return record(
            item.container,
            representation.materialize_source(
                item.container,
                anchor_device=str(method.runtime_device) if method.runtime_device is not None else "cpu",
            ),
            partition=partition,
            provenance=item.provenance,
        )

    manifest = {
        "schema_version": "4.0.0",
        "dataset_source_path": str(catalog.dataset.source_path),
        "dataset_source_sha256": catalog.dataset.source_sha256,
        "train": [
            record_item(item, partition="train", representation=method.representation)
            for item in catalog.dataset.train.records
        ],
        "evaluation": {
            suite: [
                record_item(item, partition=f"evaluation.{suite}", representation=method.fixed_representation)
                for item in partition.records
            ]
            for suite, partition in catalog.dataset.evaluation.items()
        },
    }
    validate_asset_manifest_isolation(manifest)
    return manifest


def start_physical_audit(method: Any, catalog: Any) -> PhysicalAuditHandle:


    cancel_event = Event()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ssl-physical-audit")
    future = executor.submit(asset_manifest, method, catalog, cancel_event=cancel_event)
    return PhysicalAuditHandle(future, executor, cancel_event)


__all__ = [
    "LazyGeometrySources",
    "LazySobolSamplers",
    "MultiAnchorGaussianSession",
    "PhysicalAuditHandle",
    "SourcePrefetchHandle",
    "_derive_padding",
    "asset_manifest",
    "configure_source_artifacts",
    "lazy_sources",
    "materialize_or_load_core",
    "preflight_source_artifacts",
    "prepare_source_artifacts",
    "require_train_sources",
    "source_artifact_identity",
    "source_partitions",
    "split_asset_count",
    "split_names",
    "start_physical_audit",
]
