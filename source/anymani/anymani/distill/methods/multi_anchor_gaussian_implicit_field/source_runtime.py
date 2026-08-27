r"""Geometry SSL 的 lazy source、稀疏 Sobol 与 split-session 生命周期。

本模块拥有训练数据的运行时资源边界，而不拥有模型、objective 或 FairGrad 数学：
``LazyGeometrySources`` 把固定资产轴映射为受 arena 约束的按需 CPU core；current/next 双 worker
prefetch 只重叠 CPU source 构建与当前 GPU subwindow；``LazySobolSamplers`` 只保存已访问资产的
cursor；``MultiAnchorGaussianSession`` 封装 source、sampler 与 resident Warp lease。

资产、q-block、Sobol cursor 与 resident bank 的顺序共同定义训练随机轨迹；性能重排不得改变这些身份。
"""

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
    r"""由 typed semantics 的活动 JOINT/TIP 轴推导全 catalog padding。

    此扫描不构造 float64 POE 或 graph 张量；collision component、home pose、screw 与 joint limits 不改变
    离散轴长度。当前 8192-train preset 因而只需整数扫描即可恢复 $N_J^{max}$ 与 $N_{tip}^{max}$。
    """

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
    r"""下一 8-asset CPU core buffer 的 futures 与提交时刻。"""

    asset_ids: tuple[str, ...]  # 与 logical asset chunk 同序的稳定 ID
    futures: tuple[Future[GeometrySourceCore], ...]  # 两个 worker 共享 arena 的逐资产结果
    started: float  # ``perf_counter`` 提交时刻，用于可消费延迟证据


class LazyGeometrySources(Sequence[GeometrySourceCore]):
    r"""把固定 HandContainer 轴映射为按资产 demand-load 的 CPU source-core 轴。

    只有显式索引或 slice 会触发 source arena 读取/物化。Provider 不持有历史 source dict，因而完整
    catalog 的访问量不会越过 arena 的 16 项/512 MiB 上限。
    """

    def __init__(
        self,
        assets: Sequence[HandContainer],
        *,
        cache: GeometrySourceArena,
        config: Any,
        materialize: Callable[[HandContainer], GeometrySourceCore],
    ) -> None:
        r"""保存轻量资产轴和 source 构造函数，不执行资产几何 IO。"""

        self.assets = tuple(assets)  # catalog 已冻结的资产轴
        self.asset_ids = tuple(asset.asset_id for asset in self.assets)  # schedule 共用的稳定身份
        if len(set(self.asset_ids)) != len(self.asset_ids):
            raise ValueError("lazy geometry source asset IDs must be unique")
        self.cache = cache  # 16-entry/512 MiB CPU core arena
        self.config = config  # q-independent source realization 配置
        self.materialize = materialize  # cache miss 的唯一构造入口
        self._index_by_id = {asset_id: index for index, asset_id in enumerate(self.asset_ids)}
        self._prefetch_executor: ThreadPoolExecutor | None = None  # 首次请求时才创建两个 CPU workers
        self._prefetch_stats: dict[str, int | float] = {
            "subwindow_count": 0,
            "asset_count": 0,
            "ready_latency_seconds": 0.0,
            "blocked_wait_seconds": 0.0,
        }  # current/next pipeline 的累计服务时间与主线程阻塞时间
        self._ready_latencies: list[float] = []  # 每个 8-asset subwindow 的 submit→ready wall time，s
        self._blocked_waits: list[float] = []  # 主线程 await 的未重叠尾延迟，s

    def __len__(self) -> int:
        r"""返回不触发 source 物化的资产数量。"""

        return len(self.assets)

    @overload
    def __getitem__(self, index: int) -> GeometrySourceCore: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[GeometrySourceCore, ...]: ...

    def __getitem__(self, index: int | slice) -> GeometrySourceCore | tuple[GeometrySourceCore, ...]:
        r"""按索引读取 source core；slice 只展开对应范围。"""

        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        asset = self.assets[index]  # 当前固定 catalog row
        core = self.cache.load_or_create(
            asset,
            config=self.config,
            materialize=lambda asset=asset: self.materialize(asset),
        )
        if not isinstance(core, GeometrySourceCore):
            raise TypeError("training source arena returned a finalized source instead of GeometrySourceCore")
        return core

    def get(self, asset_id: str) -> GeometrySourceCore:
        r"""按稳定 asset ID 读取 source core，供 resident window demand-load。"""

        try:
            return self[self._index_by_id[asset_id]]
        except KeyError as exc:
            raise KeyError(f"unknown geometry asset ID={asset_id!r}") from exc

    def prefetch_async(self, asset_ids: Sequence[str]) -> SourcePrefetchHandle:
        r"""异步准备下一 device subwindow 的 CPU core，不等待 GPU 当前组完成。"""

        requested = tuple(asset_ids)  # 保持 logical asset chunk 顺序
        started = perf_counter()  # submit 到可消费的 wall-time 原点
        if not requested:
            return SourcePrefetchHandle(requested, (), started)
        if self._prefetch_executor is None:
            self._prefetch_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="source-prefetch")
        futures = tuple(self._prefetch_executor.submit(self.get, asset_id) for asset_id in requested)
        return SourcePrefetchHandle(requested, futures, started)

    def await_prefetch(self, handle: SourcePrefetchHandle) -> tuple[GeometrySourceCore, ...]:
        r"""等待并返回 current core buffer，同时记录可重叠与未重叠延迟。"""

        wait_started = perf_counter()  # next-buffer 已完成时应接近零
        cores = [future.result() for future in handle.futures]  # submission order 与 asset_ids 一致
        completed = perf_counter()
        ready_latency = completed - handle.started  # 完整 source subwindow 服务时间，s
        blocked_wait = completed - wait_started  # GPU 当前组未覆盖的 CPU 尾部，s
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
        r"""返回 current/next core pipeline 的累计 timing 与预算证据。"""

        evidence = dict(self._prefetch_stats)
        if self._ready_latencies:
            rank = math.ceil(0.95 * len(self._ready_latencies)) - 1  # nearest-rank $P_{95}$
            evidence["ready_latency_p95_seconds"] = sorted(self._ready_latencies)[rank]
            evidence["blocked_wait_p95_seconds"] = sorted(self._blocked_waits)[rank]
        return evidence

    def prefetch(self, asset_ids: Sequence[str]) -> None:
        r"""同步兼容入口；无外层流水时提交并等待同一 core buffer。"""

        self.await_prefetch(self.prefetch_async(asset_ids))

    def close(self) -> None:
        r"""等待已提交工作并释放 executor；arena 生命周期由 Method 管理。"""

        if self._prefetch_executor is not None:
            self._prefetch_executor.shutdown(wait=True, cancel_futures=True)
            self._prefetch_executor = None


class LazySobolSamplers(Sequence[SobolJointSampler]):
    r"""按资产 typed semantics 延迟构造独立 Sobol joint-limit sampler。

    Sampler 只读取 joint limits；不得通过 ``GeometrySource.spec_cpu`` 取 limits，否则 epoch-0 checkpoint
    会隐式物化完整 catalog 的 owner mesh。
    """

    def __init__(self, sources: LazyGeometrySources, *, seed: int) -> None:
        r"""保存轻量资产轴和 root seed，不提前 lower 全部 sampler。"""

        self.sources = sources  # 与 schedule 相同的固定 asset axis
        self.seed = int(seed)  # 每资产实际 seed 为 root + asset index
        self._samplers: dict[int, SobolJointSampler] = {}  # 只保存已访问 rows

    def __len__(self) -> int:
        r"""返回固定资产轴长度。"""

        return len(self.sources)

    @overload
    def __getitem__(self, index: int) -> SobolJointSampler: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[SobolJointSampler, ...]: ...

    def __getitem__(self, index: int | slice) -> SobolJointSampler | tuple[SobolJointSampler, ...]:
        r"""按资产索引构造或返回 Sobol sampler。"""

        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        sampler = self._samplers.get(index)
        if sampler is None:
            semantics = self.sources.assets[index].geometry_semantics
            if semantics is None:
                raise ValueError(f"asset {self.sources.asset_ids[index]!r} is missing geometry semantics")
            spec = lower_hand_geometry_semantics(semantics, dtype=torch.float64)  # limits/axis 的 CPU FP64 真源
            sampler = SobolJointSampler(spec, seed=self.seed + index)
            self._samplers[index] = sampler
        return sampler

    def clear(self) -> None:
        r"""释放 Sobol engines；cursor 可由 seed + sparse checkpoint state 重建。"""

        self._samplers.clear()

    def state_dict(self) -> dict[str, dict[str, int]]:
        r"""只保存已访问资产 cursor，不为 epoch-0 物化 8192 个 engines。"""

        return {str(index): sampler.state_dict() for index, sampler in sorted(self._samplers.items())}

    def load_state_dict(self, states: Mapping[str, object]) -> None:
        r"""按稳定资产索引恢复 sparse cursor；未来资产仍由 seed 从 cursor 0 构造。"""

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
    r"""后台完整 physical asset manifest 的可等待、可协作取消句柄。"""

    def __init__(
        self,
        future: Future[dict[str, Any]],
        executor: ThreadPoolExecutor,
        cancel_event: Event,
    ) -> None:
        r"""保存后台任务、executor 与协作取消事件。"""

        self._future = future
        self._executor = executor
        self._cancel_event = cancel_event
        self._result: dict[str, Any] | None = None
        self._closed = False

    def wait(self) -> dict[str, Any]:
        r"""等待完整 manifest，通过异常传播保证 audit gate fail-closed。"""

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
        r"""协作停止未发布 audit，并等待当前单项 source 物化退出。"""

        if self._closed:
            return
        self._cancel_event.set()
        self._future.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._closed = True


class MultiAnchorGaussianSession:
    r"""封装一个 train/validation/evaluation split 的 source、Sobol cursor 与 resident window。"""

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
        r"""建立独立 sampler 与 device window；Trainer 不读取底层数组。"""

        if not sources:
            raise ValueError(f"method session role={role!r} suite={suite!r} requires at least one asset")
        self.method = method  # concrete scientific aggregation root
        self.role = role  # train/training_evaluation/validation/evaluation
        self.suite = suite  # held-out suite 名；train 为空
        self.sources = sources  # 当前 split 的固定 lazy asset axis
        self.seed = int(seed)  # 当前 split 独立 q/query root seed
        self.samplers = method.make_independent_samplers(sources, seed=self.seed)
        loader = method.load_device_state if role == "train" else method.load_validation_device_state
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
        r"""返回当前 split 的真实资产数。"""

        return len(self.sources)

    def realize(self, schedule_item: Any, *, schedule: Any, step: int) -> PaddedOnlineGeometryBatch:
        r"""按离散 schedule item realization opaque geometry batch。"""

        del step  # 采样 identity 由 schedule cursor 与 session seed 唯一确定
        return self.method.realize_minibatch(
            schedule_item,
            sources=self.sources,
            samplers=self.samplers,
            window=self.window,
            seed=self.seed,
            schedule=schedule,
            mode="train" if self.role == "train" else "eval",
        )

    def state_dict(self) -> dict[str, object]:
        r"""返回资产轴和 sparse Sobol cursors，供 optimizer-boundary checkpoint 保存。"""

        return {"asset_ids": self.sources.asset_ids, "samplers": self.samplers.state_dict()}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        r"""严格恢复同一资产轴上的 sparse Sobol cursors。"""

        raw_asset_ids = state.get("asset_ids")
        if not isinstance(raw_asset_ids, (tuple, list)) or tuple(raw_asset_ids) != self.sources.asset_ids:
            raise ValueError("method session checkpoint asset axis does not match current split")
        raw_samplers = state.get("samplers")
        if not isinstance(raw_samplers, Mapping):
            raise ValueError("checkpoint lacks method session sparse samplers")
        self.samplers.load_state_dict(raw_samplers)

    def close(self) -> None:
        r"""幂等释放 resident Warp leases、device state 与 Sobol engines。"""

        self.window.release_all()
        self.samplers.clear()

    def drain_runtime_events(self) -> tuple[dict[str, object], ...]:
        r"""把 resident/source 生命周期事件交给 Trainer 的 append-only runtime logger。"""

        return self.window.drain_telemetry_events()


def configure_source_artifacts(
    method: Any,
    *,
    root: str,
    mode: str,
    dataset_manifest_sha256: str,
    producer_device: str,
) -> None:
    r"""配置跨 run source store；``off`` 显式关闭磁盘 artifact。"""

    method.source_artifact_store = (
        None
        if mode == "off"
        else GeometrySourceArtifactStore(
            root,
            mode=mode,
            dataset_manifest_sha256=dataset_manifest_sha256,
            producer_device=producer_device,
        )
    )


def source_artifact_identity(method: Any) -> dict[str, object]:
    r"""返回 checkpoint/stage 可比较的 source producer 身份。"""

    store = method.source_artifact_store
    return {"schema_version": "1.0.0", "mode": "off"} if store is None else store.identity()


def materialize_or_load_core(
    method: Any,
    container: HandContainer,
    representation: GeometryRepresentation,
) -> GeometrySourceCore:
    r"""按 cache mode 加载 base；仅 read-write miss/corruption 可物化并原子发布。"""

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
    r"""建立不触发 source IO 的固定资产 provider。"""

    return LazyGeometrySources(
        assets,
        cache=method.source_cache,
        config=method.config.representation.source,
        materialize=lambda container: materialize_or_load_core(method, container, representation),
    )


def source_partitions(method: Any) -> dict[str, tuple[LazyGeometrySources, int]]:
    r"""返回 prepare/preflight 所需 provider 与 anchor shard 数。"""

    partitions: dict[str, tuple[LazyGeometrySources, int]] = {
        "train": (require_train_sources(method), method.config.representation.source.anchors.bank_size),
    }
    partitions.update({f"validation.{name}": (source, 1) for name, source in method.validation_sources.items()})
    partitions.update({f"evaluation.{name}": (source, 1) for name, source in method.evaluation_sources.items()})
    return partitions


def prepare_source_artifacts(
    method: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
    partitions: tuple[str, ...] = (),
) -> dict[str, object]:
    r"""离线构建 base 与 train 0..7 / held-out bank-0 shards，并报告 cold-build 成本。"""

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
            core = sources[asset_index]  # read-write miss 构建并发布 base
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
        "schema_version": "1.0.0",
        "source_artifact_schema": "1.0.0",
        "root": str(store.root),
        "partitions": sorted(providers),
        "base_count": base_count,
        "anchor_shard_count": shard_count,
        "elapsed_seconds": perf_counter() - started,
        "disk_free_bytes": disk.free,
    }


def preflight_source_artifacts(method: Any) -> dict[str, int]:
    r"""模型初始化前校验 formal run 的全部 base/shards；任一 miss/corruption fail closed。"""

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
    r"""返回 train 或 validation/evaluation 具名 suite 轴。"""

    if role in {"train", "training_evaluation"}:
        return ("",)
    if role == "validation":
        return tuple(method.validation_sources)
    if role == "evaluation":
        return tuple(method.evaluation_sources)
    raise ValueError(f"unknown method split role={role!r}")


def require_train_sources(method: Any) -> LazyGeometrySources:
    r"""返回 prepare 后的 train provider；生命周期错误时 fail-fast。"""

    if method.train_sources is None:
        raise RuntimeError("multi-anchor method train sources have not been prepared")
    return method.train_sources


def split_asset_count(method: Any, role: str, *, suite: str = "") -> int:
    r"""返回 train 或具名 held-out suite 的资产数。"""

    if role in {"train", "training_evaluation"}:
        return len(require_train_sources(method))
    if role == "validation":
        return len(method.validation_sources.get(suite, ()))
    if role == "evaluation":
        return len(method.evaluation_sources.get(suite, ()))
    raise ValueError(f"unknown method split role={role!r}")


def asset_manifest(method: Any, catalog: Any, *, cancel_event: Event | None = None) -> dict[str, Any]:
    r"""流式记录 physical source 与 train/held-out 隔离证据。

    Audit 不遍历训练用 provider，避免完整 catalog 污染 CPU arena 或与 device prefetch 争用。
    每项 source 在 record 后释放，只保留 hash/provenance mapping。
    """

    from .provenance import (
        anchor_realization_record,
        home_surface_realization_record,
        validate_asset_manifest_isolation,
    )

    def record(asset: Any, source: GeometrySource, *, partition: str, provenance: Any) -> dict[str, Any]:
        r"""把 materialized source 规约为稳定物理与采样 provenance。"""

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
        r"""在每项昂贵 source 构造前检查 cooperative cancellation。"""

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
        "validation": {
            suite: [
                record_item(item, partition=f"validation.{suite}", representation=method.validation_representation)
                for item in partition.records
            ]
            for suite, partition in catalog.dataset.validation.items()
        },
        "evaluation": {
            suite: [
                record_item(item, partition=f"evaluation.{suite}", representation=method.validation_representation)
                for item in partition.records
            ]
            for suite, partition in catalog.dataset.evaluation.items()
        },
    }
    validate_asset_manifest_isolation(manifest)
    return manifest


def start_physical_audit(method: Any, catalog: Any) -> PhysicalAuditHandle:
    r"""后台计算完整 physical manifest；checkpoint/artifact lineage gate 等待结果。"""

    cancel_event = Event()  # teardown 可在相邻资产 source 构造之间 cooperative-cancel
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
