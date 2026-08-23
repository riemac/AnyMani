r"""Geometry SSL resident-window 的有界设备资源生命周期。

训练 minibatch 与 Sobol 游标由 ``runtime.sampling`` 和 Method session 拥有。本层只把 CPU
catalog 的指定资产 window 映射到 GPU，并在驱逐时释放 owner BVH lease。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from time import perf_counter

import torch

from anymani.distill.representations.geometry import GeometryRepresentationState
from anymani.distill.representations.sources.geometry_source import GeometrySource, GeometrySourceCore


class ResidentGeometryAssetWindow:
    r"""把 CPU catalog 映射为有界 GPU asset window。

    ``loader`` 在正式运行中是 method 的 ``load_device_state``；参数注入使纯 contract 可以用
    synthetic loader 验证 resident cap 与 eviction，而不启动 Warp/Isaac Sim。
    """

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
        self._resident: dict[str, GeometryRepresentationState] = {}
        self._telemetry_events: list[dict[str, object]] = []  # 窗口/BVH 生命周期的 append-only 事件

    @property
    def resident_asset_ids(self) -> tuple[str, ...]:
        return tuple(self._resident)

    def ensure(
        self,
        asset_ids: tuple[str, ...] | list[str],
        *,
        prefetch_sources: bool = True,
        prepared_sources: Mapping[str, GeometrySource | GeometrySourceCore] | None = None,
    ) -> tuple[GeometryRepresentationState, ...]:
        r"""确保一个完整 resident window 已驻留，并记录 lease/BVH/memory 生命周期证据。

        ``prefetch_sources=False`` 表示 Method 已通过 current/next 双缓冲等待对应 CPU core；
        ``prepared_sources`` 直接 pin 该 current buffer，使 next-buffer LRU 插入不能触发重复物化。
        普通调用保持同步 prefetch 默认行为。
        """

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
        if tuple(self._resident) == requested:
            return tuple(self._resident[asset_id] for asset_id in requested)  # 稳态 minibatch 不同步 CUDA
        before_memory = self._memory_snapshot()  # 切窗前设备 free/allocator 状态
        started = perf_counter()  # 同一进程内的 wall-clock 生命周期起点
        released_asset_ids: list[str] = []  # 本次切窗实际释放的资产
        loaded_asset_ids: list[str] = []  # 本次切窗实际新加载的资产
        release_started = perf_counter()  # release 子阶段起点
        for asset_id in tuple(self._resident):
            if asset_id not in requested:
                self._evict(asset_id)
                released_asset_ids.append(asset_id)
        release_seconds = perf_counter() - release_started  # lease 释放和 registry eviction 时间
        load_started = perf_counter()  # load 子阶段起点
        if prefetch_sources and self.source_provider is not None:
            prefetch_fn = getattr(self.source_provider, "prefetch", None)
            if prefetch_fn is not None:
                prefetch_fn(requested)
        for asset_id in requested:
            if asset_id not in self._resident:
                self._resident[asset_id] = self.loader(
                    prepared_sources[asset_id] if prepared_sources is not None else self._source_for_asset(asset_id),
                    device=self.device,
                    dtype=self.dtype,
                )
                loaded_asset_ids.append(asset_id)
        if len(self._resident) > self.max_resident_assets:
            raise RuntimeError("resident asset window exceeded configured cap")
        load_seconds = perf_counter() - load_started  # CPU->GPU evidence/Warp BVH 构造时间
        if loaded_asset_ids or released_asset_ids:
            after_memory = self._memory_snapshot()  # 切窗后设备状态；同步后才读取 allocator
            self._telemetry_events.append(
                {
                    "event": "resident_window",
                    "requested_asset_ids": list(requested),
                    "loaded_asset_ids": loaded_asset_ids,
                    "released_asset_ids": released_asset_ids,
                    "resident_asset_ids": list(self.resident_asset_ids),
                    "resident_asset_count": len(self._resident),
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
        r"""释放单项 GPU asset lease，并记录独立 eviction 生命周期事件。"""

        if asset_id not in self._resident:
            raise KeyError(f"asset is not resident: {asset_id!r}")
        before_memory = self._memory_snapshot()  # eviction 前 device 状态
        started = perf_counter()  # 单项 release wall-clock 起点
        self._evict(asset_id)  # registry lease 归零后丢弃 device-state 强引用
        after_memory = self._memory_snapshot()  # eviction 后 device 状态
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
        r"""释放当前 window 的全部 lease，并写入一次聚合生命周期事件。"""

        if not self._resident:
            return
        before_memory = self._memory_snapshot()  # 全量释放前 device 状态
        started = perf_counter()  # 全量 release wall-clock 起点
        released_asset_ids = tuple(self._resident)
        for asset_id in released_asset_ids:
            self._evict(asset_id)  # 不为每项制造重复事件，聚合事件保存同一切窗边界
        after_memory = self._memory_snapshot()  # 全量释放后 device 状态
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
        r"""取出自上次 drain 以来的窗口事件，避免 runtime.jsonl 重复写入。"""

        events = tuple(self._telemetry_events)
        self._telemetry_events.clear()
        return events

    def state_dict(self) -> dict[str, object]:
        return {"resident_asset_ids": self.resident_asset_ids, "max_resident_assets": self.max_resident_assets}

    def _evict(self, asset_id: str) -> None:
        """执行不产生额外 telemetry 的底层 eviction。"""

        state = self._resident.pop(asset_id)
        self.releaser(state)

    def _source_for_asset(self, asset_id: str) -> GeometrySource | GeometrySourceCore:
        r"""按 stable ID 取得 CPU source/core；lazy provider 只在此处触发资产加载。"""

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
        """返回当前 window 中真实上传的 owner-local BVH 数。"""

        return sum(len(getattr(state.warp_cache, "handles", ())) for state in self._resident.values())

    def _resident_triangle_count(self) -> int:
        """返回当前 window 中 Warp 实际接收的 float32 有效三角形总数。"""

        return sum(
            handle.face_count
            for state in self._resident.values()
            for handle in getattr(state.warp_cache, "handles", ())
        )

    def _memory_snapshot(self) -> dict[str, int | None]:
        """同步后读取 CUDA free/total 与 PyTorch allocator，明确区分两种记账口径。"""

        if not str(self.device).startswith("cuda") or not torch.cuda.is_available():
            return {
                "cuda_free_bytes": None,
                "cuda_total_bytes": None,
                "torch_allocated_bytes": None,
                "torch_reserved_bytes": None,
            }
        torch.cuda.synchronize(self.device)  # Warp/PyTorch kernel 完成后再采设备状态
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)  # driver 视角的全局 free/total
        return {
            "cuda_free_bytes": int(free_bytes),
            "cuda_total_bytes": int(total_bytes),
            "torch_allocated_bytes": int(torch.cuda.memory_allocated(self.device)),
            "torch_reserved_bytes": int(torch.cuda.memory_reserved(self.device)),
        }


def _memory_used_delta(
    before: dict[str, int | None],
    after: dict[str, int | None],
) -> int | None:
    r"""由 CUDA free-memory 差得到本次窗口转换的设备已用字节增量。"""

    before_free = before["cuda_free_bytes"]  # driver 口径，包含 Warp 与 PyTorch
    after_free = after["cuda_free_bytes"]  # 同一同步边界后的 driver free bytes
    if before_free is None or after_free is None:
        return None
    return before_free - after_free  # 正值表示转换后设备占用增加


def _memory_allocator_delta(
    before: dict[str, int | None],
    after: dict[str, int | None],
    *,
    key: str,
) -> int | None:
    r"""计算 PyTorch caching allocator 指定字段的窗口转换增量。"""

    before_value = before[key]
    after_value = after[key]
    if before_value is None or after_value is None:
        return None
    return after_value - before_value  # 正值表示 PyTorch allocator 占用增加


__all__ = [
    "ResidentGeometryAssetWindow",
]
