r"""多锚点 Gaussian 隐式场的科学聚合根。

本类对内显式耦合 representation、model 与 objectives，对外只给 SSL trainer 封闭接口：
prepare、realize_minibatch、forward/backward update、reduce、evaluate、retained export。
Trainer 不得读取 `representation.config.field` 或 padding layout。
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from threading import Event
from time import perf_counter
from typing import Any, overload

import torch

from anymani.assets.bank.hand_container import HandContainer
from anymani.distill.methods.contracts import FeatureSpec, MethodEvaluationReport, MethodStep, MethodUpdate
from anymani.distill.models.geometry_ssl import GeometrySSLModel
from anymani.distill.models.input_adapters.geometry import GeometryPaddingCfg
from anymani.distill.objectives.contracts import AdditiveStatistic, ObjectiveTermResult
from anymani.distill.representations.geometry import GeometryRepresentation
from anymani.distill.representations.sources.cache import GeometrySourceArena
from anymani.distill.representations.sources.geometry_source import GeometrySource, GeometrySourceCore
from anymani.distill.representations.sources.kinematics import lower_hand_geometry_semantics
from anymani.distill.representations.targets.geometry_field import fixed_validation_gaussian_field_config

from .augmentation import maybe_rewrite_batch
from .batch import (
    PaddedOnlineGeometryBatch,
    attach_static_evidence,
    method_batch_views,
    pad_online_geometry_samples,
    split_padded_online_geometry_batch,
)
from .config import MultiAnchorGaussianMethodCfg
from .context import MultiAnchorObjectiveContext
from .objectives import (
    evaluate_objectives,
    finalize_teacher_baselines,
    merge_teacher_baseline_statistics,
    reduce_method_steps,
    teacher_baseline_sufficient_statistics,
)
from .state_measure import SobolJointSampler

_TRAIN_FORWARD_MICROBATCH_SAMPLES = 64
"""rho/kappa 普通参数反向的 `(asset,q)` 样本上限。"""

_EVALUATION_FORWARD_MICROBATCH_SAMPLES = 64
"""固定评估的单次样本上限，与训练使用同一张量形状合同。"""

_CALIBRATION_FORWARD_MICROBATCH_SAMPLES = 64
"""预实验在 `no_grad` 下使用的单次样本上限。"""

_DEVICE_SUBWINDOW_ASSETS = 8
"""单次 device source/Warp lease 的资产上限；不改变 logical resident window 顺序。"""


def _forward_microbatch_samples(mode: str) -> int:
    r"""按运行阶段返回 rho/kappa 普通前向的 sample 上限。

    训练逐块立即反向，以保持完整 minibatch 的精确充分统计，
    calibration/evaluation 则在 ``no_grad`` 下复用同一块大小。
    """

    if mode == "train":
        return _TRAIN_FORWARD_MICROBATCH_SAMPLES
    if mode == "calibration":
        return _CALIBRATION_FORWARD_MICROBATCH_SAMPLES
    return _EVALUATION_FORWARD_MICROBATCH_SAMPLES


def _merge_microbatch_steps(steps: tuple[MethodStep, ...]) -> MethodStep:
    r"""精确合并 microbatch 的 additive statistics，保留可反向传播的 numerator 图。"""

    if not steps:
        raise ValueError("microbatch reduction requires at least one MethodStep")
    totals: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    metric_values: dict[str, dict[str, list[torch.Tensor]]] = {}
    for step in steps:
        for term_name, result in step.objectives.items():
            term_totals = totals.setdefault(term_name, {})
            term_metrics = metric_values.setdefault(term_name, {})
            for component in result.components:
                previous = term_totals.get(component.name)
                if previous is None:
                    term_totals[component.name] = (component.numerator, component.denominator)
                else:
                    term_totals[component.name] = (
                        previous[0] + component.numerator,
                        previous[1] + component.denominator,
                    )
            for metric_name, metric in result.metrics.items():
                if isinstance(metric, torch.Tensor):
                    term_metrics.setdefault(metric_name, []).append(metric)
    merged: dict[str, ObjectiveTermResult] = {}
    for term_name, component_totals in totals.items():
        components = tuple(
            AdditiveStatistic(name, numerator, denominator)
            for name, (numerator, denominator) in component_totals.items()
        )
        metrics = {
            metric_name: sum(values[1:], values[0]) / len(values)
            for metric_name, values in metric_values.get(term_name, {}).items()
            if values
        }
        merged[term_name] = ObjectiveTermResult(term_name, components, metrics)
    return MethodStep(
        objectives=merged,
        sample_count=sum(step.sample_count for step in steps),
    )


def _detach_method_step(step: MethodStep) -> MethodStep:
    r"""释放 calibration microbatch 的 autograd graph，只保留可加统计量。"""

    objectives = {
        term_name: ObjectiveTermResult(
            term_name,
            tuple(
                AdditiveStatistic(
                    component.name,
                    component.numerator.detach(),
                    component.denominator.detach(),
                )
                for component in result.components
            ),
            {
                metric_name: metric.detach()
                for metric_name, metric in result.metrics.items()
                if isinstance(metric, torch.Tensor)
            },
        )
        for term_name, result in step.objectives.items()
    }
    return MethodStep(objectives=objectives, sample_count=step.sample_count)


def _derive_padding(assets: Sequence[HandContainer], *, max_graph_distance: int) -> GeometryPaddingCfg:
    r"""由 typed semantics 的离散轴长度直接推导 padding。

    padding 只依赖活动 JOINT 数、TIP owner 数和 backbone graph-distance 上限；空间旋量、
    owner home pose、collision component 与 joint limits 都不改变这些离散轴长度。资产层的
    ``HandGeometrySemanticsCfg`` 已经验证 ``active_joint_names`` 与 revolute joint 轴一一对应，
    owner ``role`` 也已经闭合于 PALM/JOINT/TIP，因此无需为完整 catalog 重复执行
    ``lower_hand_geometry_semantics``。对当前 8192-train preset，这使准备阶段从逐资产构造
    float64 POE/graph 张量退化为纯整数扫描，同时保持 $N_J^{max}=16, N_{tip}^{max}=4$。

    Args:
        assets (Sequence[HandContainer]): train/validation/evaluation 的完整 typed asset 轴。
        max_graph_distance (int): backbone 离散图距离截断，直接进入 ``GeometryPaddingCfg``。

    Returns:
        GeometryPaddingCfg: 全部资产都可容纳的 JOINT/TIP/graph padding 上限。

    Raises:
        ValueError: 资产轴为空、任一资产缺少 typed semantics，或 catalog 没有 JOINT/TIP。
    """

    if not assets:
        raise ValueError("padding derivation requires at least one materialized source")
    max_joint = 0
    max_tip = 0
    for asset in assets:
        semantics = asset.geometry_semantics
        if semantics is None:
            raise ValueError(f"asset {asset.asset_id!r} is missing geometry semantics")
        max_joint = max(max_joint, len(semantics.active_joint_names))  # $N_J$：规范活动关节轴长度
        max_tip = max(max_tip, sum(owner.role == "tip" for owner in semantics.owners))  # $N_{tip}$：TIP owner 数
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
    r"""把 resolved HandContainer 轴映射为按资产 demand-load 的 CPU source-core 轴。

    资产顺序和 ID 在 catalog resolve 后已经固定；只有索引、稳定 ID 或完整迭代会触发
    source arena 读取/物化。provider 自身不持有 source dict，因而完整训练轴的历史访问量
    不会越过 arena 的 16 项/512 MiB 上限。
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

        self.assets = tuple(assets)
        self.asset_ids = tuple(asset.asset_id for asset in self.assets)
        if len(set(self.asset_ids)) != len(self.asset_ids):
            raise ValueError("lazy geometry source asset IDs must be unique")
        self.cache = cache
        self.config = config
        self.materialize = materialize
        self._index_by_id = {asset_id: index for index, asset_id in enumerate(self.asset_ids)}
        self._prefetch_executor: ThreadPoolExecutor | None = None  # 首次请求时才创建两个 CPU workers
        self._prefetch_stats: dict[str, int | float] = {
            "subwindow_count": 0,
            "asset_count": 0,
            "ready_latency_seconds": 0.0,
            "blocked_wait_seconds": 0.0,
        }  # current/next pipeline 的累计可消费延迟与主线程实际阻塞时间
        self._ready_latencies: list[float] = []  # 每个 8-asset subwindow 从 submit 到可消费的 wall latency，s
        self._blocked_waits: list[float] = []  # 主线程 await 的未重叠尾延迟，s

    def __len__(self) -> int:
        r"""返回不触发 source 物化的资产数量。"""

        return len(self.assets)

    @overload
    def __getitem__(self, index: int) -> GeometrySourceCore: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[GeometrySourceCore, ...]: ...

    def __getitem__(self, index: int | slice) -> GeometrySourceCore | tuple[GeometrySourceCore, ...]:
        r"""按索引读取 source core；slice 只在显式请求时展开对应范围。"""

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
        r"""按稳定 asset ID 读取 source core，供 resident window demand-load。"""

        try:
            return self[self._index_by_id[asset_id]]
        except KeyError as exc:
            raise KeyError(f"unknown geometry asset ID={asset_id!r}") from exc

    def prefetch_async(self, asset_ids: Sequence[str]) -> SourcePrefetchHandle:
        r"""异步准备下一 device subwindow 的 CPU core，不等待 GPU 当前组完成。

        每项资产继续通过 arena 的 per-key 锁保证幂等。executor 跨 subwindow 复用，避免每 8 项
        重建线程；两个 worker 与 16-entry arena 对应 current/next 双缓冲，不扩大历史驻留量。
        """

        requested = tuple(asset_ids)
        started = perf_counter()
        if not requested:
            return SourcePrefetchHandle(requested, (), started)
        if self._prefetch_executor is None:
            self._prefetch_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="source-prefetch")
        futures = tuple(self._prefetch_executor.submit(self.get, asset_id) for asset_id in requested)
        return SourcePrefetchHandle(requested, futures, started)

    def await_prefetch(self, handle: SourcePrefetchHandle) -> tuple[GeometrySourceCore, ...]:
        r"""等待并直接返回 current core buffer，避免 next-buffer LRU 插入使其重复物化。"""

        wait_started = perf_counter()  # next-buffer 若已与 GPU 重叠完成，此处应接近零阻塞
        cores: list[GeometrySourceCore] = []
        for future in handle.futures:
            cores.append(future.result())
        completed = perf_counter()
        ready_latency = completed - handle.started  # CPU current/next pipeline 对该 subwindow 的完整服务时间，s
        blocked_wait = completed - wait_started  # GPU 当前组未覆盖掉的 CPU 尾部等待，s
        self._prefetch_stats["subwindow_count"] = int(self._prefetch_stats["subwindow_count"]) + 1
        self._prefetch_stats["asset_count"] = int(self._prefetch_stats["asset_count"]) + len(handle.asset_ids)
        self._prefetch_stats["ready_latency_seconds"] = float(
            self._prefetch_stats["ready_latency_seconds"]
        ) + ready_latency
        self._prefetch_stats["blocked_wait_seconds"] = float(
            self._prefetch_stats["blocked_wait_seconds"]
        ) + blocked_wait
        self._ready_latencies.append(ready_latency)
        self._blocked_waits.append(blocked_wait)
        return tuple(cores)  # futures 的 submission order 与 asset_ids 严格一致

    def prefetch_stats(self) -> dict[str, int | float]:
        r"""返回 current/next core pipeline 的累计 timing 与预算证据。"""

        evidence = dict(self._prefetch_stats)
        if self._ready_latencies:
            rank = math.ceil(0.95 * len(self._ready_latencies)) - 1  # nearest-rank $P_{95}$，保留实测 subwindow 值
            evidence["ready_latency_p95_seconds"] = sorted(self._ready_latencies)[rank]
            evidence["blocked_wait_p95_seconds"] = sorted(self._blocked_waits)[rank]
        return evidence

    def prefetch(self, asset_ids: Sequence[str]) -> None:
        r"""同步兼容入口；scheduler 未提供外层流水时提交并等待同一 core buffer。"""

        self.await_prefetch(self.prefetch_async(asset_ids))

    def close(self) -> None:
        r"""等待已提交 core 工作并释放持久 executor；arena 生命周期由 Method 统一管理。"""

        if self._prefetch_executor is not None:
            self._prefetch_executor.shutdown(wait=True, cancel_futures=True)
            self._prefetch_executor = None


class LazySobolSamplers(Sequence[SobolJointSampler]):
    r"""按资产 typed semantics 延迟构造独立 Sobol joint-limit sampler。

    sampler 只需要 joint limits；它不得通过 ``GeometrySource.spec_cpu`` 取 limits，否则 checkpoint
    保存全部 cursor 会隐式物化完整 asset catalog 的 owner mesh。
    """

    def __init__(self, sources: LazyGeometrySources, *, seed: int) -> None:
        r"""保存轻量资产轴和 seed，不提前 lower 全部 sampler。"""

        self.sources = sources
        self.seed = int(seed)
        self._samplers: dict[int, SobolJointSampler] = {}

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
            spec = lower_hand_geometry_semantics(semantics, dtype=torch.float64)
            sampler = SobolJointSampler(spec, seed=self.seed + index)
            self._samplers[index] = sampler
        return sampler

    def clear(self) -> None:
        r"""释放 Sobol engine；cursor 已由 checkpoint 保存时可从 seed+cursor 重建。"""

        self._samplers.clear()

    def state_dict(self) -> tuple[dict[str, int], ...]:
        r"""完整 materialized sampler 状态，供 checkpoint 保存。"""

        return tuple(self[index].state_dict() for index in range(len(self)))

    def load_state_dict(self, states: Sequence[object]) -> None:
        r"""按固定资产轴恢复所有 sampler cursor。"""

        if len(states) != len(self):
            raise ValueError("sampler state count does not match lazy asset axis")
        for index, state in enumerate(states):
            if not isinstance(state, Mapping):
                raise ValueError("lazy sampler state must be a mapping")
            if not all(isinstance(key, str) and isinstance(value, int) for key, value in state.items()):
                raise ValueError("lazy sampler state keys/values must be str/int")
            parsed = {str(key): int(value) for key, value in state.items()}
            self[index].load_state_dict(parsed)


class PhysicalAuditHandle:
    r"""后台完整 physical asset manifest 的可等待句柄。

    轻量 catalog 身份检查完成后，完整 ``physical_geometry_hash``、anchor realization
    和 home-surface realization 在独立 CPU worker 中构造。训练可以先准备首个 resident
    window；任何 checkpoint 或 artifact 写入仍必须调用 ``wait``，因此审计失败不会被
    训练产物隐藏。
    """

    def __init__(
        self,
        future: Future[dict[str, Any]],
        executor: ThreadPoolExecutor,
        cancel_event: Event,
    ) -> None:
        r"""保存后台任务和 executor 生命周期。"""

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
        r"""协作停止未发布的 audit，并等待当前单项 source 物化退出。"""

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
        method: MultiAnchorGaussianMethod,
        *,
        role: str,
        suite: str,
        sources: LazyGeometrySources,
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
        max_resident_assets: int,
        window_factory: Any,
    ) -> None:
        r"""建立独立 sampler 与 device window；Trainer 只保留本 session，不读取底层数组。"""

        if not sources:
            raise ValueError(f"method session role={role!r} suite={suite!r} requires at least one asset")
        self.method = method
        self.role = role
        self.suite = suite
        self.sources = sources
        self.seed = int(seed)
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
        r"""返回资产轴和每资产 Sobol cursor，供 optimizer-boundary checkpoint 保存。"""

        return {
            "asset_ids": self.sources.asset_ids,
            "samplers": tuple(sampler.state_dict() for sampler in self.samplers),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        r"""严格恢复同一资产轴上的 Sobol cursor。"""

        expected_ids = self.sources.asset_ids
        raw_asset_ids = state.get("asset_ids")
        if not isinstance(raw_asset_ids, (tuple, list)) or tuple(raw_asset_ids) != expected_ids:
            raise ValueError("method session checkpoint asset axis does not match current split")
        raw_samplers = state.get("samplers")
        if not isinstance(raw_samplers, (tuple, list)) or len(raw_samplers) != len(self.samplers):
            raise ValueError("method session checkpoint sampler count does not match current split")
        for sampler, sampler_state in zip(self.samplers, raw_samplers):
            if not isinstance(sampler_state, dict):
                raise ValueError("method session sampler state must be a mapping")
            sampler.load_state_dict(sampler_state)

    def close(self) -> None:
        r"""幂等释放 resident Warp lease、device state 与本 session 的 Sobol engines。"""

        self.window.release_all()
        self.window.drain_telemetry_events()
        self.samplers.clear()


class MultiAnchorGaussianMethod:
    r"""显式装配 GeometryRepresentation、GeometrySSLModel 与 rho/kappa 双 objective。"""

    def __init__(self, config: MultiAnchorGaussianMethodCfg) -> None:
        r"""保存配置并构造无 IO 的 representation runtime。"""

        self.config = config
        self.representation = GeometryRepresentation(config.representation)
        self.validation_representation = GeometryRepresentation(
            replace(
                config.representation,
                field=fixed_validation_gaussian_field_config(config.representation.field),
            )
        )
        self.model: GeometrySSLModel | None = None
        self.train_sources: LazyGeometrySources | None = None
        self.source_cache = GeometrySourceArena()
        self.validation_sources: dict[str, LazyGeometrySources] = {}
        self.evaluation_sources: dict[str, LazyGeometrySources] = {}
        self.padding: GeometryPaddingCfg | None = None
        self.runtime_device: torch.device | None = None  # physical audit 与 resident source 共用 classifier device
        self.teacher_baselines: dict[str, float] | None = None  # pretrain 前由 schema-7 artifact 固定
        self._anchor_classification: dict[str, int | float] = {
            "asset_count": 0,
            "query_point_count": 0,
            "kernel_launch_count": 0,
            "boundary_recheck_count": 0,
            "boundary_disagreement_count": 0,
            "elapsed_seconds": 0.0,
        }  # 所有 split 共用的 append-only GPU/CPU anchor 分类证据

    def prepare(self, catalog: Any, *, device: torch.device, dtype: torch.dtype) -> None:
        r"""建立 train/validation/evaluation source provider，并推导全局 padding。"""

        del dtype
        self.runtime_device = torch.device(device)  # audit manifest 必须重建训练实际使用的 CUDA anchors
        print(f"[Method] Indexing lazy train sources: {len(catalog.train)} assets")
        self.train_sources = self._lazy_sources(catalog.train, self.representation)

        print("[Method] Indexing lazy validation sources...")
        self.validation_sources = {
            suite_name: self._lazy_sources(suite_assets, self.validation_representation)
            for suite_name, suite_assets in catalog.validation.items()
        }

        print("[Method] Indexing lazy evaluation sources...")
        self.evaluation_sources = {
            suite_name: self._lazy_sources(suite_assets, self.validation_representation)
            for suite_name, suite_assets in catalog.evaluation.items()
        }

        all_assets = tuple(catalog.train)
        all_assets += tuple(asset for assets in catalog.validation.values() for asset in assets)
        all_assets += tuple(asset for assets in catalog.evaluation.values() for asset in assets)
        self.padding = _derive_padding(
            all_assets,
            max_graph_distance=self.config.model.encoder.backbone.max_graph_distance,
        )

    def _lazy_sources(
        self,
        assets: Sequence[HandContainer],
        representation: GeometryRepresentation,
    ) -> LazyGeometrySources:
        r"""建立一个不触发 source IO 的 provider。"""

        return LazyGeometrySources(
            assets,
            cache=self.source_cache,
            config=self.config.representation.source,
            materialize=representation.materialize_core,
        )

    def split_names(self, role: str) -> tuple[str, ...]:
        r"""返回 validation/evaluation 的具名 suites；train 没有 suite 子轴。"""

        if role == "train":
            return ("",)
        if role == "training_evaluation":
            return ("",)
        if role == "validation":
            return tuple(self.validation_sources)
        if role == "evaluation":
            return tuple(self.evaluation_sources)
        raise ValueError(f"unknown method split role={role!r}")

    def require_train_sources(self) -> LazyGeometrySources:
        r"""返回 prepare 后的 train provider；生命周期顺序错误时 fail-fast。"""

        if self.train_sources is None:
            raise RuntimeError("multi-anchor method train sources have not been prepared")
        return self.train_sources

    def split_asset_count(self, role: str, *, suite: str = "") -> int:
        r"""返回 train 或具名 held-out suite 的真实资产数。"""

        if role == "train":
            return len(self.require_train_sources())
        if role == "training_evaluation":
            return len(self.require_train_sources())
        if role == "validation":
            return len(self.validation_sources.get(suite, ()))
        if role == "evaluation":
            return len(self.evaluation_sources.get(suite, ()))
        raise ValueError(f"unknown method split role={role!r}")

    def asset_manifest(self, catalog: Any, *, cancel_event: Event | None = None) -> dict[str, Any]:
        r"""流式记录 physical source 与 train/held-out 隔离证据。

        audit 不遍历训练用 ``LazyGeometrySources``，避免完整 catalog 污染其 LRU 或与 device prefetch
        争夺同一 arena。每项 source 在 ``record`` 返回后即可释放，只保留小型 hash/provenance mapping。
        """

        from .provenance import (
            anchor_realization_record,
            home_surface_realization_record,
            validate_asset_manifest_isolation,
        )

        def record(asset: Any, source: GeometrySource, *, partition: str, provenance: Any) -> dict[str, Any]:
            r"""把一项 materialized source 规约成稳定的物理与采样 provenance。"""

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
                    anchor_device=str(self.runtime_device) if self.runtime_device is not None else "cpu",
                ),
                partition=partition,
                provenance=item.provenance,
            )

        manifest = {
            "schema_version": "4.0.0",
            "dataset_source_path": str(catalog.dataset.source_path),
            "dataset_source_sha256": catalog.dataset.source_sha256,
            "train": [
                record_item(
                    item,
                    partition="train",
                    representation=self.representation,
                )
                for item in catalog.dataset.train.records
            ],
            "validation": {
                suite: [
                    record_item(
                        item,
                        partition=f"validation.{suite}",
                        representation=self.validation_representation,
                    )
                    for item in partition.records
                ]
                for suite, partition in catalog.dataset.validation.items()
            },
            "evaluation": {
                suite: [
                    record_item(
                        item,
                        partition=f"evaluation.{suite}",
                        representation=self.validation_representation,
                    )
                    for item in partition.records
                ]
                for suite, partition in catalog.dataset.evaluation.items()
            },
        }
        validate_asset_manifest_isolation(manifest)
        return manifest

    def start_physical_audit(self, catalog: Any) -> PhysicalAuditHandle:
        r"""启动完整 physical manifest 计算；结果由 checkpoint/artifact gate 等待。"""

        cancel_event = Event()  # teardown 可在两项 source 之间停止尚未发布的审计
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ssl-physical-audit")
        future = executor.submit(self.asset_manifest, catalog, cancel_event=cancel_event)
        return PhysicalAuditHandle(future, executor, cancel_event)

    def open_session(
        self,
        role: str,
        *,
        suite: str = "",
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
        max_resident_assets: int,
        window_factory: Any,
    ) -> MultiAnchorGaussianSession:
        r"""打开一个不向 Trainer 暴露 sources/samplers/loaders 的 split session。"""

        if role in {"train", "training_evaluation"}:
            sources = self.require_train_sources()
        elif role == "validation":
            sources = self.validation_sources.get(suite)
        elif role == "evaluation":
            sources = self.evaluation_sources.get(suite)
        else:
            raise ValueError(f"unknown method split role={role!r}")
        if sources is None:
            raise KeyError(f"unknown method split suite role={role!r} suite={suite!r}")
        return MultiAnchorGaussianSession(
            self,
            role=role,
            suite=suite,
            sources=sources,
            seed=seed,
            device=device,
            dtype=dtype,
            max_resident_assets=max_resident_assets,
            window_factory=window_factory,
        )

    def make_independent_samplers(
        self,
        sources: LazyGeometrySources,
        *,
        seed: int,
    ) -> LazySobolSamplers:
        r"""为给定 CPU sources 建立不复用训练 cursor 的独立 Sobol 引擎。"""

        return LazySobolSamplers(sources, seed=seed)

    def initialize_model(self, *, device: torch.device, dtype: torch.dtype) -> GeometrySSLModel:
        r"""在 Trainer 已冻结 device/dtype 后一次性构造 learned model。"""

        if self.model is not None:
            raise RuntimeError("multi-anchor method model is already initialized")
        self.model = GeometrySSLModel(self.config.model).to(device=device, dtype=dtype)
        return self.model

    def require_model(self) -> GeometrySSLModel:
        r"""返回已初始化模型；setup 顺序错误时明确失败。"""

        if self.model is None:
            raise RuntimeError("multi-anchor method model has not been initialized")
        return self.model

    def parameters(self):
        r"""返回完整 learned method 的 optimizer parameters。"""

        return self.require_model().parameters()

    def train_mode(self) -> None:
        r"""启用训练期 dropout/normalization 行为。"""

        self.require_model().train()

    def eval_mode(self) -> None:
        r"""启用固定评估行为；评估生命周期在 ``no_grad`` 下运行。"""

        self.require_model().eval()

    def require_padding(self) -> GeometryPaddingCfg:
        r"""返回已由 dataset/model 推导的 padding 上限。"""

        if self.padding is None:
            raise RuntimeError("multi-anchor method padding has not been derived")
        return self.padding

    def load_device_state(self, source: GeometrySourceCore, *, device: torch.device | str, dtype: torch.dtype):
        r"""把一项 CPU core 完成 anchors 并上传为训练期 device state。"""

        state = self.representation.to_device(source, device=device, dtype=dtype)
        self._record_anchor_classification(state)
        return state

    def load_validation_device_state(
        self, source: GeometrySourceCore, *, device: torch.device | str, dtype: torch.dtype
    ):
        r"""把一项 CPU core 完成 anchors 并上传为固定 validation sigma 的 device state。"""

        state = self.validation_representation.to_device(source, device=device, dtype=dtype)
        self._record_anchor_classification(state)
        return state

    def _record_anchor_classification(self, state: Any) -> None:
        r"""把一项 resident source 的 anchor backend 证据规约到 run 累计量。"""

        stats = getattr(state, "anchor_classification", None)
        if stats is None:
            return
        self._anchor_classification["asset_count"] = int(self._anchor_classification["asset_count"]) + 1
        for name in (
            "query_point_count",
            "kernel_launch_count",
            "boundary_recheck_count",
            "boundary_disagreement_count",
        ):
            self._anchor_classification[name] = int(self._anchor_classification[name]) + int(getattr(stats, name))
        self._anchor_classification["elapsed_seconds"] = float(
            self._anchor_classification["elapsed_seconds"]
        ) + float(stats.elapsed_seconds)

    def declared_objective_weights(self) -> dict[str, float]:
        r"""返回固定 density/kappa normalized vanilla 权重。"""

        return {name: float(term.weight) for name, term in self.config.objectives.enabled().items()}

    def formula_identity(self) -> dict[str, str]:
        r"""返回 density/kappa objective 公式身份：模块级函数的完整限定名。"""

        return {name: term.qualified_func_name() for name, term in self.config.objectives.enabled().items()}

    def runtime_resource_evidence(self) -> dict[str, object]:
        r"""返回 CPU core arena 与 GPU/CPU anchor classifier 的有界资源证据。"""

        evidence: dict[str, object] = {
            "geometry_source_core_arena": self.source_cache.stats(),
            "anchor_classifier": dict(self._anchor_classification),
        }
        if self.train_sources is not None:
            evidence["geometry_core_prefetch"] = self.train_sources.prefetch_stats()
        if self.model is not None:
            parameter = next(self.model.parameters(), None)
            if parameter is not None and parameter.device.type == "cuda":
                evidence["cuda_memory"] = {
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(parameter.device)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(parameter.device)),
                    "current_allocated_bytes": int(torch.cuda.memory_allocated(parameter.device)),
                    "current_reserved_bytes": int(torch.cuda.memory_reserved(parameter.device)),
                }  # PyTorch allocator 口径；Warp BVH 由 scheduler 的 driver free/total 另行记录
        return evidence

    def realize_minibatch(
        self,
        schedule_item: Any,
        *,
        sources: LazyGeometrySources,
        samplers: LazySobolSamplers,
        window: Any,
        seed: int,
        schedule: Any,
        mode: str = "train",
    ) -> PaddedOnlineGeometryBatch:
        r"""由 schedule item realization 一次同资产 q block，并在 window 内复用 device state。"""

        del schedule  # q-block、window 与随机身份均由 ScheduledMinibatch 显式携带
        representation = self.representation if mode == "train" else self.validation_representation
        padding = self.require_padding()
        catalog_ids = sources.asset_ids
        resident_indices = tuple(schedule_item.resident_asset_indices)
        if not resident_indices:
            raise ValueError("schedule item must declare the complete resident window, not only the minibatch")
        samples = []
        q_block_index = int(schedule_item.q_block_index)  # 同一资产获得第几个新 q-block
        resident_set = set(resident_indices)
        logical_indices = tuple(schedule_item.asset_indices)
        if any(asset_index not in resident_set for asset_index in logical_indices):
            raise ValueError("logical minibatch assets must belong to its declared resident window")
        asset_chunks = tuple(
            logical_indices[chunk_start : chunk_start + _DEVICE_SUBWINDOW_ASSETS]
            for chunk_start in range(0, len(logical_indices), _DEVICE_SUBWINDOW_ASSETS)
        )  # logical 64-asset batch 的有序 8-asset device subwindows
        first_ids = tuple(catalog_ids[index] for index in asset_chunks[0])
        prefetch_handle = sources.prefetch_async(first_ids)  # 首组无可重叠 GPU 工作，立即异步提交
        for chunk_index, asset_chunk in enumerate(asset_chunks):
            current_cores = sources.await_prefetch(prefetch_handle)  # 强引用 pin current，不依赖 arena LRU
            next_handle = None
            if chunk_index + 1 < len(asset_chunks):
                next_ids = tuple(catalog_ids[index] for index in asset_chunks[chunk_index + 1])
                next_handle = sources.prefetch_async(next_ids)  # CPU next 与当前 GPU finalization/teacher 重叠
            states = window.ensure(
                tuple(catalog_ids[index] for index in asset_chunk),
                prefetch_sources=False,
                prepared_sources={core.asset_id: core for core in current_cores},
            )
            states_by_id = {state.source.asset_id: state for state in states}
            for asset_index in asset_chunk:
                asset_id = catalog_ids[asset_index]  # logical catalog index 对应的稳定 source identity
                state = states_by_id[asset_id]
                source = state.source  # 当前 device state 持有已经完成 GPU anchor bank 的最终 source
                q_count = schedule_item.q_per_asset
                q = samplers[asset_index].draw(
                    q_count, device=state.spec.space_screws.device, dtype=state.spec.space_screws.dtype
                )
                q_start = samplers[asset_index].cursor - q_count
                bank = source.anchor_bank
                if not bank:
                    raise ValueError(f"asset {source.asset_id!r} has an empty physical anchor bank")
                anchor_index = 0 if mode != "train" else int(q_block_index % len(bank))
                schedule_index = (
                    int(schedule_item.minibatch_index) * 1_000_003
                    + int(schedule_item.window_index) * 10_007
                    + int(schedule_item.asset_group)
                )
                physical = representation.sample(
                    state,
                    q,
                    sampling_seed=seed + schedule_index,
                    q_index=torch.arange(q_start, q_start + q_count, device=q.device, dtype=torch.long),
                    anchor_index=anchor_index,
                    supervision_split="train" if mode == "train" else "eval",
                )
                samples.extend(
                    attach_static_evidence(
                        physical,
                        source=source,
                        spec=state.spec,
                        anchors=bank[anchor_index],
                        device=q.device,
                        dtype=q.dtype,
                    )
                )
            if next_handle is not None:
                prefetch_handle = next_handle
        return pad_online_geometry_samples(samples, padding=padding)

    def forward_objectives(
        self,
        batch: PaddedOnlineGeometryBatch,
        *,
        step: int,
        mode: str = "train",
        microbatch_size: int | None = None,
    ) -> MethodStep:
        r"""完成一次 logical minibatch 前向，并按 phase-specific sample budget 做 microbatch。

        logical minibatch 的 sampling budget、joint-sign rewrite 随机身份和 additive
        objective 统计不变；microbatch 只限制单次 encoder activation 的峰值显存。
        """

        microbatch_samples = int(microbatch_size or _forward_microbatch_samples(mode))
        q_per_asset = self._q_per_asset_block(batch)
        if batch.q.shape[0] % microbatch_samples != 0:
            raise ValueError("microbatch_size must exactly divide the realized minibatch")
        if microbatch_samples % q_per_asset != 0:
            raise ValueError("microbatch_size must preserve complete per-asset q blocks")
        if batch.q.shape[0] <= microbatch_samples:
            result, _prediction = self._forward_with_prediction(batch, step=step, mode=mode)
            return result
        rewritten = (
            maybe_rewrite_batch(
                batch,
                config=self.config.joint_sign_rewrite,
                step=step,
                seed=step,
            )
            if mode in {"train", "calibration"}
            else batch
        )
        steps = []
        for microbatch in split_padded_online_geometry_batch(
            rewritten,
            microbatch_size=microbatch_samples,
        ):
            micro_step = self._forward_with_prediction(
                microbatch,
                step=step,
                mode=mode,
                apply_augmentation=False,
            )[0]
            steps.append(_detach_method_step(micro_step) if mode == "calibration" else micro_step)
        return _merge_microbatch_steps(tuple(steps))

    def teacher_baseline_statistics(self, batch: PaddedOnlineGeometryBatch) -> dict[str, torch.Tensor]:
        r"""只读取 teacher truth，返回 constant-rho/zero-kappa 的单批充分统计。"""

        return teacher_baseline_sufficient_statistics(batch)

    def merge_teacher_baseline_statistics(
        self,
        total: dict[str, torch.Tensor] | None,
        block: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        r"""合并完整 catalog 单遍中的 teacher-only 充分统计。"""

        return merge_teacher_baseline_statistics(total, block)

    def finalize_teacher_baselines(self, statistics: dict[str, torch.Tensor]) -> dict[str, object]:
        r"""闭合 teacher-only 充分统计，形成 artifact 中的固定 normalization constants。"""

        return finalize_teacher_baselines(statistics)

    def set_teacher_baselines(self, payload: Mapping[str, object]) -> None:
        r"""从已核验 artifact 装载严格正的 $B_\rho,B_\kappa$，拒绝缺项或额外 objective。"""

        expected = tuple(self.config.objectives.enabled())
        if expected != ("density", "kappa"):
            raise ValueError("unified Geometry SSL requires exactly density and kappa objectives")
        values: dict[str, float] = {}
        for name in expected:
            record = payload.get(name)
            if not isinstance(record, Mapping):
                raise ValueError(f"teacher baseline artifact lacks mapping for {name}")
            value = record.get("baseline_mse")
            if not isinstance(value, (float, int)) or float(value) <= 0.0:
                raise ValueError(f"teacher baseline {name}.baseline_mse must be positive")
            values[name] = float(value)
        self.teacher_baselines = values

    def require_teacher_baselines(self) -> dict[str, float]:
        r"""返回训练固定分母；未装载 artifact 时禁止计算 optimizer loss。"""

        if self.teacher_baselines is None:
            raise RuntimeError("pretraining requires configured teacher-only density/kappa baselines")
        return dict(self.teacher_baselines)

    def _forward_with_prediction(
        self,
        batch: PaddedOnlineGeometryBatch,
        *,
        step: int,
        mode: str,
        apply_augmentation: bool = True,
    ) -> tuple[MethodStep, Any]:
        r"""共享 objective 与固定评估所需的同一次模型预测，避免 validation 双前向。"""

        model = self.require_model()
        if mode in {"train", "calibration"} and apply_augmentation:
            batch = maybe_rewrite_batch(
                batch,
                config=self.config.joint_sign_rewrite,
                step=step,
                seed=step,
            )
        views = method_batch_views(batch)
        q, evidence = views.model_input
        query_points, bandwidths, owner_index, query_index, joint_index = views.readout_condition
        q = q.detach()  # 物理构型是模型条件；本方法只对模型参数建立普通一阶梯度图
        prediction = model(
            q,
            evidence,
            query_points,
            bandwidths,
            owner_index=owner_index,
            query_index=query_index,
            joint_index=joint_index,
        )
        context = MultiAnchorObjectiveContext(
            prediction=prediction,
            batch=batch,
        )
        results = evaluate_objectives(context, self.config.objectives)
        return MethodStep(objectives=results, sample_count=int(batch.q.shape[0])), prediction

    def reduce_update(self, steps: tuple[MethodStep, ...]) -> MethodUpdate:
        r"""按 $(asset,q)$ 等权合并一个 optimizer update。"""

        return reduce_method_steps(steps, self.config.objectives, self.require_teacher_baselines())

    @staticmethod
    def _q_per_asset_block(batch: PaddedOnlineGeometryBatch) -> int:
        r"""由连续 asset-major 样本轴恢复每资产完整 q-block 长度。"""

        if not batch.asset_ids:
            raise ValueError("training minibatch must contain at least one asset/q pair")
        first_asset = batch.asset_ids[0]
        q_per_asset = 0
        for asset_id in batch.asset_ids:
            if asset_id != first_asset:
                break
            q_per_asset += 1
        if q_per_asset < 1 or len(batch.asset_ids) % q_per_asset != 0:
            raise ValueError("training minibatch asset axis does not contain uniform q blocks")
        for start in range(0, len(batch.asset_ids), q_per_asset):
            if len(set(batch.asset_ids[start : start + q_per_asset])) != 1:
                raise ValueError("training minibatch must remain asset-major with contiguous q blocks")
        return q_per_asset

    def _training_minibatch_denominators(
        self,
        batch: PaddedOnlineGeometryBatch,
    ) -> dict[str, torch.Tensor]:
        r"""在模型 forward 前由监督 mask 计算完整 minibatch 的样本 denominator。

        两项 objective 都先在每个 $(asset,q)$ 行内归约，再对有效行等权。density 行有效当且仅当
        任一 owner/query 有效；kappa 行有效当且仅当任一 sampled edge 有效。joint-sign
        rewrite 不改变 mask，因此 denominator 可在 augmentation/forward 前精确得到。
        """

        device = batch.q.device
        dtype = batch.q.dtype
        density = torch.zeros((), device=device, dtype=dtype)  # 有效 density $(asset,q)$ 行数
        edge = torch.zeros((), device=device, dtype=dtype)  # 有效 kappa $(asset,q)$ 行数
        density += batch.field_targets.valid_mask.reshape(batch.q.shape[0], -1).any(dim=-1).sum().to(dtype)
        edge += batch.sensitivity_targets.valid_mask.reshape(batch.q.shape[0], -1).any(dim=-1).sum().to(dtype)
        available = {
            "density": density,
            "kappa": edge,
        }
        denominators = {name: available[name] for name in self.config.objectives.enabled()}
        invalid = [name for name, denominator in denominators.items() if float(denominator) <= 0.0]
        if invalid:
            raise ValueError(f"streaming backward minibatch has no valid samples for objectives={invalid}")
        return denominators

    def backward_update(
        self,
        batch: PaddedOnlineGeometryBatch,
        *,
        forward_step: int,
        microbatch_size: int,
        collect_z_gradients: bool = False,
    ) -> MethodUpdate:
        r"""按完整 minibatch denominator 对显式 microbatches 流式反传。

        对 objective $j$、microbatch $m$ 的充分统计 numerator $N_{j,m}$ 与完整 minibatch
        denominator $D_j$，利用微分线性性执行：
        $$
        \nabla_\theta\mathcal L
        =\sum_m\sum_j w_j\nabla_\theta\frac{N_{j,m}}{D_j}.
        $$
        每个 microbatch 调用 ``backward`` 后即可释放普通参数图；这与先构造
        $\sum_mN_{j,m}/D_j$ 再统一 backward 数学等价，但峰值不随 minibatch 增长。

        Args:
            batch (PaddedOnlineGeometryBatch): 一次 optimizer update 的完整统计 minibatch。
            forward_step (int): 当前 update 的 augmentation/forward 随机身份。
            microbatch_size (int): 一次 forward/backward 的 $(asset,q)$ pair 数。
            collect_z_gradients (bool): 是否累计 rho/kappa 对 unified $Z$ 的稀疏充分统计。

        Returns:
            MethodUpdate: detached normalized loss、raw/normalized 双项均值与样本数；参数 ``.grad`` 已累积。
        """

        q_per_asset = self._q_per_asset_block(batch)
        if batch.q.shape[0] % microbatch_size != 0:
            raise ValueError("microbatch_size must exactly divide the realized minibatch")
        if microbatch_size % q_per_asset != 0:
            raise ValueError("microbatch_size must preserve complete per-asset q blocks")
        denominators = self._training_minibatch_denominators(batch)  # 完整 minibatch 的固定 $D_j$
        numerators = {
            name: torch.zeros_like(denominator) for name, denominator in denominators.items()
        }  # detached $\sum_mN_{j,m}$，只服务日志/等价性检查
        observed_denominators = {
            name: torch.zeros_like(denominator) for name, denominator in denominators.items()
        }
        enabled = self.config.objectives.enabled()
        baselines = self.require_teacher_baselines()  # 固定 $B_\rho,B_\kappa$，整个 run 不更新
        sample_count = 0
        z_gradient_squares = {name: 0.0 for name in enabled}  # $\sum_m\|\nabla_{Z_m}L_j\|^2$
        z_gradient_dot = 0.0  # $\sum_m\langle\nabla_{Z_m}L_\rho,\nabla_{Z_m}L_\kappa\rangle$
        diagnostic_totals: dict[str, list[float]] = {}  # 名称 -> [平方和或计数, denominator]

        def accumulate_diagnostic(name: str, values: torch.Tensor, mask: torch.Tensor) -> None:
            r"""沿完整 microbatch 累加逐元素诊断，不改变 objective 的 per-sample reduction。"""

            weight = mask.to(values.dtype)
            while weight.ndim < values.ndim:
                weight = weight.unsqueeze(-1)
            weight = weight.expand_as(values)
            current = diagnostic_totals.setdefault(name, [0.0, 0.0])
            current[0] += float((values * weight).sum().detach())
            current[1] += float(weight.sum().detach())

        rewritten = maybe_rewrite_batch(
            batch,
            config=self.config.joint_sign_rewrite,
            step=int(forward_step),
            seed=int(forward_step),
        )
        for microbatch in split_padded_online_geometry_batch(
            rewritten,
            microbatch_size=microbatch_size,
        ):
            micro_step, prediction = self._forward_with_prediction(
                microbatch,
                step=int(forward_step),
                mode="train",
                apply_augmentation=False,
            )
            micro_loss: torch.Tensor | None = None
            z_gradients: dict[str, torch.Tensor] = {}
            field_valid = microbatch.field_targets.valid_mask
            edge_valid = microbatch.sensitivity_targets.valid_mask
            active_mask = microbatch.sensitivity_targets.active_mask
            if active_mask.ndim == 1:
                active_mask = active_mask.unsqueeze(0).expand_as(edge_valid)
            density_error_sq = (prediction.density - microbatch.field_targets.density).square()
            kappa_error_sq = (prediction.kappa - microbatch.sensitivity_targets.kappa).square()
            accumulate_diagnostic("density/prediction_square", prediction.density.square(), field_valid)
            accumulate_diagnostic("density/target_square", microbatch.field_targets.density.square(), field_valid)
            accumulate_diagnostic("density/error_square", density_error_sq, field_valid)
            accumulate_diagnostic("kappa/prediction_square", prediction.kappa.square(), edge_valid)
            accumulate_diagnostic("kappa/target_square", microbatch.sensitivity_targets.kappa.square(), edge_valid)
            accumulate_diagnostic("kappa/error_square", kappa_error_sq, edge_valid)
            accumulate_diagnostic("kappa/active_error_square", kappa_error_sq, edge_valid & active_mask)
            accumulate_diagnostic("kappa/zero_error_square", kappa_error_sq, edge_valid & ~active_mask)
            diagnostic_totals.setdefault("density/valid_ratio", [0.0, 0.0])
            diagnostic_totals["density/valid_ratio"][0] += float(field_valid.sum())
            diagnostic_totals["density/valid_ratio"][1] += float(field_valid.numel())
            diagnostic_totals.setdefault("kappa/valid_ratio", [0.0, 0.0])
            diagnostic_totals["kappa/valid_ratio"][0] += float(edge_valid.sum())
            diagnostic_totals["kappa/valid_ratio"][1] += float(edge_valid.numel())
            for term_name, result in micro_step.objectives.items():
                if len(result.components) != 1 or result.components[0].name != term_name:
                    raise ValueError("streaming backward requires one same-name additive component per term")
                component = result.components[0]
                raw_term = component.numerator / denominators[term_name]  # 完整 minibatch denominator 下的 $L_j$
                if collect_z_gradients:
                    z_gradients[term_name] = torch.autograd.grad(
                        raw_term,
                        prediction.latents.entities,
                        retain_graph=True,
                    )[0].detach()  # 当前 microbatch 的 unified-Z proxy，不写回 ``.grad``
                weighted = (
                    float(enabled[term_name].weight)
                    * raw_term
                    / float(baselines[term_name])
                )
                micro_loss = weighted if micro_loss is None else micro_loss + weighted
                numerators[term_name] += component.numerator.detach()
                observed_denominators[term_name] += component.denominator.detach()
            if micro_loss is None:
                raise ValueError("streaming backward microbatch contains no enabled objective")
            if collect_z_gradients:
                if set(z_gradients) != {"density", "kappa"}:
                    raise RuntimeError("unified-Z gradient evidence requires density and kappa gradients")
                rho_gradient = z_gradients["density"]
                kappa_gradient = z_gradients["kappa"]
                z_gradient_squares["density"] += float(rho_gradient.square().sum())
                z_gradient_squares["kappa"] += float(kappa_gradient.square().sum())
                z_gradient_dot += float((rho_gradient * kappa_gradient).sum())
            micro_loss.backward()  # 当前普通参数图立即释放；梯度只累计到当前完整 minibatch
            sample_count += micro_step.sample_count

        for name, expected in denominators.items():
            if not torch.equal(observed_denominators[name], expected):
                raise RuntimeError(
                    f"streaming backward denominator mismatch for {name}: "
                    f"observed={float(observed_denominators[name])}, expected={float(expected)}"
                )
        terms = {name: float(numerators[name] / denominators[name]) for name in denominators}
        normalized_terms = {name: terms[name] / float(baselines[name]) for name in denominators}
        skills = {name: 1.0 - normalized_terms[name] for name in denominators}
        detached_loss = torch.zeros_like(next(iter(denominators.values())))
        for name in denominators:
            detached_loss += float(enabled[name].weight) * normalized_terms[name]
        gradient_evidence: dict[str, float] = {}
        if collect_z_gradients:
            epsilon = 1.0e-30
            raw_rho_sq = z_gradient_squares["density"]
            raw_kappa_sq = z_gradient_squares["kappa"]
            raw_dot = z_gradient_dot
            rho_scale = float(baselines["density"])
            kappa_scale = float(baselines["kappa"])
            normalized_rho_sq = raw_rho_sq / (rho_scale * rho_scale)
            normalized_kappa_sq = raw_kappa_sq / (kappa_scale * kappa_scale)
            normalized_dot = raw_dot / (rho_scale * kappa_scale)
            trace = normalized_rho_sq + normalized_kappa_sq
            determinant = max(normalized_rho_sq * normalized_kappa_sq - normalized_dot * normalized_dot, 0.0)
            discriminant = max(trace * trace - 4.0 * determinant, 0.0)
            largest = 0.5 * (trace + math.sqrt(discriminant))
            smallest = 0.5 * (trace - math.sqrt(discriminant))
            joint_sq = normalized_rho_sq + normalized_kappa_sq + 2.0 * normalized_dot
            joint_norm = math.sqrt(max(joint_sq, 0.0))
            gradient_evidence = {
                "raw/rho_norm": math.sqrt(max(raw_rho_sq, 0.0)),
                "raw/kappa_norm": math.sqrt(max(raw_kappa_sq, 0.0)),
                "raw/dot": raw_dot,
                "raw/cosine": raw_dot / math.sqrt(max(raw_rho_sq * raw_kappa_sq, epsilon)),
                "normalized/rho_norm": math.sqrt(max(normalized_rho_sq, 0.0)),
                "normalized/kappa_norm": math.sqrt(max(normalized_kappa_sq, 0.0)),
                "normalized/dot": normalized_dot,
                "normalized/cosine": normalized_dot
                / math.sqrt(max(normalized_rho_sq * normalized_kappa_sq, epsilon)),
                "normalized/gram_determinant": determinant,
                "normalized/gram_condition": largest / max(smallest, epsilon),
                "vanilla/joint_norm": joint_norm,
                "vanilla/rho_projection": (normalized_rho_sq + normalized_dot) / max(joint_norm, epsilon),
                "vanilla/kappa_projection": (normalized_kappa_sq + normalized_dot) / max(joint_norm, epsilon),
            }
        diagnostics = {
            name: numerator / max(denominator, 1.0)
            for name, (numerator, denominator) in diagnostic_totals.items()
        }
        for name in (
            "density/prediction_square",
            "density/target_square",
            "kappa/prediction_square",
            "kappa/target_square",
        ):
            diagnostics[name.replace("_square", "_rms")] = math.sqrt(max(diagnostics.pop(name), 0.0))
        return MethodUpdate(
            loss=detached_loss.detach(),
            terms=terms,
            sample_count=sample_count,
            denominators={name: float(value) for name, value in denominators.items()},
            normalized_terms=normalized_terms,
            skills=skills,
            gradient_evidence=gradient_evidence,
            diagnostics=diagnostics,
        )

    def evaluate_session(
        self,
        session: MultiAnchorGaussianSession,
        schedule: Any,
        *,
        include_ablations: bool = False,
    ) -> MethodEvaluationReport:
        r"""流式执行固定 $A^{(0)}$/4-16-64 mm/4+4 edge 测度。

        Trainer 只决定何时评估；本函数拥有固定 sigma、anchor、edge、
        分层轴与具体 ablation。所有 objective 只累计 detached numerator/denominator，不保留跨 batch 图。
        """

        from anymani.distill.diagnostics.evaluation.geometry_ssl import (
            aggregate_geometry_ssl_stratified_components,
            geometry_ssl_stratified_components_per_sample,
        )

        from .evaluation import fixed_evaluation_ablation_evidence, update_evaluation_digest

        self.eval_mode()
        totals: dict[str, list[float]] = {}
        stratified_blocks: list[tuple[tuple[str, ...], Any]] = []
        ablation_records: list[dict[str, object]] = []
        ablation_names: tuple[str, ...] | None = None
        digest = hashlib.sha256(b"multi-anchor-fixed-evaluation-v1\0")
        block_index = 0
        with torch.no_grad():
            while not schedule.complete:
                batch = session.realize(schedule.next(), schedule=schedule, step=block_index)
                step_result, prediction = self._forward_with_prediction(batch, step=block_index, mode="eval")
                for objective in step_result.objectives.values():
                    for component in objective.components:
                        current = totals.setdefault(component.name, [0.0, 0.0])
                        current[0] += float(component.numerator.detach())
                        current[1] += float(component.denominator.detach())
                stratified_blocks.append(
                    (batch.asset_ids, geometry_ssl_stratified_components_per_sample(prediction, batch))
                )
                update_evaluation_digest(digest, batch)
                if include_ablations:
                    with torch.no_grad():
                        evidence = fixed_evaluation_ablation_evidence(self.require_model(), (batch,))
                    raw_names = evidence.get("ablations")
                    if not isinstance(raw_names, (tuple, list)):
                        raise ValueError("method ablation evidence names must be a sequence")
                    names = tuple(str(name) for name in raw_names)
                    ablation_names = names if ablation_names is None else ablation_names
                    if names != ablation_names:
                        raise ValueError("method ablation names changed within one evaluation session")
                    raw_records = evidence.get("records")
                    if not isinstance(raw_records, list):
                        raise ValueError("method ablation evidence records must be a list")
                    for record in raw_records:
                        copied = dict(record)
                        copied["block_index"] = block_index
                        ablation_records.append(copied)
                block_index += 1
        strata = aggregate_geometry_ssl_stratified_components(tuple(stratified_blocks))
        raw_metrics = strata.get("metric_scores")
        if not isinstance(raw_metrics, dict):
            raise ValueError("method evaluation strata lack metric_scores")
        metrics = {str(name): float(value) for name, value in raw_metrics.items()}
        strata["objective_terms"] = {
            name: numerator / denominator for name, (numerator, denominator) in totals.items()
        }
        strata["bank_digest_sha256"] = digest.hexdigest()
        ablations = None
        if include_ablations:
            ablations = {
                "split": f"{session.role}.{session.suite}" if session.suite else session.role,
                "pairing_key": ["asset_id", "q_index"],
                "ablations": ablation_names or (),
                "records": ablation_records,
            }
        return MethodEvaluationReport(metrics=metrics, strata=strata, ablations=ablations)

    def analyze_ablations(
        self,
        evidence: Mapping[str, Any],
        *,
        bootstrap_replicates: int,
        seed: int,
    ) -> dict[str, Any]:
        r"""执行 morphology/q 两级配对 bootstrap，不把 geometry 统计暴露给 Trainer。"""

        from anymani.distill.diagnostics.analysis.geometry_ssl import analyze_geometry_ssl_ablation_evidence

        return analyze_geometry_ssl_ablation_evidence(
            dict(evidence),
            bootstrap_samples=bootstrap_replicates,
            seed=seed,
        )

    def feature_spec(self) -> FeatureSpec:
        r"""返回下游消费的 unified entity sequence 与 JOINT gather view 合同。"""

        return FeatureSpec(
            entity_width=self.config.model.encoder.backbone.hidden_width,
        )

    def retained_state_dict(self) -> dict[str, torch.Tensor]:
        r"""返回只含 retained encoder namespace 的 standalone transfer state。"""

        return self.require_model().retained_state_dict()

    def training_state_dict(self) -> dict[str, torch.Tensor]:
        r"""返回 encoder 与两个 SSL-only readers 的完整 resume state。"""

        return self.require_model().state_dict()

    def load_training_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        r"""严格恢复完整 learned state；任一 namespace 漂移均失败。"""

        self.require_model().load_state_dict(dict(state), strict=True)

    def retained_artifact_payload(
        self,
        *,
        metadata: Mapping[str, Any],
        source_checkpoint: Path,
    ) -> dict[str, Any]:
        r"""构造只含 retained encoder 的 method-owned standalone artifact。"""

        if not source_checkpoint.is_file():
            raise FileNotFoundError(f"retained artifact source checkpoint does not exist: {source_checkpoint}")
        retained = self.retained_state_dict()
        if not retained or any(not key.startswith("encoder.") for key in retained):
            raise ValueError("retained artifact requires a non-empty encoder-only state")
        return {
            "schema_version": "5.0.0",
            "artifact_type": "retained_geometry_encoder",
            "retained_state": retained,
            "retained_model_config": {"encoder": asdict(self.config.model.encoder)},
            "feature_spec": asdict(self.feature_spec()),
            "input_contract": {
                "frame": "query/closest/surface in hand frame {h}",
                "units": "length=m,joint=rad,density=dimensionless,kappa=m/rad",
                "retained_inputs": "physical q + static geometry evidence",
            },
            "lineage": {
                "source_checkpoint": str(source_checkpoint),
                "code_revision": metadata.get("code_revision", "unknown"),
                "package_version": metadata.get("package_version", "unknown"),
                "geometry_semantics_schema": metadata.get("geometry_semantics_schema", "unknown"),
                "asset_manifest": dict(metadata.get("asset_manifest", {})),
                "dataset_identity": dict(metadata.get("dataset_identity", {})),
            },
        }

    def close(self) -> None:
        r"""关闭全部 core prefetch executors 并释放共享 arena；GPU lease 已由 session teardown。"""

        providers = [
            *(tuple([self.train_sources]) if self.train_sources is not None else ()),
            *self.validation_sources.values(),
            *self.evaluation_sources.values(),
        ]
        seen: set[int] = set()
        for provider in providers:
            if id(provider) not in seen:
                provider.close()
                seen.add(id(provider))
        self.source_cache.clear()


MultiAnchorGaussianMethodCfg.runtime_type = MultiAnchorGaussianMethod  # type: ignore[misc, assignment]


__all__ = [
    "MultiAnchorGaussianMethod",
    "MultiAnchorGaussianSession",
    "_derive_padding",
    "_forward_microbatch_samples",
]
