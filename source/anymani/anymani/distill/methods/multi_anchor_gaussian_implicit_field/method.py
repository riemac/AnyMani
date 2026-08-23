r"""多锚点 Gaussian 隐式场的科学聚合根。

本类对内显式耦合 representation、model 与 objectives，对外只给 SSL trainer 封闭接口：
prepare、realize_minibatch、forward_objectives、reduce_update、evaluate、retained export。
Trainer 不得读取 `representation.config.field` 或 padding layout。
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, replace
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
from anymani.distill.representations.sources.geometry_source import GeometrySource
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
from .objectives import evaluate_objectives, reduce_method_steps
from .state_measure import SobolJointSampler

_FORWARD_MICROBATCH_SAMPLES = 16
"""一次保留在 GPU activation graph 中的 `(asset,q)` 样本上限。"""

_DEVICE_SUBWINDOW_ASSETS = 8
"""单次 device source/Warp lease 的资产上限；不改变 logical resident window 顺序。"""


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
    r"""由 typed semantics 推导 padding，不物化 owner collision geometry。

    padding 只依赖 JOINT/TIP 数量和 graph distance，不依赖 surface、anchor 或 query。
    因而这里直接 lower 轻量运动学规格，避免小规模探针为了得到全局 shape 上限而提前
    生成全部静态 geometry source。
    """

    if not assets:
        raise ValueError("padding derivation requires at least one materialized source")
    max_joint = 0
    max_tip = 0
    for asset in assets:
        semantics = asset.geometry_semantics
        if semantics is None:
            raise ValueError(f"asset {asset.asset_id!r} is missing geometry semantics")
        spec = lower_hand_geometry_semantics(semantics, dtype=torch.float64)
        max_joint = max(max_joint, int(spec.space_screws.shape[0]))
        max_tip = max(max_tip, sum(role == "tip" for role in spec.owner_roles))
    if max_joint < 1 or max_tip < 1:
        raise ValueError("resolved dataset must contain at least one JOINT and one TIP owner")
    return GeometryPaddingCfg(
        max_joint_count=max_joint,
        max_tip_count=max_tip,
        max_graph_distance=max_graph_distance,
    )


class LazyGeometrySources(Sequence[GeometrySource]):
    r"""把 resolved HandContainer 轴映射为按资产 demand-load 的 CPU source 轴。

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
        materialize: Callable[[HandContainer], GeometrySource],
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

    def __len__(self) -> int:
        r"""返回不触发 source 物化的资产数量。"""

        return len(self.assets)

    @overload
    def __getitem__(self, index: int) -> GeometrySource: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[GeometrySource, ...]: ...

    def __getitem__(self, index: int | slice) -> GeometrySource | tuple[GeometrySource, ...]:
        r"""按索引读取 source；slice 只在显式请求时展开对应范围。"""

        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        asset = self.assets[index]
        return self.cache.load_or_create(
            asset,
            config=self.config,
            materialize=lambda asset=asset: self.materialize(asset),
        )

    def get(self, asset_id: str) -> GeometrySource:
        r"""按稳定 asset ID 读取 source，供 resident window demand-load。"""

        try:
            return self[self._index_by_id[asset_id]]
        except KeyError as exc:
            raise KeyError(f"unknown geometry asset ID={asset_id!r}") from exc

    def prefetch(self, asset_ids: Sequence[str]) -> None:
        r"""并行准备 resident window 的 CPU source，GPU 上传仍由主线程完成。

        每项资产继续通过 arena 的 per-key 锁保证幂等；线程数固定为 2 的上限。
        owner union 与 mesh 文件访问同时运行时会争用 CPU 内存带宽，过度并行反而延长
        总 wall time，因此这里使用保守的双 worker，而不是按逻辑核数扩张。
        """

        requested = tuple(asset_ids)
        if not requested:
            return
        worker_count = min(2, len(requested))
        started = perf_counter()
        print(f"[Source] CPU prefetch start: assets={len(requested)} workers={worker_count}", flush=True)
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="source-prefetch") as executor:
            futures = [executor.submit(self.get, asset_id) for asset_id in requested]
            for index, future in enumerate(futures, start=1):
                future.result()
                if index % 8 == 0 or index == len(futures):
                    print(f"[Source] CPU prefetch progress: {index}/{len(futures)}", flush=True)
        print(f"[Source] CPU prefetch done: seconds={perf_counter() - started:.3f}", flush=True)


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
    r"""显式装配 GeometryRepresentation、GeometrySSLModel 与五项 objective。"""

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

    def prepare(self, catalog: Any, *, device: torch.device, dtype: torch.dtype) -> None:
        r"""建立 train/validation/evaluation source provider，并推导全局 padding。"""

        del device, dtype
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
            materialize=representation.materialize_source,
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
                representation.materialize_source(item.container),
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
        r"""启用固定评估行为，但保留 Sobolev/JVP 所需的局部 autograd。"""

        self.require_model().eval()

    def require_padding(self) -> GeometryPaddingCfg:
        r"""返回已由 dataset/model 推导的 padding 上限。"""

        if self.padding is None:
            raise RuntimeError("multi-anchor method padding has not been derived")
        return self.padding

    def load_device_state(self, source: GeometrySource, *, device: torch.device | str, dtype: torch.dtype):
        r"""把一项 CPU source 上传为训练期 device state；trainer 不直接读 representation。"""

        return self.representation.to_device(source, device=device, dtype=dtype)

    def load_validation_device_state(
        self, source: GeometrySource, *, device: torch.device | str, dtype: torch.dtype
    ):
        r"""把一项 CPU source 上传为固定 validation sigma 的 device state。"""

        return self.validation_representation.to_device(source, device=device, dtype=dtype)

    def declared_objective_weights(self) -> dict[str, float]:
        r"""返回 OBJECTIVES_CFG 中显式写出的五项权重。"""

        return {name: float(term.weight) for name, term in self.config.objectives.enabled().items()}

    def formula_identity(self) -> dict[str, str]:
        r"""返回五项 objective 公式身份：模块级函数的完整限定名。"""

        return {name: term.qualified_func_name() for name, term in self.config.objectives.enabled().items()}

    def runtime_resource_evidence(self) -> dict[str, dict[str, int]]:
        r"""返回 CPU source arena 的命中、驱逐与硬容量证据。"""

        return {"geometry_source_arena": self.source_cache.stats()}

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
        for chunk_start in range(0, len(logical_indices), _DEVICE_SUBWINDOW_ASSETS):
            asset_chunk = logical_indices[chunk_start : chunk_start + _DEVICE_SUBWINDOW_ASSETS]
            states = window.ensure(tuple(catalog_ids[index] for index in asset_chunk))
            states_by_id = {state.source.asset_id: state for state in states}
            for asset_index in asset_chunk:
                source = sources[asset_index]
                state = states_by_id[source.asset_id]
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
        return pad_online_geometry_samples(samples, padding=padding)

    def forward_objectives(self, batch: PaddedOnlineGeometryBatch, *, step: int, mode: str = "train") -> MethodStep:
        r"""完成一次 logical minibatch 前向，并在 GPU 上按 64 samples 做 microbatch。

        logical minibatch 的 sampling budget、joint-sign rewrite 随机身份和 additive
        objective 统计不变；microbatch 只限制单次 encoder activation 的峰值显存。
        """

        if batch.q.shape[0] <= _FORWARD_MICROBATCH_SAMPLES:
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
            microbatch_size=_FORWARD_MICROBATCH_SAMPLES,
        ):
            micro_step = self._forward_with_prediction(
                microbatch,
                step=step,
                mode=mode,
                apply_augmentation=False,
            )[0]
            steps.append(_detach_method_step(micro_step) if mode == "calibration" else micro_step)
        return _merge_microbatch_steps(tuple(steps))

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
        q = q.detach().requires_grad_(True)
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
            model=model,
            q=q,
            prediction=prediction,
            batch=batch,
            create_graph=mode != "calibration",
        )
        results = evaluate_objectives(context, self.config.objectives)
        return MethodStep(objectives=results, sample_count=int(batch.q.shape[0])), prediction

    def reduce_update(self, steps: tuple[MethodStep, ...]) -> MethodUpdate:
        r"""按 $(asset,q)$ 等权合并一个 optimizer update。"""

        return reduce_method_steps(steps, self.config.objectives)

    def evaluate_session(
        self,
        session: MultiAnchorGaussianSession,
        schedule: Any,
        *,
        include_ablations: bool = False,
    ) -> MethodEvaluationReport:
        r"""流式执行固定 $A^{(0)}$/4-16-64 mm/4+4 edge 测度。

        Trainer 只决定何时评估和如何使用三项 selection metrics；本函数拥有固定 sigma、anchor、edge、
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
        with torch.enable_grad():
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
        r"""返回下游消费的零阶实体序列与逐 JOINT 一阶序列合同。"""

        heads = self.config.model.encoder.heads
        return FeatureSpec(
            zero_order_width=heads.zero_order_width,
            first_order_width=heads.first_order_width,
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
            "schema_version": "4.0.0",
            "artifact_type": "retained_geometry_encoder",
            "retained_state": retained,
            "retained_model_config": {"encoder": asdict(self.config.model.encoder)},
            "feature_spec": asdict(self.feature_spec()),
            "input_contract": {
                "frame": "query/closest/surface in hand frame {h}",
                "units": "length=m,joint=rad,density=dimensionless,kappa=m/rad,g=1/rad",
                "retained_inputs": "physical q + static geometry evidence",
            },
            "lineage": {
                "source_checkpoint": str(source_checkpoint),
                "code_revision": metadata.get("code_revision", "unknown"),
                "package_version": metadata.get("package_version", "unknown"),
                "geometry_semantics_schema": metadata.get("geometry_semantics_schema", "unknown"),
                "asset_manifest": dict(metadata.get("asset_manifest", {})),
            },
        }

    def close(self) -> None:
        r"""释放共享 CPU source arena；GPU lease 已由各 session window 先行 teardown。"""

        self.source_cache.clear()


MultiAnchorGaussianMethodCfg.runtime_type = MultiAnchorGaussianMethod  # type: ignore[misc, assignment]


__all__ = ["MultiAnchorGaussianMethod", "MultiAnchorGaussianSession", "_derive_padding"]
