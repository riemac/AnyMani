r"""多锚点 Gaussian 隐式场的科学聚合根。

本类对内显式耦合 representation、model 与 objectives，对外只给 SSL trainer 封闭接口：
prepare、realize_minibatch、forward/backward update、reduce、evaluate、retained export。
Trainer 不得读取 `representation.config.field` 或 padding layout。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from threading import Event, Lock
from typing import Any

import torch

from anymani.assets.bank.hand_container import HandContainer
from anymani.distill.methods.contracts import (
    FeatureSpec,
    MethodEvaluationReport,
    MethodParameterGroup,
    MethodStep,
    MethodUpdate,
)
from anymani.distill.models.geometry_ssl import GeometrySSLForward, GeometrySSLModel
from anymani.distill.models.input_adapters.geometry import GeometryPaddingCfg
from anymani.distill.objectives.contracts import AdditiveStatistic, ObjectiveTermResult
from anymani.distill.representations.geometry import GeometryRepresentation
from anymani.distill.representations.sources.artifacts import GeometrySourceArtifactStore
from anymani.distill.representations.sources.cache import GeometrySourceArena
from anymani.distill.representations.sources.geometry_source import GeometrySource, GeometrySourceCore
from anymani.distill.representations.targets.geometry_field import fixed_validation_gaussian_field_config

from .artifact import build_retained_geometry_artifact
from .augmentation import maybe_rewrite_batch, sample_entity_permutation
from .batch import (
    PaddedOnlineGeometryBatch,
    attach_static_evidence_block,
    method_batch_views,
    pad_online_geometry_blocks,
    split_padded_online_geometry_batch,
)
from .config import MultiAnchorGaussianMethodCfg
from .context import MultiAnchorObjectiveContext
from .evaluation import (
    analyze_ablations,
    evaluate_method_session,
    evaluate_z_compression_session,
    fit_z_compression_basis,
)
from .objectives import (
    evaluate_objectives,
    finalize_teacher_baselines,
    merge_teacher_baseline_statistics,
    reduce_method_steps,
    teacher_baseline_sufficient_statistics,
)
from .source_runtime import (
    LazyGeometrySources,
    LazySobolSamplers,
    MultiAnchorGaussianSession,
    PhysicalAuditHandle,
    _derive_padding,
    materialize_or_load_core,
)
from .source_runtime import (
    asset_manifest as build_asset_manifest,
)
from .source_runtime import (
    configure_source_artifacts as configure_method_source_artifacts,
)
from .source_runtime import (
    lazy_sources as build_lazy_sources,
)
from .source_runtime import (
    preflight_source_artifacts as preflight_method_source_artifacts,
)
from .source_runtime import (
    prepare_source_artifacts as prepare_method_source_artifacts,
)
from .source_runtime import (
    require_train_sources as require_method_train_sources,
)
from .source_runtime import (
    source_artifact_identity as method_source_artifact_identity,
)
from .source_runtime import (
    source_partitions as method_source_partitions,
)
from .source_runtime import (
    split_asset_count as method_split_asset_count,
)
from .source_runtime import (
    split_names as method_split_names,
)
from .source_runtime import (
    start_physical_audit as start_method_physical_audit,
)
from .training import _q_per_asset_block, backward_method_update

_TRAIN_FORWARD_MICROBATCH_SAMPLES = 64
"""rho/kappa 普通参数反向的 `(asset,q)` 样本上限。"""

_EVALUATION_FORWARD_MICROBATCH_SAMPLES = 64
"""固定评估的单次样本上限，与训练使用同一张量形状合同。"""

_DEVICE_SUBWINDOW_ASSETS = 8
"""单次 device source/Warp lease 的资产上限；不改变 logical resident window 顺序。"""


def _forward_microbatch_samples(mode: str) -> int:
    r"""按运行阶段返回 rho/kappa 普通前向的 sample 上限。

    训练逐块立即反向，以保持完整 minibatch 的精确充分统计，
    evaluation 在 ``no_grad`` 下复用同一块大小。
    """

    if mode == "train":
        return _TRAIN_FORWARD_MICROBATCH_SAMPLES
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


class MultiAnchorGaussianMethod:
    r"""显式装配 GeometryRepresentation、GeometrySSLModel 与 rho/kappa 双 objective。"""

    _q_per_asset_block = staticmethod(_q_per_asset_block)  # forward/backward 共用 q-block layout contract

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
        self.source_artifact_store: GeometrySourceArtifactStore | None = None
        self._source_artifact_lock = Lock()
        self._base_artifact_refs: dict[str, object] = {}
        self._pending_source_artifact_refs: list[dict[str, object]] = []
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

    def configure_source_artifacts(
        self,
        *,
        root: str,
        mode: str,
        dataset_manifest_sha256: str,
        producer_device: str,
    ) -> None:
        r"""配置跨 run source store；Trainer 不读取 artifact 的 geometry 内部字段。"""

        configure_method_source_artifacts(
            self,
            root=root,
            mode=mode,
            dataset_manifest_sha256=dataset_manifest_sha256,
            producer_device=producer_device,
        )

    def source_artifact_identity(self) -> dict[str, object]:
        """返回 checkpoint/stage 可比较的 source producer 身份；off 模式显式保留。"""

        return method_source_artifact_identity(self)

    def _materialize_or_load_core(
        self,
        container: HandContainer,
        representation: GeometryRepresentation,
    ) -> GeometrySourceCore:
        """按 cache mode 读取 base；只有 read-write miss/corruption 才允许物化并发布。"""

        return materialize_or_load_core(self, container, representation)

    def _lazy_sources(
        self,
        assets: Sequence[HandContainer],
        representation: GeometryRepresentation,
    ) -> LazyGeometrySources:
        r"""建立一个不触发 source IO 的 provider。"""

        return build_lazy_sources(self, assets, representation)

    def _source_partitions(self) -> dict[str, tuple[LazyGeometrySources, int]]:
        """返回 prepare CLI/preflight 的 source provider 与所需 anchor shard 数。"""

        return method_source_partitions(self)

    def prepare_source_artifacts(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        partitions: tuple[str, ...] = (),
    ) -> dict[str, object]:
        r"""离线构建 base 与 train 0..7 / held-out bank-0 shards，并报告冷构建成本。"""

        return prepare_method_source_artifacts(
            self,
            device=device,
            dtype=dtype,
            partitions=partitions,
        )

    def preflight_source_artifacts(self) -> dict[str, int]:
        r"""在模型初始化前完整校验 formal run 所需 base/shards，任一 miss/corruption fail closed。"""

        return preflight_method_source_artifacts(self)

    def split_names(self, role: str) -> tuple[str, ...]:
        r"""返回 validation/evaluation 的具名 suites；train 没有 suite 子轴。"""

        return method_split_names(self, role)

    def require_train_sources(self) -> LazyGeometrySources:
        r"""返回 prepare 后的 train provider；生命周期顺序错误时 fail-fast。"""

        return require_method_train_sources(self)

    def split_asset_count(self, role: str, *, suite: str = "") -> int:
        r"""返回 train 或具名 held-out suite 的真实资产数。"""

        return method_split_asset_count(self, role, suite=suite)

    def asset_manifest(self, catalog: Any, *, cancel_event: Event | None = None) -> dict[str, Any]:
        r"""流式记录 physical source 与 train/held-out 隔离证据。

        audit 不遍历训练用 ``LazyGeometrySources``，避免完整 catalog 污染其 LRU 或与 device prefetch
        争夺同一 arena。每项 source 在 ``record`` 返回后即可释放，只保留小型 hash/provenance mapping。
        """

        return build_asset_manifest(self, catalog, cancel_event=cancel_event)

    def start_physical_audit(self, catalog: Any) -> PhysicalAuditHandle:
        r"""启动完整 physical manifest 计算；结果由 checkpoint/artifact gate 等待。"""

        return start_method_physical_audit(self, catalog)

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
        resource_profile: bool = False,
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
            resource_profile=resource_profile,
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

    def optimizer_parameter_groups(self) -> tuple[MethodParameterGroup, ...]:
        r"""显式返回 retained shared encoder 与两个 disposable private readers。

        该边界使 Trainer 无需依赖 ``encoder.`` 等字符串命名；返回前同时验证三组互斥且完整覆盖
        所有可训练参数，防止新模块被静默遗漏在 optimizer 之外。
        """

        model = self.require_model()
        groups = (
            MethodParameterGroup("shared_encoder", tuple(model.encoder.parameters())),
            MethodParameterGroup("density_reader", tuple(model.density_decoder.parameters())),
            MethodParameterGroup("kappa_reader", tuple(model.sensitivity_decoder.parameters())),
        )
        grouped = tuple(parameter for group in groups for parameter in group.parameters if parameter.requires_grad)
        trainable = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
        if len({id(parameter) for parameter in grouped}) != len(grouped):
            raise RuntimeError("Geometry SSL optimizer parameter groups overlap")
        if {id(parameter) for parameter in grouped} != {id(parameter) for parameter in trainable}:
            raise RuntimeError("Geometry SSL optimizer parameter groups do not cover the trainable model")
        return groups

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

    def load_device_state(
        self,
        source: GeometrySourceCore,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        bank_index: int = 0,
    ):
        r"""把一项 CPU core 完成 anchors 并上传为训练期 device state。"""

        state = self._load_device_state_with_artifact(
            source,
            representation=self.representation,
            bank_index=bank_index,
            device=device,
            dtype=dtype,
        )
        self._record_anchor_classification(state)
        return state

    def load_validation_device_state(
        self,
        source: GeometrySourceCore,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        bank_index: int = 0,
    ):
        r"""把一项 CPU core 完成 anchors 并上传为固定 validation sigma 的 device state。"""

        state = self._load_device_state_with_artifact(
            source,
            representation=self.validation_representation,
            bank_index=bank_index,
            device=device,
            dtype=dtype,
        )
        self._record_anchor_classification(state)
        return state

    def _load_device_state_with_artifact(
        self,
        core: GeometrySourceCore,
        *,
        representation: GeometryRepresentation,
        bank_index: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ):
        r"""从 selected shard 构造 resident state；read-write miss 才调用 anchor classifier。"""

        store = self.source_artifact_store
        if store is None:
            return representation.to_device(core, device=device, dtype=dtype, bank_index=bank_index)
        try:
            realization, stats, anchor_reference = store.load_anchor(
                core.container,
                self.config.representation.source,
                bank_index,
            )
            source = GeometrySource.from_core(
                core,
                anchor_bank=(realization.samples,),
                anchor_realization=realization,
            )
            device_source = source.to_device(device=device, dtype=dtype)
            state = representation.assemble_device_state(
                source,
                device_source,
                anchor_stats=stats,
                device=device,
                dtype=dtype,
            )
        except (FileNotFoundError, ValueError):
            if store.mode != "read-write":
                raise
            source, device_source, stats = core.finalize_selected_on_device(
                config=self.config.representation.source,
                bank_index=bank_index,
                device=device,
                dtype=dtype,
            )
            realization = source.anchor_realization
            if realization is None:
                device_source.release()
                raise RuntimeError("selected source finalization did not return AnchorRealization")
            store.write_anchor(core.container, self.config.representation.source, realization, stats)
            _loaded, _loaded_stats, anchor_reference = store.load_anchor(
                core.container,
                self.config.representation.source,
                bank_index,
            )
            state = representation.assemble_device_state(
                source,
                device_source,
                anchor_stats=stats,
                device=device,
                dtype=dtype,
            )
        with self._source_artifact_lock:
            base_reference = self._base_artifact_refs.get(core.asset_id)
            resident_realization = state.source.anchor_realization
            if resident_realization is None:
                raise RuntimeError("artifact-backed resident source lost selected anchor realization")
            self._pending_source_artifact_refs.append(
                {
                    "asset_id": core.asset_id,
                    "bank_index": bank_index,
                    "artifact_key": anchor_reference.artifact_key,
                    "base_manifest_digest": getattr(base_reference, "manifest_digest", ""),
                    "anchor_manifest_digest": anchor_reference.manifest_digest,
                    "anchor_realization_hash": resident_realization.realization_hash,
                    "input_fingerprint": core.identity.physical_geometry_hash,
                }
            )
        return state

    def drain_source_artifact_references(self) -> tuple[dict[str, object], ...]:
        """返回自上次 drain 后实际加载的 `(asset,bank)` artifact refs。"""

        with self._source_artifact_lock:
            references = tuple(self._pending_source_artifact_refs)
            self._pending_source_artifact_refs.clear()
        return references

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
        r"""返回 objective presence；schema-8 不再以 scalar task weight 表达 shared 优先级。"""

        return {name: 1.0 for name in self.config.objectives.enabled()}

    def formula_identity(self) -> dict[str, str]:
        r"""返回 density/kappa objective 公式身份：模块级函数的完整限定名。"""

        return {name: term.qualified_func_name() for name, term in self.config.objectives.enabled().items()}

    def optimization_identity(self) -> dict[str, object]:
        r"""向 Trainer 暴露 checkpoint 所需的 FairGrad 公式身份，不泄漏 concrete config。"""

        return {
            "algorithm": self.config.fairgrad.algorithm,
            "near_opposition_tolerance": self.config.fairgrad.near_opposition_tolerance,
        }

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
            selected_bank_index = 0 if mode != "train" else int(q_block_index % self.config.representation.source.anchors.bank_size)
            states = window.ensure(
                tuple(catalog_ids[index] for index in asset_chunk),
                prefetch_sources=False,
                prepared_sources={core.asset_id: core for core in current_cores},
                bank_index=selected_bank_index,
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
                realization = source.anchor_realization
                if realization is None:
                    raise ValueError(f"asset {source.asset_id!r} lacks selected anchor realization")
                anchor_index = realization.bank_index
                schedule_index = (
                    int(schedule_item.minibatch_index) * 1_000_003
                    + int(schedule_item.window_index) * 10_007
                    + int(schedule_item.asset_group)
                )
                physical = representation.sample(
                    state,
                    q,
                    sampling_seed=seed + schedule_index,
                    # q identity 属于 host-side Sobol/provenance；留在 CPU，避免采样类别与 1% audit 各自 D2H 同步。
                    q_index=torch.arange(q_start, q_start + q_count, device="cpu", dtype=torch.long),
                    anchor_index=anchor_index,
                    supervision_split="train" if mode == "train" else "eval",
                )
                samples.append(
                    attach_static_evidence_block(
                        physical,
                        source=source,
                        spec=state.spec,
                        anchors=realization.samples,
                        device=q.device,
                        dtype=q.dtype,
                        entity_permutation=(
                            sample_entity_permutation(
                                len(source.container.geometry_semantics.owners),
                                asset_id=asset_id,
                                q_block_start=q_start,
                                root_seed=seed + schedule_index,
                                config=self.config.entity_permutation,
                            )
                            if mode == "train" and source.container.geometry_semantics is not None
                            else None
                        ),
                    )
                )
            if next_handle is not None:
                prefetch_handle = next_handle
        return pad_online_geometry_blocks(samples, padding=padding)

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
            if mode == "train"
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
            steps.append(micro_step)
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
        if mode == "train" and apply_augmentation:
            batch = maybe_rewrite_batch(
                batch,
                config=self.config.joint_sign_rewrite,
                step=step,
                seed=step,
            )
        views = method_batch_views(batch)
        q, evidence, evidence_row_index = views.model_input
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
            evidence_row_index=evidence_row_index,
        )
        context = MultiAnchorObjectiveContext(
            prediction=prediction,
            batch=batch,
        )
        results = evaluate_objectives(context, self.config.objectives)
        return MethodStep(objectives=results, sample_count=int(batch.q.shape[0])), prediction

    def dense_snapshot(
        self,
        batch: PaddedOnlineGeometryBatch,
        *,
        microbatch_size: int,
    ) -> tuple[GeometrySSLForward, PaddedOnlineGeometryBatch]:
        r"""在当前参数下重放首个完整 q-block microbatch，供 checkpoint-cadence NPZ 记录。"""

        snapshot_batch = split_padded_online_geometry_batch(batch, microbatch_size=microbatch_size)[0]
        model = self.require_model()
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                _step, prediction = self._forward_with_prediction(
                    snapshot_batch,
                    step=0,
                    mode="eval",
                    apply_augmentation=False,
                )
        finally:
            model.train(was_training)
        return prediction, snapshot_batch

    def reduce_update(self, steps: tuple[MethodStep, ...]) -> MethodUpdate:
        r"""按 $(asset,q)$ 等权合并一个 optimizer update。"""

        return reduce_method_steps(steps, self.config.objectives)

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

        return backward_method_update(
            self,
            batch,
            forward_step=forward_step,
            microbatch_size=microbatch_size,
            collect_z_gradients=collect_z_gradients,
            rewrite_batch_fn=maybe_rewrite_batch,
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

        return evaluate_method_session(
            self,
            session,
            schedule,
            include_ablations=include_ablations,
        )

    def fit_z_compression_basis(self, session: MultiAnchorGaussianSession, schedule: Any):
        r"""在独立 training-q bank 上流式拟合一个统一 PALM/JOINT/TIP PCA basis。"""

        return fit_z_compression_basis(self, session, schedule)

    def evaluate_z_compression_session(
        self,
        session: MultiAnchorGaussianSession,
        schedule: Any,
        *,
        basis: Any,
        ranks: tuple[int, ...],
    ) -> dict[str, object]:
        r"""在固定 suite bank 上以原 readers 重放 32/64/96/128 维 $Z$。"""

        return evaluate_z_compression_session(
            self,
            session,
            schedule,
            basis=basis,
            ranks=ranks,
        )

    def analyze_ablations(
        self,
        evidence: Mapping[str, Any],
        *,
        bootstrap_replicates: int,
        seed: int,
    ) -> dict[str, Any]:
        r"""执行 morphology/q 两级配对 bootstrap，不把 geometry 统计暴露给 Trainer。"""

        return analyze_ablations(
            evidence,
            bootstrap_replicates=bootstrap_replicates,
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

        return build_retained_geometry_artifact(
            self,
            metadata=metadata,
            source_checkpoint=source_checkpoint,
        )

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
