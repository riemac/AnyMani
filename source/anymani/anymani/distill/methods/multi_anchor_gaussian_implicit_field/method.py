"""Shared source, cache, session, and evaluation lifecycle inherited by N040."""


from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from threading import Event, Lock
from typing import Any

import torch
from torch._functorch import config as functorch_config  # pyright: ignore[reportPrivateImportUsage]

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
from anymani.distill.representations.targets.geometry_field import fixed_gaussian_field_config

from .artifact import build_retained_geometry_artifact
from .augmentation import maybe_rewrite_batch, sample_entity_permutation
from .batch import (
    OnlineGeometrySample,
    PaddedOnlineGeometryBatch,
    attach_static_evidence_block,
    method_batch_views,
    pad_online_geometry_blocks,
    restore_padded_batch_from_replay,
    split_padded_online_geometry_batch,
    stage_padded_batch_for_replay,
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
from .training import _q_per_asset_block, backward_method_update, backward_method_update_units

_TRAIN_FORWARD_MICROBATCH_SAMPLES = 64
"""Maximum asset-state pairs per ordinary density/gradient backward pass."""

_EVALUATION_FORWARD_MICROBATCH_SAMPLES = 64
"""Maximum evaluation batch size under the shared tensor contract."""

_DEVICE_SUBWINDOW_ASSETS = 8
"""Maximum assets per device/Warp lease; preserve logical window order."""


def _forward_microbatch_samples(mode: str) -> int:


    if mode == "train":
        return _TRAIN_FORWARD_MICROBATCH_SAMPLES
    return _EVALUATION_FORWARD_MICROBATCH_SAMPLES


def _merge_microbatch_steps(steps: tuple[MethodStep, ...]) -> MethodStep:


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


    _q_per_asset_block = staticmethod(_q_per_asset_block)

    def __init__(self, config: MultiAnchorGaussianMethodCfg) -> None:


        self.config = config
        self.representation = GeometryRepresentation(config.representation)
        self.fixed_representation = GeometryRepresentation(
            replace(
                config.representation,
                field=fixed_gaussian_field_config(config.representation.field),
            )
        )
        self.model: GeometrySSLModel | None = None
        self.execution_policy: Any | None = None
        self._compiled_forward: Any | None = None
        self.train_sources: LazyGeometrySources | None = None
        self.source_cache = GeometrySourceArena()
        self.evaluation_sources: dict[str, LazyGeometrySources] = {}
        self.active_role: str | None = None
        self.padding: GeometryPaddingCfg | None = None
        self.runtime_device: torch.device | None = None
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
        }

    def prepare(self, catalog: Any, *, role: str, device: torch.device, dtype: torch.dtype) -> None:


        del dtype
        if role not in {"train", "evaluation"}:
            raise ValueError(f"Geometry SSL method role must be train or evaluation, got {role!r}")
        if self.active_role is not None:
            raise RuntimeError("Geometry SSL method can prepare exactly one runtime role")
        self.active_role = role
        self.runtime_device = torch.device(device)
        if role == "train":
            print(f"[Method] Indexing lazy train sources: {len(catalog.train)} assets")
            self.train_sources = self._lazy_sources(catalog.train, self.representation)
            active_assets = tuple(catalog.train)
        else:
            print("[Method] Indexing lazy evaluation sources...")
            self.evaluation_sources = {
                suite_name: self._lazy_sources(suite_assets, self.fixed_representation)
                for suite_name, suite_assets in catalog.evaluation.items()
            }
            active_assets = tuple(asset for assets in catalog.evaluation.values() for asset in assets)
        self.padding = _derive_padding(
            active_assets,
            max_graph_distance=self.config.model.encoder.backbone.max_graph_distance,
        )

    def configure_source_artifacts(
        self,
        *,
        root: str,
        mode: str,
        dataset_manifest_sha256: str,
        producer_device: str,
        role: str = "train",
    ) -> None:


        configure_method_source_artifacts(
            self,
            root=root,
            mode=mode,
            dataset_manifest_sha256=dataset_manifest_sha256,
            producer_device=producer_device,
            role=role,
        )

    def source_artifact_identity(self) -> dict[str, object]:


        return method_source_artifact_identity(self)

    def _materialize_or_load_core(
        self,
        container: HandContainer,
        representation: GeometryRepresentation,
    ) -> GeometrySourceCore:


        return materialize_or_load_core(self, container, representation)

    def _lazy_sources(
        self,
        assets: Sequence[HandContainer],
        representation: GeometryRepresentation,
    ) -> LazyGeometrySources:


        return build_lazy_sources(self, assets, representation)

    def _source_partitions(self) -> dict[str, tuple[LazyGeometrySources, int]]:


        return method_source_partitions(self)

    def prepare_source_artifacts(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        partitions: tuple[str, ...] = (),
    ) -> dict[str, object]:


        return prepare_method_source_artifacts(
            self,
            device=device,
            dtype=dtype,
            partitions=partitions,
        )

    def preflight_source_artifacts(self) -> dict[str, int]:


        return preflight_method_source_artifacts(self)

    def split_names(self, role: str) -> tuple[str, ...]:


        return method_split_names(self, role)

    def require_train_sources(self) -> LazyGeometrySources:


        return require_method_train_sources(self)

    def split_asset_count(self, role: str, *, suite: str = "") -> int:


        return method_split_asset_count(self, role, suite=suite)

    def asset_manifest(self, catalog: Any, *, cancel_event: Event | None = None) -> dict[str, Any]:


        return build_asset_manifest(self, catalog, cancel_event=cancel_event)

    def start_physical_audit(self, catalog: Any) -> PhysicalAuditHandle:


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


        if role == "train":
            sources = self.require_train_sources()
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


        return LazySobolSamplers(sources, seed=seed)

    def initialize_model(self, *, device: torch.device, dtype: torch.dtype) -> GeometrySSLModel:


        if self.model is not None:
            raise RuntimeError("multi-anchor method model is already initialized")
        self.model = GeometrySSLModel(self.config.model).to(device=device, dtype=dtype)
        if self.execution_policy is not None and bool(self.execution_policy.compile_enabled):
            # FairGrad obtains two task gradients from one activation graph. AOTAutograd's donated
            # buffers are incompatible with the required first retain_graph=True gradient call. The
            # setting must remain active after forward returns because the backward calls occur in
            # the method's streaming reducer, outside this initialization scope.
            functorch_config.donated_buffer = False
            self._compiled_forward = torch.compile(
                self.model,
                mode=str(self.execution_policy.compile_mode),
                fullgraph=True,
            )
        return self.model

    def configure_execution(self, policy: Any) -> None:


        if self.model is not None:
            raise RuntimeError("execution policy must be configured before model initialization")
        required = (
            "teacher_dtype",
            "parameter_dtype",
            "model_autocast_dtype",
            "loss_dtype",
            "fairgrad_accumulation_dtype",
            "allow_tf32",
            "compile_enabled",
            "compile_mode",
        )
        missing = tuple(name for name in required if not hasattr(policy, name))
        if missing:
            raise TypeError(f"Geometry SSL execution policy lacks fields={missing}")
        self.execution_policy = policy

    def require_model(self) -> GeometrySSLModel:


        if self.model is None:
            raise RuntimeError("multi-anchor method model has not been initialized")
        return self.model

    def parameters(self):


        return self.require_model().parameters()

    def optimizer_parameter_groups(self) -> tuple[MethodParameterGroup, ...]:


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


        self.require_model().train()

    def eval_mode(self) -> None:


        self.require_model().eval()

    def require_padding(self) -> GeometryPaddingCfg:


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


        state = self._load_device_state_with_artifact(
            source,
            representation=self.representation,
            bank_index=bank_index,
            device=device,
            dtype=dtype,
        )
        self._record_anchor_classification(state)
        return state

    def load_fixed_device_state(
        self,
        source: GeometrySourceCore,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        bank_index: int = 0,
    ):


        state = self._load_device_state_with_artifact(
            source,
            representation=self.fixed_representation,
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


        with self._source_artifact_lock:
            references = tuple(self._pending_source_artifact_refs)
            self._pending_source_artifact_refs.clear()
        return references

    def _record_anchor_classification(self, state: Any) -> None:


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


        return {name: 1.0 for name in self.config.objectives.enabled()}

    def formula_identity(self) -> dict[str, str]:


        return {name: term.qualified_func_name() for name, term in self.config.objectives.enabled().items()}

    def optimization_identity(self) -> dict[str, object]:


        return {
            "algorithm": self.config.fairgrad.algorithm,
            "near_opposition_tolerance": self.config.fairgrad.near_opposition_tolerance,
        }

    def runtime_resource_evidence(self) -> dict[str, object]:


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
                }
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


        blocks = [
            block
            for unit in self._realize_minibatch_blocks(
                schedule_item,
                sources=sources,
                samplers=samplers,
                window=window,
                seed=seed,
                schedule=schedule,
                mode=mode,
            )
            for block in unit
        ]
        return pad_online_geometry_blocks(blocks, padding=self.require_padding())

    def realize_minibatch_units(
        self,
        schedule_item: Any,
        *,
        sources: LazyGeometrySources,
        samplers: LazySobolSamplers,
        window: Any,
        seed: int,
        schedule: Any,
        mode: str = "train",
    ) -> Iterator[PaddedOnlineGeometryBatch]:


        padding = self.require_padding()
        for blocks in self._realize_minibatch_blocks(
            schedule_item,
            sources=sources,
            samplers=samplers,
            window=window,
            seed=seed,
            schedule=schedule,
            mode=mode,
        ):
            yield pad_online_geometry_blocks(blocks, padding=padding)

    def _realize_minibatch_blocks(
        self,
        schedule_item: Any,
        *,
        sources: LazyGeometrySources,
        samplers: LazySobolSamplers,
        window: Any,
        seed: int,
        schedule: Any,
        mode: str,
    ) -> Iterator[list[OnlineGeometrySample]]:


        del schedule
        representation = self.representation if mode == "train" else self.fixed_representation
        catalog_ids = sources.asset_ids
        resident_indices = tuple(schedule_item.resident_asset_indices)
        if not resident_indices:
            raise ValueError("schedule item must declare the complete resident window, not only the minibatch")
        q_block_index = int(schedule_item.q_block_index)
        resident_set = set(resident_indices)
        logical_indices = tuple(schedule_item.asset_indices)
        if any(asset_index not in resident_set for asset_index in logical_indices):
            raise ValueError("logical minibatch assets must belong to its declared resident window")
        asset_chunks = tuple(
            logical_indices[chunk_start : chunk_start + _DEVICE_SUBWINDOW_ASSETS]
            for chunk_start in range(0, len(logical_indices), _DEVICE_SUBWINDOW_ASSETS)
        )
        first_ids = tuple(catalog_ids[index] for index in asset_chunks[0])
        prefetch_handle = sources.prefetch_async(first_ids)
        for chunk_index, asset_chunk in enumerate(asset_chunks):
            samples: list[OnlineGeometrySample] = []
            current_cores = sources.await_prefetch(prefetch_handle)
            next_handle = None
            if chunk_index + 1 < len(asset_chunks):
                next_ids = tuple(catalog_ids[index] for index in asset_chunks[chunk_index + 1])
                next_handle = sources.prefetch_async(next_ids)
            selected_bank_index = 0 if mode != "train" else int(q_block_index % self.config.representation.source.anchors.bank_size)
            states = window.ensure(
                tuple(catalog_ids[index] for index in asset_chunk),
                prefetch_sources=False,
                prepared_sources={core.asset_id: core for core in current_cores},
                bank_index=selected_bank_index,
            )
            states_by_id = {state.source.asset_id: state for state in states}
            for asset_index in asset_chunk:
                asset_id = catalog_ids[asset_index]
                state = states_by_id[asset_id]
                source = state.source
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
            yield samples
            if next_handle is not None:
                prefetch_handle = next_handle

    def forward_objectives(
        self,
        batch: PaddedOnlineGeometryBatch,
        *,
        step: int,
        mode: str = "train",
        microbatch_size: int | None = None,
    ) -> MethodStep:


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


        return teacher_baseline_sufficient_statistics(batch)

    def merge_teacher_baseline_statistics(
        self,
        total: dict[str, torch.Tensor] | None,
        block: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:


        return merge_teacher_baseline_statistics(total, block)

    def finalize_teacher_baselines(self, statistics: dict[str, torch.Tensor]) -> dict[str, object]:


        return finalize_teacher_baselines(statistics)

    def _forward_with_prediction(
        self,
        batch: PaddedOnlineGeometryBatch,
        *,
        step: int,
        mode: str,
        apply_augmentation: bool = True,
    ) -> tuple[MethodStep, Any]:


        model = self.require_model()
        if mode == "train" and apply_augmentation:
            batch = maybe_rewrite_batch(
                batch,
                config=self.config.joint_sign_rewrite,
                step=step,
                seed=step,
            )
        views = method_batch_views(batch)
        q, evidence, evidence_row_index, joint_coordinate_sign = views.model_input
        query_points, bandwidths, owner_index, query_index, joint_index = views.readout_condition
        q = q.detach()
        forward = self._compiled_forward if self._compiled_forward is not None else model
        autocast_name = str(getattr(self.execution_policy, "model_autocast_dtype", "float32"))
        if self._compiled_forward is not None and q.device.type == "cuda":

            torch.compiler.cudagraph_mark_step_begin()
        if q.device.type == "cuda" and autocast_name == "bfloat16":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                raw_prediction = forward(
                    q,
                    evidence,
                    query_points,
                    bandwidths,
                    owner_index=owner_index,
                    query_index=query_index,
                    joint_index=joint_index,
                    evidence_row_index=evidence_row_index,
                    joint_coordinate_sign=joint_coordinate_sign,
                )
        else:
            raw_prediction = forward(
                q,
                evidence,
                query_points,
                bandwidths,
                owner_index=owner_index,
                query_index=query_index,
                joint_index=joint_index,
                evidence_row_index=evidence_row_index,
                joint_coordinate_sign=joint_coordinate_sign,
            )
        prediction = GeometrySSLForward(
            latents=raw_prediction.latents,
            query_features=raw_prediction.query_features,
            density=raw_prediction.density.to(torch.float32),
            kappa=raw_prediction.kappa.to(torch.float32),
        )
        context = MultiAnchorObjectiveContext(
            prediction=prediction,
            batch=batch,
        )
        results = evaluate_objectives(context, self.config.objectives)
        return MethodStep(objectives=results, sample_count=int(batch.q.shape[0])), prediction

    def _forward_latent_diagnostic(
        self,
        batch: PaddedOnlineGeometryBatch,
        prediction: GeometrySSLForward,
    ) -> tuple[MethodStep, GeometrySSLForward]:


        model = self.require_model()
        views = method_batch_views(batch)
        _q, evidence, evidence_row_index, _joint_coordinate_sign = views.model_input
        _query_points, bandwidths, owner_index, query_index, joint_index = views.readout_condition


        diagnostic_entities = prediction.latents.entities.detach().requires_grad_(True)
        diagnostic_latents = replace(prediction.latents, entities=diagnostic_entities)  # typed unified latent
        diagnostic_query_features = prediction.query_features.detach()


        entity_valid = evidence.entity_valid_mask
        if entity_valid is not None:
            if evidence_row_index is not None and entity_valid.ndim == 2:
                entity_valid = entity_valid[evidence_row_index]
            if entity_valid.ndim == 1:
                entity_valid = entity_valid.unsqueeze(0).expand(batch.q.shape[0], -1)

        joint_entity_index = (
            evidence.joint_entity_index[evidence_row_index]
            if evidence_row_index is not None and evidence.joint_entity_index.ndim == 2
            else evidence.joint_entity_index
        )


        autocast_name = str(getattr(self.execution_policy, "model_autocast_dtype", "float32"))
        if diagnostic_entities.device.type == "cuda" and autocast_name == "bfloat16":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                raw_diagnostic = model.decode_latents(
                    diagnostic_latents,
                    diagnostic_query_features,
                    bandwidths=bandwidths,
                    entity_valid_mask=entity_valid,
                    joint_entity_index=joint_entity_index,
                    owner_index=owner_index,
                    query_index=query_index,
                    joint_index=joint_index,
                )
        else:
            raw_diagnostic = model.decode_latents(
                diagnostic_latents,
                diagnostic_query_features,
                bandwidths=bandwidths,
                entity_valid_mask=entity_valid,
                joint_entity_index=joint_entity_index,
                owner_index=owner_index,
                query_index=query_index,
                joint_index=joint_index,
            )


        diagnostic_prediction = GeometrySSLForward(
            latents=raw_diagnostic.latents,
            query_features=raw_diagnostic.query_features,
            density=raw_diagnostic.density.to(torch.float32),
            kappa=raw_diagnostic.kappa.to(torch.float32),
        )
        context = MultiAnchorObjectiveContext(prediction=diagnostic_prediction, batch=batch)
        results = evaluate_objectives(context, self.config.objectives)
        return MethodStep(objectives=results, sample_count=int(batch.q.shape[0])), diagnostic_prediction

    def dense_snapshot(
        self,
        batch: PaddedOnlineGeometryBatch,
        *,
        microbatch_size: int,
    ) -> tuple[GeometrySSLForward, PaddedOnlineGeometryBatch]:


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


        return reduce_method_steps(steps, self.config.objectives)

    def backward_update(
        self,
        batch: PaddedOnlineGeometryBatch,
        *,
        forward_step: int,
        microbatch_size: int,
        collect_z_gradients: bool = False,
    ) -> MethodUpdate:


        return backward_method_update(
            self,
            batch,
            forward_step=forward_step,
            microbatch_size=microbatch_size,
            collect_z_gradients=collect_z_gradients,
            rewrite_batch_fn=maybe_rewrite_batch,
        )

    def backward_update_units(
        self,
        units: Iterator[PaddedOnlineGeometryBatch],
        *,
        forward_step: int,
        logical_sample_count: int,
        microbatch_size: int,
        collect_z_gradients: bool = False,
    ) -> MethodUpdate:


        return backward_method_update_units(
            self,
            units,
            forward_step=forward_step,
            logical_sample_count=logical_sample_count,
            microbatch_size=microbatch_size,
            collect_z_gradients=collect_z_gradients,
            rewrite_batch_fn=maybe_rewrite_batch,
        )

    def stage_replay_unit(self, unit: PaddedOnlineGeometryBatch) -> PaddedOnlineGeometryBatch:


        return stage_padded_batch_for_replay(unit)

    def restore_replay_unit(
        self,
        unit: PaddedOnlineGeometryBatch,
        *,
        device: torch.device,
    ) -> PaddedOnlineGeometryBatch:


        return restore_padded_batch_from_replay(unit, device=device)

    def evaluate_session(
        self,
        session: MultiAnchorGaussianSession,
        schedule: Any,
        *,
        include_ablations: bool = False,
    ) -> MethodEvaluationReport:


        return evaluate_method_session(
            self,
            session,
            schedule,
            include_ablations=include_ablations,
        )

    def fit_z_compression_basis(self, session: MultiAnchorGaussianSession, schedule: Any):


        return fit_z_compression_basis(self, session, schedule)

    def evaluate_z_compression_session(
        self,
        session: MultiAnchorGaussianSession,
        schedule: Any,
        *,
        basis: Any,
        ranks: tuple[int, ...],
    ) -> dict[str, object]:


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


        return analyze_ablations(
            evidence,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        )

    def feature_spec(self) -> FeatureSpec:


        return FeatureSpec(
            entity_width=self.config.model.encoder.backbone.hidden_width,
        )

    def retained_state_dict(self) -> dict[str, torch.Tensor]:


        return self.require_model().retained_state_dict()

    def training_state_dict(self) -> dict[str, torch.Tensor]:


        return self.require_model().state_dict()

    def load_training_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:


        self.require_model().load_state_dict(dict(state), strict=True)

    def retained_artifact_payload(
        self,
        *,
        metadata: Mapping[str, Any],
        source_checkpoint: Path,
    ) -> dict[str, Any]:


        return build_retained_geometry_artifact(
            self,
            metadata=metadata,
            source_checkpoint=source_checkpoint,
        )

    def close(self) -> None:


        providers = [
            *(tuple([self.train_sources]) if self.train_sources is not None else ()),
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
