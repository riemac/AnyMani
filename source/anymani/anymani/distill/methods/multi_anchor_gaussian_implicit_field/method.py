r"""多锚点 Gaussian 隐式场的科学聚合根。

本类对内显式耦合 representation、model 与 objectives，对外只给 SSL trainer 封闭接口：
prepare、realize_minibatch、forward_objectives、reduce_update、evaluate、retained export。
Trainer 不得读取 `representation.config.field` 或 padding layout。
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from anymani.assets.bank.hand_container import HandContainer
from anymani.distill.methods.contracts import FeatureSpec, MethodEvaluationReport, MethodStep, MethodUpdate
from anymani.distill.models.geometry_ssl import GeometrySSLModel
from anymani.distill.models.input_adapters.geometry import GeometryPaddingCfg
from anymani.distill.representations.geometry import GeometryRepresentation
from anymani.distill.representations.sources.geometry_source import GeometrySource
from anymani.distill.representations.targets.geometry_field import fixed_validation_gaussian_field_config

from .augmentation import maybe_rewrite_batch
from .batch import PaddedOnlineGeometryBatch, attach_static_evidence, method_batch_views, pad_online_geometry_samples
from .config import MultiAnchorGaussianMethodCfg
from .context import MultiAnchorObjectiveContext
from .objectives import evaluate_objectives, reduce_method_steps
from .state_measure import SobolJointSampler


def _derive_padding(sources: tuple[GeometrySource, ...], *, max_graph_distance: int) -> GeometryPaddingCfg:
    r"""由 resolved 资产实际最大结构推导稠密容器上限；超出则失败。"""

    if not sources:
        raise ValueError("padding derivation requires at least one materialized source")
    max_joint = max(source.spec_cpu.space_screws.shape[0] for source in sources)
    max_tip = max(sum(role == "tip" for role in source.spec_cpu.owner_roles) for source in sources)
    if max_joint < 1 or max_tip < 1:
        raise ValueError("resolved dataset must contain at least one JOINT and one TIP owner")
    return GeometryPaddingCfg(
        max_joint_count=max_joint,
        max_tip_count=max_tip,
        max_graph_distance=max_graph_distance,
    )


class MultiAnchorGaussianSession:
    r"""封装一个 train/validation/evaluation split 的 source、Sobol cursor 与 resident window。"""

    def __init__(
        self,
        method: MultiAnchorGaussianMethod,
        *,
        role: str,
        suite: str,
        sources: tuple[GeometrySource, ...],
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
            "asset_ids": tuple(source.asset_id for source in self.sources),
            "samplers": tuple(sampler.state_dict() for sampler in self.samplers),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        r"""严格恢复同一资产轴上的 Sobol cursor。"""

        expected_ids = tuple(source.asset_id for source in self.sources)
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
        r"""释放 resident window 中的 Warp lease 和 device state。"""

        self.window.release_all()
        self.window.drain_telemetry_events()


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
        self.train_sources: tuple[GeometrySource, ...] = ()
        self.validation_sources: dict[str, tuple[GeometrySource, ...]] = {}
        self.evaluation_sources: dict[str, tuple[GeometrySource, ...]] = {}
        self.padding: GeometryPaddingCfg | None = None

    def prepare(self, catalog: Any, *, device: torch.device, dtype: torch.dtype) -> None:
        r"""物化 train/validation/evaluation sources，并按 dataset 实际结构推导 padding。"""

        del device, dtype
        self.train_sources = tuple(self.representation.materialize_source(asset) for asset in catalog.train)
        self.validation_sources = {
            suite_name: tuple(self.validation_representation.materialize_source(asset) for asset in suite_assets)
            for suite_name, suite_assets in catalog.validation.items()
        }
        self.evaluation_sources = {
            suite_name: tuple(self.validation_representation.materialize_source(asset) for asset in suite_assets)
            for suite_name, suite_assets in catalog.evaluation.items()
        }
        all_sources = list(self.train_sources)
        for suite_sources in self.validation_sources.values():
            all_sources.extend(suite_sources)
        for suite_sources in self.evaluation_sources.values():
            all_sources.extend(suite_sources)
        self.padding = _derive_padding(
            tuple(all_sources),
            max_graph_distance=self.config.model.encoder.backbone.max_graph_distance,
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

    def split_asset_count(self, role: str, *, suite: str = "") -> int:
        r"""返回 train 或具名 held-out suite 的真实资产数。"""

        if role == "train":
            return len(self.train_sources)
        if role == "training_evaluation":
            return len(self.train_sources)
        if role == "validation":
            return len(self.validation_sources.get(suite, ()))
        if role == "evaluation":
            return len(self.evaluation_sources.get(suite, ()))
        raise ValueError(f"unknown method split role={role!r}")

    def asset_manifest(self, catalog: Any) -> dict[str, Any]:
        r"""记录本 Method 实际使用的 physical source 与 train/held-out 隔离证据。"""

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

        train_by_id = {source.asset_id: source for source in self.train_sources}
        validation_by_suite = {
            suite: {source.asset_id: source for source in sources}
            for suite, sources in self.validation_sources.items()
        }
        evaluation_by_suite = {
            suite: {source.asset_id: source for source in sources}
            for suite, sources in self.evaluation_sources.items()
        }
        manifest = {
            "schema_version": "4.0.0",
            "dataset_source_path": str(catalog.dataset.source_path),
            "dataset_source_sha256": catalog.dataset.source_sha256,
            "train": [
                record(
                    item.container,
                    train_by_id[item.container.asset_id],
                    partition="train",
                    provenance=item.provenance,
                )
                for item in catalog.dataset.train.records
            ],
            "validation": {
                suite: [
                    record(
                        item.container,
                        validation_by_suite[suite][item.container.asset_id],
                        partition=f"validation.{suite}",
                        provenance=item.provenance,
                    )
                    for item in partition.records
                ]
                for suite, partition in catalog.dataset.validation.items()
            },
            "evaluation": {
                suite: [
                    record(
                        item.container,
                        evaluation_by_suite[suite][item.container.asset_id],
                        partition=f"evaluation.{suite}",
                        provenance=item.provenance,
                    )
                    for item in partition.records
                ]
                for suite, partition in catalog.dataset.evaluation.items()
            },
        }
        validate_asset_manifest_isolation(manifest)
        return manifest

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
            sources = self.train_sources
        elif role == "validation":
            sources = self.validation_sources.get(suite, ())
        elif role == "evaluation":
            sources = self.evaluation_sources.get(suite, ())
        else:
            raise ValueError(f"unknown method split role={role!r}")
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
        sources: tuple[GeometrySource, ...],
        *,
        seed: int,
    ) -> tuple[SobolJointSampler, ...]:
        r"""为给定 CPU sources 建立不复用训练 cursor 的独立 Sobol 引擎。"""

        return tuple(SobolJointSampler(source.spec_cpu, seed=seed + index) for index, source in enumerate(sources))

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

    def materialize_sources(self, assets: tuple[HandContainer, ...]) -> tuple[GeometrySource, ...]:
        r"""按 representation source 配置物化 CPU physical sources。"""

        return tuple(self.representation.materialize_source(asset) for asset in assets)

    def realize_minibatch(
        self,
        schedule_item: Any,
        *,
        sources: tuple[GeometrySource, ...],
        samplers: tuple[SobolJointSampler, ...],
        window: Any,
        seed: int,
        schedule: Any,
        mode: str = "train",
    ) -> PaddedOnlineGeometryBatch:
        r"""由 schedule item realization 一次同资产 q block，并在 window 内复用 device state。"""

        del schedule  # q-block、window 与随机身份均由 ScheduledMinibatch 显式携带
        representation = self.representation if mode == "train" else self.validation_representation
        padding = self.require_padding()
        catalog_ids = tuple(source.asset_id for source in sources)
        resident_indices = tuple(schedule_item.resident_asset_indices)
        if not resident_indices:
            raise ValueError("schedule item must declare the complete resident window, not only the minibatch")
        window_ids = tuple(catalog_ids[index] for index in resident_indices)
        states = window.ensure(window_ids)
        states_by_id = {state.source.asset_id: state for state in states}
        samples = []
        q_block_index = int(schedule_item.q_block_index)  # 同一资产获得第几个新 q-block
        for asset_index in schedule_item.asset_indices:
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
        r"""完成一次基础模型前向，并让五项 terms 共享惰性计算上下文。"""

        result, _prediction = self._forward_with_prediction(batch, step=step, mode=mode)
        return result

    def _forward_with_prediction(
        self,
        batch: PaddedOnlineGeometryBatch,
        *,
        step: int,
        mode: str,
    ) -> tuple[MethodStep, Any]:
        r"""共享 objective 与固定评估所需的同一次模型预测，避免 validation 双前向。"""

        model = self.require_model()
        if mode == "train":
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
        context = MultiAnchorObjectiveContext(model=model, q=q, prediction=prediction, batch=batch)
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
        r"""当前 method 不额外持有 GPU lease；resident window 由 trainer teardown。"""


MultiAnchorGaussianMethodCfg.runtime_type = MultiAnchorGaussianMethod  # type: ignore[misc, assignment]


__all__ = ["MultiAnchorGaussianMethod", "MultiAnchorGaussianSession", "_derive_padding"]
