r"""多锚点 Gaussian 隐式场的科学聚合根。

本类对内显式耦合 representation、model 与 objectives，对外只给 SSL trainer 封闭接口：
prepare、realize_minibatch、forward_objectives、reduce_update、evaluate、retained export。
Trainer 不得读取 `representation.config.field` 或 padding layout。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from anymani.assets.bank.hand_container import HandContainer
from anymani.distill.methods.contracts import FeatureSpec, MethodStep, MethodUpdate
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
        self.padding: GeometryPaddingCfg | None = None
        self.train_samplers: tuple[SobolJointSampler, ...] = ()
        self.validation_samplers: dict[str, tuple[SobolJointSampler, ...]] = {}

    def prepare(self, catalog: Any, *, device: torch.device, dtype: torch.dtype) -> None:
        r"""物化 train/validation sources，并按 dataset 实际结构推导 padding。"""

        del device, dtype
        self.train_sources = tuple(self.representation.materialize_source(asset) for asset in catalog.train)
        self.validation_sources = {
            suite_name: tuple(self.validation_representation.materialize_source(asset) for asset in suite_assets)
            for suite_name, suite_assets in catalog.validation.items()
        }
        all_sources = list(self.train_sources)
        for suite_sources in self.validation_sources.values():
            all_sources.extend(suite_sources)
        self.padding = _derive_padding(
            tuple(all_sources),
            max_graph_distance=self.config.model.encoder.backbone.max_graph_distance,
        )

    def initialize_samplers(self, *, train_seed: int, validation_seeds: dict[str, int]) -> None:
        r"""为每资产建立独立 scrambled Sobol cursor。"""

        self.train_samplers = self.make_independent_samplers(self.train_sources, seed=train_seed)
        self.validation_samplers = {
            suite_name: self.make_independent_samplers(suite_sources, seed=validation_seeds[suite_name])
            for suite_name, suite_sources in self.validation_sources.items()
        }

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
        q_block_index = schedule_item.epoch * schedule.q_rounds_per_epoch + schedule_item.q_round
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
                ((int(schedule_item.epoch) * 1_000_003 + int(schedule_item.window_index)) * 10_007
                 + int(schedule_item.q_round))
                * 97
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
        return MethodStep(objectives=results, sample_count=int(batch.q.shape[0]))

    def reduce_update(self, steps: tuple[MethodStep, ...]) -> MethodUpdate:
        r"""按 $(asset,q)$ 等权合并一个 optimizer update。"""

        return reduce_method_steps(steps, self.config.objectives)

    def evaluate(self, batches: tuple[PaddedOnlineGeometryBatch, ...]) -> dict[str, float]:
        r"""在固定 validation bank 上聚合五项 term；JVP 需要 autograd 但不更新参数。"""

        self.require_model().eval()
        steps: list[MethodStep] = []
        with torch.enable_grad():
            for index, batch in enumerate(batches):
                steps.append(self.forward_objectives(batch, step=index, mode="eval"))
        update = self.reduce_update(tuple(steps))
        return dict(update.terms)

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

    def close(self) -> None:
        r"""当前 method 不额外持有 GPU lease；resident window 由 trainer teardown。"""


MultiAnchorGaussianMethodCfg.runtime_type = MultiAnchorGaussianMethod  # type: ignore[misc, assignment]


__all__ = ["MultiAnchorGaussianMethod", "_derive_padding"]
