r"""Gaussian density + anchor-relational Material-point Jacobian 的 concrete SSL method。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch._functorch import config as functorch_config  # pyright: ignore[reportPrivateImportUsage]

from anymani.distill.methods.contracts import (
    FeatureSpec,
    MethodEvaluationReport,
    MethodParameterGroup,
    MethodStep,
    MethodUpdate,
)
from anymani.distill.methods.multi_anchor_gaussian_implicit_field.method import MultiAnchorGaussianMethod
from anymani.distill.methods.multi_anchor_gaussian_implicit_field.training import combine_fairgrad
from anymani.distill.models.density_material_jacobian_ssl import (
    DensityMaterialJacobianForward,
    DensityMaterialJacobianSSLModel,
)

from .artifact import build_retained_artifact
from .augmentation import maybe_rewrite_density_gamma_batch, permute_density_gamma_sample, sample_entity_permutation
from .batch import (
    PaddedDensityGammaBatch,
    pad_density_gamma_blocks,
    restore_padded_batch_from_replay,
    sample_density_gamma_block,
    stage_padded_batch_for_replay,
)
from .config import DensityMaterialJacobianMethodCfg
from .objectives import (
    DensityGammaObjectiveContext,
    evaluate_objectives,
    finalize_teacher_baselines,
    merge_teacher_baseline_statistics,
    reduce_method_steps,
    teacher_baseline_statistics,
)

_DEVICE_SUBWINDOW_ASSETS = 8


class DensityMaterialJacobianMethod(MultiAnchorGaussianMethod):
    r"""复用 source/cache/session 生命周期并独立拥有 batch/model/objective/evaluation 的联合方法。

    继承边界只覆盖已验证的 source artifact、lazy provider、resident window 与 physical audit 基础设施；
    本类覆盖所有会生成旧 κ target、旧 batch、旧 readers 或旧梯度的路径。
    """

    def __init__(self, config: DensityMaterialJacobianMethodCfg) -> None:
        r"""构造无 IO source runtime，并清空 learned model。"""

        super().__init__(config)  # type: ignore[arg-type]  # 两种 config 共享 source/model/augmentation 字段合同
        self.config = config
        self.model: DensityMaterialJacobianSSLModel | None = None
        self._compiled_forward: Any | None = None

    @property
    def dense_snapshot(self) -> None:
        r"""新 batch/schema 的 dense artifact 实现前，显式关闭旧 κ snapshot hook。"""

        return None

    def initialize_model(self, *, device: torch.device, dtype: torch.dtype) -> DensityMaterialJacobianSSLModel:
        r"""一次性构造 unified encoder、density reader 与 Gamma reader。"""

        if self.model is not None:
            raise RuntimeError("density/Gamma model is already initialized")
        self.model = DensityMaterialJacobianSSLModel(self.config.model).to(device=device, dtype=dtype)
        if self.execution_policy is not None and bool(self.execution_policy.compile_enabled):
            functorch_config.donated_buffer = False
            self._compiled_forward = torch.compile(
                self.model,
                mode=str(self.execution_policy.compile_mode),
                fullgraph=True,
            )
        return self.model

    def require_model(self) -> DensityMaterialJacobianSSLModel:
        r"""返回已初始化联合模型。"""

        if self.model is None:
            raise RuntimeError("density/Gamma model has not been initialized")
        return self.model

    def optimizer_parameter_groups(self) -> tuple[MethodParameterGroup, ...]:
        r"""返回 shared encoder 与两个互斥、完整覆盖的 private readers。"""

        model = self.require_model()
        groups = (
            MethodParameterGroup("shared_encoder", tuple(model.encoder.parameters())),
            MethodParameterGroup("density_reader", tuple(model.density_decoder.parameters())),
            MethodParameterGroup("material_jacobian_reader", tuple(model.material_jacobian_decoder.parameters())),
        )
        grouped = tuple(parameter for group in groups for parameter in group.parameters if parameter.requires_grad)
        trainable = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
        if len({id(parameter) for parameter in grouped}) != len(grouped):
            raise RuntimeError("density/Gamma optimizer parameter groups overlap")
        if {id(parameter) for parameter in grouped} != {id(parameter) for parameter in trainable}:
            raise RuntimeError("density/Gamma parameter groups do not cover the trainable model")
        return groups

    def feature_spec(self) -> FeatureSpec:
        r"""声明 unified entity/JOINT view 与两个 observable 的 coordinate contract。"""

        return FeatureSpec(
            entity_width=self.config.model.encoder.backbone.hidden_width,
            coordinate_rewrite_contract="density invariant; material_jacobian selected-column sign-equivariant",
        )

    def declared_objective_weights(self) -> dict[str, float]:
        r"""两项目标均存在；shared priority 由 FairGrad 决定。"""

        return {name: 1.0 for name in self.config.objectives.enabled()}

    def formula_identity(self) -> dict[str, str]:
        r"""记录 density 与 Gamma objective 的限定名。"""

        return {name: term.qualified_func_name() for name, term in self.config.objectives.enabled().items()}

    def optimization_identity(self) -> dict[str, object]:
        r"""记录两任务 FairGrad 配置。"""

        return {
            "algorithm": self.config.fairgrad.algorithm,
            "tasks": ["density", "material_jacobian"],
            "near_opposition_tolerance": self.config.fairgrad.near_opposition_tolerance,
        }

    def _realize_minibatch_blocks(
        self,
        schedule_item: Any,
        *,
        sources: Any,
        samplers: Any,
        window: Any,
        seed: int,
        schedule: Any,
        mode: str,
    ):
        r"""按 8-asset device subwindow 流式生成 density/Gamma blocks，不调用旧 κ representation.sample。"""

        del schedule
        catalog_ids = sources.asset_ids
        logical_indices = tuple(schedule_item.asset_indices)
        chunks = tuple(
            logical_indices[start : start + _DEVICE_SUBWINDOW_ASSETS]
            for start in range(0, len(logical_indices), _DEVICE_SUBWINDOW_ASSETS)
        )
        q_block_index = int(schedule_item.q_block_index)
        prefetch = sources.prefetch_async(tuple(catalog_ids[index] for index in chunks[0]))
        for chunk_index, asset_chunk in enumerate(chunks):
            cores = sources.await_prefetch(prefetch)
            next_handle = None
            if chunk_index + 1 < len(chunks):
                next_ids = tuple(catalog_ids[index] for index in chunks[chunk_index + 1])
                next_handle = sources.prefetch_async(next_ids)
            bank_index = 0 if mode != "train" else q_block_index % self.config.representation.source.anchors.bank_size
            states = window.ensure(
                tuple(catalog_ids[index] for index in asset_chunk),
                prefetch_sources=False,
                prepared_sources={core.asset_id: core for core in cores},
                bank_index=bank_index,
            )
            state_by_id = {state.source.asset_id: state for state in states}
            blocks = []
            for asset_index in asset_chunk:
                asset_id = catalog_ids[asset_index]
                state = state_by_id[asset_id]
                q_count = int(schedule_item.q_per_asset)
                q = samplers[asset_index].draw(
                    q_count,
                    device=state.spec.space_screws.device,
                    dtype=state.spec.space_screws.dtype,
                )
                q_start = samplers[asset_index].cursor - q_count
                schedule_seed = (
                    int(schedule_item.minibatch_index) * 1_000_003
                    + int(schedule_item.window_index) * 10_007
                    + int(schedule_item.asset_group)
                )
                block = sample_density_gamma_block(
                        state,
                        q,
                        self.config,
                        sampling_seed=seed + schedule_seed,
                        q_index=torch.arange(q_start, q_start + q_count, device="cpu", dtype=torch.long),
                        anchor_index=bank_index,
                        supervision_split="train" if mode == "train" else "eval",
                    )
                if mode == "train" and self.config.entity_permutation.enabled:
                    permutation = sample_entity_permutation(
                        len(state.source.container.geometry_semantics.owners),
                        asset_id=asset_id,
                        q_block_start=q_start,
                        root_seed=seed + schedule_seed,
                        config=self.config.entity_permutation,
                    )
                    block = permute_density_gamma_sample(block, permutation)
                blocks.append(block)
            yield blocks
            if next_handle is not None:
                prefetch = next_handle

    def realize_minibatch(self, schedule_item: Any, **kwargs: Any) -> PaddedDensityGammaBatch:
        r"""收集当前 schedule item 的全部 blocks，供 fixed evaluation 使用。"""

        blocks = [block for unit in self._realize_minibatch_blocks(schedule_item, **kwargs) for block in unit]
        return pad_density_gamma_blocks(blocks, padding=self.require_padding())

    def realize_minibatch_units(self, schedule_item: Any, **kwargs: Any):
        r"""逐个交付当前 logical update 的 8-asset density/Gamma units。"""

        for blocks in self._realize_minibatch_blocks(schedule_item, **kwargs):
            yield pad_density_gamma_blocks(blocks, padding=self.require_padding())

    def _forward_with_prediction(
        self,
        batch: PaddedDensityGammaBatch,
        *,
        mode: str,
    ) -> tuple[MethodStep, DensityMaterialJacobianForward]:
        r"""共享训练/evaluation 的单次联合模型前向与 objective 构造。"""

        if mode == "train":
            batch = maybe_rewrite_density_gamma_batch(
                batch,
                config=self.config.joint_sign_rewrite,
                step=int(batch.q_index[0]) if batch.q_index.numel() else 0,
                seed=int(batch.anchor_index[0]) if batch.anchor_index.numel() else 0,
            )
        model = self.require_model()
        forward = self._compiled_forward if self._compiled_forward is not None else model
        autocast_name = str(getattr(self.execution_policy, "model_autocast_dtype", "float32"))
        if self._compiled_forward is not None and batch.q.device.type == "cuda":
            torch.compiler.cudagraph_mark_step_begin()
        kwargs = {
            "evidence_row_index": batch.evidence_row_index,
            "joint_coordinate_sign": batch.joint_coordinate_sign,
        }
        if batch.q.device.type == "cuda" and autocast_name == "bfloat16":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                raw = forward(
                    batch.q.detach(),
                    batch.evidence,
                    batch.queries.query_points_h,
                    batch.field_targets.bandwidths,
                    batch.material_targets.owner_index,
                    batch.material_targets.joint_index,
                    batch.material_point_index,
                    **kwargs,
                )
        else:
            raw = forward(
                batch.q.detach(),
                batch.evidence,
                batch.queries.query_points_h,
                batch.field_targets.bandwidths,
                batch.material_targets.owner_index,
                batch.material_targets.joint_index,
                batch.material_point_index,
                **kwargs,
            )
        prediction = DensityMaterialJacobianForward(
            latents=raw.latents,
            query_features=raw.query_features,
            material_pair_features=raw.material_pair_features,
            density=raw.density.float(),
            material_jacobian=raw.material_jacobian.float(),
        )
        results = evaluate_objectives(DensityGammaObjectiveContext(prediction, batch), self.config.objectives)
        return MethodStep(results, int(batch.q.shape[0])), prediction

    def forward_objectives(
        self,
        batch: PaddedDensityGammaBatch,
        *,
        step: int,
        mode: str = "train",
        microbatch_size: int | None = None,
    ) -> MethodStep:
        r"""普通合同前向；streaming 正式训练使用 `backward_update_units`。"""

        del step, microbatch_size
        return self._forward_with_prediction(batch, mode=mode)[0]

    @staticmethod
    def _accumulate_gradient(
        accumulator: list[torch.Tensor | None],
        gradients: tuple[torch.Tensor | None, ...],
    ) -> None:
        r"""跨 stream units 累计同一参数布局的 detached gradients。"""

        for index, gradient in enumerate(gradients):
            if gradient is None:
                continue
            current = accumulator[index]
            accumulator[index] = gradient.detach().clone() if current is None else current + gradient.detach()

    def backward_update_units(
        self,
        units: Iterable[PaddedDensityGammaBatch],
        *,
        forward_step: int,
        logical_sample_count: int,
        microbatch_size: int,
        collect_z_gradients: bool = False,
    ) -> MethodUpdate:
        r"""逐 unit 形成两项 shared task gradients，最后以 FairGrad 写回一次 optimizer update。"""

        del forward_step, microbatch_size, collect_z_gradients
        model = self.require_model()
        shared = tuple(model.encoder.parameters())
        density_private = tuple(model.density_decoder.parameters())
        gamma_private = tuple(model.material_jacobian_decoder.parameters())
        density_shared: list[torch.Tensor | None] = [None] * len(shared)
        gamma_shared: list[torch.Tensor | None] = [None] * len(shared)
        density_reader: list[torch.Tensor | None] = [None] * len(density_private)
        gamma_reader: list[torch.Tensor | None] = [None] * len(gamma_private)
        numerator = {"density": 0.0, "material_jacobian": 0.0}
        observed = 0
        for unit in units:
            step, _prediction = self._forward_with_prediction(unit, mode="train")
            observed += int(unit.q.shape[0])
            density_component = step.objectives["density"].components[0]
            gamma_component = step.objectives["material_jacobian"].components[0]
            density_term = density_component.numerator / float(logical_sample_count)
            gamma_term = gamma_component.numerator / float(logical_sample_count)
            d_grad = torch.autograd.grad(
                density_term,
                (*shared, *density_private),
                retain_graph=True,
                allow_unused=True,
            )
            g_grad = torch.autograd.grad(
                gamma_term,
                (*shared, *gamma_private),
                allow_unused=True,
            )
            self._accumulate_gradient(density_shared, d_grad[: len(shared)])
            self._accumulate_gradient(density_reader, d_grad[len(shared) :])
            self._accumulate_gradient(gamma_shared, g_grad[: len(shared)])
            self._accumulate_gradient(gamma_reader, g_grad[len(shared) :])
            numerator["density"] += float(density_component.numerator.detach())
            numerator["material_jacobian"] += float(gamma_component.numerator.detach())
        if observed != logical_sample_count:
            raise RuntimeError(f"stream units contain {observed} samples; expected {logical_sample_count}")

        fairgrad = combine_fairgrad(
            density_shared,
            gamma_shared,
            near_opposition_tolerance=self.config.fairgrad.near_opposition_tolerance,
        )
        for parameter, gradient in zip(shared, fairgrad.combined, strict=True):
            parameter.grad = gradient
        for parameter, gradient in zip(density_private, density_reader, strict=True):
            parameter.grad = gradient
        for parameter, gradient in zip(gamma_private, gamma_reader, strict=True):
            parameter.grad = gradient
        evidence = asdict(fairgrad.evidence)
        evidence = {key.replace("kappa", "material_jacobian"): float(value) for key, value in evidence.items()}
        return MethodUpdate(
            terms={name: value / logical_sample_count for name, value in numerator.items()},
            sample_count=observed,
            denominators={name: float(logical_sample_count) for name in numerator},
            gradient_evidence={f"fairgrad/{name}": value for name, value in evidence.items()},
        )

    def backward_update(
        self,
        batch: PaddedDensityGammaBatch,
        *,
        forward_step: int,
        microbatch_size: int,
        collect_z_gradients: bool = False,
    ) -> MethodUpdate:
        r"""把一个完整 opaque batch 作为单 unit 送入同一 FairGrad 实现，供 admission parity 使用。"""

        return self.backward_update_units(
            (batch,),
            forward_step=forward_step,
            logical_sample_count=int(batch.q.shape[0]),
            microbatch_size=microbatch_size,
            collect_z_gradients=collect_z_gradients,
        )

    def reduce_update(self, steps: tuple[MethodStep, ...]) -> MethodUpdate:
        r"""普通 forward steps 的 additive reduction。"""

        return reduce_method_steps(steps, self.config.objectives)

    def teacher_baseline_statistics(self, batch: PaddedDensityGammaBatch) -> dict[str, torch.Tensor]:
        r"""返回 constant-density 与 zero-Gamma baseline 充分统计。"""

        return teacher_baseline_statistics(batch, self.config.objectives)

    def merge_teacher_baseline_statistics(
        self,
        total: dict[str, torch.Tensor] | None,
        block: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        r"""合并 teacher baseline blocks。"""

        return merge_teacher_baseline_statistics(total, block)

    def finalize_teacher_baselines(self, statistics: dict[str, torch.Tensor]) -> dict[str, object]:
        r"""形成 run-local density/Gamma baselines。"""

        return finalize_teacher_baselines(statistics)

    def stage_replay_unit(self, unit: PaddedDensityGammaBatch) -> PaddedDensityGammaBatch:
        r"""把 opaque unit 暂存到 pinned CPU。"""

        return stage_padded_batch_for_replay(unit)

    def restore_replay_unit(self, unit: PaddedDensityGammaBatch, *, device: torch.device) -> PaddedDensityGammaBatch:
        r"""恢复当前 replay unit 到训练 device。"""

        return restore_padded_batch_from_replay(unit, device=device)

    def training_state_dict(self) -> dict[str, torch.Tensor]:
        r"""返回完整 encoder + readers learned state。"""

        return {name: value.detach().clone() for name, value in self.require_model().state_dict().items()}

    def load_training_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        r"""严格恢复完整 learned state。"""

        self.require_model().load_state_dict(dict(state), strict=True)

    def retained_state_dict(self) -> dict[str, torch.Tensor]:
        r"""返回 encoder-only FP32 master state。"""

        return self.require_model().retained_state_dict()

    def retained_artifact_payload(
        self,
        *,
        metadata: Mapping[str, Any],
        source_checkpoint: Path,
    ) -> dict[str, Any]:
        r"""构造 density/Gamma method-owned schema-5 artifact。"""

        return build_retained_artifact(self, metadata=metadata, source_checkpoint=source_checkpoint)

    def evaluate_session(
        self,
        session: Any,
        schedule: Any,
        *,
        include_ablations: bool = False,
    ) -> MethodEvaluationReport:
        r"""流式计算 canonical fixed-bank density/Gamma objectives 与 zero-baseline skill。"""

        numerator = {"density": 0.0, "material_jacobian": 0.0}
        denominator = {"density": 0.0, "material_jacobian": 0.0}
        baseline_total: dict[str, torch.Tensor] | None = None
        channel_error = torch.zeros(4, dtype=torch.float64)
        channel_baseline = torch.zeros(4, dtype=torch.float64)
        channel_count = torch.zeros(4, dtype=torch.float64)
        channel_sign_correct = torch.zeros(4, dtype=torch.float64)
        channel_sign_count = torch.zeros(4, dtype=torch.float64)
        zero_prediction_square = torch.zeros((), dtype=torch.float64)
        zero_prediction_count = torch.zeros((), dtype=torch.float64)
        ablation_names = ("full", "query_only", "same_asset_q_shuffle", "cross_asset_shuffle", "joint_token_shuffle")
        ablation_totals = {
            ablation: {
                term: [0.0, 0.0]
                for term in ("density", "material_jacobian")
            }
            for ablation in ablation_names
        }
        step_index = 0
        self.eval_mode()
        with torch.no_grad():
            while not schedule.complete:
                batch = session.realize(schedule.next(), schedule=schedule, step=step_index)
                step, prediction = self._forward_with_prediction(batch, mode="eval")
                for name, result in step.objectives.items():
                    component = result.components[0]
                    numerator[name] += float(component.numerator)
                    denominator[name] += float(component.denominator)
                    ablation_totals["full"][name][0] += float(component.numerator)
                    ablation_totals["full"][name][1] += float(component.denominator)
                baseline_total = self.merge_teacher_baseline_statistics(
                    baseline_total,
                    {name: value.cpu().double() for name, value in self.teacher_baseline_statistics(batch).items()},
                )
                target = batch.material_targets.relation_sensitivity_per_rad
                anchor_valid = batch.evidence.anchor_valid_mask
                if anchor_valid is None:
                    anchor_valid = torch.ones(
                        batch.evidence.anchors.shape[:-1],
                        device=target.device,
                        dtype=torch.bool,
                    )
                anchor_valid = anchor_valid[batch.evidence_row_index]
                valid = batch.edge_valid_mask[:, :, None, None] & anchor_valid[:, None, :, None]
                radius_valid = torch.ones_like(target, dtype=torch.bool)
                radius_valid[..., 1] = batch.material_targets.radius_valid_mask
                valid = valid & radius_valid
                active = valid & batch.material_targets.ancestor_mask[:, :, None, None]
                zero = valid & ~batch.material_targets.ancestor_mask[:, :, None, None]
                error = prediction.material_jacobian - target
                for channel in range(4):
                    mask = active[..., channel]
                    channel_error[channel] += error[..., channel][mask].double().square().sum().cpu()
                    channel_baseline[channel] += target[..., channel][mask].double().square().sum().cpu()
                    channel_count[channel] += mask.sum().double().cpu()
                    nonzero = mask & (target[..., channel].abs() >= 1.0e-5)
                    channel_sign_correct[channel] += (
                        torch.sign(prediction.material_jacobian[..., channel][nonzero])
                        == torch.sign(target[..., channel][nonzero])
                    ).sum().double().cpu()
                    channel_sign_count[channel] += nonzero.sum().double().cpu()
                zero_prediction_square += prediction.material_jacobian[zero].double().square().sum().cpu()
                zero_prediction_count += zero.sum().double().cpu()
                if include_ablations:
                    model = self.require_model()
                    entities = prediction.latents.entities
                    rows_by_asset: dict[str, list[int]] = {}
                    for row, asset_id in enumerate(batch.asset_ids):
                        rows_by_asset.setdefault(asset_id, []).append(row)
                    same_q_index = torch.arange(len(batch.asset_ids), device=entities.device)
                    groups = list(rows_by_asset.values())
                    for rows in groups:
                        index = torch.tensor(rows, device=entities.device)
                        same_q_index[index] = torch.tensor(list(reversed(rows)), device=entities.device)
                    cross_asset_index = torch.empty_like(same_q_index)
                    for group_index, rows in enumerate(groups):
                        source_rows = groups[(group_index + 1) % len(groups)]
                        if len(source_rows) != len(rows):
                            raise ValueError("cross-asset ablation requires equal q-block lengths")
                        cross_asset_index[torch.tensor(rows, device=entities.device)] = torch.tensor(
                            source_rows,
                            device=entities.device,
                        )
                    joint_shuffled = entities.clone()
                    joint_entities = batch.evidence.joint_entity_index[batch.evidence_row_index]
                    joint_valid = batch.evidence.joint_valid_mask
                    if joint_valid is None:
                        joint_valid = torch.ones_like(joint_entities, dtype=torch.bool)
                    else:
                        joint_valid = joint_valid[batch.evidence_row_index]
                    for row in range(entities.shape[0]):
                        slots = joint_entities[row, joint_valid[row]]
                        if slots.numel() > 1:
                            joint_shuffled[row, slots] = entities[row, torch.roll(slots, shifts=1)]
                    variants = {
                        "query_only": torch.zeros_like(entities),
                        "same_asset_q_shuffle": entities[same_q_index],
                        "cross_asset_shuffle": entities[cross_asset_index],
                        "joint_token_shuffle": joint_shuffled,
                    }
                    for ablation, ablated_entities in variants.items():
                        ablated_latents = type(prediction.latents)(entities=ablated_entities)
                        ablated_prediction = model.decode_features(
                            ablated_latents,
                            prediction.query_features,
                            prediction.material_pair_features,
                            batch.field_targets.bandwidths,
                            batch.evidence,
                            batch.material_targets.owner_index,
                            batch.material_targets.joint_index,
                            evidence_row_index=batch.evidence_row_index,
                            entity_valid_mask=batch.evidence.entity_valid_mask[batch.evidence_row_index]
                            if batch.evidence.entity_valid_mask is not None
                            else None,
                        )
                        ablated_step = evaluate_objectives(
                            DensityGammaObjectiveContext(ablated_prediction, batch),
                            self.config.objectives,
                        )
                        for name, result in ablated_step.items():
                            component = result.components[0]
                            ablation_totals[ablation][name][0] += float(component.numerator)
                            ablation_totals[ablation][name][1] += float(component.denominator)
                step_index += 1
        if baseline_total is None:
            raise RuntimeError("evaluation produced no teacher baseline statistics")
        baselines = self.finalize_teacher_baselines(baseline_total)
        metrics = {name: numerator[name] / denominator[name] for name in numerator}
        for name in numerator:
            baseline = baselines[name]
            if not isinstance(baseline, Mapping):
                raise TypeError(f"teacher baseline {name!r} must be a mapping")
            metrics[f"{name}_skill"] = 1.0 - metrics[name] / float(baseline["baseline_mse"])
        channel_names = ("height", "radius", "dot", "chirality")
        channel_metrics: dict[str, object] = {}
        for channel, name in enumerate(channel_names):
            mse = channel_error[channel] / channel_count[channel].clamp_min(1.0)
            zero_mse = channel_baseline[channel] / channel_count[channel].clamp_min(1.0)
            channel_metrics[name] = {
                "active_mse": float(mse),
                "active_zero_baseline": float(zero_mse),
                "active_skill": float(1.0 - mse / zero_mse.clamp_min(1.0e-30)),
                "active_sign_accuracy": float(
                    channel_sign_correct[channel] / channel_sign_count[channel].clamp_min(1.0)
                ),
                "active_scalar_count": int(channel_count[channel]),
            }
        metrics["material_jacobian_structural_zero_prediction_rms"] = float(
            torch.sqrt(zero_prediction_square / zero_prediction_count.clamp_min(1.0))
        )
        ablation_payload = None
        if include_ablations:
            aggregate = {
                ablation: {
                    name: values[0] / values[1]
                    for name, values in terms.items()
                }
                for ablation, terms in ablation_totals.items()
            }
            ablation_payload = {
                "pairing_key": ["asset_id", "q_index"],
                "ablations": ablation_names,
                "aggregate_metrics": aggregate,
                "records": [],
            }
        return MethodEvaluationReport(
            metrics=metrics,
            strata={
                "metric_scores": metrics,
                "material_jacobian_channels": channel_metrics,
                "batch_count": step_index,
            },
            teacher_baselines=baselines,
            ablations=ablation_payload,
        )

    def analyze_ablations(self, evidence: Mapping[str, Any], *, bootstrap_replicates: int, seed: int) -> dict[str, Any]:
        r"""首个 method 在无 final ablation evidence 时返回显式空报告。"""

        return {
            "record_count": len(evidence.get("records", ())),
            "bootstrap_replicates": bootstrap_replicates,
            "seed": seed,
            "aggregate_metrics": evidence.get("aggregate_metrics", {}),
            "note": "aggregate-only first formal ablation; paired bootstrap records are deferred",
        }


DensityMaterialJacobianMethodCfg.runtime_type = DensityMaterialJacobianMethod


__all__ = ["DensityMaterialJacobianMethod"]
