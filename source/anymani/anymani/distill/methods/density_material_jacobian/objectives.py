r"""Gaussian density 与四通道 relational Material Jacobian 的 additive objectives。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch

from anymani.distill.methods.contracts import AdditiveStatistic, MethodStep, MethodUpdate, ObjectiveTermResult
from anymani.distill.models.density_material_jacobian_ssl import DensityMaterialJacobianForward

from .batch import PaddedDensityGammaBatch
from .config import (
    DensityMaterialJacobianObjectivesCfg,
    DensityObjectiveCfg,
    MaterialJacobianObjectiveCfg,
)

ACTIVE_FRACTION = 2.0 / 3.0
STRUCTURAL_ZERO_FRACTION = 1.0 / 3.0


@dataclass(frozen=True)
class DensityGammaObjectiveContext:
    r"""联合 prediction、teacher truth 与 masks 的单次 objective 输入。"""

    prediction: DensityMaterialJacobianForward
    batch: PaddedDensityGammaBatch


def _per_sample_mean(error: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""先在每个 `(asset,q)` 行内平均，再对有效 rows 等权。"""

    while mask.ndim < error.ndim:
        mask = mask.unsqueeze(-1)
    weight = mask.to(error.dtype).expand_as(error)
    denominator = weight.reshape(error.shape[0], -1).sum(dim=-1)
    numerator = (weight * error.square()).reshape(error.shape[0], -1).sum(dim=-1)
    valid = denominator > 0.0
    mean = torch.where(valid, numerator / denominator.clamp_min(1.0), torch.zeros_like(numerator))
    if not bool(valid.any()):
        raise ValueError("objective received no valid (asset,q) rows")
    return mean[valid].sum(), valid.to(error.dtype).sum()


def _active_zero_mean(
    error: torch.Tensor,
    valid_mask: torch.Tensor,
    active_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""每 row 分别平均 active/zero scalar targets，再固定按 2:1 组合。"""

    while valid_mask.ndim < error.ndim:
        valid_mask = valid_mask.unsqueeze(-1)
    while active_mask.ndim < error.ndim:
        active_mask = active_mask.unsqueeze(-1)
    active_weight = (valid_mask & active_mask).to(error.dtype).expand_as(error)
    zero_weight = (valid_mask & ~active_mask).to(error.dtype).expand_as(error)
    active_den = active_weight.reshape(error.shape[0], -1).sum(dim=-1)
    zero_den = zero_weight.reshape(error.shape[0], -1).sum(dim=-1)
    active_num = (active_weight * error.square()).reshape(error.shape[0], -1).sum(dim=-1)
    zero_num = (zero_weight * error.square()).reshape(error.shape[0], -1).sum(dim=-1)
    active_present = active_den > 0.0
    zero_present = zero_den > 0.0
    row_valid = active_present | zero_present
    row_value = ACTIVE_FRACTION * torch.where(
        active_present,
        active_num / active_den.clamp_min(1.0),
        0.0,
    ) + STRUCTURAL_ZERO_FRACTION * torch.where(
        zero_present,
        zero_num / zero_den.clamp_min(1.0),
        0.0,
    )
    if not bool(row_valid.any()):
        raise ValueError("Gamma objective received no active or structural-zero rows")
    return row_value[row_valid].sum(), row_valid.to(error.dtype).sum()


def density_objective(context: DensityGammaObjectiveContext) -> ObjectiveTermResult:
    r"""完整 owner/query/sigma Gaussian density raw MSE。"""

    numerator, denominator = _per_sample_mean(
        context.prediction.density - context.batch.field_targets.density,
        context.batch.field_targets.valid_mask,
    )
    statistic = AdditiveStatistic("density", numerator, denominator)
    return ObjectiveTermResult("density", (statistic,), {"loss": statistic.mean})


def material_jacobian_objective(
    context: DensityGammaObjectiveContext,
    config: MaterialJacobianObjectiveCfg,
) -> ObjectiveTermResult:
    r"""四通道 Gamma 残差按固定 physical scales、channel mask 与 active/zero 2:1 归约。"""

    prediction = context.prediction.material_jacobian
    target = context.batch.material_targets.relation_sensitivity_per_rad
    scale = torch.tensor(config.channel_scale.values, device=prediction.device, dtype=prediction.dtype)
    error = (prediction - target) / scale  # `[B,E,K,4]`，无量纲 residual
    evidence_rows = context.batch.evidence_row_index
    anchor_valid = context.batch.evidence.anchor_valid_mask
    if anchor_valid is None:
        anchor_valid = torch.ones(
            context.batch.evidence.anchors.shape[:-1],
            device=prediction.device,
            dtype=torch.bool,
        )
    anchor_valid = anchor_valid[evidence_rows]  # `[B,K]`
    channel_valid = torch.ones_like(target, dtype=torch.bool)
    channel_valid[..., 1] = context.batch.material_targets.radius_valid_mask  # radius 独立奇点 mask
    valid = (
        context.batch.edge_valid_mask[:, :, None, None]
        & anchor_valid[:, None, :, None]
        & channel_valid
    )
    numerator, denominator = _active_zero_mean(
        error,
        valid,
        context.batch.material_targets.ancestor_mask,
    )
    statistic = AdditiveStatistic("material_jacobian", numerator, denominator)
    metrics: dict[str, torch.Tensor] = {"loss": statistic.mean}
    for channel, name in enumerate(("height", "radius", "dot", "chirality")):
        channel_num, channel_den = _active_zero_mean(
            prediction[..., channel] - target[..., channel],
            valid[..., channel],
            context.batch.material_targets.ancestor_mask,
        )
        metrics[f"{name}_physical_rms"] = (channel_num / channel_den).clamp_min(0.0).sqrt()
    return ObjectiveTermResult("material_jacobian", (statistic,), metrics)


DensityObjectiveCfg.func = density_objective
MaterialJacobianObjectiveCfg.func = material_jacobian_objective


def evaluate_objectives(
    context: DensityGammaObjectiveContext,
    config: DensityMaterialJacobianObjectivesCfg,
) -> dict[str, ObjectiveTermResult]:
    r"""按配置稳定顺序计算 density 与 Gamma。"""

    return {
        "density": density_objective(context),
        "material_jacobian": material_jacobian_objective(context, config.material_jacobian),
    }


def reduce_method_steps(
    steps: tuple[MethodStep, ...],
    config: DensityMaterialJacobianObjectivesCfg,
) -> MethodUpdate:
    r"""跨 units 合并 additive numerators/denominators。"""

    totals: dict[str, list[torch.Tensor]] = {}
    sample_count = 0
    for step in steps:
        sample_count += int(step.sample_count)
        for result in step.objectives.values():
            for component in result.components:
                pair = totals.get(component.name)
                if pair is None:
                    totals[component.name] = [component.numerator, component.denominator]
                else:
                    pair[0] = pair[0] + component.numerator
                    pair[1] = pair[1] + component.denominator
    terms = {name: float((totals[name][0] / totals[name][1]).detach()) for name in config.enabled()}
    return MethodUpdate(
        terms=terms,
        sample_count=sample_count,
        denominators={name: float(totals[name][1].detach()) for name in config.enabled()},
    )


def teacher_baseline_statistics(batch: PaddedDensityGammaBatch, config: DensityMaterialJacobianObjectivesCfg) -> dict[str, torch.Tensor]:
    r"""累计 constant-density 与 zero-Gamma baseline 的充分统计。"""

    density = batch.field_targets.density.detach()
    valid = batch.field_targets.valid_mask.detach()
    weight = valid.to(density.dtype)
    row_count = weight.reshape(density.shape[0], -1).sum(dim=-1)
    row_valid = row_count > 0.0
    normalized = weight / row_count.clamp_min(1.0)[:, None, None]
    slot_mean = (density * normalized.unsqueeze(-1)).sum(dim=(1, 2))
    slot_second = (density.square() * normalized.unsqueeze(-1)).sum(dim=(1, 2))

    zero_prediction = DensityMaterialJacobianForward(
        latents=cast(Any, None),
        query_features=density.new_empty(0),
        material_pair_features=density.new_empty(0),
        density=density,
        material_jacobian=torch.zeros_like(batch.material_targets.relation_sensitivity_per_rad),
    )
    gamma_result = material_jacobian_objective(
        DensityGammaObjectiveContext(zero_prediction, batch),
        config.material_jacobian,
    )
    gamma_stat = gamma_result.components[0]
    return {
        "density_slot_mean_sum": slot_mean[row_valid].sum(dim=0),
        "density_slot_second_sum": slot_second[row_valid].sum(dim=0),
        "density_slot_sample_count": row_valid.to(density.dtype).sum().expand(slot_mean.shape[-1]).clone(),
        "material_jacobian_zero_numerator": gamma_stat.numerator.detach(),
        "material_jacobian_zero_denominator": gamma_stat.denominator.detach(),
    }


def merge_teacher_baseline_statistics(
    total: dict[str, torch.Tensor] | None,
    block: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    r"""逐字段相加 teacher baseline 充分统计。"""

    if total is None:
        return {name: value.detach().clone() for name, value in block.items()}
    if total.keys() != block.keys():
        raise ValueError("teacher baseline fields changed within one run")
    return {name: total[name] + block[name].detach() for name in total}


def finalize_teacher_baselines(statistics: dict[str, torch.Tensor]) -> dict[str, object]:
    r"""形成 canonical density constant baseline 与 zero-Gamma objective baseline。"""

    mean_sum = statistics["density_slot_mean_sum"].double()
    second_sum = statistics["density_slot_second_sum"].double()
    count = statistics["density_slot_sample_count"].double()
    constant = mean_sum / count
    density_sse = second_sum - 2.0 * constant * mean_sum + constant.square() * count
    density_baseline = density_sse.sum() / count.sum()
    gamma_num = statistics["material_jacobian_zero_numerator"].double()
    gamma_den = statistics["material_jacobian_zero_denominator"].double()
    gamma_baseline = gamma_num / gamma_den
    if float(density_baseline) <= 0.0 or float(gamma_baseline) <= 0.0:
        raise ValueError("density and Gamma teacher baselines must be positive")
    return {
        "density": {
            "predictor": "constant_teacher_mean_per_bandwidth_slot",
            "constant_mean": constant.tolist(),
            "baseline_mse": float(density_baseline),
            "sample_count_per_slot": count.tolist(),
        },
        "material_jacobian": {
            "predictor": "zero",
            "reduction": "four scaled channels; per-sample active/structural-zero means combined 2:1",
            "baseline_mse": float(gamma_baseline),
            "sample_count": float(gamma_den),
        },
    }


__all__ = [
    "DensityGammaObjectiveContext",
    "density_objective",
    "evaluate_objectives",
    "finalize_teacher_baselines",
    "material_jacobian_objective",
    "merge_teacher_baseline_statistics",
    "reduce_method_steps",
    "teacher_baseline_statistics",
]
