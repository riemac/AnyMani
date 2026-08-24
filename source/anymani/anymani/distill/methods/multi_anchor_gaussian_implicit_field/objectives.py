r"""三项主 objective：比较预测与 representation 真值，并按 $(asset,q)$ 等权归约。

每个 term 只读取 method 组装好的 typed context，不重新采样 query，也不重新运行 encoder。
一阶 active/zero 在样本内先分别平均，再按 1:1 合并，避免 active 被最近点 mask 后让 zero 主导。
"""

from __future__ import annotations

import torch

from anymani.distill.methods.contracts import AdditiveStatistic, MethodStep, MethodUpdate, ObjectiveTermResult

from .config import (
    DensityObjectiveCfg,
    DerivedFieldObjectiveCfg,
    KappaObjectiveCfg,
    MultiAnchorGaussianObjectivesCfg,
)
from .context import MultiAnchorObjectiveContext


def _per_sample_square_mean(error: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""对每个 batch 行独立计算有效标量 MSE，并返回 $(N_b, D_b)$ 再等权合成。"""

    while mask.ndim < error.ndim:
        mask = mask.unsqueeze(-1)
    weight = mask.to(error.dtype).expand_as(error)
    per_sample_denominator = weight.reshape(error.shape[0], -1).sum(dim=-1)
    per_sample_numerator = (weight * error.square()).reshape(error.shape[0], -1).sum(dim=-1)
    valid = per_sample_denominator > 0.0
    per_sample_mean = torch.where(
        valid,
        per_sample_numerator / per_sample_denominator.clamp_min(1.0),
        torch.zeros_like(per_sample_numerator),
    )
    numerator = per_sample_mean.sum()
    denominator = valid.to(error.dtype).sum()
    if float(denominator.detach()) <= 0.0:
        raise ValueError("objective term received no valid (asset,q) realizations")
    return numerator, denominator


def _split_active_zero_mean(
    error: torch.Tensor,
    valid_mask: torch.Tensor,
    active_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""每个 $(asset,q)$ 先分别平均 active 与 zero，再 1:1 合并。"""

    if active_mask.ndim == 1:
        active_mask = active_mask.unsqueeze(0).expand(error.shape[0], -1)
    while valid_mask.ndim < error.ndim:
        valid_mask = valid_mask.unsqueeze(-1)
    while active_mask.ndim < error.ndim:
        active_mask = active_mask.unsqueeze(-1)
    active_valid = valid_mask & active_mask
    zero_valid = valid_mask & ~active_mask
    active_weight = active_valid.to(error.dtype).expand_as(error)
    zero_weight = zero_valid.to(error.dtype).expand_as(error)
    active_den = active_weight.reshape(error.shape[0], -1).sum(dim=-1)
    zero_den = zero_weight.reshape(error.shape[0], -1).sum(dim=-1)
    active_num = (active_weight * error.square()).reshape(error.shape[0], -1).sum(dim=-1)
    zero_num = (zero_weight * error.square()).reshape(error.shape[0], -1).sum(dim=-1)
    sample_value = torch.zeros(error.shape[0], device=error.device, dtype=error.dtype)
    sample_count = torch.zeros(error.shape[0], device=error.device, dtype=error.dtype)
    both = (active_den > 0.0) & (zero_den > 0.0)
    only_active = (active_den > 0.0) & (zero_den <= 0.0)
    only_zero = (zero_den > 0.0) & (active_den <= 0.0)
    sample_value = torch.where(both, 0.5 * (active_num / active_den.clamp_min(1.0) + zero_num / zero_den.clamp_min(1.0)), sample_value)
    sample_value = torch.where(only_active, active_num / active_den.clamp_min(1.0), sample_value)
    sample_value = torch.where(only_zero, zero_num / zero_den.clamp_min(1.0), sample_value)
    sample_count = torch.where(both | only_active | only_zero, torch.ones_like(sample_count), sample_count)
    if float(sample_count.sum().detach()) <= 0.0:
        raise ValueError("kappa/g term received no valid active or zero edges")
    return (sample_value * sample_count).sum(), sample_count.sum()


def density_objective(context: MultiAnchorObjectiveContext) -> ObjectiveTermResult:
    r"""$\mathcal L_\rho=\mathbb E_{(asset,q)}[\mathrm{MSE}(\hat\rho,\rho)]$。"""

    numerator, denominator = _per_sample_square_mean(
        context.density_prediction - context.density_target,
        context.density_valid_mask,
    )
    statistic = AdditiveStatistic("density", numerator, denominator)
    return ObjectiveTermResult("density", (statistic,), {"loss": statistic.mean})


def kappa_objective(context: MultiAnchorObjectiveContext) -> ObjectiveTermResult:
    r"""$\mathcal L_\kappa$：active/zero 先分别平均，再 1:1 合并。"""

    numerator, denominator = _split_active_zero_mean(
        context.kappa_prediction - context.kappa_target,
        context.edge_valid_mask,
        context.active_mask,
    )
    statistic = AdditiveStatistic("kappa", numerator, denominator)
    return ObjectiveTermResult("kappa", (statistic,), {"loss": statistic.mean})


def derived_field_objective(context: MultiAnchorObjectiveContext) -> ObjectiveTermResult:
    r"""$\mathcal L_g^{(\kappa)}$：$\hat g^{(\kappa)}$ 对齐 teacher $g$。"""

    numerator, denominator = _split_active_zero_mean(
        context.derived_field_sensitivity - context.field_sensitivity_target,
        context.edge_valid_mask,
        context.active_mask,
    )
    statistic = AdditiveStatistic("derived_field", numerator, denominator)
    return ObjectiveTermResult("derived_field", (statistic,), {"loss": statistic.mean})


DensityObjectiveCfg.func = density_objective
KappaObjectiveCfg.func = kappa_objective
DerivedFieldObjectiveCfg.func = derived_field_objective


def evaluate_objectives(
    context: MultiAnchorObjectiveContext, config: MultiAnchorGaussianObjectivesCfg
) -> dict[str, ObjectiveTermResult]:
    r"""按配置顺序计算全部开启的 objective。"""

    results: dict[str, ObjectiveTermResult] = {}
    for name, term_cfg in config.enabled().items():
        func = type(term_cfg).func
        if func is None:
            raise RuntimeError(f"{type(term_cfg).__name__} has not bound its objective function")
        results[name] = func(context)
    return results


def reduce_method_steps(
    steps: tuple[MethodStep, ...],
    config: MultiAnchorGaussianObjectivesCfg,
) -> MethodUpdate:
    r"""跨 accumulation 的 $(asset,q)$ 等权合并。"""

    totals: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    sample_count = 0
    for step in steps:
        sample_count += int(step.sample_count)
        for name, result in step.objectives.items():
            for component in result.components:
                previous = totals.get(component.name)
                if previous is None:
                    totals[component.name] = (component.numerator, component.denominator)
                else:
                    totals[component.name] = (previous[0] + component.numerator, previous[1] + component.denominator)
    loss: torch.Tensor | None = None
    terms: dict[str, float] = {}
    enabled = config.enabled()
    for name, term_cfg in enabled.items():
        numerator, denominator = totals[name]
        mean = numerator / denominator
        terms[name] = float(mean.detach())
        weighted = float(term_cfg.weight) * mean
        loss = weighted if loss is None else loss + weighted
    if loss is None:
        raise ValueError("method update contains no enabled objectives")
    return MethodUpdate(
        loss=loss,
        terms=terms,
        sample_count=sample_count,
        denominators={name: float(totals[name][1].detach()) for name in enabled},
    )


__all__ = [
    "density_objective",
    "derived_field_objective",
    "evaluate_objectives",
    "kappa_objective",
    "reduce_method_steps",
]
