r"""rho/kappa 双主 objective：比较预测与 representation 真值，并按 $(asset,q)$ 等权归约。

每个 term 只读取 method 组装好的 typed context，不重新采样 query，也不重新运行 encoder。
一阶 active/zero 在样本内先分别平均，再按固定 2:1 合并；任一类缺失时其固定质量为零，不把另一类
重归一化到 1。训练残差除以固定 $L_{ref}=0.1$ m，使 $\kappa$ objective 无量纲。
"""

from __future__ import annotations

from typing import Any

import torch

from anymani.distill.methods.contracts import AdditiveStatistic, MethodStep, MethodUpdate, ObjectiveTermResult

from .config import (
    DensityObjectiveCfg,
    KappaObjectiveCfg,
    MultiAnchorGaussianObjectivesCfg,
)
from .context import MultiAnchorObjectiveContext

KAPPA_PHYSICAL_SCALE_M = 0.1
"""距离 Jacobian 残差的全手固定参考尺度，单位 m/rad。"""

KAPPA_ACTIVE_FRACTION = 2.0 / 3.0
"""每个 $(asset,q)$ 的 active Jacobian 监督质量。"""

KAPPA_STRUCTURAL_ZERO_FRACTION = 1.0 / 3.0
"""每个 $(asset,q)$ 的 non-ancestor structural-zero 监督质量。"""


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
    r"""每个 $(asset,q)$ 先分别平均 active 与 zero，再按固定 2:1 合并。"""

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
    active_present = active_den > 0.0
    zero_present = zero_den > 0.0
    active_mean = torch.where(active_present, active_num / active_den.clamp_min(1.0), 0.0)
    zero_mean = torch.where(zero_present, zero_num / zero_den.clamp_min(1.0), 0.0)
    sample_value = KAPPA_ACTIVE_FRACTION * active_mean + KAPPA_STRUCTURAL_ZERO_FRACTION * zero_mean
    sample_count = torch.where(active_present | zero_present, torch.ones_like(sample_count), sample_count)
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
    r"""$\mathcal L_\kappa$：残差除以 0.1 m/rad 后按 active/zero 2:1 合并。"""

    physical_error = context.kappa_prediction - context.kappa_target
    numerator, denominator = _split_active_zero_mean(
        physical_error / KAPPA_PHYSICAL_SCALE_M,
        context.edge_valid_mask,
        context.active_mask,
    )
    physical_numerator, physical_denominator = _split_active_zero_mean(
        physical_error,
        context.edge_valid_mask,
        context.active_mask,
    )
    if not torch.equal(denominator, physical_denominator):
        raise RuntimeError("normalized and physical kappa reductions disagree on valid sample count")
    statistic = AdditiveStatistic("kappa", numerator, denominator)
    physical_mse = physical_numerator / physical_denominator
    return ObjectiveTermResult(
        "kappa",
        (statistic,),
        {
            "loss": statistic.mean,
            "physical_mse": physical_mse,
            "physical_rms": physical_mse.clamp_min(0.0).sqrt(),
        },
    )


def teacher_baseline_sufficient_statistics(batch: Any) -> dict[str, torch.Tensor]:
    r"""从 teacher-only batch 累计 constant-density 与 zero-kappa baseline 的充分统计。

    对 bandwidth slot $s$，每个有效样本先在 owner/query 轴求一阶矩 $m_{b,s}$ 与二阶矩
    $v_{b,s}$。跨样本等权累加后，最优常数与 baseline MSE 为：

    $$
    c_s=\frac{\sum_bm_{b,s}}{N_s},\qquad
    B_\rho=\frac{\sum_{b,s}(v_{b,s}-2c_sm_{b,s}+c_s^2)}{\sum_sN_s}.
    $$

    query stratum 只重复记录同类矩，不参与 $c_s$ 或 $B_\rho$ 的估计。$B_\kappa$ 令预测为零，
    并直接调用正式 objective 的无量纲 active/zero 归约，确保 structural-zero 权重完全一致。
    """

    field = batch.field_targets  # concrete method batch；truth 不进入 model
    sensitivity = batch.sensitivity_targets
    density = field.density.detach()  # `[B,G,N_Q,N_sigma]`，teacher-only
    valid = field.valid_mask.detach()  # `[B,G,N_Q]`
    valid_weight = valid.to(density.dtype)
    scalar_count = valid_weight.reshape(density.shape[0], -1).sum(dim=-1)  # 每行有效 owner/query 数
    sample_valid = scalar_count > 0.0
    normalized_weight = valid_weight / scalar_count.clamp_min(1.0)[:, None, None]
    slot_mean = (density * normalized_weight.unsqueeze(-1)).sum(dim=(1, 2))  # `[B,N_sigma]`
    slot_second = (density.square() * normalized_weight.unsqueeze(-1)).sum(dim=(1, 2))
    slot_weight = sample_valid.to(density.dtype).unsqueeze(-1).expand_as(slot_mean)

    # 三个 query strata 只作可重算诊断；它们不能改变 constant predictor 或训练分母。
    stratum_mean_sum = torch.zeros(3, density.shape[-1], device=density.device, dtype=density.dtype)
    stratum_second_sum = torch.zeros_like(stratum_mean_sum)
    stratum_sample_count = torch.zeros_like(stratum_mean_sum)
    for stratum in range(3):
        stratum_valid = valid & (field.query_stratum == stratum)
        stratum_weight = stratum_valid.to(density.dtype)
        stratum_count = stratum_weight.reshape(density.shape[0], -1).sum(dim=-1)
        row_valid = stratum_count > 0.0
        row_weight = stratum_weight / stratum_count.clamp_min(1.0)[:, None, None]
        row_mean = (density * row_weight.unsqueeze(-1)).sum(dim=(1, 2))
        row_second = (density.square() * row_weight.unsqueeze(-1)).sum(dim=(1, 2))
        stratum_mean_sum[stratum] = row_mean[row_valid].sum(dim=0)
        stratum_second_sum[stratum] = row_second[row_valid].sum(dim=0)
        stratum_sample_count[stratum] = row_valid.to(density.dtype).sum()

    kappa_num, kappa_den = _split_active_zero_mean(
        sensitivity.kappa.detach() / KAPPA_PHYSICAL_SCALE_M,
        sensitivity.valid_mask.detach(),
        sensitivity.active_mask.detach(),
    )
    active_mask = sensitivity.active_mask.detach()
    if active_mask.ndim == 1:
        active_mask = active_mask.unsqueeze(0).expand_as(sensitivity.valid_mask)
    kappa_stratum_sum = torch.zeros(2, device=density.device, dtype=density.dtype)
    kappa_stratum_count = torch.zeros_like(kappa_stratum_sum)
    for stratum, mask in enumerate(
        (sensitivity.valid_mask & active_mask, sensitivity.valid_mask & ~active_mask)
    ):
        weight = mask.to(density.dtype)
        row_count = weight.sum(dim=-1)
        row_valid = row_count > 0.0
        row_second = (sensitivity.kappa.detach().square() * weight).sum(dim=-1) / row_count.clamp_min(1.0)
        kappa_stratum_sum[stratum] = row_second[row_valid].sum()
        kappa_stratum_count[stratum] = row_valid.to(density.dtype).sum()
    return {
        "density_slot_mean_sum": (slot_mean * slot_weight).sum(dim=0),
        "density_slot_second_sum": (slot_second * slot_weight).sum(dim=0),
        "density_slot_sample_count": slot_weight.sum(dim=0),
        "density_stratum_mean_sum": stratum_mean_sum,
        "density_stratum_second_sum": stratum_second_sum,
        "density_stratum_sample_count": stratum_sample_count,
        "kappa_zero_numerator": kappa_num,
        "kappa_zero_denominator": kappa_den,
        "kappa_stratum_second_sum": kappa_stratum_sum,
        "kappa_stratum_sample_count": kappa_stratum_count,
    }


def merge_teacher_baseline_statistics(
    total: dict[str, torch.Tensor] | None,
    block: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    r"""逐字段相加 teacher baseline 充分统计，不保留 batch 或 autograd 图。"""

    if total is None:
        return {name: value.detach().clone() for name, value in block.items()}
    if total.keys() != block.keys():
        raise ValueError("teacher baseline statistic fields changed within one catalog pass")
    return {name: total[name] + block[name].detach() for name in total}


def finalize_teacher_baselines(statistics: dict[str, torch.Tensor]) -> dict[str, object]:
    r"""把单遍充分统计闭合为训练使用的 $B_\rho,B_\kappa$ 与诊断矩。"""

    mean_sum = statistics["density_slot_mean_sum"].to(torch.float64)
    second_sum = statistics["density_slot_second_sum"].to(torch.float64)
    count = statistics["density_slot_sample_count"].to(torch.float64)
    if torch.any(count <= 0.0):
        raise ValueError("teacher density baseline has an empty bandwidth slot")
    constant_mean = mean_sum / count  # 每个 bandwidth slot 的 teacher-only constant predictor
    density_sse = second_sum - 2.0 * constant_mean * mean_sum + constant_mean.square() * count
    density_baseline = density_sse.sum() / count.sum()  # 与正式 per-sample density reduction 同测度
    kappa_den = statistics["kappa_zero_denominator"].to(torch.float64)
    if float(kappa_den) <= 0.0:
        raise ValueError("teacher kappa baseline has no valid active/zero samples")
    kappa_baseline = statistics["kappa_zero_numerator"].to(torch.float64) / kappa_den
    if float(density_baseline) <= 0.0 or float(kappa_baseline) <= 0.0:
        raise ValueError("teacher baselines must be strictly positive normalization constants")

    stratum_mean_sum = statistics["density_stratum_mean_sum"].to(torch.float64)
    stratum_second_sum = statistics["density_stratum_second_sum"].to(torch.float64)
    stratum_count = statistics["density_stratum_sample_count"].to(torch.float64)
    stratum_mean = stratum_mean_sum / stratum_count.clamp_min(1.0)
    stratum_second = stratum_second_sum / stratum_count.clamp_min(1.0)
    kappa_stratum_count = statistics["kappa_stratum_sample_count"].to(torch.float64)
    kappa_stratum_second = (
        statistics["kappa_stratum_second_sum"].to(torch.float64) / kappa_stratum_count.clamp_min(1.0)
    )
    return {
        "density": {
            "predictor": "constant_teacher_mean_per_bandwidth_slot",
            "constant_mean": constant_mean.tolist(),
            "teacher_second_moment": (second_sum / count).tolist(),
            "baseline_mse": float(density_baseline),
            "sample_count_per_slot": count.tolist(),
            "query_strata_diagnostic": {
                "mean": stratum_mean.tolist(),
                "second_moment": stratum_second.tolist(),
                "sample_count": stratum_count.tolist(),
            },
        },
        "kappa": {
            "predictor": "zero",
            "reduction": "per-sample active/structural-zero means combined 2:1",
            "physical_scale_m": KAPPA_PHYSICAL_SCALE_M,
            "baseline_mse": float(kappa_baseline),  # 训练 objective 口径，残差已除以 0.1 m/rad
            "objective_baseline_mse": float(kappa_baseline),
            "physical_baseline_mse": float(kappa_baseline * KAPPA_PHYSICAL_SCALE_M**2),
            "sample_count": float(kappa_den),
            "strata_diagnostic": {
                "names": ["active", "structural_zero"],
                "second_moment": kappa_stratum_second.tolist(),
                "sample_count": kappa_stratum_count.tolist(),
            },
        },
    }


DensityObjectiveCfg.func = density_objective
KappaObjectiveCfg.func = kappa_objective


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
    r"""跨 accumulation 合并分任务 MSE；不构造可反传统一总损失。"""

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
    terms: dict[str, float] = {}
    enabled = config.enabled()
    for name in enabled:
        numerator, denominator = totals[name]
        mean = numerator / denominator
        terms[name] = float(mean.detach())
    if not terms:
        raise ValueError("method update contains no enabled objectives")
    return MethodUpdate(
        terms=terms,
        sample_count=sample_count,
        denominators={name: float(totals[name][1].detach()) for name in enabled},
    )


__all__ = [
    "density_objective",
    "evaluate_objectives",
    "finalize_teacher_baselines",
    "kappa_objective",
    "merge_teacher_baseline_statistics",
    "reduce_method_steps",
    "teacher_baseline_sufficient_statistics",
]
