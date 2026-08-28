r"""Geometry SSL 的两任务 shared-gradient 聚合数学。

对 retained encoder 的 density 与 $\kappa$ 参数梯度分别记为 $g_\rho,g_\kappa$。两任务
$\alpha=1$ FairGrad 在非零、非反向情形使用解析解：

$$
w_j=\frac{1}{\|g_j\|\sqrt{1+c}},\qquad
d=w_\rho g_\rho+w_\kappa g_\kappa,
\qquad c=\frac{g_\rho^Tg_\kappa}{\|g_\rho\|\|g_\kappa\|}.
$$

本模块只拥有纯梯度几何，不拥有 forward、loss、optimizer 或训练循环。所有范数、点积与余弦都
在 FP64 中累计；组合梯度再写回原参数 dtype/device。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from anymani.distill.methods.contracts import MethodParameterGroup, MethodUpdate
from anymani.distill.representations.targets.field_samples import (
    QueryStratum,
    SensitivityOwnerCategory,
    SensitivitySamplingRole,
)

from .augmentation import maybe_rewrite_batch
from .batch import PaddedOnlineGeometryBatch, split_padded_online_geometry_batch


@dataclass(frozen=True)
class FairGradEvidence:
    """一次 shared 聚合的可审计标量证据。"""

    density_norm: float
    kappa_norm: float
    cosine: float
    density_weight: float
    kappa_weight: float
    combined_norm: float
    shared_conflict_blocked: bool
    active_tasks: int


@dataclass(frozen=True)
class FairGradResult:
    """与 shared 参数一一对应的组合梯度及其数值证据。"""

    combined: tuple[torch.Tensor | None, ...]
    evidence: FairGradEvidence


@dataclass(frozen=True)
class GradientGroupEvidence:
    """一个 named optimizer parameter group 的独立裁剪证据。"""

    pre_clip_norm: float
    post_clip_norm: float
    clip_ratio: float
    hit: bool
    active_parameter_count: int


def _validate_gradient_layout(
    density_gradients: Sequence[torch.Tensor | None],
    kappa_gradients: Sequence[torch.Tensor | None],
) -> None:
    """保证两个任务描述同一组 shared 参数，并在聚合前拒绝非有限张量。"""

    if len(density_gradients) != len(kappa_gradients) or not density_gradients:
        raise ValueError("FairGrad task gradients must describe the same non-empty parameter sequence")
    for index, (density, kappa) in enumerate(zip(density_gradients, kappa_gradients, strict=True)):
        if density is not None and kappa is not None and density.shape != kappa.shape:
            raise ValueError(f"FairGrad gradient shape mismatch at shared parameter {index}")
        for task_name, gradient in (("density", density), ("kappa", kappa)):
            if gradient is not None and not bool(torch.isfinite(gradient).all().item()):
                raise FloatingPointError(f"non-finite {task_name} shared gradient at parameter {index}")


def _collect_latent_gradients(
    method: Any,
    batch: PaddedOnlineGeometryBatch,
    prediction: Any,
    denominators: Mapping[str, torch.Tensor],
    *,
    normalize: bool,
) -> dict[str, torch.Tensor]:
    r"""在独立 leaf $Z$ 上提取 density/$\kappa$ objective 的 latent gradients。

    AOTAutograd 对 compiled forward 的参数反向是稳定的，但 compiled function 返回的中间 $Z$ 不保证
    仍是 readers 输出图中的可微输入节点。因此 concrete Method 使用已返回的数值 $Z$ 重算 readers，
    这里负责读取该独立图的两个 objective 梯度。完整 batch 路径传入 ``normalize=True``，直接使用
    完整 $D_j$；streaming 路径传入 ``normalize=False``，跨 unit 累计 raw numerator gradient，并在
    所有 unit 完成后只除以一次完整 $D_j$。若任一 objective 没有连接到 $Z$，直接抛出错误；将断图
    误记为零梯度会污染统一表示的 gradient Gram 诊断。

    Args:
        method (Any): 拥有具体 model/reader 结构的 MultiAnchorGaussianMethod。
        batch (PaddedOnlineGeometryBatch): 当前 unit 的 truth mask 与 objective 条件。
        prediction (Any): 主 forward 返回的 compiled prediction，提供数值 unified $Z$。
        denominators (Mapping[str, torch.Tensor]): 完整 logical minibatch 的 task denominator $D_j$。
        normalize (bool): 是否在本 unit 内按完整 denominator 形成 normalized objective gradient。

    Returns:
        dict[str, torch.Tensor]: density 与 κ 对 `[B,G,D]` latent $Z$ 的 detached FP64 梯度。

    Raises:
        RuntimeError: objective 结构不是单一 additive component，或 objective 与 $Z$ 断开时抛出。
    """

    diagnostic_step, diagnostic_prediction = method._forward_latent_diagnostic(batch, prediction)
    gradients: dict[str, torch.Tensor] = {}  # 两项 objective 的 $\partial L_j/\partial Z$，后续用 FP64 统计
    entities = diagnostic_prediction.latents.entities  # 独立 leaf unified $Z$，形状 `[B,G,D]`
    for task_name in ("density", "kappa"):
        result = diagnostic_step.objectives.get(task_name)
        if result is None or len(result.components) != 1 or result.components[0].name != task_name:
            raise RuntimeError(f"latent diagnostic requires one same-name additive component for {task_name}")
        term = (
            result.components[0].numerator / denominators[task_name]
            if normalize
            else result.components[0].numerator
        )  # 完整路径除以 $D_j$；streaming 路径保留 raw numerator
        try:
            gradient = torch.autograd.grad(term, entities, retain_graph=True, allow_unused=False)[0]
        except RuntimeError as error:
            raise RuntimeError(f"{task_name} latent diagnostic objective is disconnected from unified Z") from error
        if gradient is None:
            raise RuntimeError(f"{task_name} latent diagnostic objective returned no unified-Z gradient")
        gradients[task_name] = gradient.detach().to(torch.float64)  # Gram 统计的统一 FP64 累计输入
    return gradients


def _norm_squared(gradients: Sequence[torch.Tensor | None]) -> torch.Tensor:
    """以 FP64 累计一个分布式参数序列的平方范数。"""

    terms = [gradient.detach().to(torch.float64).square().sum() for gradient in gradients if gradient is not None]
    if not terms:
        return torch.zeros((), dtype=torch.float64)
    return torch.stack(terms).sum()


def _dot(
    left: Sequence[torch.Tensor | None],
    right: Sequence[torch.Tensor | None],
) -> torch.Tensor:
    """以 FP64 累计两个同参数布局梯度的内积；unused 参数按零处理。"""

    terms = [
        density.detach().to(torch.float64).mul(kappa.detach().to(torch.float64)).sum()
        for density, kappa in zip(left, right, strict=True)
        if density is not None and kappa is not None
    ]
    if not terms:
        return torch.zeros((), dtype=torch.float64)
    return torch.stack(terms).sum()


def _unit_gradient(gradients: Sequence[torch.Tensor | None], norm: float) -> tuple[torch.Tensor | None, ...]:
    """把单个非零任务梯度单位化，同时保留 unused 参数的 ``None``。"""

    return tuple(None if gradient is None else gradient / norm for gradient in gradients)


def combine_fairgrad(
    density_gradients: Sequence[torch.Tensor | None],
    kappa_gradients: Sequence[torch.Tensor | None],
    *,
    near_opposition_tolerance: float = 1.0e-6,
) -> FairGradResult:
    r"""返回两任务 $\alpha=1$ FairGrad 组合方向。

    ``near_opposition_tolerance`` 作用于 $1+c$。若两任务近乎完全反向，解析权重发散且共同下降方向
    不再可辨，因此返回全 ``None``，由调用者跳过 shared AdamW 更新；两个 private reader 不受影响。
    """

    if not 0.0 < near_opposition_tolerance < 1.0:
        raise ValueError("near_opposition_tolerance must lie in (0,1)")
    _validate_gradient_layout(density_gradients, kappa_gradients)
    density_norm = math.sqrt(float(_norm_squared(density_gradients).item()))
    kappa_norm = math.sqrt(float(_norm_squared(kappa_gradients).item()))
    if not math.isfinite(density_norm) or not math.isfinite(kappa_norm):
        raise FloatingPointError("non-finite FairGrad task norm")

    if density_norm == 0.0 and kappa_norm == 0.0:
        evidence = FairGradEvidence(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, 0)
        return FairGradResult(tuple(None for _ in density_gradients), evidence)
    if density_norm == 0.0:
        combined = _unit_gradient(kappa_gradients, kappa_norm)
        evidence = FairGradEvidence(0.0, kappa_norm, 0.0, 0.0, 1.0 / kappa_norm, 1.0, False, 1)
        return FairGradResult(combined, evidence)
    if kappa_norm == 0.0:
        combined = _unit_gradient(density_gradients, density_norm)
        evidence = FairGradEvidence(density_norm, 0.0, 0.0, 1.0 / density_norm, 0.0, 1.0, False, 1)
        return FairGradResult(combined, evidence)

    dot = float(_dot(density_gradients, kappa_gradients).item())
    cosine = max(-1.0, min(1.0, dot / (density_norm * kappa_norm)))
    if not math.isfinite(dot) or not math.isfinite(cosine):
        raise FloatingPointError("non-finite FairGrad dot product or cosine")
    if 1.0 + cosine <= near_opposition_tolerance:
        evidence = FairGradEvidence(
            density_norm,
            kappa_norm,
            cosine,
            0.0,
            0.0,
            0.0,
            True,
            2,
        )
        return FairGradResult(tuple(None for _ in density_gradients), evidence)

    common = math.sqrt(1.0 + cosine)
    density_weight = 1.0 / (density_norm * common)
    kappa_weight = 1.0 / (kappa_norm * common)
    combined_values: list[torch.Tensor | None] = []
    for density, kappa in zip(density_gradients, kappa_gradients, strict=True):
        if density is None and kappa is None:
            combined_values.append(None)
            continue
        density_term = torch.zeros_like(kappa) if density is None and kappa is not None else density
        kappa_term = torch.zeros_like(density) if kappa is None and density is not None else kappa
        if density_term is None or kappa_term is None:
            raise RuntimeError("FairGrad internal task-gradient branch did not resolve unused parameter")
        combined_values.append(density_term * density_weight + kappa_term * kappa_weight)
    combined = tuple(combined_values)
    combined_norm = math.sqrt(float(_norm_squared(combined).item()))
    if not math.isfinite(combined_norm):
        raise FloatingPointError("non-finite FairGrad combined gradient")
    evidence = FairGradEvidence(
        density_norm,
        kappa_norm,
        cosine,
        density_weight,
        kappa_weight,
        combined_norm,
        False,
        2,
    )
    return FairGradResult(combined, evidence)


def clip_parameter_groups(
    groups: Sequence[MethodParameterGroup],
    *,
    max_norm: float,
) -> dict[str, GradientGroupEvidence]:
    r"""分别裁剪 shared/density-private/$\kappa$-private 梯度并返回同一次 update 的证据。

    ``grad=None`` 的 shared 参数不进入范数也不被改写，因此 near-opposite 阻塞时 AdamW 不会推进其
    momentum、step 或 weight decay。任一组出现非有限梯度时，PyTorch 在 optimizer step 前抛错。
    """

    if max_norm <= 0.0:
        raise ValueError("gradient group max_norm must be strictly positive")
    evidence: dict[str, GradientGroupEvidence] = {}
    for group in groups:
        active = tuple(parameter for parameter in group.parameters if parameter.grad is not None)
        if not active:
            evidence[group.name] = GradientGroupEvidence(0.0, 0.0, 1.0, False, 0)
            continue
        pre_clip = float(
            torch.nn.utils.clip_grad_norm_(
                active,
                max_norm,
                error_if_nonfinite=True,
            ).detach()
        )
        post_square = sum(
            float(parameter.grad.detach().to(torch.float64).square().sum())
            for parameter in active
            if parameter.grad is not None
        )
        post_clip = math.sqrt(post_square)
        ratio = min(1.0, max_norm / max(pre_clip, torch.finfo(torch.float64).tiny))
        evidence[group.name] = GradientGroupEvidence(
            pre_clip,
            post_clip,
            ratio,
            pre_clip > max_norm,
            len(active),
        )
    return evidence


def _q_per_asset_block(batch: PaddedOnlineGeometryBatch) -> int:
    r"""由 asset-major 样本轴恢复每资产完整 q-block 长度。"""

    if not batch.asset_ids:
        raise ValueError("training minibatch must contain at least one asset/q pair")
    first_asset = batch.asset_ids[0]  # 第一段连续 asset rows 定义 $Q$
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


def _training_minibatch_denominators(method: Any, batch: PaddedOnlineGeometryBatch) -> dict[str, torch.Tensor]:
    r"""在模型 forward 前由 truth masks 形成完整 optimizer-minibatch denominator。

    Density 与 $\kappa$ 都先在每个 $(asset,q)$ 行内归约，再对有效行等权。Joint-sign rewrite 不改变
    mask，因此 denominator 可在 augmentation 前确定，并被全部 64-pair microbatches 共享。
    """

    dtype = batch.q.dtype  # objective numerator 的训练 dtype
    density = batch.field_targets.valid_mask.reshape(batch.q.shape[0], -1).any(dim=-1).sum().to(dtype)
    edge = batch.sensitivity_targets.valid_mask.reshape(batch.q.shape[0], -1).any(dim=-1).sum().to(dtype)
    available = {"density": density, "kappa": edge}  # 两项任务各自的有效 $(asset,q)$ 数
    denominators = {name: available[name] for name in method.config.objectives.enabled()}
    invalid = [name for name, denominator in denominators.items() if float(denominator) <= 0.0]
    if invalid:
        raise ValueError(f"streaming backward minibatch has no valid samples for objectives={invalid}")
    return denominators


def _accumulate_masked(
    totals: dict[str, list[torch.Tensor]],
    name: str,
    values: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    r"""沿 microbatch 累加 detached numerator/count，不改变 objective 的 per-sample reduction。"""

    weight = mask.to(values.dtype)  # bool support -> 与 values 同 dtype 的统计权重
    while weight.ndim < values.ndim:
        weight = weight.unsqueeze(-1)
    weight = weight.expand_as(values)
    current = totals.get(name)
    if current is None:
        current = [values.new_zeros(()), values.new_zeros(())]
        totals[name] = current
    current[0] = current[0] + (values * weight).sum().detach()
    current[1] = current[1] + weight.sum().detach()


def _accumulate_gradients(
    buffers: list[torch.Tensor | None],
    gradients: tuple[torch.Tensor | None, ...],
) -> None:
    r"""累加一个 task/microbatch 参数梯度，同时保留 unused 参数的 ``None``。"""

    if len(buffers) != len(gradients):
        raise RuntimeError("microbatch gradient layout changed within one optimizer update")
    for index, gradient in enumerate(gradients):
        if gradient is None:
            continue
        detached = gradient.detach()  # buffer 不保留 autograd graph
        previous = buffers[index]
        buffers[index] = detached.clone() if previous is None else previous + detached


def _accumulate_prediction_diagnostics(
    method: Any,
    microbatch: PaddedOnlineGeometryBatch,
    prediction: Any,
    totals: dict[str, list[torch.Tensor]],
) -> None:
    r"""累计物理误差、edge coverage 与 1% 中心差分分层证据。

    这些统计全部 detached，不进入 density/$\kappa$ objective、FairGrad 或 checkpoint selection。
    Relative source error 使用 $10^{-6}\,\mathrm{m/rad}$ 分母地板；distance shell 使用该样本实际 sigma；
    feature-margin 档位相对正式 smoothness threshold 定义。
    """

    field_valid = microbatch.field_targets.valid_mask  # `[B,G,N_Q]`
    edge_valid = microbatch.sensitivity_targets.valid_mask  # `[B,E]`
    active_mask = microbatch.sensitivity_targets.active_mask
    if active_mask.ndim == 1:
        active_mask = active_mask.unsqueeze(0).expand_as(edge_valid)
    density_error_sq = (prediction.density - microbatch.field_targets.density).square()
    kappa_error_sq = (prediction.kappa - microbatch.sensitivity_targets.kappa).square()
    _accumulate_masked(totals, "density/prediction_square", prediction.density.square(), field_valid)
    _accumulate_masked(totals, "density/target_square", microbatch.field_targets.density.square(), field_valid)
    _accumulate_masked(totals, "density/error_square", density_error_sq, field_valid)
    _accumulate_masked(totals, "kappa/prediction_square", prediction.kappa.square(), edge_valid)
    _accumulate_masked(totals, "kappa/target_square", microbatch.sensitivity_targets.kappa.square(), edge_valid)
    _accumulate_masked(totals, "kappa/error_square", kappa_error_sq, edge_valid)
    _accumulate_masked(totals, "kappa/active_error_square", kappa_error_sq, edge_valid & active_mask)
    _accumulate_masked(totals, "kappa/zero_error_square", kappa_error_sq, edge_valid & ~active_mask)
    active_sign_mask = edge_valid & active_mask & (microbatch.sensitivity_targets.kappa.abs() >= 1.0e-6)
    active_sign = (torch.sign(prediction.kappa) == torch.sign(microbatch.sensitivity_targets.kappa)).to(
        prediction.kappa.dtype
    )
    _accumulate_masked(totals, "kappa/active_sign_accuracy", active_sign, active_sign_mask)
    density_stat_dtype = prediction.density.dtype
    density_valid = field_valid.sum().to(dtype=density_stat_dtype)
    density_total = field_valid.new_tensor(field_valid.numel(), dtype=density_stat_dtype)
    totals.setdefault("density/valid_ratio", [density_valid.new_zeros(()), density_valid.new_zeros(())])
    totals["density/valid_ratio"][0] = totals["density/valid_ratio"][0] + density_valid
    totals["density/valid_ratio"][1] = totals["density/valid_ratio"][1] + density_total
    kappa_stat_dtype = prediction.kappa.dtype
    kappa_valid = edge_valid.sum().to(dtype=kappa_stat_dtype)
    kappa_total = edge_valid.new_tensor(edge_valid.numel(), dtype=kappa_stat_dtype)
    totals.setdefault("kappa/valid_ratio", [kappa_valid.new_zeros(()), kappa_valid.new_zeros(())])
    totals["kappa/valid_ratio"][0] = totals["kappa/valid_ratio"][0] + kappa_valid
    totals["kappa/valid_ratio"][1] = totals["kappa/valid_ratio"][1] + kappa_total

    # Coverage 按真实 sampled edge 槽统计；padding provenance 固定为 -1，不进入 denominator。
    owner_category = microbatch.sensitivity_targets.owner_category
    query_stratum = microbatch.sensitivity_targets.query_stratum
    sampling_role = microbatch.sensitivity_targets.sampling_role
    fallback_category = microbatch.sensitivity_targets.fallback_category
    if owner_category is not None:
        sampled = owner_category >= 0
        for category in SensitivityOwnerCategory:
            selected = sampled & (owner_category == int(category))
            _accumulate_masked(
                totals,
                f"edge_coverage/owner_category/{category.name.lower()}",
                selected.to(prediction.kappa.dtype),
                sampled,
            )
        if fallback_category is not None:
            _accumulate_masked(
                totals,
                "edge_coverage/fallback_rate",
                (fallback_category >= 0).to(prediction.kappa.dtype),
                sampled,
            )
    if query_stratum is not None:
        sampled = query_stratum >= 0
        for stratum in QueryStratum:
            selected = sampled & (query_stratum == int(stratum))
            _accumulate_masked(
                totals,
                f"edge_coverage/query_stratum/{stratum.name.lower()}",
                selected.to(prediction.kappa.dtype),
                sampled,
            )
    if sampling_role is not None:
        sampled = sampling_role >= 0
        for role in SensitivitySamplingRole:
            selected = sampled & (sampling_role == int(role))
            _accumulate_masked(
                totals,
                f"edge_coverage/sampling_role/{role.name.lower()}",
                selected.to(prediction.kappa.dtype),
                sampled,
            )

    # 稳定 digest 命中的约 1% q rows 才有 central_valid；其余 rows 以全 False mask 表示未审计。
    central_valid = microbatch.sensitivity_targets.central_difference_valid_mask
    central = microbatch.sensitivity_targets.central_difference
    if central_valid is None or central is None:
        return
    source_error = central - microbatch.sensitivity_targets.kappa  # analytic teacher 与 q± 数值导数差，m/rad
    _accumulate_masked(totals, "source_audit/error_square", source_error.square(), central_valid)
    _accumulate_masked(totals, "source_audit/absolute_error", source_error.abs(), central_valid)
    relative_error = source_error.abs() / central.abs().clamp_min(1.0e-6)
    _accumulate_masked(totals, "source_audit/relative_error", relative_error, central_valid)
    sign_agreement = (torch.sign(central) == torch.sign(microbatch.sensitivity_targets.kappa)).to(source_error.dtype)
    _accumulate_masked(totals, "source_audit/sign_agreement", sign_agreement, central_valid)

    if query_stratum is not None:
        for stratum in QueryStratum:
            mask = central_valid & (query_stratum == int(stratum))
            name = stratum.name.lower()
            _accumulate_masked(totals, f"source_audit/stratum/{name}/error_square", source_error.square(), mask)
            _accumulate_masked(totals, f"source_audit/stratum/{name}/sign_agreement", sign_agreement, mask)
    if owner_category is not None:
        for category in SensitivityOwnerCategory:
            mask = central_valid & (owner_category == int(category))
            name = category.name.lower()
            _accumulate_masked(totals, f"source_audit/owner_category/{name}/error_square", source_error.square(), mask)
            _accumulate_masked(totals, f"source_audit/owner_category/{name}/sign_agreement", sign_agreement, mask)

    owner_role = microbatch.field_targets.owner_role
    if owner_role.ndim == 1:
        owner_role = owner_role.unsqueeze(0).expand(microbatch.q.shape[0], -1)
    edge_owner_role = owner_role.gather(1, microbatch.sensitivity_targets.owner_index)
    for role_value, role_name in ((0, "palm"), (1, "joint"), (2, "tip")):
        mask = central_valid & (edge_owner_role == role_value)
        _accumulate_masked(totals, f"source_audit/owner_role/{role_name}/error_square", source_error.square(), mask)
        _accumulate_masked(totals, f"source_audit/owner_role/{role_name}/sign_agreement", sign_agreement, mask)

    # Distance shell 使用每行实际 jittered sigma，并构成互斥分区。
    batch_axis = torch.arange(microbatch.q.shape[0], device=microbatch.q.device).unsqueeze(1)
    edge_distance = microbatch.field_targets.distance[
        batch_axis,
        microbatch.sensitivity_targets.owner_index,
        microbatch.sensitivity_targets.query_index,
    ]
    bandwidths = microbatch.field_targets.bandwidths
    if bandwidths.ndim == 1:
        bandwidths = bandwidths.unsqueeze(0).expand(microbatch.q.shape[0], -1)
    lower: torch.Tensor | None = None
    for sigma_index in range(bandwidths.shape[1]):
        upper = bandwidths[:, sigma_index].unsqueeze(1)
        shell = edge_distance <= upper if lower is None else (edge_distance > lower) & (edge_distance <= upper)
        shell &= central_valid
        name = "le_sigma_0" if lower is None else f"sigma_{sigma_index - 1}_to_{sigma_index}"
        _accumulate_masked(totals, f"source_audit/distance_shell/{name}/error_square", source_error.square(), shell)
        _accumulate_masked(totals, f"source_audit/distance_shell/{name}/sign_agreement", sign_agreement, shell)
        lower = upper
    if lower is not None:
        far = central_valid & (edge_distance > lower)
        _accumulate_masked(totals, "source_audit/distance_shell/gt_sigma_last/error_square", source_error.square(), far)
        _accumulate_masked(totals, "source_audit/distance_shell/gt_sigma_last/sign_agreement", sign_agreement, far)

    # 当前 triangle-feature margin 相对正式 smoothness threshold 分档。
    margin = microbatch.sensitivity_targets.uniqueness_margin
    representation = getattr(method.config, "representation", None)
    target = getattr(representation, "target", None)
    threshold = float(getattr(target, "feature_margin_min_m", 1.0e-6))
    margin_masks = {
        "below_threshold": margin < threshold,
        "one_to_two_thresholds": (margin >= threshold) & (margin < 2.0 * threshold),
        "ge_two_thresholds": margin >= 2.0 * threshold,
    }
    for name, margin_mask in margin_masks.items():
        mask = central_valid & margin_mask
        _accumulate_masked(totals, f"source_audit/feature_margin/{name}/error_square", source_error.square(), mask)
        _accumulate_masked(totals, f"source_audit/feature_margin/{name}/sign_agreement", sign_agreement, mask)

    # Stable 要求原始、q+、q- 三次查询落在同一 union face；其余统一记为 switch。
    plus_face = microbatch.sensitivity_targets.central_difference_plus_face
    minus_face = microbatch.sensitivity_targets.central_difference_minus_face
    if plus_face is not None and minus_face is not None:
        original_face = microbatch.sensitivity_targets.closest_source.bitwise_and(0xFFFFFFFF)
        stable = central_valid & (original_face == plus_face) & (plus_face == minus_face)
        switched = central_valid & ~stable
        _accumulate_masked(totals, "source_audit/face_stable", stable.to(source_error.dtype), central_valid)
        _accumulate_masked(totals, "source_audit/face/stable/error_square", source_error.square(), stable)
        _accumulate_masked(totals, "source_audit/face/switch/error_square", source_error.square(), switched)


def backward_method_update(
    method: Any,
    batch: PaddedOnlineGeometryBatch,
    *,
    forward_step: int,
    microbatch_size: int,
    collect_z_gradients: bool = False,
    rewrite_batch_fn: Any = maybe_rewrite_batch,
) -> MethodUpdate:
    r"""形成完整 512-pair update 的 shared/private task gradients，并写入参数 ``.grad``。

    对 task $j$、microbatch $m$ 的 numerator $N_{j,m}$ 与完整 minibatch denominator $D_j$：
    $$
    \nabla_\theta\mathcal L_j=\sum_m\nabla_\theta\frac{N_{j,m}}{D_j}.
    $$
    每个 64-pair microbatch只前向一次；density 与 $\kappa$ 分别通过 ``autograd.grad`` 提取 shared
    task gradient 与对应 private reader gradient。全部 microbatches 完成后，shared 两任务梯度才进入
    精确 $\alpha=1$ FairGrad；private readers 保持普通 task gradient。
    """

    q_per_asset = _q_per_asset_block(batch)  # 正式值为 8
    if batch.q.shape[0] % microbatch_size != 0:
        raise ValueError("microbatch_size must exactly divide the realized minibatch")
    if microbatch_size % q_per_asset != 0:
        raise ValueError("microbatch_size must preserve complete per-asset q blocks")
    denominators = _training_minibatch_denominators(method, batch)  # 完整 512-pair 固定 $D_j$
    numerators = {name: torch.zeros_like(value) for name, value in denominators.items()}
    observed_denominators = {name: torch.zeros_like(value) for name, value in denominators.items()}
    enabled = method.config.objectives.enabled()
    parameter_groups = method.optimizer_parameter_groups()
    shared_parameters = parameter_groups[0].parameters
    density_parameters = parameter_groups[1].parameters
    kappa_parameters = parameter_groups[2].parameters
    shared_buffers: dict[str, list[torch.Tensor | None]] = {
        name: [None for _parameter in shared_parameters] for name in enabled
    }
    density_private: list[torch.Tensor | None] = [None for _parameter in density_parameters]
    kappa_private: list[torch.Tensor | None] = [None for _parameter in kappa_parameters]
    sample_count = 0
    z_gradient_squares = {
        name: torch.zeros((), device=batch.q.device, dtype=torch.float64) for name in enabled
    }  # latent Gram 对角项，FP64 标量
    z_gradient_dot = torch.zeros((), device=batch.q.device, dtype=torch.float64)  # $\langle\nabla_ZL_\rho,\nabla_ZL_\kappa\rangle$
    diagnostic_totals: dict[str, list[torch.Tensor]] = {}

    rewritten = rewrite_batch_fn(
        batch,
        config=method.config.joint_sign_rewrite,
        step=int(forward_step),
        seed=int(forward_step),
    )
    for microbatch in split_padded_online_geometry_batch(rewritten, microbatch_size=microbatch_size):
        micro_step, prediction = method._forward_with_prediction(
            microbatch,
            step=int(forward_step),
            mode="train",
            apply_augmentation=False,
        )
        _accumulate_prediction_diagnostics(method, microbatch, prediction, diagnostic_totals)
        raw_terms: dict[str, torch.Tensor] = {}
        for term_name, result in micro_step.objectives.items():
            if len(result.components) != 1 or result.components[0].name != term_name:
                raise ValueError("streaming backward requires one same-name additive component per term")
            component = result.components[0]
            raw_terms[term_name] = component.numerator / denominators[term_name]
            numerators[term_name] += component.numerator.detach()
            observed_denominators[term_name] += component.denominator.detach()
        if set(raw_terms) != {"density", "kappa"}:
            raise ValueError("streaming backward microbatch must contain density and kappa")

        if collect_z_gradients:
            z_gradients = _collect_latent_gradients(
                method,
                microbatch,
                prediction,
                denominators,
                normalize=True,
            )
            rho_gradient = z_gradients["density"]
            kappa_gradient = z_gradients["kappa"]
            z_gradient_squares["density"] = z_gradient_squares["density"] + rho_gradient.square().sum().detach()
            z_gradient_squares["kappa"] = z_gradient_squares["kappa"] + kappa_gradient.square().sum().detach()
            z_gradient_dot = z_gradient_dot + (rho_gradient * kappa_gradient).sum().detach()

        density_gradients = torch.autograd.grad(
            raw_terms["density"],
            (*shared_parameters, *density_parameters),
            retain_graph=True,
            allow_unused=True,
        )
        shared_count = len(shared_parameters)
        _accumulate_gradients(shared_buffers["density"], tuple(density_gradients[:shared_count]))
        _accumulate_gradients(density_private, tuple(density_gradients[shared_count:]))
        kappa_gradients = torch.autograd.grad(
            raw_terms["kappa"],
            (*shared_parameters, *kappa_parameters),
            retain_graph=False,
            allow_unused=True,
        )
        _accumulate_gradients(shared_buffers["kappa"], tuple(kappa_gradients[:shared_count]))
        _accumulate_gradients(kappa_private, tuple(kappa_gradients[shared_count:]))
        sample_count += micro_step.sample_count

    for name, expected in denominators.items():
        if not torch.equal(observed_denominators[name], expected):
            raise RuntimeError(
                f"streaming backward denominator mismatch for {name}: "
                f"observed={float(observed_denominators[name])}, expected={float(expected)}"
            )
    fairgrad = combine_fairgrad(
        tuple(shared_buffers["density"]),
        tuple(shared_buffers["kappa"]),
        near_opposition_tolerance=method.config.fairgrad.near_opposition_tolerance,
    )
    for parameter, gradient in zip(shared_parameters, fairgrad.combined, strict=True):
        parameter.grad = gradient
    for parameter, gradient in zip(density_parameters, density_private, strict=True):
        parameter.grad = gradient
    for parameter, gradient in zip(kappa_parameters, kappa_private, strict=True):
        parameter.grad = gradient

    terms = {name: float(numerators[name] / denominators[name]) for name in denominators}
    gradient_evidence: dict[str, float] = {
        "fairgrad/density_norm": fairgrad.evidence.density_norm,
        "fairgrad/kappa_norm": fairgrad.evidence.kappa_norm,
        "fairgrad/cosine": fairgrad.evidence.cosine,
        "fairgrad/density_weight": fairgrad.evidence.density_weight,
        "fairgrad/kappa_weight": fairgrad.evidence.kappa_weight,
        "fairgrad/combined_norm": fairgrad.evidence.combined_norm,
        "fairgrad/shared_conflict_blocked": float(fairgrad.evidence.shared_conflict_blocked),
        "fairgrad/active_tasks": float(fairgrad.evidence.active_tasks),
    }
    if collect_z_gradients:
        epsilon = 1.0e-30
        rho_sq = float(z_gradient_squares["density"].detach())
        kappa_sq = float(z_gradient_squares["kappa"].detach())
        dot = float(z_gradient_dot.detach())
        trace = rho_sq + kappa_sq
        determinant = max(rho_sq * kappa_sq - dot * dot, 0.0)
        discriminant = max(trace * trace - 4.0 * determinant, 0.0)
        largest = 0.5 * (trace + math.sqrt(discriminant))
        smallest = 0.5 * (trace - math.sqrt(discriminant))
        gradient_evidence.update(
            {
                "raw/rho_norm": math.sqrt(max(rho_sq, 0.0)),
                "raw/kappa_norm": math.sqrt(max(kappa_sq, 0.0)),
                "raw/dot": dot,
                "raw/cosine": dot / math.sqrt(max(rho_sq * kappa_sq, epsilon)),
                "raw/gram_determinant": determinant,
                "raw/gram_condition": largest / max(smallest, epsilon),
                "raw/joint_norm": math.sqrt(max(rho_sq + kappa_sq + 2.0 * dot, 0.0)),
            }
        )
    diagnostics = {
        name: float((numerator / denominator.clamp_min(1.0)).detach())
        for name, (numerator, denominator) in diagnostic_totals.items()
    }
    diagnostics["source_audit/wall_time_seconds"] = batch.sensitivity_targets.central_difference_elapsed_seconds
    for name in (
        "density/prediction_square",
        "density/target_square",
        "kappa/prediction_square",
        "kappa/target_square",
    ):
        diagnostics[name.replace("_square", "_rms")] = math.sqrt(max(diagnostics.pop(name), 0.0))
    return MethodUpdate(
        terms=terms,
        sample_count=sample_count,
        denominators={name: float(value) for name, value in denominators.items()},
        gradient_evidence=gradient_evidence,
        diagnostics=diagnostics,
    )


def backward_method_update_units(
    method: Any,
    units: Iterable[PaddedOnlineGeometryBatch],
    *,
    forward_step: int,
    logical_sample_count: int,
    microbatch_size: int,
    collect_z_gradients: bool = False,
    rewrite_batch_fn: Any = maybe_rewrite_batch,
) -> MethodUpdate:
    r"""从有界 64-pair units 闭合一个数学上仍为 512-pair 的 FairGrad update。

    每个任务先对 unit numerator $N_{j,u}$ 求参数梯度，全部 unit 完成后才除以完整 denominator
    $D_j=\sum_uD_{j,u}$：
    $$
    \nabla_\theta\mathcal L_j=\frac{\sum_u\nabla_\theta N_{j,u}}{\sum_uD_{j,u}}.
    $$
    这与完整 batch 归约严格等价，但每次只保留一个 8-assets × 8-q unit 的 activation。
    """

    if logical_sample_count < 1 or microbatch_size < 1:
        raise ValueError("streamed update sample and microbatch sizes must be positive")
    enabled = method.config.objectives.enabled()
    parameter_groups = method.optimizer_parameter_groups()
    shared_parameters = parameter_groups[0].parameters
    density_parameters = parameter_groups[1].parameters
    kappa_parameters = parameter_groups[2].parameters
    shared_buffers: dict[str, list[torch.Tensor | None]] = {
        name: [None for _parameter in shared_parameters] for name in enabled
    }
    density_private: list[torch.Tensor | None] = [None for _parameter in density_parameters]
    kappa_private: list[torch.Tensor | None] = [None for _parameter in kappa_parameters]
    numerators: dict[str, torch.Tensor] = {}
    denominators: dict[str, torch.Tensor] = {}
    diagnostic_totals: dict[str, list[torch.Tensor]] = {}
    z_gradient_squares: dict[str, torch.Tensor] = {}  # 只在 collect_z_gradients 时按 FP64 累计 latent Gram
    z_gradient_dot: torch.Tensor | None = None
    sample_count = 0

    # 每个 unit 独立 realization/forward/backward，参数级 detached buffers 是唯一跨 unit GPU state。
    for unit in units:
        unit_size = int(unit.q.shape[0])
        q_per_asset = _q_per_asset_block(unit)
        if unit_size > microbatch_size or unit_size % microbatch_size != 0:
            raise ValueError("stream unit must be no larger than and exactly divisible by microbatch_size")
        if microbatch_size % q_per_asset != 0:
            raise ValueError("microbatch_size must preserve complete per-asset q blocks")
        rewritten = rewrite_batch_fn(
            unit,
            config=method.config.joint_sign_rewrite,
            step=int(forward_step),
            seed=int(forward_step),
            row_offset=sample_count,
            logical_batch_size=logical_sample_count,
        )
        for microbatch in split_padded_online_geometry_batch(rewritten, microbatch_size=microbatch_size):
            micro_step, prediction = method._forward_with_prediction(
                microbatch,
                step=int(forward_step),
                mode="train",
                apply_augmentation=False,
            )
            _accumulate_prediction_diagnostics(method, microbatch, prediction, diagnostic_totals)
            raw_terms: dict[str, torch.Tensor] = {}
            for term_name, result in micro_step.objectives.items():
                if len(result.components) != 1 or result.components[0].name != term_name:
                    raise ValueError("streaming backward requires one same-name additive component per term")
                component = result.components[0]
                raw_terms[term_name] = component.numerator
                numerators[term_name] = numerators.get(term_name, torch.zeros_like(component.numerator)) + component.numerator.detach()
                denominators[term_name] = denominators.get(
                    term_name, torch.zeros_like(component.denominator)
                ) + component.denominator.detach()
            if set(raw_terms) != {"density", "kappa"}:
                raise ValueError("streaming backward microbatch must contain density and kappa")

            if collect_z_gradients:
                z_gradients = _collect_latent_gradients(
                    method,
                    microbatch,
                    prediction,
                    denominators,
                    normalize=False,
                )
                rho_gradient = z_gradients["density"]
                kappa_gradient = z_gradients["kappa"]
                z_gradient_squares["density"] = z_gradient_squares.get(
                    "density", rho_gradient.new_zeros(())
                ) + rho_gradient.square().sum()
                z_gradient_squares["kappa"] = z_gradient_squares.get(
                    "kappa", kappa_gradient.new_zeros(())
                ) + kappa_gradient.square().sum()
                z_gradient_dot = (
                    rho_gradient.new_zeros(()) if z_gradient_dot is None else z_gradient_dot
                ) + (rho_gradient * kappa_gradient).sum()

            density_gradients = torch.autograd.grad(
                raw_terms["density"],
                (*shared_parameters, *density_parameters),
                retain_graph=True,
                allow_unused=True,
            )
            shared_count = len(shared_parameters)
            _accumulate_gradients(shared_buffers["density"], tuple(density_gradients[:shared_count]))
            _accumulate_gradients(density_private, tuple(density_gradients[shared_count:]))
            kappa_gradients = torch.autograd.grad(
                raw_terms["kappa"],
                (*shared_parameters, *kappa_parameters),
                retain_graph=False,
                allow_unused=True,
            )
            _accumulate_gradients(shared_buffers["kappa"], tuple(kappa_gradients[:shared_count]))
            _accumulate_gradients(kappa_private, tuple(kappa_gradients[shared_count:]))
            sample_count += micro_step.sample_count
            del microbatch, micro_step, prediction, raw_terms, density_gradients, kappa_gradients
        del rewritten, unit  # pinned replay 在恢复下一 unit 前释放当前 CUDA tensors

    if sample_count != logical_sample_count:
        raise RuntimeError(f"streamed update sample count mismatch: observed={sample_count}, expected={logical_sample_count}")
    if set(denominators) != {"density", "kappa"} or any(float(value) <= 0.0 for value in denominators.values()):
        raise ValueError("streamed update requires positive density and kappa denominators")

    # Numerator gradients只有在完整 $D_j$ 已知后归一化；private 与 shared 使用相同 task denominator。
    normalized_shared = {
        name: tuple(None if gradient is None else gradient / denominators[name] for gradient in gradients)
        for name, gradients in shared_buffers.items()
    }
    density_private = [None if gradient is None else gradient / denominators["density"] for gradient in density_private]
    kappa_private = [None if gradient is None else gradient / denominators["kappa"] for gradient in kappa_private]
    fairgrad = combine_fairgrad(
        normalized_shared["density"],
        normalized_shared["kappa"],
        near_opposition_tolerance=method.config.fairgrad.near_opposition_tolerance,
    )
    for parameter, gradient in zip(shared_parameters, fairgrad.combined, strict=True):
        parameter.grad = gradient
    for parameter, gradient in zip(density_parameters, density_private, strict=True):
        parameter.grad = gradient
    for parameter, gradient in zip(kappa_parameters, kappa_private, strict=True):
        parameter.grad = gradient

    terms = {name: float(numerators[name] / denominators[name]) for name in denominators}
    gradient_evidence: dict[str, float] = {
        "fairgrad/density_norm": fairgrad.evidence.density_norm,
        "fairgrad/kappa_norm": fairgrad.evidence.kappa_norm,
        "fairgrad/cosine": fairgrad.evidence.cosine,
        "fairgrad/density_weight": fairgrad.evidence.density_weight,
        "fairgrad/kappa_weight": fairgrad.evidence.kappa_weight,
        "fairgrad/combined_norm": fairgrad.evidence.combined_norm,
        "fairgrad/shared_conflict_blocked": float(fairgrad.evidence.shared_conflict_blocked),
        "fairgrad/active_tasks": float(fairgrad.evidence.active_tasks),
    }
    if collect_z_gradients:
        epsilon = 1.0e-30
        rho_sq = float(z_gradient_squares["density"] / denominators["density"].square())
        kappa_sq = float(z_gradient_squares["kappa"] / denominators["kappa"].square())
        dot = float(z_gradient_dot / (denominators["density"] * denominators["kappa"])) if z_gradient_dot is not None else 0.0
        trace = rho_sq + kappa_sq
        determinant = max(rho_sq * kappa_sq - dot * dot, 0.0)
        discriminant = max(trace * trace - 4.0 * determinant, 0.0)
        largest = 0.5 * (trace + math.sqrt(discriminant))
        smallest = 0.5 * (trace - math.sqrt(discriminant))
        gradient_evidence.update(
            {
                "raw/rho_norm": math.sqrt(max(rho_sq, 0.0)),
                "raw/kappa_norm": math.sqrt(max(kappa_sq, 0.0)),
                "raw/dot": dot,
                "raw/cosine": dot / math.sqrt(max(rho_sq * kappa_sq, epsilon)),
                "raw/gram_determinant": determinant,
                "raw/gram_condition": largest / max(smallest, epsilon),
                "raw/joint_norm": math.sqrt(max(rho_sq + kappa_sq + 2.0 * dot, 0.0)),
            }
        )
    diagnostics = {
        name: float((numerator / denominator.clamp_min(1.0)).detach())
        for name, (numerator, denominator) in diagnostic_totals.items()
    }
    for name in ("density/prediction_square", "density/target_square", "kappa/prediction_square", "kappa/target_square"):
        diagnostics[name.replace("_square", "_rms")] = math.sqrt(max(diagnostics.pop(name), 0.0))
    return MethodUpdate(
        terms=terms,
        sample_count=sample_count,
        denominators={name: float(value) for name, value in denominators.items()},
        gradient_evidence=gradient_evidence,
        diagnostics=diagnostics,
    )


__all__ = [
    "FairGradEvidence",
    "FairGradResult",
    "GradientGroupEvidence",
    "backward_method_update",
    "backward_method_update_units",
    "clip_parameter_groups",
    "combine_fairgrad",
]
