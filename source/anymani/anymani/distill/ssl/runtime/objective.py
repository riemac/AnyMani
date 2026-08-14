r"""Geometry SSL 单 microbatch 前向与 optimizer-step 全局归一化。

物理 loss 公式由 ``objectives.representations`` 定义；本模块只把 runtime batch 与模型装配起来，
执行真实 joint-sign coordinate rewrite，并在 gradient accumulation 时按整个 optimizer step 的有效
标量总数归一化。这样尾资产组、不同 uniqueness mask 数和 padding 比例不会改变监督权重。
"""

from __future__ import annotations

import torch  # q-autograd、Sobolev 二阶图与 denominator tensors

from anymani.distill.models.geometry_ssl import GeometrySSLForward, GeometrySSLModel
from anymani.distill.objectives.representations.field_reconstruction import (
    GeometrySSLObjective,
    GeometrySSLTerms,
    GeometrySSLWeights,
)
from anymani.distill.objectives.representations.gauge_consistency import (
    deterministic_partial_joint_sign,
    joint_sign_paired_loss_additive_components,
    joint_sign_paired_loss_components,
    rewrite_joint_sign_coordinates,
)
from anymani.distill.ssl.dataset import PaddedOnlineGeometryBatch


def forward_objective(
    model: GeometrySSLModel,
    objective: GeometrySSLObjective,
    batch: PaddedOnlineGeometryBatch,
    pair_step: int = 0,
) -> tuple[GeometrySSLForward, GeometrySSLTerms]:
    r"""在物理 q 上保留 Sobolev 图，并用真实 sign rewrite 计算 paired parity。

    sampler 与 Warp teacher 停止梯度；新建 leaf q 后通过 encoder/decoder 对物理 rad 求导：
    $\hat g^{auto}=\partial\hat\rho/\partial q_i$。每个样本另选择一个有效 JOINT，同步改写
    $(q_i,q_{home,i},\mathcal S_i)$ 后再次运行共享 encoder，约束 $Z^{(0)}$ 偶与对应
    $z_i^{(1)}$ 奇。
    """

    q = batch.q.detach().requires_grad_(True)  # teacher stop-gradient；模型仍对物理 rad q 求导
    prediction = model(
        q,
        batch.evidence,
        batch.queries.query_points_h,
        batch.field_targets.bandwidths,  # `[B,N_σ]` actual sigma，q 导数中保持固定
        owner_index=batch.sensitivity_targets.owner_index,
        query_index=batch.sensitivity_targets.query_index,
        joint_index=batch.sensitivity_targets.joint_index,
    )
    joint_valid = batch.evidence.joint_valid_mask  # `[B,N_J]`，padding JOINT 不参与 rewrite
    if joint_valid is None:
        joint_valid = torch.ones_like(q, dtype=torch.bool)  # 非 padding integration contract
    if joint_valid.ndim == 1:
        joint_valid = joint_valid.unsqueeze(0).expand(q.shape[0], -1)
    joint_sign = deterministic_partial_joint_sign(
        joint_valid,
        step=pair_step,
        dtype=q.dtype,
    )  # 每个样本恰改写一个有效 JOINT
    rewritten_q, rewritten_evidence, joint_sign = rewrite_joint_sign_coordinates(
        q,
        batch.evidence,
        joint_sign=joint_sign,
    )  # 同一 physical hand 的另一套关节坐标记号
    paired_latents = model.encoder(rewritten_q, rewritten_evidence)  # 不重复 disposable decoders
    pair_additive_components = joint_sign_paired_loss_additive_components(
        prediction.latents,
        paired_latents,
        joint_sign=joint_sign,
        entity_valid_mask=batch.evidence.entity_valid_mask,
        joint_valid_mask=joint_valid,
    )  # `(N_0,D_0,N_1,D_1)`，服务不等大小 microbatch accumulation
    pair_loss, pair_numerator, pair_denominator = joint_sign_paired_loss_components(
        prediction.latents,
        paired_latents,
        joint_sign=joint_sign,
        entity_valid_mask=batch.evidence.entity_valid_mask,
        joint_valid_mask=joint_valid,
    )  # 保持 `MSE(Z0 parity)+MSE(z1 parity)` 原公式
    terms = objective(
        q=q,
        density_prediction=prediction.density,
        kappa_prediction=prediction.kappa,
        field_targets=batch.field_targets,
        sensitivity_targets=batch.sensitivity_targets,
        paired_loss=pair_loss,
        paired_components=(pair_numerator, pair_denominator),
        paired_additive_components=pair_additive_components,
    )
    return prediction, terms


def accumulated_objective(
    terms: GeometrySSLTerms,
    denominator_totals: tuple[torch.Tensor, ...],
    paired_denominator_totals: tuple[torch.Tensor, torch.Tensor],
    weights: GeometrySSLWeights,
) -> torch.Tensor:
    r"""把一个 microbatch 的 numerators 按整个 optimizer step 的 denominator 缩放。

    前五项使用 $\sum_bN_{t,b}/\sum_bD_{t,b}$；paired 项保持
    $\sum_bN_{0,b}/\sum_bD_{0,b}+\sum_bN_{1,b}/\sum_bD_{1,b}$。调用方逐 microbatch
    backward，显存仍只持有一份 Sobolev 图。
    """

    if len(terms.numerators) != 6 or len(denominator_totals) != 5:
        raise ValueError("accumulated geometry SSL objective requires six terms and five field totals")
    if len(terms.paired_additive_numerators) != 2 or len(terms.paired_additive_denominators) != 2:
        raise ValueError("accumulated geometry SSL objective requires paired additive components")
    field_weights = (weights.density, weights.kappa, weights.derived_field, weights.sobolev, weights.chain)
    total = sum(
        weight * numerator / denominator_total
        for weight, numerator, denominator_total in zip(
            field_weights,
            terms.numerators[:5],
            denominator_totals,
        )
    )
    paired = sum(
        numerator / denominator_total
        for numerator, denominator_total in zip(
            terms.paired_additive_numerators,
            paired_denominator_totals,
        )
    )
    return torch.as_tensor(total, device=terms.total.device, dtype=terms.total.dtype) + weights.paired * paired


def objective_denominators_from_batch(
    batch: PaddedOnlineGeometryBatch,
    model: GeometrySSLModel,
) -> tuple[tuple[float, ...], tuple[float, float]]:
    r"""由 masks 与 latent widths 预计算五项 field 及 paired 两支有效标量数。"""

    bandwidth_count = batch.field_targets.bandwidths.shape[-1]  # $N_\sigma$ 数据采样轴
    field_count = float(batch.field_targets.valid_mask.sum()) * bandwidth_count
    edge_count = float(batch.sensitivity_targets.valid_mask.sum())
    edge_band_count = edge_count * bandwidth_count
    entity_valid = batch.evidence.entity_valid_mask
    joint_valid = batch.evidence.joint_valid_mask
    if entity_valid is None or joint_valid is None:
        raise ValueError("accumulation denominator preflight requires padded entity/joint masks")
    zero_count = float(entity_valid.sum()) * model.config.encoder.zero_order_width
    first_count = float(joint_valid.sum()) * model.config.encoder.first_order_width
    values = (field_count, edge_count, edge_band_count, edge_band_count, edge_band_count)
    if min(*values, zero_count, first_count) <= 0.0:
        raise ValueError("accumulation denominator preflight found an empty objective term")
    return values, (zero_count, first_count)


__all__ = ["accumulated_objective", "forward_objective", "objective_denominators_from_batch"]
