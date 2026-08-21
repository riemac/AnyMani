r"""训练期单 JOINT 坐标符号改写。

被选中的样本只做一次主 forward，不另算 paired latent MSE。物理 surface/query/closest point 不变，
因此 density/distance 不变；对应 JOINT 的 $\kappa/g$ 随坐标翻号。
"""

from __future__ import annotations

import torch

from anymani.distill.objectives.representations.gauge_consistency import rewrite_joint_sign_coordinates
from anymani.distill.representations.targets.field_samples import SensitivityTargetBatch

from .batch import PaddedOnlineGeometryBatch
from .config import JointSignRewriteCfg


def maybe_rewrite_batch(
    batch: PaddedOnlineGeometryBatch,
    *,
    config: JointSignRewriteCfg,
    step: int,
    seed: int,
) -> PaddedOnlineGeometryBatch:
    r"""按 20% 概率、每个选中样本恰好一个有效 JOINT 改写输入与一阶 target。"""

    joint_valid = batch.evidence.joint_valid_mask
    if joint_valid is None:
        joint_valid = torch.ones_like(batch.q, dtype=torch.bool)
    if joint_valid.ndim == 1:
        joint_valid = joint_valid.unsqueeze(0).expand(batch.q.shape[0], -1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(config.seed_offset) + int(step))
    selected = torch.rand(batch.q.shape[0], generator=generator) < config.probability
    joint_sign = torch.ones_like(batch.q)
    for batch_index, is_selected in enumerate(selected.tolist()):
        if not is_selected:
            continue
        valid_indices = torch.where(joint_valid[batch_index])[0]
        if len(valid_indices) == 0:
            raise ValueError("joint-sign rewrite requires at least one valid JOINT")
        cursor = 0
        if batch.q_index is not None:
            cursor = int(batch.q_index[batch_index].item())
        chosen = valid_indices[(int(step) + cursor + batch_index) % len(valid_indices)]
        joint_sign[batch_index, chosen] = -1.0
    rewritten_q, rewritten_evidence, joint_sign = rewrite_joint_sign_coordinates(
        batch.q,
        batch.evidence,
        joint_sign=joint_sign,
    )
    sensitivity = _rewrite_sensitivity_targets(batch.sensitivity_targets, joint_sign)
    return PaddedOnlineGeometryBatch(
        asset_ids=batch.asset_ids,
        q=rewritten_q,
        evidence=rewritten_evidence,
        queries=batch.queries,
        field_targets=batch.field_targets,
        sensitivity_targets=sensitivity,
        q_index=batch.q_index,
    )


def _rewrite_sensitivity_targets(
    targets: SensitivityTargetBatch,
    joint_sign: torch.Tensor,
) -> SensitivityTargetBatch:
    r"""只翻被改写 JOINT 的 $\kappa$ 与 $g$；非该列保持不变。"""

    if targets.joint_index.ndim == 1:
        sign = joint_sign[:, targets.joint_index]
    else:
        sign = torch.gather(joint_sign, 1, targets.joint_index)
    kappa = targets.kappa * sign
    field_sensitivity = targets.field_sensitivity * sign.unsqueeze(-1)
    return SensitivityTargetBatch(
        owner_index=targets.owner_index,
        query_index=targets.query_index,
        joint_index=targets.joint_index,
        ancestor_mask=targets.ancestor_mask,
        active_mask=targets.active_mask,
        closest_point=targets.closest_point,
        closest_source=targets.closest_source,
        uniqueness_margin=targets.uniqueness_margin,
        kappa=kappa,
        field_sensitivity=field_sensitivity,
        valid_mask=targets.valid_mask,
        provenance=targets.provenance,
    )


__all__ = ["maybe_rewrite_batch"]
