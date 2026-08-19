r"""Paired URDF reparameterization 的 gauge-consistency objective contract。

Field reconstruction 定义 latent 的主物理语义；gauge loss 只约束同一 physical hand、同一
$q$ 在两种 local-frame parameterization A/B 下保持一致：

$$
\mathcal L
=
\mathcal L_{field}
+
\lambda_g\mathcal L_{gauge},
\qquad
\lambda_g\ge0.
$$

候选包括 latent consistency $\|z^A-z^B\|_2^2$ 与 query-space field consistency。前者直接服务 policy，但可能过度限制
等价内部参数化；后者只要求物理输出一致，约束较弱。二者都必须以同步变换 mesh、joint
axis/sign、zero offset、limits、drive/target/action labels 的合法 paired asset 为证据，不能
只随机旋转一个 feature vector 就声称完成 URDF gauge augmentation。

真实改变 $q$、tip 装配方向、尺度或 wedge orientation 时 representation 必须变化；gauge
loss 不能误把 physical equivariance 压成 rotation invariance。
"""

from __future__ import annotations

from dataclasses import replace

import torch

from anymani.distill.models.input_adapters.geometry import GeometryLatents, StaticGeometryEvidence


def rewrite_joint_sign_coordinates(
    q: torch.Tensor,
    evidence: StaticGeometryEvidence,
    *,
    joint_index: int | None = None,
    joint_sign: torch.Tensor | None = None,
) -> tuple[torch.Tensor, StaticGeometryEvidence, torch.Tensor]:
    r"""同步改写物理 $q$、$q_{home}$ 与空间旋量的 JOINT 坐标符号。

    可传单个 `joint_index` 形成 partial rewrite，或传 `[N_J]`/`[B,N_J]` 的
    `joint_sign`。每个元素只能取 $\{-1,+1\}$；padding JOINT 应由调用方保持 $+1$。
    本函数不改 owner surface、anchors、topology 或 limits，因为它们不属于 retained
    encoder 输入；策略/控制侧若保存 limits/action label，必须在其边界同步改写。

    Returns:
        tuple: 改写后的 q、静态证据，以及实际使用的 sign tensor。
    """

    joint_count = q.shape[1]
    if (joint_index is None) == (joint_sign is None):
        raise ValueError("provide exactly one of joint_index or joint_sign")
    if joint_sign is None:
        if joint_index is None or not 0 <= joint_index < joint_count:
            raise IndexError(f"joint_index must lie in [0,{joint_count})")
        joint_sign = torch.ones(joint_count, device=q.device, dtype=q.dtype)
        joint_sign[joint_index] = -1.0
    else:
        joint_sign = joint_sign.to(device=q.device, dtype=q.dtype)
    if joint_sign.shape not in {(joint_count,), q.shape}:
        raise ValueError("joint_sign must have shape [N_J] or [B,N_J]")
    if not torch.all((joint_sign == 1.0) | (joint_sign == -1.0)):
        raise ValueError("joint_sign entries must be exactly -1 or +1")

    rewritten_q = q * joint_sign  # $q_i' = s_i q_i$
    screw_sign = joint_sign.unsqueeze(-1)  # `[N_J,1]` 或 `[B,N_J,1]`
    q_home_sign = joint_sign
    if evidence.space_screws.ndim == 2 and joint_sign.ndim == 2:
        raise ValueError("per-sample joint_sign requires batched StaticGeometryEvidence")
    rewritten = replace(
        evidence,
        space_screws=evidence.space_screws * screw_sign,
        q_home=evidence.q_home * q_home_sign,
    )
    return rewritten_q, rewritten, joint_sign


def joint_sign_paired_loss(
    reference: GeometryLatents,
    rewritten: GeometryLatents,
    *,
    joint_sign: torch.Tensor,
    entity_valid_mask: torch.Tensor | None = None,
    joint_valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""约束 paired rewrite 下 $Z^{(0)}$ 偶、逐 JOINT $z_i^{(1)}$ 奇。

    $$
    \mathcal L_{pair}
    =\operatorname{MSE}(Z^{(0)}_{s},Z^{(0)})
    +\operatorname{MSE}(z^{(1)}_{s},s\odot z^{(1)}).
    $$

    `joint_sign` 可以由 batch 共享 `[N_J]`，也可以逐样本为 `[B,N_J]`。padding
    mask 只改变归约分母，不改变目标变换律。
    """

    loss, _numerator, _denominator = joint_sign_paired_loss_components(
        reference,
        rewritten,
        joint_sign=joint_sign,
        entity_valid_mask=entity_valid_mask,
        joint_valid_mask=joint_valid_mask,
    )
    return loss


def joint_sign_paired_loss_components(
    reference: GeometryLatents,
    rewritten: GeometryLatents,
    *,
    joint_sign: torch.Tensor,
    entity_valid_mask: torch.Tensor | None = None,
    joint_valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""返回 paired parity 的 loss、平方误差 numerator 与有效标量 denominator。"""

    zero_numerator, zero_denominator, first_numerator, first_denominator = (
        joint_sign_paired_loss_additive_components(
            reference,
            rewritten,
            joint_sign=joint_sign,
            entity_valid_mask=entity_valid_mask,
            joint_valid_mask=joint_valid_mask,
        )
    )
    numerator = (
        zero_numerator * first_denominator + first_numerator * zero_denominator
    )  # 等价分式分子，保持 $N_0/D_0+N_1/D_1$
    denominator = zero_denominator * first_denominator  # 等价分式分母 $D_0D_1$
    return numerator / denominator, numerator, denominator


def joint_sign_paired_loss_additive_components(
    reference: GeometryLatents,
    rewritten: GeometryLatents,
    *,
    joint_sign: torch.Tensor,
    entity_valid_mask: torch.Tensor | None = None,
    joint_valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""分别返回零阶偶性与一阶奇性的 numerator/denominator。

    paired loss 是两个 MSE 的和，而不是一个共享 denominator 的 MSE：

    $$
    \mathcal L_{pair}=N_0/D_0+N_1/D_1.
    $$

    分开返回四个标量，使 gradient accumulation 能先跨不等大小 minibatches 累加
    $(N_0,D_0)$ 与 $(N_1,D_1)$，再保持原公式归一化；尾资产组不会因合并分母
    $D_0D_1$ 而获得与 $B^2$ 成正比的错误权重。
    """

    if reference.zero_order.shape != rewritten.zero_order.shape:
        raise ValueError("paired zero-order latents must share shape")
    if reference.first_order.shape != rewritten.first_order.shape:
        raise ValueError("paired first-order latents must share shape")
    sign = joint_sign.to(reference.first_order).reshape(
        (1, -1, 1) if joint_sign.ndim == 1 else (*joint_sign.shape, 1)
    )
    zero_error = rewritten.zero_order - reference.zero_order
    first_error = rewritten.first_order - sign * reference.first_order
    zero_mask = _latent_mask(entity_valid_mask, zero_error)
    first_mask = _latent_mask(joint_valid_mask, first_error)
    zero_numerator, zero_denominator = _masked_latent_mse_components(zero_error, zero_mask)
    first_numerator, first_denominator = _masked_latent_mse_components(first_error, first_mask)
    return zero_numerator, zero_denominator, first_numerator, first_denominator


def deterministic_partial_joint_sign(
    joint_valid_mask: torch.Tensor,
    *,
    step: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    r"""为每个样本确定性选择一个有效 JOINT 做 partial sign rewrite。

    对第 $b$ 个样本，把有效 joint 列表中的第 $(step+b)\bmod N_J^{valid}$ 项置为
    $-1$，其余坐标保持 $+1$。该规则不消耗全局 RNG；恢复相同采样 step 后会得到
    完全相同的 paired augmentation。
    """

    if joint_valid_mask.ndim == 1:
        joint_valid_mask = joint_valid_mask.unsqueeze(0)
    if joint_valid_mask.ndim != 2 or joint_valid_mask.dtype != torch.bool:
        raise ValueError("joint_valid_mask must have bool shape [N_J] or [B,N_J]")
    signs = torch.ones(joint_valid_mask.shape, device=joint_valid_mask.device, dtype=dtype)
    for batch_index in range(joint_valid_mask.shape[0]):
        valid_indices = torch.where(joint_valid_mask[batch_index])[0]
        if len(valid_indices) == 0:
            raise ValueError("partial joint-sign rewrite requires at least one valid JOINT per sample")
        selected = valid_indices[(int(step) + batch_index) % len(valid_indices)]
        signs[batch_index, selected] = -1.0
    return signs


def _latent_mask(mask: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor:
    r"""把可选 `[B,N]`/`[N]` validity mask 广播到 latent 尾轴。"""

    if mask is None:
        return torch.ones_like(value, dtype=torch.bool)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0).expand(value.shape[0], -1)
    if mask.shape != value.shape[:2]:
        raise ValueError("paired latent validity mask must align with [B,N]")
    return mask.unsqueeze(-1).expand_as(value)


def _masked_latent_mse(error: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    r"""按有效 latent 标量数归一化 paired 平方误差。"""

    numerator, denominator = _masked_latent_mse_components(error, mask)
    return numerator / denominator


def _masked_latent_mse_components(error: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""返回 latent parity 平方误差的 numerator 与有效标量 denominator。"""

    weight = mask.to(error.dtype)
    denominator = weight.sum()
    if int(denominator.detach()) == 0:
        raise ValueError("paired latent loss received no valid coordinates")
    return torch.sum(weight * error.square()), denominator


__all__ = [
    "deterministic_partial_joint_sign",
    "joint_sign_paired_loss",
    "joint_sign_paired_loss_additive_components",
    "joint_sign_paired_loss_components",
    "rewrite_joint_sign_coordinates",
]
