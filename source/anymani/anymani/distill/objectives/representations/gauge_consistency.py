r"""JOINT 坐标符号改写的物理输入工具。

本模块只构造同一 physical hand 的等价坐标描述，不定义 latent parity loss。统一
$Z\in\mathbb R^{B\times G\times D}$ 没有被指定为 joint-sign 偶或奇；完整模型只在
``distill.diagnostics`` 中检查 observable density 不变与对应 $\kappa$ 变号。

retained encoder 不读取 joint limits、速度、PD target、previous action 或 action，因此这里仅同步
改写 $q$、$q_{home}$ 与空间旋量。策略和控制侧若消费其余物理量，必须在各自边界执行完整变换：

$$
(q_i,\mathcal S_i,q_i^{min},q_i^{max})
\longleftrightarrow
(-q_i,-\mathcal S_i,-q_i^{max},-q_i^{min}).
$$
"""

from __future__ import annotations

from dataclasses import replace

import torch

from anymani.distill.models.input_adapters.geometry import StaticGeometryEvidence


def rewrite_joint_sign_coordinates(
    q: torch.Tensor,
    evidence: StaticGeometryEvidence,
    *,
    joint_index: int | None = None,
    joint_sign: torch.Tensor | None = None,
) -> tuple[torch.Tensor, StaticGeometryEvidence, torch.Tensor]:
    r"""同步改写物理 $q$、$q_{home}$ 与空间旋量的 JOINT 坐标符号。

    可传单个 ``joint_index`` 形成 partial rewrite，或传 `[N_J]`/`[B,N_J]` 的
    ``joint_sign``。每个元素只能取 $\{-1,+1\}$；padding JOINT 应由调用方保持 $+1$。
    owner surface、anchors 与 topology 描述同一物理对象，不随坐标正方向改变。

    Args:
        q (torch.Tensor): `[B,N_J]` 的物理关节坐标，单位 rad。
        evidence (StaticGeometryEvidence): 含 $q_{home}$ 与空间旋量的 retained 输入。
        joint_index (int | None): 全 batch 共用的单个待翻转 JOINT。
        joint_sign (torch.Tensor | None): `[N_J]` 或 `[B,N_J]` 的显式 $\pm1$ 改写。

    Returns:
        tuple[torch.Tensor, StaticGeometryEvidence, torch.Tensor]: 改写后的 $q$、静态证据与实际 sign。
    """

    joint_count = q.shape[1]  # 当前统一/原生 JOINT 轴长度 $N_J$
    if (joint_index is None) == (joint_sign is None):
        raise ValueError("provide exactly one of joint_index or joint_sign")
    if joint_sign is None:
        if joint_index is None or not 0 <= joint_index < joint_count:
            raise IndexError(f"joint_index must lie in [0,{joint_count})")
        joint_sign = torch.ones(joint_count, device=q.device, dtype=q.dtype)  # 默认保持全部坐标方向
        joint_sign[joint_index] = -1.0  # 只重写指定 JOINT
    else:
        joint_sign = joint_sign.to(device=q.device, dtype=q.dtype)
    if joint_sign.shape not in {(joint_count,), q.shape}:
        raise ValueError("joint_sign must have shape [N_J] or [B,N_J]")
    if not torch.all((joint_sign == 1.0) | (joint_sign == -1.0)):
        raise ValueError("joint_sign entries must be exactly -1 or +1")

    rewritten_q = q * joint_sign  # $q_i'=s_iq_i$
    screw_sign = joint_sign.unsqueeze(-1)  # `[N_J,1]` 或 `[B,N_J,1]`
    if evidence.space_screws.ndim == 2 and joint_sign.ndim == 2:
        raise ValueError("per-sample joint_sign requires batched StaticGeometryEvidence")
    rewritten = replace(
        evidence,
        space_screws=evidence.space_screws * screw_sign,  # $\mathcal S_i'=s_i\mathcal S_i$
        q_home=evidence.q_home * joint_sign,  # $q_{home,i}'=s_iq_{home,i}$
    )
    return rewritten_q, rewritten, joint_sign


def deterministic_partial_joint_sign(
    joint_valid_mask: torch.Tensor,
    *,
    step: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    r"""为每个样本确定性选择一个有效 JOINT 做 partial sign rewrite。

    对第 $b$ 个样本，把有效 joint 列表中的第 $(step+b)\bmod N_J^{valid}$ 项置为
    $-1$，其余坐标保持 $+1$。该规则不消耗全局 RNG；恢复相同 sampling step 后会得到
    完全相同的训练增强。
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


__all__ = ["deterministic_partial_joint_sign", "rewrite_joint_sign_coordinates"]
