r"""TIP-silent 状态下被 joint limit 拒绝的动作均值代价。

该模块只承载一个无状态的纯 Torch 数学 helper。输入的 ``target_normalized`` 与 ``limits_normalized``
均以 ``rad / pi`` 表示，``mean`` 是无量纲 policy action；因此每个 policy step 的物理权限
``1 / 24 rad`` 在归一化目标空间中对应 ``1 / (24*pi)``。若令 $u$ 为当前接受目标、$[l,h]$ 为
合法归一化限位，则理论动作和被拒绝部分为

$$
\hat q = \operatorname{clip}\left(u + \frac{\mu}{24\pi}, l, h\right),
\qquad
r = \mu - 24\pi(\hat q-u).
$$

实现改写为 action-space bounds 的等价式

$$
r = \mu - \operatorname{clip}\left(\mu,
  24\pi(l-u), 24\pi(h-u)\right),
$$

从而不需要在接受动作上做 ``pred - u`` 的浮点相减。cost 只在所有有效 TIP 均无接触、且至少有一个
有效 TIP 的整手 gate 开启时计算；非 TIP 接触、力、物体状态、当前 $q$、history 和 previous action
都不属于本 helper 的信息边界。ghost JOINT 的 ``mean``、target 和 limits 会先由 ``torch.where``
清成零，因此 ghost 上游的 NaN 不会进入前向值或反向梯度。
"""

from __future__ import annotations

import math

import torch

_POLICY_STEP_AUTHORITY_RAD = 1.0 / 24.0  # 每个 policy step 的固定物理权限，单位 rad
_REJECTION_DIAGNOSTIC_THRESHOLD = 1.0e-6  # rejected action 的诊断事件阈值，单位为无量纲 mean action


def _validate_action_regularization_shapes(
    mean: torch.Tensor,
    target_normalized: torch.Tensor,
    limits_normalized: torch.Tensor,
    joint_valid: torch.Tensor,
    tip_contact: torch.Tensor,
    tip_valid: torch.Tensor,
) -> None:
    r"""只按 rank、axis、dtype 和 device 检查 helper 的静态 ABI。

    正式批量视图是 ``[B,J]``、``[B,J,2]``、``[B,F]``；为了让 ``torch.func.vmap`` 能把同一纯函数
    映射到样本轴，也接受相应的 rank-1 单样本视图 ``[J]``、``[J,2]``、``[F]``。这里不读取 tensor
    内容，不使用 ``torch.all(...).item()`` 或 data-dependent assertion；合法 runtime 的 ``u`` 位于 limits
    内、数值 mask 为 0/1 等数据语义由上游 contract 负责。
    """

    rank = mean.ndim  # rank=2 是公开 batch ABI；rank=1 是 vmap 对单个 sample 的函数视图
    if rank not in (1, 2):  # 只允许可静态解释的 JOINT 轴布局
        raise ValueError(f"mean must have shape [B,J] (or rank-1 vmap sample), got {tuple(mean.shape)}")

    expected_joint_shape = tuple(mean.shape)  # [B,J] 或 [J]，作为 target 与 joint mask 的共同轴
    if target_normalized.shape != mean.shape:  # target 必须逐 JOINT 对齐 policy mean
        raise ValueError(
            f"target_normalized must have shape {expected_joint_shape} matching mean, got {tuple(target_normalized.shape)}"
        )
    if tuple(limits_normalized.shape[:-1]) != expected_joint_shape or limits_normalized.shape[-1] != 2:
        raise ValueError(
            "limits_normalized must have shape [B,J,2] (or [J,2] for vmap sample), "
            f"got {tuple(limits_normalized.shape)}"
        )
    if joint_valid.shape != mean.shape:  # joint_valid 的 True/False 轴必须逐元素对应 mean
        raise ValueError(f"joint_valid must have shape {expected_joint_shape}, got {tuple(joint_valid.shape)}")

    expected_tip_prefix = tuple(mean.shape[:-1])  # batch 视图为 [B]，单样本视图为空 tuple
    for name, value in (("tip_contact", tip_contact), ("tip_valid", tip_valid)):
        if value.ndim != rank or tuple(value.shape[:-1]) != expected_tip_prefix:
            raise ValueError(
                f"{name} must have shape [B,F] (or [F] for vmap sample) aligned with mean, got {tuple(value.shape)}"
            )
    if tip_contact.shape != tip_valid.shape:  # contact 与有效性共享同一 TIP 轴 F
        raise ValueError(
            f"tip_contact and tip_valid must share shape, got {tuple(tip_contact.shape)} and {tuple(tip_valid.shape)}"
        )

    floating_inputs = (mean, target_normalized, limits_normalized)  # 三个连续物理量必须可进行浮点微分
    if any(not torch.is_floating_point(value) for value in floating_inputs):  # dtype 判定只读取静态类型元数据
        raise TypeError("mean, target_normalized and limits_normalized must be floating-point tensors")
    if any(value.dtype != mean.dtype for value in floating_inputs[1:]):  # 避免隐式混合精度改变 action bounds
        raise TypeError(
            "mean, target_normalized and limits_normalized must share dtype, "
            f"got {mean.dtype}, {target_normalized.dtype}, {limits_normalized.dtype}"
        )

    all_inputs = (target_normalized, limits_normalized, joint_valid, tip_contact, tip_valid)  # 与 mean 对比 device
    if any(value.device != mean.device for value in all_inputs):  # device 对象比较不触发 tensor-data 同步
        raise ValueError("mean, target_normalized, limits_normalized and masks must share one device")


def tip_silent_rejected_action_cost(
    *,
    mean: torch.Tensor,
    target_normalized: torch.Tensor,
    limits_normalized: torch.Tensor,
    joint_valid: torch.Tensor,
    tip_contact: torch.Tensor,
    tip_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""计算 TIP-silent 状态下每个 sample 的未加权 rejected-action cost 与比例诊断。

    对每个有效 JOINT，``mean`` 是无量纲动作均值 $\mu_j$，``target_normalized`` 是当前接受目标
    $u_j=q_j^{target}/\pi$，``limits_normalized[...,0/1]`` 是 $l_j/h_j=q_{min/max}/\pi$。每个 policy
    step 的固定物理权限为 $1/24\,\mathrm{rad}$，所以 action-space 中的限位距离是
    $24\pi(l_j-u_j)$ 与 $24\pi(h_j-u_j)$。实现计算

    $$
    r_j = \mu_j - \operatorname{clip}\!\left(\mu_j,
      24\pi(l_j-u_j), 24\pi(h_j-u_j)\right),
    \qquad
    c = g\,\frac{\sum_j m_j r_j^2}{\max(\sum_j m_j,1)},
    $$

    其中 $m_j$ 是 ``joint_valid``，$g$ 是整手 gate。gate 仅在至少一个 ``tip_valid`` 且所有有效
    TIP 的 ``tip_contact`` 都为 False 时为 1；invalid TIP 的 contact 不参与判定。返回的比例是

    $$
    f = g\,\frac{\sum_j m_j\,\mathbf{1}(|r_j|>10^{-6})}{\max(\sum_j m_j,1)}.
    $$

    ``1e-6`` 是诊断事件阈值，不是物理权限、训练权重或可微 loss 的平滑项。ghost JOINT 的三个连续
    输入在代数运算前由 ``torch.where`` 清为零；因此即使 ghost 含 NaN，也不会污染值或梯度。函数没有
    参数、buffer、RNG 或其他副作用，可被 ``torch.func.vmap`` 和 ``torch.func.grad`` 组合。

    Args:
        mean: policy action mean，形状 ``[B,J]``（vmap 单样本可为 ``[J]``），无量纲。
        target_normalized: 当前接受目标 $u$，与 ``mean`` 同形状，单位 rad/$\pi$。
        limits_normalized: 每个 JOINT 的 $[l,h]$，形状 ``[B,J,2]``（vmap 单样本可为 ``[J,2]``），单位 rad/$\pi$。
        joint_valid: 有效 JOINT mask，形状 ``[B,J]``（vmap 单样本可为 ``[J]``）；False 表示 ghost。
        tip_contact: TIP 接触 mask，形状 ``[B,F]``（vmap 单样本可为 ``[F]``）；非零值解释为接触。
        tip_valid: 有效 TIP mask，形状 ``[B,F]``（vmap 单样本可为 ``[F]``）；False TIP 不进入 gate。

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(cost, rejected_fraction)``，batch 视图均为 ``[B]``，
            单样本视图均为 scalar；两者都保持 ``mean`` 的 floating dtype。

    Raises:
        ValueError: 输入 rank、轴形状或 device 不符合静态 ABI。
        TypeError: 三个连续输入不是同 dtype 的 floating tensors。
    """

    _validate_action_regularization_shapes(  # 在所有运算前锁定可供 vmap 静态解释的轴合同
        mean,
        target_normalized,
        limits_normalized,
        joint_valid,
        tip_contact,
        tip_valid,
    )

    joint_mask = joint_valid.to(dtype=torch.bool)  # bool$[B,J]$，numeric 0/1 transport 也规约到同一语义
    tip_mask = tip_valid.to(dtype=torch.bool)  # bool$[B,F]$，只有有效 TIP 能触发 hand-wide gate
    contact_mask = tip_contact.to(dtype=torch.bool)  # bool$[B,F]$，非零接触表示为 True

    # ghost 的连续输入必须在代数前清零；这样 NaN ghost 不会进入 clamp 的 min/max 或其 autograd 路径。
    clean_mean = torch.where(joint_mask, mean, torch.zeros_like(mean))  # $\mu_j\leftarrow0$ for ghost JOINT
    clean_target = torch.where(joint_mask, target_normalized, torch.zeros_like(target_normalized))  # $u_j\leftarrow0$
    clean_limits = torch.where(
        joint_mask.unsqueeze(-1), limits_normalized, torch.zeros_like(limits_normalized)
    )  # $[l_j,h_j]\leftarrow[0,0]$ for ghost JOINT

    # 只保留 valid TIP 的 contact；invalid TIP 即使上游带有接触 bit，也不能关闭整手 gate。
    effective_contact = torch.where(tip_mask, contact_mask, torch.zeros_like(contact_mask))  # ghost TIP contact→False
    silent_gate = tip_mask.any(dim=-1) & ~effective_contact.any(dim=-1)  # $g=1$ iff valid TIP exists and all are silent

    # 把 target-space limit 距离一次换成 action-space bounds，精确实现 $24\pi(l-u),24\pi(h-u)$。
    authority_scale = clean_mean.new_tensor(1.0 / _POLICY_STEP_AUTHORITY_RAD)  # $24$，与 mean 保持 dtype/device
    lower_action = authority_scale * math.pi * (clean_limits[..., 0] - clean_target)  # lower bound $24\pi(l-u)$
    upper_action = authority_scale * math.pi * (clean_limits[..., 1] - clean_target)  # upper bound $24\pi(h-u)$
    accepted_action = torch.clamp(clean_mean, min=lower_action, max=upper_action)  # action-space accepted portion
    rejected = clean_mean - accepted_action  # $r=\mu-\operatorname{clip}(\mu,24\pi(l-u),24\pi(h-u))$

    # 仅按有效 JOINT 计数；全 ghost sample 的分母用 1 保持有限且不引入额外数据依赖。
    joint_weight = joint_mask.to(dtype=mean.dtype)  # $m_j\in\{0,1\}$，与 cost 共 dtype
    valid_joint_count = joint_weight.sum(dim=-1).clamp_min(1.0)  # $\max(\sum_jm_j,1)$，形状 [B] 或 scalar
    gate_weight = silent_gate.to(dtype=mean.dtype)  # $g$，按 sample 广播到 cost/fraction

    # 未加权 cost 是有效 JOINT rejected action 平方的均值；rejected 本身仍保留对 mean 的梯度。
    cost = gate_weight * (joint_weight * rejected.square()).sum(dim=-1) / valid_joint_count  # $c[B]$
    rejection_event = rejected.abs() > _REJECTION_DIAGNOSTIC_THRESHOLD  # $|r_j|>10^{-6}$ 的诊断事件
    rejected_fraction = (
        gate_weight * (joint_weight * rejection_event.to(dtype=mean.dtype)).sum(dim=-1) / valid_joint_count
    )  # $f[B]$
    return cost, rejected_fraction


__all__ = ["tip_silent_rejected_action_cost"]
