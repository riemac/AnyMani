r"""Palm-up DexCube rotation任务的纯Torch$SO(3)$、reward与评估数学。

本模块固定quaternion顺序$(w,x,y,z)$，不导入Isaac Lab。相邻policy frame的旋转远小于$\pi$，因此signed
progress使用principal quaternion logarithm；输入$q$或$-q$经relative quaternion canonicalization得到同一
结果。RewardManager的时间积分位于Isaac层，本模块只返回rate或无量纲shape reward。
"""

from __future__ import annotations

import math

import torch


def active_reference_sum(
    values: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    reference_dof: int = 16,
) -> torch.Tensor:
    r"""把逐关节和式折算到固定参考DoF，兼顾N000数值锚点与异构公平。

    对第$i$只手的$n_i$个有效关节，返回：

    $$
    \widetilde P_i=\frac{n_{ref}}{n_i}\sum_{j\in\mathcal J_i}p_{i,j},
    \qquad n_{ref}=16.
    $$

    因而16-DoF手严格恢复N000的原始sum；少DoF手按有效关节均值再投影到同一个16-DoF参考尺度。
    Canonical ghost只属于padding，即使其storage中出现任意有限值，也不进入分子或分母。

    Args:
        values (torch.Tensor): 逐关节非负penalty contributions，末维为canonical joint轴。
        active_mask (torch.Tensor): 与``values``同shape的bool有效关节mask。
        reference_dof (int): 参考手有效DoF；当前N000锚点固定16。

    Returns:
        torch.Tensor: 去除joint轴后的reference-DoF equivalent penalty。
    """

    if values.shape != active_mask.shape or active_mask.dtype != torch.bool:
        raise ValueError("reference-DoF reward reduction requires matching values and bool active_mask")
    if reference_dof < 1:
        raise ValueError("reference_dof must be positive")
    weights = active_mask.to(dtype=values.dtype)  # $m_{i,j}\in\{0,1\}$，shape与values一致
    active_count = weights.sum(dim=-1)  # $n_i$，每只手真实有效DoF
    if bool((active_count < 1).any().item()):
        raise ValueError("reference-DoF reward reduction requires at least one active joint per environment")
    active_sum = (values * weights).sum(dim=-1)  # $\sum_{j\in\mathcal J_i}p_{i,j}$
    return active_sum * (float(reference_dof) / active_count)  # $n_{ref}/n_i$折算后的N000尺度


def active_reference_l2(
    values: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    reference_dof: int = 16,
) -> torch.Tensor:
    r"""返回固定参考DoF下的masked $L_2$幅度。

    $$
    \widetilde L_i
    =\sqrt{\frac{n_{ref}}{n_i}\sum_{j\in\mathcal J_i}x_{i,j}^2}.
    $$

    该定义在$n_i=16$时逐值等于N000的$\|x_i\|_2$，在不同DoF间比较时保持相同的典型
    per-joint偏移对应相同惩罚量级。

    Args:
        values (torch.Tensor): 可带符号逐关节物理量，末维为canonical joint轴。
        active_mask (torch.Tensor): bool有效关节mask，与``values``同shape。
        reference_dof (int): 参考有效DoF，当前固定16。

    Returns:
        torch.Tensor: 去除joint轴后的reference-DoF equivalent $L_2$幅度。
    """

    return torch.sqrt(active_reference_sum(values.square(), active_mask, reference_dof=reference_dof))


def normalize_quaternion_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    r"""归一化最后一维为4的finite quaternion batch。"""

    if quaternion.shape[-1] != 4 or not bool(torch.isfinite(quaternion).all().item()):
        raise ValueError("quaternion must be finite with final dimension four")
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if bool((norm < 1.0e-12).any().item()):
        raise ValueError("quaternion norm must be non-zero")
    return quaternion / norm


def quaternion_multiply_wxyz(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    r"""计算Hamilton product$q=q_l\otimes q_r$，支持broadcast batch axes。"""

    if left.shape[-1] != 4 or right.shape[-1] != 4:
        raise ValueError("quaternion operands must end in dimension four")
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def quaternion_inverse_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    r"""返回unit quaternion inverse$q^{-1}=(w,-x,-y,-z)$。"""

    normalized = normalize_quaternion_wxyz(quaternion)
    inverse = normalized.clone()
    inverse[..., 1:] *= -1.0
    return inverse


def quaternion_to_matrix_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    r"""把unit quaternion转换为rotation matrix$R\in SO(3)$。"""

    q = normalize_quaternion_wxyz(quaternion)
    w, x, y, z = q.unbind(dim=-1)
    two = 2.0
    return torch.stack(
        (
            1.0 - two * (y * y + z * z),
            two * (x * y - z * w),
            two * (x * z + y * w),
            two * (x * y + z * w),
            1.0 - two * (x * x + z * z),
            two * (y * z - x * w),
            two * (x * z - y * w),
            two * (y * z + x * w),
            1.0 - two * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def quaternion_from_angle_axis_wxyz(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    r"""由angle$[B]$与非零axis$[B,3]$构造unit quaternion。"""

    if angle.shape != axis.shape[:-1] or axis.shape[-1] != 3:
        raise ValueError("angle and axis must have shapes [...], [...,3]")
    norm = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
    if bool((norm < 1.0e-12).any().item()) or not bool(torch.isfinite(angle).all().item()):
        raise ValueError("axis must be finite/non-zero and angle finite")
    normalized_axis = axis / norm
    half = 0.5 * angle
    return torch.cat((torch.cos(half).unsqueeze(-1), normalized_axis * torch.sin(half).unsqueeze(-1)), dim=-1)


def quaternion_apply_wxyz(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    r"""应用rotation$v'=R(q)v$。"""

    if vector.shape[-1] != 3 or quaternion.shape[:-1] != vector.shape[:-1]:
        raise ValueError("quaternion/vector batch shapes must match with final dimensions 4/3")
    return torch.einsum("...ij,...j->...i", quaternion_to_matrix_wxyz(quaternion), vector)


def axis_angle_from_quaternion_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    r"""返回principal rotation vector$\log(R)^\vee$，angle范围$[0,\pi]$。

    Quaternion先按$w\ge0$选择双覆盖代表。对于接近identity的输入，极限为$2v$，避免除以零。
    """

    q = normalize_quaternion_wxyz(quaternion)
    q = torch.where((q[..., :1] < 0.0).expand_as(q), -q, q)  # canonical representative$q\sim-q$
    scalar = torch.clamp(q[..., 0], min=0.0, max=1.0)
    vector = q[..., 1:]
    vector_norm = torch.linalg.vector_norm(vector, dim=-1)
    angle = 2.0 * torch.atan2(vector_norm, scalar)
    scale = torch.where(vector_norm > 1.0e-8, angle / vector_norm, torch.full_like(vector_norm, 2.0))
    return vector * scale.unsqueeze(-1)


def projected_space_rotation_delta(
    previous_quat_w: torch.Tensor,
    current_quat_w: torch.Tensor,
    axis_w: torch.Tensor,
) -> torch.Tensor:
    r"""计算相邻姿态绕world-frame有向space axis的signed增量。

    $$
    \Delta R_t=R_tR_{t-1}^{\mathsf T},\qquad
    \Delta\psi_t=\hat k^{w\mathsf T}\log(\Delta R_t)^\vee.
    $$
    """

    if previous_quat_w.shape != current_quat_w.shape or previous_quat_w.shape[:-1] != axis_w.shape[:-1]:
        raise ValueError("previous/current quaternion and axis batches must align")
    delta_quaternion = quaternion_multiply_wxyz(
        normalize_quaternion_wxyz(current_quat_w), quaternion_inverse_wxyz(previous_quat_w)
    )
    delta_rotation_vector = axis_angle_from_quaternion_wxyz(delta_quaternion)
    axis_norm = torch.linalg.vector_norm(axis_w, dim=-1, keepdim=True)
    if bool((axis_norm < 1.0e-12).any().item()):
        raise ValueError("progress axis must be non-zero")
    return torch.sum(delta_rotation_vector * (axis_w / axis_norm), dim=-1)


def hand_axis_to_world(
    axis_h: torch.Tensor,
    root_quat_wxyz: torch.Tensor,
    semantic_R_ha: torch.Tensor,
) -> torch.Tensor:
    r"""按$v^a=R_{ha}^{\mathsf T}v^h$与$v^w=R_{wa}v^a$转换hand axis。"""

    if semantic_R_ha.shape != (3, 3):
        raise ValueError("semantic_R_ha must have shape [3,3]")
    axis_a = axis_h @ semantic_R_ha  # row-vector form$v_a^T=v_h^TR_{ha}$
    return quaternion_apply_wxyz(root_quat_wxyz, axis_a)


def moving_goal_quaternion(
    current_quat_w: torch.Tensor,
    axis_w: torch.Tensor,
    *,
    subgoal_angle_rad: float = math.pi / 6.0,
) -> torch.Tensor:
    r"""从当前object pose左乘30°space rotation生成下一moving goal。

    $$
    q_{g,k+1}=q_\Delta(\hat k^w,\pi/6)\otimes q_{o,t_k}.
    $$
    """

    if not math.isfinite(subgoal_angle_rad) or subgoal_angle_rad <= 0.0:
        raise ValueError("subgoal angle must be finite and positive")
    angle = torch.full(axis_w.shape[:-1], subgoal_angle_rad, dtype=axis_w.dtype, device=axis_w.device)
    delta = quaternion_from_angle_axis_wxyz(angle, axis_w)
    return normalize_quaternion_wxyz(quaternion_multiply_wxyz(delta, current_quat_w))


def orientation_keypoint_distance(
    current_quat_w: torch.Tensor,
    goal_quat_w: torch.Tensor,
    *,
    radius_m: float = 0.05,
) -> torch.Tensor:
    r"""计算中心对齐的六轴orientation-only keypoint平均距离，单位m。"""

    if not math.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError("keypoint radius must be finite and positive")
    keypoints = torch.tensor(
        (
            (radius_m, 0.0, 0.0),
            (-radius_m, 0.0, 0.0),
            (0.0, radius_m, 0.0),
            (0.0, -radius_m, 0.0),
            (0.0, 0.0, radius_m),
            (0.0, 0.0, -radius_m),
        ),
        dtype=current_quat_w.dtype,
        device=current_quat_w.device,
    )
    current_points = torch.einsum("bij,kj->bki", quaternion_to_matrix_wxyz(current_quat_w), keypoints)
    goal_points = torch.einsum("bij,kj->bki", quaternion_to_matrix_wxyz(goal_quat_w), keypoints)
    return torch.linalg.vector_norm(current_points - goal_points, dim=-1).mean(dim=-1)


def goal_errors_and_success(
    object_pos_w: torch.Tensor,
    object_quat_w: torch.Tensor,
    position_anchor_w: torch.Tensor,
    goal_quat_w: torch.Tensor,
    *,
    keypoint_radius_m: float = 0.05,
    orientation_threshold_m: float = 0.005,
    position_threshold_m: float = 0.025,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""返回orientation/position/alignment与严格双门success。"""

    orientation_error = orientation_keypoint_distance(
        object_quat_w, goal_quat_w, radius_m=keypoint_radius_m
    )
    position_error = torch.linalg.vector_norm(object_pos_w - position_anchor_w, dim=-1)
    object_z_w = quaternion_to_matrix_wxyz(object_quat_w)[..., :, 2]
    goal_z_w = quaternion_to_matrix_wxyz(goal_quat_w)[..., :, 2]
    normal_alignment = torch.sum(object_z_w * goal_z_w, dim=-1)  # signed$z_o^Tz_g$
    success = (orientation_error < orientation_threshold_m) & (position_error < position_threshold_m)
    return orientation_error, position_error, normal_alignment, success


def full_pose_keypoint_reward(
    object_pos_w: torch.Tensor,
    object_quat_w: torch.Tensor,
    position_anchor_w: torch.Tensor,
    goal_quat_w: torch.Tensor,
    *,
    keypoint_radius_m: float = 0.05,
) -> torch.Tensor:
    r"""计算六点full-pose kernel reward，返回每env无量纲均值。"""

    keypoints = torch.tensor(
        (
            (keypoint_radius_m, 0.0, 0.0),
            (-keypoint_radius_m, 0.0, 0.0),
            (0.0, keypoint_radius_m, 0.0),
            (0.0, -keypoint_radius_m, 0.0),
            (0.0, 0.0, keypoint_radius_m),
            (0.0, 0.0, -keypoint_radius_m),
        ),
        dtype=object_pos_w.dtype,
        device=object_pos_w.device,
    )
    current = object_pos_w.unsqueeze(1) + torch.einsum(
        "bij,kj->bki", quaternion_to_matrix_wxyz(object_quat_w), keypoints
    )
    goal = position_anchor_w.unsqueeze(1) + torch.einsum(
        "bij,kj->bki", quaternion_to_matrix_wxyz(goal_quat_w), keypoints
    )
    distance = torch.linalg.vector_norm(current - goal, dim=-1)
    exponent = torch.clamp(50.0 * distance, min=0.0, max=30.0)
    kernel = 4.0 / (torch.exp(exponent) + 2.0 + torch.exp(-exponent))
    return kernel.mean(dim=-1)


def task_termination_flags(
    position_error_m: torch.Tensor,
    normal_alignment: torch.Tensor,
    *,
    drop_distance_m: float = 0.07,
    max_axis_angle_deg: float = 45.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""返回drop与signed normal-axis failure bool tensors。"""

    drop = position_error_m >= drop_distance_m  # 闭边界7 cm
    threshold = math.cos(math.radians(max_axis_angle_deg))
    axis_failure = normal_alignment < threshold  # 不取abs；反向法向必须失败
    return drop, axis_failure


def impulse_to_rate(impulse: torch.Tensor, step_dt_s: float) -> torch.Tensor:
    r"""把one-step impulse转换成RewardManager积分前的rate。"""

    if not math.isfinite(step_dt_s) or step_dt_s <= 0.0:
        raise ValueError("step_dt_s must be finite and positive")
    return impulse.to(dtype=torch.float32) / step_dt_s


def contact_role_reward(
    tip_contact_bits: torch.Tensor,
    tip_active_mask: torch.Tensor,
    finger_non_tip_bits: torch.Tensor,
    finger_non_tip_active_mask: torch.Tensor,
    *,
    minimum_tip_contacts: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""返回good-TIP与bad-finger-non-tip indicator；palm不进入坏接触。"""

    if tip_contact_bits.shape != tip_active_mask.shape or finger_non_tip_bits.shape != finger_non_tip_active_mask.shape:
        raise ValueError("contact bits and active masks must share role-specific shapes")
    if any(tensor.dtype != torch.bool for tensor in (tip_contact_bits, tip_active_mask, finger_non_tip_bits, finger_non_tip_active_mask)):
        raise TypeError("contact bits and masks must be bool")
    tip_count = (tip_contact_bits & tip_active_mask).sum(dim=-1)
    good_tip = tip_count >= minimum_tip_contacts
    bad_non_tip = (finger_non_tip_bits & finger_non_tip_active_mask).any(dim=-1)
    return good_tip.to(dtype=torch.float32), bad_non_tip.to(dtype=torch.float32)


def equal_asset_mean(metric_sum: torch.Tensor, episode_count: torch.Tensor) -> torch.Tensor:
    r"""由per-asset sum/count计算unique-asset等权均值。"""

    if metric_sum.shape != episode_count.shape or metric_sum.ndim != 1:
        raise ValueError("metric_sum and episode_count must share rank-1 asset axis")
    valid = episode_count > 0
    if not bool(valid.any().item()):
        raise ValueError("equal-asset mean requires at least one observed asset")
    per_asset = metric_sum[valid] / episode_count[valid]
    return per_asset.mean()


__all__ = [
    "active_reference_l2",
    "active_reference_sum",
    "axis_angle_from_quaternion_wxyz",
    "contact_role_reward",
    "equal_asset_mean",
    "full_pose_keypoint_reward",
    "goal_errors_and_success",
    "hand_axis_to_world",
    "impulse_to_rate",
    "moving_goal_quaternion",
    "normalize_quaternion_wxyz",
    "orientation_keypoint_distance",
    "projected_space_rotation_delta",
    "quaternion_apply_wxyz",
    "quaternion_from_angle_axis_wxyz",
    "quaternion_inverse_wxyz",
    "quaternion_multiply_wxyz",
    "quaternion_to_matrix_wxyz",
    "task_termination_flags",
]
