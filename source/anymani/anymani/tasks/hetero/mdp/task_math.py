(
    'Pure-Torch SO(3), reward, and evaluation math for palm-up DexCube rotation. '
    'Quaternion order is (w,x,y,z). Adjacent-frame motion is far below pi, so '
    'signed progress uses the principal quaternion log with q/-q '
    'canonicalization. Isaac owns reward time integration; this module returns '
    'rates or dimensionless shape rewards.'
)

from __future__ import annotations

import math

import torch


def active_reference_sum(
    values: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    reference_dof: int = 16,
) -> torch.Tensor:
    (
        'Scale each per-joint sum by 16/n_active for comparison at the N000 reference '
        'DoF. The 16-DoF hand exactly recovers the original sum; lower-DoF hands use '
        'the active-joint mean projected to the same 16-DoF scale. Ghost padding '
        'never contributes, even if stored values are finite.'
    )

    if values.shape != active_mask.shape or active_mask.dtype != torch.bool:
        raise ValueError("reference-DoF reward reduction requires matching values and bool active_mask")
    if reference_dof < 1:
        raise ValueError("reference_dof must be positive")
    weights = active_mask.to(dtype=values.dtype)  # Active-joint mask m_ij in {0,1}; same shape as values.
    active_count = weights.sum(dim=-1)  # n_i is the number of active DoFs for hand i.
    if bool((active_count < 1).any().item()):
        raise ValueError("reference-DoF reward reduction requires at least one active joint per environment")
    active_sum = (values * weights).sum(dim=-1)  # Sum of active per-joint penalties.
    return active_sum * (float(reference_dof) / active_count)  # Scale by n_ref/n_i to the N000 reference.


def active_reference_l2(
    values: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    reference_dof: int = 16,
) -> torch.Tensor:
    (
        'Return the masked L2 magnitude at the reference DoF: sqrt((16/n_active) * '
        'sum(x_j^2)). At 16 active DoFs this equals the N000 L2 norm exactly; across '
        'hands, the same typical per-joint offset has a comparable penalty. Ghost '
        'slots are excluded.'
    )

    return torch.sqrt(active_reference_sum(values.square(), active_mask, reference_dof=reference_dof))


def normalize_quaternion_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    'Normalize a finite quaternion batch with final width 4.'

    if quaternion.shape[-1] != 4 or not bool(torch.isfinite(quaternion).all().item()):
        raise ValueError("quaternion must be finite with final dimension four")
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if bool((norm < 1.0e-12).any().item()):
        raise ValueError("quaternion norm must be non-zero")
    return quaternion / norm


def quaternion_multiply_wxyz(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    'Compute Hamilton product q_left * q_right with broadcast batch axes.'

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
    'Return the unit-quaternion inverse (w,-x,-y,-z).'

    normalized = normalize_quaternion_wxyz(quaternion)
    inverse = normalized.clone()
    inverse[..., 1:] *= -1.0
    return inverse


def quaternion_to_matrix_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    'Convert a unit quaternion to a rotation matrix in SO(3).'

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
    'Build unit quaternions from angles [B] and nonzero axes [B,3].'

    if angle.shape != axis.shape[:-1] or axis.shape[-1] != 3:
        raise ValueError("angle and axis must have shapes [...], [...,3]")
    norm = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
    if bool((norm < 1.0e-12).any().item()) or not bool(torch.isfinite(angle).all().item()):
        raise ValueError("axis must be finite/non-zero and angle finite")
    normalized_axis = axis / norm
    half = 0.5 * angle
    return torch.cat((torch.cos(half).unsqueeze(-1), normalized_axis * torch.sin(half).unsqueeze(-1)), dim=-1)


def quaternion_apply_wxyz(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    'Apply rotation v_out = R(q) * v.'

    if vector.shape[-1] != 3 or quaternion.shape[:-1] != vector.shape[:-1]:
        raise ValueError("quaternion/vector batch shapes must match with final dimensions 4/3")
    return torch.einsum("...ij,...j->...i", quaternion_to_matrix_wxyz(quaternion), vector)


def axis_angle_from_quaternion_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    (
        'Return the principal rotation vector log(R), with angle in [0,pi]. '
        'Canonicalize quaternion sign so w >= 0; near identity use the 2v limit to '
        'avoid division by zero.'
    )

    q = normalize_quaternion_wxyz(quaternion)
    q = torch.where((q[..., :1] < 0.0).expand_as(q), -q, q)  # Canonical quaternion representative; q and -q are equivalent.
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
    (
        'Compute signed rotation increments between adjacent poses about a directed '
        'world-space axis: delta_R = R_t * transpose(R_(t-1)); delta_psi = '
        'dot(k_world, log(delta_R)). Units are rad.'
    )

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


def rotation_frontier_update(
    net_rotation_rad: torch.Tensor,
    previous_max_positive_rad: torch.Tensor,
    previous_frontier_count: torch.Tensor,
    *,
    frontier_interval_rad: float = math.pi / 6.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    (
        'Update the monotonic positive-rotation frontier from signed net rotation '
        'Psi_t: M_t=max(M_prev,max(0,Psi_t)), K_t=floor(M_t/delta), '
        'delta_K=K_t-K_prev, delta=pi/6 (30 deg). Reverse motion cannot lower the '
        'frontier or earn repeat rewards. Preserve delta_K when one step crosses '
        'multiple thresholds; return M, K, delta_K, and pulse on env axis [N].'
    )

    if net_rotation_rad.ndim != 1 or previous_max_positive_rad.shape != net_rotation_rad.shape:
        raise ValueError("frontier net rotation and historical maximum must share rank-1 environment axis")
    if previous_frontier_count.shape != net_rotation_rad.shape or previous_frontier_count.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise TypeError("previous frontier count must be an integer tensor on the same environment axis")
    if not math.isfinite(frontier_interval_rad) or frontier_interval_rad <= 0.0:
        raise ValueError("frontier interval must be finite and positive")

    positive_net = torch.clamp(net_rotation_rad, min=0.0)  # max(0,Psi_t), rad.
    maximum = torch.maximum(previous_max_positive_rad, positive_net)  # Historical positive frontier M_t.
    count = torch.floor(maximum / frontier_interval_rad).to(dtype=previous_frontier_count.dtype)  # Integer frontier count K_t.
    delta = torch.clamp(count - previous_frontier_count, min=0)  # Nonnegative increment delta_K_t; one step may cross multiple frontiers.
    return maximum, count, delta, delta > 0


def hand_axis_to_world(
    axis_h: torch.Tensor,
    root_quat_wxyz: torch.Tensor,
    semantic_R_ha: torch.Tensor,
) -> torch.Tensor:
    'Transform the hand axis using v_a = transpose(R_ha)*v_h and v_w = R_wa*v_a.'

    if semantic_R_ha.shape != (3, 3):
        raise ValueError("semantic_R_ha must have shape [3,3]")
    axis_a = axis_h @ semantic_R_ha  # Row-vector transform: v_a = v_h * R_ha.
    return quaternion_apply_wxyz(root_quat_wxyz, axis_a)


def moving_goal_quaternion(
    current_quat_w: torch.Tensor,
    axis_w: torch.Tensor,
    *,
    subgoal_angle_rad: float = math.pi / 6.0,
) -> torch.Tensor:
    'Create the next moving goal by left-multiplying a 30-degree world-space rotation onto the current object pose.'

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
    'Compute the mean six-axis keypoint distance for center-aligned orientation-only error, in meters.'

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
    'Return orientation, position, alignment, and strict two-gate success.'

    orientation_error = orientation_keypoint_distance(
        object_quat_w, goal_quat_w, radius_m=keypoint_radius_m
    )
    position_error = torch.linalg.vector_norm(object_pos_w - position_anchor_w, dim=-1)
    object_z_w = quaternion_to_matrix_wxyz(object_quat_w)[..., :, 2]
    goal_z_w = quaternion_to_matrix_wxyz(goal_quat_w)[..., :, 2]
    normal_alignment = torch.sum(object_z_w * goal_z_w, dim=-1)  # Signed normal alignment z_o dot z_g.
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
    'Compute the six-point full-pose kernel reward; return a dimensionless mean per env.'

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


def orientation_tracking_flags(
    orientation_error_rad: torch.Tensor,
    position_error_m: torch.Tensor,
    *,
    angle_tolerance_rad: float = 0.2,
    position_tolerance_m: float = 0.025,
) -> tuple[torch.Tensor, torch.Tensor]:
    (
        'Separate angle-based goal advancement from qualified pose bonus. Advance '
        'when theta <= theta_tol; qualify only when advance and d <= d_tol. Clear '
        'both signals after one goal consumption so persistent threshold satisfaction '
        'cannot earn repeated bonuses.'
    )
    if angle_tolerance_rad <= 0 or position_tolerance_m <= 0:
        raise ValueError('orientation/position tolerances must be positive')
    advance = orientation_error_rad <= angle_tolerance_rad
    return advance, advance & (position_error_m <= position_tolerance_m)


def inverse_orientation_step_reward(orientation_error_rad: torch.Tensor, epsilon: float = 0.1) -> torch.Tensor:
    'Return the same inverse-angle form as Isaac Lab, as the actual per-policy-step reward before weight.'
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('inverse orientation epsilon must be finite and positive')
    return 1.0 / (orientation_error_rad.abs() + epsilon)  # Theta is the shortest SO(3) rotation angle, rad.


def exponential_orientation_step_reward(
    orientation_error_rad: torch.Tensor, slope_rad_inv: float = 4.0, denominator_epsilon: float = 0.1,
) -> torch.Tensor:
    (
        'Return the per-step exponential pose kernel before weight: '
        '1/(exp(k*abs(theta))+epsilon). Theta is shortest SO(3) angle in rad, k in '
        'rad^-1, and epsilon is dimensionless. With k=4 and epsilon=0.1, values are '
        '0.1216467 at 30 deg and 0.4300075 at 0.2 rad. Do not rescale the peak or '
        'subtract a baseline; use the stable equivalent form to avoid exponent '
        'clipping.'
    )
    if not math.isfinite(slope_rad_inv) or slope_rad_inv <= 0:  # Positive slope makes reward decrease with angle error.
        raise ValueError("orientation exponential slope must be finite and positive")
    if not math.isfinite(denominator_epsilon) or denominator_epsilon < 0:  # Denominator is at least 1.
        raise ValueError("orientation exponential epsilon must be finite and nonnegative")
    decay = torch.exp(-slope_rad_inv * orientation_error_rad.abs())  # Dimensionless; shape [N] or input shape.
    return decay / (1.0 + denominator_epsilon * decay)  # Exactly equivalent to 1/(exp(k*abs(theta))+epsilon).


def task_termination_flags(
    position_error_m: torch.Tensor,
    normal_alignment: torch.Tensor,
    *,
    drop_distance_m: float = 0.07,
    max_axis_angle_deg: float = 45.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    'Return drop and signed normal-axis failure bool tensors.'

    drop = position_error_m >= drop_distance_m  # Closed 7 cm boundary.
    threshold = math.cos(math.radians(max_axis_angle_deg))
    axis_failure = normal_alignment < threshold  # Keep the sign; an opposite-facing normal must fail.
    return drop, axis_failure


def impulse_to_rate(impulse: torch.Tensor, step_dt_s: float) -> torch.Tensor:
    'Convert a one-step impulse to a rate before RewardManager integration.'

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
    'Return good-TIP and bad-finger-non-tip indicators; PALM is not a bad contact.'

    if tip_contact_bits.shape != tip_active_mask.shape or finger_non_tip_bits.shape != finger_non_tip_active_mask.shape:
        raise ValueError("contact bits and active masks must share role-specific shapes")
    if any(tensor.dtype != torch.bool for tensor in (tip_contact_bits, tip_active_mask, finger_non_tip_bits, finger_non_tip_active_mask)):
        raise TypeError("contact bits and masks must be bool")
    tip_count = (tip_contact_bits & tip_active_mask).sum(dim=-1)
    good_tip = tip_count >= minimum_tip_contacts
    bad_non_tip = (finger_non_tip_bits & finger_non_tip_active_mask).any(dim=-1)
    return good_tip.to(dtype=torch.float32), bad_non_tip.to(dtype=torch.float32)


def equal_asset_mean(metric_sum: torch.Tensor, episode_count: torch.Tensor) -> torch.Tensor:
    'Compute the equal-weight unique-asset mean from per-asset sums/counts.'

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
    "rotation_frontier_update",
    "quaternion_apply_wxyz",
    "quaternion_from_angle_axis_wxyz",
    "quaternion_inverse_wxyz",
    "quaternion_multiply_wxyz",
    "quaternion_to_matrix_wxyz",
    "task_termination_flags",
]
