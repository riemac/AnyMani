(
    'Heterogeneous tactile-rotation baseline reward terms. Each term returns its '
    'value before RewardManager multiplies by step_dt. Pose kernel is '
    'dimensionless; rotation reward is in rad/s; one-step success/failure '
    'impulses are divided by policy dt. Contact terms read mask-aware bits from '
    'the task-owned contact state.'
)

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from isaaclab.assets import Articulation

from ..contact_layout import HeterogeneousContactLayout
from .commands import get_rotation_command
from .contact_state import get_contact_state
from .curriculums import reward_release_gain
from .runtime_state import HETERO_PREGRASP_STATE_ATTR, HeterogeneousPregraspState
from .task_math import (
    active_reference_l2,
    active_reference_sum,
    contact_role_reward,
    exponential_orientation_step_reward,
    full_pose_keypoint_reward,
    impulse_to_rate,
    inverse_orientation_step_reward,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from isaaclab.envs import ManagerBasedRLEnv


def pose_keypoint_reward(env: ManagerBasedRLEnv, command_name: str, *, position_only: bool = False) -> torch.Tensor:
    (
        'Return full-pose or position-only kernel [N] before weight and policy dt. '
        'Measure six local points at +/-r along x/y/z: '
        'd_i=norm(p_o+R_o*b_i-p_star-R_g*b_i), in meters. Average '
        'kappa(d_i)=4/(exp(50*d_i)+2+exp(-50*d_i))=sech^2(25*d_i), with 50 in 1/m. '
        'Cap s=50*d at 30 in the stable implementation; this is exact over the '
        'original nonterminal range at r=0.05 m. position_only sets r=0, reducing the '
        'six points to center distance and excluding orientation. Command keypoints '
        'still govern strict goal error, and this function does not mutate command '
        'state. At 20 Hz and weight 1, center reward is 0.05 per step.'
    )

    command = get_rotation_command(env, command_name)  # Read only the object and goal state.
    reward_radius_m = 0.0 if position_only else float(command.cfg.keypoint_radius_m)  # Use reward-specific measurement geometry.
    return full_pose_keypoint_reward(
        command.object.data.root_pos_w,  # World-frame object center [N,3], meters.
        command.object.data.root_quat_w,  # Object quaternion [N,4], order wxyz; orientation has no effect when r=0.
        command.position_anchor_w,  # Original position anchor [N,3], meters.
        command.goal_quat_w,  # Original goal quaternion; strict goal checks continue to use it.
        keypoint_radius_m=reward_radius_m,  # Average six points; at r=0 all use the same center-distance kernel.
    )


def signed_rotation_progress_rate(
    env: ManagerBasedRLEnv,
    command_name: str,
    *,
    clip_rad_per_step: float = 0.025,
) -> torch.Tensor:
    'Return clip(delta_psi, -0.025, 0.025) / dt; reverse rotation remains penalized.'

    command = get_rotation_command(env, command_name)
    return torch.clamp(command.delta_psi, min=-clip_rad_per_step, max=clip_rad_per_step) / float(env.step_dt)


def track_orientation_inv_l2(env: ManagerBasedRLEnv, command_name: str, *, rot_eps: float = 0.1) -> torch.Tensor:
    (
        'Return the legacy in-hand inverse-angle reward w/(theta+epsilon), where '
        'theta=norm(Log(R_goal * transpose(R_object))) in rad. Divide by policy dt to '
        'cancel RewardManager integration, matching official_orientation when w=1; '
        'position is excluded.'
    )
    command = get_rotation_command(env, command_name)
    return inverse_orientation_step_reward(command.orientation_error_rad, rot_eps) / float(env.step_dt)


def track_orientation_exponential(
    env: ManagerBasedRLEnv, command_name: str, *, slope_rad_inv: float = 4.0, denominator_epsilon: float = 0.1,
) -> torch.Tensor:
    (
        'Return the actual per-step exponential reward w/(exp(k*theta)+epsilon). '
        'Theta is in rad, k in rad^-1, and epsilon is dimensionless. Divide by policy '
        'dt to cancel RewardManager integration; with weight 1, a 30-degree error '
        'gives 0.1216467 per step. Position, goal advancement, and bonuses remain '
        'separate.'
    )
    command = get_rotation_command(env, command_name)  # Consume this physics step's goal-angle error [N].
    value = exponential_orientation_step_reward(
        command.orientation_error_rad, slope_rad_inv, denominator_epsilon,
    )  # Unweighted per-policy-step reward [N]; command state is unchanged.
    return value / float(env.step_dt)  # Manager multiplies by dt afterward, restoring the per-step value.


def goal_success_impulse_rate(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    'Convert the strict orientation-and-position success pulse to a one-step rate.'

    command = get_rotation_command(env, command_name)
    return impulse_to_rate(command.goal_success_pulse, float(env.step_dt)).to(device=command.device)


def failure_termination_impulse_rate(
    env: ManagerBasedRLEnv,
    *,
    command_name: str,
    termination_term_names: Sequence[str],
    layout: HeterogeneousContactLayout,
    active_joint_mask_by_env: Sequence[Sequence[bool]],
    ema_alpha: float = 0.5,
    force_threshold_N: float = 0.25,
) -> torch.Tensor:
    'Freeze the pre-reset evaluation snapshot, OR non-timeout failures, and convert to a rate.'

    if not termination_term_names:
        raise ValueError("failure reward requires at least one termination term")
    termination_bits = {
        term_name: env.termination_manager.get_term(term_name)
        for term_name in (*termination_term_names, "time_out")
    }
    command = get_rotation_command(env, command_name)
    contact = get_contact_state(
        env,
        layout=layout,
        active_joint_mask_by_env=active_joint_mask_by_env,
        ema_alpha=ema_alpha,
        force_threshold_N=force_threshold_N,
    )
    command.capture_post_physics_evaluation_snapshot(contact, termination_bits)
    failure = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for term_name in termination_term_names:
        failure |= termination_bits[term_name]
    return impulse_to_rate(failure, float(env.step_dt)).to(device=env.device)


def good_tip_contact(
    env: ManagerBasedRLEnv,
    *,
    layout: HeterogeneousContactLayout,
    active_joint_mask_by_env: Sequence[Sequence[bool]],
    minimum_tip_contacts: int = 2,
    ema_alpha: float = 0.5,
    force_threshold_N: float = 0.25,
) -> torch.Tensor:
    'Return a mask-aware indicator for at least two active TIP contacts.'

    state = get_contact_state(
        env,
        layout=layout,
        active_joint_mask_by_env=active_joint_mask_by_env,
        ema_alpha=ema_alpha,
        force_threshold_N=force_threshold_N,
    )
    good, _ = contact_role_reward(
        state.tip_bits,
        state.active_sensor_mask[:, :4],
        state.finger_non_tip_bits,
        state.active_sensor_mask[:, 4:23],
        minimum_tip_contacts=minimum_tip_contacts,
    )
    return good


def bad_finger_non_tip_contact(
    env: ManagerBasedRLEnv,
    *,
    layout: HeterogeneousContactLayout,
    active_joint_mask_by_env: Sequence[Sequence[bool]],
    ema_alpha: float = 0.5,
    force_threshold_N: float = 0.25,
) -> torch.Tensor:
    'Return the mask-aware finger non-tip OR; exclude PALM support.'

    state = get_contact_state(
        env,
        layout=layout,
        active_joint_mask_by_env=active_joint_mask_by_env,
        ema_alpha=ema_alpha,
        force_threshold_N=force_threshold_N,
    )
    _, bad = contact_role_reward(
        state.tip_bits,
        state.active_sensor_mask[:, :4],
        state.finger_non_tip_bits,
        state.active_sensor_mask[:, 4:23],
    )
    return bad


def good_tip_contact_curriculum(
    env: ManagerBasedRLEnv,
    *,
    layout: HeterogeneousContactLayout,
    active_joint_mask_by_env: Sequence[Sequence[bool]],
    minimum_tip_contacts: int = 2,
    ema_alpha: float = 0.5,
    force_threshold_N: float = 0.25,
) -> torch.Tensor:
    'Return lambda_cell times the at-least-two-TIP indicator, preserving N000 contact-release semantics.'

    return good_tip_contact(
        env,
        layout=layout,
        active_joint_mask_by_env=active_joint_mask_by_env,
        minimum_tip_contacts=minimum_tip_contacts,
        ema_alpha=ema_alpha,
        force_threshold_N=force_threshold_N,
    ) * reward_release_gain(env)


def bad_finger_non_tip_contact_curriculum(
    env: ManagerBasedRLEnv,
    *,
    layout: HeterogeneousContactLayout,
    active_joint_mask_by_env: Sequence[Sequence[bool]],
    ema_alpha: float = 0.5,
    force_threshold_N: float = 0.25,
) -> torch.Tensor:
    'Return lambda_cell times any finger non-tip contact; PALM remains neutral.'

    return bad_finger_non_tip_contact(
        env,
        layout=layout,
        active_joint_mask_by_env=active_joint_mask_by_env,
        ema_alpha=ema_alpha,
        force_threshold_N=force_threshold_N,
    ) * reward_release_gain(env)


def object_axis_speed_band_curriculum(
    env: ManagerBasedRLEnv,
    command_name: str,
    *,
    speed_min_rad_s: float = 0.6,
    speed_max_rad_s: float = 0.833,
) -> torch.Tensor:
    'N000 soft speed band: square shortfall below omega_min and excess above omega_max.'

    if speed_max_rad_s <= speed_min_rad_s:
        raise ValueError("speed_max_rad_s must exceed speed_min_rad_s")
    command = get_rotation_command(env, command_name)
    below = torch.clamp(speed_min_rad_s - command.axis_speed_ema_rad_s, min=0.0)
    above = torch.clamp(command.axis_speed_ema_rad_s - speed_max_rad_s, min=0.0)
    return (below.square() + above.square()) * reward_release_gain(env)


def object_axis_speed_jitter_curriculum(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    'Penalize squared residual of instantaneous axis speed from its 0.25 s EMA.'

    command = get_rotation_command(env, command_name)
    return (command.axis_speed_rad_s - command.axis_speed_ema_rad_s).square() * reward_release_gain(env)


def object_off_axis_angular_velocity_curriculum(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    'Penalize squared object angular velocity orthogonal to the target axis.'

    command = get_rotation_command(env, command_name)
    angular_velocity = command.object.data.root_ang_vel_w
    parallel = torch.sum(angular_velocity * command.axis_w, dim=-1, keepdim=True) * command.axis_w
    return torch.sum((angular_velocity - parallel).square(), dim=-1) * reward_release_gain(env)


def object_linear_velocity_curriculum(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    'Penalize squared world linear speed to reduce palm sliding and bouncing.'

    command = get_rotation_command(env, command_name)
    return torch.sum(command.object.data.root_lin_vel_w.square(), dim=-1) * reward_release_gain(env)


def _active_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    'Read the active-joint mask published by the pregrasp sidecar.'

    sidecar = getattr(env, HETERO_PREGRASP_STATE_ATTR, None)
    if not isinstance(sidecar, HeterogeneousPregraspState) or not bool(sidecar.valid.all().item()):
        raise RuntimeError("stable reward requires resolved good-pregrasp sidecar")
    return sidecar.active_joint_mask


def joint_pose_anchor_curriculum(
    env: ManagerBasedRLEnv,
    *,
    robot_name: str = "robot",
) -> torch.Tensor:
    'Return the 16-DoF-equivalent reference-pose offset: sqrt((16/n_active) * sum((q_j-q0_j)^2)).'

    robot = cast(Articulation, env.scene[robot_name])
    sidecar = cast(HeterogeneousPregraspState, getattr(env, HETERO_PREGRASP_STATE_ATTR))
    penalty = active_reference_l2(robot.data.joint_pos - sidecar.q_state_rad, _active_mask(env))
    return penalty * reward_release_gain(env)


def joint_mechanical_power_curriculum(
    env: ManagerBasedRLEnv,
    *,
    robot_name: str = "robot",
) -> torch.Tensor:
    'Return 16-DoF-equivalent mechanical power: (16/n_active) * sum(abs(tau_j*qdot_j)), in watts.'

    robot = cast(Articulation, env.scene[robot_name])
    power = torch.abs(robot.data.computed_torque * robot.data.joint_vel)
    return active_reference_sum(power, _active_mask(env)) * reward_release_gain(env)


def torque_l2_curriculum(env: ManagerBasedRLEnv, *, robot_name: str = "robot") -> torch.Tensor:
    'Return 16-DoF-equivalent torque squared: (16/n_active) * sum(tau_j^2), in (N m)^2.'

    robot = cast(Articulation, env.scene[robot_name])
    return active_reference_sum(robot.data.computed_torque.square(), _active_mask(env)) * reward_release_gain(env)


def action_l2_curriculum(env: ManagerBasedRLEnv) -> torch.Tensor:
    'Return 16-DoF-equivalent action squared: (16/n_active) * sum(a_j^2).'

    return active_reference_sum(env.action_manager.action.square(), _active_mask(env)) * reward_release_gain(env)


def action_rate_l2_curriculum(env: ManagerBasedRLEnv) -> torch.Tensor:
    'Return 16-DoF-equivalent action-rate squared: (16/n_active) * sum((a_t,j-a_(t-1),j)^2).'

    difference = env.action_manager.action - env.action_manager.prev_action
    return active_reference_sum(difference.square(), _active_mask(env)) * reward_release_gain(env)


__all__ = [
    "bad_finger_non_tip_contact",
    "bad_finger_non_tip_contact_curriculum",
    "action_l2_curriculum",
    "action_rate_l2_curriculum",
    "failure_termination_impulse_rate",
    "good_tip_contact",
    "good_tip_contact_curriculum",
    "goal_success_impulse_rate",
    "pose_keypoint_reward",
    "joint_mechanical_power_curriculum",
    "joint_pose_anchor_curriculum",
    "object_axis_speed_band_curriculum",
    "object_axis_speed_jitter_curriculum",
    "object_linear_velocity_curriculum",
    "object_off_axis_angular_velocity_curriculum",
    "signed_rotation_progress_rate",
    "torque_l2_curriculum",
]
