from __future__ import annotations

import torch
from isaaclab.utils.math import quat_conjugate, quat_mul


@torch.jit.script
def scale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor):
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def rotation_distance(object_rot: torch.Tensor, target_rot: torch.Tensor):
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))


@torch.jit.script
def compute_rewards(
    reset_buf: torch.Tensor,
    reset_goal_buf: torch.Tensor,
    successes: torch.Tensor,
    consecutive_successes: torch.Tensor,
    max_episode_length: float,
    fingertip_pos: torch.Tensor,
    object_pos: torch.Tensor,
    object_rot: torch.Tensor,
    target_pos: torch.Tensor,
    target_rot: torch.Tensor,
    object_linvel: torch.Tensor,
    object_angvel: torch.Tensor,
    dist_reward_scale: float,
    rot_reward_scale: float,
    rot_eps: float,
    actions: torch.Tensor,
    action_penalty_scale: float,
    pose_diff_penalty: torch.Tensor,
    pose_diff_penalty_scale: float,
    torque_penalty: torch.Tensor,
    torque_penalty_scale: float,
    success_tolerance: float,
    reach_goal_bonus: float,
    fall_dist: float,
    fall_penalty: float,
    av_factor: float,
):
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)
    rot_dist = rotation_distance(object_rot, target_rot)

    dist_rew = goal_dist * dist_reward_scale
    rot_rew = 1.0 / (torch.abs(rot_dist) + rot_eps) * rot_reward_scale
    action_penalty = torch.sum(actions**2, dim=-1)
    pose_penalty = pose_diff_penalty * pose_diff_penalty_scale

    reward = dist_rew + rot_rew + action_penalty * action_penalty_scale + pose_penalty + torque_penalty * torque_penalty_scale

    goal_resets = torch.where(
        (torch.abs(rot_dist) <= success_tolerance) & (goal_dist <= 0.025),
        torch.ones_like(reset_goal_buf),
        reset_goal_buf,
    )
    successes = successes + goal_resets
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)
    reward = torch.where((object_angvel[:, 2] > 0.25) & (object_angvel[:, 2] < 1.5), reward + 1, reward)
    reward = torch.where(goal_dist >= fall_dist, reward + fall_penalty, reward)

    resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())
    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return reward, goal_resets, successes, cons_successes
