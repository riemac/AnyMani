r"""Heterogeneous tactile-rotation baseline reward terms。

每个term返回RewardManager乘``step_dt``之前的值。Pose kernel无量纲；rotation是rad/s；success/failure是
one-step impulse除以policy dt形成的rate。Contact terms在hetero-owned contact state模块中取mask-aware bits。
"""

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
    full_pose_keypoint_reward,
    impulse_to_rate,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from isaaclab.envs import ManagerBasedRLEnv


def pose_keypoint_reward(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    r"""返回六点full-pose kernel$\frac16\sum_i4/(e^{50x_i}+2+e^{-50x_i})$。"""

    command = get_rotation_command(env, command_name)
    return full_pose_keypoint_reward(
        command.object.data.root_pos_w,
        command.object.data.root_quat_w,
        command.position_anchor_w,
        command.goal_quat_w,
        keypoint_radius_m=float(command.cfg.keypoint_radius_m),
    )


def signed_rotation_progress_rate(
    env: ManagerBasedRLEnv,
    command_name: str,
    *,
    clip_rad_per_step: float = 0.025,
) -> torch.Tensor:
    r"""返回$\operatorname{clip}(\Delta\psi,-0.025,0.025)/\Delta t$，保留反向惩罚。"""

    command = get_rotation_command(env, command_name)
    return torch.clamp(command.delta_psi, min=-clip_rad_per_step, max=clip_rad_per_step) / float(env.step_dt)


def goal_success_impulse_rate(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    r"""把strict orientation+position success pulse转换成one-step rate。"""

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
    r"""冻结pre-reset evaluation snapshot，再对非timeout failures做OR并转换为rate。"""

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
    r"""返回mask-aware$[n_{tip}\ge2]$ indicator。"""

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
    r"""返回mask-aware finger non-tip OR；PALM support不进入输入。"""

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
    r"""返回$\lambda_{cell}[n_{tip}\ge2]$，保持N000 contact release语义。"""

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
    r"""返回$\lambda_{cell}[n_{finger\ non-tip}>0]$；PALM仍保持中性。"""

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
    r"""N000速度带$[\max(0,\omega_{min}-\bar\omega)]^2+[\max(0,\bar\omega-\omega_{max})]^2$。"""

    if speed_max_rad_s <= speed_min_rad_s:
        raise ValueError("speed_max_rad_s must exceed speed_min_rad_s")
    command = get_rotation_command(env, command_name)
    below = torch.clamp(speed_min_rad_s - command.axis_speed_ema_rad_s, min=0.0)
    above = torch.clamp(command.axis_speed_ema_rad_s - speed_max_rad_s, min=0.0)
    return (below.square() + above.square()) * reward_release_gain(env)


def object_axis_speed_jitter_curriculum(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    r"""惩罚瞬时轴速相对0.25 s EMA的残差平方。"""

    command = get_rotation_command(env, command_name)
    return (command.axis_speed_rad_s - command.axis_speed_ema_rad_s).square() * reward_release_gain(env)


def object_off_axis_angular_velocity_curriculum(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    r"""惩罚object angular velocity正交于目标轴的平方范数。"""

    command = get_rotation_command(env, command_name)
    angular_velocity = command.object.data.root_ang_vel_w
    parallel = torch.sum(angular_velocity * command.axis_w, dim=-1, keepdim=True) * command.axis_w
    return torch.sum((angular_velocity - parallel).square(), dim=-1) * reward_release_gain(env)


def object_linear_velocity_curriculum(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    r"""惩罚object world linear velocity平方范数，抑制掌面滑移/弹跳。"""

    command = get_rotation_command(env, command_name)
    return torch.sum(command.object.data.root_lin_vel_w.square(), dim=-1) * reward_release_gain(env)


def _active_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    r"""读取pregrasp sidecar发布的真实active-joint mask。"""

    sidecar = getattr(env, HETERO_PREGRASP_STATE_ATTR, None)
    if not isinstance(sidecar, HeterogeneousPregraspState) or not bool(sidecar.valid.all().item()):
        raise RuntimeError("stable reward requires resolved good-pregrasp sidecar")
    return sidecar.active_joint_mask


def joint_pose_anchor_curriculum(
    env: ManagerBasedRLEnv,
    *,
    robot_name: str = "robot",
) -> torch.Tensor:
    r"""返回16-DoF参考姿态偏移$\sqrt{\frac{16}{n_i}\sum_j(q_j-q_{0,j})^2}$。"""

    robot = cast(Articulation, env.scene[robot_name])
    sidecar = cast(HeterogeneousPregraspState, getattr(env, HETERO_PREGRASP_STATE_ATTR))
    penalty = active_reference_l2(robot.data.joint_pos - sidecar.q_state_rad, _active_mask(env))
    return penalty * reward_release_gain(env)


def joint_mechanical_power_curriculum(
    env: ManagerBasedRLEnv,
    *,
    robot_name: str = "robot",
) -> torch.Tensor:
    r"""返回16-DoF参考机械功率$\frac{16}{n_i}\sum_j|\tau_j\dot q_j|$，单位W。"""

    robot = cast(Articulation, env.scene[robot_name])
    power = torch.abs(robot.data.computed_torque * robot.data.joint_vel)
    return active_reference_sum(power, _active_mask(env)) * reward_release_gain(env)


def torque_l2_curriculum(env: ManagerBasedRLEnv, *, robot_name: str = "robot") -> torch.Tensor:
    r"""返回16-DoF参考torque平方和$\frac{16}{n_i}\sum_j\tau_j^2$，单位$(N\,m)^2$。"""

    robot = cast(Articulation, env.scene[robot_name])
    return active_reference_sum(robot.data.computed_torque.square(), _active_mask(env)) * reward_release_gain(env)


def action_l2_curriculum(env: ManagerBasedRLEnv) -> torch.Tensor:
    r"""返回16-DoF参考action平方和$\frac{16}{n_i}\sum_ja_j^2$。"""

    return active_reference_sum(env.action_manager.action.square(), _active_mask(env)) * reward_release_gain(env)


def action_rate_l2_curriculum(env: ManagerBasedRLEnv) -> torch.Tensor:
    r"""返回16-DoF参考action-rate平方和$\frac{16}{n_i}\sum_j(a_{t,j}-a_{t-1,j})^2$。"""

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
