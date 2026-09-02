r"""Heterogeneous tactile-rotation baseline reward terms。

每个term返回RewardManager乘``step_dt``之前的值。Pose kernel无量纲；rotation是rad/s；success/failure是
one-step impulse除以policy dt形成的rate。Contact terms在hetero-owned contact state模块中取mask-aware bits。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..contact_layout import HeterogeneousContactLayout
from .commands import get_rotation_command
from .contact_state import get_contact_state
from .task_math import contact_role_reward, full_pose_keypoint_reward, impulse_to_rate

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
    termination_term_names: Sequence[str],
) -> torch.Tensor:
    r"""对非timeout failure terms做OR并转换为one-step rate。"""

    if not termination_term_names:
        raise ValueError("failure reward requires at least one termination term")
    failure = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for term_name in termination_term_names:
        failure |= env.termination_manager.get_term(term_name)
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


__all__ = [
    "bad_finger_non_tip_contact",
    "failure_termination_impulse_rate",
    "good_tip_contact",
    "goal_success_impulse_rate",
    "pose_keypoint_reward",
    "signed_rotation_progress_rate",
]
