r"""ManagerBased structured actor/critic observation terms。

每个term返回一个具名tensor；group不concatenate。Actor只读取JOINT proprioception/history与TIP contact；critic
额外读取joint velocity、all-owner contact、object和task。Asset row与morphology cell不进入数值observation。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import torch
from isaaclab.assets import Articulation

from ..contact_layout import HeterogeneousContactLayout
from .actions import PreloadAwareMaskedRelativeJointPositionAction
from .commands import get_rotation_command
from .contact_state import get_contact_state
from .object_state import object_state_in_hand_frame, task_state
from .observation_state import (
    actor_joint_contact_frame,
    actor_joint_current,
    actor_joint_history_frame,
    actor_joint_limits,
    actor_owner_contact,
    actor_tip_contact,
    critic_joint_state,
)
from .runtime_state import derive_tip_and_owner_masks

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _mask_tensor(env: ManagerBasedRLEnv, active_joint_mask_by_env: Sequence[Sequence[bool]]) -> torch.Tensor:
    r"""把config-static routing恢复为device bool$[N,16]$。"""

    mask = torch.tensor(active_joint_mask_by_env, dtype=torch.bool, device=env.device)
    if mask.shape != (env.num_envs, 16):
        raise ValueError("observation routing must provide [num_envs,16] mask")
    return mask


def _action_term(env: ManagerBasedRLEnv, action_name: str) -> PreloadAwareMaskedRelativeJointPositionAction:
    r"""解析hetero preload-aware action term。"""

    term = env.action_manager.get_term(action_name)
    if not isinstance(term, PreloadAwareMaskedRelativeJointPositionAction):
        raise TypeError(f"action {action_name!r} is not preload-aware heterogeneous action")
    return term


def palm_valid(env: ManagerBasedRLEnv) -> torch.Tensor:
    r"""返回PALM validity$[N,1]$；PALM本身不伪造proprioceptive features。"""

    return torch.ones(env.num_envs, 1, device=env.device)


def joint_valid(env: ManagerBasedRLEnv, active_joint_mask_by_env: Sequence[Sequence[bool]]) -> torch.Tensor:
    r"""返回float$[N,16]$ JOINT mask，供Gym space/transport；内部真源仍为bool。"""

    return _mask_tensor(env, active_joint_mask_by_env).to(dtype=torch.float32)


def tip_valid(env: ManagerBasedRLEnv, active_joint_mask_by_env: Sequence[Sequence[bool]]) -> torch.Tensor:
    r"""返回float$[N,4]$ TIP mask。"""

    tip_mask, _ = derive_tip_and_owner_masks(_mask_tensor(env, active_joint_mask_by_env))
    return tip_mask.to(dtype=torch.float32)


def owner_valid(env: ManagerBasedRLEnv, active_joint_mask_by_env: Sequence[Sequence[bool]]) -> torch.Tensor:
    r"""返回float$[N,21]$ PALM/JOINT/TIP owner mask。"""

    _, owner_mask = derive_tip_and_owner_masks(_mask_tensor(env, active_joint_mask_by_env))
    return owner_mask.to(dtype=torch.float32)


def actor_joint_current_term(
    env: ManagerBasedRLEnv,
    *,
    active_joint_mask_by_env: Sequence[Sequence[bool]],
    action_name: str,
    robot_name: str = "robot",
) -> torch.Tensor:
    r"""返回$[N,16,3]$ actor current JOINT block。"""

    robot = cast(Articulation, env.scene[robot_name])
    action = _action_term(env, action_name)
    mask = _mask_tensor(env, active_joint_mask_by_env)
    return actor_joint_current(robot.data.joint_pos, action.current_targets, action.executed_actions, mask)


def actor_joint_history_frame_term(
    env: ManagerBasedRLEnv,
    *,
    active_joint_mask_by_env: Sequence[Sequence[bool]],
    action_name: str,
    layout: HeterogeneousContactLayout,
    robot_name: str = "robot",
    ema_alpha: float = 0.5,
    force_threshold_N: float = 0.25,
) -> torch.Tensor:
    r"""返回供ObservationManager形成History30的当前$[N,16,4]$ raw frame。"""

    robot = cast(Articulation, env.scene[robot_name])
    action = _action_term(env, action_name)
    mask = _mask_tensor(env, active_joint_mask_by_env)
    contact = get_contact_state(
        env,
        layout=layout,
        active_joint_mask_by_env=active_joint_mask_by_env,
        ema_alpha=ema_alpha,
        force_threshold_N=force_threshold_N,
    )
    return actor_joint_history_frame(
        robot.data.joint_pos,
        action.current_targets,
        action.executed_actions,
        contact.tip_bits,
        mask,
    )


def actor_joint_contact_frame_term(
    env: ManagerBasedRLEnv,
    *,
    active_joint_mask_by_env: Sequence[Sequence[bool]],
    action_name: str,
    layout: HeterogeneousContactLayout,
    robot_name: str = "robot",
    ema_alpha: float = 0.5,
    force_threshold_N: float = 0.25,
) -> torch.Tensor:
    r"""返回MVP actor当前/History30共用的`[N,16,5]` own-JOINT＋TIP-contact帧。"""

    robot = cast(Articulation, env.scene[robot_name])
    action = _action_term(env, action_name)
    mask = _mask_tensor(env, active_joint_mask_by_env)
    contact = get_contact_state(
        env,
        layout=layout,
        active_joint_mask_by_env=active_joint_mask_by_env,
        ema_alpha=ema_alpha,
        force_threshold_N=force_threshold_N,
    )
    _, owner_bits = contact.owner_force_and_bits()
    return actor_joint_contact_frame(
        robot.data.joint_pos,
        action.current_targets,
        action.executed_actions,
        owner_bits[:, 1:17],
        contact.tip_bits,
        mask,
    )


def actor_owner_contact_term(
    env: ManagerBasedRLEnv,
    *,
    active_joint_mask_by_env: Sequence[Sequence[bool]],
    layout: HeterogeneousContactLayout,
    ema_alpha: float = 0.5,
    force_threshold_N: float = 0.25,
) -> torch.Tensor:
    r"""返回MVP global residual消费的`[N,21,1]` owner binary contact。"""

    mask = _mask_tensor(env, active_joint_mask_by_env)
    contact = get_contact_state(
        env,
        layout=layout,
        active_joint_mask_by_env=active_joint_mask_by_env,
        ema_alpha=ema_alpha,
        force_threshold_N=force_threshold_N,
    )
    _, owner_bits = contact.owner_force_and_bits()
    return actor_owner_contact(owner_bits, mask)


def actor_joint_limits_term(
    env: ManagerBasedRLEnv,
    *,
    active_joint_mask_by_env: Sequence[Sequence[bool]],
    robot_name: str = "robot",
) -> torch.Tensor:
    r"""返回$[N,16,2]$ static normalized soft limits。"""

    robot = cast(Articulation, env.scene[robot_name])
    return actor_joint_limits(robot.data.soft_joint_pos_limits, _mask_tensor(env, active_joint_mask_by_env))


def actor_tip_contact_term(
    env: ManagerBasedRLEnv,
    *,
    active_joint_mask_by_env: Sequence[Sequence[bool]],
    layout: HeterogeneousContactLayout,
    ema_alpha: float = 0.5,
    force_threshold_N: float = 0.25,
) -> torch.Tensor:
    r"""返回TIP-only$[N,4,1]$ actor tactile bits。"""

    contact = get_contact_state(
        env,
        layout=layout,
        active_joint_mask_by_env=active_joint_mask_by_env,
        ema_alpha=ema_alpha,
        force_threshold_N=force_threshold_N,
    )
    return actor_tip_contact(contact.tip_bits, _mask_tensor(env, active_joint_mask_by_env))


def critic_joint_state_term(
    env: ManagerBasedRLEnv,
    *,
    active_joint_mask_by_env: Sequence[Sequence[bool]],
    action_name: str,
    robot_name: str = "robot",
) -> torch.Tensor:
    r"""返回privileged$[N,16,4]$ JOINT block。"""

    robot = cast(Articulation, env.scene[robot_name])
    action = _action_term(env, action_name)
    return critic_joint_state(
        robot.data.joint_pos,
        robot.data.joint_vel,
        action.current_targets,
        action.executed_actions,
        _mask_tensor(env, active_joint_mask_by_env),
    )


def critic_owner_contact_term(
    env: ManagerBasedRLEnv,
    *,
    active_joint_mask_by_env: Sequence[Sequence[bool]],
    layout: HeterogeneousContactLayout,
    ema_alpha: float = 0.5,
    force_threshold_N: float = 0.25,
) -> torch.Tensor:
    r"""返回all-owner$[N,21,2]$ privileged$[f_{EMA}\,N,c]$。"""

    contact = get_contact_state(
        env,
        layout=layout,
        active_joint_mask_by_env=active_joint_mask_by_env,
        ema_alpha=ema_alpha,
        force_threshold_N=force_threshold_N,
    )
    owner_force, owner_bits = contact.owner_force_and_bits()
    return torch.stack((owner_force, owner_bits.to(dtype=owner_force.dtype)), dim=-1)


def critic_object_term(
    env: ManagerBasedRLEnv,
    *,
    command_name: str,
    semantic_R_ha: Sequence[float],
) -> torch.Tensor:
    r"""返回hand-frame privileged object$[N,1,15]$。"""

    command = get_rotation_command(env, command_name)
    rotation = torch.tensor(semantic_R_ha, dtype=torch.float32, device=env.device).reshape(3, 3)
    return object_state_in_hand_frame(
        root_quat_wxyz=command.robot.data.root_quat_w,
        semantic_R_ha=rotation,
        object_pos_w=command.object.data.root_pos_w,
        object_quat_wxyz=command.object.data.root_quat_w,
        position_anchor_w=command.position_anchor_w,
        object_linear_velocity_w=command.object.data.root_lin_vel_w,
        object_angular_velocity_w=command.object.data.root_ang_vel_w,
    )


def critic_task_term(env: ManagerBasedRLEnv, *, command_name: str) -> torch.Tensor:
    r"""返回privileged task$[N,1,8]$，含signed net progress。"""

    command = get_rotation_command(env, command_name)
    return task_state(
        command.axis_h,
        command.goal_error_so3_h,
        command.net_rotation_rad,
        subgoal_angle_rad=float(command.cfg.subgoal_angle_rad),
    )


__all__ = [
    "actor_joint_current_term",
    "actor_joint_history_frame_term",
    "actor_joint_contact_frame_term",
    "actor_joint_limits_term",
    "actor_tip_contact_term",
    "actor_owner_contact_term",
    "critic_joint_state_term",
    "critic_object_term",
    "critic_owner_contact_term",
    "critic_task_term",
    "joint_valid",
    "owner_valid",
    "palm_valid",
    "tip_valid",
]
