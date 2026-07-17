r"""GM tactile rotation 的固定 52D actor frame 与 152D privileged critic state。

Composite functions 在一个位置锁住 feature 顺序，避免 ObservationGroup 字段重排造成 checkpoint
语义漂移。Actor 只含 deployment-available proprio/target/action/tip bits；object、goal、palm、
finger non-tip、ADR 与 reward curriculum 只进入 central critic。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from ..adr_state import gm_adr_state_observation
from ..commands.tactile_rotation_command import ensure_post_physics_progress_updated
from ..tactile_contact_state import get_tactile_contact_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


TACTILE_ACTOR_FRAME_DIM = 52
TACTILE_PRIVILEGED_TASK_DIM = 103
TACTILE_CRITIC_STATE_DIM = 152


def tactile_rotation_policy_frame(
    env: ManagerBasedRLEnv,
    fingertip_sensor_names: Sequence[str],
    finger_non_tip_sensor_names: Sequence[str],
    palm_sensor_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*"], preserve_order=True),
    action_name: str = "hand_joint_pos",
    ema_alpha: float = 0.5,
    force_threshold: float = 0.25,
) -> torch.Tensor:
    r"""构造严格按 canonical joint/finger order 排列的单帧 52D actor observation。

    $$x_t=[q_t/\pi,\ u_t/\pi,\ a_{t-1}^{policy},\ c_t^{tip}].$$

    `raw_actions` 是 wrapper clamp 后送入 ActionManager 的无量纲 policy command；不读取
    `executed_actions` 或 `processed_actions`，因此 ADR noise/latency 不泄漏给 actor。
    """

    robot: Articulation = env.scene[robot_cfg.name]
    joint_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]
    action_term = env.action_manager.get_term(action_name)
    current_targets = getattr(action_term, "current_targets", None)
    raw_actions = getattr(action_term, "raw_actions", None)
    if not isinstance(current_targets, torch.Tensor) or not isinstance(raw_actions, torch.Tensor):
        raise RuntimeError(
            f"Action term '{action_name}' must expose tensor current_targets/raw_actions for tactile actor state."
        )
    contact = get_tactile_contact_state(
        env,
        fingertip_sensor_names,
        finger_non_tip_sensor_names,
        palm_sensor_name,
        ema_alpha,
        force_threshold,
    )
    frame = torch.cat(
        (joint_pos / torch.pi, current_targets / torch.pi, raw_actions, contact.tip_bits.float()), dim=-1
    )
    if frame.shape[-1] != TACTILE_ACTOR_FRAME_DIM:
        raise RuntimeError(
            "Tactile actor frame must be 52D = 16 q + 16 target + 16 last policy action + 4 tip bits; "
            f"got shape {tuple(frame.shape)}."
        )
    return frame


def tactile_rotation_privileged_task_state(
    env: ManagerBasedRLEnv,
    command_name: str,
    fingertip_sensor_names: Sequence[str],
    finger_non_tip_sensor_names: Sequence[str],
    palm_sensor_name: str,
    semantic_R_ha: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*"], preserve_order=True),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    action_name: str = "hand_joint_pos",
    ema_alpha: float = 0.5,
    force_threshold: float = 0.25,
) -> torch.Tensor:
    r"""构造不含 ADR/curriculum 的 103D privileged task/contact state。

    顺序固定为 `[q16,dq16,u16,a16,pos3,goal-relative-rot6d-6,v_h3,omega_h3,
    tip-force4,palm-force1,finger-non-tip-bits19]`。
    """

    command = ensure_post_physics_progress_updated(env, command_name)
    robot: Articulation = env.scene[robot_cfg.name]
    object_asset: RigidObject = env.scene[object_cfg.name]
    action_term = env.action_manager.get_term(action_name)
    current_targets = getattr(action_term, "current_targets", None)
    raw_actions = getattr(action_term, "raw_actions", None)
    if not isinstance(current_targets, torch.Tensor) or not isinstance(raw_actions, torch.Tensor):
        raise RuntimeError(f"Action term '{action_name}' lacks privileged current_targets/raw_actions tensors.")

    R_ha = torch.tensor(semantic_R_ha, dtype=torch.float32, device=env.device).reshape(3, 3)
    position_delta_w = object_asset.data.root_pos_w - command.position_anchor_w
    position_delta_a = math_utils.quat_apply_inverse(robot.data.root_quat_w, position_delta_w)
    position_delta_h = position_delta_a @ R_ha.T
    linear_velocity_a = math_utils.quat_apply_inverse(robot.data.root_quat_w, object_asset.data.root_lin_vel_w)
    angular_velocity_a = math_utils.quat_apply_inverse(robot.data.root_quat_w, object_asset.data.root_ang_vel_w)
    linear_velocity_h = linear_velocity_a @ R_ha.T
    angular_velocity_h = angular_velocity_a @ R_ha.T

    current_rot_w = math_utils.matrix_from_quat(object_asset.data.root_quat_w)
    goal_rot_w = math_utils.matrix_from_quat(command.goal_quat_w)
    relative_rot = goal_rot_w.transpose(-1, -2) @ current_rot_w  # $R_g^{-1}R_o$
    relative_rot6d = torch.cat((relative_rot[:, :, 0], relative_rot[:, :, 1]), dim=-1)

    contact = get_tactile_contact_state(
        env,
        fingertip_sensor_names,
        finger_non_tip_sensor_names,
        palm_sensor_name,
        ema_alpha,
        force_threshold,
    )
    state = torch.cat(
        (
            robot.data.joint_pos[:, robot_cfg.joint_ids],
            robot.data.joint_vel[:, robot_cfg.joint_ids],
            current_targets,
            raw_actions,
            position_delta_h,
            relative_rot6d,
            linear_velocity_h,
            angular_velocity_h,
            contact.tip_force_ema,
            contact.palm_force_ema,
            contact.finger_non_tip_bits.float(),
        ),
        dim=-1,
    )
    if state.shape[-1] != TACTILE_PRIVILEGED_TASK_DIM:
        raise RuntimeError(
            "Tactile privileged task state must be 103D; "
            f"got shape {tuple(state.shape)}. Check 16-DOF/4-tip/19-finger-non-tip schema."
        )
    return state


def tactile_rotation_critic_state(
    env: ManagerBasedRLEnv,
    command_name: str,
    fingertip_sensor_names: Sequence[str],
    finger_non_tip_sensor_names: Sequence[str],
    palm_sensor_name: str,
    semantic_R_ha: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*"], preserve_order=True),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    action_name: str = "hand_joint_pos",
    ema_alpha: float = 0.5,
    force_threshold: float = 0.25,
    reward_lambda_attr_name: str = "_gm_reward_curriculum_lambda",
) -> torch.Tensor:
    r"""拼接 103D task/contact、48D actual ADR 与 1D reward release，得到 152D `states`。"""

    task_state = tactile_rotation_privileged_task_state(
        env=env,
        command_name=command_name,
        fingertip_sensor_names=fingertip_sensor_names,
        finger_non_tip_sensor_names=finger_non_tip_sensor_names,
        palm_sensor_name=palm_sensor_name,
        semantic_R_ha=semantic_R_ha,
        robot_cfg=robot_cfg,
        object_cfg=object_cfg,
        action_name=action_name,
        ema_alpha=ema_alpha,
        force_threshold=force_threshold,
    )
    adr_state = gm_adr_state_observation(env, action_dim=16)
    reward_lambda = getattr(env, reward_lambda_attr_name, 0.0)
    reward_lambda_tensor = torch.as_tensor(reward_lambda, dtype=torch.float32, device=env.device)
    if reward_lambda_tensor.ndim == 0:
        reward_lambda_tensor = reward_lambda_tensor.expand(env.num_envs)
    reward_lambda_tensor = reward_lambda_tensor.reshape(env.num_envs, 1)
    critic_state = torch.cat((task_state, adr_state, reward_lambda_tensor), dim=-1)
    if critic_state.shape[-1] != TACTILE_CRITIC_STATE_DIM:
        raise RuntimeError(f"Tactile central critic state must be 152D, got shape {tuple(critic_state.shape)}.")
    return critic_state


__all__ = [
    "TACTILE_ACTOR_FRAME_DIM",
    "TACTILE_CRITIC_STATE_DIM",
    "TACTILE_PRIVILEGED_TASK_DIM",
    "tactile_rotation_critic_state",
    "tactile_rotation_policy_frame",
    "tactile_rotation_privileged_task_state",
]
