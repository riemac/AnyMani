r"""GM tactile rotation 的语义 observation terms 与固定 52D/152D wire contract。

ObservationGroup 按 config 声明顺序拼接独立物理块。Actor 只含 deployment-available
proprioception、target、policy action 与 tip bits；object、goal、palm、finger non-tip、ADR
和 reward curriculum 只进入 central critic。文件末尾保留 composite helpers，供 runtime smoke
直接核对最终 wire；正式环境配置不再把整个 observation 藏在单一 MDP term 中。
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


def tactile_joint_position(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*"], preserve_order=True),
) -> torch.Tensor:
    r"""返回 canonical joint position $q_t$，形状 `[B,16]`，单位 rad。"""

    robot: Articulation = env.scene[robot_cfg.name]  # generated articulation canonical source
    return robot.data.joint_pos[:, robot_cfg.joint_ids]  # $q_t$，由 ObsTerm.scale 决定是否除以 $\pi$


def tactile_joint_velocity(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*"], preserve_order=True),
) -> torch.Tensor:
    r"""返回 canonical joint velocity $\dot q_t$，形状 `[B,16]`，单位 rad/s。"""

    robot: Articulation = env.scene[robot_cfg.name]  # critic privileged proprioception source
    return robot.data.joint_vel[:, robot_cfg.joint_ids]  # 保留真实单位，central RMS 再做统计标准化


def tactile_joint_target(env: ManagerBasedRLEnv, action_name: str = "hand_joint_pos") -> torch.Tensor:
    r"""返回 policy-step target buffer $u_t$，形状 `[B,16]`，单位 rad。"""

    current_targets, _ = _action_term_tensors(env, action_name)
    return current_targets  # actor/critic 共用同一 source；ObsTerm.scale 统一为 $1/\pi$


def tactile_last_policy_action(env: ManagerBasedRLEnv, action_name: str = "hand_joint_pos") -> torch.Tensor:
    r"""返回 wrapper-clamped raw policy action $a_{t-1}^{policy}$，形状 `[B,16]`。"""

    _, raw_actions = _action_term_tensors(env, action_name)
    return raw_actions  # 不读取 ADR noise/latency 后的 executed action，避免向 actor 泄漏 corruption


def tactile_object_task_state(
    env: ManagerBasedRLEnv,
    command_name: str,
    semantic_R_ha: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    r"""构造 15D object/goal privileged task state。

    顺序为 `[position_delta_h3, goal-relative-rot6d6, linear_velocity_h3,
    angular_velocity_h3]`。位置单位 m，速度单位 m/s 与 rad/s；旋转使用
    $R_g^{-1}R_o$ 的前两列 6D 表示。
    """

    command = ensure_post_physics_progress_updated(env, command_name)  # 当前 post-physics command snapshot
    robot: Articulation = env.scene[robot_cfg.name]  # `{a}->{w}` hand root pose
    object_asset: RigidObject = env.scene[object_cfg.name]  # object canonical pose/velocity source
    R_ha = torch.tensor(semantic_R_ha, dtype=torch.float32, device=env.device).reshape(3, 3)  # $R_{ha}$

    # 先从 world 变换到 raw asset `{a}`，再应用静态语义标定得到 hand frame `{h}`。
    position_delta_w = object_asset.data.root_pos_w - command.position_anchor_w  # `[B,3]`，m
    position_delta_a = math_utils.quat_apply_inverse(robot.data.root_quat_w, position_delta_w)  # $R_{aw}\Delta p^w$
    position_delta_h = position_delta_a @ R_ha.T  # row-vector form of $R_{ha}\Delta p^a$
    linear_velocity_a = math_utils.quat_apply_inverse(robot.data.root_quat_w, object_asset.data.root_lin_vel_w)
    angular_velocity_a = math_utils.quat_apply_inverse(robot.data.root_quat_w, object_asset.data.root_ang_vel_w)
    linear_velocity_h = linear_velocity_a @ R_ha.T  # `[B,3]`，m/s
    angular_velocity_h = angular_velocity_a @ R_ha.T  # `[B,3]`，rad/s

    # $R_{go}=R_{wg}^{T}R_{wo}$；取前两列得到连续 6D orientation feature。
    current_rot_w = math_utils.matrix_from_quat(object_asset.data.root_quat_w)  # `[B,3,3]`
    goal_rot_w = math_utils.matrix_from_quat(command.goal_quat_w)  # `[B,3,3]`
    relative_rot = goal_rot_w.transpose(-1, -2) @ current_rot_w  # $R_g^{-1}R_o$
    relative_rot6d = torch.cat((relative_rot[:, :, 0], relative_rot[:, :, 1]), dim=-1)  # `[B,6]`
    return torch.cat((position_delta_h, relative_rot6d, linear_velocity_h, angular_velocity_h), dim=-1)  # `[B,15]`


def tactile_reward_release_coefficient(
    env: ManagerBasedRLEnv,
    reward_lambda_attr_name: str = "_gm_reward_curriculum_lambda",
) -> torch.Tensor:
    r"""返回每个 env 的 reward-release 系数 $\lambda_{rew}\in[0,1]$，形状 `[B,1]`。"""

    reward_lambda = getattr(env, reward_lambda_attr_name, 0.0)  # scalar 或 `[B]` runtime curriculum state
    coefficient = torch.as_tensor(reward_lambda, dtype=torch.float32, device=env.device)
    if coefficient.ndim == 0:
        coefficient = coefficient.expand(env.num_envs)  # global curriculum scalar -> per-env view
    return coefficient.reshape(env.num_envs, 1)  # `[B,1]`


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

    joint_pos = tactile_joint_position(env, robot_cfg) / torch.pi  # actor fixed physical scale
    current_targets = tactile_joint_target(env, action_name) / torch.pi  # 与 critic shared field 同口径
    raw_actions = tactile_last_policy_action(env, action_name)  # 无量纲 policy-facing action
    contact = get_tactile_contact_state(
        env,
        fingertip_sensor_names,
        finger_non_tip_sensor_names,
        palm_sensor_name,
        ema_alpha,
        force_threshold,
    )
    frame = torch.cat(
        (joint_pos, current_targets, raw_actions, contact.tip_bits.float()), dim=-1
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

    joint_pos = tactile_joint_position(env, robot_cfg) / torch.pi  # shared actor/critic $q/\pi$
    joint_velocity = tactile_joint_velocity(env, robot_cfg)  # privileged raw rad/s
    current_targets = tactile_joint_target(env, action_name) / torch.pi  # shared actor/critic $u/\pi$
    raw_actions = tactile_last_policy_action(env, action_name)  # shared raw policy action
    object_task_state = tactile_object_task_state(
        env,
        command_name=command_name,
        semantic_R_ha=semantic_R_ha,
        robot_cfg=robot_cfg,
        object_cfg=object_cfg,
    )  # `[B,15]`

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
            joint_pos,
            joint_velocity,
            current_targets,
            raw_actions,
            object_task_state,
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
    reward_lambda_tensor = tactile_reward_release_coefficient(env, reward_lambda_attr_name)
    critic_state = torch.cat((task_state, adr_state, reward_lambda_tensor), dim=-1)
    if critic_state.shape[-1] != TACTILE_CRITIC_STATE_DIM:
        raise RuntimeError(f"Tactile central critic state must be 152D, got shape {tuple(critic_state.shape)}.")
    return critic_state


def _action_term_tensors(env: ManagerBasedRLEnv, action_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    r"""读取 action term 的 target/raw-policy tensors，并在 schema 不兼容时 fail fast。"""

    action_term = env.action_manager.get_term(action_name)
    current_targets = getattr(action_term, "current_targets", None)
    raw_actions = getattr(action_term, "raw_actions", None)
    if not isinstance(current_targets, torch.Tensor) or not isinstance(raw_actions, torch.Tensor):
        raise RuntimeError(
            f"Action term '{action_name}' must expose tensor current_targets/raw_actions for tactile observations."
        )
    return current_targets, raw_actions


__all__ = [
    "TACTILE_ACTOR_FRAME_DIM",
    "TACTILE_CRITIC_STATE_DIM",
    "TACTILE_PRIVILEGED_TASK_DIM",
    "tactile_joint_position",
    "tactile_joint_target",
    "tactile_joint_velocity",
    "tactile_last_policy_action",
    "tactile_object_task_state",
    "tactile_reward_release_coefficient",
    "tactile_rotation_critic_state",
    "tactile_rotation_policy_frame",
    "tactile_rotation_privileged_task_state",
]
