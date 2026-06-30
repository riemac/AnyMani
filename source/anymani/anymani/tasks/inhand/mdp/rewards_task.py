# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""LeapHand与任务相关的奖励函数。主要由状态决定，衡量当前状态与目标状态的距离。它是稀疏的（如成功/失败）或稠密的（如距离度量）。

简言之，该奖励文件决定 “做什么”，特定于任务（Task-Specific）。

"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

# from isaaclab.markers import VisualizationMarkers
# from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

###
# 旋转和重定向奖励
###


def _resolve_goal_pose_from_command_term(
    env: ManagerBasedRLEnv, command_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """从命令项解析 goal pose（pos_e, quat_w）。

    背景：RelativeSO3Command 的 `command()` 输出是 3 维 rotvec（phi_ref），不再是 7D pose。
    但任务侧的 success/track 奖励仍需要一个 *内部* 目标姿态用于计算误差。

    兼容策略：
        1) 优先读取 command term 的内部 buffer（pos_command_e / quat_command_w）。
        2) 否则回退到旧接口：command tensor 的 (pos, quat) 拼接形式。

    Returns:
        goal_pos_e: (num_envs, 3)
        goal_quat_w: (num_envs, 4)
    """

    term = env.command_manager.get_term(command_name)

    goal_pos_e = getattr(term, "pos_command_e", None)
    goal_quat_w = getattr(term, "quat_command_w", None)

    if isinstance(goal_pos_e, torch.Tensor) and isinstance(goal_quat_w, torch.Tensor):
        return goal_pos_e, goal_quat_w

    # fallback to legacy pose command: (pos_e, quat_w)
    cmd = env.command_manager.get_command(command_name)
    if not (isinstance(cmd, torch.Tensor) and cmd.shape[-1] >= 7):
        raise RuntimeError(
            f"Cannot resolve goal pose from command '{command_name}'. Expected term buffers (pos_command_e/quat_command_w) "
            f"or a pose-like command tensor with dim>=7. Got: {type(cmd)} {getattr(cmd, 'shape', None)}"
        )
    return cmd[:, :3], cmd[:, -4:]


def compute_official_leap_reward(
    goal_dist: torch.Tensor,
    rot_dist: torch.Tensor,
    executed_actions: torch.Tensor,
    pose_diff_penalty: torch.Tensor,
    torque_penalty: torch.Tensor,
    object_angvel_z: torch.Tensor,
    dist_reward_scale: float,
    rot_reward_scale: float,
    rot_eps: float,
    action_penalty_scale: float,
    pose_diff_penalty_scale: float,
    torque_penalty_scale: float,
    success_tolerance: float,
    position_success_threshold: float,
    reach_goal_bonus: float,
    fall_dist: float,
    fall_penalty: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""计算官方 LEAP DirectRLEnv 的单步 reward。

    官方核心 reward 为：
    $$
    r = -10\|p_o-p_g\|_2 + \frac{1}{|\theta| + 0.1}
        - 0.0002\|a\|_2^2 - 0.3\|q^{target} - q^{pregrasp}\|_2^2
        + 250\cdot\mathbf{1}_{success}
        - 10\cdot\mathbf{1}_{fall}
        + \mathbf{1}_{0.25 < \omega_z < 1.5}.
    $$

    Args:
        goal_dist: 物体位置与目标位置的环境系距离 ``[N]``。
        rot_dist: 物体姿态与目标姿态的角距离 ``[N]``，单位 rad。
        executed_actions: 已执行的规范化动作 $a_t^{exec}$，形状 ``[N, J]``。
        pose_diff_penalty: 当前 target buffer 相对 pregrasp 的二范数平方 ``[N]``。
        torque_penalty: 当前关节力矩平方和 ``[N]``。
        object_angvel_z: 物体世界系 z 轴角速度 ``[N]``。

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
        - 单步 reward ``[N]``；
        - success mask ``[N]``，表示本步是否完成一个小目标。
    """

    dist_rew = goal_dist * dist_reward_scale
    rot_rew = rot_reward_scale / (torch.abs(rot_dist) + rot_eps)
    action_penalty = torch.sum(executed_actions**2, dim=-1)

    reward = (
        dist_rew
        + rot_rew
        + action_penalty * action_penalty_scale
        + pose_diff_penalty * pose_diff_penalty_scale
        + torque_penalty * torque_penalty_scale
    )

    success_mask = (rot_dist <= success_tolerance) & (goal_dist <= position_success_threshold)
    reward = torch.where(success_mask, reward + reach_goal_bonus, reward)

    z_spin_bonus = (object_angvel_z > 0.25) & (object_angvel_z < 1.5)
    reward = torch.where(z_spin_bonus, reward + 1.0, reward)

    fall_mask = goal_dist >= fall_dist
    reward = torch.where(fall_mask, reward + fall_penalty, reward)
    return reward, success_mask


class OfficialLeapReward(ManagerTermBase):
    r"""官方 LEAP reward 与 reset-time 日志项的 ManagerBased 等价封装。

    这个 term 的职责有两层：

    1. 在 `RewardManager.compute()` 中返回与官方 DirectRLEnv **每个 policy step 等价**的 reward；
    2. 在 `reset()` 中把官方 TensorBoard 同名日志写入 `env.extras["log"]`，以便 AnyMani 的
       rl_games observer 仍生成 `Episode/consecutive_successes`、`Episode/adr_criteria` 等官方标量。

    注意 ManagerBased `RewardManager` 会自动乘 `dt`，因此为了恢复官方 reward 数值，
    本 term 返回的是：
    $$
    \frac{r_t^{official}}{\Delta t}.
    $$
    随后 `RewardManager` 再乘 `\Delta t`，净效果正好等于官方 DirectRLEnv 的单步 reward。
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._action_term_name = cfg.params.get("action_term_name", "hand_joint_pos")
        self._command_name = cfg.params.get("command_name", "goal_pose")
        self._object_cfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        self._robot_cfg = cfg.params.get("robot_cfg", SceneEntityCfg("robot"))
        self._dist_reward_scale = float(cfg.params.get("dist_reward_scale", -10.0))
        self._rot_reward_scale = float(cfg.params.get("rot_reward_scale", 1.0))
        self._rot_eps = float(cfg.params.get("rot_eps", 0.1))
        self._action_penalty_scale = float(cfg.params.get("action_penalty_scale", -0.0002))
        self._pose_diff_penalty_scale = float(cfg.params.get("pose_diff_penalty_scale", -0.3))
        self._torque_penalty_scale = float(cfg.params.get("torque_penalty_scale", -0.0))
        self._success_tolerance = float(cfg.params.get("success_tolerance", 0.2))
        self._position_success_threshold = float(cfg.params.get("position_success_threshold", 0.025))
        self._reach_goal_bonus = float(cfg.params.get("reach_goal_bonus", 250.0))
        self._fall_dist = float(cfg.params.get("fall_dist", 0.07))
        self._fall_penalty = float(cfg.params.get("fall_penalty", -10.0))
        self._z_rotation_steps = int(cfg.params.get("z_rotation_steps", 16))

    def _compute_state(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        object_asset: RigidObject = self._env.scene[self._object_cfg.name]
        robot_asset: Articulation = self._env.scene[self._robot_cfg.name]
        action_term = self._env.action_manager.get_term(self._action_term_name)
        goal_pos_e, goal_quat_w = _resolve_goal_pose_from_command_term(self._env, self._command_name)

        object_pos_e = object_asset.data.root_pos_w - self._env.scene.env_origins
        goal_dist = torch.norm(object_pos_e - goal_pos_e, p=2, dim=-1)
        rot_dist = math_utils.quat_error_magnitude(goal_quat_w, object_asset.data.root_quat_w)
        pose_diff_penalty = torch.sum((action_term.current_targets - action_term.pregrasp_targets) ** 2, dim=-1)
        torque_penalty = torch.sum(robot_asset.data.computed_torque**2, dim=-1)
        object_angvel_z = object_asset.data.root_ang_vel_w[:, 2]
        return goal_dist, rot_dist, action_term.executed_actions, pose_diff_penalty, torque_penalty, object_angvel_z

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        action_term_name: str = "hand_joint_pos",
        command_name: str = "goal_pose",
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        dist_reward_scale: float = -10.0,
        rot_reward_scale: float = 1.0,
        rot_eps: float = 0.1,
        action_penalty_scale: float = -0.0002,
        pose_diff_penalty_scale: float = -0.3,
        torque_penalty_scale: float = -0.0,
        success_tolerance: float = 0.2,
        position_success_threshold: float = 0.025,
        reach_goal_bonus: float = 250.0,
        fall_dist: float = 0.07,
        fall_penalty: float = -10.0,
        z_rotation_steps: int = 16,
    ) -> torch.Tensor:
        goal_dist, rot_dist, executed_actions, pose_diff_penalty, torque_penalty, object_angvel_z = (
            self._compute_state()
        )
        reward, _ = compute_official_leap_reward(
            goal_dist=goal_dist,
            rot_dist=rot_dist,
            executed_actions=executed_actions,
            pose_diff_penalty=pose_diff_penalty,
            torque_penalty=torque_penalty,
            object_angvel_z=object_angvel_z,
            dist_reward_scale=dist_reward_scale,
            rot_reward_scale=rot_reward_scale,
            rot_eps=rot_eps,
            action_penalty_scale=action_penalty_scale,
            pose_diff_penalty_scale=pose_diff_penalty_scale,
            torque_penalty_scale=torque_penalty_scale,
            success_tolerance=success_tolerance,
            position_success_threshold=position_success_threshold,
            reach_goal_bonus=reach_goal_bonus,
            fall_dist=fall_dist,
            fall_penalty=fall_penalty,
        )
        return reward / float(self._env.step_dt)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)

        object_asset: RigidObject = self._env.scene[self._object_cfg.name]
        command_term = self._env.command_manager.get_term(self._command_name)
        _, _, _, pose_diff_penalty, torque_penalty, _ = self._compute_state()

        log = getattr(self._env, "extras", {}).get("log")
        if isinstance(log, dict):
            log["consecutive_successes"] = torch.mean(command_term.success_counter[env_ids]).item() / float(
                self._z_rotation_steps
            )
            log["pose_diff_penalty"] = torch.mean(pose_diff_penalty[env_ids]).item()
            log["torque_info"] = torch.mean(torque_penalty[env_ids]).item()
            log["object_linvel"] = torch.norm(object_asset.data.root_lin_vel_w[env_ids], p=1, dim=-1).mean().item()
            log["roll"] = object_asset.data.root_ang_vel_w[env_ids, 0].mean().item()
            log["pitch"] = object_asset.data.root_ang_vel_w[env_ids, 1].mean().item()
            log["yaw"] = object_asset.data.root_ang_vel_w[env_ids, 2].mean().item()
            log["num_adr_increases"] = float(getattr(self._env, "leap_adr_increment", 0))
            log["adr_criteria"] = float(getattr(self._env, "leap_adr_criteria", 0.0))
            if hasattr(self._env, "leap_adr_episode_lengths"):
                lengths_s = self._env.leap_adr_episode_lengths.float() * float(self._env.step_dt)
                log["avg_episode_length_s"] = lengths_s.mean().item()
                log["min_episode_length_s"] = lengths_s.min().item()
                log["max_episode_length_s"] = lengths_s.max().item()


def track_orientation_inv_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "goal_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    rot_eps: float = 1e-3,
) -> torch.Tensor:
    """方向跟踪奖励 - 使用方向误差的倒数。

    奖励为物体当前姿态与目标姿态之间的方向误差倒数，误差越小奖励越大。

    Args:
        env: ManagerBasedRLEnv - 环境实例
        command_name: str - 命令项名称（用于获取目标姿态）
        object_cfg: SceneEntityCfg - 物体资产配置
        rot_eps: float - 防止除零的小常数（默认 1e-3）

    Returns:
        (num_envs,) 张量，方向跟踪奖励

    NOTE:
        - 奖励公式：R = 1 / (eps + |dtheta|)
    """
    # 获取物体资产
    asset: RigidObject = env.scene[object_cfg.name]

    # 获取目标姿态（优先使用 command term 内部 buffer；兼容旧 pose command）
    _, goal_quat_w = _resolve_goal_pose_from_command_term(env, command_name)

    # 计算方向误差（轴角表示的 L2 范数）
    # q_goal ⊖ q_current^(-1) -> 轴角对 -> 角误差（L2范数，单位轴化1，剩下角度）
    dtheta = math_utils.quat_error_magnitude(goal_quat_w, asset.data.root_quat_w)

    # 计算奖励：误差越小，奖励越大
    reward = 1.0 / (dtheta + rot_eps)

    return reward


def success_bonus(
    env: ManagerBasedRLEnv,
    command_name: str = "goal_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    orientation_threshold: float = 0.2,
    position_threshold: float = 0.025,
) -> torch.Tensor:
    """成功奖励 - 达到目标旋转时的稀疏奖励

    Args:
        env: ManagerBasedRLEnv - 环境实例
        command_name: str - 命令项名称（用于获取目标姿态）
        object_cfg: SceneEntityCfg - 物体资产配置
        threshold: float - 成功容忍度（弧度）

    Returns:
        (num_envs,) 张量，成功奖励
    """
    # 获取物体资产
    asset: RigidObject = env.scene[object_cfg.name]
    # act = env.action_manager.get_term(action_name)
    # act

    # 获取目标姿态/位置（优先使用 command term 内部 buffer；兼容旧 pose command）
    goal_pos_e, goal_quat_w = _resolve_goal_pose_from_command_term(env, command_name)

    # 计算方向误差（轴角表示的 L2 范数）
    dtheta = math_utils.quat_error_magnitude(goal_quat_w, asset.data.root_quat_w)

    # 计算位置误差（目标位置在环境坐标系下）
    object_pos_e = asset.data.root_pos_w - env.scene.env_origins
    goal_dist = torch.norm(object_pos_e - goal_pos_e, p=2, dim=-1)

    # 计算成功奖励：姿态和位置双重满足
    success_reward = torch.where(
        (dtheta <= orientation_threshold) & (goal_dist <= position_threshold),
        torch.ones_like(dtheta),
        torch.zeros_like(dtheta),
    )

    return success_reward


def track_pos_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "goal_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """位置跟踪惩罚：物体位置与目标位置的 L2 距离。

    该项用于“位置约束/别漂移”：
    - 目标位置来自命令项（RelativeSO3Command / ContinuousRotationCommand 等）的 pos_command_e；
    - 物体当前位置取 root_pos_w 并减去 env_origins，得到环境坐标系 {e} 下的位置。

    Note:
        IsaacLab 官方 inhand 任务中也提供了类似的 track_pos_l2（默认在配置里注释掉）。
        AnyMani 这里保留同名接口，便于直接对齐上游配置风格。

    Returns:
        (num_envs,) 张量，位置误差的 L2 范数（越大越“坏”，通常配负权重）。
    """

    asset: RigidObject = env.scene[object_cfg.name]

    goal_pos_e, _ = _resolve_goal_pose_from_command_term(env, command_name)
    object_pos_e = asset.data.root_pos_w - env.scene.env_origins
    return torch.norm(object_pos_e - goal_pos_e, p=2, dim=-1)


def goal_position_distance(
    env: ManagerBasedRLEnv,
    command_name: str = "goal_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """兼容接口：goal_position_distance 等价于 track_pos_l2。

    历史原因：部分环境配置（例如 leaphand_round 的早期版本）使用了该函数名。
    为避免配置失效，这里提供别名。
    """

    return track_pos_l2(env=env, command_name=command_name, object_cfg=object_cfg)


def fall_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "goal_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    fall_distance: float = 0.07,
) -> torch.Tensor:
    """计算掉落惩罚

    Args:
        env: 环境实例
        asset_cfg: 物体资产配置
        fall_distance: 掉落距离阈值

    Returns:
        掉落惩罚 (num_envs,)
    """
    # 获取物体资产
    asset: RigidObject = env.scene[object_cfg.name]

    goal_pos_e, _ = _resolve_goal_pose_from_command_term(env, command_name)
    object_pos_e = asset.data.root_pos_w - env.scene.env_origins
    distance = torch.norm(object_pos_e - goal_pos_e, p=2, dim=-1)

    return torch.where(distance > fall_distance, torch.ones_like(distance), torch.zeros_like(distance))


def track_rotation_velocity_alignment(
    env: ManagerBasedRLEnv,
    command_name: str = "goal_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    eps: float = 1e-6,
    clamp_negative: bool = True,
) -> torch.Tensor:
    """rolling_goal 推荐奖励：角速度与指令轴对齐 + 幅值。

    rolling_goal 下目标会随当前姿态滚动更新，基于姿态误差的 tracking reward 将退化为常数。
    因此用“执行旋转”的信号更合适：鼓励物体角速度沿指令轴方向旋转，并鼓励一定的角速度幅值。

    计算：
        - 指令轴：u_ref = phi_ref / ||phi_ref||
        - 角速度方向：u_omega = omega / ||omega||
        - reward = ||omega|| * max(0, <u_ref, u_omega>)   (默认)

    Args:
        env: 环境
        command_name: 命令项名称（RelativeSO3Command）
        object_cfg: 物体资产
        eps: 数值稳定
        clamp_negative: 是否将反向旋转的 dot 裁剪为 0

    Returns:
        (num_envs,) reward
    """

    asset: RigidObject = env.scene[object_cfg.name]
    omega_w = asset.data.root_ang_vel_w  # (num_envs, 3)
    omega_norm = torch.linalg.norm(omega_w, dim=-1)
    omega_hat = omega_w / (omega_norm.unsqueeze(-1) + eps)

    term = env.command_manager.get_term(command_name)
    phi = getattr(term, "phi_ref_e", None)
    if not (isinstance(phi, torch.Tensor) and phi.shape[-1] == 3):
        # fallback: command tensor itself
        cmd = env.command_manager.get_command(command_name)
        phi = (cmd[:, 3:6] if cmd.shape[-1] == 6 else cmd) if isinstance(cmd, torch.Tensor) else cmd

    if not (isinstance(phi, torch.Tensor) and phi.shape[-1] == 3):
        raise RuntimeError(
            f"track_rotation_velocity_alignment expects command '{command_name}' to provide phi_ref_e as (num_envs,3) "
            f"or a command tensor with dim=3 (phi_ref_e) / dim=6 (pos_e, phi_ref_e). Got: {type(phi)} {getattr(phi, 'shape', None)}"
        )

    phi_norm = torch.linalg.norm(phi, dim=-1)
    phi_hat = phi / (phi_norm.unsqueeze(-1) + eps)

    dot = torch.sum(phi_hat * omega_hat, dim=-1)
    if clamp_negative:
        dot = torch.clamp(dot, min=0.0)
    return omega_norm * dot


###
#  参考LEAP_Hand_Isaac_Lab奖励项
###
def pose_diff_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    natural_pose: dict[str, float] | None = None,
) -> torch.Tensor:
    """计算手部姿态偏差惩罚 - 鼓励保持接近人手的自然姿态

    关键点：**通过关节名对齐自然姿态和当前姿态**，避免依赖“隐含的关节索引顺序”。

    这样既保证了目标姿态和当前姿态在关节维度上的一一对应，也与动作空间中
    使用 ``preserve_order=True`` 时的关节选择逻辑兼容。

    Args:
        env: 环境实例。
        asset_cfg: 机器人资产配置。
        natural_pose: 以关节名为键的自然姿态字典；若为 ``None`` 则使用默认的 LeapHand 自然姿态。

    Returns:
        姿态偏差惩罚 (num_envs,)。
    """

    # 获取机器人资产
    asset: Articulation = env.scene[asset_cfg.name]

    # 1) 定义 LeapHand 的自然姿态（以「关节名 -> 目标角度」的形式），
    #    数值与 InHandSceneCfg.robot.init_state.joint_pos 完全一致。
    if natural_pose is None:
        natural_pose = {
            "a_1": 0.000,
            "a_12": 0.500,
            "a_5": 0.000,
            "a_9": 0.000,
            "a_0": -0.750,
            "a_13": 1.300,
            "a_4": 0.000,
            "a_8": 0.750,
            "a_2": 1.750,
            "a_14": 1.500,
            "a_6": 1.750,
            "a_10": 1.750,
            "a_3": 0.000,
            "a_15": 1.000,
            "a_7": 0.000,
            "a_11": 0.000,
        }

    # 2) 缓存：按照 Articulation 中 **关节名解析结果** 的顺序，构建
    #    - 关节索引 joint_ids
    #    - 自然姿态向量 natural_joint_pos
    #
    #    这里通过名字解析来对齐，而不是假定某个固定的关节索引顺序，
    #    这样可以避免与动作空间（尤其是使用 preserve_order=True 的 se3 动作项）
    #    之间出现隐式的索引错位。
    if not hasattr(env, "_leaphand_natural_joint_ids") or not hasattr(env, "_leaphand_natural_joint_pos"):
        # 使用关节名列表作为解析顺序；在 Python 3.7+ 中 dict 保证插入顺序，
        # 因此 natural_pose 的键顺序是显式且可控的。
        natural_joint_names = list(natural_pose.keys())
        joint_ids, joint_names = asset.find_joints(natural_joint_names, preserve_order=True)

        # 根据解析后的 joint_names 顺序生成自然姿态向量
        natural_joint_list = [float(natural_pose[name]) for name in joint_names]

        # 保存到 env 上以避免每步重复解析
        env._leaphand_natural_joint_ids = torch.as_tensor(joint_ids, device=env.device, dtype=torch.long)
        env._leaphand_natural_joint_pos = torch.tensor(
            natural_joint_list, device=env.device, dtype=torch.float32
        ).unsqueeze(0)  # 形状: (1, num_natural_joints)

    joint_ids = env._leaphand_natural_joint_ids
    # 扩展到所有 env：形状 (num_envs, num_natural_joints)
    natural_joint_pos = env._leaphand_natural_joint_pos.expand(env.num_envs, -1)

    # 3) 计算当前关节位置与自然姿态的差异（仅对配置了自然姿态的那几个关节）
    current_joint_pos = asset.data.joint_pos[:, joint_ids]
    pose_diff = current_joint_pos - natural_joint_pos

    # 4) 计算 L2 平方惩罚：对每个 env 在关节维度求和
    pose_diff_penalty = torch.sum(pose_diff**2, dim=-1)

    return pose_diff_penalty
