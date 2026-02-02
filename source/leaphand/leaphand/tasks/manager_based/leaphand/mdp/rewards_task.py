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

import torch
import isaaclab.utils.math as math_utils

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
# from isaaclab.markers import VisualizationMarkers
# from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

###
# 旋转和重定向奖励
###
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

    # 获取目标姿态（从命令管理器）
    # goal_pose 通常是 (pos, quat)，我们取后4维作为目标四元数
    goal_pose = env.command_manager.get_command(command_name)
    goal_quat_w = goal_pose[:, -4:]  # (num_envs, 4) in (w, x, y, z)

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

    # 获取目标姿态（从命令管理器）
    goal_pose = env.command_manager.get_command(command_name)
    goal_quat_w = goal_pose[:, -4:]  # (num_envs, 4) in (w, x, y, z)

    # 计算方向误差（轴角表示的 L2 范数）
    dtheta = math_utils.quat_error_magnitude(goal_quat_w, asset.data.root_quat_w)

    # 计算位置误差（目标位置在环境坐标系下）
    goal_pos = goal_pose[:, :3]
    object_pos = asset.data.root_pos_w - env.scene.env_origins
    goal_dist = torch.norm(object_pos - goal_pos, p=2, dim=-1)

    # 计算成功奖励：姿态和位置双重满足
    success_reward = torch.where(
        (dtheta <= orientation_threshold) & (goal_dist <= position_threshold),
        torch.ones_like(dtheta),
        torch.zeros_like(dtheta),
    )

    return success_reward

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

    goal_pose = env.command_manager.get_command(command_name)
    goal_pos = goal_pose[:, :3]

    object_pos = asset.data.root_pos_w - env.scene.env_origins

    distance = torch.norm(object_pos - goal_pos, p=2, dim=-1)

    return torch.where(distance > fall_distance, torch.ones_like(distance), torch.zeros_like(distance))


def goal_position_distance(
    env: ManagerBasedRLEnv,
    command_name: str = "goal_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """物体与目标位置之间的欧氏距离。"""

    asset: RigidObject = env.scene[object_cfg.name]

    goal_pose = env.command_manager.get_command(command_name)
    goal_pos = goal_pose[:, :3]

    object_pos = asset.data.root_pos_w - env.scene.env_origins

    return torch.norm(object_pos - goal_pos, p=2, dim=-1)


def fingertip_distance_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    fingertip_body_names: Sequence[str] | None = None,
) -> torch.Tensor:
    """指尖到物体中心距离的平均值。"""

    if fingertip_body_names is None or len(fingertip_body_names) == 0:
        raise ValueError("fingertip_body_names 不能为空，需指定指尖 body 名称。")

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    if not hasattr(env, "_leaphand_fingertip_body_ids"):
        body_ids, _ = robot.find_bodies(list(fingertip_body_names), preserve_order=True)
        env._leaphand_fingertip_body_ids = torch.as_tensor(body_ids, device=env.device, dtype=torch.long)

    fingertip_pos = robot.data.body_pos_w[:, env._leaphand_fingertip_body_ids]
    fingertip_pos = fingertip_pos - env.scene.env_origins.unsqueeze(1)

    object_pos = obj.data.root_pos_w - env.scene.env_origins

    distances = torch.norm(fingertip_pos - object_pos.unsqueeze(1), p=2, dim=-1)

    return torch.mean(distances, dim=-1)


###
#  参考LEAP_Hand_Isaac_Lab奖励项
###
def pose_diff_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    natural_pose: dict[str, float] | None = None
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
        env._leaphand_natural_joint_ids = torch.as_tensor(
            joint_ids, device=env.device, dtype=torch.long
        )
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
    pose_diff_penalty = torch.sum(pose_diff ** 2, dim=-1)

    return pose_diff_penalty


###
# 接触奖励
###
