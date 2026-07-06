# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LeapHand连续旋转任务的观测函数

提供sim和real都能使用的观测值
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


###
# 触觉相关
###


def fingertip_contact_data(
    env: ManagerBasedRLEnv,
    sensor_names: Sequence[str],
    output_type: str = "force",
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """获取指尖触觉数据（支持力信号或二值化接触状态）。

    此函数从ContactSensor中提取触觉信息，支持两种输出模式：

    1. **力信号模式** (`output_type="force"`):
       - 返回每个指尖的总接触合力（法向力 + 摩擦力）在世界坐标系下的矢量
       - 计算公式：f_total = force_matrix_w + friction_forces_w
       - 输出形状：(num_envs, num_sensors * 3)
       - 用于teacher policy的精确力控制

    2. **二值信号模式** (`output_type="binary"`):
       - 返回每个指尖是否接触（0或1）
       - 判断标准：||f_total|| > force_threshold
       - 输出形状：(num_envs, num_sensors)
       - 用于student policy的sim2real部署

    Args:
        env: 强化学习环境实例
        sensor_names: ContactSensor名称列表，如 ["contact_index", "contact_middle", ...]
        output_type: 输出类型，"force"（默认）或 "binary"
        force_threshold: 二值化阈值（仅在output_type="binary"时使用）

    Returns:
        力信号：(num_envs, num_sensors * 3) 的张量
        二值信号：(num_envs, num_sensors) 的张量

    Raises:
        ValueError: 如果 output_type 不是 "force" 或 "binary"
        RuntimeError: 如果传感器未启用 track_friction_forces（在 force 模式下）

    Notes:
        - 对于力信号模式，ContactSensor 必须配置 `track_friction_forces=True`
        - 形状说明：
          - force_matrix_w: (num_envs, num_bodies, num_filters, 3)
          - friction_forces_w: (num_envs, num_bodies, num_filters, 3)
          - 由于每个指尖传感器只有1个body、1个filter，所以取 [0, 0] 即可
        - 无接触时，force_matrix_w 和 friction_forces_w 均为零向量（不会产生NaN）
    """
    if output_type not in ["force", "binary"]:
        raise ValueError(f"output_type must be 'force' or 'binary', got '{output_type}'")

    forces = []

    for sensor_name in sensor_names:
        # 获取传感器实例
        sensor = env.scene[sensor_name]

        # 获取法向力 (num_envs, num_bodies, num_filters, 3)
        normal_force = sensor.data.force_matrix_w  # 默认值为 0

        if output_type == "force":
            # 获取摩擦力（切向力）
            if sensor.data.friction_forces_w is None:
                raise RuntimeError(
                    f"Sensor '{sensor_name}' does not have friction_forces_w enabled. "
                    "Please set track_friction_forces=True in ContactSensorCfg."
                )
            friction_force = sensor.data.friction_forces_w  # 默认值为 0

            # 计算总合力（法向 + 切向）
            # 形状：(num_envs, num_bodies, num_filters, 3)
            total_force_w = normal_force + friction_force

            # 提取第一个 body、第一个 filter 的力
            # 形状：(num_envs, 3)
            force = total_force_w[:, 0, 0, :]
            forces.append(force)

        else:  # output_type == "binary"
            # 计算总合力的模
            # 如果没有摩擦力数据，只用法向力判断
            if sensor.data.friction_forces_w is not None:
                friction_force = sensor.data.friction_forces_w
                total_force_w = normal_force + friction_force
            else:
                total_force_w = normal_force

            force = total_force_w[:, 0, 0, :]  # (num_envs, 3)
            force_norm = torch.norm(force, dim=-1)  # (num_envs,)
            is_contact = (force_norm > force_threshold).float()  # (num_envs,)
            forces.append(is_contact)

    # 拼接所有传感器的数据
    if output_type == "force":
        # (num_envs, num_sensors * 3)
        return torch.cat(forces, dim=1)
    else:
        # (num_envs, num_sensors)
        return torch.stack(forces, dim=1)


def goal_quat_diff(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    command_name: str = "goal_pose",
    make_quat_unique: bool = True,
) -> torch.Tensor:
    r"""物体当前姿态与目标姿态的四元数差。

    该项曾位于 `observations_privileged.py`，当前主线只保留这一项，因此直接并回 `observations.py`。
    它的职责是为旧 critic / debug 路径提供一个四元数形式的方向误差观测：

    $$
    Q_{err}=Q_g^w\otimes (Q_o^w)^{-1}.
    $$

    其中 $Q_o^w$ 是物体当前姿态，$Q_g^w$ 是目标姿态。若 `make_quat_unique=True`，则再做
    `quat_unique`，消除 $Q$ 与 $-Q$ 的双覆盖歧义。
    """

    obj = env.scene[asset_cfg.name]  # 物体刚体对象，用于读取当前 root quaternion。
    current_quat = obj.data.root_quat_w  # 当前姿态 $Q_o^w$，形状 `[N,4]`。

    term = env.command_manager.get_term(command_name)  # 命令项对象，优先读取内部目标四元数 buffer。
    target_quat = getattr(term, "quat_command_w", None)  # 期望形状 `[N,4]`。
    if not (isinstance(target_quat, torch.Tensor) and target_quat.shape[-1] == 4):
        goal_pose = env.command_manager.get_command(command_name)  # legacy pose-like command fallback。
        if not (isinstance(goal_pose, torch.Tensor) and goal_pose.shape[-1] >= 7):
            raise RuntimeError(
                f"goal_quat_diff expects command '{command_name}' to provide term.quat_command_w or a pose-like tensor. "
                f"Got: {type(goal_pose)} {getattr(goal_pose, 'shape', None)}"
            )
        target_quat = goal_pose[:, -4:]  # legacy 7D command 的最后四维为目标四元数。

    current_quat_inv = math_utils.quat_inv(current_quat)  # $(Q_o^w)^{-1}$。
    quat_diff = math_utils.quat_mul(target_quat, current_quat_inv)  # $Q_g^w\otimes(Q_o^w)^{-1}$。
    if make_quat_unique:
        quat_diff = math_utils.quat_unique(quat_diff)  # 约定 $w\ge0$，保持观测表示连续。
    return quat_diff


def quat_command(env: ManagerBasedRLEnv, command_name: str, make_quat_unique: bool = True) -> torch.Tensor:
    r"""读取目标姿态四元数命令 $Q_g^w$。

    该观测项用于替代旧的 3D rotvec 指令观测。对于当前 `RelativeSO3Command`，
    `command()` 只暴露 $(p_g^e,\phi_g^e)$ 的 6D 张量，但目标四元数仍作为命令项内部状态
    `quat_command_w` 维护；policy/critic 需要四元数版目标时，应显式读取该 buffer，
    而不是把 6D command 的后三维误当成姿态观测。

    数学语义：
    $$
    o_g = Q_g^w = (w,x,y,z) \in \mathbb{S}^3
    $$
    其中 $Q_g^w$ 与物体根姿态 `root_quat_w` 使用相同的世界系和 wxyz 排列。

    Args:
        env: 强化学习环境实例。
        command_name: CommandManager 中的命令项名称，例如 `goal_pose`。
        make_quat_unique: 是否施加 $w\ge 0$ 的符号约定，消除 $Q$ 与 $-Q$ 的双覆盖歧义。

    Returns:
        torch.Tensor: 形状为 `(num_envs, 4)` 的目标四元数 $Q_g^w$。

    Raises:
        RuntimeError: 当命令项既没有 `quat_command_w`，也没有 legacy 7D pose-like command 时抛出。
    """
    import isaaclab.utils.math as math_utils

    # 优先读取命令项内部 buffer：这是 `RelativeSO3Command.command()` 已改为 6D 后的唯一无歧义四元数来源。
    term = env.command_manager.get_term(command_name)
    quat = getattr(term, "quat_command_w", None)  # 目标姿态 $Q_g^w$，期望形状 [num_envs, 4]
    if not (isinstance(quat, torch.Tensor) and quat.shape[-1] == 4):
        # 兼容历史 continuous-rotation 命令：旧接口直接返回 `[p_g^e, Q_g^w]` 的 7D goal pose。
        cmd = env.command_manager.get_command(command_name)
        if not (isinstance(cmd, torch.Tensor) and cmd.shape[-1] >= 7):
            raise RuntimeError(
                f"quat_command expects command '{command_name}' to provide term.quat_command_w or a pose-like tensor. "
                f"Got: {type(cmd)} {getattr(cmd, 'shape', None)}"
            )
        quat = cmd[:, -4:]  # legacy pose-like command 的最后四维为 $Q_g^w$，形状 [num_envs, 4]

    # 统一四元数符号分支：$Q$ 与 $-Q$ 表示同一物理姿态，但神经网络输入不应随机翻号。
    if make_quat_unique:
        quat = math_utils.quat_unique(quat)  # 约定 $w\ge0$，保持观测表示连续且可学习

    return quat


def pos_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """读取目标位置命令 pos_command_e（环境系 {e}）。

    该观测项用于“位置约束”：命令项可同时输出 (pos_command_e, phi_ref_e) 的 6D command。

    优先从命令项对象读取 `pos_command_e`（更明确的语义）；若不存在则回退到
    `env.command_manager.get_command()`。

    Args:
        env: 强化学习环境实例
        command_name: CommandManager 中的命令项名称

    Returns:
        (num_envs, 3) 张量，目标位置（环境系）
    """

    term = env.command_manager.get_term(command_name)
    if hasattr(term, "pos_command_e"):
        pos = getattr(term, "pos_command_e")
        if isinstance(pos, torch.Tensor) and pos.shape[-1] == 3:
            return pos

    cmd = env.command_manager.get_command(command_name)
    if isinstance(cmd, torch.Tensor):
        # 新接口：6D command = (pos_e, phi_ref_e)
        if cmd.shape[-1] == 6:
            return cmd[:, 0:3]
        # 旧接口：pose-like command = (pos_e, quat_w) with dim>=7
        if cmd.shape[-1] >= 7:
            return cmd[:, 0:3]

    raise RuntimeError(
        f"pos_command expects command '{command_name}' to provide term.pos_command_e, a (num_envs,6) tensor (pos_e, phi_ref_e), "
        f"or a pose-like tensor with dim>=7. Got: {type(cmd)} {getattr(cmd, 'shape', None)}"
    )


def official_policy_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    action_term_name: str = "hand_joint_pos",
) -> torch.Tensor:
    r"""官方 LEAP actor 的单帧 32D 本体控制观测。

    该函数只返回当前时刻的
    $$
    o_t^{frame} = [\tilde q_t, q_t^{target}] \in \mathbb{R}^{32},
    $$
    而历史拼接完全交给 IsaacLab `ObservationTermCfg.history_length` 内置的 `CircularBuffer`。
    这样可直接复用官方框架语义：reset 后第一次 append 会自动把整段历史窗口填满当前帧。

    Args:
        env: ManagerBasedRLEnv 运行时对象。
        asset_cfg: 机器人资产配置，需解析出 LEAP 的 16 个动作关节。
        action_term_name: ActionManager 中 official target-buffer 动作项的名称。

    Returns:
        torch.Tensor: 形状为 ``[N, 32]`` 的单帧观测，其中前 16 维是归一化关节位置，
        后 16 维是当前关节目标 `cur_targets`。
    """

    robot: Articulation = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    lower = robot.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    upper = robot.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    joint_pos_norm = math_utils.scale_transform(joint_pos, lower, upper)
    action_term = env.action_manager.get_term(action_term_name)
    return torch.cat((joint_pos_norm, action_term.current_targets), dim=-1).clone()


def raw_policy_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    action_term_name: str = "hand_joint_pos",
    joint_scale_rad: float = torch.pi,
) -> torch.Tensor:
    r"""N040 / heterogeneous-ready actor 的单帧 32D raw observation。

    当前设计目标是把 official actor 单帧观测

    $$
    o_t^{frame}=[\tilde q_t, q_t^{target}]\in\mathbb R^{32}
    $$

    迁移到 unit-scaled raw-rad 语义：

    $$
    o_t^{frame}=\left[\frac{q_t}{\pi},\frac{q_t^{cmd}}{\pi}\right]\in\mathbb R^{32}.
    $$

    其中：

    - 前 16 维是当前实际关节角 $q_t$ 的 unit-scaled raw rad 表达；
    - 后 16 维是动作项暴露的 `current_targets`，在 official target-buffer action 下它表示
      $q_t^{target}$，在 N040 `ADRRelativeJointPositionAction(reference="current")` 下它表示本步 command target
      $q_t^{cmd}$。

    这样设计的目的不是让 official 与 N040 完全同义，而是让 N040 第一刀保持 PPO 输入维度
    仍为 96D（history 3 帧后），同时把 per-joint-limit normalization 换成跨 variant 更稳定的
    unit-scaled raw coordinates。

    Args:
        env: ManagerBasedRLEnv 运行时对象。
        asset_cfg: 机器人资产配置，需解析出 actor 消费的关节槽位。
        action_term_name: ActionManager 中暴露 `current_targets` 的动作项名称。
        joint_scale_rad: raw rad 到无量纲输入的全局尺度，当前默认 $\pi$。

    Returns:
        torch.Tensor: 形状为 ``[N, 32]`` 的单帧观测。

    Raises:
        RuntimeError: 当动作项没有 `current_targets` 字段时抛出，避免静默退化成错误 obs。
    """

    robot: Articulation = env.scene[asset_cfg.name]  # 机器人 articulation，用于读取当前关节角 $q_t$。
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]  # 当前实际关节角，形状 `[N,16]`、单位 rad。
    action_term = env.action_manager.get_term(action_term_name)  # 动作项；N040 需要它暴露 `current_targets`。

    if not hasattr(action_term, "current_targets"):
        raise RuntimeError(
            f"raw_policy_frame expects action term '{action_term_name}' to expose current_targets for q_cmd/q_target."
        )

    q_obs = joint_pos / float(joint_scale_rad)  # $q_t/\pi$，跨 variant 共享的 unit-scaled raw rad 表达。
    q_cmd_obs = action_term.current_targets / float(joint_scale_rad)  # $q_t^{cmd}/\pi$ 或 $q_t^{target}/\pi$。
    return torch.cat((q_obs, q_cmd_obs), dim=-1).clone()  # 单帧 32D obs，history 由 ObservationTermCfg 负责。
