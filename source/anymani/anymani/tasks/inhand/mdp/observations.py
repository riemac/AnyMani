# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LeapHand连续旋转任务的观测函数

提供sim和real都能使用的观测值
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

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
    
    num_sensors = len(sensor_names)
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


def so3_command(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    """读取 so(3) 相对增量指令（rotvec）。

    该观测项用于对齐方案：policy 侧只接收 3 维 rotvec 指令，而不依赖物体绝对姿态。

    优先从命令项对象读取 `phi_ref_e`（便于后续扩展为手掌系 {s} 等），若不存在则回退到
    `env.command_manager.get_command()`。

    Args:
        env: 强化学习环境实例
        command_name: CommandManager 中的命令项名称

    Returns:
        (num_envs, 3) 张量，so(3) 指令 rotvec
    """

    term = env.command_manager.get_term(command_name)
    if hasattr(term, "phi_ref_e"):
        phi = getattr(term, "phi_ref_e")
        if isinstance(phi, torch.Tensor) and phi.shape[-1] == 3:
            return phi

    cmd = env.command_manager.get_command(command_name)
    # fallback: 假设 command 格式为 (pos_e, phi_ref_e) = 6D，取后 3 维
    if isinstance(cmd, torch.Tensor):
        if cmd.shape[-1] == 6:
            return cmd[:, 3:6]
        if cmd.shape[-1] == 3:
            return cmd

    raise RuntimeError(
        f"so3_command expects command '{command_name}' to provide a (num_envs,3) tensor or a (num_envs,6) tensor (pos_e, phi_ref_e). "
        f"Got: {type(cmd)} {getattr(cmd, 'shape', None)}"
    )


def pos_command(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
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