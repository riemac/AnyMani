# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""触觉相关的奖励函数

基于ContactSensor数据计算的奖励项，包括：
1. 负载分配奖励：鼓励手指承担垂直载荷
2. Good Contact奖励：鼓励多指尖接触
3. Bad Contact惩罚：惩罚非指尖部位接触
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _compute_reward_curriculum_lambda(
    env: ManagerBasedRLEnv,
    command_name: str,
    g_min: float,
    g_max: float,
    metric_key: str = "consecutive_success",
) -> torch.Tensor:
    """Compute adaptive reward curriculum coefficient.

    Uses a linear schedule based on a command term metric (per environment):

    .. math::

        \lambda = \mathrm{clip}\left(\frac{g_{\text{eval}} - g_{\min}}{g_{\max} - g_{\min}}, 0, 1\right)

    If the metric is missing, returns zeros.
    """
    # Guard against degenerate ranges.
    denom = float(g_max) - float(g_min)
    if denom <= 0.0:
        return torch.zeros(env.num_envs, device=env.device)

    cmd_term = env.command_manager.get_term(command_name)
    g_eval = cmd_term.metrics.get(metric_key, None)
    if g_eval is None:
        return torch.zeros(env.num_envs, device=env.device)

    lam = (g_eval - float(g_min)) / denom
    return torch.clamp(lam, 0.0, 1.0)


def load_distribution_reward(
    env: ManagerBasedRLEnv,
    fingertip_sensor_names: Sequence[str],
    palm_sensor_names: Sequence[str],
    gravity_axis: int = 2,  # z轴
    epsilon: float = 1e-6,
) -> torch.Tensor:
    r"""负载分配奖励：鼓励手指承担的垂直载荷占总载荷的比例趋近于1。
    
    参考 AnyRotate 论文的负载分配策略，计算手指承担的垂直方向合力占比：
    
    .. math::
        
        r_{\text{load}} = \frac{\sum_{i \in \mathcal{F}} (\mathbf{f}_i \cdot \hat{k})}{\sum_{j \in \mathcal{F} \cup \mathcal{P}} (\mathbf{f}_j \cdot \hat{k}) + \epsilon}
        
        \mathbf{f}_i = \mathbf{f}_{n,i} + \mathbf{f}_{t,i}
    
    其中：
    - :math:`\mathcal{F}` 是指尖集合，:math:`\mathcal{P}` 是手掌/非指尖集合
    - :math:`\mathbf{f}_{n,i}` 是法向力（来自 ``force_matrix_w``）
    - :math:`\mathbf{f}_{t,i}` 是切向力（来自 ``friction_forces_w``）
    - :math:`\hat{k}` 是重力方向单位向量（默认 z 轴，可配置）
    - :math:`\epsilon` 是数值稳定项，防止除零
    
    算法流程：
    
    1. 对每个指尖传感器：
       
       a. 读取 ``force_matrix_w[:, 0, 0, :]`` (法向力)
       b. 读取 ``friction_forces_w[:, 0, 0, :]`` (切向力)
       c. 计算总合力：:math:`\mathbf{f}_{\text{total}} = \mathbf{f}_n + \mathbf{f}_t`
       d. 提取垂直分量：:math:`f_{z} = \mathbf{f}_{\text{total}} \cdot \hat{k}`
    
    2. 对每个手掌/非指尖传感器，执行相同操作
    3. 计算奖励比例：
       
       .. math::
           
           r = \frac{\text{fingers\_vertical\_load}}{\text{total\_vertical\_load} + \epsilon}
    
    Args:
        env: 强化学习环境实例
        fingertip_sensor_names: 指尖ContactSensor名称列表
            例如：``["contact_index", "contact_middle", "contact_ring", "contact_thumb"]``
        palm_sensor_names: 手掌/非指尖ContactSensor名称列表
            例如：``["contact_palm", "contact_index_mcp", ...]``
        gravity_axis: 重力方向轴（0=x, 1=y, 2=z），默认2（z轴向上）
        epsilon: 数值稳定项，防止分母为零
    
    Returns:
        奖励张量，形状 (num_envs,)；为手指垂直合力占总垂直合力的比例。
        在采用“完整垂直合力分量”的定义下，该比例不再严格限制在 [0, 1]。
    
    Raises:
        RuntimeError: 如果传感器未启用 ``track_friction_forces=True``
    
    Notes:
        - 所有指尖传感器必须配置 ``track_friction_forces=True``
        - 计算的是垂直方向的**完整合力分量**（包含向上与向下），不进行 clamp。
        - 如果总垂直合力接近0（例如接触很弱或正负抵消），返回0而非NaN。
    
    Example:
        在 ``RewardsCfg`` 中配置：
        
        .. code-block:: python
            
            load_distribution = RewTerm(
                func=leap_mdp.load_distribution_reward,
                weight=0.5,
                params={
                    "fingertip_sensor_names": ["contact_index", "contact_middle", "contact_ring", "contact_thumb"],
                    "palm_sensor_names": ["contact_palm"],
                    "gravity_axis": 2,
                },
            )
    """
    num_envs = env.num_envs
    device = env.device
    
    # 初始化垂直载荷累加器
    fingers_vertical_load = torch.zeros(num_envs, device=device)
    palm_vertical_load = torch.zeros(num_envs, device=device)
    
    # 定义重力方向单位向量（世界坐标系）
    gravity_dir = torch.zeros(3, device=device)
    gravity_dir[gravity_axis] = 1.0  # 例如 [0, 0, 1] 表示 z 轴向上
    
    # 辅助函数：计算垂直载荷
    def compute_vertical_load(sensor_names: Sequence[str]) -> torch.Tensor:
        """计算一组传感器的垂直载荷总和"""
        vertical_load = torch.zeros(num_envs, device=device)
        
        for sensor_name in sensor_names:
            sensor = env.scene[sensor_name]
            
            # 获取法向力和切向力
            normal_force = sensor.data.force_matrix_w  # (num_envs, num_bodies, num_filters, 3)
            
            if sensor.data.friction_forces_w is None:
                raise RuntimeError(
                    f"Sensor '{sensor_name}' does not have friction_forces_w enabled. "
                    "Please set track_friction_forces=True in ContactSensorCfg."
                )
            friction_force = sensor.data.friction_forces_w
            
            # 计算总合力（法向 + 切向）
            total_force_w = normal_force + friction_force  # (num_envs, num_bodies, num_filters, 3)
            
            # 提取第一个 body、第一个 filter 的力
            force = total_force_w[:, 0, 0, :]  # (num_envs, 3)
            
            # 计算垂直分量（点积）
            vertical_component = torch.sum(force * gravity_dir, dim=-1)  # (num_envs,)

            # 累加完整的垂直方向合力（不做 clamp）
            vertical_load += vertical_component
        
        return vertical_load
    
    # 计算手指和手掌的垂直载荷
    fingers_vertical_load = compute_vertical_load(fingertip_sensor_names)
    palm_vertical_load = compute_vertical_load(palm_sensor_names)
    
    # 总垂直载荷
    total_vertical_load = fingers_vertical_load + palm_vertical_load
    
    # 计算奖励比例（避免除零）
    reward = fingers_vertical_load / (total_vertical_load + epsilon)

    # 如果总合力接近0（数值不稳定），奖励设为0
    reward = torch.where(torch.abs(total_vertical_load) < epsilon, torch.zeros_like(reward), reward)
    
    return reward


def good_fingertip_contact(
    env: ManagerBasedRLEnv,
    sensor_names: Sequence[str],
    min_contacts: int = 2,
    force_threshold: float = 1.0,
    reward_type: str = "binary",
    use_curriculum: bool = False,
    command_name: str = "goal_pose",
    g_min: float = 1.0,
    g_max: float = 2.0,
    metric_key: str = "consecutive_success",
) -> torch.Tensor:
    r"""Good Contact奖励：鼓励至少 ``min_contacts`` 个指尖与物体接触。
    
    参考 AnyRotate 论文公式 (6)：
    
    .. math::
        
        r_{\text{gc}} = \begin{cases}
            1 & \text{if } n_{\text{tip-contact}} \ge \text{min\_contacts} \\
            0 & \text{otherwise}
        \end{cases}
    
    算法流程：
    
    1. 对每个指尖传感器：
       
       a. 读取 ``force_matrix_w`` 和 ``friction_forces_w``
       b. 计算总合力的模：:math:`\|\mathbf{f}_{\text{total}}\|`
       c. 判断是否接触：:math:`\|\mathbf{f}_{\text{total}}\| > \text{force\_threshold}`
    
    2. 统计接触的指尖数量：:math:`n_{\text{tip-contact}}`
    3. 返回二值奖励：:math:`n \ge \text{min\_contacts}` 时为 1，否则为 0
    
    Args:
        env: 强化学习环境实例
        sensor_names: 指尖ContactSensor名称列表
        min_contacts: 触发奖励所需的最少接触指尖数量（默认2）
        force_threshold: 判断接触的力阈值（默认1.0 N）
    
    Returns:
        奖励张量，形状 (num_envs,)，值为 0 或 1
    
    Notes:
        - 接触判断基于**总合力**（法向+切向）的模
        - 此函数实现了"多指协同抓取"的奖励机制
        - 配合 ``bad_palm_contact`` 使用可引导策略形成自然的finger gaiting
    
    Example:
        在 ``RewardsCfg`` 中配置：
        
        .. code-block:: python
            
            good_fingertip_contact = RewTerm(
                func=leap_mdp.good_fingertip_contact,
                weight=1.0,
                params={
                    "sensor_names": ["contact_index", "contact_middle", "contact_ring", "contact_thumb"],
                    "min_contacts": 2,  # 至少2个指尖接触
                    "force_threshold": 1.0,
                },
            )
    """
    num_envs = env.num_envs
    device = env.device
    
    # 统计每个环境中接触的指尖数量
    contact_count = torch.zeros(num_envs, device=device, dtype=torch.int32)
    
    for sensor_name in sensor_names:
        sensor = env.scene[sensor_name]
        
        # 获取法向力
        normal_force = sensor.data.force_matrix_w
        
        # 获取摩擦力（如果可用）
        if sensor.data.friction_forces_w is not None:
            friction_force = sensor.data.friction_forces_w
            total_force_w = normal_force + friction_force
        else:
            # 如果没有摩擦力数据，只用法向力判断
            total_force_w = normal_force
        
        # 提取力矢量 (num_envs, 3)
        force = total_force_w[:, 0, 0, :]
        
        # 计算力的模
        force_norm = torch.norm(force, dim=-1)  # (num_envs,)
        
        # 判断是否接触
        is_contact = (force_norm > force_threshold).int()
        
        # 累加接触计数
        contact_count += is_contact
    
    if reward_type == "binary":
        reward = (contact_count >= min_contacts).float()
    elif reward_type in {"count", "continuous"}:
        reward = contact_count.float()
    else:
        raise ValueError(f"Unknown reward_type: {reward_type}. Expected 'binary' or 'count'.")

    if use_curriculum:
        lam = _compute_reward_curriculum_lambda(env, command_name, g_min, g_max, metric_key)
        reward = lam * reward

    return reward


def bad_palm_contact(
    env: ManagerBasedRLEnv,
    sensor_names: Sequence[str],
    force_threshold: float = 0.5,
    reward_type: str = "binary",
    use_curriculum: bool = False,
    command_name: str = "goal_pose",
    g_min: float = 1.0,
    g_max: float = 2.0,
    metric_key: str = "consecutive_success",
) -> torch.Tensor:
    r"""Bad Contact惩罚：惩罚手掌或非指尖部位与物体的接触。
    
    参考 AnyRotate 论文公式 (7)：
    
    .. math::
        
        r_{\text{bc}} = \begin{cases}
            1 & \text{if } n_{\text{non-tip-contact}} > 0 \\
            0 & \text{otherwise}
        \end{cases}
    
    此函数返回 **正值**，需在 ``RewardsCfg`` 中配置 **负权重** 以实现惩罚。
    
    算法流程：
    
    1. 遍历所有非期望接触的传感器（手掌、手指关节等）
    2. 检测是否有任何一个传感器发生接触
    3. 如果有接触，返回 1；否则返回 0
    
    Args:
        env: 强化学习环境实例
        sensor_names: 非期望接触部位的ContactSensor名称列表
            例如：``["contact_palm", "contact_index_mcp", "contact_thumb_pip", ...]``
        force_threshold: 判断接触的力阈值（默认0.5 N）
    
    Returns:
        惩罚指示张量，形状 (num_envs,)，值为 0 或 1
        
        - 1：有非期望接触（需配置负权重）
        - 0：无非期望接触
    
    Notes:
        - 此惩罚鼓励策略避免用手掌托举物体
        - 配合 ``good_fingertip_contact`` 使用，引导策略仅用指尖操作
        - 阈值 ``force_threshold`` 通常设置比指尖接触低（避免误报幽灵力）
    
    Example:
        在 ``RewardsCfg`` 中配置：
        
        .. code-block:: python
            
            bad_palm_contact = RewTerm(
                func=leap_mdp.bad_palm_contact,
                weight=-1.0,  # 注意：负权重实现惩罚
                params={
                    "sensor_names": [
                        "contact_palm",
                        "contact_index_mcp", "contact_index_pip", "contact_index_dip",
                        "contact_middle_mcp", "contact_middle_pip", "contact_middle_dip",
                        "contact_ring_mcp", "contact_ring_pip", "contact_ring_dip",
                        "contact_thumb_base", "contact_thumb_pip", "contact_thumb_dip",
                    ],
                    "force_threshold": 0.5,
                },
            )
    """
    num_envs = env.num_envs
    device = env.device
    
    # 统计非期望接触的数量（按传感器计数）
    bad_contact_count = torch.zeros(num_envs, device=device, dtype=torch.int32)
    
    for sensor_name in sensor_names:
        sensor = env.scene[sensor_name]
        
        # 获取法向力
        normal_force = sensor.data.force_matrix_w
        
        # 获取摩擦力（如果可用）
        if sensor.data.friction_forces_w is not None:
            friction_force = sensor.data.friction_forces_w
            total_force_w = normal_force + friction_force
        else:
            total_force_w = normal_force
        
        # 提取力矢量 (num_envs, 3)
        force = total_force_w[:, 0, 0, :]
        
        # 计算力的模
        force_norm = torch.norm(force, dim=-1)  # (num_envs,)
        
        # 检测接触
        is_contact = force_norm > force_threshold

        bad_contact_count += is_contact.int()

    if reward_type == "binary":
        penalty = (bad_contact_count > 0).float()
    elif reward_type in {"count", "continuous"}:
        penalty = bad_contact_count.float()
    else:
        raise ValueError(f"Unknown reward_type: {reward_type}. Expected 'binary' or 'count'.")

    if use_curriculum:
        lam = _compute_reward_curriculum_lambda(env, command_name, g_min, g_max, metric_key)
        penalty = lam * penalty

    return penalty
