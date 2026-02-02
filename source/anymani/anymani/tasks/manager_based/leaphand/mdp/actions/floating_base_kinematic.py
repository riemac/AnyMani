# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Floating base kinematic action term implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, axis_angle_from_quat, quat_from_axis_angle

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from . import floating_base_kinematic_cfg


class FloatingBaseKinematicAction(ActionTerm):
    """运动学浮动基座动作项实现
    
    该类将6维动作空间（位置增量3 + 旋转增量3）映射到浮动基座的目标位姿。
    
    核心机制：
    
    1. **运动学控制**：通过`write_root_pose_to_sim()`直接设置基座位姿，
       跳过物理仿真（无重力、无碰撞响应）
    
    2. **增量控制**：动作表示相对于当前位姿的增量：
       
       .. math::
           p_{t+1} = p_t + \\alpha \\cdot (\\Delta p \\cdot s_p)
           q_{t+1} = q_t \\otimes \\Delta q(\\Delta \\omega \\cdot s_r)
       
       其中 :math:`s_p`, :math:`s_r` 是位置和旋转的缩放因子，
       :math:`\\alpha` 是EMA平滑系数。
    
    3. **边界约束**：可选的位置边界，防止手移动到工作空间外。
    
    Note:
        需要设置`articulation_props.fix_root_link = False`才能使用此动作项。
        
    Example:
        在ActionsCfg中使用::
        
            floating_base = FloatingBaseKinematicActionCfg(
                asset_name="robot",
                position_scale=0.01,  # 1cm per action unit
                rotation_scale=0.1,   # 0.1rad per action unit
                smoothing_alpha=0.3,
            )
    """
    
    cfg: floating_base_kinematic_cfg.FloatingBaseKinematicActionCfg
    """动作项配置"""
    
    _asset: Articulation
    """目标关节资产"""
    
    def __init__(self, cfg: floating_base_kinematic_cfg.FloatingBaseKinematicActionCfg, env: ManagerBasedEnv):
        """初始化浮动基座动作项
        
        Args:
            cfg: 动作项配置
            env: 环境实例
        """
        super().__init__(cfg, env)
        
        # 获取资产
        self._asset: Articulation = env.scene[cfg.asset_name]
        
        # 验证资产配置
        # Note: fix_root_link在仿真启动后无法直接检查，依赖用户正确配置
        
        # 动作空间维度：6 (3位置 + 3旋转)
        self._num_actions = 6
        
        # 缓冲区
        self._target_pos = torch.zeros(env.num_envs, 3, device=env.device)
        self._target_quat = torch.zeros(env.num_envs, 4, device=env.device)
        self._target_quat[:, 0] = 1.0  # 初始化为单位四元数 (w=1)
        
        # 初始位姿（用于重置）
        self._init_pos = self._asset.data.default_root_state[:, :3].clone()
        self._init_quat = self._asset.data.default_root_state[:, 3:7].clone()
        
        # 当前目标（用于EMA平滑）
        self._current_target_pos = self._init_pos.clone()
        self._current_target_quat = self._init_quat.clone()
        
        # 原始动作存储
        self._raw_actions = torch.zeros(env.num_envs, 6, device=env.device)
        
    @property
    def action_dim(self) -> int:
        """返回动作空间维度"""
        return self._num_actions
    
    @property
    def raw_actions(self) -> torch.Tensor:
        """返回原始动作"""
        return self._raw_actions
    
    @property
    def processed_actions(self) -> torch.Tensor:
        """返回处理后的目标位姿 (pos + quat)"""
        return torch.cat([self._target_pos, self._target_quat], dim=-1)
    
    def process_actions(self, actions: torch.Tensor):
        """处理动作并计算目标位姿
        
        Args:
            actions: 原始动作 (num_envs, 6)，前3维是位置增量，后3维是旋转增量
        """
        self._raw_actions[:] = actions
        
        # 分解动作
        delta_pos = actions[:, :3] * self.cfg.position_scale
        delta_rot = actions[:, 3:6] * self.cfg.rotation_scale
        
        # 计算新的位置目标
        new_target_pos = self._current_target_pos + delta_pos
        
        # 应用位置边界约束
        if self.cfg.position_bounds is not None:
            lower = torch.tensor(self.cfg.position_bounds[0], device=self._device)
            upper = torch.tensor(self.cfg.position_bounds[1], device=self._device)
            new_target_pos = torch.clamp(new_target_pos, lower, upper)
        
        # 计算新的旋转目标
        if self.cfg.rotation_type == "axis_angle":
            # 将轴角转换为四元数增量
            delta_quat = quat_from_axis_angle(delta_rot)
        elif self.cfg.rotation_type == "euler":
            # 将欧拉角转换为四元数增量
            delta_quat = quat_from_euler_xyz(delta_rot[:, 0], delta_rot[:, 1], delta_rot[:, 2])
        else:
            raise ValueError(f"Unknown rotation type: {self.cfg.rotation_type}")
        
        # 四元数乘法：q_new = q_current * q_delta（局部坐标系旋转）
        new_target_quat = quat_mul(self._current_target_quat, delta_quat)
        
        # 归一化四元数
        new_target_quat = new_target_quat / torch.norm(new_target_quat, dim=-1, keepdim=True)
        
        # EMA平滑
        alpha = self.cfg.smoothing_alpha
        self._target_pos = alpha * new_target_pos + (1 - alpha) * self._current_target_pos
        self._target_quat = self._slerp(self._current_target_quat, new_target_quat, alpha)
        
        # 更新当前目标
        self._current_target_pos = self._target_pos.clone()
        self._current_target_quat = self._target_quat.clone()
    
    def apply_actions(self):
        """将目标位姿应用到仿真中
        
        通过运动学方式直接设置基座位姿，跳过物理仿真。
        """
        # 构造目标位姿 (pos + quat)
        target_pose = torch.cat([self._target_pos, self._target_quat], dim=-1)
        
        # 直接写入仿真（运动学控制）
        self._asset.write_root_pose_to_sim(target_pose)
        
        # 设置速度为零（避免漂移）
        zero_velocity = torch.zeros(self._num_envs, 6, device=self._device)
        self._asset.write_root_velocity_to_sim(zero_velocity)
    
    def reset(self, env_ids: torch.Tensor) -> None:
        """重置指定环境的动作状态
        
        Args:
            env_ids: 需要重置的环境索引
        """
        if len(env_ids) == 0:
            return
        
        if self.cfg.reset_to_init:
            # 重置到初始位姿
            self._current_target_pos[env_ids] = self._init_pos[env_ids]
            self._current_target_quat[env_ids] = self._init_quat[env_ids]
        else:
            # 保持当前位姿
            self._current_target_pos[env_ids] = self._asset.data.root_pos_w[env_ids]
            self._current_target_quat[env_ids] = self._asset.data.root_quat_w[env_ids]
        
        # 重置目标缓冲区
        self._target_pos[env_ids] = self._current_target_pos[env_ids]
        self._target_quat[env_ids] = self._current_target_quat[env_ids]
        self._raw_actions[env_ids] = 0.0
    
    def _slerp(self, q0: torch.Tensor, q1: torch.Tensor, t: float) -> torch.Tensor:
        """球面线性插值（SLERP）
        
        Args:
            q0: 起始四元数 (N, 4)
            q1: 目标四元数 (N, 4)
            t: 插值参数 [0, 1]
            
        Returns:
            插值后的四元数 (N, 4)
        """
        # 计算点积
        dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
        
        # 确保走短路径
        q1 = torch.where(dot < 0, -q1, q1)
        dot = torch.abs(dot)
        
        # 处理接近平行的情况
        # 当dot接近1时，使用线性插值
        linear_threshold = 0.9995
        linear_mask = (dot > linear_threshold).squeeze(-1)
        
        # 初始化输出
        result = torch.zeros_like(q0)
        
        # 线性插值（接近平行）
        if linear_mask.any():
            result[linear_mask] = (1 - t) * q0[linear_mask] + t * q1[linear_mask]
        
        # SLERP（其他情况）
        slerp_mask = ~linear_mask
        if slerp_mask.any():
            theta = torch.acos(dot[slerp_mask])
            sin_theta = torch.sin(theta)
            s0 = torch.sin((1 - t) * theta) / sin_theta
            s1 = torch.sin(t * theta) / sin_theta
            result[slerp_mask] = s0 * q0[slerp_mask] + s1 * q1[slerp_mask]
        
        # 归一化
        result = result / torch.norm(result, dim=-1, keepdim=True)
        
        return result
