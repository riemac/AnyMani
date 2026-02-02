# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
仿射编队动作项实现 - 基于编队控制思想的灵巧手指尖控制

该动作项将编队飞行中的仿射变换思想应用到手内旋转任务：
1. 预设标称手型 P* (4指×3坐标)
2. RL学习9维参数：旋转tau(3) + 缩放scales(3) + 平移b(3)
3. 通过仿射变换计算期望指尖位置：P = (R @ D) @ P* + b
4. 用IK将期望位置映射到关节空间

坐标系约定：
- {w}: 世界坐标系
- {h}: 手部基座局部坐标系（robot.data.root_pos_w）
- {b}: 指尖刚体坐标系
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets.articulation import Articulation
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    
    from . import affine_formation_cfg


class AffineFormationAction(ActionTerm):
    """仿射编队动作项实现
    
    该类将9维动作空间（旋转3+缩放3+平移3）映射到16维关节空间。
    
    核心流程：
    1. 接收网络输出的9维动作
    2. 约束动作到安全范围
    3. 通过Rodrigues公式计算旋转矩阵R
    4. 构造对角缩放矩阵D
    5. 应用仿射变换：P = (R @ D) @ P* + b
    6. 为每根手指独立求解IK
    7. 输出16维关节位置目标
    """
    
    cfg: affine_formation_cfg.AffineFormationActionCfg
    _asset: Articulation
    
    def __init__(self, cfg: affine_formation_cfg.AffineFormationActionCfg, env: ManagerBasedEnv):
        """初始化仿射编队动作项
        
        Args:
            cfg: 仿射编队动作配置
            env: 环境实例
        """
        super().__init__(cfg, env)
        
        # 获取资产
        self._asset: Articulation = env.scene[cfg.asset_name]
        self._num_envs = env.num_envs
        self._device = self._asset.device
        
        # ========== 解析指尖刚体索引 ==========
        self._fingertip_body_ids = []
        for body_name in cfg.finger_bodies:
            body_ids, _ = self._asset.find_bodies(body_name)
            if len(body_ids) != 1:
                raise ValueError(
                    f"指尖刚体'{body_name}'匹配到{len(body_ids)}个，期望恰好1个"
                )
            self._fingertip_body_ids.append(body_ids[0])
        
        # ========== 解析关节索引 ==========
        self._finger_names = ["index", "middle", "ring", "thumb"]
        self._finger_joint_ids = {}  # {finger_name: [joint_indices]}
        
        all_joint_ids = []
        for finger_name in self._finger_names:
            joint_names = cfg.finger_joints[finger_name]
            joint_ids, _ = self._asset.find_joints(joint_names, preserve_order=True)
            self._finger_joint_ids[finger_name] = joint_ids
            all_joint_ids.extend(joint_ids)
        
        # 验证关节数量
        if len(all_joint_ids) != 16:
            raise ValueError(
                f"总关节数为{len(all_joint_ids)}，期望16个（4指×4关节）"
            )
        
        # ========== 创建IK控制器（每根手指独立） ==========
        ik_cfg = DifferentialIKControllerCfg(
            command_type="position",  # 只控制位置
            use_relative_mode=False,  # 绝对位置目标
            ik_method=cfg.ik_method,
            ik_params=cfg.ik_params
        )
        
        self._ik_controllers = {}
        for finger_name in self._finger_names:
            self._ik_controllers[finger_name] = DifferentialIKController(
                ik_cfg, num_envs=self._num_envs, device=self._device
            )
        
        # ========== 标称指尖位置 P* ==========
        # 根据配置方式选择初始化策略
        if cfg.nominal_joint_angles is not None:
            # 方式1: 从关节角度通过FK计算（推荐）
            self._P_star = self._compute_nominal_fingertip_from_joints(cfg.nominal_joint_angles)
        else:
            # 方式2: 直接使用指定的指尖位置
            P_star_list = list(cfg.nominal_fingertip_config)
            self._P_star = torch.tensor(
                P_star_list, dtype=torch.float32, device=self._device
            ).unsqueeze(0).expand(self._num_envs, -1, -1)  # (N, 4, 3)
        
        # ========== 缓存变量 ==========
        self._raw_actions = torch.zeros(self._num_envs, self.action_dim, device=self._device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._P_desired = torch.zeros(self._num_envs, 4, 3, device=self._device)
        self._joint_pos_target = torch.zeros(self._num_envs, 16, device=self._device)
        
        # ========== 调试统计 ==========
        self._ik_failure_count = 0
        self._ik_total_count = 0
    
    # ========== ActionTerm 必需接口 ==========
    
    @property
    def action_dim(self) -> int:
        """动作空间维度：9维 (tau:3 + scales:3 + translation:3)"""
        return 9
    
    @property
    def raw_actions(self) -> torch.Tensor:
        """原始动作（网络输出）"""
        return self._raw_actions
    
    @property
    def processed_actions(self) -> torch.Tensor:
        """处理后的动作（约束后）"""
        return self._processed_actions
    
    def process_actions(self, actions: torch.Tensor):
        """处理和约束动作
        
        Args:
            actions: (N, 9) 网络输出的原始动作
        """
        self._raw_actions[:] = actions
        
        # 分解动作：tau, scales, translation
        tau = actions[:, 0:3]
        scales_raw = actions[:, 3:6]
        translation = actions[:, 6:9]
        
        # 1. 约束旋转切向量模长
        tau_norm = torch.norm(tau, dim=1, keepdim=True)
        tau_clamped = torch.where(
            tau_norm > self.cfg.rotation_limit,
            tau * (self.cfg.rotation_limit / (tau_norm + 1e-8)),
            tau
        )
        
        # 2. 约束缩放因子（学习偏差后映射到[scale_min, scale_max]）
        scale_min, scale_max = self.cfg.scale_range
        scales = torch.clamp(scales_raw, min=-0.5, max=0.5)  # 偏差限制
        scales = 1.0 + scales  # 映射到[0.5, 1.5]
        scales = torch.clamp(scales, min=scale_min, max=scale_max)
        
        # 3. 约束平移向量
        translation = torch.clamp(
            translation, 
            min=-self.cfg.translation_limit, 
            max=self.cfg.translation_limit
        )
        
        # 保存处理后的动作
        self._processed_actions = torch.cat([tau_clamped, scales, translation], dim=1)
    
    def apply_actions(self):
        """应用动作：仿射变换 → IK求解 → 关节位置控制"""
        # 1. 计算期望指尖位置
        self._P_desired = self._compute_affine_transform(self._processed_actions)
        
        # 2. 为每根手指求解IK
        self._solve_ik_all_fingers()
        
        # 3. 设置关节位置目标
        self._asset.set_joint_position_target(self._joint_pos_target)
    
    # ========== 核心算法实现 ==========
    
    def _compute_affine_transform(self, actions: torch.Tensor) -> torch.Tensor:
        """计算仿射变换后的期望指尖位置
        
        公式：P = (R @ D) @ P* + b
        
        Args:
            actions: (N, 9) 处理后的动作 [tau, scales, translation]
        
        Returns:
            P_desired: (N, 4, 3) 期望的指尖位置（手部基座局部坐标系）
        """
        tau = actions[:, 0:3]
        scales = actions[:, 3:6]
        translation = actions[:, 6:9]
        
        # 1. 旋转：切向量 → 旋转矩阵（Rodrigues公式）
        R = self._tangent_to_rotation(tau)  # (N, 3, 3)
        
        # 2. 缩放：对角矩阵
        D = torch.diag_embed(scales)  # (N, 3, 3)
        
        # 3. 组合仿射矩阵 A = R @ D
        A = R @ D  # (N, 3, 3)
        
        # 4. 应用变换：P = A @ P* + b
        # P*: (N, 4, 3), A: (N, 3, 3) → 使用einsum广播
        P_transformed = torch.einsum('nij,nfj->nfi', A, self._P_star)  # (N, 4, 3)
        P_desired = P_transformed + translation.unsqueeze(1)  # 广播平移
        
        return P_desired
    
    def _tangent_to_rotation(self, tau: torch.Tensor) -> torch.Tensor:
        """切向量转旋转矩阵（Rodrigues公式）
        
        公式：R = I + sin(θ)/θ * [tau]× + (1-cos(θ))/θ² * [tau]×²
        其中 θ = ||tau||，[tau]× 是反对称矩阵
        
        Args:
            tau: (N, 3) 旋转切向量
        
        Returns:
            R: (N, 3, 3) 旋转矩阵 ∈ SO(3)
        """
        N = tau.shape[0]
        theta = torch.norm(tau, dim=1, keepdim=True)  # (N, 1)
        
        # 处理theta≈0的情况（一阶泰勒展开）
        small_angle = theta < 1e-6
        theta_safe = torch.where(small_angle, torch.ones_like(theta), theta)
        
        # 归一化得到旋转轴
        k = tau / theta_safe  # (N, 3)
        
        # 构造反对称矩阵 [k]×
        k_hat = self._skew_symmetric(k)  # (N, 3, 3)
        k_hat_sq = k_hat @ k_hat  # (N, 3, 3)
        
        # Rodrigues公式
        I = torch.eye(3, device=self._device).unsqueeze(0).expand(N, -1, -1)
        
        # 系数
        coeff1 = torch.where(
            small_angle,
            torch.ones_like(theta),  # theta→0时，sin(θ)/θ → 1
            torch.sin(theta) / theta
        ).unsqueeze(-1)  # (N, 1, 1)
        
        coeff2 = torch.where(
            small_angle,
            0.5 * torch.ones_like(theta),  # theta→0时，(1-cos(θ))/θ² → 0.5
            (1 - torch.cos(theta)) / (theta ** 2)
        ).unsqueeze(-1)  # (N, 1, 1)
        
        R = I + coeff1 * k_hat + coeff2 * k_hat_sq
        
        return R
    
    @staticmethod
    def _skew_symmetric(v: torch.Tensor) -> torch.Tensor:
        """构造反对称矩阵
        
        输入：v = [v1, v2, v3]
        输出：[v]× = [[ 0  -v3  v2],
                      [ v3  0  -v1],
                      [-v2  v1  0 ]]
        
        Args:
            v: (N, 3) 向量
        
        Returns:
            v_hat: (N, 3, 3) 反对称矩阵
        """
        N = v.shape[0]
        v_hat = torch.zeros(N, 3, 3, device=v.device, dtype=v.dtype)
        
        v_hat[:, 0, 1] = -v[:, 2]
        v_hat[:, 0, 2] = v[:, 1]
        v_hat[:, 1, 0] = v[:, 2]
        v_hat[:, 1, 2] = -v[:, 0]
        v_hat[:, 2, 0] = -v[:, 1]
        v_hat[:, 2, 1] = v[:, 0]
        
        return v_hat
    
    def _solve_ik_all_fingers(self):
        """为所有手指求解IK"""
        # 获取当前手部基座位置（用于坐标转换）
        hand_base_pos = self._asset.data.root_pos_w  # (N, 3)
        
        # 获取所有关节的当前位置和雅可比
        all_jacobians = self._asset.root_physx_view.get_jacobians()  # (N, num_bodies, 6, num_dofs)
        joint_pos_current = self._asset.data.joint_pos  # (N, num_joints)
        
        # 遍历每根手指
        for i, finger_name in enumerate(self._finger_names):
            # 1. 获取当前指尖状态
            body_idx = self._fingertip_body_ids[i]
            
            # 指尖位置（世界坐标系）
            ee_pos_w = self._asset.data.body_pos_w[:, body_idx]  # (N, 3)
            # 转换到手部局部坐标系
            ee_pos_local = ee_pos_w - hand_base_pos if self.cfg.use_body_frame else ee_pos_w
            
            # 指尖姿态（用于IK锚定）
            ee_quat_w = self._asset.data.body_quat_w[:, body_idx]  # (N, 4)
            
            # 2. 设置IK目标（期望位置）
            P_target = self._P_desired[:, i, :]  # (N, 3)
            
            ik_controller = self._ik_controllers[finger_name]
            ik_controller.set_command(
                command=P_target,
                ee_pos=ee_pos_local,
                ee_quat=ee_quat_w  # 姿态保持当前值
            )
            
            # 3. 提取该手指的雅可比
            joint_ids = self._finger_joint_ids[finger_name]
            
            # 调整雅可比索引（考虑固定基座）
            if self._asset.is_fixed_base:
                jacobi_body_idx = body_idx - 1
                jacobi_joint_ids = joint_ids
            else:
                jacobi_body_idx = body_idx
                jacobi_joint_ids = [idx + 6 for idx in joint_ids]
            
            # 提取position部分的雅可比 (前3行)
            jacobian = all_jacobians[:, jacobi_body_idx, 0:3, :][:, :, jacobi_joint_ids]  # (N, 3, 4)
            
            # 4. 求解IK
            joint_pos_finger = joint_pos_current[:, joint_ids]  # (N, 4)
            
            try:
                joint_pos_target = ik_controller.compute(
                    ee_pos=ee_pos_local,
                    ee_quat=ee_quat_w,
                    jacobian=jacobian,
                    joint_pos=joint_pos_finger
                )
                # 更新全局关节目标
                self._joint_pos_target[:, joint_ids] = joint_pos_target
                
            except Exception as e:
                # IK失败时保持当前关节位置
                print(f"[WARNING] IK失败 {finger_name}: {e}")
                self._joint_pos_target[:, joint_ids] = joint_pos_finger
                self._ik_failure_count += 1
            
            self._ik_total_count += 1
    
    def _compute_nominal_fingertip_from_joints(self, joint_angles: dict[str, float]) -> torch.Tensor:
        """通过正运动学从关节角度计算标称指尖位置
        
        Args:
            joint_angles: 关节角度配置字典 {joint_name: angle}
        
        Returns:
            P_star: (N, 4, 3) 标称指尖位置
        """
        # 1. 保存当前关节状态
        current_joint_pos = self._asset.data.joint_pos.clone()
        
        # 2. 设置机器人到标称构型
        target_joint_pos = torch.zeros(
            self._num_envs, self._asset.num_joints, device=self._device
        )
        
        # 填充目标关节角度
        for joint_name, angle in joint_angles.items():
            joint_ids, _ = self._asset.find_joints(joint_name)
            if len(joint_ids) == 1:
                target_joint_pos[:, joint_ids[0]] = angle
        
        # 临时写入关节位置（不影响仿真，仅用于FK计算）
        self._asset.write_joint_state_to_sim(target_joint_pos, torch.zeros_like(target_joint_pos))
        
        # 3. 更新articulation状态以触发FK计算
        self._asset.update(dt=0.0)  # dt=0表示仅更新状态不推进时间
        
        # 4. 读取指尖位置（世界坐标系）
        hand_base_pos = self._asset.data.root_pos_w  # (N, 3)
        fingertip_pos_w = self._asset.data.body_pos_w[:, self._fingertip_body_ids]  # (N, 4, 3)
        
        # 5. 转换到手部局部坐标系
        if self.cfg.use_body_frame:
            P_star = fingertip_pos_w - hand_base_pos.unsqueeze(1)
        else:
            P_star = fingertip_pos_w
        
        # 6. 恢复原始关节状态
        self._asset.write_joint_state_to_sim(current_joint_pos, torch.zeros_like(current_joint_pos))
        self._asset.update(dt=0.0)
        
        return P_star
    
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """重置动作项状态"""
        if env_ids is None:
            env_ids = slice(None)
        
        # 重置缓存
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._P_desired[env_ids] = self._P_star[env_ids]
        
        # 重置IK控制器
        for ik_controller in self._ik_controllers.values():
            ik_controller.reset(env_ids)
