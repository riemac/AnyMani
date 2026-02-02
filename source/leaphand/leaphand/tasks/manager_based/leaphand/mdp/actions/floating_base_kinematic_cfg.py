# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for floating base kinematic action term."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.utils import configclass

from . import floating_base_kinematic


@configclass
class FloatingBaseKinematicActionCfg(ActionTermCfg):
    """运动学浮动基座动作项配置
    
    该动作项将6维动作（位置增量3 + 旋转增量3）映射到浮动基座的目标位姿。
    基座通过运动学方式（直接设置位姿）控制，跳过物理仿真。
    
    适用场景：
    - 灵巧手手内操作，需要解耦机械臂型号
    - 需要任意手腕朝向的泛化训练
    - 模拟"完美机械臂"控制的末端位姿
    
    Note:
        基座不受重力和碰撞影响，但手指和物体仍然是物理仿真的。
    """
    
    class_type: type = floating_base_kinematic.FloatingBaseKinematicAction
    
    asset_name: str = MISSING
    """要控制的关节资产名称（必须设置fix_root_link=False）"""
    
    # 位置控制参数
    position_scale: float = 0.01
    """位置增量的缩放因子 [m]。默认0.01表示动作±1对应±1cm的位移"""
    
    position_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    """基座位置的边界约束 [(x_min, y_min, z_min), (x_max, y_max, z_max)]。
    None表示不限制。默认为None"""
    
    # 旋转控制参数
    rotation_scale: float = 0.1
    """旋转增量的缩放因子 [rad]。默认0.1表示动作±1对应±0.1rad的旋转"""
    
    rotation_type: str = "axis_angle"
    """旋转表示类型：'axis_angle' 或 'euler'。默认'axis_angle'"""
    
    # 平滑参数
    smoothing_alpha: float = 0.3
    """EMA平滑系数。0=无平滑，1=完全使用新目标。默认0.3"""
    
    # 重置行为
    reset_to_init: bool = True
    """重置时是否恢复到初始位姿。默认True"""
