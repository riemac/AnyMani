# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
仿射编队动作项配置 - 用于手内旋转任务的对比实验

该配置类实现了编队控制中的仿射变换思想，将其应用到灵巧手指尖位置控制。
核心思想：预设一个标称手型 P*，RL学习仿射变换参数（旋转+缩放+平移），
通过IK将期望指尖位置映射到关节空间。
"""

from collections.abc import Sequence
from typing import Literal

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.configclass import MISSING

from . import affine_formation as affine


@configclass
class AffineFormationActionCfg(ActionTermCfg):
    """仿射编队动作项配置类
    
    动作空间维度：9维
    - tau (3维): 旋转切向量，通过Rodrigues公式映射到SO(3)
    - scales (3维): 对角缩放因子 [d_x, d_y, d_z]
    - translation (3维): 平移向量 [b_x, b_y, b_z]
    
    变换公式：P = (R @ D) @ P* + b
    其中：
    - P: 期望的4指指尖位置 (4, 3)
    - R: 旋转矩阵 ∈ SO(3)，由tau通过Rodrigues公式计算
    - D: 对角缩放矩阵
    - P*: 标称指尖位置（预设的手型）
    - b: 平移向量
    """

    class_type: type[ActionTerm] = affine.AffineFormationAction
    
    # ========== 基本配置 ==========
    asset_name: str = MISSING
    """资产名称（通常是"robot"）"""
    
    # ========== 标称构型配置 ==========
    nominal_joint_angles: dict[str, float] | None = None
    """标称构型的关节角度配置（推荐方式）
    
    通过关节角度定义标称手型P*，初始化时会自动通过正运动学转换为指尖位置。
    这种方式比直接指定坐标更直观和易于调配。
    
    示例：
    {
        "a_1": 0.000, "a_0": -0.750, "a_2": 1.750, "a_3": 0.000,  # index
        "a_5": 0.000, "a_4": 0.000, "a_6": 1.750, "a_7": 0.000,  # middle
        "a_9": 0.000, "a_8": 0.750, "a_10": 1.750, "a_11": 0.000,  # ring
        "a_12": 0.500, "a_13": 1.300, "a_14": 1.500, "a_15": 1.000,  # thumb
    }
    
    注意：如果提供此参数，将忽略 nominal_fingertip_config。
    """
    
    nominal_fingertip_config: tuple[tuple[float, ...], ...] | None = (
        (0.05, -0.03, 0.10),  # index finger
        (0.05,  0.00, 0.12),  # middle finger
        (0.05,  0.03, 0.10),  # ring finger
        (-0.03, 0.00, 0.08),  # thumb
    )
    """标称指尖位置 P* (4指 × 3坐标)，相对于手部基座的局部坐标系（备用方式）
    
    直接指定指尖位置坐标。如果提供了 nominal_joint_angles，此参数将被忽略。
    仅在无法提供关节角度时使用此方式。
    """
    
    # ========== 仿射变换限制 ==========
    rotation_limit: float = 0.5
    """旋转切向量的范数限制（单位：rad），限制旋转角度在±28.6度内"""
    
    scale_range: tuple[float, float] = (0.7, 1.3)
    """缩放因子的允许范围，防止过度缩放或反转"""
    
    translation_limit: float = 0.05
    """平移向量每个分量的限制（单位：m），防止指尖偏移过远"""
    
    # ========== IK控制器配置 ==========
    ik_method: Literal["pinv", "svd", "trans", "dls"] = "dls"
    """IK求解方法：
    - "pinv": Moore-Penrose伪逆
    - "svd": 奇异值分解（自适应）
    - "trans": 雅可比转置
    - "dls": 阻尼最小二乘（推荐，最稳定）
    """
    
    ik_params: dict[str, float] = None
    """IK方法参数，None时使用默认值
    - dls默认: {"lambda_val": 0.05}  # 阻尼系数
    """
    
    # ========== 手指关节映射 ==========
    finger_joints: dict[str, list[str]] = None
    """每根手指的关节名称映射，None时使用默认LeapHand配置"""
    
    finger_bodies: tuple[str, ...] = (
        "fingertip", "fingertip_2", "fingertip_3", "thumb_fingertip"
    )
    """指尖刚体名称列表（顺序：食指、中指、无名指、拇指）"""
    
    # ========== 坐标系配置 ==========
    use_body_frame: bool = True
    """是否在手部基座局部坐标系下表达指尖位置
    
    True: 位置相对于robot.data.root_pos_w（推荐）
    False: 位置相对于环境原点
    """
    
    def __post_init__(self):
        """配置验证和默认值设置"""
        # 设置默认手指关节映射（LeapHand标准配置）
        if self.finger_joints is None:
            self.finger_joints = {
                "index": ["a_1", "a_0", "a_2", "a_3"],
                "middle": ["a_5", "a_4", "a_6", "a_7"],
                "ring": ["a_9", "a_8", "a_10", "a_11"],
                "thumb": ["a_12", "a_13", "a_14", "a_15"],
            }
        
        # 验证手指数量匹配
        if len(self.nominal_fingertip_config) != len(self.finger_bodies):
            raise ValueError(
                f"标称构型指尖数量 ({len(self.nominal_fingertip_config)}) "
                f"与finger_bodies数量 ({len(self.finger_bodies)}) 不匹配"
            )
        
        # 设置IK默认参数
        if self.ik_params is None:
            if self.ik_method == "dls":
                self.ik_params = {"lambda_val": 0.05}  # 增加阻尼提高稳定性
            elif self.ik_method == "pinv":
                self.ik_params = {"k_val": 0.5}
            elif self.ik_method == "svd":
                self.ik_params = {"k_val": 0.5, "min_singular_value": 1e-5}
            elif self.ik_method == "trans":
                self.ik_params = {"k_val": 0.5}
