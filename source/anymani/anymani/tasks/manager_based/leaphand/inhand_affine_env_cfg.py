# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
仿射编队对比实验环境配置

基于inhand_base_env_cfg.py创建，仅修改动作空间为仿射编队控制。
其他所有组件(observations, rewards, events等)保持不变，确保对比实验的公平性。
"""

from isaaclab.utils import configclass
from . import inhand_base_env_cfg
from . import mdp as leap_mdp


@configclass
class AffineActionsCfg:
    """仿射编队动作配置"""
    
    affine_formation = leap_mdp.AffineFormationActionCfg(
        asset_name="robot",
        # 通过关节角度定义标称构型（推荐方式）
        nominal_joint_angles={
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
        },
        # 仿射变换限制
        rotation_limit=0.5,  # rad
        scale_range=(0.7, 1.3),
        translation_limit=0.05,  # m
        # IK配置
        ik_method="dls",
        ik_params={"lambda_val": 0.05},
        # 手指配置（使用LeapHand默认映射）
        finger_joints={
            "index": ["a_1", "a_0", "a_2", "a_3"],
            "middle": ["a_5", "a_4", "a_6", "a_7"],
            "ring": ["a_9", "a_8", "a_10", "a_11"],
            "thumb": ["a_12", "a_13", "a_14", "a_15"],
        },
        finger_bodies=(
            "fingertip", "fingertip_2", "fingertip_3", "thumb_fingertip"
        ),
        use_body_frame=True,
    )


@configclass
class InHandAffineEnvCfg(inhand_base_env_cfg.InHandObjectEnvCfg):
    """仿射编队对比实验环境配置
    
    该环境与baseline (inhand_base_env_cfg.InHandObjectEnvCfg) 的唯一区别：
    - Baseline: 16维关节空间动作（RelativeJointPositionActionCfg）
    - Affine: 9维仿射编队动作（AffineFormationActionCfg）
    
    其他所有配置保持一致，确保实验对比的公平性。
    """
    
    # 替换动作配置为仿射编队
    actions: AffineActionsCfg = AffineActionsCfg()
    
    def __post_init__(self):
        """后初始化 - 继承父类所有配置"""
        super().__post_init__()
        # 所有配置已由父类处理，这里无需额外操作
