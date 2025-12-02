# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""LeapHand连续旋转任务环境配置 - ManagerBasedRLEnv架构

主要是动作空间从关节空间被改为se3动作空间

"""

from isaaclab.managers import RewardTermCfg as RewTerm

from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
from . import mdp as leap_mdp
from . import inhand_base_env_cfg


@configclass
class ActionsCfg:
    """动作配置 - SE(3) 旋量动作空间
    
    每根手指配置独立的 se(3) 动作项，共4根手指 × 6维旋量 = 24维动作空间。
    
    Note:
        虚拟Xform到父刚体的映射关系：
        - index_tip_head  → fingertip       (食指)
        - middle_tip_head → fingertip_2     (中指)
        - ring_tip_head   → fingertip_3     (无名指)
        - thumb_tip_head  → thumb_fingertip (拇指)
    """
    index_se3 = leap_mdp.se3dlsActionsCfg(
        asset_name="robot",
        joint_names=["a_1", "a_0", "a_2", "a_3"],
        preserve_order=True,
        is_xform=True,
        target="index_tip_head",
        parent="fingertip",  # 食指末端刚体
        use_pd=True,
        angular_limits=2,
        linear_limits=0.2356,  # 估算逻辑为：指长0.15m左右，设每秒最多沿圆周转90度，则线速度约0.15*π/2=0.2356m/s
        damping=0.01,
        use_joint_limits=True,
    )
    middle_se3 = leap_mdp.se3dlsActionsCfg(
        asset_name="robot",
        joint_names=["a_5", "a_4", "a_6", "a_7"],
        preserve_order=True,
        is_xform=True,
        target="middle_tip_head",
        parent="fingertip_2",  # 中指末端刚体
        use_pd=True,
        angular_limits=2,
        linear_limits=0.2356,
        damping=0.01,
        use_joint_limits=True,
    )
    ring_se3 = leap_mdp.se3dlsActionsCfg(
        asset_name="robot",
        joint_names=["a_9", "a_8", "a_10", "a_11"],
        preserve_order=True,
        is_xform=True,
        target="ring_tip_head",
        parent="fingertip_3",  # 无名指末端刚体
        use_pd=True,
        angular_limits=2,
        linear_limits=0.2356,
        damping=0.01,
        use_joint_limits=True,
    )
    thumb_se3 = leap_mdp.se3dlsActionsCfg(
        asset_name="robot",
        joint_names=["a_12", "a_13", "a_14", "a_15"],
        preserve_order=True,
        is_xform=True,
        target="thumb_tip_head",
        parent="thumb_fingertip",  # 拇指末端刚体
        use_pd=True,
        angular_limits=2,
        linear_limits=0.2356,
        damping=0.01,
        use_joint_limits=True,
    )


@configclass
class RewardCfg(inhand_base_env_cfg.RewardsCfg):
    """奖励配置"""
    # TODO:这里到时候要加一些适配se3动作的奖励
    se3 = RewTerm(
    )

    # def __post_init__(self):
    #     super().__post_init__()
    


@configclass
class InHandse3EnvCfg(inhand_base_env_cfg.InHandObjectEnvCfg):
    """LeapHand连续旋转任务环境配置 - 使用se3相对刚体末端旋量动作空间"""
    actions: ActionsCfg = ActionsCfg()
    reward: RewardCfg = RewardCfg()
    def __post_init__(self):
        # post init of parent
        super().__post_init__()