# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""LeapHand连续旋转任务环境配置 - ManagerBasedRLEnv架构

- 主要是动作空间从关节空间被改为se3动作空间，奖励和观察组件也改为适配se3动作空间的版本

"""

import math
from shlex import join

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg

from isaaclab.managers import RecorderManagerBaseCfg as DefaultEmptyRecorderManagerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise

from isaaclab.envs.ui import ManagerBasedRLEnvWindow
from isaaclab.envs.common import ViewerCfg
from isaaclab.devices.openxr import XrCfg

import isaaclab.envs.mdp as mdp
from leaphand.robots.leap import LEAP_HAND_CFG
from . import mdp as leap_mdp
from . import inhand_base_env_cfg


@configclass
class se3awdlsActionsCfg:
    """动作配置 - SE(3) 旋量动作空间
    
    每根手指配置独立的 se(3) 动作项，共4根手指 × 6维旋量 = 24维动作空间。
    
    Note:
        虚拟Xform到父刚体的映射关系：
        - index_tip_head  → fingertip       (食指)
        - middle_tip_head → fingertip_2     (中指)
        - ring_tip_head   → fingertip_3     (无名指)
        - thumb_tip_head  → thumb_fingertip (拇指)
    """
    index_se3 = leap_mdp.se3awdlsActionsCfg(
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
        singular_threshold=0.05,
        W_x=[1, 1, 1, 1, 1, 1],
    )
    middle_se3 = leap_mdp.se3awdlsActionsCfg(
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
        singular_threshold=0.05,
        W_x=[1, 1, 1, 1, 1, 1],
    )
    ring_se3 = leap_mdp.se3awdlsActionsCfg(
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
        singular_threshold=0.05,
        W_x=[1, 1, 1, 1, 1, 1],
    )
    thumb_se3 = leap_mdp.se3awdlsActionsCfg(
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
        singular_threshold=0.05,
        W_x=[1, 1, 1, 1, 1, 1],
    )

@configclass
class se3dlsActionsCfg:
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
class ObservationsCfg:
    """观测配置 - 支持非对称Actor-Critic"""

    @configclass
    class PrivilegedObsCfg(ObsGroup):
        """Actor策略观测 - 包含大量仅仿真可用的特权信息"""
        # -- robot terms
        joint_pos = ObsTerm(
            func=mdp.joint_pos_limit_normalized,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        body_twists = ObsTerm(
            func=leap_mdp.body_twists,
            params={"asset_cfg": SceneEntityCfg("robot"), "action_names": ["index_se3", "middle_se3", "ring_se3", "thumb_se3"]},
        )

        # -- object terms
        object_pos = ObsTerm(func=mdp.root_pos_w, noise=Gnoise(std=0.002), params={"asset_cfg": SceneEntityCfg("object")})
        object_quat = ObsTerm( # IDEA:该项添加噪音可能会破坏归一化约束？
            func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object"), "make_quat_unique": False}
        )

        # -- command terms
        goal_pose = ObsTerm(func=mdp.generated_commands, params={"command_name": "goal_pose"})
        goal_quat_diff = ObsTerm(
            func=leap_mdp.goal_quat_diff,
            params={"asset_cfg": SceneEntityCfg("object"), "command_name": "goal_pose", "make_quat_unique": True},
        )

        # -- action terms
        last_action = ObsTerm(func=mdp.last_action) # 返回的是 策略输出的规范化后值（通常是 -1 到 1）动作步a_{t-1}

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class ProprioceptionObsCfg(PrivilegedObsCfg):
        """Actor策略观测 - 仅包含灵巧手本体感受的信息"""
        # 仅保留关节位置信息，上一步动作信息，目标位姿信息
        def __post_init__(self):
            super().__post_init__()
            self.object_pos = None
            self.object_quat = None
            self.goal_quat_diff = None

    @configclass
    class CriticCfg(PrivilegedObsCfg):
        """Critic价值函数观测 - 包含大量仅仿真可用的特权信息"""

    # 观测组配置
    policy: ObsGroup = PrivilegedObsCfg(history_length=2)
    critic: ObsGroup = CriticCfg(history_length=2)


@configclass
class RewardsCfg:
    """奖励配置 - 连续旋转任务奖励机制"""

    # -- task
    track_orientation_inv_l2 = RewTerm(
        func=leap_mdp.track_orientation_inv_l2, weight=1.0,
        params={"object_cfg": SceneEntityCfg("object"), "rot_eps": 0.1, "command_name": "goal_pose"},
    )
    goal_position_distance = RewTerm(
        func=leap_mdp.goal_position_distance, weight=-10.0,
        params={"object_cfg": SceneEntityCfg("object"), "command_name": "goal_pose"},
    )
    success_bonus = RewTerm(
        func=leap_mdp.success_bonus, weight=250.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "command_name": "goal_pose",
            "orientation_threshold": 0.2,
            "position_threshold": 0.025,
        },
    )
    fingertip_distance = RewTerm(
        func=leap_mdp.fingertip_distance_penalty, weight=-2.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfg": SceneEntityCfg("object"),
            "fingertip_body_names": ["fingertip", "thumb_fingertip", "fingertip_2", "fingertip_3"],
        },
    )
    fall_penalty = RewTerm(
        func=leap_mdp.fall_penalty, weight=-10.0,
        params={"object_cfg": SceneEntityCfg("object"), "command_name": "goal_pose", "fall_distance": 0.07},
    )
    pose_diff = RewTerm(func=leap_mdp.pose_diff_penalty, weight=-0.3)

    # -- action
    manipulability = RewTerm(
        func=leap_mdp.jacobian_manipulability, weight=0.1,  # 按照最大可操作度约10来设定权重（奖励为0.25），鼓励手指保持良好可操作度
        params={"action_names": ["index_se3", "middle_se3", "ring_se3", "thumb_se3"]},
    )
    kinetic_energy = RewTerm(  # 动能
        func=leap_mdp.se3_kinetic_energy, weight=-1,
        params={"action_names": ["index_se3", "middle_se3", "ring_se3", "thumb_se3"]},
    )
    action_smooth = RewTerm(
        func=leap_mdp.se3_action_smooth, weight=-1, # 鼓励动作平滑
        params={"action_names": ["index_se3", "middle_se3", "ring_se3", "thumb_se3"],
                "use_processed": False, "norm": 1},
    )


@configclass
class CurriculumCfg:
    """课程学习配置 - 提供各种课程学习策略"""
    
    pass


@configclass
class InHandse3EnvCfg(inhand_base_env_cfg.InHandObjectEnvCfg):
    """LeapHand连续旋转任务环境配置 - 使用se3相对刚体末端旋量动作空间"""
    actions: se3dlsActionsCfg = se3dlsActionsCfg()
    # actions: se3awdlsActionsCfg = se3awdlsActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    def __post_init__(self):
        # post init of parent
        super().__post_init__()