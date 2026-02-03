# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LeapHand 手内操作任务环境配置

本模块定义所有 LeapHand 手型的环境变体，通过继承通用基类
并在 __post_init__ 中修改特定配置来实现不同的动作空间和观测设置。

环境变体（训练）:
    - LeapHandJointEnvCfg: 关节空间动作（16 维）
    - LeapHandSe3EnvCfg: SE(3) 旋量动作（24 维）
    - LeapHandTactileEnvCfg: 关节空间 + 触觉观测
    - LeapHandSe3TactileEnvCfg: SE(3) + 触觉观测
    - LeapHandAffineEnvCfg: 仿射编队动作（9 维）

环境变体（Play/可视化）:
    - LeapHandJointEnvCfg_PLAY
    - LeapHandSe3EnvCfg_PLAY
    - LeapHandTactileEnvCfg_PLAY
    - LeapHandSe3TactileEnvCfg_PLAY
    - LeapHandAffineEnvCfg_PLAY

Usage:
    from anymani.tasks.inhand.config.leaphand import LeapHandJointEnvCfg
"""

from __future__ import annotations

import math

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.ui import ManagerBasedRLEnvWindow
from isaaclab.envs.common import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

# 导入通用 MDP 组件
from anymani.tasks.inhand.inhand_env_cfg import (
    # 场景
    InHandObjectSceneCfg,
    TactileSceneCfg,
    # 观测
    JointSpaceObservationsCfg,
    Se3ObservationsCfg,
    TactileObsGroupCfg,
    TactileCriticObsGroupCfg,
    TactileObservationsCfg,
    Se3TactileObservationsCfg,
    # 动作
    JointSpaceActionsCfg,
    Se3ActionsCfg,
    Se3EmaActionsCfg,
    AffineActionsCfg,
    # 奖励
    CommonRewardsCfg,
    Se3RewardsCfg,
    TactileRewardsCfg,
    Se3TactileRewardsCfg,
    # 事件
    CommonEventCfg,
    # 终止
    CommonTerminationsCfg,
    # 命令
    ContinuousRotationCommandsCfg,
    # 课程
    EmptyCurriculumCfg,
    # 触觉超参数
    TACTILE_FORCE_THRESHOLD,
    TACTILE_CONTACT_REWARD_TYPE,
    TACTILE_USE_REWARD_CURRICULUM,
    TACTILE_CURRICULUM_METRIC_KEY,
    TACTILE_G_MIN,
    TACTILE_G_MAX,
)
from anymani.robots.leap import LEAP_HAND_CFG


##############################################################################
# LeapHand 场景配置
##############################################################################

@configclass
class LeapHandSceneCfg(InHandObjectSceneCfg):
    """LeapHand 场景配置
    
    继承通用场景，指定 LeapHand 机器人及其初始姿态。
    """
    
    robot: ArticulationCfg = LEAP_HAND_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(0.5, 0.5, -0.5, 0.5),  # 手掌朝上
            joint_pos={
                "a_1": 0.000, "a_12": 0.500, "a_5": 0.000, "a_9": 0.000,
                "a_0": -0.750, "a_13": 1.300, "a_4": 0.000, "a_8": 0.750,
                "a_2": 1.750, "a_14": 1.500, "a_6": 1.750, "a_10": 1.750,
                "a_3": 0.000, "a_15": 1.000, "a_7": 0.000, "a_11": 0.000,
            },
            joint_vel={"a_.*": 0.0},
        ),
    )


@configclass
class LeapHandTactileSceneCfg(TactileSceneCfg):
    """LeapHand 触觉场景配置
    
    在触觉场景基础上指定 LeapHand 机器人。

    注意：该场景只包含指尖 + 手掌的接触传感器（见 :class:`TactileSceneCfg`）。
    如需包含所有关节（非指尖）的接触传感器，请使用 :class:`LeapHandFullTactileSceneCfg`。
    """
    
    robot: ArticulationCfg = LEAP_HAND_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(0.5, 0.5, -0.5, 0.5),
            joint_pos={
                "a_1": 0.000, "a_12": 0.500, "a_5": 0.000, "a_9": 0.000,
                "a_0": -0.750, "a_13": 1.300, "a_4": 0.000, "a_8": 0.750,
                "a_2": 1.750, "a_14": 1.500, "a_6": 1.750, "a_10": 1.750,
                "a_3": 0.000, "a_15": 1.000, "a_7": 0.000, "a_11": 0.000,
            },
            joint_vel={"a_.*": 0.0},
        ),
    )


##############################################################################
# 完整触觉场景配置（包含所有关节接触传感器）
##############################################################################

from isaaclab.sensors import ContactSensorCfg


@configclass
class LeapHandFullTactileSceneCfg(LeapHandTactileSceneCfg):
    """LeapHand 完整触觉场景配置
    
    在基础触觉场景上添加所有关节的接触传感器，用于检测非期望接触。
    """
    
    # 禁用物理复制以支持域随机化
    replicate_physics = False
    
    # ===== 食指关节（非指尖）=====
    contact_index_mcp = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/mcp_joint",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
        update_period=0.0,
        max_contact_data_count_per_prim=64,
        force_threshold=0.5,
        track_friction_forces=True,
        debug_vis=False,
    )
    contact_index_pip = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pip",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
        update_period=0.0,
        max_contact_data_count_per_prim=64,
        force_threshold=0.5,
        track_friction_forces=True,
        debug_vis=False,
    )
    contact_index_dip = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/dip",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
        update_period=0.0,
        max_contact_data_count_per_prim=64,
        force_threshold=0.5,
        track_friction_forces=True,
        debug_vis=False,
    )
    
    # ===== 中指关节（非指尖）=====
    contact_middle_mcp = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/mcp_joint_2",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
        update_period=0.0,
        max_contact_data_count_per_prim=64,
        force_threshold=0.5,
        track_friction_forces=True,
        debug_vis=False,
    )
    contact_middle_pip = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pip_2",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
        update_period=0.0,
        max_contact_data_count_per_prim=64,
        force_threshold=0.5,
        track_friction_forces=True,
        debug_vis=False,
    )
    contact_middle_dip = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/dip_2",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
        update_period=0.0,
        max_contact_data_count_per_prim=64,
        force_threshold=0.5,
        track_friction_forces=True,
        debug_vis=False,
    )
    
    # ===== 无名指关节（非指尖）=====
    contact_ring_mcp = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/mcp_joint_3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
        update_period=0.0,
        max_contact_data_count_per_prim=64,
        force_threshold=0.5,
        track_friction_forces=True,
        debug_vis=False,
    )
    contact_ring_pip = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pip_3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
        update_period=0.0,
        max_contact_data_count_per_prim=64,
        force_threshold=0.5,
        track_friction_forces=True,
        debug_vis=False,
    )
    contact_ring_dip = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/dip_3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
        update_period=0.0,
        max_contact_data_count_per_prim=64,
        force_threshold=0.5,
        track_friction_forces=True,
        debug_vis=False,
    )
    
    # ===== 拇指关节（非指尖）=====
    contact_thumb_base = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/thumb_temp_base",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
        update_period=0.0,
        max_contact_data_count_per_prim=64,
        force_threshold=0.5,
        track_friction_forces=True,
        debug_vis=False,
    )
    contact_thumb_pip = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/thumb_pip",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
        update_period=0.0,
        max_contact_data_count_per_prim=64,
        force_threshold=0.5,
        track_friction_forces=True,
        debug_vis=False,
    )
    contact_thumb_dip = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/thumb_dip",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
        update_period=0.0,
        max_contact_data_count_per_prim=64,
        force_threshold=0.5,
        track_friction_forces=True,
        debug_vis=False,
    )


##############################################################################
# 环境配置类
##############################################################################

@configclass
class LeapHandJointEnvCfg(ManagerBasedRLEnvCfg):
    """LeapHand 关节空间环境配置（Baseline）
    
    使用 16 维关节位置动作空间，适合作为基准对比。
    
    动作空间: 16 维（4 根手指 × 4 关节）
    观测空间: 关节位置 + 物体位姿 + 目标位姿
    """
    
    ui_window_class_type: type | None = ManagerBasedRLEnvWindow
    is_finite_horizon: bool = True
    
    # 场景配置
    scene: InteractiveSceneCfg = LeapHandSceneCfg(
        num_envs=4096, env_spacing=0.75, replicate_physics=False
    )
    viewer: ViewerCfg = ViewerCfg()
    sim: SimulationCfg = SimulationCfg(
        physics_material=RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
        ),
    )
    seed: int | None = 42
    
    # MDP 配置
    observations: JointSpaceObservationsCfg = JointSpaceObservationsCfg()
    actions: JointSpaceActionsCfg = JointSpaceActionsCfg()
    commands: ContinuousRotationCommandsCfg = ContinuousRotationCommandsCfg()
    rewards: CommonRewardsCfg = CommonRewardsCfg()
    terminations: CommonTerminationsCfg = CommonTerminationsCfg()
    events: CommonEventCfg = CommonEventCfg()
    curriculum: EmptyCurriculumCfg = EmptyCurriculumCfg()
    
    def __post_init__(self):
        """后初始化"""
        super().__post_init__()
        self.decimation = 4
        self.episode_length_s = 30.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (2.0, 2.0, 2.0)


@configclass
class LeapHandSe3EnvCfg(LeapHandJointEnvCfg):
    """LeapHand SE(3) 动作空间环境配置
    
    使用 24 维 SE(3) 旋量动作空间（每根手指 6 维）。
    
    动作空间: 24 维（4 根手指 × 6 维旋量）
    观测空间: 末端旋量 + 物体位姿 + 目标位姿
    """
    
    actions: Se3ActionsCfg = Se3ActionsCfg()
    observations: Se3ObservationsCfg = Se3ObservationsCfg()
    rewards: Se3RewardsCfg = Se3RewardsCfg()
    
    def __post_init__(self):
        super().__post_init__()


@configclass
class LeapHandTactileEnvCfg(LeapHandJointEnvCfg):
    """LeapHand 触觉增强环境配置
    
    在关节空间基础上添加触觉观测和奖励。
    
    动作空间: 16 维
    观测空间: 关节位置 + 触觉信号 + 物体位姿 + 目标位姿
    """
    
    scene: InteractiveSceneCfg = LeapHandTactileSceneCfg(
        num_envs=4096, env_spacing=0.6, replicate_physics=False
    )
    observations: TactileObservationsCfg = TactileObservationsCfg()
    rewards: TactileRewardsCfg = TactileRewardsCfg()
    
    def __post_init__(self):
        super().__post_init__()


@configclass
class LeapHandSe3TactileEnvCfg(LeapHandJointEnvCfg):
    """LeapHand SE(3) + 触觉环境配置
    
    结合 SE(3) 动作空间和触觉传感，适合 sim2real 迁移。
    
    动作空间: 24 维（4 根手指 × 6 维旋量）
    观测空间: 末端旋量 + 触觉信号 + 目标位姿（带历史）
    """
    
    scene: InteractiveSceneCfg = LeapHandFullTactileSceneCfg(
        num_envs=4096, env_spacing=0.6, replicate_physics=False
    )
    actions: Se3EmaActionsCfg = Se3EmaActionsCfg()
    observations: Se3TactileObservationsCfg = Se3TactileObservationsCfg()
    rewards: Se3TactileRewardsCfg = Se3TactileRewardsCfg()
    
    def __post_init__(self):
        super().__post_init__()


@configclass
class LeapHandAffineEnvCfg(LeapHandJointEnvCfg):
    """LeapHand 仿射编队环境配置
    
    使用 9 维仿射编队动作空间控制指尖编队形状。
    
    动作空间: 9 维（旋转 3 + 缩放 3 + 平移 3）
    观测空间: 关节位置 + 物体位姿 + 目标位姿
    """
    
    actions: AffineActionsCfg = AffineActionsCfg()
    
    def __post_init__(self):
        super().__post_init__()


##############################################################################
# Play 配置（用于可视化和评估）
##############################################################################

@configclass
class LeapHandJointEnvCfg_PLAY(LeapHandJointEnvCfg):
    """关节空间环境 Play 配置"""
    
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
        self.terminations.time_out = None


@configclass
class LeapHandSe3EnvCfg_PLAY(LeapHandSe3EnvCfg):
    """SE(3) 环境 Play 配置"""
    
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
        self.terminations.time_out = None


@configclass
class LeapHandTactileEnvCfg_PLAY(LeapHandTactileEnvCfg):
    """触觉环境 Play 配置"""
    
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
        self.terminations.time_out = None


@configclass
class LeapHandSe3TactileEnvCfg_PLAY(LeapHandSe3TactileEnvCfg):
    """SE(3) + 触觉环境 Play 配置"""
    
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
        self.terminations.time_out = None


@configclass
class LeapHandAffineEnvCfg_PLAY(LeapHandAffineEnvCfg):
    """仿射编队环境 Play 配置"""
    
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
        self.terminations.time_out = None
