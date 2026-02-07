# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AnyMani In-Hand 操作任务的 MDP 组件库

本模块提供可复用的 MDP 配置组件，用于组装各种环境变体。
设计目标是将 Obs/Actions/Rewards 等配置解耦，便于灵活组合。

Usage:
    from anymani.tasks.inhand.inhand_env_cfg import (
        JointSpaceObservationsCfg,
        JointSpaceActionsCfg,
        CommonRewardsCfg,
    )

组件分类:
    - Scene: 场景配置（地面、光照、物体）
    - Observations: 观测配置（关节空间/SE3/触觉）
    - Actions: 动作空间配置（关节/SE3/仿射编队）
    - Rewards: 奖励配置（通用/任务特定）
    - Events: 域随机化配置
    - Terminations: 终止条件

Note:
    各手型的具体环境配置（如 LeapHandJointEnvCfg）在 config/<hand_type>/ 下定义，
    它们通过组合本模块的组件来构建完整环境。
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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
from isaaclab.sensors import ContactSensorCfg

import isaaclab.envs.mdp as mdp
from anymani.tasks.inhand import mdp as leap_mdp


##############################################################################
# 全局超参数
##############################################################################

# rl_games PPO 配置中的 horizon_length 和 epochs_num，
# 用于控制某些随机化事件的最小间隔
HORIZON_LENGTH = 32
EPOCHS_NUM = 5

# 默认物体 USD 资产路径
DEFAULT_OBJECT_USD = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"


##############################################################################
# 场景配置组件
##############################################################################

@configclass
class InHandObjectSceneCfg(InteractiveSceneCfg):
    """手内操作任务的基础场景配置
    
    包含地面、物体、光照等共用场景元素。
    机器人（robot）由子类或具体手型配置指定。
    """

    # ===== 地面 =====
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.1)),
    )

    # ===== 被操作物体 =====
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=DEFAULT_OBJECT_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
            scale=(1.2, 1.2, 1.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, -0.1, 0.56),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # ===== 光照 =====
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


##############################################################################
# 观测配置组件
##############################################################################

@configclass
class JointSpaceObsGroupCfg(ObsGroup):
    """关节空间观测组
    
    适用于关节位置控制的任务，观测包括：
    - 关节位置（归一化到限位范围）
    - 物体位姿
    - so(3) 指令（3D rotvec）
    - 上一步动作
    """
    
    # -- robot terms
    joint_pos = ObsTerm(
        func=mdp.joint_pos_limit_normalized,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # -- object terms
    object_pos = ObsTerm(
        func=mdp.root_pos_w,
        noise=Gnoise(std=0.002),
        params={"asset_cfg": SceneEntityCfg("object")},
    )
    object_quat = ObsTerm(
        func=mdp.root_quat_w,
        params={"asset_cfg": SceneEntityCfg("object"), "make_quat_unique": False},
    )

    # -- command terms
    so3_command = ObsTerm(
        func=leap_mdp.so3_command,
        params={"command_name": "goal_pose"},
    )

    # NOTE:
    #   位置目标（pos_command_e）是命令项提供的“别乱跑”约束信号。
    #   为了保持 policy 观测可部署（不依赖物体绝对位姿），该项仅在 Critic/特权观测组中启用。
    pos_command = ObsTerm(
        func=leap_mdp.pos_command,
        params={"command_name": "goal_pose"},
    )

    # -- action terms
    last_action = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class ProprioceptionObsGroupCfg(JointSpaceObsGroupCfg):
    """本体感受观测组（无视觉/物体信息）
    
    仅包含机器人自身可感知的信息，适用于 sim2real 场景。
    """
    
    def __post_init__(self):
        super().__post_init__()
        # 移除需要外部感知的项
        self.object_pos = None
        self.object_quat = None
        # so3_command 是可部署指令输入，保留
        # pos_command 属于特权信息（目标位置约束），仅在 Critic/特权观测中启用
        self.pos_command = None


@configclass
class Se3ObsGroupCfg(ObsGroup):
    """SE(3) 旋量观测组
    
    适用于 SE(3) 动作空间的任务，观测包括：
    - 指尖刚体旋量（body_twists）
    - 物体位姿
    - so(3) 指令（3D rotvec）
    - 上一步动作
    
    Note:
        action_names 参数需要与 ActionsCfg 中的 se3 动作项名称一致
    """
    
    # -- robot terms (SE3 specific)
    body_twists = ObsTerm(
        func=leap_mdp.body_twists,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "action_names": ["index_se3", "middle_se3", "ring_se3", "thumb_se3"],
            "use_body_frame": False,
        },
    )

    # -- object terms
    object_pos = ObsTerm(
        func=mdp.root_pos_w,
        noise=Gnoise(std=0.002),
        params={"asset_cfg": SceneEntityCfg("object")},
    )
    object_quat = ObsTerm(
        func=mdp.root_quat_w,
        params={"asset_cfg": SceneEntityCfg("object"), "make_quat_unique": False},
    )

    # -- command terms
    so3_command = ObsTerm(
        func=leap_mdp.so3_command,
        params={"command_name": "goal_pose"},
    )

    # NOTE: 同 JointSpaceObsGroupCfg，仅用于 Critic/特权观测。
    pos_command = ObsTerm(
        func=leap_mdp.pos_command,
        params={"command_name": "goal_pose"},
    )

    # -- action terms
    last_action = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class Se3ProprioceptionObsGroupCfg(Se3ObsGroupCfg):
    """SE(3) 本体感受观测组（无物体绝对位姿）。

    用于 sim2real：policy 不依赖物体的绝对位置/姿态，只读本体感受与 so(3) 指令。
    """

    def __post_init__(self):
        super().__post_init__()
        self.object_pos = None
        self.object_quat = None
        # pos_command 属于特权信息（目标位置约束），仅在 Critic/特权观测中启用
        self.pos_command = None


@configclass
class JointSpaceObservationsCfg:
    """关节空间完整观测配置（Policy + Critic）"""
    
    @configclass
    class PolicyCfg(ProprioceptionObsGroupCfg):
        """策略观测（可部署）：无物体绝对位姿"""
        pass
    
    @configclass
    class CriticCfg(JointSpaceObsGroupCfg):
        """Critic 观测（特权信息）"""
        pass
    
    policy: ObsGroup = PolicyCfg(history_length=1)
    critic: ObsGroup = CriticCfg(history_length=1)


@configclass
class Se3ObservationsCfg:
    """SE(3) 动作空间完整观测配置（Policy + Critic）"""
    
    @configclass
    class PolicyCfg(Se3ProprioceptionObsGroupCfg):
        """策略观测（可部署）：无物体绝对位姿"""
        pass
    
    @configclass
    class CriticCfg(Se3ObsGroupCfg):
        """Critic 观测（特权信息）"""
        pass
    
    policy: ObsGroup = PolicyCfg(history_length=3)
    critic: ObsGroup = CriticCfg(history_length=3)


##############################################################################
# 动作配置组件
##############################################################################

@configclass
class JointSpaceActionsCfg:
    """关节空间动作配置
    
    使用相对关节位置控制，动作范围 [-1, 1] 映射到关节增量。
    """
    hand_joint_pos = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "a_1", "a_0", "a_2", "a_3",      # 食指
            "a_5", "a_4", "a_6", "a_7",      # 中指
            "a_9", "a_8", "a_10", "a_11",    # 无名指
            "a_12", "a_13", "a_14", "a_15",  # 拇指
        ],
        scale=1 / 10,
        preserve_order=True,
    )


@configclass
class Se3ActionsCfg:
    """SE(3) 旋量动作配置
    
    每根手指独立的 se(3) 动作，共 4 × 6 = 24 维动作空间。
    使用阻尼最小二乘（DLS）逆运动学求解关节增量。
    """
    index_se3 = leap_mdp.se3dlsActionsCfg(
        asset_name="robot",
        joint_names=["a_1", "a_0", "a_2", "a_3"],
        preserve_order=True,
        is_xform=True,
        use_body_frame=False,
        target="index_tip_head",
        parent="fingertip",
        use_pd=True,
        angular_limits=2,
        linear_limits=0.2356,
        damping=0.01,
        use_joint_limits=True,
    )
    middle_se3 = leap_mdp.se3dlsActionsCfg(
        asset_name="robot",
        joint_names=["a_5", "a_4", "a_6", "a_7"],
        preserve_order=True,
        is_xform=True,
        use_body_frame=False,
        target="middle_tip_head",
        parent="fingertip_2",
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
        use_body_frame=False,
        target="ring_tip_head",
        parent="fingertip_3",
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
        use_body_frame=False,
        target="thumb_tip_head",
        parent="thumb_fingertip",
        use_pd=True,
        angular_limits=2,
        linear_limits=0.2356,
        damping=0.01,
        use_joint_limits=True,
    )


##############################################################################
# SE(3) + EMA 动作配置
##############################################################################

# NOTE:
# 这里的常量服务于 Se3EmaActionsCfg / Se3TactileObservationsCfg 的内部配置。
# 它们属于“可复用 MDP 组件”的一部分，因此放在 inhand_env_cfg.py 中统一维护。
USE_BODY_FRAME_STUDENT = True
ENCODER_HISTORY_LENGTH = 50


@configclass
class Se3EmaActionsCfg:
    """SE(3) 旋量动作 + EMA 平滑

    每根手指独立的 se(3) 动作，共 4 × 6 = 24 维动作空间。
    添加 EMA 平滑以减少动作抖动。
    """

    index_se3 = leap_mdp.se3dlsEmaActionsCfg(
        asset_name="robot",
        joint_names=["a_1", "a_0", "a_2", "a_3"],
        preserve_order=True,
        is_xform=True,
        use_body_frame=USE_BODY_FRAME_STUDENT,
        target="index_tip_head",
        parent="fingertip",
        use_pd=True,
        angular_limits=2,
        linear_limits=0.2356,
        damping=0.01,
        use_joint_limits=True,
        alpha=1 / 24,
    )
    middle_se3 = leap_mdp.se3dlsEmaActionsCfg(
        asset_name="robot",
        joint_names=["a_5", "a_4", "a_6", "a_7"],
        preserve_order=True,
        is_xform=True,
        use_body_frame=USE_BODY_FRAME_STUDENT,
        target="middle_tip_head",
        parent="fingertip_2",
        use_pd=True,
        angular_limits=2,
        linear_limits=0.2356,
        damping=0.01,
        use_joint_limits=True,
        alpha=1 / 24,
    )
    ring_se3 = leap_mdp.se3dlsEmaActionsCfg(
        asset_name="robot",
        joint_names=["a_9", "a_8", "a_10", "a_11"],
        preserve_order=True,
        is_xform=True,
        use_body_frame=USE_BODY_FRAME_STUDENT,
        target="ring_tip_head",
        parent="fingertip_3",
        use_pd=True,
        angular_limits=2,
        linear_limits=0.2356,
        damping=0.01,
        use_joint_limits=True,
        alpha=1 / 24,
    )
    thumb_se3 = leap_mdp.se3dlsEmaActionsCfg(
        asset_name="robot",
        joint_names=["a_12", "a_13", "a_14", "a_15"],
        preserve_order=True,
        is_xform=True,
        use_body_frame=USE_BODY_FRAME_STUDENT,
        target="thumb_tip_head",
        parent="thumb_fingertip",
        use_pd=True,
        angular_limits=2,
        linear_limits=0.2356,
        damping=0.01,
        use_joint_limits=True,
        alpha=1 / 24,
    )


@configclass
class AffineActionsCfg:
    """仿射编队动作配置
    
    9 维动作空间：旋转(3) + 缩放(3) + 平移(3)
    通过仿射变换控制指尖编队形状。
    """
    affine_formation = leap_mdp.AffineFormationActionCfg(
        asset_name="robot",
        nominal_joint_angles={
            "a_1": 0.000, "a_12": 0.500, "a_5": 0.000, "a_9": 0.000,
            "a_0": -0.750, "a_13": 1.300, "a_4": 0.000, "a_8": 0.750,
            "a_2": 1.750, "a_14": 1.500, "a_6": 1.750, "a_10": 1.750,
            "a_3": 0.000, "a_15": 1.000, "a_7": 0.000, "a_11": 0.000,
        },
        rotation_limit=0.5,
        scale_range=(0.7, 1.3),
        translation_limit=0.05,
        ik_method="dls",
        ik_params={"lambda_val": 0.05},
        finger_joints={
            "index": ["a_1", "a_0", "a_2", "a_3"],
            "middle": ["a_5", "a_4", "a_6", "a_7"],
            "ring": ["a_9", "a_8", "a_10", "a_11"],
            "thumb": ["a_12", "a_13", "a_14", "a_15"],
        },
        finger_bodies=("fingertip", "fingertip_2", "fingertip_3", "thumb_fingertip"),
        use_body_frame=True,
    )


##############################################################################
# 奖励配置组件
##############################################################################

@configclass
class CommonRewardsCfg:
    """通用奖励配置
    
    适用于大多数手内操作任务的基础奖励项。
    """
    
    # ===== 任务奖励 =====
    track_orientation_inv_l2 = RewTerm(
        func=leap_mdp.track_orientation_inv_l2,
        weight=1.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "rot_eps": 0.1,
            "command_name": "goal_pose",
        },
    )

    # NOTE:
    #   对齐 IsaacLab 官方 inhand 环境：位置跟踪奖励项在上游默认是“可选关闭”的。
    #   - 该项度量物体位置与 pos_command_e 的 L2 距离（环境系 {e}）。
    #   - 若希望显式约束“物体别漂移”，将 weight 从 0.0 调整为负值（例如 -10.0）。
    track_pos_l2 = RewTerm(
        func=leap_mdp.track_pos_l2,
        weight=0.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "command_name": "goal_pose",
        },
    )
    
    success_bonus = RewTerm(
        func=leap_mdp.success_bonus,
        weight=250.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "command_name": "goal_pose",
            "orientation_threshold": 0.2,
            "position_threshold": 0.025,
        },
    )
    
    # ===== 动作正则化 =====
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-2.5e-5)
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.0001)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    torque_l2 = RewTerm(func=leap_mdp.torque_l2_penalty, weight=-1e-5)


@configclass
class Se3RewardsCfg(CommonRewardsCfg):
    """SE(3) 动作空间专用奖励配置
    
    在通用奖励基础上增加：
    - 可操作度奖励
    - 动能惩罚
    - 动作平滑奖励
    """
    
    fall_penalty = RewTerm(
        func=leap_mdp.fall_penalty,
        weight=-10.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "command_name": "goal_pose",
            "fall_distance": 0.07,
        },
    )
    
    pose_diff = RewTerm(func=leap_mdp.pose_diff_penalty, weight=-0.3)
    
    # SE(3) 专用
    manipulability = RewTerm(
        func=leap_mdp.jacobian_manipulability,
        weight=1,
        params={"action_names": ["index_se3", "middle_se3", "ring_se3", "thumb_se3"]},
    )
    
    kinetic_energy = RewTerm(
        func=leap_mdp.se3_kinetic_energy,
        weight=-1,
        params={"action_names": ["index_se3", "middle_se3", "ring_se3", "thumb_se3"]},
    )
    
    action_smooth = RewTerm(
        func=leap_mdp.se3_action_smooth,
        weight=-0.25,
        params={
            "action_names": ["index_se3", "middle_se3", "ring_se3", "thumb_se3"],
            "use_processed": False,
            "norm": 1,
        },
    )


##############################################################################
# 事件配置组件
##############################################################################

@configclass
class CommonEventCfg:
    """通用域随机化配置
    
    包含物体和机器人的常用随机化项。
    """
    
    # ===== 物体随机化 =====
    randomized_object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.25, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomized_object_com = EventTerm(
        func=leap_mdp.randomize_rigid_object_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "com_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
        },
    )

    randomized_object_scale = EventTerm(
        func=mdp.randomize_rigid_body_scale,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "scale_range": (0.8, 1.2),
        },
    )

    randomized_object_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.2, 1.0),
            "dynamic_friction_range": (0.15, 0.6),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 250,
            "make_consistent": True,
        },
    )

    randomized_object_force_disturbance = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        min_step_count_between_reset=EPOCHS_NUM * HORIZON_LENGTH,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "force_range": (-1.0, 1.0),
            "torque_range": (-0.1, 0.1),
        },
    )

    # ===== 机器人随机化 =====
    randomized_hand_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="a_.*"),
            "friction_distribution_params": (0.8, 1.2),
            "armature_distribution_params": (0.6, 1.5),
            "lower_limit_distribution_params": (0.975, 1.025),
            "upper_limit_distribution_params": (0.975, 1.025),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomized_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.8, 1.2),
            "distribution": "uniform",
            "operation": "scale",
        },
    )

    randomized_robot_force_disturbance = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        min_step_count_between_reset=EPOCHS_NUM * HORIZON_LENGTH,
        params={
            "asset_cfg": SceneEntityCfg(name="robot", body_names=".*"),
            "force_range": (-0.5, 0.5),
            "torque_range": (-0.025, 0.025),
        },
    )

    robot_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # ===== 重置事件 =====
    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.01, 0.01],
                "y": [-0.01, 0.01],
                "z": [-0.01, 0.01],
                "roll": [-0.0, 0.0],
                "pitch": [-0.0, 0.0],
                "yaw": [-math.pi, math.pi],
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (0.0, 0.0),
        },
    )


##############################################################################
# 终止条件组件
##############################################################################

@configclass
class CommonTerminationsCfg:
    """通用终止条件配置"""

    object_falling = DoneTerm(
        func=leap_mdp.object_falling_termination,
        params={"fall_dist": 0.1, "target_pos_offset": (0.0, -0.1, 0.56)},
    )

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##############################################################################
# 命令配置组件
##############################################################################

@configclass
class ReorientationCommandsCfg:
    """so(3) 相对增量指令命令配置。

    NOTE:
        这里保留类名以避免大范围重命名，但内部命令项已替换为 RelativeSO3Command。
    """

    goal_pose = leap_mdp.RelativeSO3CommandCfg(
        asset_name="object",
        resampling_time_range=(1e6, 1e6),
        init_pos_offset=(0.0, 0.0, 0.0),
        theta_min=0.0,
        theta_max=math.pi / 2.0,
        mode="fixed_goal",
        make_quat_unique=True,
        update_goal_on_success=True,
    )


##############################################################################
# 课程学习组件
##############################################################################

@configclass
class EmptyCurriculumCfg:
    """空课程学习配置（占位符）"""
    pass


##############################################################################
# 触觉配置组件
##############################################################################

# 触觉超参数
TACTILE_FORCE_THRESHOLD = 0.2
TACTILE_CONTACT_REWARD_TYPE = "binary"
TACTILE_USE_REWARD_CURRICULUM = True
TACTILE_CURRICULUM_METRIC_KEY = "consecutive_success"
TACTILE_G_MIN = 0.0
TACTILE_G_MAX = 8.0


@configclass
class TactileSceneCfg(InHandObjectSceneCfg):
    """触觉增强场景配置
    
    在基础场景上添加指尖和非指尖的接触传感器。
    """
    
    # ===== 指尖触觉传感器 =====
    contact_index = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/fingertip",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
        update_period=0.0,
        history_length=3,
        track_air_time=True,
        track_friction_forces=True,
        max_contact_data_count_per_prim=64,
        force_threshold=0.125,
        debug_vis=True,
    )
    
    contact_middle = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/fingertip_2",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
        update_period=0.0,
        history_length=3,
        track_air_time=True,
        track_friction_forces=True,
        max_contact_data_count_per_prim=64,
        force_threshold=0.125,
        debug_vis=True,
    )
    
    contact_ring = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/fingertip_3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
        update_period=0.0,
        history_length=3,
        track_air_time=True,
        track_friction_forces=True,
        max_contact_data_count_per_prim=64,
        force_threshold=0.125,
        debug_vis=True,
    )
    
    contact_thumb = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/thumb_fingertip",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
        update_period=0.0,
        history_length=3,
        track_air_time=True,
        track_friction_forces=True,
        max_contact_data_count_per_prim=64,
        force_threshold=0.125,
        debug_vis=True,
    )
    
    # ===== 手掌接触传感器（用于惩罚非期望接触）=====
    contact_palm = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
        update_period=0.0,
        history_length=3,
        track_friction_forces=True,
        max_contact_data_count_per_prim=64,
        force_threshold=0.5,
        debug_vis=False,
    )


@configclass
class TactileObsGroupCfg(ProprioceptionObsGroupCfg):
    """触觉增强观测组
    
    在关节空间观测基础上添加二值化触觉信号。
    """
    
    fingertip_contact_binary = ObsTerm(
        func=leap_mdp.fingertip_contact_data,
        params={
            "sensor_names": ["contact_index", "contact_middle", "contact_ring", "contact_thumb"],
            "output_type": "binary",
            "force_threshold": TACTILE_FORCE_THRESHOLD,
        },
    )


@configclass
class TactileCriticObsGroupCfg(JointSpaceObsGroupCfg):
    """触觉 Critic 观测组
    
    包含精确的力矢量信息用于 Teacher Policy。
    """
    
    fingertip_contact_force = ObsTerm(
        func=leap_mdp.fingertip_contact_data,
        params={
            "sensor_names": ["contact_index", "contact_middle", "contact_ring", "contact_thumb"],
            "output_type": "force",
        },
        clip=(-50.0, 50.0),
        scale=0.1,
    )


@configclass
class TactileObservationsCfg:
    """触觉增强观测配置（Policy + Critic）"""

    @configclass
    class PolicyCfg(TactileObsGroupCfg):
        """策略观测：关节 + 二值触觉"""

        pass

    @configclass
    class CriticCfg(TactileCriticObsGroupCfg):
        """Critic 观测：特权 + 力触觉"""

        pass

    policy: ObsGroup = PolicyCfg(history_length=1)
    critic: ObsGroup = CriticCfg(history_length=1)


@configclass
class Se3TactileObservationsCfg:
    """SE(3) + 触觉观测配置

    Policy: 本体感受 (body_twists) + 触觉二值信号 + 历史
    Critic: 特权信息 (物体位姿) + 连续力触觉
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """策略观测：本体感受 + 二值触觉 + 历史"""

        body_twists = ObsTerm(
            func=leap_mdp.body_twists,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "action_names": ["index_se3", "middle_se3", "ring_se3", "thumb_se3"],
                "use_body_frame": USE_BODY_FRAME_STUDENT,
            },
            history_length=ENCODER_HISTORY_LENGTH,
        )
        so3_command = ObsTerm(func=leap_mdp.so3_command, params={"command_name": "goal_pose"})
        last_action = ObsTerm(func=mdp.last_action, history_length=ENCODER_HISTORY_LENGTH)
        fingertip_contact_binary = ObsTerm(
            func=leap_mdp.fingertip_contact_data,
            params={
                "sensor_names": ["contact_index", "contact_middle", "contact_ring", "contact_thumb"],
                "output_type": "binary",
                "force_threshold": TACTILE_FORCE_THRESHOLD,
            },
            history_length=ENCODER_HISTORY_LENGTH,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic 观测：特权信息 + 连续力"""

        body_twists = ObsTerm(
            func=leap_mdp.body_twists,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "action_names": ["index_se3", "middle_se3", "ring_se3", "thumb_se3"],
                "use_body_frame": USE_BODY_FRAME_STUDENT,
            },
        )
        object_pos = ObsTerm(
            func=mdp.root_pos_w,
            noise=Gnoise(std=0.002),
            params={"asset_cfg": SceneEntityCfg("object")},
        )
        object_quat = ObsTerm(
            func=mdp.root_quat_w,
            params={"asset_cfg": SceneEntityCfg("object"), "make_quat_unique": False},
        )
        so3_command = ObsTerm(func=leap_mdp.so3_command, params={"command_name": "goal_pose"})

        # NOTE: 位置目标仅用于 Critic/特权观测，policy 侧不暴露以保持可部署假设。
        pos_command = ObsTerm(func=leap_mdp.pos_command, params={"command_name": "goal_pose"})
        last_action = ObsTerm(func=mdp.last_action)
        fingertip_contact_force = ObsTerm(
            func=leap_mdp.fingertip_contact_data,
            params={
                "sensor_names": ["contact_index", "contact_middle", "contact_ring", "contact_thumb"],
                "output_type": "force",
            },
            clip=(-50.0, 50.0),
            scale=0.1,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: ObsGroup = PolicyCfg(history_length=None)
    critic: ObsGroup = CriticCfg(history_length=1)


@configclass
class TactileRewardsCfg(CommonRewardsCfg):
    """触觉增强奖励配置
    
    在通用奖励基础上添加接触相关奖励。
    """
    
    load_distribution = RewTerm(
        func=leap_mdp.load_distribution_reward,
        weight=1,
        params={
            "fingertip_sensor_names": [
                "contact_index", "contact_middle", "contact_ring", "contact_thumb"
            ],
            "palm_sensor_names": ["contact_palm"],
            "gravity_axis": 2,
            "epsilon": 1e-3,
        },
    )
    
    good_fingertip_contact = RewTerm(
        func=leap_mdp.good_fingertip_contact,
        weight=1.0,
        params={
            "sensor_names": ["contact_index", "contact_middle", "contact_ring", "contact_thumb"],
            "min_contacts": 2,
            "force_threshold": TACTILE_FORCE_THRESHOLD,
            "reward_type": TACTILE_CONTACT_REWARD_TYPE,
            "use_curriculum": TACTILE_USE_REWARD_CURRICULUM,
            "command_name": "goal_pose",
            "g_min": TACTILE_G_MIN,
            "g_max": TACTILE_G_MAX,
            "metric_key": TACTILE_CURRICULUM_METRIC_KEY,
        },
    )
    
    bad_palm_contact = RewTerm(
        func=leap_mdp.bad_palm_contact,
        weight=-1.0,
        params={
            "sensor_names": ["contact_palm"],
            "force_threshold": TACTILE_FORCE_THRESHOLD,
            "reward_type": TACTILE_CONTACT_REWARD_TYPE,
            "use_curriculum": TACTILE_USE_REWARD_CURRICULUM,
            "command_name": "goal_pose",
            "g_min": TACTILE_G_MIN,
            "g_max": TACTILE_G_MAX,
            "metric_key": TACTILE_CURRICULUM_METRIC_KEY,
        },
    )


@configclass
class Se3TactileRewardsCfg(Se3RewardsCfg):
    """SE(3) + 触觉奖励配置

    在 SE(3) 奖励基础上添加触觉接触 shaping。
    """

    load_distribution = RewTerm(
        func=leap_mdp.load_distribution_reward,
        weight=1.0,
        params={
            "fingertip_sensor_names": ["contact_index", "contact_middle", "contact_ring", "contact_thumb"],
            "palm_sensor_names": [
                "contact_palm",
                "contact_index_mcp",
                "contact_index_pip",
                "contact_index_dip",
                "contact_middle_mcp",
                "contact_middle_pip",
                "contact_middle_dip",
                "contact_ring_mcp",
                "contact_ring_pip",
                "contact_ring_dip",
                "contact_thumb_base",
                "contact_thumb_pip",
                "contact_thumb_dip",
            ],
            "gravity_axis": 2,
            "epsilon": 1e-3,
        },
    )

    good_fingertip_contact = RewTerm(
        func=leap_mdp.good_fingertip_contact,
        weight=1.0,
        params={
            "sensor_names": ["contact_index", "contact_middle", "contact_ring", "contact_thumb"],
            "min_contacts": 2,
            "force_threshold": TACTILE_FORCE_THRESHOLD,
            "reward_type": TACTILE_CONTACT_REWARD_TYPE,
            "use_curriculum": TACTILE_USE_REWARD_CURRICULUM,
            "command_name": "goal_pose",
            "g_min": TACTILE_G_MIN,
            "g_max": TACTILE_G_MAX,
            "metric_key": TACTILE_CURRICULUM_METRIC_KEY,
        },
    )

    bad_palm_contact = RewTerm(
        func=leap_mdp.bad_palm_contact,
        weight=-1.0,
        params={
            "sensor_names": [
                "contact_palm",
                "contact_index_mcp",
                "contact_index_pip",
                "contact_index_dip",
                "contact_middle_mcp",
                "contact_middle_pip",
                "contact_middle_dip",
                "contact_ring_mcp",
                "contact_ring_pip",
                "contact_ring_dip",
                "contact_thumb_base",
                "contact_thumb_pip",
                "contact_thumb_dip",
            ],
            "force_threshold": TACTILE_FORCE_THRESHOLD,
            "reward_type": TACTILE_CONTACT_REWARD_TYPE,
            "use_curriculum": TACTILE_USE_REWARD_CURRICULUM,
            "command_name": "goal_pose",
            "g_min": TACTILE_G_MIN,
            "g_max": TACTILE_G_MAX,
            "metric_key": TACTILE_CURRICULUM_METRIC_KEY,
        },
    )


##############################################################################
# 公共导出（供各 hand config 组合使用）
##############################################################################


__all__ = [
    # 场景
    "InHandObjectSceneCfg",
    "TactileSceneCfg",
    # 观测
    "ProprioceptionObsGroupCfg",
    "Se3ProprioceptionObsGroupCfg",
    "JointSpaceObservationsCfg",
    "Se3ObservationsCfg",
    "TactileObsGroupCfg",
    "TactileCriticObsGroupCfg",
    "TactileObservationsCfg",
    "Se3TactileObservationsCfg",
    # 动作
    "JointSpaceActionsCfg",
    "Se3ActionsCfg",
    "Se3EmaActionsCfg",
    "AffineActionsCfg",
    # 奖励
    "CommonRewardsCfg",
    "Se3RewardsCfg",
    "TactileRewardsCfg",
    "Se3TactileRewardsCfg",
    # 事件/终止/命令/课程
    "CommonEventCfg",
    "CommonTerminationsCfg",
    "ReorientationCommandsCfg",
    "EmptyCurriculumCfg",
    # 触觉超参数
    "TACTILE_FORCE_THRESHOLD",
    "TACTILE_CONTACT_REWARD_TYPE",
    "TACTILE_USE_REWARD_CURRICULUM",
    "TACTILE_CURRICULUM_METRIC_KEY",
    "TACTILE_G_MIN",
    "TACTILE_G_MAX",
    # SE(3)+EMA / SE(3)+触觉超参数
    "USE_BODY_FRAME_STUDENT",
    "ENCODER_HISTORY_LENGTH",
]
