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
    - Observations: 观测配置（关节空间/触觉）
    - Actions: 动作空间配置（关节）
    - Rewards: 奖励配置（通用/触觉）
    - Events: 域随机化配置
    - Terminations: 终止条件

Note:
    各手型的具体环境配置（如 LeapHandJointEnvCfg）在 config/<hand_type>/ 下定义，
    它们通过组合本模块的组件来构建完整环境。
"""

from __future__ import annotations

import math

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise

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

# 默认清天 HDR 天空贴图；与 GM scene 保持一致，只改视觉不改动力学。
INHAND_CLEAR_SKY_TEXTURE_FILE = (
    f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"
)

# 默认天空光强度；沿用 GM scene 的数值锚点，避免纯灰 DomeLight 把地面/手色洗白。
INHAND_CLEAR_SKY_LIGHT_INTENSITY = 750.0


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
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=INHAND_CLEAR_SKY_LIGHT_INTENSITY,
            texture_file=INHAND_CLEAR_SKY_TEXTURE_FILE,
        ),
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
    - 目标姿态四元数（4D wxyz）
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
    quat_command = ObsTerm(
        func=leap_mdp.quat_command,
        params={"command_name": "goal_pose", "make_quat_unique": True},
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
        # quat_command 是任务目标输入，保留；policy 不再消费旧的 3D rotvec 指令。
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
class ContinuousRotationObsGroupCfg(ObsGroup):
    r"""a51c666 连续旋转任务的特权观测组。

    该观测组刻意不同于当前 `JointSpaceObsGroupCfg` 的 so(3) rotvec command：
    这里恢复历史黄金 tactile 版使用的 7D goal pose 观测，直接暴露
    `ContinuousRotationCommand.command = [p_g^e, Q_g^w]`。

    数学上，critic 看到的是：
    $$
    o_t^V = [\tilde q_t,\ p_o^w,\ Q_o^w,\ p_g^e,\ Q_g^w,\
             Q_g^w \otimes (Q_o^w)^{-1},\ a_{t-1}].
    $$

    其中 $\tilde q_t$ 是关节限位归一化位置，$Q_g^w$ 是固定 z 轴连续推进的目标四元数。
    该设计牺牲一部分 deployable purity，但恢复了历史训练成功时的 Markov 信息量。
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
    goal_pose = ObsTerm(
        func=mdp.generated_commands,
        params={"command_name": "goal_pose"},
    )
    goal_quat_diff = ObsTerm(
        func=leap_mdp.goal_quat_diff,
        params={"asset_cfg": SceneEntityCfg("object"), "command_name": "goal_pose", "make_quat_unique": True},
    )

    # -- action terms
    last_action = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class ContinuousRotationProprioceptionObsGroupCfg(ContinuousRotationObsGroupCfg):
    r"""a51c666 actor 观测组：本体感受 + 7D quaternion goal pose。

    actor 不看物体绝对位姿，也不看 $Q_g \otimes Q_o^{-1}$ 这种仿真特权误差信息；
    它只接收关节状态、连续旋转目标姿态和上一帧动作：
    $$
    o_t^\pi = [\tilde q_t,\ p_g^e,\ Q_g^w,\ a_{t-1}].
    $$

    tactile 子类会在该向量后追加四个指尖的二值接触信号 $b_t\in\{0,1\}^4$。
    """

    def __post_init__(self):
        super().__post_init__()
        # 恢复 a51 tactile actor 的“本体 + 目标姿态”设定：不把物体绝对 pose 暴露给 policy。
        self.object_pos = None
        self.object_quat = None
        self.goal_quat_diff = None


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
            "a_1",
            "a_0",
            "a_2",
            "a_3",  # 食指
            "a_5",
            "a_4",
            "a_6",
            "a_7",  # 中指
            "a_9",
            "a_8",
            "a_10",
            "a_11",  # 无名指
            "a_12",
            "a_13",
            "a_14",
            "a_15",  # 拇指
        ],
        scale=1 / 10,
        preserve_order=True,
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
        func=leap_mdp.official_orientation,
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
        func=leap_mdp.official_goal_distance,
        weight=0.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "command_name": "goal_pose",
        },
    )

    success_bonus = RewTerm(
        func=leap_mdp.official_success_bonus,
        weight=250.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "command_name": "goal_pose",
            "success_tolerance": 0.2,
            "position_success_threshold": 0.025,
        },
    )

    # ===== 动作正则化 =====
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-2.5e-5)
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.0001)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    torque_l2 = None


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

    randomized_object_com = None
    # 历史组件仓中的刚体 COM randomization 已出清；当前 inhand 主线不再维护这一路径。

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
    """历史 reorientation 命令配置占位。

    NOTE:
        `RelativeSO3CommandCfg` 已从 inhand 主线移除，因为相对增量 so(3) 对 RL 表现为明显非平稳。
        这里暂保留历史类名，内部退化为 generic continuous rotation command，只为保持旧组件仓可编译。
    """

    goal_pose = leap_mdp.ContinuousRotationCommandCfg(
        asset_name="object",
        resampling_time_range=(1e6, 1e6),
        init_pos_offset=(0.0, 0.0, 0.0),
        rotation_axis="z",
        delta_angle=math.pi / 8.0,
        make_quat_unique=True,
        update_goal_on_success=True,
    )


@configclass
class ContinuousRotationCommandsCfg:
    r"""a51c666 黄金 tactile 版的固定轴连续旋转命令。

    该命令项把 hand-object 任务重新锚定为局部连续旋转，而不是当前 Joint-v0 使用的
    随机 SO(3) fixed-goal reorientation。每个目标只推进一个小角度：
    $$
    Q_g^{k+1} = R_z(\Delta\theta) Q_g^k,\qquad \Delta\theta=\frac{\pi}{8}.
    $$

    这个小步长相当于内建 curriculum：策略每次只需要学习稳定抓握下的局部滚动原语，
    成功后再把目标沿同一世界系 z 轴推进。`command()` 输出 7D `goal_pose`：
    $$
    c_t=[p_g^e, Q_g^w]\in\mathbb{R}^7.
    $$
    """

    goal_pose = leap_mdp.ContinuousRotationCommandCfg(
        asset_name="object",
        resampling_time_range=(1e6, 1e6),
        init_pos_offset=(0.0, 0.0, 0.0),
        rotation_axis="z",
        delta_angle=math.pi / 8.0,
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
class TactileObsGroupCfg(ContinuousRotationProprioceptionObsGroupCfg):
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
class TactileCriticObsGroupCfg(ContinuousRotationObsGroupCfg):
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
class TactileRewardsCfg(CommonRewardsCfg):
    """触觉增强奖励配置

    在通用奖励基础上添加接触相关奖励。
    """

    # a51c666 中位置项是主动约束，而不是当前 Joint-v0 的可选关闭项。
    # 这里用历史名称保留 TensorBoard 语义：$r_p=-10\|p_o^e-p_g^e\|_2$。
    track_pos_l2 = None
    goal_position_distance = RewTerm(
        func=leap_mdp.official_goal_distance,
        weight=-10.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "command_name": "goal_pose",
        },
    )

    # official LEAP 训练中显式加入了跌落惩罚：当物体相对目标位置漂移超过 7 cm 时，
    # 除了触发 reset，还额外给一次负奖励，避免 policy 在早期通过“撞到一次 goal 然后掉落”获得净收益。
    # 公式：$r_{fall}=-10\,\mathbf{1}[\|p_o^e-p_g^e\|_2 \ge 0.07]$。
    fall_penalty = RewTerm(
        func=leap_mdp.official_fall_penalty,
        weight=-10.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "command_name": "goal_pose",
            "fall_dist": 0.07,
        },
    )

    load_distribution = RewTerm(
        func=leap_mdp.load_distribution_reward,
        weight=1,
        params={
            "fingertip_sensor_names": ["contact_index", "contact_middle", "contact_ring", "contact_thumb"],
            # a51c666 把所有非指尖 link 都放进分母，避免 policy 用关节/手掌“托住”物体骗过 load reward。
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
    "JointSpaceObservationsCfg",
    "ContinuousRotationObsGroupCfg",
    "ContinuousRotationProprioceptionObsGroupCfg",
    "TactileObsGroupCfg",
    "TactileCriticObsGroupCfg",
    "TactileObservationsCfg",
    # 动作
    "JointSpaceActionsCfg",
    # 奖励
    "CommonRewardsCfg",
    "TactileRewardsCfg",
    # 事件/终止/命令/课程
    "CommonEventCfg",
    "CommonTerminationsCfg",
    "ReorientationCommandsCfg",
    "ContinuousRotationCommandsCfg",
    "EmptyCurriculumCfg",
    # 触觉超参数
    "TACTILE_FORCE_THRESHOLD",
    "TACTILE_CONTACT_REWARD_TYPE",
    "TACTILE_USE_REWARD_CURRICULUM",
    "TACTILE_CURRICULUM_METRIC_KEY",
    "TACTILE_G_MIN",
    "TACTILE_G_MAX",
]
