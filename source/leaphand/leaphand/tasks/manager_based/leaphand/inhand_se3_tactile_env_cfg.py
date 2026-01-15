"""LeapHand 连续旋转任务：SE(3) 动作空间 + 触觉观测/奖励（ManagerBasedRLEnv）。

该配置用于 se3-policy（学生策略）训练：
- 动作：每根手指一个 se(3) DLS（可选 EMA 平滑）动作项
- 观测：本体感受（末端 twist + last_action）+ 指令（goal）+ 触觉二值信号（sim2real 友好）
- Critic：可使用特权信息（物体位姿 + 连续力触觉等）以稳定训练
- 奖励：在 se3 任务奖励基础上，加入 tactile 的 good/bad contact shaping

Notes:
    - 该文件不修改 IsaacLab 核心，仅在 AnyRotate 项目侧做组合配置。
    - 当前默认 student 使用 {b}/{b'} 参考系的 twist（use_body_frame=True），与“跨手型”泛化目标一致。
"""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise

import isaaclab.envs.mdp as mdp

from . import inhand_base_env_cfg
from . import mdp as leap_mdp
from .inhand_se3_env_cfg import RewardsCfg as Se3RewardsCfg
from .inhand_tactile_env_cfg import (
    InHandTactileSceneCfg,
    TACTILE_CONTACT_REWARD_TYPE,
    TACTILE_CURRICULUM_METRIC_KEY,
    TACTILE_FORCE_THRESHOLD,
    TACTILE_G_MAX,
    TACTILE_G_MIN,
    TACTILE_USE_REWARD_CURRICULUM,
)


USE_BODY_FRAME_STUDENT = True
ENCODER_HISTORY_LENGTH = 50


@configclass
class Se3EmaActionsCfg:
    """动作配置：se(3) DLS + EMA（每根手指一个 6D twist）。"""

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
class Se3TactileObservationsCfg:
    """观测配置：student 用本体感受+触觉；critic 用特权信息+连续力。"""

    @configclass
    class PolicyCfg(ObsGroup):
        # -- robot terms
        body_twists = ObsTerm(
            func=leap_mdp.body_twists,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "action_names": ["index_se3", "middle_se3", "ring_se3", "thumb_se3"],
                "use_body_frame": USE_BODY_FRAME_STUDENT,
            },
            history_length=ENCODER_HISTORY_LENGTH,
        )

        # -- command terms
        goal_pose = ObsTerm(func=mdp.generated_commands, params={"command_name": "goal_pose"})

        # -- action terms
        last_action = ObsTerm(func=mdp.last_action, history_length=ENCODER_HISTORY_LENGTH)

        # -- tactile terms (sim2real friendly)
        fingertip_contact_binary = ObsTerm(
            func=leap_mdp.fingertip_contact_data,
            params={
                "sensor_names": [
                    "contact_index",
                    "contact_middle",
                    "contact_ring",
                    "contact_thumb",
                ],
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
        # -- robot terms
        body_twists = ObsTerm(
            func=leap_mdp.body_twists,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "action_names": ["index_se3", "middle_se3", "ring_se3", "thumb_se3"],
                "use_body_frame": USE_BODY_FRAME_STUDENT,
            },
        )

        # -- object terms (privileged)
        object_pos = ObsTerm(func=mdp.root_pos_w, noise=Gnoise(std=0.002), params={"asset_cfg": SceneEntityCfg("object")})
        object_quat = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object"), "make_quat_unique": False})

        # -- command terms
        goal_pose = ObsTerm(func=mdp.generated_commands, params={"command_name": "goal_pose"})
        goal_quat_diff = ObsTerm(
            func=leap_mdp.goal_quat_diff,
            params={"asset_cfg": SceneEntityCfg("object"), "command_name": "goal_pose", "make_quat_unique": True},
        )

        # -- action terms
        last_action = ObsTerm(func=mdp.last_action)

        # -- tactile force (privileged)
        fingertip_contact_force = ObsTerm(
            func=leap_mdp.fingertip_contact_data,
            params={
                "sensor_names": [
                    "contact_index",
                    "contact_middle",
                    "contact_ring",
                    "contact_thumb",
                ],
                "output_type": "force",
            },
            clip=(-50.0, 50.0),
            scale=0.1,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # Keep group history_length=None to allow per-term history settings.
    policy: ObsGroup = PolicyCfg(history_length=None)
    critic: ObsGroup = CriticCfg(history_length=1)


@configclass
class Se3TactileRewardsCfg(Se3RewardsCfg):
    """奖励配置：se3 任务奖励 + tactile 接触 shaping。"""

    load_distribution = RewTerm(
        func=leap_mdp.load_distribution_reward,
        weight=1.0,
        params={
            "fingertip_sensor_names": [
                "contact_index",
                "contact_middle",
                "contact_ring",
                "contact_thumb",
            ],
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
            "sensor_names": [
                "contact_index",
                "contact_middle",
                "contact_ring",
                "contact_thumb",
            ],
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


@configclass
class InHandSe3TactileEnvCfg(inhand_base_env_cfg.InHandObjectEnvCfg):
    """主环境配置：SE(3)+tactile。"""

    scene: InHandTactileSceneCfg = InHandTactileSceneCfg(num_envs=4096, env_spacing=0.6)
    actions: Se3EmaActionsCfg = Se3EmaActionsCfg()
    observations: Se3TactileObservationsCfg = Se3TactileObservationsCfg()
    rewards: Se3TactileRewardsCfg = Se3TactileRewardsCfg()

    commands: inhand_base_env_cfg.CommandsCfg = inhand_base_env_cfg.CommandsCfg()
    events: inhand_base_env_cfg.EventCfg = inhand_base_env_cfg.EventCfg()
    terminations: inhand_base_env_cfg.TerminationsCfg = inhand_base_env_cfg.TerminationsCfg()
    curriculum: inhand_base_env_cfg.CurriculumCfg = inhand_base_env_cfg.CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
