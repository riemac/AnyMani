r"""Official-aligned LEAP ADR env for AnyMani.

本文件把 `AnyMani-LeapHand-Tactile-ADR-v0` 重定义为“尽量复刻官方 LEAP_Hand_Isaac_Lab”的
ManagerBasedRLEnv 等价实现，而不再沿用 AnyMani tactile ADR 研究语义。

核心对齐目标：

1. actor 观测：
   $$
   o_t^\pi = [\tilde q_t, q_t^{target}]_{t-2:t} \in \mathbb{R}^{96};
   $$
2. action 语义：
   $$
   q_t^{target}=\operatorname{clip}(q_{t-1}^{target}+\tfrac{1}{24}a_t^{exec});
   $$
3. reward 语义：对齐官方 DirectRLEnv，并在 term 内显式除以 `env.step_dt`，抵消
   ManagerBased RewardManager 的自动 `*dt`；
4. ADR / horizon / action noise / latency / wrench / spawn ranges：尽量照抄官方。
"""

from __future__ import annotations

import math

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.ui import ManagerBasedRLEnvWindow
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from anymani.robots.leap import LEAP_HAND_CFG
from anymani.tasks.inhand import mdp as leap_mdp

# 官方 LEAP runtime 的 16 维 sim order 不是按 joint name 字典序，而是“关节层级交织”顺序。
# 该顺序由官方 `override_default_joint_pos` 向量和 deployment 注释共同确定。
OFFICIAL_JOINT_ORDER = (
    "a_1",
    "a_12",
    "a_5",
    "a_9",
    "a_0",
    "a_13",
    "a_4",
    "a_8",
    "a_2",
    "a_14",
    "a_6",
    "a_10",
    "a_3",
    "a_15",
    "a_7",
    "a_11",
)

OFFICIAL_PREGRASP_BY_NAME = {
    "a_0": -0.750,
    "a_1": 0.000,
    "a_2": 1.750,
    "a_3": 0.000,
    "a_4": 0.000,
    "a_5": 0.000,
    "a_6": 1.750,
    "a_7": 0.000,
    "a_8": 0.750,
    "a_9": 0.000,
    "a_10": 1.750,
    "a_11": 0.000,
    "a_12": 0.500,
    "a_13": 1.300,
    "a_14": 1.500,
    "a_15": 1.000,
}
OFFICIAL_PREGRASP_VECTOR = tuple(OFFICIAL_PREGRASP_BY_NAME[name] for name in OFFICIAL_JOINT_ORDER)


@configclass
class LeapHandOfficialADRSceneCfg(InteractiveSceneCfg):
    r"""官方 LEAP 对齐场景：无 tactile sensors，仅 robot/object/ground/light。"""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.1)),
    )

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.1, 0.56), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    robot: ArticulationCfg = LEAP_HAND_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(0.5, 0.5, -0.5, 0.5),
            joint_pos=OFFICIAL_PREGRASP_BY_NAME,
            joint_vel={"a_.*": 0.0},
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class LeapHandOfficialADRPolicyObsCfg(ObsGroup):
    r"""官方 actor 观测组：单帧 32D，再由 IsaacLab 内置 history buffer 叠成 96D。"""

    frame = ObsTerm(
        func=leap_mdp.official_policy_frame,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(OFFICIAL_JOINT_ORDER), preserve_order=True),
            "action_term_name": "hand_joint_pos",
        },
        history_length=3,
        flatten_history_dim=True,
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class LeapHandOfficialADRObservationsCfg:
    r"""官方 LEAP 对齐观测：仅暴露 policy 96D proprio-target history。"""

    @configclass
    class PolicyCfg(LeapHandOfficialADRPolicyObsCfg):
        pass

    policy: ObsGroup = PolicyCfg()


@configclass
class LeapHandOfficialADRActionsCfg:
    r"""官方 LEAP target-buffer relative action。"""

    hand_joint_pos = leap_mdp.OfficialADRTargetJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(OFFICIAL_JOINT_ORDER),
        scale=1.0 / 24.0,
        preserve_order=True,
        use_zero_offset=True,
        max_latency=3,
        latency_rand=1,
        pregrasp_joint_pos=OFFICIAL_PREGRASP_VECTOR,
    )


@configclass
class LeapHandOfficialADRCommandsCfg:
    r"""官方连续 z 轴目标推进命令。"""

    goal_pose = leap_mdp.OfficialContinuousRotationCommandCfg(
        asset_name="object",
        resampling_time_range=(1e6, 1e6),
        init_pos_offset=(0.0, 0.0, 0.0),
        rotation_axis="z",
        delta_angle=math.pi / 8.0,
        make_quat_unique=True,
        orientation_success_threshold=0.2,
        position_success_threshold=0.025,
        update_goal_on_success=True,
    )


@configclass
class LeapHandOfficialADRRewardsCfg:
    r"""官方 LEAP reward：单一 combined term，内部抵消 ManagerBased dt。"""

    official_reward = RewTerm(
        func=leap_mdp.OfficialLeapReward,
        weight=1.0,
        params={
            "action_term_name": "hand_joint_pos",
            "command_name": "goal_pose",
            "object_cfg": SceneEntityCfg("object"),
            "robot_cfg": SceneEntityCfg("robot"),
            "dist_reward_scale": -10.0,
            "rot_reward_scale": 1.0,
            "rot_eps": 0.1,
            "action_penalty_scale": -0.0002,
            "pose_diff_penalty_scale": -0.3,
            "torque_penalty_scale": -0.0,
            "success_tolerance": 0.2,
            "position_success_threshold": 0.025,
            "reach_goal_bonus": 250.0,
            "fall_dist": 0.07,
            "fall_penalty": -10.0,
            "z_rotation_steps": 16,
        },
    )


@configclass
class LeapHandOfficialADRTerminationsCfg:
    r"""官方 LEAP done：out_of_reach 或 flipped，再加 randomized timeout。"""

    object_falling = DoneTerm(
        func=leap_mdp.official_out_of_reach_or_flipped,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "command_name": "goal_pose",
            "fall_dist": 0.07,
            "flipped_dot_threshold": 0.5,
        },
    )
    time_out = DoneTerm(func=leap_mdp.adr_randomized_time_out, time_out=True)


@configclass
class LeapHandOfficialADREventCfg:
    r"""官方 ADR event 集合。"""

    randomized_object_scale = EventTerm(
        func=mdp.randomize_rigid_body_scale,
        mode="prestartup",
        params={"asset_cfg": SceneEntityCfg("object"), "scale_range": (1.1, 1.25)},
    )

    randomized_object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomized_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (3.0, 3.0),
            "damping_distribution_params": (0.1, 0.1),
            "operation": "abs",
            "distribution": "uniform",
        },
    )

    randomized_object_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
            "make_consistent": True,
        },
    )
    randomized_object_friction_adr = EventTerm(
        func=leap_mdp.resample_adr_material_buckets,
        mode="reset",
        min_step_count_between_reset=720,
        params={"term_name": "randomized_object_friction", "range_attr": "leap_adr_object_material_ranges"},
    )

    randomized_robot_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
            "make_consistent": True,
        },
    )
    randomized_robot_material_adr = EventTerm(
        func=leap_mdp.resample_adr_material_buckets,
        mode="reset",
        min_step_count_between_reset=720,
        params={"term_name": "randomized_robot_material", "range_attr": "leap_adr_robot_material_ranges"},
    )

    reset_episode_length = EventTerm(
        func=leap_mdp.reset_adr_episode_length,
        mode="reset",
        params={"min_episode_length_s": 20.0},
    )
    reset_object = EventTerm(
        func=leap_mdp.reset_adr_object_state, mode="reset", params={"asset_cfg": SceneEntityCfg("object")}
    )
    reset_robot_joints = EventTerm(
        func=leap_mdp.reset_adr_robot_joints,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(OFFICIAL_JOINT_ORDER), preserve_order=True)},
    )
    reset_wrench_gate = EventTerm(
        func=leap_mdp.reset_adr_wrench_state,
        mode="reset",
        params={"probability": 0.5, "asset_cfg": SceneEntityCfg("object")},
    )
    object_wrench = EventTerm(
        func=leap_mdp.apply_adr_object_wrench,
        mode="interval",
        interval_range_s=(3.0, 3.0),
        params={"asset_cfg": SceneEntityCfg("object"), "torsional_radius": 0.0},
    )


@configclass
class LeapHandOfficialADRCurriculumCfg:
    r"""官方全局 ADR scheduler。"""

    adr = CurrTerm(
        func=leap_mdp.LeapADRGlobalScheduler,
        params={
            "command_name": "goal_pose",
            "metric_key": "consecutive_success",
            "num_increments": 25,
            "min_rot_adr_coeff": 0.15,
            "min_steps_for_dr_change": 240 * 4,
            "z_rotation_steps": 16,
            "ema_alpha": 0.1,
            "min_episode_length_s": 20.0,
            "episode_length_s": 120.0,
        },
    )


@configclass
class LeapHandTactileADREnvCfg(ManagerBasedRLEnvCfg):
    r"""覆盖旧 tactile ADR 语义的 official-aligned LEAP ADR env。"""

    ui_window_class_type: type | None = ManagerBasedRLEnvWindow
    is_finite_horizon: bool = True

    scene: InteractiveSceneCfg = LeapHandOfficialADRSceneCfg(num_envs=4096, env_spacing=0.75, replicate_physics=False)
    viewer: ViewerCfg = ViewerCfg()
    sim: SimulationCfg = SimulationCfg(
        physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
        ),
    )
    seed: int | None = 42

    observations: LeapHandOfficialADRObservationsCfg = LeapHandOfficialADRObservationsCfg()
    actions: LeapHandOfficialADRActionsCfg = LeapHandOfficialADRActionsCfg()
    commands: LeapHandOfficialADRCommandsCfg = LeapHandOfficialADRCommandsCfg()
    rewards: LeapHandOfficialADRRewardsCfg = LeapHandOfficialADRRewardsCfg()
    terminations: LeapHandOfficialADRTerminationsCfg = LeapHandOfficialADRTerminationsCfg()
    events: LeapHandOfficialADREventCfg = LeapHandOfficialADREventCfg()
    curriculum: LeapHandOfficialADRCurriculumCfg = LeapHandOfficialADRCurriculumCfg()

    min_episode_length_s: float = 20.0

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4
        self.episode_length_s = 120.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (2.0, 2.0, 2.0)


@configclass
class LeapHandTactileADREnvCfg_PLAY(LeapHandTactileADREnvCfg):
    r"""official-aligned ADR play 配置。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.commands.goal_pose.debug_vis = True
