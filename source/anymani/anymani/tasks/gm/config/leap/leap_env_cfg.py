r"""Official LEAP hand GM reorientation probe.

This variant keeps the current GM MDP stack fixed and swaps only the hand asset
to the project-maintained official LEAP USD. It is a clean ablation: if this
variant learns while the generated single asset does not, the failure likely
belongs to generated asset physics/contact basin/joint mechanics rather than
the reward/command/action stack itself.
"""

from __future__ import annotations

import math

import isaaclab.envs.mdp as isaac_mdp
import isaaclab.sim as sim_utils
from anymani.robots.leap import LEAP_HAND_CFG
from anymani.tools.grasp_preset import GraspPreset, asset_preset_path
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.common import ViewerCfg
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

from ... import mdp as gm_mdp
from ...contact_sensors import GmContactSensorLayout, install_contact_sensors

GM_LEAP_GRASP_PRESET_PATH = asset_preset_path("official", "leap")
r"""Official LEAP pre-grasp / contact-basin preset used by this ablation."""

GM_LEAP_GRASP_PRESET = GraspPreset.from_yaml(
    GM_LEAP_GRASP_PRESET_PATH,
    expected_hand_source=("official_leap_usd", "official_leap_urdf"),
    expected_hand_ref_contains="leap_hand",
)
r"""Runtime preset source for LEAP joint pose and object initial pose."""

GM_LEAP_ROOT_POS_E = (0.0, 0.0, 0.5)
"""Official LEAP root position in env frame, matching the existing LEAP scene seed."""

GM_LEAP_ROOT_ROT_WXYZ = (0.5, 0.5, -0.5, 0.5)
"""Official LEAP root orientation: hand palm faces the object basin."""

GM_LEAP_SEMANTIC_R_HA = (0.0, 1.0, 0.0,
                         0.0, 0.0, 1.0,
                         1.0, 0.0, 0.0)
"""Official LEAP calibrated `{a}->{h}` rotation matrix $R_{ha}$, row-major."""

GM_LEAP_SEMANTIC_P_HA = (-0.0098, 0.002, -0.011)
"""Official LEAP calibrated raw-root position $p_{ha}$ in hand semantic frame `{h}`, unit m."""

GM_LEAP_FINGER_LINK_CHAINS = (
    ("mcp_joint", "pip", "dip", "fingertip"),
    ("mcp_joint_2", "pip_2", "dip_2", "fingertip_2"),
    ("mcp_joint_3", "pip_3", "dip_3", "fingertip_3"),
    ("thumb_temp_base", "thumb_pip", "thumb_dip", "thumb_fingertip"),
)
"""Official LEAP link chains used only for contact layout semantics."""

GM_LEAP_FINGERTIP_LINK_NAMES = ("fingertip", "fingertip_2", "fingertip_3", "thumb_fingertip")
"""Links treated as task fingertips for good-contact reward and force observations."""

GM_LEAP_NON_TIP_LINK_NAMES = (
    "palm_lower",
    "mcp_joint",
    "pip",
    "dip",
    "mcp_joint_2",
    "pip_2",
    "dip_2",
    "mcp_joint_3",
    "pip_3",
    "dip_3",
    "thumb_temp_base",
    "thumb_pip",
    "thumb_dip",
)
"""Palm and non-tip links penalized by bad non-tip contact."""

GM_LEAP_CONTACT_LAYOUT = GmContactSensorLayout(
    source_asset_id="official_leap",
    palm_link_name="palm_lower",
    finger_link_chains=GM_LEAP_FINGER_LINK_CHAINS,
    fingertip_link_names=GM_LEAP_FINGERTIP_LINK_NAMES,
    non_tip_link_names=GM_LEAP_NON_TIP_LINK_NAMES,
    fingertip_sensor_names=tuple(f"contact_{link_name}" for link_name in GM_LEAP_FINGERTIP_LINK_NAMES),
    non_tip_sensor_names=tuple(f"contact_{link_name}" for link_name in GM_LEAP_NON_TIP_LINK_NAMES),
)
"""Static contact layout for official LEAP, which has no AnyMani generated sidecar."""

GM_LEAP_HAND_CFG = LEAP_HAND_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    init_state=ArticulationCfg.InitialStateCfg(
        pos=GM_LEAP_ROOT_POS_E,
        rot=GM_LEAP_ROOT_ROT_WXYZ,
        joint_pos=GM_LEAP_GRASP_PRESET.joint_pos_rad,
        joint_vel={"a_.*": 0.0},
    ),
)
"""Official LEAP articulation cfg with preset-provided reset joint pose."""


@configclass
class GmLeapSceneCfg(InteractiveSceneCfg):
    r"""Official LEAP hand + DexCube scene."""

    robot: ArticulationCfg = GM_LEAP_HAND_CFG
    """Official LEAP hand articulation."""

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
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=GM_LEAP_GRASP_PRESET.object_pos_cfg,
            rot=GM_LEAP_GRASP_PRESET.object_rot_wxyz,
        ),
    )
    """Object pose comes from the official LEAP preset, not the generated-hand preset."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.1)),
    )

    light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    def __post_init__(self):
        r"""Install official LEAP per-link contact sensors."""

        super().__post_init__()
        install_contact_sensors(self, GM_LEAP_CONTACT_LAYOUT)


@configclass
class GmLeapCommandsCfg:
    r"""Fixed z-axis command, matching the current single-asset ablation."""

    goal_pose: gm_mdp.ReorientCommandCfg = gm_mdp.ReorientCommandCfg(
        asset_name="object",
        robot_asset_name="robot",
        axis_mode="fixed",
        axis_resample_mode="episode",
        fixed_axis_h=(0.0, 0.0, 1.0),
        semantic_R_ha=GM_LEAP_SEMANTIC_R_HA,
    )


@configclass
class GmLeapActionsCfg:
    r"""Official IsaacLab relative joint-position action."""

    hand_joint_pos = isaac_mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.1,
        preserve_order=True,
    )


@configclass
class GmLeapObservationsCfg:
    r"""Actor / critic observations for the LEAP ablation."""

    @configclass
    class PolicyCfg(ObsGroup):
        r"""Actor-facing flat observation group."""

        joint_pos = ObsTerm(func=isaac_mdp.joint_pos_limit_normalized, params={"asset_cfg": SceneEntityCfg("robot")})
        last_action = ObsTerm(func=isaac_mdp.last_action)
        fingertip_force = ObsTerm(
            func=gm_mdp.fingertip_contact_force,
            params={
                "sensor_names": GM_LEAP_CONTACT_LAYOUT.fingertip_sensor_names,
                "robot_cfg": SceneEntityCfg("robot"),
                "semantic_R_ha": GM_LEAP_SEMANTIC_R_HA,
                "frame": "h",
            },
        )
        object_pos = ObsTerm(
            func=gm_mdp.object_pos,
            params={
                "object_cfg": SceneEntityCfg("object"),
                "robot_cfg": SceneEntityCfg("robot"),
                "semantic_R_ha": GM_LEAP_SEMANTIC_R_HA,
                "semantic_p_ha": GM_LEAP_SEMANTIC_P_HA,
                "frame": "h",
                "reference": "hand",
            },
        )
        object_orientation = ObsTerm(
            func=gm_mdp.object_orientation,
            params={
                "object_cfg": SceneEntityCfg("object"),
                "robot_cfg": SceneEntityCfg("robot"),
                "semantic_R_ha": GM_LEAP_SEMANTIC_R_HA,
                "frame": "h",
                "representation": "rot6d",
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        r"""Critic-facing privileged observation group."""

    policy: ObsGroup = PolicyCfg(history_length=1)
    critic: ObsGroup = CriticCfg(history_length=1)


@configclass
class GmLeapRewardsCfg:
    r"""Reward terms reused from the generated single-asset ablation."""

    track_orientation = RewTerm(
        func=gm_mdp.keypoint_reorientation_reward,
        weight=1.0,
        params={"command_name": "goal_pose", "object_cfg": SceneEntityCfg("object")},
    )
    axis_progress = RewTerm(
        func=gm_mdp.AxisDeltaRotationReward,
        weight=2.5,
        params={"command_name": "goal_pose", "object_cfg": SceneEntityCfg("object"), "clip_value": 0.025},
    )
    success_bonus = RewTerm(
        func=gm_mdp.goal_success_bonus,
        weight=5.0,
        params={"command_name": "goal_pose", "object_cfg": SceneEntityCfg("object"), "success_mode": "so3"},
    )
    good_contact = RewTerm(
        func=gm_mdp.good_fingertip_contact,
        weight=0.5,
        params={
            "sensor_names": GM_LEAP_CONTACT_LAYOUT.fingertip_sensor_names,
            "min_contacts": 2,
            "force_threshold": 0.2,
            "lambda_floor": 0.05,
        },
    )
    bad_non_tip_contact = RewTerm(
        func=gm_mdp.bad_non_tip_contact,
        weight=-0.2,
        params={
            "sensor_names": GM_LEAP_CONTACT_LAYOUT.non_tip_sensor_names,
            "force_threshold": 0.2,
            "lambda_floor": 0.0,
        },
    )
    action_l2 = RewTerm(func=gm_mdp.action_l2_curriculum, weight=-1.0e-4, params={"lambda_floor": 0.0})
    action_rate_l2 = RewTerm(func=gm_mdp.action_rate_l2_curriculum, weight=-1.0e-2, params={"lambda_floor": 0.0})


@configclass
class GmLeapEventsCfg:
    r"""Reset events for official LEAP.

    Unlike generated assets, this variant does not apply AnyMani structural
    collision filters. The official LEAP USD keeps its own collision semantics.
    """

    reset_robot_joints = EventTerm(
        func=isaac_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reset_object = EventTerm(
        func=isaac_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "yaw": (-math.pi, math.pi)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
    record_object_reset_anchor = EventTerm(
        func=gm_mdp.record_object_reset_anchor,
        mode="reset",
        params={"object_cfg": SceneEntityCfg("object")},
    )


@configclass
class GmLeapTerminationsCfg:
    r"""Timeout + object out-of-hand termination."""

    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)
    object_falling = DoneTerm(func=gm_mdp.object_out_of_hand, params={"fall_dist": 0.06})


@configclass
class GmLeapCurriculumCfg:
    r"""Reward release curriculum, kept identical to single-asset MLP probe."""

    reward_release = CurrTerm(
        func=gm_mdp.RewardCurriculumByGoalSuccess,
        params={
            "command_name": "goal_pose",
            "metric_key": "goal_success_count",
            "g_min": 1.0,
            "g_max": 2.0,
            "ema_alpha": 0.05,
        },
    )


@configclass
class GmLeapEnvCfg(ManagerBasedRLEnvCfg):
    r"""Official LEAP hand MLP MDP probe environment."""

    scene: GmLeapSceneCfg = GmLeapSceneCfg(
        num_envs=2048,
        env_spacing=0.75,
        replicate_physics=False,
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

    observations: GmLeapObservationsCfg = GmLeapObservationsCfg()
    actions: GmLeapActionsCfg = GmLeapActionsCfg()
    commands: GmLeapCommandsCfg = GmLeapCommandsCfg()
    rewards: GmLeapRewardsCfg = GmLeapRewardsCfg()
    terminations: GmLeapTerminationsCfg = GmLeapTerminationsCfg()
    events: GmLeapEventsCfg = GmLeapEventsCfg()
    curriculum: GmLeapCurriculumCfg = GmLeapCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (2.0, 2.0, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)


@configclass
class GmLeapEnvCfg_PLAY(GmLeapEnvCfg):
    r"""Official LEAP GUI / command marker check environment."""

    commands: GmLeapCommandsCfg = GmLeapCommandsCfg(
        goal_pose=gm_mdp.ReorientCommandCfg(
            asset_name="object",
            robot_asset_name="robot",
            axis_mode="fixed",
            axis_resample_mode="episode",
            debug_vis=True,
            fixed_axis_h=(0.0, 0.0, 1.0),
            semantic_R_ha=GM_LEAP_SEMANTIC_R_HA,
        )
    )

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
        self.terminations.time_out = None


__all__ = [
    "GM_LEAP_CONTACT_LAYOUT",
    "GM_LEAP_GRASP_PRESET",
    "GM_LEAP_GRASP_PRESET_PATH",
    "GM_LEAP_HAND_CFG",
    "GM_LEAP_SEMANTIC_P_HA",
    "GM_LEAP_SEMANTIC_R_HA",
    "GmLeapActionsCfg",
    "GmLeapCommandsCfg",
    "GmLeapCurriculumCfg",
    "GmLeapEnvCfg",
    "GmLeapEnvCfg_PLAY",
    "GmLeapEventsCfg",
    "GmLeapObservationsCfg",
    "GmLeapRewardsCfg",
    "GmLeapSceneCfg",
    "GmLeapTerminationsCfg",
]
