'ADR-off, scale-1.1 palm-supported rotation for generated hands. Consume rank-0 from the schema-3 good-pregrasp catalog and retain the N000 moving-goal, contact, reward-release, and termination semantics. Actor observations include each JOINT owner contact and TIP contact; object, task, and force privilege remains in the separate critic. The formal launcher sets its 80 rows and 2,560 environments before importing this config; smaller row sets are runtime smoke tests only.'

from __future__ import annotations

import math
from typing import cast

import isaaclab.envs.mdp as isaac_mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from ...mdp import commands as command_mdp
from ...mdp import observations as observation_mdp
from ...mdp import rewards as reward_mdp
from ...mdp import terminations as termination_mdp
from ...mdp.actions import POLICY_STEP_AUTHORITY_RAD, PreloadAwareMaskedRelativeJointPositionActionCfg
from ...mdp.adr import HeterogeneousAdrCfg
from ...mdp.contact_state import reset_contact_state
from ...mdp.curriculums import RewardReleaseByAssetMedianCell, reward_release_observation
from ...mdp.events import (
    apply_structural_collision_filter,
    lock_ghost_joint_limits,
    reset_from_good_pregrasp_catalog,
    validate_formal_object_physics,
)
from .good_pregrasp_identity import (
    GOOD_PREGRASP_OBJECT_SCALE,
    GOOD_PREGRASP_PHYSICS_IDENTITY,
)
from .pregrasp_identity import (
    FORMAL_CONTACT_EMA_ALPHA,
    FORMAL_CONTACT_FORCE_THRESHOLD_N,
    FORMAL_DYNAMIC_FRICTION,
    FORMAL_PHYSICS_DT_S,
    FORMAL_RESTITUTION,
    FORMAL_STATIC_FRICTION,
)
from .scene import ACTIVE_MASK_BY_ENV, ASSET_BINDING, CONTACT_LAYOUT, NUM_ENVS, GeneratedHeterogeneousSceneCfg

GOOD_PREGRASP_RESET_CFG = ASSET_BINDING.build_good_pregrasp_reset_cfg(num_envs=NUM_ENVS, rank=0)
'The current selection’s exact scale-1.1 rank-0 reset binding.'


def _contact_params() -> dict[str, object]:
    'Return the shared 20 Hz contact-state configuration used by actor, critic, and reward.'

    return {
        "layout": CONTACT_LAYOUT,
        "active_joint_mask_by_env": ACTIVE_MASK_BY_ENV,
        "ema_alpha": FORMAL_CONTACT_EMA_ALPHA,
        "force_threshold_N": FORMAL_CONTACT_FORCE_THRESHOLD_N,
    }


@configclass
class PalmRotationMvpActionsCfg:
    'Canonical 16-slot target action relative to pregrasp; maximum step is 1/24 rad per policy step.'

    hand_joint_pos = PreloadAwareMaskedRelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        preserve_order=True,
        scale=POLICY_STEP_AUTHORITY_RAD,
        use_zero_offset=True,
    )


@configclass
class PalmRotationMvpCommandsCfg:
    'Physical 30-degree frontier along hand +z and a separate strict moving-goal command.'

    goal_pose = command_mdp.HeterogeneousRotationCommandCfg(
        object_name="object",
        robot_name="robot",
        fixed_axis_h=(0.0, 0.0, 1.0),
        semantic_R_ha=tuple(ASSET_BINDING.hand_spawn_cfg.frame.semantic_R_ha),
        subgoal_angle_rad=math.pi / 6.0,
        rotation_frontier_interval_rad=math.pi / 6.0,
        keypoint_radius_m=0.05,
        orientation_success_threshold_m=0.005,
        position_success_threshold_m=0.025,
        speed_ema_time_constant_s=0.25,
        horizon_s=120.0,
        dataset_row_by_env=ASSET_BINDING.dataset_row_by_env(NUM_ENVS),
        log_asset_metrics=False,
    )


@configclass
class PalmRotationMvpActorObsCfg(ObsGroup):
    'Raw simulation-contact actor observation O-a; object, task, and force are excluded.'

    jnt_current = ObsTerm(
        func=observation_mdp.actor_joint_contact_frame_term,
        params={"action_name": "hand_joint_pos", **_contact_params()},
    )  # Shape [N,16,5]: q, u, a, current-joint contact, and tip contact.
    jnt_history = ObsTerm(
        func=observation_mdp.actor_joint_contact_frame_term,
        params={"action_name": "hand_joint_pos", **_contact_params()},
        history_length=30,
        flatten_history_dim=False,
    )  # `[N,30,16,5]` oldest-to-latest，1.5 s
    jnt_limits = ObsTerm(
        func=observation_mdp.actor_joint_limits_term,
        params={"active_joint_mask_by_env": ACTIVE_MASK_BY_ENV},
    )  # Shape [N,16,2]: joint limits divided by pi.
    owner_contact = ObsTerm(
        func=observation_mdp.actor_owner_contact_term,
        params=_contact_params(),
    )  # Shape [N,21,1]: current binary contact; global residual actor may read it.
    jnt_valid = ObsTerm(func=observation_mdp.joint_valid, params={"active_joint_mask_by_env": ACTIVE_MASK_BY_ENV})
    tip_valid = ObsTerm(func=observation_mdp.tip_valid, params={"active_joint_mask_by_env": ACTIVE_MASK_BY_ENV})
    owner_valid = ObsTerm(func=observation_mdp.owner_valid, params={"active_joint_mask_by_env": ACTIVE_MASK_BY_ENV})

    def __post_init__(self) -> None:
        'Keep all role/history axes and disable observation corruption.'

        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class PalmRotationMvpCriticObsCfg(ObsGroup):
    'Fully privileged structured critic observation O-c.'

    jnt_state = ObsTerm(
        func=observation_mdp.critic_joint_state_term,
        params={"active_joint_mask_by_env": ACTIVE_MASK_BY_ENV, "action_name": "hand_joint_pos"},
    )  # Shape [N,16,4]: q, q-dot, u, and a.
    owner_contact = ObsTerm(func=observation_mdp.critic_owner_contact_term, params=_contact_params())
    obj = ObsTerm(
        func=observation_mdp.critic_object_term,
        params={
            "command_name": "goal_pose",
            "semantic_R_ha": tuple(ASSET_BINDING.hand_spawn_cfg.frame.semantic_R_ha),
        },
    )  # `[N,1,15]` object pose/twist
    task = ObsTerm(func=observation_mdp.critic_task_term, params={"command_name": "goal_pose"})  # `[N,1,8]`
    reward_release = ObsTerm(func=reward_release_observation)  # Shape [N,1]: actual cell-level reward scale lambda.
    jnt_valid = ObsTerm(func=observation_mdp.joint_valid, params={"active_joint_mask_by_env": ACTIVE_MASK_BY_ENV})
    tip_valid = ObsTerm(func=observation_mdp.tip_valid, params={"active_joint_mask_by_env": ACTIVE_MASK_BY_ENV})
    owner_valid = ObsTerm(func=observation_mdp.owner_valid, params={"active_joint_mask_by_env": ACTIVE_MASK_BY_ENV})

    def __post_init__(self) -> None:
        'Keep named critic axes; do not inject asset-row or cell one-hot features.'

        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class PalmRotationMvpObservationsCfg:
    'Separate non-concatenated actor and critic observation groups.'

    policy: ObsGroup = PalmRotationMvpActorObsCfg()
    critic: ObsGroup = PalmRotationMvpCriticObsCfg()


@configclass
class PalmRotationMvpRewardsCfg:
    'Retain N000 pose, progress, strict-goal, and stability rewards. Training anchors are keypoint pose 1, signed progress 5, and the 10-point goal bonus gated by pose and 2.5 cm position. The 30-degree frontier is diagnostic and for scale-ready evaluation only. Preserve the 7 cm position and 45-degree alignment failure terms.'

    pose_keypoint = RewTerm(func=reward_mdp.pose_keypoint_reward, weight=1.0, params={"command_name": "goal_pose"})
    orientation_tracking: RewTerm | None = None  # The new task explicitly replaces knowledge distillation; declare before failure to preserve the final-term snapshot boundary.
    rotation_progress = RewTerm(
        func=reward_mdp.signed_rotation_progress_rate,
        weight=5.0,
        params={"command_name": "goal_pose", "clip_rad_per_step": 0.025},
    )
    goal_success = RewTerm(
        func=reward_mdp.goal_success_impulse_rate,
        weight=10.0,
        params={"command_name": "goal_pose"},
    )
    good_tip_contact = RewTerm(
        func=reward_mdp.good_tip_contact_curriculum,
        weight=0.1,
        params={**_contact_params(), "minimum_tip_contacts": 2},
    )
    bad_finger_non_tip_contact = RewTerm(
        func=reward_mdp.bad_finger_non_tip_contact_curriculum,
        weight=-0.2,
        params=_contact_params(),
    )
    speed_band = RewTerm(
        func=reward_mdp.object_axis_speed_band_curriculum,
        weight=-0.5,
        params={"command_name": "goal_pose", "speed_min_rad_s": 0.6, "speed_max_rad_s": 0.833},
    )
    speed_jitter = RewTerm(
        func=reward_mdp.object_axis_speed_jitter_curriculum,
        weight=-0.05,
        params={"command_name": "goal_pose"},
    )
    off_axis_angular_velocity = RewTerm(
        func=reward_mdp.object_off_axis_angular_velocity_curriculum,
        weight=-0.05,
        params={"command_name": "goal_pose"},
    )
    object_linear_velocity = RewTerm(
        func=reward_mdp.object_linear_velocity_curriculum,
        weight=-0.2,
        params={"command_name": "goal_pose"},
    )
    joint_pose_anchor = RewTerm(func=reward_mdp.joint_pose_anchor_curriculum, weight=-0.5)
    mechanical_power = RewTerm(func=reward_mdp.joint_mechanical_power_curriculum, weight=-0.1)
    torque_l2 = RewTerm(func=reward_mdp.torque_l2_curriculum, weight=-0.05)
    action_l2 = RewTerm(func=reward_mdp.action_l2_curriculum, weight=-1.0e-4)
    action_rate_l2 = RewTerm(func=reward_mdp.action_rate_l2_curriculum, weight=-1.0e-2)
    # The last term freezes the post-physics, pre-reset trajectory snapshot; do not add later rewards that update commands or contact state.
    failure = RewTerm(
        func=reward_mdp.failure_termination_impulse_rate,
        weight=-50.0,
        params={
            "command_name": "goal_pose",
            "termination_term_names": ("object_out_of_anchor", "goal_axis_misaligned"),
            **_contact_params(),
        },
    )


@configclass
class PalmRotationMvpTerminationsCfg:
    'N000 termination semantics: 7 cm anchor distance, signed 45-degree normal alignment, and 120-second timeout.'

    object_out_of_anchor = DoneTerm(
        func=termination_mdp.object_out_of_anchor,
        params={"command_name": "goal_pose", "drop_distance_m": 0.07},
    )
    goal_axis_misaligned = DoneTerm(
        func=termination_mdp.goal_axis_misaligned,
        params={"command_name": "goal_pose", "max_axis_angle_deg": 45.0},
    )
    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)


@configclass
class PalmRotationMvpEventsCfg:
    'Install structural filters, ghost-joint locks, the scale-1.1 physics gate, and schema-3 rank-0 reset.'

    structural_collision_filter = EventTerm(
        func=apply_structural_collision_filter,
        mode="prestartup",
        params={
            "robot_prim_path": "{ENV_REGEX_NS}/Robot",
            "palm_link_name": CONTACT_LAYOUT.palm_link,
            "finger_link_chains": CONTACT_LAYOUT.finger_link_chains,
        },
    )
    ghost_joint_lock = EventTerm(
        func=lock_ghost_joint_limits,
        mode="startup",
        params={"active_joint_mask_by_env": ACTIVE_MASK_BY_ENV, "robot_name": "robot"},
    )
    object_physics_identity = EventTerm(
        func=validate_formal_object_physics,
        mode="startup",
        params={"expected_physics_identity": dict(GOOD_PREGRASP_PHYSICS_IDENTITY)},
    )
    good_pregrasp_reset = EventTerm(
        func=reset_from_good_pregrasp_catalog,
        mode="reset",
        params={"config": GOOD_PREGRASP_RESET_CFG},
    )
    contact_reset = EventTerm(func=reset_contact_state, mode="reset", params=_contact_params())
    episode_horizon: EventTerm | None = None  # The launcher sets the planned duration; fixed evaluation does not sample training horizons.


@configclass
class PalmRotationMvpCurriculumCfg:
    'Use per-asset EMA and the median of eight cells for release; keep ADR disabled.'

    reward_release = CurrTerm(
        func=RewardReleaseByAssetMedianCell,  # pyright: ignore[reportArgumentType]  # ManagerTermBase class-term
        params={
            "command_name": "goal_pose",
            "dataset_rows_by_asset": ASSET_BINDING.dataset_rows,
            "cell_ids_by_asset": ASSET_BINDING.morphology_cell_ids,
            "asset_index_by_env": ASSET_BINDING.asset_index_by_env(NUM_ENVS),
            "release_start_turns": 1.0,
            "release_end_turns": 2.0,
            "ema_alpha": 0.05,
        },
    )


@configclass
class GeneratedPalmRotationMvpEnvCfg(ManagerBasedRLEnvCfg):
    'Main 80-hand training environment; task and object stay fixed while morphology and pregrasp vary.'

    is_finite_horizon: bool = True
    adr: HeterogeneousAdrCfg = HeterogeneousAdrCfg()  # Each component has an independent switch; legacy configs default all components off.
    seed: int | None = 42
    scene: GeneratedHeterogeneousSceneCfg = GeneratedHeterogeneousSceneCfg(
        num_envs=NUM_ENVS,
        env_spacing=0.75,
        replicate_physics=False,
        filter_collisions=True,
        clone_in_fabric=False,
    )
    viewer: ViewerCfg = ViewerCfg()
    sim: SimulationCfg = SimulationCfg(
        physics_material=RigidBodyMaterialCfg(
            static_friction=FORMAL_STATIC_FRICTION,
            dynamic_friction=FORMAL_DYNAMIC_FRICTION,
            restitution=FORMAL_RESTITUTION,
            friction_combine_mode="average",
            restitution_combine_mode="average",
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
        ),
    )
    observations: PalmRotationMvpObservationsCfg = PalmRotationMvpObservationsCfg()
    actions: PalmRotationMvpActionsCfg = PalmRotationMvpActionsCfg()
    commands: PalmRotationMvpCommandsCfg = PalmRotationMvpCommandsCfg()
    rewards: PalmRotationMvpRewardsCfg = PalmRotationMvpRewardsCfg()
    terminations: PalmRotationMvpTerminationsCfg = PalmRotationMvpTerminationsCfg()
    events: PalmRotationMvpEventsCfg = PalmRotationMvpEventsCfg()
    curriculum: PalmRotationMvpCurriculumCfg = PalmRotationMvpCurriculumCfg()

    def __post_init__(self) -> None:
        'Lock scale 1.1, 120 Hz physics, 20 Hz policy, and a fixed 120-second horizon.'

        super().__post_init__()  # pyright: ignore[reportAttributeAccessIssue]
        object_spawn = cast(sim_utils.UsdFileCfg, self.scene.object.spawn)
        object_spawn.scale = (
            GOOD_PREGRASP_OBJECT_SCALE,
            GOOD_PREGRASP_OBJECT_SCALE,
            GOOD_PREGRASP_OBJECT_SCALE,
        )  # prestartup exact collision scale
        self.decimation = 6
        self.episode_length_s = 120.0
        self.sim.dt = FORMAL_PHYSICS_DT_S
        self.sim.render_interval = self.decimation
        self.viewer.eye = (2.0, 2.0, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)


__all__ = ["GeneratedPalmRotationMvpEnvCfg"]
