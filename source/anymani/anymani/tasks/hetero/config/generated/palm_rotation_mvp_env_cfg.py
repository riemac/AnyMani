r"""MVP80 generated-hand、scale-1.1、ADR-0 palm-supported rotation task。

本配置直接消费schema-3 Top-8 good-pregrasp catalog的rank-0，并继承N000能力锚点的30°moving subgoal、
rotation/contact/stable reward-release与termination语义。Actor observation新增每个JOINT owner自身binary contact，
同时保留所属finger TIP contact；object/task/force仍只进入完全分参的privileged critic。

Formal launcher必须在import前从``ppo_mvp80.yaml``设置80个dataset rows与`N=2560` environments。少量rows只用于
runtime smoke；任务数学、object scale、reward和reset catalog保持相同。
"""

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
"""当前selection的exact scale-1.1 rank-0 reset binding。"""


def _contact_params() -> dict[str, object]:
    r"""返回actor/critic/reward共享的单一20 Hz contact-state配置。"""

    return {
        "layout": CONTACT_LAYOUT,
        "active_joint_mask_by_env": ACTIVE_MASK_BY_ENV,
        "ema_alpha": FORMAL_CONTACT_EMA_ALPHA,
        "force_threshold_N": FORMAL_CONTACT_FORCE_THRESHOLD_N,
    }


@configclass
class PalmRotationMvpActionsCfg:
    r"""Canonical 16-slot、每policy step$1/24$ rad的pregrasp-relative target action。"""

    hand_joint_pos = PreloadAwareMaskedRelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        preserve_order=True,
        scale=POLICY_STEP_AUTHORITY_RAD,
        use_zero_offset=True,
    )


@configclass
class PalmRotationMvpCommandsCfg:
    r"""N000同义的hand$+z$、30°moving-subgoal command。"""

    goal_pose = command_mdp.HeterogeneousRotationCommandCfg(
        object_name="object",
        robot_name="robot",
        fixed_axis_h=(0.0, 0.0, 1.0),
        semantic_R_ha=tuple(ASSET_BINDING.hand_spawn_cfg.frame.semantic_R_ha),
        subgoal_angle_rad=math.pi / 6.0,
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
    r"""Simulation-contact actor raw$O^a$；object/task/force不进入本组。"""

    jnt_current = ObsTerm(
        func=observation_mdp.actor_joint_contact_frame_term,
        params={"action_name": "hand_joint_pos", **_contact_params()},
    )  # `[N,16,5]`，当前$q/u/a/c_j/c_{tip}$ bypass
    jnt_history = ObsTerm(
        func=observation_mdp.actor_joint_contact_frame_term,
        params={"action_name": "hand_joint_pos", **_contact_params()},
        history_length=30,
        flatten_history_dim=False,
    )  # `[N,30,16,5]` oldest-to-latest，1.5 s
    jnt_limits = ObsTerm(
        func=observation_mdp.actor_joint_limits_term,
        params={"active_joint_mask_by_env": ACTIVE_MASK_BY_ENV},
    )  # `[N,16,2]`，$q_{min/max}/\pi$
    owner_contact = ObsTerm(
        func=observation_mdp.actor_owner_contact_term,
        params=_contact_params(),
    )  # `[N,21,1]` current binary contact，global residual读取
    jnt_valid = ObsTerm(func=observation_mdp.joint_valid, params={"active_joint_mask_by_env": ACTIVE_MASK_BY_ENV})
    tip_valid = ObsTerm(func=observation_mdp.tip_valid, params={"active_joint_mask_by_env": ACTIVE_MASK_BY_ENV})
    owner_valid = ObsTerm(func=observation_mdp.owner_valid, params={"active_joint_mask_by_env": ACTIVE_MASK_BY_ENV})

    def __post_init__(self) -> None:
        r"""保留所有role/history axes并关闭observation corruption。"""

        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class PalmRotationMvpCriticObsCfg(ObsGroup):
    r"""完全privileged、structured hand-level critic raw$O^c$。"""

    jnt_state = ObsTerm(
        func=observation_mdp.critic_joint_state_term,
        params={"active_joint_mask_by_env": ACTIVE_MASK_BY_ENV, "action_name": "hand_joint_pos"},
    )  # `[N,16,4]`，$q/\dot q/u/a$
    owner_contact = ObsTerm(func=observation_mdp.critic_owner_contact_term, params=_contact_params())
    obj = ObsTerm(
        func=observation_mdp.critic_object_term,
        params={
            "command_name": "goal_pose",
            "semantic_R_ha": tuple(ASSET_BINDING.hand_spawn_cfg.frame.semantic_R_ha),
        },
    )  # `[N,1,15]` object pose/twist
    task = ObsTerm(func=observation_mdp.critic_task_term, params={"command_name": "goal_pose"})  # `[N,1,8]`
    reward_release = ObsTerm(func=reward_release_observation)  # `[N,1]` actual cell-level$\lambda_{rew}$
    jnt_valid = ObsTerm(func=observation_mdp.joint_valid, params={"active_joint_mask_by_env": ACTIVE_MASK_BY_ENV})
    tip_valid = ObsTerm(func=observation_mdp.tip_valid, params={"active_joint_mask_by_env": ACTIVE_MASK_BY_ENV})
    owner_valid = ObsTerm(func=observation_mdp.owner_valid, params={"active_joint_mask_by_env": ACTIVE_MASK_BY_ENV})

    def __post_init__(self) -> None:
        r"""Critic tensors保持named axes；不注入asset row或cell one-hot。"""

        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class PalmRotationMvpObservationsCfg:
    r"""Actor/critic两个non-concatenated top-level observation groups。"""

    policy: ObsGroup = PalmRotationMvpActorObsCfg()
    critic: ObsGroup = PalmRotationMvpCriticObsCfg()


@configclass
class PalmRotationMvpRewardsCfg:
    r"""N000 rotation＋cell-released contact/stable＋failure impulse reward。"""

    pose_keypoint = RewTerm(func=reward_mdp.pose_keypoint_reward, weight=1.0, params={"command_name": "goal_pose"})
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
    # 最后一个term冻结post-physics/pre-reset trajectory snapshot；后续不得在其后增加会更新command/contact的reward。
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
    r"""N000的7 cm anchor、signed 45° normal alignment与120 s timeout。"""

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
    r"""结构过滤、ghost lock、scale-1.1 physics gate与schema-3 rank-0 reset。"""

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


@configclass
class PalmRotationMvpCurriculumCfg:
    r"""Per-asset EMA、8-cell median实际release；ADR保持关闭。"""

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
    r"""80手主训练环境；同一对象/任务，仅morphology与pregrasp随asset变化。"""

    is_finite_horizon: bool = True
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
        r"""锁定scale1.1、120 Hz physics、20 Hz policy与120 s fixed horizon。"""

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
