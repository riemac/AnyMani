r"""Generated canonical heterogeneous DexCube palm-up tactile-rotation ManagerBased environment。

Task科学接口为non-concatenated structured actor/critic tensors。Scene、command、contact、reward、termination、
pregrasp与action均由``tasks/hetero``拥有；资产lowering只调用``robots``，不import旧任务族。
"""

from __future__ import annotations

import math
import os

import isaaclab.envs.mdp as isaac_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from anymani.pregrasp import PregraspTier

from ...mdp import commands as command_mdp
from ...mdp import observations as observation_mdp
from ...mdp import rewards as reward_mdp
from ...mdp import terminations as termination_mdp
from ...mdp.actions import POLICY_STEP_AUTHORITY_RAD, PreloadAwareMaskedRelativeJointPositionActionCfg
from ...mdp.contact_state import reset_contact_state
from ...mdp.events import (
    apply_structural_collision_filter,
    lock_ghost_joint_limits,
    reset_from_pregrasp_cache,
    validate_formal_object_physics,
)
from .pregrasp_identity import (
    FORMAL_CONTACT_EMA_ALPHA,
    FORMAL_CONTACT_FORCE_THRESHOLD_N,
    FORMAL_DYNAMIC_FRICTION,
    FORMAL_PHYSICS_DT_S,
    FORMAL_RESTITUTION,
    FORMAL_STATIC_FRICTION,
)
from .scene import (
    ACTIVE_MASK_BY_ENV,
    ASSET_BINDING,
    CONTACT_LAYOUT,
    FORMAL_PREGRASP_IDENTITY,
    NUM_ENVS,
    OBJECT_SCALE,
    GeneratedHeterogeneousSceneCfg,
)


def _minimum_pregrasp_tier() -> PregraspTier:
    r"""解析support/contact环境合同；缺省support允许分层2-asset scene。"""

    raw = os.environ.get("ANYMANI_HETERO_MIN_PREGRASP_TIER", PregraspTier.SUPPORT_BASIN.value)
    tier = PregraspTier(raw)
    if tier not in {PregraspTier.SUPPORT_BASIN, PregraspTier.CONTACT_BASIN}:
        raise ValueError("heterogeneous env minimum pregrasp tier must be support_basin or contact_basin")
    return tier


MINIMUM_PREGRASP_TIER = _minimum_pregrasp_tier()
_exact_tier_raw = os.environ.get("ANYMANI_HETERO_EXACT_PREGRASP_TIER", "").strip()
EXACT_PREGRASP_TIER = PregraspTier(_exact_tier_raw) if _exact_tier_raw else MINIMUM_PREGRASP_TIER
PREGRASP_RESET_CFG = ASSET_BINDING.build_pregrasp_reset_cfg(
    num_envs=NUM_ENVS,
    object_scale=OBJECT_SCALE,
    minimum_tier=MINIMUM_PREGRASP_TIER,
    catalog_identity=FORMAL_PREGRASP_IDENTITY,
    exact_tier=EXACT_PREGRASP_TIER,
)


def _contact_params() -> dict[str, object]:
    r"""返回所有contact consumers共享的role/EMA/routing参数。"""

    return {
        "layout": CONTACT_LAYOUT,
        "active_joint_mask_by_env": ACTIVE_MASK_BY_ENV,
        "ema_alpha": FORMAL_CONTACT_EMA_ALPHA,
        "force_threshold_N": FORMAL_CONTACT_FORCE_THRESHOLD_N,
    }


@configclass
class GeneratedHeterogeneousActionsCfg:
    r"""Canonical 16-slot、每policy step$1/24$ rad preload-aware action。"""

    hand_joint_pos = PreloadAwareMaskedRelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        preserve_order=True,
        scale=POLICY_STEP_AUTHORITY_RAD,
        use_zero_offset=True,
    )


@configclass
class GeneratedHeterogeneousCommandsCfg:
    r"""固定hand$+z$、30°moving-subgoal command。"""

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
        log_asset_metrics=os.environ.get("ANYMANI_HETERO_LOG_ASSET_METRICS", "0") == "1",
    )


@configclass
class ActorObservationsCfg(ObsGroup):
    r"""Deployable$O^a$：JOINT current/History30/limits、TIP bits与validity masks。"""

    palm_valid = ObsTerm(func=observation_mdp.palm_valid)
    jnt_current = ObsTerm(
        func=observation_mdp.actor_joint_current_term,
        params={"active_joint_mask_by_env": ACTIVE_MASK_BY_ENV, "action_name": "hand_joint_pos"},
    )
    jnt_history = ObsTerm(
        func=observation_mdp.actor_joint_history_frame_term,
        params={
            "active_joint_mask_by_env": ACTIVE_MASK_BY_ENV,
            "action_name": "hand_joint_pos",
            **_contact_params(),
        },
        history_length=30,
        flatten_history_dim=False,
    )
    jnt_limits = ObsTerm(
        func=observation_mdp.actor_joint_limits_term,
        params={"active_joint_mask_by_env": ACTIVE_MASK_BY_ENV},
    )
    tip_contact = ObsTerm(
        func=observation_mdp.actor_tip_contact_term,
        params=_contact_params(),
    )
    jnt_valid = ObsTerm(func=observation_mdp.joint_valid, params={"active_joint_mask_by_env": ACTIVE_MASK_BY_ENV})
    tip_valid = ObsTerm(func=observation_mdp.tip_valid, params={"active_joint_mask_by_env": ACTIVE_MASK_BY_ENV})
    owner_valid = ObsTerm(
        func=observation_mdp.owner_valid, params={"active_joint_mask_by_env": ACTIVE_MASK_BY_ENV}
    )

    def __post_init__(self) -> None:
        r"""保留named role/rank tensors与term-level History30。"""

        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class CriticObservationsCfg(ObsGroup):
    r"""Privileged$O^c$：JOINT、all owners、object/task与validity masks。"""

    palm_valid = ObsTerm(func=observation_mdp.palm_valid)
    jnt_state = ObsTerm(
        func=observation_mdp.critic_joint_state_term,
        params={"active_joint_mask_by_env": ACTIVE_MASK_BY_ENV, "action_name": "hand_joint_pos"},
    )
    owner_contact = ObsTerm(func=observation_mdp.critic_owner_contact_term, params=_contact_params())
    obj = ObsTerm(
        func=observation_mdp.critic_object_term,
        params={
            "command_name": "goal_pose",
            "semantic_R_ha": tuple(ASSET_BINDING.hand_spawn_cfg.frame.semantic_R_ha),
        },
    )
    task = ObsTerm(func=observation_mdp.critic_task_term, params={"command_name": "goal_pose"})
    jnt_valid = ObsTerm(func=observation_mdp.joint_valid, params={"active_joint_mask_by_env": ACTIVE_MASK_BY_ENV})
    tip_valid = ObsTerm(func=observation_mdp.tip_valid, params={"active_joint_mask_by_env": ACTIVE_MASK_BY_ENV})
    owner_valid = ObsTerm(
        func=observation_mdp.owner_valid, params={"active_joint_mask_by_env": ACTIVE_MASK_BY_ENV}
    )

    def __post_init__(self) -> None:
        r"""Critic保持named tensors，不注入asset row或cell one-hot。"""

        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class GeneratedHeterogeneousObservationsCfg:
    r"""Actor与critic两个non-concatenated top-level groups。"""

    policy: ObsGroup = ActorObservationsCfg()
    critic: ObsGroup = CriticObservationsCfg()


@configclass
class GeneratedHeterogeneousRewardsCfg:
    r"""固定六项palm-supported tactile rotation baseline。"""

    pose_keypoint = RewTerm(func=reward_mdp.pose_keypoint_reward, weight=1.0, params={"command_name": "goal_pose"})
    rotation_progress = RewTerm(
        func=reward_mdp.signed_rotation_progress_rate,
        weight=5.0,
        params={"command_name": "goal_pose", "clip_rad_per_step": 0.025},
    )
    goal_success = RewTerm(
        func=reward_mdp.goal_success_impulse_rate, weight=10.0, params={"command_name": "goal_pose"}
    )
    good_tip_contact = RewTerm(func=reward_mdp.good_tip_contact, weight=0.1, params=_contact_params())
    bad_finger_non_tip_contact = RewTerm(
        func=reward_mdp.bad_finger_non_tip_contact, weight=-0.2, params=_contact_params()
    )
    failure = RewTerm(
        func=reward_mdp.failure_termination_impulse_rate,
        weight=-50.0,
        params={"termination_term_names": ("object_out_of_anchor", "goal_axis_misaligned")},
    )


@configclass
class GeneratedHeterogeneousTerminationsCfg:
    r"""7 cm anchor、signed 45° normal alignment与固定120 s timeout。"""

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
class GeneratedHeterogeneousEventsCfg:
    r"""Prestartup structural filter、startup ghost lock与唯一pregrasp reset writer。"""

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
        params={"expected_physics_identity": dict(FORMAL_PREGRASP_IDENTITY.physics_identity)},
    )
    pregrasp_reset = EventTerm(
        func=reset_from_pregrasp_cache,
        mode="reset",
        params={"config": PREGRASP_RESET_CFG},
    )
    contact_reset = EventTerm(
        func=reset_contact_state,
        mode="reset",
        params=_contact_params(),
    )


@configclass
class GeneratedHeterogeneousTactileRotationEnvCfg(ManagerBasedRLEnvCfg):
    r"""Formal generated support域；当前process selection由`ANYMANI_HETERO_ASSET_ROWS`决定。"""

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
    observations: GeneratedHeterogeneousObservationsCfg = GeneratedHeterogeneousObservationsCfg()
    actions: GeneratedHeterogeneousActionsCfg = GeneratedHeterogeneousActionsCfg()
    commands: GeneratedHeterogeneousCommandsCfg = GeneratedHeterogeneousCommandsCfg()
    rewards: GeneratedHeterogeneousRewardsCfg = GeneratedHeterogeneousRewardsCfg()
    terminations: GeneratedHeterogeneousTerminationsCfg = GeneratedHeterogeneousTerminationsCfg()
    events: GeneratedHeterogeneousEventsCfg = GeneratedHeterogeneousEventsCfg()
    curriculum = None  # baseline显式关闭ADR/curriculum

    def __post_init__(self) -> None:
        r"""锁定120 Hz physics、20 Hz policy与120 s fixed horizon。"""

        super().__post_init__()  # pyright: ignore[reportAttributeAccessIssue]
        self.decimation = 6
        self.episode_length_s = 120.0
        self.sim.dt = FORMAL_PHYSICS_DT_S
        self.sim.render_interval = self.decimation
        self.viewer.eye = (2.0, 2.0, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)


__all__ = ["GeneratedHeterogeneousTactileRotationEnvCfg"]
