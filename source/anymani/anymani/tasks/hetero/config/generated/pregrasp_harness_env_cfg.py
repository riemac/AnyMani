"""Cache-independent ManagerBased harness for pregrasp search and physics probes.

This unregistered harness builds the paper scene and provides a relative-action term, a lifecycle-only reward, and a timeout. Reset uses canonical default joint positions and the default DexCube pose; the search driver then writes candidate joint positions, PD targets, and the hand-object transform explicitly. Unpublished candidates are never treated as trained resets.
"""

from __future__ import annotations

import isaaclab.envs.mdp as isaac_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from ...mdp.events import apply_structural_collision_filter, lock_ghost_joint_limits
from .scene import ACTIVE_MASK_BY_ENV, CONTACT_LAYOUT, GeneratedHeterogeneousSceneCfg, NUM_ENVS


@configclass
class PregraspHarnessActionsCfg:
    """Provide the 16-slot relative target required by ManagerBased stepping."""

    hand_joint_pos = isaac_mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        preserve_order=True,
        scale=1.0 / 24.0,  # Match formal action authority; zero-action probes hold the target.
        use_zero_offset=True,
    )


@configclass
class PregraspHarnessObservationsCfg:
    """Flat proprioception for ManagerBased shape and lifecycle checks only."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Expose physical joint position and velocity without defining a training ABI."""

        joint_pos = ObsTerm(func=isaac_mdp.joint_pos)
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel)

        def __post_init__(self) -> None:
            """Disable corruption and concatenate this harness-only observation."""

            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class PregraspHarnessRewardsCfg:
    """Use a lifecycle-only alive reward that is excluded from pregrasp metrics."""

    alive = RewTerm(func=isaac_mdp.is_alive, weight=1.0)


@configclass
class PregraspHarnessTerminationsCfg:
    """Keep a long timeout; search uses an explicit physics-step budget."""

    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)


@configclass
class PregraspHarnessEventsCfg:
    """Configure structural filters, ghost locks, and deterministic default resets."""

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
    reset_robot_joints = EventTerm(
        func=isaac_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"], preserve_order=True),
        },
    )
    reset_object = EventTerm(
        func=isaac_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class GeneratedPregraspHarnessEnvCfg(ManagerBasedRLEnvCfg):
    """Unregistered search environment with batch size fixed before task import."""

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
        physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
        ),
    )
    observations: PregraspHarnessObservationsCfg = PregraspHarnessObservationsCfg()
    actions: PregraspHarnessActionsCfg = PregraspHarnessActionsCfg()
    rewards: PregraspHarnessRewardsCfg = PregraspHarnessRewardsCfg()
    terminations: PregraspHarnessTerminationsCfg = PregraspHarnessTerminationsCfg()
    events: PregraspHarnessEventsCfg = PregraspHarnessEventsCfg()
    commands = None  # Search writes the hand-object transform directly.
    curriculum = None  # Physics and pregrasp checks do not change the domain.

    def __post_init__(self) -> None:
        """Fix 120 Hz simulation, 20 Hz policy, and a 120 s timeout."""

        super().__post_init__()  # pyright: ignore[reportAttributeAccessIssue]
        self.decimation = 6
        self.episode_length_s = 120.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (2.0, 2.0, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)


__all__ = [
    "GeneratedPregraspHarnessEnvCfg",
    "PregraspHarnessActionsCfg",
    "PregraspHarnessEventsCfg",
]
