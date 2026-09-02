r"""Pregrasp搜索与physics probe共用的cache-independent ManagerBased harness。

Harness不是训练任务且不注册Gym ID。它只建立与formal task相同的physical scene，再以官方relative action、alive reward和
timeout满足ManagerBasedRLEnv生命周期。Reset写canonical default q与DexCube default pose；搜索脚本随后显式写candidate
$q_s$、PD target $q_t$与$T_{ho}$。因此未发布cache不会被误当成可训练reset，同时搜索也不形成cache→task→搜索的循环依赖。
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
    r"""只为ManagerBased step提供16-slot official relative target；搜索通常直接写PD target。"""

    hand_joint_pos = isaac_mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        preserve_order=True,
        scale=1.0 / 24.0,  # 与formal policy-step最大增量相同，physics probe的zero action不改变target
        use_zero_offset=True,
    )


@configclass
class PregraspHarnessObservationsCfg:
    r"""仅用于ManagerBased shape/lifecycle的flat proprioceptive observation。"""

    @configclass
    class PolicyCfg(ObsGroup):
        r"""返回physical joint position/velocity；不声明训练actor contract。"""

        joint_pos = ObsTerm(func=isaac_mdp.joint_pos)
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel)

        def __post_init__(self) -> None:
            r"""关闭corruption并拼接harness-only tensor。"""

            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class PregraspHarnessRewardsCfg:
    r"""Alive常数只满足RL environment接口，不进入pregrasp认证指标。"""

    alive = RewTerm(func=isaac_mdp.is_alive, weight=1.0)


@configclass
class PregraspHarnessTerminationsCfg:
    r"""只保留长timeout；搜索以显式physics-step预算终止。"""

    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)


@configclass
class PregraspHarnessEventsCfg:
    r"""结构碰撞、ghost lock及确定性default reset，不读取任何pregrasp record。"""

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
    r"""非注册prestartup/search environment；$N$由`ANYMANI_HETERO_NUM_ENVS`在import前固定。"""

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
    commands = None  # 搜索直接构造$T_{ho}$，不生成rotation subgoal
    curriculum = None  # physics/pregrasp认证不改变domain

    def __post_init__(self) -> None:
        r"""锁定P0001 physics identity：120 Hz simulation、20 Hz policy与120 s safety timeout。"""

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
