r"""2048 canonical assets 的 N000 tactile-rotation heterogeneous environment。

该环境保留 N000 的 DexCube、palm support、hand ``+z`` 30° moving subgoal、20 Hz tactile state、
pose/progress/success/contact/failure reward 与 120 Hz physics；删除 ADR、curriculum、reward release、
history/TCN、外力和 episode-length randomization。

Actor flat group 为 69D：

$$
[q/\pi]_{16}+[u/\pi]_{16}+[a_{t-1}]_{16}+[c_{tip}]_4+[asset\_row]_1+[m]_{16}.
$$

前 52D 是 deployable N000 current frame；后 17D 只路由 manifest/frozen $Z$。Privileged critic 为
103D：``q16+qd16+u16+a16+object15+tip4+palm1+finger_non_tip19``。
"""

from __future__ import annotations

import math

import isaaclab.envs.mdp as isaac_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.common import ViewerCfg
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

from anymani.tools.grasp_preset import GraspPreset, asset_preset_path

from ... import mdp as gm_mdp
from ...contact_sensors import install_contact_sensors
from .asset_runtime import (
    HETEROGENEOUS_ACTIVE_MASK_ROWS,
    HETEROGENEOUS_ASSET_ROWS,
    HETEROGENEOUS_CELL_ID_ROWS,
    HETEROGENEOUS_CONTACT_LAYOUT,
    HETEROGENEOUS_HAND_ADAPTER,
    HETEROGENEOUS_HAND_SPAWN_CFG,
    HETEROGENEOUS_OBJECT_OFFSET_ROWS,
    HETEROGENEOUS_RESET_Q_ROWS,
)

HETEROGENEOUS_JOINT_CFG = SceneEntityCfg("robot", joint_names=[".*"], preserve_order=True)
"""Action、actor、critic、reset 共用的 canonical depth-major 16-joint axis。"""

HETEROGENEOUS_GRASP_PRESET = GraspPreset.from_yaml(
    asset_preset_path("generated_asset", "right_t4_i4_m4_r4"),
    expected_hand_source="generated_bundle",
    expected_hand_ref_contains="right_t4_i4_m4_r4",
)
"""N000 DexCube object anchor；heterogeneous hand joints 则按各资产 q-home reset。"""


def _contact_params() -> dict[str, object]:
    r"""返回每个 tactile consumer 独立持有的同语义参数 mapping。"""

    return {
        "fingertip_sensor_names": HETEROGENEOUS_CONTACT_LAYOUT.fingertip_sensor_names,
        "finger_non_tip_sensor_names": HETEROGENEOUS_CONTACT_LAYOUT.finger_non_tip_sensor_names,
        "palm_sensor_name": HETEROGENEOUS_CONTACT_LAYOUT.palm_sensor_name,
        "ema_alpha": 0.5,
        "force_threshold": 0.25,
    }


@configclass
class HeterogeneousTactileRotationSceneCfg(InteractiveSceneCfg):
    r"""2048 canonical hand prototypes + DexCube + support scene。"""

    robot = HETEROGENEOUS_HAND_ADAPTER.build_articulation_cfg(prim_path="{ENV_REGEX_NS}/Robot")
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
            scale=(1.2, 1.2, 1.2),  # N000 fixed DexCube scale；不启用 ADR/random scale
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=HETEROGENEOUS_GRASP_PRESET.object_pos_cfg,
            rot=HETEROGENEOUS_GRASP_PRESET.object_rot_wxyz,
        ),
    )
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

    def __post_init__(self) -> None:
        r"""安装 24 个 object-filtered sensors，保留 friction、去除未消费 history/timers。"""

        super().__post_init__()  # pyright: ignore[reportAttributeAccessIssue]  # configclass base hook
        install_contact_sensors(
            self,
            HETEROGENEOUS_CONTACT_LAYOUT,
            history_length=0,
            track_air_time=False,
            track_friction_forces=True,
            max_contact_data_count_per_prim=16,
        )


@configclass
class HeterogeneousTactileRotationCommandsCfg:
    r"""固定 hand ``+z`` 轴、30° moving-subgoal command。"""

    goal_pose: gm_mdp.TactileRotationCommandCfg = gm_mdp.TactileRotationCommandCfg(
        asset_name="object",
        robot_asset_name="robot",
        fixed_axis_h=(0.0, 0.0, 1.0),
        semantic_R_ha=HETEROGENEOUS_HAND_SPAWN_CFG.frame.semantic_R_ha,
        subgoal_angle=math.pi / 6.0,
        keypoint_radius=0.05,
        orientation_keypoint_success_threshold=0.005,
        position_success_threshold=0.025,
        speed_ema_time_constant_s=0.25,
        diagnostics_action_name="hand_joint_pos",
        diagnostics_fingertip_sensor_names=HETEROGENEOUS_CONTACT_LAYOUT.fingertip_sensor_names,
        diagnostics_finger_non_tip_sensor_names=HETEROGENEOUS_CONTACT_LAYOUT.finger_non_tip_sensor_names,
        diagnostics_palm_sensor_name=HETEROGENEOUS_CONTACT_LAYOUT.palm_sensor_name,
        diagnostics_contact_ema_alpha=0.5,
        diagnostics_contact_force_threshold=0.25,
        resampling_time_range=(1.0e6, 1.0e6),
    )


@configclass
class HeterogeneousTactileRotationActionsCfg:
    r"""每个 20 Hz policy step 累加一次 0.1 rad masked delta；六个 substeps 只 hold。"""

    hand_joint_pos = gm_mdp.PolicyStepMaskedRelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        preserve_order=True,
        scale=0.1,
        clip={".*": (-0.1, 0.1)},
        use_zero_offset=True,
    )


@configclass
class HeterogeneousN040HistoryActionsCfg:
    r"""继承N000 accepted TCN的20 Hz、每policy-step一次$1/24$ rad相对target。"""

    hand_joint_pos = gm_mdp.PolicyStepMaskedRelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        preserve_order=True,
        scale=1.0 / 24.0,
        clip={".*": (-1.0 / 24.0, 1.0 / 24.0)},
        use_zero_offset=True,
    )


@configclass
class HeterogeneousPolicyObsCfg(ObsGroup):
    r"""52D N000 current frame + 17D manifest routing certificate。"""

    joint_pos = ObsTerm(func=gm_mdp.joint_pos_raw, params={"asset_cfg": HETEROGENEOUS_JOINT_CFG}, scale=1.0 / math.pi)
    joint_target = ObsTerm(func=gm_mdp.joint_target, params={"action_name": "hand_joint_pos"}, scale=1.0 / math.pi)
    last_policy_action = ObsTerm(func=gm_mdp.last_action, params={"action_name": "hand_joint_pos"})
    tip_contact_bits = ObsTerm(func=gm_mdp.tip_contact_bits_ema, params=_contact_params())
    asset_row = ObsTerm(func=gm_mdp.canonical_asset_row)
    active_joint_mask = ObsTerm(func=gm_mdp.canonical_active_joint_mask)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class HeterogeneousN040HistoryPolicyObsCfg(ObsGroup):
    r"""1969D flat ABI：逐JOINT History30 + static limits + routing certificate。

    `joint_history`由ObservationManager形成`[B,30,16,4]`后flatten为1920D；limits不进历史，
    保持32D；末尾仍是`asset_row1 + active_mask16`。Flat tensor只服务rl_games transport，policy
    adapter立即恢复结构化axes。
    """

    joint_history = ObsTerm(
        func=gm_mdp.per_joint_policy_frame,
        params={
            "asset_cfg": HETEROGENEOUS_JOINT_CFG,
            "action_name": "hand_joint_pos",
            **_contact_params(),
        },
        history_length=30,
        flatten_history_dim=True,
    )
    joint_limits = ObsTerm(
        func=gm_mdp.joint_soft_pos_limits,
        params={"asset_cfg": HETEROGENEOUS_JOINT_CFG},
        scale=1.0 / math.pi,
    )
    asset_row = ObsTerm(func=gm_mdp.canonical_asset_row)
    active_joint_mask = ObsTerm(func=gm_mdp.canonical_active_joint_mask)

    def __post_init__(self) -> None:
        r"""关闭corruption并保持各term自有history配置，不做group-level覆盖。"""

        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class HeterogeneousCriticObsCfg(ObsGroup):
    r"""103D privileged state；删除 48D ADR 与 1D reward-release。"""

    joint_pos = ObsTerm(func=gm_mdp.joint_pos_raw, params={"asset_cfg": HETEROGENEOUS_JOINT_CFG}, scale=1.0 / math.pi)
    joint_velocity = ObsTerm(func=gm_mdp.joint_vel_raw, params={"asset_cfg": HETEROGENEOUS_JOINT_CFG})
    joint_target = ObsTerm(func=gm_mdp.joint_target, params={"action_name": "hand_joint_pos"}, scale=1.0 / math.pi)
    last_policy_action = ObsTerm(func=gm_mdp.last_action, params={"action_name": "hand_joint_pos"})
    object_task_state = ObsTerm(
        func=gm_mdp.object_goal_task_state,
        params={
            "command_name": "goal_pose",
            "semantic_R_ha": HETEROGENEOUS_HAND_SPAWN_CFG.frame.semantic_R_ha,
            "robot_cfg": HETEROGENEOUS_JOINT_CFG,
            "object_cfg": SceneEntityCfg("object"),
        },
    )
    tip_force_ema = ObsTerm(func=gm_mdp.tip_force_magnitude_ema, params=_contact_params())
    palm_force_ema = ObsTerm(func=gm_mdp.palm_force_magnitude_ema, params=_contact_params())
    finger_non_tip_bits = ObsTerm(func=gm_mdp.finger_non_tip_contact_bits_ema, params=_contact_params())

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class HeterogeneousN040CriticObsCfg(HeterogeneousCriticObsCfg):
    r"""103D task state + active mask16 + morphology cell one-hot8 = 127D task-aware critic。"""

    active_joint_mask = ObsTerm(func=gm_mdp.canonical_active_joint_mask)
    morphology_cell = ObsTerm(func=gm_mdp.canonical_morphology_cell_one_hot)


@configclass
class HeterogeneousObservationsCfg:
    r"""Deployable policy 与 privileged central critic groups。"""

    policy: ObsGroup = HeterogeneousPolicyObsCfg(history_length=1)
    critic: ObsGroup = HeterogeneousCriticObsCfg(history_length=1)


@configclass
class HeterogeneousN040HistoryObservationsCfg:
    r"""N040 History30 actor与127D morphology-conditioned privileged critic。"""

    policy: ObsGroup = HeterogeneousN040HistoryPolicyObsCfg()
    critic: ObsGroup = HeterogeneousN040CriticObsCfg(history_length=1)


@configclass
class HeterogeneousN040HistoryLegacyCriticObservationsCfg:
    r"""仅用于恢复已持久化run001/002 checkpoint的1969D actor + 103D critic ABI。"""

    policy: ObsGroup = HeterogeneousN040HistoryPolicyObsCfg()
    critic: ObsGroup = HeterogeneousCriticObsCfg(history_length=1)


@configclass
class HeterogeneousRewardsCfg:
    r"""N000 rotation/contact/failure core，无 curriculum-gated stable group。"""

    pose_keypoint = RewTerm(
        func=gm_mdp.tactile_full_pose_keypoint_reward,
        weight=1.0,
        params={"command_name": "goal_pose", "object_cfg": SceneEntityCfg("object"), "keypoint_radius": 0.05},
    )
    rotation_progress = RewTerm(
        func=gm_mdp.tactile_axis_delta_rotation_rate,
        weight=5.0,
        params={"command_name": "goal_pose", "clip_value": 0.025},
    )
    goal_success = RewTerm(func=gm_mdp.tactile_goal_success_impulse, weight=10.0, params={"command_name": "goal_pose"})
    good_tip_contact = RewTerm(
        func=gm_mdp.tactile_good_tip_contact_ungated,
        weight=0.1,
        params={**_contact_params(), "min_contacts": 2},
    )
    bad_finger_non_tip_contact = RewTerm(
        func=gm_mdp.tactile_bad_finger_non_tip_contact_ungated,
        weight=-0.2,
        params=_contact_params(),
    )
    failure = RewTerm(
        func=gm_mdp.failure_termination_impulse,
        weight=-50.0,
        params={"termination_term_names": ("object_out_of_anchor", "goal_axis_misaligned")},
    )


@configclass
class HeterogeneousFailure100RewardsCfg(HeterogeneousRewardsCfg):
    r"""稳定化消融：每次anchor/axis failure impulse从-50提高到-100，其余reward不变。"""

    failure = RewTerm(
        func=gm_mdp.failure_termination_impulse,
        weight=-100.0,
        params={"termination_term_names": ("object_out_of_anchor", "goal_axis_misaligned")},
    )


@configclass
class HeterogeneousTerminationsCfg:
    r"""7 cm anchor、signed 45° normal alignment 与固定 120 s timeout。"""

    object_out_of_anchor = DoneTerm(
        func=gm_mdp.tactile_object_out_of_anchor,
        params={"command_name": "goal_pose", "fall_dist": 0.07},
    )
    goal_axis_misaligned = DoneTerm(
        func=gm_mdp.tactile_goal_axis_misaligned,
        params={"command_name": "goal_pose", "max_angle_deg": 45.0},
    )
    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)


@configclass
class HeterogeneousEventsCfg:
    r"""Canonical routing/ghost lock + N000 reset/contact state；无 ADR/randomization/wrench。"""

    apply_structural_collision_filter = EventTerm(
        func=gm_mdp.apply_generated_structural_collision_filter,
        mode="prestartup",
        params={
            "robot_prim_path": "{ENV_REGEX_NS}/Robot",
            "palm_link_name": HETEROGENEOUS_CONTACT_LAYOUT.palm_link_name,
            "finger_link_chains": HETEROGENEOUS_CONTACT_LAYOUT.finger_link_chains,
            "filter_palm_finger": True,
            "filter_same_finger": True,
        },
    )
    initialize_runtime = EventTerm(
        func=gm_mdp.initialize_canonical_runtime_state,
        mode="startup",
        params={
            "active_joint_mask": HETEROGENEOUS_ACTIVE_MASK_ROWS,
            "asset_rows": HETEROGENEOUS_ASSET_ROWS,
            "q_home": HETEROGENEOUS_RESET_Q_ROWS,
            "morphology_cell_ids": HETEROGENEOUS_CELL_ID_ROWS,
            "object_position_offsets": HETEROGENEOUS_OBJECT_OFFSET_ROWS,
            "routing_mode": "round_robin",
        },
    )
    lock_ghost_limits = EventTerm(
        func=gm_mdp.lock_canonical_ghost_joint_limits,
        mode="startup",
        params={"asset_cfg": HETEROGENEOUS_JOINT_CFG},
    )
    reset_robot_joints = EventTerm(
        func=gm_mdp.reset_canonical_robot_joints,
        mode="reset",
        params={"asset_cfg": HETEROGENEOUS_JOINT_CFG},
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
    apply_object_offset = EventTerm(
        func=gm_mdp.apply_canonical_object_position_offset,
        mode="reset",
        params={"object_cfg": SceneEntityCfg("object")},
    )
    record_object_anchor = EventTerm(
        func=gm_mdp.record_object_reset_anchor,
        mode="reset",
        params={"object_cfg": SceneEntityCfg("object")},
    )
    reset_contact_state = EventTerm(func=gm_mdp.reset_tactile_contact_state, mode="reset", params=_contact_params())


@configclass
class HeterogeneousTactileRotationEnvCfg(ManagerBasedRLEnvCfg):
    r"""2048×1 默认、可由 CLI 覆盖为 2048×2=4096 的 infra-stage environment。"""

    is_finite_horizon: bool = True
    seed: int | None = 42
    scene: HeterogeneousTactileRotationSceneCfg = HeterogeneousTactileRotationSceneCfg(
        num_envs=len(HETEROGENEOUS_ASSET_ROWS),
        env_spacing=0.75,
        replicate_physics=False,
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
    observations: HeterogeneousObservationsCfg = HeterogeneousObservationsCfg()
    actions: HeterogeneousTactileRotationActionsCfg = HeterogeneousTactileRotationActionsCfg()
    commands: HeterogeneousTactileRotationCommandsCfg = HeterogeneousTactileRotationCommandsCfg()
    rewards: HeterogeneousRewardsCfg = HeterogeneousRewardsCfg()
    terminations: HeterogeneousTerminationsCfg = HeterogeneousTerminationsCfg()
    events: HeterogeneousEventsCfg = HeterogeneousEventsCfg()
    curriculum = None

    def __post_init__(self) -> None:
        r"""锁定 N000 的 120 Hz physics、20 Hz policy 与固定 120 s horizon。"""

        super().__post_init__()  # pyright: ignore[reportAttributeAccessIssue]  # configclass injects base hook
        self.decimation = 6
        self.episode_length_s = 120.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (2.0, 2.0, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)


@configclass
class HeterogeneousN040HistoryTactileRotationEnvCfg(HeterogeneousTactileRotationEnvCfg):
    r"""冻结N040 policy adapter的History30 observation variant；MDP其余语义完全继承N000。"""

    observations: HeterogeneousN040HistoryObservationsCfg = HeterogeneousN040HistoryObservationsCfg()
    actions: HeterogeneousN040HistoryActionsCfg = HeterogeneousN040HistoryActionsCfg()


@configclass
class HeterogeneousN040HistoryLegacyCriticEvalEnvCfg(HeterogeneousN040HistoryTactileRotationEnvCfg):
    r"""固定评估旧103D central-critic checkpoint；actor与MDP和当前N040 task完全相同。"""

    observations: HeterogeneousN040HistoryLegacyCriticObservationsCfg = (
        HeterogeneousN040HistoryLegacyCriticObservationsCfg()
    )


@configclass
class HeterogeneousN040Failure100TactileRotationEnvCfg(HeterogeneousN040HistoryTactileRotationEnvCfg):
    r"""127D task-aware critic主线的failure=-100单变量稳定化variant。"""

    rewards: HeterogeneousFailure100RewardsCfg = HeterogeneousFailure100RewardsCfg()


__all__ = [
    "HeterogeneousN040HistoryLegacyCriticEvalEnvCfg",
    "HeterogeneousN040HistoryTactileRotationEnvCfg",
    "HeterogeneousN040Failure100TactileRotationEnvCfg",
    "HeterogeneousTactileRotationEnvCfg",
]
