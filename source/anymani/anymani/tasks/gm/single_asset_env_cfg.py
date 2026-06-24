r"""TODO: 单资产手内操作环境任务文件。

本文件是 single-asset MDP probe 的落点，而不是后续同拓扑异构并行训练环境的
简化版。它只回答一个科研问题：在固定 pre-made topology template 母体资产上，
当前 `gm` 的 command / action / observation / reward / reset / termination 设计，
能否训练出可用的手内重定向 / 旋转策略。

目标资产：

```text
AnyMani/source/anymani/anymani/assets/generated/2026-06-10_11-30-08/
  single_palm_leap/right_t4_i4_m4_r4/hand.urdf
```

该资产是 `right_t4_i4_m4_r4` pre-made topology template 母体，不是 post-mutate
leaf sample。使用它的原因不是追求泛化，而是固定 morphology 变量：

1. 若单资产母体都训不起来，问题优先归因于 MDP / reset / reward / action / 资产物理，
   而不是 transformer、asset bank routing 或 morphology generalization；
2. 若单资产母体能训起来，再进入 same-topology post-mutate 并行训练时，失败才更可能
   指向异构资产分布、网络表达或 morphology conditioning；
3. 该阶段尽量暴露应在 MDP 层解决的问题，避免把基础问题带入后续 teacher transformer。

== 当前已确认的设计边界 ==

- 新写一个 single-asset env cfg，但复用 `tasks/gm/mdp` 已有 term；不复制一套新的
  command / reward / action 数学逻辑，避免同一科研语义出现两个实现版本。
- 观测先按当前 MDP scaffold 走：raw joint state、processed last action、soft limits、
  fingertip contact binary、`[axis_h, error_so3_h]` command。暂不引入
  `distill/models` 的 PALM / JOINT / TIP geometry tokenizer，也不在这里实现 tip BPS。
- reset 第一轮采用“一事一议”的 split event：hand joint reset、object pose reset
  直接复用 IsaacLab 官方项，AnyMani 只额外记录 object reset anchor。默认初态来自
  标定台导出的 pre-grasp / contact basin；若失败再单独排查 reset 扰动与接触盆地。
- command 难度第一轮主动收窄到 fixed `{h}` z 轴 + episode 目标，贴近 LEAP 官方
  z-axis 成功基线；random-axis / subgoal 留给 single-asset 跑通后的下一轮消融。
- 训练策略第一轮使用 MLP PPO，网络复杂度不参与本文件；训练入口放在
  `anymani.distill.train_mlp_single_asset`。

== 推荐的最小验收信号 ==

该环境 cfg 完成后，不应直接用长期训练曲线判断成败，而应先看以下信号：

1. random-action smoke：env 可构造、reset、step，obs/action/reward/done shape 闭合；
2. short MLP rollout：1–5 个 epoch 内 reward、value loss、policy loss 均为 finite，无 NaN；
3. reset 统计：`object_out_of_hand` 不应在绝大多数 env reset 后立刻触发；若立刻掉落，
   优先排查 no-cache 初态而不是 reward；
4. command 统计：`orientation_error`、`goal_success_count`、`axis_progress` 能被日志读到；
5. contact 统计：fingertip / non-tip sensor 不应全零或全饱和，否则接触传感器或阈值错误。

== 失败后的排查优先级 ==

若第一轮 MLP 单资产训练不动，建议按如下顺序排查，而不是立刻改网络：

1. 资产加载与关节顺序：母体 URDF 的 joint order 是否与 action / obs 一致；
2. reset 分布：随机 no-cache 初态是否让物体经常直接掉落或完全无接触；
3. reward 数值：keypoint reward、axis progress、success bonus 的量级是否被 contact / action 项压过；
4. action scale：`scale=0.1` 是否过大导致高频接触抖动，或过小导致探索不足；
5. contact sensor：fingertip / non-tip sensor 名称与 filtered object contact 是否正确；
6. command 几何：`axis_h -> axis_e` 与 `error_so3_h` 是否符合 `{h}` 坐标系语义。

TOAGENT:
    本文件仍处于 design scaffold。实现阶段可以把本 docstring 中的 TODO 转成
    `@configclass` cfg，但不要删除上述科研边界、验收信号和排查顺序。它们是训练调优
    阶段判断曲线失败原因的上下文，不是普通注释噪声。
"""

from __future__ import annotations

import math

import isaaclab.envs.mdp as isaac_mdp
import isaaclab.sim as sim_utils
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

from anymani.assets.bank import HandBankCfg
from anymani.assets.bank.path_utils import resolve_bank_path

from . import mdp as gm_mdp
from .contact_sensors import build_contact_sensor_layout_from_hand_spawn, install_contact_sensors
from .hand_spawn import (
    DEFAULT_HAND_ANCHOR_POS_E,
    HandFrameCfg,
    HandJointInitCfg,
    HandSpawnAdapter,
    HandSpawnCfg,
    HandUrdfSpawnCfg,
)


def _single_asset_bundle_path() -> str:
    r"""返回 mother bundle 的绝对路径字符串。

    `HandBankCfg(selection_mode="explicit")` 在没有 `post_mutate_path` 时，只有绝对
    container path 才不会被错误解释为相对 sample id。因此这里在 cfg 声明处显式调用
    `resolve_bank_path(...)`，但不读取 URDF / YAML 内容。

    Returns:
        str: mother hand bundle 的绝对路径字符串。
    """

    return str(resolve_bank_path(GM_SINGLE_ASSET_PREMADE_TOPOLOGY_PATH))


GM_SINGLE_ASSET_PREMADE_TOPOLOGY_PATH = (
    "source/anymani/anymani/assets/generated/2026-06-10_11-30-08/single_palm_leap/right_t4_i4_m4_r4"
)
r"""单资产 MLP probe 绑定的 pre-made mother topology bundle 路径。"""

GM_SINGLE_ASSET_HAND_SPAWN_CFG = HandSpawnCfg(
    bank=HandBankCfg(
        source_mode="post_mutate",
        selection_mode="explicit",
        containers=(_single_asset_bundle_path(),),
        validate_mesh_relpaths=True,
        parse_visual_rgba=True,
    ),
    frame=HandFrameCfg(
        semantic_R_ha=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        semantic_p_ha=(0.0, 0.0, 0.0),
        anchor_R_eh=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        anchor_p_eh=DEFAULT_HAND_ANCHOR_POS_E,
    ),
    joint_init=HandJointInitCfg(
        joint_pos={
            "thumb_j0": 0.71999997,
            "index_j0": -0.0,
            "middle_j0": 0.0,
            "ring_j0": 0.11,
            "thumb_j1": 1.56999993,
            "index_j1": -0.52999997,
            "middle_j1": -0.12,
            "ring_j1": 0.44999999,
            "thumb_j2": 0.75999999,
            "index_j2": 1.23000002,
            "middle_j2": 1.13999999,
            "ring_j2": 1.29999995,
            "thumb_j3": 1.63,
            "index_j3": 0.94999999,
            "middle_j3": 0.91999996,
            "ring_j3": 0.66999996,
        }
    ),
    urdf=HandUrdfSpawnCfg(activate_contact_sensors=True),
    asset_routing="round_robin",
    restore_visual_materials=True,
    validate_same_schema=True,
)
r"""单资产 hand spawn 配置。

虽然这里只选择一个 mother asset，仍然通过 `HandSpawnAdapter` 和 `MultiAssetSpawnerCfg`
路径进入 IsaacLab。这样本 probe 同时核验正式 hand-spawn 入口，不引入绕过 sidecar /
contact layout 的临时 URDF loader。
"""

GM_SINGLE_ASSET_CONTACT_LAYOUT = build_contact_sensor_layout_from_hand_spawn(
    GM_SINGLE_ASSET_HAND_SPAWN_CFG,
    validate_all_assets=True,
)
"""从 mother asset sidecar 推导出的 tip / non-tip contact sensor layout。"""


def build_single_asset_hand_articulation_cfg(hand_spawn_cfg: HandSpawnCfg, *, prim_path: str) -> ArticulationCfg:
    r"""将单资产 hand spawn cfg lower 成 `scene.robot` articulation cfg。

    Args:
        hand_spawn_cfg (HandSpawnCfg): 只包含 mother asset 的 hand spawn 配置。
        prim_path (str): IsaacLab scene 中 robot articulation 的 prim path。

    Returns:
        ArticulationCfg: 可赋给 `InteractiveSceneCfg.robot` 的 articulation 配置。
    """

    return HandSpawnAdapter(hand_spawn_cfg).build_articulation_cfg(prim_path=prim_path)


@configclass
class GmSingleAssetSceneCfg(InteractiveSceneCfg):
    r"""单资产 hand + object scene。

    scene 只固定 embodiment 为 pre-made mother asset；object、ground、light 与正式 GM
    in-hand MDP 保持一致，以便 MDP probe 和后续 multi-asset teacher 的差异尽量只来自
    morphology 分布，而不是环境装配差异。
    """

    robot: ArticulationCfg = build_single_asset_hand_articulation_cfg(
        GM_SINGLE_ASSET_HAND_SPAWN_CFG,
        prim_path="{ENV_REGEX_NS}/Robot",
    )
    """固定绑定 `right_t4_i4_m4_r4` pre-made mother asset。"""

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
            scale=(1.0, 1.0, 1.0),  # 与标定台 local cube 的“无额外缩放”语义对齐，先降低 contact-basin 迁移误差
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.02, 0.08, 0.56), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    """被操作物体；默认 GUI 初态对齐标定台导出的 contact basin，episode reset 仍由 events 接管。"""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.1)),
    )
    """地面只服务物理兜底和 GUI 参照，不是手内操作 reward 的组成部分。"""

    light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    """清天 HDRI 光照；便于 GUI 检查手、物体和 debug goal marker。"""

    def __post_init__(self):
        r"""安装由 mother sidecar 推导出的 per-link contact sensors。"""

        super().__post_init__()
        install_contact_sensors(self, GM_SINGLE_ASSET_CONTACT_LAYOUT)  # 每个 tip/non-tip link 一个 filtered sensor


@configclass
class GmSingleAssetCommandsCfg:
    r"""单资产 MDP probe 的 command 配置。

    第一轮降低 command 难度，使用 fixed `{h}` z 轴与 episode-level 目标，先复刻
    LEAP 官方 z-axis reorientation 中最稳定的任务语义。若单资产 vertical slice
    能跑通，再恢复 random axis / subgoal 作为正式 GM 难度。
    """

    goal_pose: gm_mdp.ReorientCommandCfg = gm_mdp.ReorientCommandCfg(
        asset_name="object",
        robot_asset_name="robot",
        axis_mode="fixed",
        axis_resample_mode="episode",
        fixed_axis_h=(0.0, 0.0, 1.0),
        semantic_R_ha=GM_SINGLE_ASSET_HAND_SPAWN_CFG.frame.semantic_R_ha,
    )


@configclass
class GmSingleAssetActionsCfg:
    r"""单资产动作配置：raw rad relative delta + soft-limit clamp。"""

    hand_joint_pos: gm_mdp.ClampedRelativeJointActionCfg = gm_mdp.ClampedRelativeJointActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.1,
        preserve_order=True,
    )


@configclass
class GmSingleAssetObservationsCfg:
    r"""单资产 actor / critic observation 配置。

    这里保持当前 MDP scaffold，不引入 `distill/models` 的 PALM / JOINT / TIP tokenizer。
    actor 看到 raw joint state、processed last action、hand-frame fingertip force、hand-frame object pose
    和 `[axis_h, error_so3_h]` command。这里是 teacher / single-asset MDP probe，
    允许 policy 读取 object pose privileged state，先降低学习难度；后续 student /
    deployment 再决定是否遮蔽或蒸馏该信息。
    """

    @configclass
    class PolicyCfg(ObsGroup):
        r"""Actor-facing flat observation group。"""

        joint_pos = ObsTerm(func=gm_mdp.joint_pos_raw, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=gm_mdp.joint_vel_raw, params={"asset_cfg": SceneEntityCfg("robot")})
        last_action = ObsTerm(func=gm_mdp.last_processed_action, params={"action_name": "hand_joint_pos"})
        # joint_limits = ObsTerm(func=gm_mdp.joint_soft_pos_limits, params={"asset_cfg": SceneEntityCfg("robot")})
        fingertip_force_h = ObsTerm(
            func=gm_mdp.fingertip_contact_force_h,
            params={
                "sensor_names": GM_SINGLE_ASSET_CONTACT_LAYOUT.fingertip_sensor_names,
                "robot_cfg": SceneEntityCfg("robot"),
                "semantic_R_ha": GM_SINGLE_ASSET_HAND_SPAWN_CFG.frame.semantic_R_ha,
            },
        )
        command = ObsTerm(func=gm_mdp.reorient_command, params={"command_name": "goal_pose"})
        object_pos_h = ObsTerm(
            func=gm_mdp.object_pos_h,
            params={
                "object_cfg": SceneEntityCfg("object"),
                "robot_cfg": SceneEntityCfg("robot"),
                "semantic_R_ha": GM_SINGLE_ASSET_HAND_SPAWN_CFG.frame.semantic_R_ha,
            },
        )
        object_rot6d_h = ObsTerm(
            func=gm_mdp.object_rot6d_h,
            params={
                "object_cfg": SceneEntityCfg("object"),
                "robot_cfg": SceneEntityCfg("robot"),
                "semantic_R_ha": GM_SINGLE_ASSET_HAND_SPAWN_CFG.frame.semantic_R_ha,
            },
        )

        def __post_init__(self):
            r"""拼接 actor obs，并保留 IsaacLab observation corruption 开关。"""

            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        r"""Critic-facing privileged observation group。"""

    policy: ObsGroup = PolicyCfg(history_length=1)
    critic: ObsGroup = CriticCfg(history_length=1)


@configclass
class GmSingleAssetRewardsCfg:
    r"""单资产 MDP probe 的 reward 配置。

    Reward term 完全复用 `gm_mdp` 现有实现，保证该阶段核验的是同一套 MDP 数学逻辑。
    """

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
            "sensor_names": GM_SINGLE_ASSET_CONTACT_LAYOUT.fingertip_sensor_names,
            "min_contacts": 2,
            "force_threshold": 0.2,
            "lambda_floor": 0.05,
        },
    )
    bad_non_tip_contact = RewTerm(
        func=gm_mdp.bad_non_tip_contact,
        weight=-0.2,
        params={
            "sensor_names": GM_SINGLE_ASSET_CONTACT_LAYOUT.non_tip_sensor_names,
            "force_threshold": 0.2,
            "lambda_floor": 0.0,
        },
    )
    action_l2 = RewTerm(func=gm_mdp.action_l2_curriculum, weight=-1.0e-4, params={"lambda_floor": 0.0})
    action_rate_l2 = RewTerm(func=gm_mdp.action_rate_l2_curriculum, weight=-1.0e-2, params={"lambda_floor": 0.0})


@configclass
class GmSingleAssetEventsCfg:
    r"""单资产 reset / event 配置。

    采用“一事一议”的 event 组合：官方项负责写 hand/object 物理状态，AnyMani 项
    只记录 object reset anchor，供 `object_out_of_hand` 使用。初期不扰动 hand joint
    与 object position，先精确复现标定台导出的 pre-grasp / contact basin；object
    yaw 允许 $[-\pi,\pi]$ 全角度随机，避免策略只适配单一 cube 初始朝向。
    """

    apply_structural_collision_filter = EventTerm(
        func=gm_mdp.apply_generated_structural_collision_filter,
        mode="prestartup",
        params={
            "robot_prim_path": "{ENV_REGEX_NS}/Robot",
            "palm_link_name": GM_SINGLE_ASSET_CONTACT_LAYOUT.palm_link_name,
            "finger_link_chains": GM_SINGLE_ASSET_CONTACT_LAYOUT.finger_link_chains,
            "filter_palm_finger": True,
            "filter_same_finger": True,
        },
    )
    """PhysX 初始化前写入 generated structural collision groups：finger-palm 与 same-finger 不碰，finger-finger 保留。"""

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
class GmSingleAssetTerminationsCfg:
    r"""单资产 termination 配置：timeout + object out of hand。"""

    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)
    object_falling = DoneTerm(func=gm_mdp.object_out_of_hand, params={"fall_dist": 0.12})


@configclass
class GmSingleAssetCurriculumCfg:
    r"""按 `goal_success_count` 释放 contact / regularization reward 的 curriculum。"""

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
class GmSingleAssetEnvCfg(ManagerBasedRLEnvCfg):
    r"""单资产 mother hand 的 MDP probe 环境。"""

    scene: GmSingleAssetSceneCfg = GmSingleAssetSceneCfg(
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

    observations: GmSingleAssetObservationsCfg = GmSingleAssetObservationsCfg()
    actions: GmSingleAssetActionsCfg = GmSingleAssetActionsCfg()
    commands: GmSingleAssetCommandsCfg = GmSingleAssetCommandsCfg()
    rewards: GmSingleAssetRewardsCfg = GmSingleAssetRewardsCfg()
    terminations: GmSingleAssetTerminationsCfg = GmSingleAssetTerminationsCfg()
    events: GmSingleAssetEventsCfg = GmSingleAssetEventsCfg()
    curriculum: GmSingleAssetCurriculumCfg = GmSingleAssetCurriculumCfg()

    def __post_init__(self):
        r"""设置单资产 MLP probe 的仿真时序与 viewer 默认视角。"""

        super().__post_init__()
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (2.0, 2.0, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)


@configclass
class GmSingleAssetEnvCfg_PLAY(GmSingleAssetEnvCfg):
    r"""单资产 GUI / command marker 检查环境。"""

    commands: GmSingleAssetCommandsCfg = GmSingleAssetCommandsCfg(
        goal_pose=gm_mdp.ReorientCommandCfg(
            asset_name="object",
            robot_asset_name="robot",
            axis_mode="fixed",
            axis_resample_mode="episode",
            debug_vis=True,
            fixed_axis_h=(0.0, 0.0, 1.0),
            semantic_R_ha=GM_SINGLE_ASSET_HAND_SPAWN_CFG.frame.semantic_R_ha,
        )
    )

    def __post_init__(self):
        r"""缩小 GUI env 数并关闭 actor obs corruption。"""

        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
        self.terminations.time_out = None


__all__ = [
    "GM_SINGLE_ASSET_CONTACT_LAYOUT",
    "GM_SINGLE_ASSET_HAND_SPAWN_CFG",
    "GM_SINGLE_ASSET_PREMADE_TOPOLOGY_PATH",
    "GmSingleAssetActionsCfg",
    "GmSingleAssetCommandsCfg",
    "GmSingleAssetCurriculumCfg",
    "GmSingleAssetEnvCfg",
    "GmSingleAssetEnvCfg_PLAY",
    "GmSingleAssetEventsCfg",
    "GmSingleAssetObservationsCfg",
    "GmSingleAssetRewardsCfg",
    "GmSingleAssetSceneCfg",
    "GmSingleAssetTerminationsCfg",
    "build_single_asset_hand_articulation_cfg",
]
