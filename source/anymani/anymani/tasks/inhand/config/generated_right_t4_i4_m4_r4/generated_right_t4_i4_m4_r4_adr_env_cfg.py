r"""Generated `right_t4_i4_m4_r4` official-ADR probe.

本文件把 N010 official-aligned LEAP ADR 任务沿资产链继续推进一步：

```text
official USD hand  ->  official raw URDF hand  ->  generated right_t4_i4_m4_r4 hand
```

核心科研约束是只替换 embodiment / reset seed / 结构性自碰过滤，而不改 MDP 数学语义：

1. actor observation 仍为 official 96D history：
   $$
   o_t^\pi=[\tilde q_t, q_t^{target}]_{t-2:t}\in\mathbb{R}^{96};
   $$
2. action 仍为 official target-buffer relative update：
   $$
   q_t^{target}=\operatorname{clip}\left(q_{t-1}^{target}+\frac{1}{24}a_t^{exec}\right);
   $$
3. reward / ADR / termination / command 继承 N010 official-aligned 语义；
4. generated hand root pose 使用 `HandSpawnCfg` 的 flat anchor：
   $p_{eh}^{anchor}=(0,0,0.5)$、$R_{eh}^{anchor}=I$；
5. reset joint / object contact basin 只从
   `tools/presets/generated_asset/right_t4_i4_m4_r4/latest.yaml` 读取。

当前 accepted preset 数值锚点：DexCube `scale=(1.2,1.2,1.2)`，object pose 为
`pos=(0.00578245, 0.08511957, 0.55879354)`、`rot=(1,0,0,0)`。这些数值不在
class body 中手写，而是通过 `GraspPreset` 从 YAML 读取，使 calibrator 导出的
`latest.yaml` 成为 reset 初态的唯一来源。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from anymani.assets.bank import HandBankCfg
from anymani.assets.bank.path_utils import resolve_bank_path
from anymani.robots.hand_spawn import (
    DEFAULT_HAND_ANCHOR_POS_E,
    HandFrameCfg,
    HandJointInitCfg,
    HandSpawnAdapter,
    HandSpawnCfg,
    HandUrdfSpawnCfg,
)
from anymani.tasks.gm.contact_sensors import build_contact_sensor_layout_from_hand_spawn
from anymani.tasks.gm.mdp.events import apply_generated_structural_collision_filter
from anymani.tasks.inhand import mdp as leap_mdp
from anymani.tasks.inhand.config.leaphand.leaphand_adr_env_cfg import (
    LeapHandOfficialADRCommandsCfg,
    LeapHandOfficialADRCurriculumCfg,
    LeapHandOfficialADREventCfg,
    LeapHandOfficialADRRewardsCfg,
    LeapHandOfficialADRTerminationsCfg,
    LeapHandTactileADREnvCfg,
)
from anymani.tasks.inhand.mdp import rewards as official_rewards
from anymani.tasks.inhand.inhand_env_cfg import INHAND_CLEAR_SKY_LIGHT_INTENSITY, INHAND_CLEAR_SKY_TEXTURE_FILE
from anymani.tools.grasp_preset import GraspPreset, asset_preset_path

GENERATED_RIGHT_T4_I4_M4_R4_BUNDLE_ID = (
    "source/anymani/anymani/assets/generated/2026-06-10_11-30-08/single_palm_leap/right_t4_i4_m4_r4"
)
r"""训练 probe 绑定的 generated hand bundle。

该路径与标定命令中的 `--hand-bundle` 一致，指向 `right_t4_i4_m4_r4` mother topology
bundle。`HandBankCfg(selection_mode="explicit")` 消费该路径后，`HandSpawnAdapter` 负责
把 selected generated URDF lower 成 IsaacLab `ArticulationCfg`。
"""

GENERATED_RIGHT_T4_I4_M4_R4_GRASP_PRESET_PATH = asset_preset_path("generated_asset", "right_t4_i4_m4_r4")
r"""当前 generated probe 的唯一 reset seed YAML。

文件位置固定为 `tools/presets/generated_asset/right_t4_i4_m4_r4/latest.yaml`。训练时不在
env cfg 中另写 joint dict 或 object pose，避免 calibrator 与训练 reset source 分叉。
"""

GENERATED_OFFICIAL_SLOT_JOINT_ORDER = (
    "index_j0",
    "thumb_j0",
    "middle_j0",
    "ring_j0",
    "index_j1",
    "thumb_j1",
    "middle_j1",
    "ring_j1",
    "index_j2",
    "thumb_j2",
    "middle_j2",
    "ring_j2",
    "index_j3",
    "thumb_j3",
    "middle_j3",
    "ring_j3",
)
r"""16D policy/action 槽位的 generated joint-name 排列。

该顺序保留 official LEAP 的“关节层级交织”槽位语义：先遍历四指的第 0 层关节，
再遍历第 1/2/3 层关节。这样 policy 看到的第 $i$ 维 action 仍对应 official slot，
只是内部 joint name 从 `a_*` 映射为 generated schema 的 `index/thumb/middle/ring_j*`。
"""


def _generated_bundle_path() -> str:
    r"""解析 generated bundle 为绝对路径字符串。

    `HandBankCfg.containers` 在显式选择本地 bundle 时使用字符串路径。这里集中解析路径，
    使 `GraspPreset.asset.hand_ref`、contract test 和 `HandSpawnCfg` 都指向同一个 bundle 语义。

    Returns:
        str: `right_t4_i4_m4_r4` bundle 的绝对路径。
    """

    return str(resolve_bank_path(GENERATED_RIGHT_T4_I4_M4_R4_BUNDLE_ID))  # 训练 hand bundle 绝对路径。


def _require_preset_object_source(preset: GraspPreset, expected_source: str) -> str:
    r"""校验 preset 中的 object branch 与当前 probe 的 DexCube 语义一致。

    `GraspPreset.from_yaml()` 主要解析 joint pose 与 object pose；object asset branch 需要在
    env cfg 中显式校验，避免误把 `local_cube` 标定结果带入 official-ADR probe。

    Args:
        preset (GraspPreset): 已解析的 generated contact-basin preset。
        expected_source (str): 当前 env 接受的 object source，固定为 `"dex_cube_usd"`。

    Returns:
        str: preset 中声明的 object source。

    Raises:
        ValueError: 当 preset 的 `asset.object_source` 不是 DexCube USD 时抛出。
    """

    object_source = str(preset.asset.get("object_source"))  # YAML 中的 object asset 分支标签。
    if object_source != expected_source:
        raise ValueError(
            f"Generated official-ADR probe expects object_source={expected_source!r}, "
            f"got {object_source!r} in {preset.path}."
        )
    return object_source


def _require_preset_object_scale(preset: GraspPreset) -> tuple[float, float, float]:
    r"""从 preset asset branch 读取 DexCube scale。

    当前 probe 使用 Isaac Nucleus DexCube，而不是 local cube。`object_scale` 因此是 USD spawn
    的显式尺度参数：
    $$
    s_o=(1.2,1.2,1.2).
    $$

    Args:
        preset (GraspPreset): 已解析的 generated contact-basin preset。

    Returns:
        tuple[float, float, float]: DexCube USD 的三轴 scale。

    Raises:
        ValueError: 当 `asset.object_scale` 缺失或不是 3 元数值序列时抛出。
    """

    value = preset.asset.get("object_scale")  # YAML `asset.object_scale`，应为三轴 scale。
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"Preset {preset.path} must provide asset.object_scale as a 3-vector for DexCube USD.")
    return tuple(float(item) for item in value)  # $(s_x,s_y,s_z)$，无量纲 USD scale。


def _require_joint_values_by_order(joint_pos_rad: Mapping[str, float], joint_order: Sequence[str]) -> tuple[float, ...]:
    r"""按 official slot order 组装 generated pre-grasp 向量。

    `OfficialADRTargetJointPositionAction` 的 `pregrasp_joint_pos` 是一个 16D 向量，必须与
    `joint_names` 完全同序，否则 pose-diff penalty 会惩罚错误关节：
    $$
    \lVert q_t^{target}-q^{pregrasp}\rVert_2^2.
    $$

    Args:
        joint_pos_rad (Mapping[str, float]): preset 中的 joint-name -> rad 关节角。
        joint_order (Sequence[str]): 16D action/obs slot 的 joint-name 顺序。

    Returns:
        tuple[float, ...]: 与 `joint_order` 同序的 pre-grasp 向量，单位 rad。

    Raises:
        ValueError: 当 preset 缺失任一 action slot joint 时抛出。
    """

    missing = tuple(joint_name for joint_name in joint_order if joint_name not in joint_pos_rad)
    if missing:
        raise ValueError(f"Generated preset is missing official-slot joint(s): {missing!r}.")
    return tuple(float(joint_pos_rad[joint_name]) for joint_name in joint_order)  # $q^{pregrasp}$，形状 [16]。


GENERATED_RIGHT_T4_I4_M4_R4_GRASP_PRESET = GraspPreset.from_yaml(
    GENERATED_RIGHT_T4_I4_M4_R4_GRASP_PRESET_PATH,
    expected_hand_source="generated_bundle",
    expected_hand_ref_contains="right_t4_i4_m4_r4",
)
r"""Calibrator 导出的 generated pre-grasp / object contact basin。"""

GENERATED_RIGHT_T4_I4_M4_R4_OBJECT_SOURCE = _require_preset_object_source(
    GENERATED_RIGHT_T4_I4_M4_R4_GRASP_PRESET,
    expected_source="dex_cube_usd",
)
r"""Object asset source，固定为 Isaac Nucleus DexCube USD。"""

GENERATED_RIGHT_T4_I4_M4_R4_OBJECT_SCALE = _require_preset_object_scale(
    GENERATED_RIGHT_T4_I4_M4_R4_GRASP_PRESET
)
r"""DexCube USD scale，从 preset `asset.object_scale` 读取，当前为 `(1.2,1.2,1.2)`。"""

GENERATED_RIGHT_T4_I4_M4_R4_PREGRASP_BY_NAME = dict(GENERATED_RIGHT_T4_I4_M4_R4_GRASP_PRESET.joint_pos_rad)
r"""Generated reset/pre-grasp joint dict，单位 rad，按 joint name 写入 articulation default state。"""

GENERATED_RIGHT_T4_I4_M4_R4_PREGRASP_VECTOR = _require_joint_values_by_order(
    GENERATED_RIGHT_T4_I4_M4_R4_PREGRASP_BY_NAME,
    GENERATED_OFFICIAL_SLOT_JOINT_ORDER,
)
r"""与 `GENERATED_OFFICIAL_SLOT_JOINT_ORDER` 同序的 16D pre-grasp 向量，单位 rad。"""

GENERATED_RIGHT_T4_I4_M4_R4_HAND_SPAWN_CFG = HandSpawnCfg(
    bank=HandBankCfg(
        source_mode="post_mutate",
        selection_mode="explicit",
        containers=(_generated_bundle_path(),),
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
        joint_pos=GENERATED_RIGHT_T4_I4_M4_R4_PREGRASP_BY_NAME,
        joint_vel={".*": 0.0},
    ),
    urdf=HandUrdfSpawnCfg(activate_contact_sensors=False),
    asset_routing="round_robin",
    restore_visual_materials=True,
    validate_same_schema=True,
)
r"""Generated hand spawn cfg。

    该 cfg 与 calibrator 的 frame anchor 保持一致：$T_{eh}^{anchor}$ 为 flat hand semantic pose，
    不套 official LEAP root quaternion。`restore_visual_materials=True` 只恢复 generated URDF 的
    debug 颜色，不改变动力学、关节、碰撞或 reward 语义。
"""

GENERATED_RIGHT_T4_I4_M4_R4_CONTACT_LAYOUT = build_contact_sensor_layout_from_hand_spawn(
    GENERATED_RIGHT_T4_I4_M4_R4_HAND_SPAWN_CFG,
    validate_all_assets=True,
)
r"""由 generated sidecar 推导的 link layout，仅用于结构性 collision filter。"""


def build_generated_right_t4_i4_m4_r4_hand_articulation_cfg(
    hand_spawn_cfg: HandSpawnCfg,
    *,
    prim_path: str,
) -> ArticulationCfg:
    r"""把 generated hand spawn cfg lower 成 IsaacLab articulation cfg。

    Args:
        hand_spawn_cfg (HandSpawnCfg): `right_t4_i4_m4_r4` generated hand spawn 声明。
        prim_path (str): scene 中 hand articulation 的 prim path，通常为 `"{ENV_REGEX_NS}/Robot"`。

    Returns:
        ArticulationCfg: 可赋给 `InteractiveSceneCfg.robot` 的 generated hand articulation。
    """

    return HandSpawnAdapter(hand_spawn_cfg).build_articulation_cfg(prim_path=prim_path)


@configclass
class GeneratedRightT4I4M4R4OfficialADRSceneCfg(InteractiveSceneCfg):
    r"""Generated hand + DexCube scene for official-ADR MDP。

    该 scene 的核心变量只有三项：

    - hand asset：`right_t4_i4_m4_r4` generated bundle；
    - object scale：preset 记录的 DexCube `scale=(1.2,1.2,1.2)`；
    - object reset basin：preset 记录的 env-frame pose。

    Ground / light / object rigid-body properties 继承 N010 official-aligned 语义，避免引入
    GM single-asset scene 中的 local-cube 或其他 MDP probe 变量。
    """

    robot: ArticulationCfg = build_generated_right_t4_i4_m4_r4_hand_articulation_cfg(
        GENERATED_RIGHT_T4_I4_M4_R4_HAND_SPAWN_CFG,
        prim_path="{ENV_REGEX_NS}/Robot",
    )
    r"""Generated hand articulation；root pose 由 `HandSpawnCfg.frame` 的 $T_{eh}^{anchor}$ 推导。"""

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
            scale=GENERATED_RIGHT_T4_I4_M4_R4_OBJECT_SCALE,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=GENERATED_RIGHT_T4_I4_M4_R4_GRASP_PRESET.object_pos_cfg,
            rot=GENERATED_RIGHT_T4_I4_M4_R4_GRASP_PRESET.object_rot_wxyz,
        ),
    )
    r"""DexCube object；init pose 是 calibrator 导出的 env-frame contact basin。"""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.1)),
    )
    r"""Ground plane；只作为物理兜底，不参与 reward 或 observation。"""

    light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=INHAND_CLEAR_SKY_LIGHT_INTENSITY,
            texture_file=INHAND_CLEAR_SKY_TEXTURE_FILE,
        ),
    )
    r"""Dome light；保持 N010 official-aligned scene 的可视化基线。"""


@configclass
class GeneratedRightT4I4M4R4OfficialADRPolicyObsCfg(ObsGroup):
    r"""Generated hand 的 official 96D actor observation。

    单帧仍为 32D：16D normalized joint position 加 16D current target buffer；IsaacLab
    observation history 再叠成 3 帧：
    $$
    3\times(16+16)=96.
    $$
    """

    frame = ObsTerm(
        func=leap_mdp.official_policy_frame,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(GENERATED_OFFICIAL_SLOT_JOINT_ORDER),
                preserve_order=True,
            ),
            "action_term_name": "hand_joint_pos",
        },
        history_length=3,
        flatten_history_dim=True,
    )

    def __post_init__(self) -> None:
        r"""关闭 obs corruption 并拼接唯一 observation term。"""

        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class GeneratedRightT4I4M4R4OfficialADRObservationsCfg:
    r"""Generated official-ADR observation group，仅暴露 policy 96D history。"""

    @configclass
    class PolicyCfg(GeneratedRightT4I4M4R4OfficialADRPolicyObsCfg):
        r"""Actor-facing observation group。"""

    policy: ObsGroup = PolicyCfg()


@configclass
class GeneratedRightT4I4M4R4OfficialADRActionsCfg:
    r"""Generated hand 的 official target-buffer relative action。"""

    hand_joint_pos = leap_mdp.OfficialADRTargetJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(GENERATED_OFFICIAL_SLOT_JOINT_ORDER),
        scale=1.0 / 24.0,
        preserve_order=True,
        use_zero_offset=True,
        max_latency=3,
        latency_rand=1,
        pregrasp_joint_pos=GENERATED_RIGHT_T4_I4_M4_R4_PREGRASP_VECTOR,
    )
    r"""16D action term；target-buffer 与 pregrasp vector 均按 official-slot generated order 排列。"""


@configclass
class GeneratedRightT4I4M4R4OfficialADREventCfg(LeapHandOfficialADREventCfg):
    r"""Generated hand official-ADR events。

    继承 N010 ADR reset / material / wrench / horizon 事件，只替换 robot-joint reset 的
    joint-name order，并新增 generated structural collision filter：finger-palm 与 same-finger
    link pairs 不碰，cross-finger collision 保留。
    """

    apply_structural_collision_filter = EventTerm(
        func=apply_generated_structural_collision_filter,
        mode="prestartup",
        params={
            "robot_prim_path": "{ENV_REGEX_NS}/Robot",
            "palm_link_name": GENERATED_RIGHT_T4_I4_M4_R4_CONTACT_LAYOUT.palm_link_name,
            "finger_link_chains": GENERATED_RIGHT_T4_I4_M4_R4_CONTACT_LAYOUT.finger_link_chains,
            "filter_palm_finger": True,
            "filter_same_finger": True,
        },
    )
    r"""PhysX startup 前写入 generated `FilteredPairsAPI` 结构过滤。"""

    reset_robot_joints = EventTerm(
        func=leap_mdp.reset_adr_robot_joints,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(GENERATED_OFFICIAL_SLOT_JOINT_ORDER),
                preserve_order=True,
            )
        },
    )
    r"""ADR joint reset；default joint pose 来自 generated preset，噪声 curriculum 继承 N010。"""


@configclass
class GeneratedRightT4I4M4R4OfficialADRNoDtRewardsCfg:
    r"""N031 no-dt reward-scale ablation。

    该配置只改变 `OfficialLeapReward` 的一个实验开关：

    $$
    \texttt{divide\_by\_step\_dt}: \texttt{True} \rightarrow \texttt{False}.
    $$

    因此相对 N030，scene / action / obs / command / termination / ADR 都保持一致。真正进入 PPO
    的 reward 将变为：

    $$
    \Delta t\cdot r_t^{official},
    $$

    因为 `RewardManager` 仍会自动乘上 `env.step_dt`。
    """

    official_reward = RewTerm(
        func=official_rewards.OfficialLeapReward,
        weight=1.0,
        params={
            "action_term_name": "hand_joint_pos",
            "command_name": "goal_pose",
            "object_cfg": SceneEntityCfg("object"),
            "dist_reward_scale": -10.0,
            "rot_reward_scale": 1.0,
            "rot_eps": 0.1,
            "action_penalty_scale": -0.0002,
            "pose_diff_penalty_scale": -0.3,
            "success_tolerance": 0.2,
            "position_success_threshold": 0.025,
            "reach_goal_bonus": 250.0,
            "fall_dist": 0.07,
            "fall_penalty": -10.0,
            "z_rotation_steps": 16,
            "divide_by_step_dt": False,
        },
    )


@configclass
class LeapHandADRGeneratedRightT4I4M4R4EnvCfg(LeapHandTactileADREnvCfg):
    r"""Generated `right_t4_i4_m4_r4` official-ADR training env。

    尽管基类名字仍含历史 `Tactile`，本 env 的 Gym id 不再含 `Tactile`，且 scene 中没有
    tactile sensors。继承该基类只是为了复用 N010 official-aligned ADR 的 sim timing、viewer、
    reward、command、termination 和 curriculum 组合。
    """

    scene: InteractiveSceneCfg = GeneratedRightT4I4M4R4OfficialADRSceneCfg(
        num_envs=4096,
        env_spacing=0.75,
        replicate_physics=False,
    )
    observations: GeneratedRightT4I4M4R4OfficialADRObservationsCfg = GeneratedRightT4I4M4R4OfficialADRObservationsCfg()
    actions: GeneratedRightT4I4M4R4OfficialADRActionsCfg = GeneratedRightT4I4M4R4OfficialADRActionsCfg()
    commands: LeapHandOfficialADRCommandsCfg = LeapHandOfficialADRCommandsCfg()
    rewards: LeapHandOfficialADRRewardsCfg = LeapHandOfficialADRRewardsCfg()
    terminations: LeapHandOfficialADRTerminationsCfg = LeapHandOfficialADRTerminationsCfg()
    events: GeneratedRightT4I4M4R4OfficialADREventCfg = GeneratedRightT4I4M4R4OfficialADREventCfg()
    curriculum: LeapHandOfficialADRCurriculumCfg = LeapHandOfficialADRCurriculumCfg()


@configclass
class LeapHandADRGeneratedRightT4I4M4R4EnvCfg_PLAY(LeapHandADRGeneratedRightT4I4M4R4EnvCfg):
    r"""Generated official-ADR play/debug env。"""

    def __post_init__(self) -> None:
        r"""降低 env 数并打开 continuous-rotation goal marker。"""

        super().__post_init__()
        self.scene.num_envs = 50
        self.commands.goal_pose.debug_vis = True


@configclass
class LeapHandADRGeneratedRightT4I4M4R4NoDtRewardEnvCfg(LeapHandADRGeneratedRightT4I4M4R4EnvCfg):
    r"""N031 generated official-ADR env：仅取消 combined reward 的 `dt` 对齐。"""

    rewards: GeneratedRightT4I4M4R4OfficialADRNoDtRewardsCfg = GeneratedRightT4I4M4R4OfficialADRNoDtRewardsCfg()


@configclass
class LeapHandADRGeneratedRightT4I4M4R4NoDtRewardEnvCfg_PLAY(LeapHandADRGeneratedRightT4I4M4R4NoDtRewardEnvCfg):
    r"""N031 no-dt reward play/debug env。"""

    def __post_init__(self) -> None:
        r"""降低 env 数并打开 goal marker，便于 N031 与 N030 replay 对比。"""

        super().__post_init__()
        self.scene.num_envs = 50
        self.commands.goal_pose.debug_vis = True


__all__ = [
    "GENERATED_OFFICIAL_SLOT_JOINT_ORDER",
    "GENERATED_RIGHT_T4_I4_M4_R4_BUNDLE_ID",
    "GENERATED_RIGHT_T4_I4_M4_R4_CONTACT_LAYOUT",
    "GENERATED_RIGHT_T4_I4_M4_R4_GRASP_PRESET",
    "GENERATED_RIGHT_T4_I4_M4_R4_GRASP_PRESET_PATH",
    "GENERATED_RIGHT_T4_I4_M4_R4_HAND_SPAWN_CFG",
    "GENERATED_RIGHT_T4_I4_M4_R4_OBJECT_SCALE",
    "GENERATED_RIGHT_T4_I4_M4_R4_OBJECT_SOURCE",
    "GENERATED_RIGHT_T4_I4_M4_R4_PREGRASP_VECTOR",
    "GeneratedRightT4I4M4R4OfficialADRActionsCfg",
    "GeneratedRightT4I4M4R4OfficialADRNoDtRewardsCfg",
    "GeneratedRightT4I4M4R4OfficialADREventCfg",
    "GeneratedRightT4I4M4R4OfficialADRObservationsCfg",
    "GeneratedRightT4I4M4R4OfficialADRSceneCfg",
    "LeapHandADRGeneratedRightT4I4M4R4EnvCfg",
    "LeapHandADRGeneratedRightT4I4M4R4EnvCfg_PLAY",
    "LeapHandADRGeneratedRightT4I4M4R4NoDtRewardEnvCfg",
    "LeapHandADRGeneratedRightT4I4M4R4NoDtRewardEnvCfg_PLAY",
    "build_generated_right_t4_i4_m4_r4_hand_articulation_cfg",
]
