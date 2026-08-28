r"""ManagerBasedRLEnv scaffold for generalized in-hand manipulation.

本文件是 `gm` 任务线的环境装配面。它不继承旧 `tasks/inhand/config/<hand>/`
的 per-hand 配置目录，因为本研究主线不是“给 LEAP / Allegro 各写一套环境”，
而是“在 generated same-topology hand assets 上训练层次通才策略”。

核心边界：

- `assets` 负责生产 `hand.urdf` / `hand.yaml`；
- `tasks/gm` 负责定义 object manipulation MDP；
- `distill` 负责选择 asset bank、组织训练、保存 manifest、设计网络。

当前只是 distributed prompt + interface scaffold。不要把这里误读成已经可训练。
"""

from __future__ import annotations

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
from anymani.robots.hand_spawn import (
    DEFAULT_HAND_ANCHOR_POS_E,
    HandFrameCfg,
    HandSpawnAdapter,
    HandSpawnCfg,
    HandUrdfSpawnCfg,
)

from . import mdp as gm_mdp
from .contact_sensors import build_contact_sensor_layout_from_hand_spawn, install_contact_sensors

DEFAULT_OBJECT_USD = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"

GM_CLEAR_SKY_TEXTURE_FILE = (
    f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"
)
r"""GM GUI / smoke 默认使用的 clear-sky HDRI。

该视觉 preset 来自 `AnyMani-GM-Heterogeneous-Test-v0` 的可视检查路径。它不改变物理
任务语义，只让 `AnyMani-GM-InHand-*` 系列在 GUI 中共享同一套可读性更好的天空环境光。
"""

GM_CLEAR_SKY_LIGHT_INTENSITY = 750.0
"""clear-sky HDRI dome light 强度；与 heterogeneous GUI smoke 保持一致。"""

GM_DEFAULT_HAND_BANK_PATH = (
    "/home/hac/isaac/AnyMani/source/anymani/anymani/assets/generated/2026-08-16_14-55-19/"
    "single_palm_allegro/right_t4_i4_m4_r4"
)
r"""GM in-hand 默认使用的 same-topology post-mutate run。"""

GM_DEFAULT_HAND_SAMPLE_COUNT = 16
"""默认抽样 hand asset 数；包含 source topology 母体作为普通候选，可按 smoke/训练阶段调节。"""

GM_DEFAULT_HAND_SAMPLE_SEED = 42
"""默认资产抽样种子；与训练 seed 分离，只控制 hand bank 选择。"""

GM_DEFAULT_ENVS_PER_HAND = 32
"""默认每个 hand asset 分配的 env 数；当前 preset 为 32。"""

GM_DEFAULT_NUM_ENVS = GM_DEFAULT_HAND_SAMPLE_COUNT * GM_DEFAULT_ENVS_PER_HAND
"""默认总并行环境数，始终由 hand sample count 与 env-per-hand routing 相乘得到。"""

GM_DEFAULT_OBJECT_INIT_OFFSET_H = (0.0, 0.055, 0.06)
r"""默认 object root 相对 hand semantic frame `{h}` 的初始偏置，单位 m。

当前 generated hand 的 palm box 约覆盖 $y^h\in[0,0.08]$，四指从 palm 向
$+y^h$ 方向展开；因此 object 初态不能沿用旧 LEAP cfg 的 $y=-0.10$。这里把
DexCube root 放在掌心到指根之间，并让 $z^h=0.06$ 约等于 palm half-height
加 cube half-height 后的轻微离手高度，reset 后由重力落到手上。
"""

GM_DEFAULT_OBJECT_INIT_POS_E = (0.0, 0.055, 0.56)
r"""默认 object root 在 env frame `{e}` 中的位置，单位 m。

当前 $R_{eh}^{anchor}=I$ 且 $p_{eh}^{anchor}=(0,0,0.5)$，所以
$p^e_o=p^e_h+p^h_o=(0,0.055,0.56)$。若后续启用 episode 级 hand orientation
reset，应把该常量升级为 reset-time 的 $p^e_o=p^e_h+R_{eh}p^h_o$ 计算。
"""

DEFAULT_GM_HAND_SPAWN_CFG = HandSpawnCfg(
    bank=HandBankCfg(
        source_mode="mixed",
        selection_mode="explicit",
        containers=(GM_DEFAULT_HAND_BANK_PATH,),
        validate_mesh_relpaths=True,
        parse_visual_rgba=True,
    ),
    frame=HandFrameCfg(
        semantic_R_ha=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        semantic_p_ha=(0.0, 0.0, 0.0),
        anchor_R_eh=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        anchor_p_eh=DEFAULT_HAND_ANCHOR_POS_E,
    ),
    urdf=HandUrdfSpawnCfg(activate_contact_sensors=True),
    asset_routing="round_robin",
    restore_visual_materials=True,
    validate_same_schema=True,
)
r"""GM in-hand 默认 hand spawn 配置：当前默认选择一组 same-topology generated hands。

`activate_contact_sensors=True` 是 MDP contract 的一部分：本环境的 policy obs、critic
obs、good-contact reward 与 bad-contact penalty 都读取 scene 中显式声明的
`ContactSensorCfg`。若 URDF importer 不为 hand links 打开 contact report，环境能 spawn
但接触项会在第一步 observation / reward 读取时失效。
"""

GM_DEFAULT_CONTACT_LAYOUT = build_contact_sensor_layout_from_hand_spawn(
    DEFAULT_GM_HAND_SPAWN_CFG,
    validate_all_assets=False,
)
r"""默认 contact sensor layout，由第一个 selected hand sidecar 自动推导。

当前 same-topology training slice 已由 `HandSpawnCfg.validate_same_schema=True` 约束为
同一 articulation schema，因此默认只读取第一个 selected asset 的 `hand_cfg`。若后续调试
跨 topology bank，可在 helper 层打开 `validate_all_assets=True` 做全量 sidecar 对照。
"""


def build_gm_hand_articulation_cfg(hand_spawn_cfg: HandSpawnCfg, *, prim_path: str) -> ArticulationCfg:
    r"""将 GM hand spawn cfg lower 成 `scene.robot` articulation cfg。"""

    return HandSpawnAdapter(hand_spawn_cfg).build_articulation_cfg(prim_path=prim_path)


@configclass
class GmInHandSceneCfg(InteractiveSceneCfg):
    r"""Scene scaffold for object-in-hand manipulation with a generated hand.

    `robot` 故意是 `MISSING`。它应由上游选择的单个 generated hand asset
    注入，而不是在环境基类里硬编码 `LEAP_HAND_CFG`。

    这样做的科研含义是：任务环境固定“操作物体”的 MDP，embodiment 由
    asset binding 决定。same-topology 训练时，一段训练配置应绑定一组动作
    schema 一致的 assets；跨拓扑统一策略则留给 `distill/models` 的 mask/token
    表达。
    """

    robot: ArticulationCfg = build_gm_hand_articulation_cfg(
        DEFAULT_GM_HAND_SPAWN_CFG,
        prim_path="{ENV_REGEX_NS}/Robot",
    )
    r"""默认绑定当前配置选中的 same-topology generated hands；训练配置可覆盖该字段。"""

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=DEFAULT_OBJECT_USD,
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
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=GM_DEFAULT_OBJECT_INIT_POS_E,
            rot=(1.0, 0.0, 0.0, 0.0),
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
            intensity=GM_CLEAR_SKY_LIGHT_INTENSITY,
            texture_file=GM_CLEAR_SKY_TEXTURE_FILE,
        ),
    )

    def __post_init__(self):
        r"""Install sidecar-derived per-link contact sensors on the scene instance.

        `InteractiveScene` 读取 scene cfg 实例属性来发现 sensors；因此这里动态挂载
        `contact_<link>` 字段，而不是在 class body 中写死当前四指 topology。每个 sensor
        仍然是一条 link 对 object 的 filtered contact，保持 reward/obs 的物理语义。
        """

        super().__post_init__()
        install_contact_sensors(self, GM_DEFAULT_CONTACT_LAYOUT)  # per-link ContactSensorCfg，object-only filter


@configclass
class GmCommandsCfg:
    r"""Command scaffold for in-hand object reorientation.

    DONE:
        当前已切到 `gm_mdp.ReorientCommandCfg`，不再借用旧 `tasks/inhand` 的
        pose command 名字或 7D pose tensor 语义。policy-facing command 为
        `command_output` 指定的 policy-facing command；reward / termination /
        curriculum 读取 command term 内部 buffer，例如 `goal_quat_w`、`axis_e`、
        `goal_success_count`。

    NOTE:
        `theta_range` 默认 `[π/6, π/2]`，下限大于 success threshold，避免刚
        采样就成功。最终绕 `{h}` z 轴连续旋转测试时，可把 `axis_mode="fixed"`
        且 `fixed_axis_h=(0,0,1)`。
    """

    goal_pose: gm_mdp.ReorientCommandCfg = gm_mdp.ReorientCommandCfg(
        asset_name="object",
        robot_asset_name="robot",
        axis_mode="random",
        axis_resample_mode="subgoal",
        semantic_R_ha=DEFAULT_GM_HAND_SPAWN_CFG.frame.semantic_R_ha,
    )


@configclass
class GmActionsCfg:
    r"""Action scaffold for same-topology generated hands.

    NOTE(方案 C，已实现): 使用 `ClampedRelativeJointPositionAction`——
    raw relative delta (rad) + soft limits clamp，无 EMA。
    详见 `gm/mdp/actions/clamped_relative_action.py` 的设计上下文。

    动作空间：
    $$
    \Delta_t = a_t^{\text{raw}} \cdot s \quad (\text{rad}),\ s = 0.1\ (\text{preset})
    $$
    $$
    q_{t+1}^{\text{cmd}} = \text{clamp}(q_t + \Delta_t,\ q^{\min},\ q^{\max}) \quad (\text{rad})
    $$

    与 obs 侧一致性：
        - state obs 用 raw rad $q_i$，动作用 raw rad $\Delta_i$，同量纲。
        - last_action 应喂 $a_{t-1}^{\text{proc}}$（rad delta），
          不是 `isaac_mdp.last_action` 返回的 `raw_actions`（pre-scale），
          见 `mdp/observations/observations_state.py` state obs 段的对应 TODO。

    preset:
        $s = 0.1$：NN 输出 $\in[-1,1]$ 时每步增量 $\le 0.1$ rad。
        约 4× IsaacLab inhand 的每步有效增量（$\approx 0.026$ rad），
        无 EMA 拖慢，后续根据训练收敛速度和动作平滑度调参。

    TODO: 当前 `joint_names=[".*"]` 匹配所有关节，依赖 articulation 的 joint order
    与 same-topology contract 一致。若后续发现不匹配，需改为显式 per-topology 绑定。
    """

    hand_joint_pos: gm_mdp.ClampedRelativeJointActionCfg = gm_mdp.ClampedRelativeJointActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.1,
        preserve_order=True,
    )


@configclass
class GmObservationsCfg:
    r"""Policy / critic observation scaffold.

    初期 teacher RL 使用 flat observation 进入 `distill/rl` 的 Transformer adapter：
    关节位置、关节速度、last action、joint limits 和 reorient command 会被 adapter
    重新组织为轻量 token，而不是在 `tasks/gm` 中写网络结构。
    """

    @configclass
    class PolicyCfg(ObsGroup):
        r"""Deployable actor observation group scaffold.

        DONE(state obs 路线 B): 已出清 IsaacLab 默认的
        `joint_pos_limit_normalized` 与 `isaac_mdp.last_action` 占位。当前 actor
        侧使用 `gm_mdp` 的 raw-rad state obs：
        $$
        [q_i,\ \dot q_i,\ \Delta a_{t-1},\ q_i^{\min},\ q_i^{\max}],
        $$
        其中 $\Delta a_{t-1}$ 来自动作项 `processed_actions`，单位 rad，和
        `ClampedRelativeJointPositionAction` 的动作语义一致。teacher actor 还读取
        hand-frame object pose $(p_o^h, R_{ho})$，先让专家策略拥有充分物体状态。

        NOTE: 关节限位 $q_i^{\min}, q_i^{\max}$ 作为静态形态量单独提供，
        当前 `history_length=1` 不会重复堆叠。若后续给 dynamic state 开启
        $H>1$ history，应把 limits 拆到不参与 history 的 geometry/static group，
        不能让静态 morphology 被时间窗口复制 $H$ 次。
        """

        joint_pos = ObsTerm(func=gm_mdp.joint_pos_raw, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=gm_mdp.joint_vel_raw, params={"asset_cfg": SceneEntityCfg("robot")})
        last_action = ObsTerm(func=gm_mdp.last_processed_action, params={"action_name": "hand_joint_pos"})
        joint_limits = ObsTerm(func=gm_mdp.joint_soft_pos_limits, params={"asset_cfg": SceneEntityCfg("robot")})
        fingertip_contact = ObsTerm(
            func=gm_mdp.fingertip_contact_binary,
            params={"sensor_names": GM_DEFAULT_CONTACT_LAYOUT.fingertip_sensor_names, "force_threshold": 0.2},
        )
        object_pos = ObsTerm(
            func=gm_mdp.object_pos,
            params={
                "object_cfg": SceneEntityCfg("object"),
                "robot_cfg": SceneEntityCfg("robot"),
                "semantic_R_ha": DEFAULT_GM_HAND_SPAWN_CFG.frame.semantic_R_ha,
                "semantic_p_ha": DEFAULT_GM_HAND_SPAWN_CFG.frame.semantic_p_ha,
                "frame": "h",
                "reference": "hand",
            },
        )
        object_orientation = ObsTerm(
            func=gm_mdp.object_orientation,
            params={
                "object_cfg": SceneEntityCfg("object"),
                "robot_cfg": SceneEntityCfg("robot"),
                "semantic_R_ha": DEFAULT_GM_HAND_SPAWN_CFG.frame.semantic_R_ha,
                "frame": "h",
                "representation": "rot6d",
            },
        )
        command = ObsTerm(func=gm_mdp.reorient_command, params={"command_name": "goal_pose"})

        def __post_init__(self):
            r"""Configure actor observation concatenation semantics."""

            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        r"""Privileged critic observation group scaffold.

        critic 继承 policy 的 hand-frame object pose；额外读取 hand-frame fingertip force。
        """

        fingertip_force = ObsTerm(
            func=gm_mdp.fingertip_contact_force,
            params={
                "sensor_names": GM_DEFAULT_CONTACT_LAYOUT.fingertip_sensor_names,
                "robot_cfg": SceneEntityCfg("robot"),
                "semantic_R_ha": DEFAULT_GM_HAND_SPAWN_CFG.frame.semantic_R_ha,
                "frame": "h",
            },
        )

    policy: ObsGroup = PolicyCfg(history_length=1)
    critic: ObsGroup = CriticCfg(history_length=1)


@configclass
class GmRewardsCfg:
    r"""Reward scaffold for in-hand manipulation.

    DONE(奖励分组已落脚手架):
        奖励按 AnyRotate 风格拆成 `r_reorient / r_axis_progress / r_regularize`。
        当前已接入 `ReorientCommand`，因此主任务项可以读取 `goal_quat_w` 与
        `axis_e`，不再使用全零 placeholder。

    NOTE:
        动作正则已切到 curriculum-gated wrapper。由于动作空间本身是
        `scale=0.1` 的 raw rad delta 且会 clamp 到 soft limits，第一版允许
        严格模仿 AnyRotate：action / action-rate 正则在 curriculum 释放前
        默认不参与优化。若实际训练早期动作仍抖，可单独给这些 term 设置
        `lambda_floor > 0`。
    """

    track_orientation = RewTerm(
        func=gm_mdp.keypoint_reorientation_reward,
        weight=1.0,
        params={"command_name": "goal_pose", "object_cfg": SceneEntityCfg("object")},
    )
    axis_progress = RewTerm(
        func=gm_mdp.AxisDeltaRotationReward,
        weight=0.25,
        params={"command_name": "goal_pose", "object_cfg": SceneEntityCfg("object"), "clip_value": 0.025},
    )
    success_bonus = RewTerm(
        func=gm_mdp.goal_success_bonus,
        weight=2.0,
        params={"command_name": "goal_pose", "object_cfg": SceneEntityCfg("object"), "success_mode": "so3"},
    )
    good_contact = RewTerm(
        func=gm_mdp.good_fingertip_contact,
        weight=0.5,
        params={
            "sensor_names": GM_DEFAULT_CONTACT_LAYOUT.fingertip_sensor_names,
            "min_contacts": 2,
            "force_threshold": 0.2,
            "lambda_floor": 0.05,
        },
    )
    bad_non_tip_contact = RewTerm(
        func=gm_mdp.bad_non_tip_contact,
        weight=-0.2,
        params={
            "sensor_names": GM_DEFAULT_CONTACT_LAYOUT.non_tip_sensor_names,
            "force_threshold": 0.2,
            "lambda_floor": 0.0,
        },
    )
    action_l2 = RewTerm(func=gm_mdp.action_l2_curriculum, weight=-1.0e-4, params={"lambda_floor": 0.0})
    action_rate_l2 = RewTerm(func=gm_mdp.action_rate_l2_curriculum, weight=-1.0e-2, params={"lambda_floor": 0.0})


@configclass
class GmEventsCfg:
    r"""Domain randomization and reset scaffold。

    DONE(拆分 reset 语义):
        第一版 runnable slice 不再使用聚合式 wrapper，而是把 reset 拆成独立
        `EventTerm`：hand joint state 由 IsaacLab 官方 `reset_joints_by_offset`
        写入，object root pose 由 IsaacLab 官方 `reset_root_state_uniform`
        写入，AnyMani 只额外记录 object reset anchor。

    DR 阶段：
        - object scale 是 startup / usd-time 离散 bucket，不是 episode reset 噪声；
        - object mass / CoM / friction、robot link material / mass / CoM、actuator
          stiffness / damping、joint friction / armature 默认 startup 采样；
        - joint limit DR、collider offset DR、fixed tendon DR、interval 外力暂缓。

    NOTE:
        这里保留原先 runnable slice 的轻扰动幅度：hand joint 约 $\pm0.05$ rad，
        object 平移厘米级、姿态小角度扰动。后续若单资产标定 basin 更稳定，可继续
        把这些扰动收窄或分阶段释放。
    """

    reset_robot_joints = EventTerm(
        func=isaac_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reset_object = EventTerm(
        func=isaac_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.01, 0.01),
                "y": (-0.01, 0.01),
                "z": (-0.005, 0.005),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.2, 0.2),
            },
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
class GmTerminationsCfg:
    r"""Termination scaffold for object-in-hand episodes.

    DONE(第一版边界):
        - `time_out` 使用 IsaacLab 内置项，不额外包装。
        - `object_falling` 使用 `object_out_of_hand`：object root 相对 reset/default
          anchor 的 3D L2 距离超过 `0.12m` 即 reset。

    NOTE:
        不做 max success termination；不做 axis deviation；不做 joint-limit / 卡死
        termination。卡死相持先由较短 episode timeout 兜底，避免误杀能自行恢复的
        finger-gaiting 状态。
    """

    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)
    object_falling = DoneTerm(func=gm_mdp.object_out_of_hand, params={"fall_dist": 0.12})


@configclass
class GmCurriculumCfg:
    r"""Curriculum scaffold.

    DONE(Reward curriculum 落子):
        采用 AnyRotate 风格 adaptive reward curriculum。其物理/学习语义是：
        先让策略学会完成随机重定向子目标，再逐步释放 contact / stable / action
        正则项，避免一开始被吸到“稳定抓住但不旋转”的局部最优。

    进度指标：
        `goal_success_count`，即单个 episode 中完成了多少个重定向子目标。
        该指标应由 `ReorientCommand` 维护，而不是用 IsaacLab 官方 inhand 中
        命名含糊的 `consecutive_success`。

    preset:
        `g_min=1.0, g_max=2.0` 对齐 AnyRotate 的直觉：平均每 episode 约完成
        1 个子目标后开始释放，约完成 2 个子目标后完全释放。所有数值保持
        cfg 可调，后续可扫 `[0,2] / [1,2] / [1,4]` 等区间。
    """

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
class GmInHandEnvCfg(ManagerBasedRLEnvCfg):
    r"""Generalized in-hand manipulation environment config scaffold.

    该 cfg 是 task-level assembly surface。它表达“环境由哪些 MDP 组件组成”，并给出
    first runnable slice 的默认 generated-hand binding：固定 post-mutate run、
    默认 hand sample count、asset-sampling seed 与 env-per-hand routing。

    DONE:
        1. `scene.robot` 已通过 `HandSpawnAdapter` 绑定当前默认选择的 same-topology generated hands；
        2. action joint order 由 `preserve_order=True` 与 same-topology sidecar schema 共同约束；
        3. command / reward 已接入 `ReorientCommand`、keypoint orientation reward、axis progress
           与 goal-success curriculum；
        4. reset 已拆成 hand joint / object pose / object anchor 三个独立 event，
           便于逐项调试初始接触盆地与扰动分布。

    TODO:
        仍需用 Isaac Lab headless random-agent smoke 验证真实 articulation loading、contact sensor
        report、reset/step 张量输出与 `rl_games` rollout。该验证依赖仿真运行，不伪装成纯单测完成。
    """

    scene: GmInHandSceneCfg = GmInHandSceneCfg(num_envs=GM_DEFAULT_NUM_ENVS, env_spacing=0.75, replicate_physics=False)
    viewer: ViewerCfg = ViewerCfg()
    sim: SimulationCfg = SimulationCfg(
        physics_material=RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
        ),
    )

    observations: GmObservationsCfg = GmObservationsCfg()
    actions: GmActionsCfg = GmActionsCfg()
    commands: GmCommandsCfg = GmCommandsCfg()
    rewards: GmRewardsCfg = GmRewardsCfg()
    terminations: GmTerminationsCfg = GmTerminationsCfg()
    events: GmEventsCfg = GmEventsCfg()
    curriculum: GmCurriculumCfg = GmCurriculumCfg()

    def __post_init__(self):
        r"""Set simulation timing defaults for high-throughput RL."""

        super().__post_init__()
        self.decimation = 4
        self.episode_length_s = 30.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (2.0, 2.0, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)  # 对准 hand/object anchor 高度，让天空-地板-操作区同时进入视野


@configclass
class GmInHandEnvCfg_PLAY(GmInHandEnvCfg):
    r"""Small-scene variant for visual review and smoke tests."""

    commands: GmCommandsCfg = GmCommandsCfg(
        goal_pose=gm_mdp.ReorientCommandCfg(
            asset_name="object",
            robot_asset_name="robot",
            axis_mode="fixed",
            axis_resample_mode="episode",
            debug_vis=True,
            fixed_axis_h=(0.0, 0.0, 1.0),
            semantic_R_ha=DEFAULT_GM_HAND_SPAWN_CFG.frame.semantic_R_ha,
        )
    )

    def __post_init__(self):
        r"""Disable training-only noise for visual inspection."""

        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
        self.terminations.time_out = None


__all__ = [
    "DEFAULT_GM_HAND_SPAWN_CFG",
    "GM_CLEAR_SKY_LIGHT_INTENSITY",
    "GM_CLEAR_SKY_TEXTURE_FILE",
    "GM_DEFAULT_ENVS_PER_HAND",
    "GM_DEFAULT_HAND_BANK_PATH",
    "GM_DEFAULT_HAND_SAMPLE_COUNT",
    "GM_DEFAULT_HAND_SAMPLE_SEED",
    "GM_DEFAULT_CONTACT_LAYOUT",
    "GM_DEFAULT_NUM_ENVS",
    "GM_DEFAULT_OBJECT_INIT_OFFSET_H",
    "GM_DEFAULT_OBJECT_INIT_POS_E",
    "GmActionsCfg",
    "GmCommandsCfg",
    "GmCurriculumCfg",
    "GmEventsCfg",
    "GmInHandEnvCfg",
    "GmInHandEnvCfg_PLAY",
    "GmInHandSceneCfg",
    "GmObservationsCfg",
    "GmRewardsCfg",
    "GmTerminationsCfg",
]
