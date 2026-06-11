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

from dataclasses import MISSING

import isaaclab.envs.mdp as isaac_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
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

from . import mdp as gm_mdp

DEFAULT_OBJECT_USD = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"


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

    robot: ArticulationCfg = MISSING

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
            pos=(0.0, -0.10, 0.56),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.1)),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class GmCommandsCfg:
    r"""Command scaffold for in-hand object reorientation.

    DONE:
        当前已切到 `gm_mdp.ReorientCommandCfg`，不再借用旧 `tasks/inhand` 的
        pose command 名字或 7D pose tensor 语义。policy-facing command 为
        `[axis_h, error_so3_h]`；reward / termination / curriculum 读取 command
        term 内部 buffer，例如 `goal_quat_w`、`axis_e`、`goal_success_count`。

    NOTE:
        `theta_range` 默认 `[π/6, π/2]`，下限大于 success threshold，避免刚
        采样就成功。最终绕 `{h}` z 轴连续旋转测试时，可把 `axis_mode="fixed"`
        且 `fixed_axis_h=(0,0,1)`。
    """

    goal_pose: object = MISSING


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
          见 `mdp/observations.py` state obs 段的对应 TODO。

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

    初期 teacher RL 可以保持 MLP 友好的状态观测：关节位置、关节速度、
    object pose、goal command、last action。后续 student / unified policy 的
    joint-centric token、mesh feature、mask / padding 不应塞进本 cfg，而应由
    `distill/models` 与训练 wrapper 明确接管。
    """

    @configclass
    class PolicyCfg(ObsGroup):
        r"""Deployable actor observation group scaffold.

        TODO(state obs 路线 B): 当前 `joint_pos` 暂用 IsaacLab 默认的
        `joint_pos_limit_normalized`（即 $q_i^{\text{norm}} \in [-1, 1]$），
        这与已对齐的设计决策相悖。正式实现时应替换为 `gm_mdp` 中基于
        **raw rad** 的关节观测项（$q_i$ + $\dot q_i$），理由见
        `mdp/observations.py` state obs 段的四条论证（跨 variant 语义不变性、
        post-mutate 只变 limit、$q_i^{\text{norm}}$ 还原 $q_i$ 需乘性算子、raw 尺度本就友好）。

        NOTE: 关节限位 $q_i^{\min}, q_i^{\max}$ 作为静态形态量单独提供，
        不进本 group 的时间历史（history_length），避免 $H > 1$ 时重复堆叠。
        """

        joint_pos = ObsTerm(func=isaac_mdp.joint_pos_limit_normalized, params={"asset_cfg": SceneEntityCfg("robot")})
        last_action = ObsTerm(func=isaac_mdp.last_action)

        def __post_init__(self):
            r"""Configure actor observation concatenation semantics."""

            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        r"""Privileged critic observation group scaffold."""

        object_pos = ObsTerm(func=isaac_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        object_quat = ObsTerm(
            func=isaac_mdp.root_quat_w,
            params={"asset_cfg": SceneEntityCfg("object"), "make_quat_unique": False},
        )

    policy: ObsGroup = PolicyCfg(history_length=1)
    critic: ObsGroup = CriticCfg(history_length=1)


@configclass
class GmRewardsCfg:
    r"""Reward scaffold for in-hand manipulation.

    DONE(奖励分组已落脚手架):
        奖励按 AnyRotate 风格拆成 `r_reorient / r_contact / r_stable /
        r_terminate`。当前 env cfg 仍保留 `reorientation_reward_placeholder`，
        因为真正的 `keypoint_reorientation_reward` / `AxisDeltaRotationReward`
        依赖 `ReorientCommand` 暴露 `goal_quat_w`、`axis_e`、
        `goal_success_count` 等 buffer / metric；这些 command contract 尚未正式
        实现。

    NOTE:
        动作正则已切到 curriculum-gated wrapper。由于动作空间本身是
        `scale=0.1` 的 raw rad delta 且会 clamp 到 soft limits，第一版允许
        严格模仿 AnyRotate：action / action-rate 正则在 curriculum 释放前
        默认不参与优化。若实际训练早期动作仍抖，可单独给这些 term 设置
        `lambda_floor > 0`。
    """

    track_orientation = RewTerm(func=gm_mdp.reorientation_reward_placeholder, weight=1.0)
    action_l2 = RewTerm(func=gm_mdp.action_l2_curriculum, weight=-1.0e-4, params={"lambda_floor": 0.0})
    action_rate_l2 = RewTerm(func=gm_mdp.action_rate_l2_curriculum, weight=-1.0e-2, params={"lambda_floor": 0.0})


@configclass
class GmEventsCfg:
    r"""Domain randomization and reset scaffold.

    DONE(语义主次已固定): 第一版主线是 cache-driven reset，而不是普通
    object pose DR + random joint reset。正式实现应新增一个 cache reset event：
    $$
    (q, T^h_o) \sim
    \mathcal{D}_{\text{grasp}}(q,T^h_o\mid a,o,s,\rho),
    $$
    并在 reset 时写 hand joint position、object pose、零速度以及 action target。

    互斥关系：
        - `reset_grasp_cache` 启用时，不应同时启用 random object pose reset；
        - `reset_grasp_cache` 启用时，不应再叠加 random hand joint offset；
        - 无 cache 消融才启用 `random_reset_object_ablation` 与
          `random_reset_robot_joints_ablation`。

    DR 阶段：
        - object scale 是 startup / usd-time 离散 bucket，不是 episode reset 噪声；
        - object mass / CoM / friction、robot link material / mass / CoM、actuator
          stiffness / damping、joint friction / armature 默认 startup 采样；
        - joint limit DR、collider offset DR、fixed tendon DR、interval 外力暂缓。

    NOTE:
        这里暂不放任何 active `EventTerm`，避免当前 scaffold 在无 cache reset 实现
        时悄悄退化成与主线不一致的随机初态环境。下面三个字段只是命名锚点，
        后续实现时再替换为真实 Isaac Lab `EventTermCfg`。
    """

    reset_grasp_cache = None  # 主线占位：未来写入 cache sample $(q,T^h_o)$
    random_reset_object_ablation = None  # no-cache 消融占位：才允许 object pose DR
    random_reset_robot_joints_ablation = None  # no-cache 消融占位：才允许 random joint reset


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

    该 cfg 是 task-level assembly surface。它表达“环境由哪些 MDP 组件组成”，
    不表达“训练时选哪 64 个 assets”。

    TODO:
        正式实现前必须解决：
        1. `scene.robot` 的 generated hand binding；
        2. same-topology action joint order contract；
        3. command / reward 函数从 placeholder 变成可验证实现；
        4. random-agent smoke test。
    """

    scene: GmInHandSceneCfg = GmInHandSceneCfg(num_envs=4096, env_spacing=0.75, replicate_physics=False)
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
        self.viewer.eye = (2.0, 2.0, 2.0)


@configclass
class GmInHandEnvCfg_PLAY(GmInHandEnvCfg):
    r"""Small-scene variant for visual review and smoke tests."""

    def __post_init__(self):
        r"""Disable training-only noise for visual inspection."""

        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
        self.terminations.time_out = None


__all__ = [
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
