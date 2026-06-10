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

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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

import isaaclab.envs.mdp as isaac_mdp
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

    TODO:
        初期可以复用旧 `inhand` 的 relative SO(3) command 语义；但命令项名字、
        success threshold、是否连续旋转，应在 `gm/mdp/commands.py` 中固定成
        本项目自己的语义，而不是继续从旧 LEAP task 借名。
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

    TODO:
        先把“物体姿态跟踪 + 成功 bonus + 动作/力矩正则”整理成 gm 自己的奖励项。
        grasp 不是当前目标，不要提前扩展目录树；若未来需要抓取，只通过 reward /
        command / termination 组合表达任务差异。
    """

    track_orientation = RewTerm(func=gm_mdp.reorientation_reward_placeholder, weight=1.0)
    action_l2 = RewTerm(func=isaac_mdp.action_l2, weight=-1.0e-4)
    action_rate_l2 = RewTerm(func=isaac_mdp.action_rate_l2, weight=-1.0e-2)


@configclass
class GmEventsCfg:
    r"""Domain randomization and reset scaffold.

    初期只保留对 object / robot 的常规随机化锚点。资产形态多样性已经来自
    pre-made / post-mutate，不应在 reset 事件里偷偷改变 hand topology。
    """

    reset_object = EventTerm(
        func=isaac_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [-0.01, 0.01], "y": [-0.01, 0.01], "z": [-0.01, 0.01], "yaw": [-math.pi, math.pi]},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
        },
    )

    reset_robot_joints = EventTerm(
        func=isaac_mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.2, 0.2), "velocity_range": (0.0, 0.0)},
    )


@configclass
class GmTerminationsCfg:
    r"""Termination scaffold for object-in-hand episodes."""

    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)
    object_falling = DoneTerm(func=gm_mdp.object_falling_placeholder, params={"fall_dist": 0.1})


@configclass
class GmCurriculumCfg:
    r"""Curriculum scaffold.

    TODO:
        初期可为空。若加入 curriculum，应服务任务难度或 command 分布，不要把
        asset-bank 采样策略混进来；后者属于 `distill`。
    """

    pass


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
