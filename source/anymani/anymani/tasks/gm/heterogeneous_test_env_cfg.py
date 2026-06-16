r"""同拓扑异构 generated hand 的 IsaacLab 环境 MVP。

本文件只拥有 `tasks/gm` 层的环境语义：scene、obs、action、reward、termination、
viewer 和 Gym 注册所需 env cfg。它不拥有训练算法、网络结构、checkpoint 或
asset-bank split；这些由 `distill` 消费本环境后自包含实现。

科研目标是用最小可运行闭环验证同一 schema 下的异构手能进入 batched IsaacLab env：

1. 随机动作 / GUI MVP：用 `AnyMani-GM-Heterogeneous-Test-v0` 检查 3 个 URDF hand
   variants 能 round-robin spawn、reset、step，并且材质颜色、天空、手部姿态可视可信。
2. MLP 训练 MVP：用 `distill` 中的 `AnyMani-GM-Heterogeneous-MLP-Smoke-v0` 绑定本环境，
   以 $3\times100=300$ envs 和极简 MLP PPO 只验证 rollout / backward / checkpoint 闭环，
   不评价 reward 表现或 policy 质量。

本轮实现的关键变化：hand asset set 不再由本文件维护私有 `HeterogeneousHandSetCfg`。
本文件改为通过 `HandSpawnCfg(bank=HandBankCfg(...))` 调用 `HandSpawnAdapter`，从而让
`tasks/gm` 的 smoke 直接验证 asset bank 到 IsaacLab `ArticulationCfg` 的接口边界。

目标 URDF 固定为同一 post-mutate run 下的 3 个 same-schema variants：

- `source/anymani/anymani/assets/generated/2026-06-10_11-30-08/single_palm_leap/right_t4_i4_m4_r4/2026-06-11_14-20-22/0b6fbfce/hand.urdf`
- `source/anymani/anymani/assets/generated/2026-06-10_11-30-08/single_palm_leap/right_t4_i4_m4_r4/2026-06-11_14-20-22/0bdf0eca/hand.urdf`
- `source/anymani/anymani/assets/generated/2026-06-10_11-30-08/single_palm_leap/right_t4_i4_m4_r4/2026-06-11_14-20-22/00d68163/hand.urdf`

推荐验证命令：

```bash
python scripts/random_agent.py --task AnyMani-GM-Heterogeneous-Test-v0 --num_envs 9
python -m anymani.distill.train_mvp --task AnyMani-GM-Heterogeneous-MLP-Smoke-v0 --num_envs 300 --max_iterations 1 --headless
```
"""

from __future__ import annotations

import isaaclab.envs.mdp as isaac_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from anymani.assets.bank import HandBankCfg

from .hand_spawn import DEFAULT_HAND_ANCHOR_POS_E, HandFrameCfg, HandSpawnAdapter, HandSpawnCfg

HETEROGENEOUS_RUN_PATH = (
    "source/anymani/anymani/assets/generated/2026-06-10_11-30-08/"
    "single_palm_leap/right_t4_i4_m4_r4/2026-06-11_14-20-22"
)
r"""本 MVP 固定使用的 post-mutate run 路径。

该路径按 AnyMani repo root 解析，而不是按 shell cwd 解析。路径解析由
`assets.bank.path_utils.resolve_bank_path` 完成，因此 VSCode、pytest 和脚本入口共享
同一语义。
"""

HETEROGENEOUS_HAND_IDS = ("0b6fbfce", "0bdf0eca", "00d68163")
r"""异构 smoke 固定选中的 3 个 same-schema post-mutate sample id。

`HandBankCfg.containers` 接受字符串简写；运行时会 lower 成 `HandContainerCfg(path=...)`。
`asset_routing="round_robin"` 时 IsaacLab 会按 `env_id % 3` 轮转，因此 `num_envs=9`
应在 GUI 中看到 A/B/C/A/B/C/A/B/C。
"""

HETEROGENEOUS_HAND_INIT_ROT = (1.0, 0.0, 0.0, 0.0)
r"""异构可视化 smoke 的 hand raw root 初始姿态 `(w,x,y,z)`。

该测试环境的目的不是复刻旧 LeapHand object-in-hand 任务的腕部姿态，而是核对
same-schema generated hand 在世界系中的可视语义。因此这里显式采用单位四元数，
表达第一版假设 `{a}\approx{h}` 且 $R_{wh}=I$：手心语义法向 $z^h$ 指向世界上方
$z^w$，$x^h,y^h$ 与 $x^w,y^w$ 同向。若后续资产 metadata 给出严格 `{a}->{h}`
校准矩阵，应在 `HandFrameCfg.semantic_R_ha` 中组合该固定校准。
"""

DEFAULT_HETEROGENEOUS_HAND_SPAWN_CFG = HandSpawnCfg(
    bank=HandBankCfg(
        source_mode="post_mutate",
        selection_mode="explicit",
        post_mutate_path=HETEROGENEOUS_RUN_PATH,
        containers=HETEROGENEOUS_HAND_IDS,
        validate_mesh_relpaths=True,
        parse_visual_rgba=True,
    ),
    frame=HandFrameCfg(
        semantic_R_ha=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        semantic_p_ha=(0.0, 0.0, 0.0),
        anchor_R_eh=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        anchor_p_eh=DEFAULT_HAND_ANCHOR_POS_E,
    ),
    asset_routing="round_robin",
    restore_visual_materials=True,
    validate_same_schema=True,
)
r"""第一版异构 smoke 的 hand spawn cfg。

该 cfg 是本文件的核心验证对象：它把 asset bank 的 3 个 container 通过
`HandSpawnAdapter` lower 成 IsaacLab `ArticulationCfg`。hand orientation reset 本轮不接入
本环境；这里只验证 spawn anchor $T_{ea}^{anchor}=T_{eh}^{anchor}T_{ha}$，其中当前
generated hand 采用 identity $T_{ha}$。
"""


def build_heterogeneous_hand_articulation_cfg(
    hand_spawn_cfg: HandSpawnCfg,
    *,
    prim_path: str,
) -> ArticulationCfg:
    r"""将声明式 hand spawn cfg 绑定为一个 batched `ArticulationCfg`。

    Args:
        hand_spawn_cfg (HandSpawnCfg): 通过 asset bank 描述的一组 same-schema hand variants。
        prim_path (str): IsaacLab scene 中 robot articulation 的 prim path。

    Returns:
        ArticulationCfg: 可作为 `scene.robot` 的异构 articulation 配置。
    """

    adapter = HandSpawnAdapter(hand_spawn_cfg)  # runtime adapter；首次 build 时 lazy resolve asset bank
    return adapter.build_articulation_cfg(prim_path=prim_path)  # lower 到 IsaacLab `scene.robot` cfg


@configclass
class HeterogeneousHandTestSceneCfg(InteractiveSceneCfg):
    r"""只包含异构 hand articulation 的最小 scene。

    没有 object；目标是隔离验证同一 `Articulation` batch 能否持有多个
    same-schema post-mutate hand variants。
    """

    robot: ArticulationCfg = build_heterogeneous_hand_articulation_cfg(
        DEFAULT_HETEROGENEOUS_HAND_SPAWN_CFG,
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.1)),
    )

    light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class HeterogeneousHandTestActionsCfg:
    r"""官方相对关节位置动作。

    第一版不使用 GM clamp action，避免把异构 articulation smoke 与自定义 MDP
    组件耦合。`scale=0.05` 让随机 agent 的 joint target 做较慢随机游走。
    """

    joint_pos = isaac_mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.05,
        preserve_order=True,
    )


@configclass
class HeterogeneousHandTestObservationsCfg:
    r"""最小 policy observation：关节位置与速度。"""

    @configclass
    class PolicyCfg(ObsGroup):
        r"""Actor-facing flat observation group。"""

        joint_pos = ObsTerm(func=isaac_mdp.joint_pos)
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel)

        def __post_init__(self) -> None:
            r"""关闭噪声并拼接成单个 flat tensor。"""

            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class HeterogeneousHandTestRewardsCfg:
    r"""最小 alive reward，只保证 RL env reward manager 有合法输出。"""

    alive = RewTerm(func=isaac_mdp.is_alive, weight=1.0)


@configclass
class HeterogeneousHandTestTerminationsCfg:
    r"""最小 termination：只按 episode time limit reset。"""

    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)


@configclass
class HeterogeneousHandTestEnvCfg(ManagerBasedRLEnvCfg):
    r"""异构 hand articulation smoke env。

    该环境不是正式 GM teacher，也不表达 object manipulation 任务；它只验证
    IsaacLab 能否在一个 batched `Articulation` 中承载 3 个 same-schema URDF hand
    variants，并能被 `scripts/random_agent.py` reset/step。
    """

    scene: HeterogeneousHandTestSceneCfg = HeterogeneousHandTestSceneCfg(
        num_envs=9,
        env_spacing=0.75,
        replicate_physics=False,
        clone_in_fabric=False,
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

    observations: HeterogeneousHandTestObservationsCfg = HeterogeneousHandTestObservationsCfg()
    actions: HeterogeneousHandTestActionsCfg = HeterogeneousHandTestActionsCfg()
    rewards: HeterogeneousHandTestRewardsCfg = HeterogeneousHandTestRewardsCfg()
    terminations: HeterogeneousHandTestTerminationsCfg = HeterogeneousHandTestTerminationsCfg()
    commands = None
    curriculum = None

    def __post_init__(self) -> None:
        r"""设置随机可视化 smoke 的仿真时序。"""

        super().__post_init__()
        self.decimation = 4
        self.episode_length_s = 4.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (2.0, 2.0, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.4)


__all__ = [
    "DEFAULT_HETEROGENEOUS_HAND_SPAWN_CFG",
    "HETEROGENEOUS_HAND_IDS",
    "HETEROGENEOUS_HAND_INIT_ROT",
    "HETEROGENEOUS_RUN_PATH",
    "HeterogeneousHandTestActionsCfg",
    "HeterogeneousHandTestEnvCfg",
    "HeterogeneousHandTestObservationsCfg",
    "HeterogeneousHandTestRewardsCfg",
    "HeterogeneousHandTestSceneCfg",
    "HeterogeneousHandTestTerminationsCfg",
    "build_heterogeneous_hand_articulation_cfg",
]
