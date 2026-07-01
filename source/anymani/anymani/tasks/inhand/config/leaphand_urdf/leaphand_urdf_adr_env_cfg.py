from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from anymani.robots.leap_urdf import LEAP_HAND_URDF_CFG
from anymani.tasks.inhand.config.leaphand.leaphand_adr_env_cfg import (
    OFFICIAL_PREGRASP_BY_NAME,
    LeapHandOfficialADRSceneCfg,
    LeapHandTactileADREnvCfg,
)


@configclass
class LeapHandOfficialADRURDFSceneCfg(LeapHandOfficialADRSceneCfg):
    r"""Official-aligned LEAP ADR scene backed by the raw official URDF asset.

    该 scene 只替换 robot asset backend：object、ground、light、DexCube scale/mass、
    object 初始位姿与 N010 official-aligned USD baseline 保持一致。核心实验变量为
    hand asset 从历史 USD 路线切换到 raw URDF importer 路线。

    数值锚点：
    - hand root pose in env frame: $p_e=(0,0,0.5)$, $q_e=(0.5,0.5,-0.5,0.5)$；
    - official pre-grasp: `OFFICIAL_PREGRASP_BY_NAME`；
    - object contact basin 继承自 `LeapHandOfficialADRSceneCfg`：
      $p_o^e=(0,-0.1,0.56)$, $q_o^e=(1,0,0,0)$。
    """

    # 仅替换 robot backend，保持 N010 的 hand root `{e}` pose 与官方 pre-grasp joint anchor。
    robot: ArticulationCfg = LEAP_HAND_URDF_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),  # hand root 在 env frame `{e}` 下的位置，单位 m。
            rot=(0.5, 0.5, -0.5, 0.5),  # hand root 在 `{e}` 下的 wxyz quaternion，与官方 USD baseline 对齐。
            joint_pos=OFFICIAL_PREGRASP_BY_NAME,  # 官方 reset/pre-grasp 关节角，单位 rad，按 joint name 写入。
            joint_vel={"a_.*": 0.0},  # reset 时所有 LEAP actuated joints 的初始速度，单位 rad/s。
        ),
    )


@configclass
class LeapHandTactileADRURDFEnvCfg(LeapHandTactileADREnvCfg):
    r"""N010 official-aligned ADR env with only the hand backend changed to URDF.

    继承 `LeapHandTactileADREnvCfg` 的 obs/action/reward/ADR/termination/rl timing
    语义，避免复制整套 MDP 后产生漂移。该类的研究问题是：在官方 pre-grasp 与
    object contact basin 均固定为官方值时，raw URDF backend 是否能复现 USD
    baseline 的早期学习曲线。
    """

    # 训练规模与 N010 保持一致：4096 env、0.75 m spacing、非 replicate physics。
    scene: InteractiveSceneCfg = LeapHandOfficialADRURDFSceneCfg(
        num_envs=4096,
        env_spacing=0.75,
        replicate_physics=False,
    )


@configclass
class LeapHandTactileADRURDFEnvCfg_PLAY(LeapHandTactileADRURDFEnvCfg):
    r"""Play/debug variant for the URDF-backed official-aligned ADR env."""

    def __post_init__(self):
        r"""Reduce env count for visualization and enable goal-pose debug markers."""

        super().__post_init__()
        self.scene.num_envs = 50  # 可视化时降低并行环境数量，保持与 N010 Play variant 一致。
        self.commands.goal_pose.debug_vis = True  # 显示连续 z-axis rotation target，便于肉眼核对任务语义。
