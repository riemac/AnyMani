r"""N050 / N051 observation ablations for generated `right_t4_i4_m4_r4`.

本文件承接 N030 generated official-ADR 主线，只改 actor observation 的关节位置表征，
不改 action law、reward、ADR、command、termination、reset basin 或 generated asset。

N030 official observation 单帧为混合量纲：

$$
o_t^{N030,frame}=[\tilde q_t,u_t^{rad}]\in\mathbb R^{32},
$$

其中 $\tilde q_t$ 是 joint-limit normalized joint position，$u_t^{rad}$ 是 official
target-buffer action 当前下发给 PD controller 的 raw-rad target。LEAP 官方 demo 也是这个
混合量纲 contract。

N050 第一刀只去掉 $q_t$ 通道的 joint-limit normalization：

$$
o_t^{N050,frame}=[q_t^{rad},u_t^{rad}]\in\mathbb R^{32}.
$$

N051 再把所有 joint-position-like channel 统一到 unit-scaled raw-rad 坐标：

$$
o_t^{N051,frame}=\left[\frac{q_t}{\pi},\frac{u_t}{\pi}\right]\in\mathbb R^{32}.
$$

三帧 history 均由 IsaacLab `ObservationTermCfg.history_length=3` 提供，因此 policy-facing
observation 维度仍为 96D，方便与 N030 / N040 / N041 做同网络宽度对照。
"""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from anymani.tasks.inhand import mdp as leap_mdp
from anymani.tasks.inhand.config.leaphand.leaphand_adr_env_cfg import (
    LeapHandOfficialADRCommandsCfg,
    LeapHandOfficialADRCurriculumCfg,
    LeapHandOfficialADRRewardsCfg,
    LeapHandOfficialADRTerminationsCfg,
)

from .generated_right_t4_i4_m4_r4_adr_env_cfg import (
    GENERATED_OFFICIAL_SLOT_JOINT_ORDER,
    GeneratedRightT4I4M4R4OfficialADRActionsCfg,
    GeneratedRightT4I4M4R4OfficialADREventCfg,
    GeneratedRightT4I4M4R4OfficialADRSceneCfg,
    LeapHandADRGeneratedRightT4I4M4R4EnvCfg,
)

UNIT_RAW_OBS_JOINT_SCALE_RAD = 3.141592653589793
r"""N051 unit-scaled raw-rad observation 的固定尺度 $\pi$ rad。"""


@configclass
class GeneratedRightT4I4M4R4RawRadObsPolicyObsCfg(ObsGroup):
    r"""N050 actor obs：三帧 history 的 `[q_rad, u_rad]`。

    单帧 observation 为：

    $$
    o_t^{frame}=[q_t,u_t]\in\mathbb R^{32},
    $$

    其中 $q_t$ 是当前真实关节角，$u_t$ 是当前 official target-buffer PD target，二者单位均为 rad。
    N050 不把 $q_t$ 除以 $\pi$，也不把 $u_t$ 重标定，目的是只证伪一件事：
    LEAP official 的 joint-limit normalized proprioception 是否是 N030 成功的必要因素。
    """

    frame = ObsTerm(
        func=leap_mdp.official_policy_frame_raw_rad,
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
    r"""唯一 actor obs term：`[q_t,u_t]_{t-2:t}`，shape `[B,96]`。"""

    def __post_init__(self) -> None:
        r"""关闭 observation corruption，并拼接唯一 term。"""

        self.enable_corruption = False  # N050 是 observation 表征消融，不引入额外观测噪声。
        self.concatenate_terms = True  # 单 term 仍显式拼接，保持 ManagerBased obs contract 清晰。


@configclass
class GeneratedRightT4I4M4R4RawRadObsObservationsCfg:
    r"""N050 observations 组合面，仅替换 actor-facing policy group。"""

    @configclass
    class PolicyCfg(GeneratedRightT4I4M4R4RawRadObsPolicyObsCfg):
        r"""Actor-facing raw-rad proprio-target observation group。"""

    policy: ObsGroup = PolicyCfg()
    r"""训练唯一 observation group；critic 仍沿用 rl_games shared actor-critic 输入。"""


@configclass
class GeneratedRightT4I4M4R4UnitRawObsPolicyObsCfg(ObsGroup):
    r"""N051 actor obs：三帧 history 的 `[q/pi, u/pi]`。

    单帧 observation 为：

    $$
    o_t^{frame}=\left[\frac{q_t}{\pi},\frac{u_t}{\pi}\right]\in\mathbb R^{32}.
    $$

    与 N050 相比，N051 同时重标定 $q_t$ 与 $u_t$ 两个 joint-position-like channel；
    与 N040 相比，N051 保持 N030 的 target-buffer action law，因此可以隔离 unit-scaled raw observation
    自身的训练动力学。
    """

    frame = ObsTerm(
        func=leap_mdp.raw_policy_frame,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(GENERATED_OFFICIAL_SLOT_JOINT_ORDER),
                preserve_order=True,
            ),
            "action_term_name": "hand_joint_pos",
            "joint_scale_rad": UNIT_RAW_OBS_JOINT_SCALE_RAD,
        },
        history_length=3,
        flatten_history_dim=True,
    )
    r"""唯一 actor obs term：`[q_t/pi,u_t/pi]_{t-2:t}`，shape `[B,96]`。"""

    def __post_init__(self) -> None:
        r"""关闭 observation corruption，并拼接唯一 term。"""

        self.enable_corruption = False  # 与 N030 对照时不额外加入 observation noise。
        self.concatenate_terms = True  # 保持 policy obs 为一个 flat 96D tensor。


@configclass
class GeneratedRightT4I4M4R4UnitRawObsObservationsCfg:
    r"""N051 observations 组合面，仅替换 actor-facing policy group。"""

    @configclass
    class PolicyCfg(GeneratedRightT4I4M4R4UnitRawObsPolicyObsCfg):
        r"""Actor-facing unit-scaled raw proprio-target observation group。"""

    policy: ObsGroup = PolicyCfg()
    r"""训练唯一 observation group；网络输入维度保持 N030 的 96D。"""


@configclass
class LeapHandADRGeneratedRightT4I4M4R4RawRadObsEnvCfg(LeapHandADRGeneratedRightT4I4M4R4EnvCfg):
    r"""N050 generated raw-rad-observation train env。

    继承 N030 generated official-ADR env，只替换 actor observation：

    $$
    [\tilde q_t,u_t]_{t-2:t}\rightarrow[q_t,u_t]_{t-2:t}.
    $$

    其余 MDP 项保持 N030：official target-buffer action、official reward、continuous z command、
    reset-hook ADR、generated structural collision filter 与 DexCube contact basin。
    """

    scene: InteractiveSceneCfg = GeneratedRightT4I4M4R4OfficialADRSceneCfg(
        num_envs=4096,
        env_spacing=0.75,
        replicate_physics=False,
    )
    observations: GeneratedRightT4I4M4R4RawRadObsObservationsCfg = GeneratedRightT4I4M4R4RawRadObsObservationsCfg()
    actions: GeneratedRightT4I4M4R4OfficialADRActionsCfg = GeneratedRightT4I4M4R4OfficialADRActionsCfg()
    commands: LeapHandOfficialADRCommandsCfg = LeapHandOfficialADRCommandsCfg()
    rewards: LeapHandOfficialADRRewardsCfg = LeapHandOfficialADRRewardsCfg()
    terminations: LeapHandOfficialADRTerminationsCfg = LeapHandOfficialADRTerminationsCfg()
    events: GeneratedRightT4I4M4R4OfficialADREventCfg = GeneratedRightT4I4M4R4OfficialADREventCfg()
    curriculum: LeapHandOfficialADRCurriculumCfg = LeapHandOfficialADRCurriculumCfg()


@configclass
class LeapHandADRGeneratedRightT4I4M4R4RawRadObsEnvCfg_PLAY(LeapHandADRGeneratedRightT4I4M4R4RawRadObsEnvCfg):
    r"""N050 raw-rad-observation play/debug env。"""

    def __post_init__(self) -> None:
        r"""降低 env 数并打开 goal marker，便于人工 replay 检查。"""

        super().__post_init__()
        self.scene.num_envs = 50  # Play 模式沿用 N030/N040/N041 的小规模可视化约定。
        self.commands.goal_pose.debug_vis = True  # 显示 continuous rotation goal marker。


@configclass
class LeapHandADRGeneratedRightT4I4M4R4UnitRawObsEnvCfg(LeapHandADRGeneratedRightT4I4M4R4EnvCfg):
    r"""N051 generated unit-raw-observation train env。

    继承 N030 generated official-ADR env，只替换 actor observation：

    $$
    [\tilde q_t,u_t]_{t-2:t}\rightarrow\left[\frac{q_t}{\pi},\frac{u_t}{\pi}\right]_{t-2:t}.
    $$

    该节点回答的是 unit-scaled raw-rad coordinate 本身是否改变 PPO 学习动力学；它不改变
    target-buffer action，因此不同于 N040 current-relative raw-delta action 旁支。
    """

    scene: InteractiveSceneCfg = GeneratedRightT4I4M4R4OfficialADRSceneCfg(
        num_envs=4096,
        env_spacing=0.75,
        replicate_physics=False,
    )
    observations: GeneratedRightT4I4M4R4UnitRawObsObservationsCfg = GeneratedRightT4I4M4R4UnitRawObsObservationsCfg()
    actions: GeneratedRightT4I4M4R4OfficialADRActionsCfg = GeneratedRightT4I4M4R4OfficialADRActionsCfg()
    commands: LeapHandOfficialADRCommandsCfg = LeapHandOfficialADRCommandsCfg()
    rewards: LeapHandOfficialADRRewardsCfg = LeapHandOfficialADRRewardsCfg()
    terminations: LeapHandOfficialADRTerminationsCfg = LeapHandOfficialADRTerminationsCfg()
    events: GeneratedRightT4I4M4R4OfficialADREventCfg = GeneratedRightT4I4M4R4OfficialADREventCfg()
    curriculum: LeapHandOfficialADRCurriculumCfg = LeapHandOfficialADRCurriculumCfg()


@configclass
class LeapHandADRGeneratedRightT4I4M4R4UnitRawObsEnvCfg_PLAY(LeapHandADRGeneratedRightT4I4M4R4UnitRawObsEnvCfg):
    r"""N051 unit-raw-observation play/debug env。"""

    def __post_init__(self) -> None:
        r"""降低 env 数并打开 goal marker，便于人工 replay 检查。"""

        super().__post_init__()
        self.scene.num_envs = 50  # Play 模式保持跨 N03x/N04x/N05x 一致。
        self.commands.goal_pose.debug_vis = True  # 显示 continuous rotation goal marker。


__all__ = [
    "UNIT_RAW_OBS_JOINT_SCALE_RAD",
    "GeneratedRightT4I4M4R4RawRadObsObservationsCfg",
    "GeneratedRightT4I4M4R4RawRadObsPolicyObsCfg",
    "GeneratedRightT4I4M4R4UnitRawObsObservationsCfg",
    "GeneratedRightT4I4M4R4UnitRawObsPolicyObsCfg",
    "LeapHandADRGeneratedRightT4I4M4R4RawRadObsEnvCfg",
    "LeapHandADRGeneratedRightT4I4M4R4RawRadObsEnvCfg_PLAY",
    "LeapHandADRGeneratedRightT4I4M4R4UnitRawObsEnvCfg",
    "LeapHandADRGeneratedRightT4I4M4R4UnitRawObsEnvCfg_PLAY",
]
