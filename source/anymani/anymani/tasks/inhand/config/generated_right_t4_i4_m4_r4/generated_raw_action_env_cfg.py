r"""N040 `raw-delta-action` env cfg for generated `right_t4_i4_m4_r4`.

本文件把先前的设计 scaffold 晋升为真实组合面，但仍严格保持“只换 action 与 actor obs”的
科研边界：scene / reward / command / termination / ADR / structural collision filter / reset basin
都继承 N030 generated official-ADR 主线。

N040 第一刀的目标不是一步到位进入 heterogeneous MDP，而是在 single-asset generated probe 上做
最小动作语义迁移：

1. official target-buffer action
   $$
   q_t^{target}=\operatorname{clip}\left(q_{t-1}^{target}+\frac{1}{24}a_t^{exec},q^{min},q^{max}\right)
   $$
   替换为 ADR-aware raw-rad current-relative action：
   $$
   q_t^{cmd}=\operatorname{clip}\left(q_t+\frac{1}{24}a_t^{exec},q^{min},q^{max}\right).
   $$
2. official actor obs
   $$
   o_t^{frame}=[\tilde q_t,q_t^{target}]\in\mathbb R^{32}
   $$
   替换为 unit-scaled raw actor obs：
   $$
   o_t^{frame}=\left[\frac{q_t}{\pi},\frac{q_t^{cmd}}{\pi}\right]\in\mathbb R^{32}.
   $$

外部接口仍刻意保持 96D actor obs 与同一 generated scene，使 N040 能直接和 N030 比较，而不把
network input dim、reward scale、object contact basin 等变量混进来。
"""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from anymani.tasks.gm.mdp.actions import ADRRelativeJointPositionActionCfg
from anymani.tasks.inhand import mdp as leap_mdp
from anymani.tasks.inhand.config.leaphand.leaphand_adr_env_cfg import (
    LeapHandOfficialADRCommandsCfg,
    LeapHandOfficialADRCurriculumCfg,
    LeapHandOfficialADRRewardsCfg,
    LeapHandOfficialADRTerminationsCfg,
)

from .generated_right_t4_i4_m4_r4_adr_env_cfg import (
    GENERATED_OFFICIAL_SLOT_JOINT_ORDER,
    GENERATED_RIGHT_T4_I4_M4_R4_PREGRASP_VECTOR,
    GeneratedRightT4I4M4R4OfficialADREventCfg,
    GeneratedRightT4I4M4R4OfficialADRSceneCfg,
    LeapHandADRGeneratedRightT4I4M4R4EnvCfg,
)

RAW_DELTA_ACTION_SCALE_RAD = 1.0 / 24.0
r"""N040 第一刀固定的 raw-rad delta scale。

这里刻意沿用 N030 official action 的每步角度尺度：

$$
\alpha=\frac{1}{24}\ \mathrm{rad}.
$$

这样 N040 第一刀主要比较的是动作参考点从 $q_{t-1}^{target}$ 切到 $q_t$，而不是把每步动作幅值
同时改大到 GM 历史 scaffold 中的 $0.1$ rad。
"""


@configclass
class GeneratedRightT4I4M4R4RawDeltaPolicyObsCfg(ObsGroup):
    r"""N040 actor obs：三帧 history 的 `[q/\pi, q_cmd/\pi]`。

    单帧 32D：

    $$
    o_t^{frame}=\left[\frac{q_t}{\pi},\frac{q_t^{cmd}}{\pi}\right]\in\mathbb R^{32}.
    $$

    三帧 history 后仍为 96D：

    $$
    o_t^\pi=[o_{t-2}^{frame},o_{t-1}^{frame},o_t^{frame}]\in\mathbb R^{96}.
    $$
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
            "joint_scale_rad": 3.141592653589793,
        },
        history_length=3,
        flatten_history_dim=True,
    )
    r"""唯一 actor obs term：保持与 N030 一样的 96D shape，但内部语义改成 raw/unit-scaled。"""

    def __post_init__(self) -> None:
        r"""关闭 obs corruption，并显式拼接唯一 obs term。"""

        self.enable_corruption = False  # N040 第一刀不引入额外 obs noise，避免 action ablation 归因混乱。
        self.concatenate_terms = True  # 单 term 仍显式拼接，保持 ManagerBased obs contract 清晰。


@configclass
class GeneratedRightT4I4M4R4RawDeltaObservationsCfg:
    r"""N040 observations 组合面，仅替换 actor obs。"""

    @configclass
    class PolicyCfg(GeneratedRightT4I4M4R4RawDeltaPolicyObsCfg):
        r"""Actor-facing observation group。"""

    policy: ObsGroup = PolicyCfg()


@configclass
class GeneratedRightT4I4M4R4RawDeltaActionsCfg:
    r"""N040 action 组合面：ADR-aware raw-rad current-relative delta。

    当前动作项来自 `gm.mdp.actions.ADRRelativeJointPositionActionCfg(reference="current")`，保留：

    - official ADR action noise；
    - official ADR action latency；
    - current-relative raw-rad delta；
    - soft joint limit clamp；
    - `current_targets` 运行态，供 obs 读取。
    """

    hand_joint_pos = ADRRelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(GENERATED_OFFICIAL_SLOT_JOINT_ORDER),
        reference="current",
        scale=RAW_DELTA_ACTION_SCALE_RAD,
        preserve_order=True,
        use_zero_offset=True,
        use_adr=True,
        pregrasp_joint_pos=GENERATED_RIGHT_T4_I4_M4_R4_PREGRASP_VECTOR,
        clip={".*": (-RAW_DELTA_ACTION_SCALE_RAD, RAW_DELTA_ACTION_SCALE_RAD)},
    )
    r"""16D raw-rad delta action；clip 仅作为每步角增量的安全网，不改变主语义。"""


@configclass
class LeapHandADRGeneratedRightT4I4M4R4RawDeltaEnvCfg(LeapHandADRGeneratedRightT4I4M4R4EnvCfg):
    r"""N040 generated raw-delta-action train env。

    继承 N030 generated official-ADR env 的 scene / reward / command / termination / events / curriculum，
    只替换：

    - `actions`: official target-buffer -> ADR-aware raw-rad current-relative delta；
    - `observations`: `[\tilde q,q_target]` -> `[q/\pi,q_cmd/\pi]`。
    """

    scene: InteractiveSceneCfg = GeneratedRightT4I4M4R4OfficialADRSceneCfg(
        num_envs=4096,
        env_spacing=0.75,
        replicate_physics=False,
    )
    observations: GeneratedRightT4I4M4R4RawDeltaObservationsCfg = GeneratedRightT4I4M4R4RawDeltaObservationsCfg()
    actions: GeneratedRightT4I4M4R4RawDeltaActionsCfg = GeneratedRightT4I4M4R4RawDeltaActionsCfg()
    commands: LeapHandOfficialADRCommandsCfg = LeapHandOfficialADRCommandsCfg()
    rewards: LeapHandOfficialADRRewardsCfg = LeapHandOfficialADRRewardsCfg()
    terminations: LeapHandOfficialADRTerminationsCfg = LeapHandOfficialADRTerminationsCfg()
    events: GeneratedRightT4I4M4R4OfficialADREventCfg = GeneratedRightT4I4M4R4OfficialADREventCfg()
    curriculum: LeapHandOfficialADRCurriculumCfg = LeapHandOfficialADRCurriculumCfg()


@configclass
class LeapHandADRGeneratedRightT4I4M4R4RawDeltaEnvCfg_PLAY(LeapHandADRGeneratedRightT4I4M4R4RawDeltaEnvCfg):
    r"""N040 raw-delta-action play/debug env。"""

    def __post_init__(self) -> None:
        r"""降低 env 数并显示 continuous z-axis goal marker。"""

        super().__post_init__()
        self.scene.num_envs = 50  # Play 模式保持与 N030 / N031 一致，便于肉眼对比。
        self.commands.goal_pose.debug_vis = True  # 显示目标姿态，确认 command 主线未被改坏。


__all__ = [
    "RAW_DELTA_ACTION_SCALE_RAD",
    "GeneratedRightT4I4M4R4RawDeltaActionsCfg",
    "GeneratedRightT4I4M4R4RawDeltaObservationsCfg",
    "GeneratedRightT4I4M4R4RawDeltaPolicyObsCfg",
    "LeapHandADRGeneratedRightT4I4M4R4RawDeltaEnvCfg",
    "LeapHandADRGeneratedRightT4I4M4R4RawDeltaEnvCfg_PLAY",
]
