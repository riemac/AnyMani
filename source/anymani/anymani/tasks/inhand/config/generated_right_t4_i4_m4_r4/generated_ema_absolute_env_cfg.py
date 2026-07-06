r"""N041 `EMAAbsolute` env cfg for generated `right_t4_i4_m4_r4`.

本文件定义 N041：以 N030 generated official-ADR 为基准，只把动作更新律从 LEAP current config 的
target-buffer raw-delta relative 换成 LEAP absolute branch / IsaacLab EMA 同构的 joint-limit absolute EMA。

N030 动作语义：

$$
u_t=\operatorname{clip}\left(u_{t-1}+\frac{1}{24}a_t^{exec},q_{min},q_{max}\right).
$$

N041 动作语义：

$$
v_t=S(a_t^{exec})=q_{min}+\frac{a_t^{exec}+1}{2}(q_{max}-q_{min}),
$$

$$
u_t=\operatorname{clip}\left(\alpha v_t+(1-\alpha)u_{t-1},q_{min},q_{max}\right),
\quad \alpha=\frac{1}{24}.
$$

科研边界：scene / observation / reward / command / termination / ADR / curriculum / reset basin 均继承 N030。
本 run 不引入 `[q/\pi,u/\pi]` raw obs，因此 N041 是 action-law ablation，而不是 obs 表征 ablation。
"""

from __future__ import annotations

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from anymani.tasks.gm.mdp.actions import ADREMAJointPositionToLimitsActionCfg
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
    GeneratedRightT4I4M4R4OfficialADRObservationsCfg,
    GeneratedRightT4I4M4R4OfficialADRSceneCfg,
    LeapHandADRGeneratedRightT4I4M4R4EnvCfg,
)

EMA_ABSOLUTE_ALPHA = 1.0 / 24.0
r"""N041 EMA absolute 的 LEAP-style blend 系数 $\alpha=1/24$。"""


@configclass
class GeneratedRightT4I4M4R4EMAAbsoluteActionsCfg:
    r"""N041 action 组合面：joint-limit absolute target + target-buffer EMA。

    与 N030 的共同点：

    - policy-facing action 仍是 16D normalized action $a_t\in[-1,1]^{16}$；
    - official ADR action noise / latency 仍默认开启；
    - action term 仍暴露 `current_targets` / `executed_actions` / `pregrasp_targets`，供 official obs/reward 使用。

    与 N030 的差异：N041 不再把 $a_t^{exec}$ 解释为 raw-rad delta，而是先映射到 joint-limit
    absolute target $v_t=S(a_t^{exec})$，再与上一帧 target buffer $u_{t-1}$ 做 EMA。
    """

    hand_joint_pos = ADREMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=list(GENERATED_OFFICIAL_SLOT_JOINT_ORDER),
        reference="target",
        alpha=EMA_ABSOLUTE_ALPHA,
        scale=1.0,
        offset=0.0,
        preserve_order=True,
        use_adr=True,
        pregrasp_joint_pos=GENERATED_RIGHT_T4_I4_M4_R4_PREGRASP_VECTOR,
    )
    r"""16D EMA absolute action；`reference="target"` 表示 $r_t=u_{t-1}$。"""


@configclass
class LeapHandADRGeneratedRightT4I4M4R4EMAAbsoluteEnvCfg(LeapHandADRGeneratedRightT4I4M4R4EnvCfg):
    r"""N041 generated EMA-absolute-action train env。

    继承 N030 generated official-ADR env，只替换 `actions`。因此 actor obs 仍是 official 96D：

    $$
    o_t^\pi=[\tilde q_t,u_t]_{t-2:t}.
    $$
    """

    scene: InteractiveSceneCfg = GeneratedRightT4I4M4R4OfficialADRSceneCfg(
        num_envs=4096,
        env_spacing=0.75,
        replicate_physics=False,
    )
    observations: GeneratedRightT4I4M4R4OfficialADRObservationsCfg = GeneratedRightT4I4M4R4OfficialADRObservationsCfg()
    actions: GeneratedRightT4I4M4R4EMAAbsoluteActionsCfg = GeneratedRightT4I4M4R4EMAAbsoluteActionsCfg()
    commands: LeapHandOfficialADRCommandsCfg = LeapHandOfficialADRCommandsCfg()
    rewards: LeapHandOfficialADRRewardsCfg = LeapHandOfficialADRRewardsCfg()
    terminations: LeapHandOfficialADRTerminationsCfg = LeapHandOfficialADRTerminationsCfg()
    events: GeneratedRightT4I4M4R4OfficialADREventCfg = GeneratedRightT4I4M4R4OfficialADREventCfg()
    curriculum: LeapHandOfficialADRCurriculumCfg = LeapHandOfficialADRCurriculumCfg()


@configclass
class LeapHandADRGeneratedRightT4I4M4R4EMAAbsoluteEnvCfg_PLAY(LeapHandADRGeneratedRightT4I4M4R4EMAAbsoluteEnvCfg):
    r"""N041 EMAAbsolute play/debug env。"""

    def __post_init__(self) -> None:
        r"""降低 env 数并打开 goal marker，便于与 N030/N040 replay 对比。"""

        super().__post_init__()
        self.scene.num_envs = 50  # Play 模式沿用 N030/N040 的小规模可视化约定。
        self.commands.goal_pose.debug_vis = True  # 显示 continuous rotation goal marker。


__all__ = [
    "EMA_ABSOLUTE_ALPHA",
    "GeneratedRightT4I4M4R4EMAAbsoluteActionsCfg",
    "LeapHandADRGeneratedRightT4I4M4R4EMAAbsoluteEnvCfg",
    "LeapHandADRGeneratedRightT4I4M4R4EMAAbsoluteEnvCfg_PLAY",
]
