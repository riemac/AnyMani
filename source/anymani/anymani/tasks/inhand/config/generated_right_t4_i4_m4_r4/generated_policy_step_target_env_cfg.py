r"""Policy-step target-buffer lifecycle ablation for generated `right_t4_i4_m4_r4`。

本文件以 UnitRawObs 环境为冻结父语义，只替换 action lifecycle。父环境在每个
`ManagerBasedRLEnv.step()` 内复用同一 $a_t^{exec}$，并由四次 `apply_actions()` 推进四次 target；
本环境改为：

$$
u_{t+1}=\operatorname{clip}
\left(u_t+\frac{1}{24}a_t^{exec},q_{min},q_{max}\right),
$$

且每个 policy step 只递推一次，随后四个 physics substep hold 同一 $u_{t+1}$。

科研边界：96D $[q/\pi,u/\pi]_{t-2:t}$ observation、generated asset、DexCube、grasp preset、
reward、ADR、command、termination、reset、`sim.dt=1/120 s`、`decimation=4` 与 PPO 全部继承父环境。
因此该环境只回答 target-buffer 更新频率是否影响学习动力学和 finger gait。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from anymani.tasks.inhand import mdp as leap_mdp

from .generated_raw_observation_env_cfg import LeapHandADRGeneratedRightT4I4M4R4UnitRawObsEnvCfg
from .generated_right_t4_i4_m4_r4_adr_env_cfg import (
    GENERATED_OFFICIAL_SLOT_JOINT_ORDER,
    GENERATED_RIGHT_T4_I4_M4_R4_PREGRASP_VECTOR,
)

POLICY_STEP_TARGET_SCALE_RAD = 1.0 / 24.0
r"""每个 policy step 的 normalized action 到 target increment 尺度，单位 rad。"""


@configclass
class GeneratedRightT4I4M4R4PolicyStepTargetActionsCfg:
    r"""每个 policy step 只推进一次的 16D target-buffer action group。

    Joint order、pre-grasp target、action noise 与 latency 参数与父环境完全一致。唯一变化是
    `PolicyStepADRTargetJointPositionAction` 在 `process_actions()` 中更新 target，而不是在每次
    `apply_actions()` 中更新。
    """

    hand_joint_pos = leap_mdp.PolicyStepADRTargetJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(GENERATED_OFFICIAL_SLOT_JOINT_ORDER),
        scale=POLICY_STEP_TARGET_SCALE_RAD,
        preserve_order=True,
        use_zero_offset=True,
        max_latency=3,
        latency_rand=1,
        pregrasp_joint_pos=GENERATED_RIGHT_T4_I4_M4_R4_PREGRASP_VECTOR,
    )
    r"""16D normalized action；一次 policy step 的最大未裁剪 target increment 为 $1/24$ rad。"""


@configclass
class LeapHandADRGeneratedRightT4I4M4R4PolicyStepTargetEnvCfg(
    LeapHandADRGeneratedRightT4I4M4R4UnitRawObsEnvCfg
):
    r"""UnitRawObs 父环境的 policy-step target-buffer train variant。

    该子类只覆盖 `actions`。所有未显式声明的 scene / observation / reward / command / termination /
    event / curriculum / simulation / training contract 均继承父环境。
    """

    actions: GeneratedRightT4I4M4R4PolicyStepTargetActionsCfg = GeneratedRightT4I4M4R4PolicyStepTargetActionsCfg()


@configclass
class LeapHandADRGeneratedRightT4I4M4R4PolicyStepTargetEnvCfg_PLAY(
    LeapHandADRGeneratedRightT4I4M4R4PolicyStepTargetEnvCfg
):
    r"""Policy-step target-buffer play/debug variant。"""

    def __post_init__(self) -> None:
        r"""降低并行环境数并打开 continuous-rotation goal marker。"""

        super().__post_init__()
        self.scene.num_envs = 50  # 与其它 generated Play 环境一致，便于同屏 gait 对比。
        self.commands.goal_pose.debug_vis = True  # 显示连续 z-axis 目标，确认 command 未被 action 消融改变。


__all__ = [
    "POLICY_STEP_TARGET_SCALE_RAD",
    "GeneratedRightT4I4M4R4PolicyStepTargetActionsCfg",
    "LeapHandADRGeneratedRightT4I4M4R4PolicyStepTargetEnvCfg",
    "LeapHandADRGeneratedRightT4I4M4R4PolicyStepTargetEnvCfg_PLAY",
]
