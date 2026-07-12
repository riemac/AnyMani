"""Generated `right_t4_i4_m4_r4` official-ADR task registrations."""

from __future__ import annotations

import gymnasium as gym

from anymani.tasks.inhand.config.leaphand import agents

from .generated_ema_absolute_env_cfg import (
    LeapHandADRGeneratedRightT4I4M4R4EMAAbsoluteEnvCfg,
    LeapHandADRGeneratedRightT4I4M4R4EMAAbsoluteEnvCfg_PLAY,
)
from .generated_policy_step_target_env_cfg import (
    LeapHandADRGeneratedRightT4I4M4R4PolicyStepTargetEnvCfg,
    LeapHandADRGeneratedRightT4I4M4R4PolicyStepTargetEnvCfg_PLAY,
)
from .generated_raw_action_env_cfg import (
    LeapHandADRGeneratedRightT4I4M4R4RawDeltaEnvCfg,
    LeapHandADRGeneratedRightT4I4M4R4RawDeltaEnvCfg_PLAY,
)
from .generated_raw_observation_env_cfg import (
    LeapHandADRGeneratedRightT4I4M4R4RawRadObsEnvCfg,
    LeapHandADRGeneratedRightT4I4M4R4RawRadObsEnvCfg_PLAY,
    LeapHandADRGeneratedRightT4I4M4R4UnitRawObsEnvCfg,
    LeapHandADRGeneratedRightT4I4M4R4UnitRawObsEnvCfg_PLAY,
)
from .generated_right_t4_i4_m4_r4_adr_env_cfg import (
    LeapHandADRGeneratedRightT4I4M4R4EnvCfg,
    LeapHandADRGeneratedRightT4I4M4R4EnvCfg_PLAY,
    LeapHandADRGeneratedRightT4I4M4R4NoDtRewardEnvCfg,
    LeapHandADRGeneratedRightT4I4M4R4NoDtRewardEnvCfg_PLAY,
)

# 训练入口：沿 N010 official-ADR MDP 语义，只把 hand backend 换为 generated right_t4_i4_m4_r4。
gym.register(
    id="AnyMani-LeapHand-ADR-Generated-right_t4_i4_m4_r4-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.generated_right_t4_i4_m4_r4_adr_env_cfg:"
        "LeapHandADRGeneratedRightT4I4M4R4EnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_official_adr.yaml",
    },
)

# 可视化入口：同一 generated asset / reset seed / MDP，仅降低 env count 并打开 goal marker。
gym.register(
    id="AnyMani-LeapHand-ADR-Generated-right_t4_i4_m4_r4-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.generated_right_t4_i4_m4_r4_adr_env_cfg:"
        "LeapHandADRGeneratedRightT4I4M4R4EnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_official_adr.yaml",
    },
)

# N031 dt-ablation：与 generated official-ADR 完全同构，只取消 combined official reward 的 dt 对齐。
gym.register(
    id="AnyMani-LeapHand-ADR-Generated-right_t4_i4_m4_r4-NoDtReward-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.generated_right_t4_i4_m4_r4_adr_env_cfg:"
        "LeapHandADRGeneratedRightT4I4M4R4NoDtRewardEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_official_adr.yaml",
    },
)

# N031 Play：便于与 N030 replay 做纯 reward-scale 对比，不改 asset / obs / action / ADR。
gym.register(
    id="AnyMani-LeapHand-ADR-Generated-right_t4_i4_m4_r4-NoDtReward-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.generated_right_t4_i4_m4_r4_adr_env_cfg:"
        "LeapHandADRGeneratedRightT4I4M4R4NoDtRewardEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_official_adr.yaml",
    },
)

# N040 raw-delta-action：继承 N030 generated official-ADR，只替换 action 与 actor obs。
gym.register(
    id="AnyMani-LeapHand-ADR-Generated-right_t4_i4_m4_r4-RawDelta-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.generated_raw_action_env_cfg:"
        "LeapHandADRGeneratedRightT4I4M4R4RawDeltaEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_official_adr.yaml",
    },
)

# N040 Play：便于与 N030/N031 做 raw-delta-action replay 对比。
gym.register(
    id="AnyMani-LeapHand-ADR-Generated-right_t4_i4_m4_r4-RawDelta-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.generated_raw_action_env_cfg:"
        "LeapHandADRGeneratedRightT4I4M4R4RawDeltaEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_official_adr.yaml",
    },
)

# N041 EMAAbsolute：继承 N030 generated official-ADR，只替换 action law，obs 保持 official 96D。
gym.register(
    id="AnyMani-LeapHand-ADR-Generated-right_t4_i4_m4_r4-EMAAbsolute-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.generated_ema_absolute_env_cfg:"
        "LeapHandADRGeneratedRightT4I4M4R4EMAAbsoluteEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_official_adr.yaml",
    },
)

# N041 Play：仅降低 env count 并打开 goal marker，便于检查 EMA absolute command target 行为。
gym.register(
    id="AnyMani-LeapHand-ADR-Generated-right_t4_i4_m4_r4-EMAAbsolute-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.generated_ema_absolute_env_cfg:"
        "LeapHandADRGeneratedRightT4I4M4R4EMAAbsoluteEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_official_adr.yaml",
    },
)

# N050 RawRadObs：继承 N030 generated official-ADR，只把 actor obs 改成 `[q_rad, u_rad]`。
gym.register(
    id="AnyMani-LeapHand-ADR-Generated-right_t4_i4_m4_r4-RawRadObs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.generated_raw_observation_env_cfg:"
        "LeapHandADRGeneratedRightT4I4M4R4RawRadObsEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_official_adr.yaml",
    },
)

# N050 Play：用于人工 replay 检查 raw-rad observation 不改变 scene/action/reward 主线。
gym.register(
    id="AnyMani-LeapHand-ADR-Generated-right_t4_i4_m4_r4-RawRadObs-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.generated_raw_observation_env_cfg:"
        "LeapHandADRGeneratedRightT4I4M4R4RawRadObsEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_official_adr.yaml",
    },
)

# N051 UnitRawObs：继承 N030 generated official-ADR，只把 actor obs 改成 `[q/pi, u/pi]`。
gym.register(
    id="AnyMani-LeapHand-ADR-Generated-right_t4_i4_m4_r4-UnitRawObs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.generated_raw_observation_env_cfg:"
        "LeapHandADRGeneratedRightT4I4M4R4UnitRawObsEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_official_adr.yaml",
    },
)

# N051 Play：用于人工 replay 检查 unit-scaled raw observation 不改变 scene/action/reward 主线。
gym.register(
    id="AnyMani-LeapHand-ADR-Generated-right_t4_i4_m4_r4-UnitRawObs-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.generated_raw_observation_env_cfg:"
        "LeapHandADRGeneratedRightT4I4M4R4UnitRawObsEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_official_adr.yaml",
    },
)

# PolicyStepTarget：继承 UnitRawObs，只把 target-buffer 更新从 physics substep 移到 policy step。
gym.register(
    id="AnyMani-LeapHand-ADR-Generated-right_t4_i4_m4_r4-PolicyStepTarget-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.generated_policy_step_target_env_cfg:"
        "LeapHandADRGeneratedRightT4I4M4R4PolicyStepTargetEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_official_adr.yaml",
    },
)

# PolicyStepTarget Play：同一 action lifecycle，只降低 env count 并打开 goal marker。
gym.register(
    id="AnyMani-LeapHand-ADR-Generated-right_t4_i4_m4_r4-PolicyStepTarget-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.generated_policy_step_target_env_cfg:"
        "LeapHandADRGeneratedRightT4I4M4R4PolicyStepTargetEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_official_adr.yaml",
    },
)


__all__ = [
    "LeapHandADRGeneratedRightT4I4M4R4EnvCfg",
    "LeapHandADRGeneratedRightT4I4M4R4EnvCfg_PLAY",
    "LeapHandADRGeneratedRightT4I4M4R4NoDtRewardEnvCfg",
    "LeapHandADRGeneratedRightT4I4M4R4NoDtRewardEnvCfg_PLAY",
    "LeapHandADRGeneratedRightT4I4M4R4RawDeltaEnvCfg",
    "LeapHandADRGeneratedRightT4I4M4R4RawDeltaEnvCfg_PLAY",
    "LeapHandADRGeneratedRightT4I4M4R4EMAAbsoluteEnvCfg",
    "LeapHandADRGeneratedRightT4I4M4R4EMAAbsoluteEnvCfg_PLAY",
    "LeapHandADRGeneratedRightT4I4M4R4RawRadObsEnvCfg",
    "LeapHandADRGeneratedRightT4I4M4R4RawRadObsEnvCfg_PLAY",
    "LeapHandADRGeneratedRightT4I4M4R4UnitRawObsEnvCfg",
    "LeapHandADRGeneratedRightT4I4M4R4UnitRawObsEnvCfg_PLAY",
    "LeapHandADRGeneratedRightT4I4M4R4PolicyStepTargetEnvCfg",
    "LeapHandADRGeneratedRightT4I4M4R4PolicyStepTargetEnvCfg_PLAY",
]
