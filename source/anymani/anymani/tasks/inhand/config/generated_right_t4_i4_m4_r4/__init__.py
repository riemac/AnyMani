"""Generated `right_t4_i4_m4_r4` official-ADR task registrations."""

from __future__ import annotations

import gymnasium as gym

from anymani.tasks.inhand.config.leaphand import agents

from .generated_right_t4_i4_m4_r4_adr_env_cfg import (
    LeapHandADRGeneratedRightT4I4M4R4EnvCfg,
    LeapHandADRGeneratedRightT4I4M4R4EnvCfg_PLAY,
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


__all__ = [
    "LeapHandADRGeneratedRightT4I4M4R4EnvCfg",
    "LeapHandADRGeneratedRightT4I4M4R4EnvCfg_PLAY",
]
