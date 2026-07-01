"""URDF-backed LeapHand official-aligned ADR task registrations."""

from __future__ import annotations

import gymnasium as gym

from anymani.tasks.inhand.config.leaphand import agents

from .leaphand_urdf_adr_env_cfg import LeapHandTactileADRURDFEnvCfg, LeapHandTactileADRURDFEnvCfg_PLAY

# 训练入口：保留 N010 official-aligned ADR 的 MDP 与 rl_games 配置，只把 hand backend 换成 raw URDF。
gym.register(
    id="AnyMani-LeapHand-Tactile-ADR-URDF-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leaphand_urdf_adr_env_cfg:LeapHandTactileADRURDFEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_official_adr.yaml",
    },
)

# 可视化入口：同一 URDF backend 和 MDP，仅降低 env count 并打开 goal marker。
gym.register(
    id="AnyMani-LeapHand-Tactile-ADR-URDF-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leaphand_urdf_adr_env_cfg:LeapHandTactileADRURDFEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_official_adr.yaml",
    },
)


__all__ = ["LeapHandTactileADRURDFEnvCfg", "LeapHandTactileADRURDFEnvCfg_PLAY"]
