r"""RL teacher training entry package for AnyMani distill.

本包自包含 GM teacher 的训练注册、rl_games backend 固定、网络 builder 和训练入口。
外层 `scripts/rl_games/train.py` 仍可作为历史通用工具存在，但层次通才策略训练主线
从这里进入，避免训练逻辑散落在项目根脚本目录。
"""

from __future__ import annotations

import gymnasium as gym

from . import agents

gym.register(
    id="AnyMani-GM-Teacher-Debug-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gm_teacher_env_cfg:GmTeacherDebugEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:gm_teacher_transformer_ppo.yaml",
    },
)

gym.register(
    id="AnyMani-GM-Teacher-Debug-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gm_teacher_env_cfg:GmTeacherDebugEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:gm_teacher_transformer_ppo.yaml",
    },
)

gym.register(
    id="AnyMani-GM-Heterogeneous-MLP-Smoke-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gm_heterogeneous_mlp_smoke_env_cfg:HeterogeneousMlpSmokeEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:gm_heterogeneous_mlp_ppo_smoke.yaml",
    },
)


__all__ = []
