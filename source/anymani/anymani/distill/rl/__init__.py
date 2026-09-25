"""Register the paper's heterogeneous palm-rotation PPO route."""

from __future__ import annotations

import gymnasium as gym

from . import agents

gym.register(
    id="AnyMani-Hetero-Generated-PalmRotation-MVP-RLGames-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "anymani.tasks.hetero.config.generated.palm_rotation_mvp_env_cfg:GeneratedPalmRotationMvpEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:heterogeneous_palm_rotation_mvp_ppo.yaml",
    },
)

__all__ = []
