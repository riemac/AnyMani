"""Generated heterogeneous-hand palm-rotation task registration."""

from __future__ import annotations

import gymnasium as gym

gym.register(
    id="AnyMani-Hetero-Generated-PalmRotation-MVP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.config.generated.palm_rotation_mvp_env_cfg:GeneratedPalmRotationMvpEnvCfg"
        )
    },
)
