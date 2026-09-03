r"""Generated heterogeneous-hand ManagerBased task family。

根package不导入Isaac runtime modules，保证纯contract测试、schema工具和文档构建不会在collection阶段启动Kit。
Gym registration只会在完整scene/config闭合后加入。
"""

from __future__ import annotations

import gymnasium as gym

gym.register(
    id="AnyMani-Hetero-Generated-TactileRotation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.config.generated.tactile_rotation_env_cfg:GeneratedHeterogeneousTactileRotationEnvCfg"
        )
    },
)

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
