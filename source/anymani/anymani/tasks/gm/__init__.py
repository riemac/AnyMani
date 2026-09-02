r"""Single-asset generated-hand与LEAP generalized-manipulation task family。

本包是新一代“手型泛化的手内操作”任务环境入口。`gm` 只注册 Isaac Lab
环境语义：scene、obs、action、command、reward、reset、termination。训练算法、
rl_games YAML、checkpoint、rollout dataset 与网络结构仍由 `distill` 消费本包后
自包含维护。

跨拓扑generated-hand任务由独立``tasks/hetero``拥有；本package只注册single-asset probe与LEAP对照，
不保留旧same-topology/canonical heterogeneous aliases。
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym

gym.register(
    id="AnyMani-GM-SingleAsset-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.single_asset.single_asset_env_cfg:GmSingleAssetEnvCfg",
    },
)

gym.register(
    id="AnyMani-GM-SingleAsset-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.single_asset.single_asset_env_cfg:GmSingleAssetEnvCfg_PLAY",
    },
)

gym.register(
    id="AnyMani-GM-SingleAsset-TactileRotation-CurrentObs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.config.single_asset.tactile_rotation_env_cfg:GmTactileRotationCurrentEnvCfg"
        ),
    },
)

gym.register(
    id="AnyMani-GM-SingleAsset-TactileRotation-CurrentObs-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.config.single_asset.tactile_rotation_env_cfg:GmTactileRotationCurrentEnvCfg_PLAY"
        ),
    },
)

gym.register(
    id="AnyMani-GM-SingleAsset-TactileRotation-History30Obs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.config.single_asset.tactile_rotation_env_cfg:GmTactileRotationHistory30EnvCfg"
        ),
    },
)

gym.register(
    id="AnyMani-GM-SingleAsset-TactileRotation-History30Obs-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.config.single_asset.tactile_rotation_env_cfg:GmTactileRotationHistory30EnvCfg_PLAY"
        ),
    },
)

gym.register(
    id="AnyMani-GM-Leap-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.leap.leap_env_cfg:GmLeapEnvCfg",
    },
)

gym.register(
    id="AnyMani-GM-Leap-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.leap.leap_env_cfg:GmLeapEnvCfg_PLAY",
    },
)

__all__ = [
    "GmLeapEnvCfg",
    "GmLeapEnvCfg_PLAY",
    "GmLeapSceneCfg",
    "GmSingleAssetEnvCfg",
    "GmSingleAssetEnvCfg_PLAY",
    "GmSingleAssetSceneCfg",
    "GmTactileRotationCurrentEnvCfg",
    "GmTactileRotationCurrentEnvCfg_PLAY",
    "GmTactileRotationHistory30EnvCfg",
    "GmTactileRotationHistory30EnvCfg_PLAY",
]


def __getattr__(name: str) -> Any:
    r"""Lazily expose GM env cfg classes without import-time Isaac Sim coupling.

    `tasks/gm/tests`是纯tensor/config contract tests，pytest收集时会先导入父包；这里保持lazy exports，
    避免在collection阶段加载single-asset/LEAP env、USD与Isaac runtime bindings。
    """

    if name in __all__:
        from .config.leap.leap_env_cfg import GmLeapEnvCfg, GmLeapEnvCfg_PLAY, GmLeapSceneCfg
        from .config.single_asset.single_asset_env_cfg import (
            GmSingleAssetEnvCfg,
            GmSingleAssetEnvCfg_PLAY,
            GmSingleAssetSceneCfg,
        )
        from .config.single_asset.tactile_rotation_env_cfg import (
            GmTactileRotationCurrentEnvCfg,
            GmTactileRotationCurrentEnvCfg_PLAY,
            GmTactileRotationHistory30EnvCfg,
            GmTactileRotationHistory30EnvCfg_PLAY,
        )
        exports = {
            "GmLeapEnvCfg": GmLeapEnvCfg,
            "GmLeapEnvCfg_PLAY": GmLeapEnvCfg_PLAY,
            "GmLeapSceneCfg": GmLeapSceneCfg,
            "GmSingleAssetEnvCfg": GmSingleAssetEnvCfg,
            "GmSingleAssetEnvCfg_PLAY": GmSingleAssetEnvCfg_PLAY,
            "GmSingleAssetSceneCfg": GmSingleAssetSceneCfg,
            "GmTactileRotationCurrentEnvCfg": GmTactileRotationCurrentEnvCfg,
            "GmTactileRotationCurrentEnvCfg_PLAY": GmTactileRotationCurrentEnvCfg_PLAY,
            "GmTactileRotationHistory30EnvCfg": GmTactileRotationHistory30EnvCfg,
            "GmTactileRotationHistory30EnvCfg_PLAY": GmTactileRotationHistory30EnvCfg_PLAY,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
