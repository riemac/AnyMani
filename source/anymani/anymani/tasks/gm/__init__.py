r"""Generalized manipulation task family for AnyMani.

本包是新一代“手型泛化的手内操作”任务环境入口。`gm` 只注册 Isaac Lab
环境语义：scene、obs、action、command、reward、reset、termination。训练算法、
rl_games YAML、checkpoint、rollout dataset 与网络结构仍由 `distill` 消费本包后
自包含维护。

当前主环境 `AnyMani-GM-InHand-v0` 绑定的是 `inhand_env_cfg.GmInHandEnvCfg`：
它使用 same-topology post-mutate hand selection，并按默认 env-per-hand routing
给出 teacher RL 并行规模。该注册只表达任务默认 contract；正式实验仍应在
`distill` 侧记录 asset selection、训练 seed 与网络配置。
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym

gym.register(
    id="AnyMani-GM-Heterogeneous-Test-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.heterogeneous_test_env_cfg:HeterogeneousHandTestEnvCfg",
    },
)

gym.register(
    id="AnyMani-GM-InHand-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.inhand_env_cfg:GmInHandEnvCfg",
    },
)

gym.register(
    id="AnyMani-GM-InHand-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.inhand_env_cfg:GmInHandEnvCfg_PLAY",
    },
)

__all__ = [
    "GmInHandEnvCfg",
    "GmInHandEnvCfg_PLAY",
    "GmInHandSceneCfg",
    "HeterogeneousHandTestEnvCfg",
]


def __getattr__(name: str) -> Any:
    r"""Lazily expose GM env cfg classes without import-time Isaac Sim coupling.

    `tasks/gm/tests` 是纯 tensor / config contract tests，pytest 收集时会先导入
    `anymani.tasks.gm` 父包。如果这里急切导入 `inhand_env_cfg`，就会把测试收集
    拖进 IsaacLab env / USD binding。实际训练和 smoke 均直接导入
    `anymani.tasks.gm.inhand_env_cfg`，不依赖父包急切 re-export。
    """

    if name in __all__:
        if name == "HeterogeneousHandTestEnvCfg":
            from .heterogeneous_test_env_cfg import HeterogeneousHandTestEnvCfg

            return HeterogeneousHandTestEnvCfg

        from .inhand_env_cfg import GmInHandEnvCfg, GmInHandEnvCfg_PLAY, GmInHandSceneCfg

        exports = {
            "GmInHandEnvCfg": GmInHandEnvCfg,
            "GmInHandEnvCfg_PLAY": GmInHandEnvCfg_PLAY,
            "GmInHandSceneCfg": GmInHandSceneCfg,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
