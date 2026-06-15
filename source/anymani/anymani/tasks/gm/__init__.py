r"""Generalized manipulation task family for AnyMani.

本包是新一代“手型泛化的手内操作”任务环境入口。

当前处于 design scaffold 阶段：这里先稳定 package 边界和研究语义，
暂不注册一个会被误认为可直接训练的 Gym 环境。正式注册应等到
`GmInHandEnvCfg` 至少满足以下条件后再加入：

- 能把一个已选定的 generated hand asset 绑定为 `scene.robot`；
- action joint order 与该 asset 的 same-topology contract 对齐；
- policy / critic observation schema 在 `distill` 侧可被记录到训练 manifest；
- random agent smoke test 可在 Isaac Lab 中启动并完成若干步。

NOTE:
    `tasks` 只定义环境；`distill` 负责训练入口、rl_games 配置、asset-bank
    split、checkpoint、rollout dataset 和模型结构。
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
