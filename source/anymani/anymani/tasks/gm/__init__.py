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

from .inhand_env_cfg import GmInHandEnvCfg, GmInHandEnvCfg_PLAY, GmInHandSceneCfg

__all__ = [
    "GmInHandEnvCfg",
    "GmInHandEnvCfg_PLAY",
    "GmInHandSceneCfg",
]
