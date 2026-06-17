r"""Distill-owned MLP smoke cfg for the default GM in-hand task.

本文件不重新定义 MDP。`tasks/gm/inhand_env_cfg.py` 已经拥有 hand asset binding、
scene、obs、action、command、reward、reset、termination 与 curriculum；这里位于
`distill/rl`，只表达训练管线选择：用 rl_games 内置 `actor_critic` MLP 跑当前
GM 默认 hand selection 与 env-per-hand routing。

该 smoke 的科研含义很窄：验证 flat obs/action 与 PPO rollout/backward/checkpoint
闭环能吃下当前默认同拓扑 generated hand selection。它不评价最终策略质量，也不引入
Transformer、mesh encoder 或 token representation。
"""

from __future__ import annotations

from anymani.tasks.gm.inhand_env_cfg import GM_DEFAULT_ENVS_PER_HAND, GM_DEFAULT_NUM_ENVS, GmInHandEnvCfg
from isaaclab.utils import configclass

GM_INHAND_MLP_SMOKE_HAND_COUNT = GM_DEFAULT_NUM_ENVS // GM_DEFAULT_ENVS_PER_HAND
r"""MLP smoke 使用的 hand asset 数，由 `tasks/gm` 默认 env 规模反推得到。"""

GM_INHAND_MLP_SMOKE_ENVS_PER_HAND = GM_DEFAULT_ENVS_PER_HAND
r"""每个 hand asset 的 env 数，保持与 `tasks/gm` 默认 round-robin routing 一致。"""

GM_INHAND_MLP_SMOKE_NUM_ENVS = GM_DEFAULT_NUM_ENVS
r"""MLP smoke 默认总并行环境数，跟随 `tasks/gm` 的当前 hand selection preset。"""

GM_INHAND_MLP_SMOKE_EPISODE_LENGTH_S = 10.0
r"""训练 smoke 的短 episode 长度，优先暴露 reset/contact/reward contract 问题。"""


@configclass
class GmInHandMlpSmokeEnvCfg(GmInHandEnvCfg):
    r"""GM in-hand 默认 hand selection 的简单 MLP PPO smoke 环境。

    继承关系刻意保持单向：`tasks/gm` 给出环境语义，本类只在 `distill` 内给
    rl_games alias 一个稳定入口。网络仍由 YAML 的 `actor_critic` MLP 决定，避免把
    训练算法或网络结构写回 `tasks/gm`。
    """

    def __post_init__(self) -> None:
        r"""保持 GM 默认并行规模，并缩短 episode 方便 smoke 统计。"""

        super().__post_init__()
        self.scene.num_envs = GM_INHAND_MLP_SMOKE_NUM_ENVS  # 与 GM 默认 hand routing 合同一致
        self.scene.replicate_physics = False  # 多 URDF prototype 的 batched scene 必须保持非 replicate physics
        self.episode_length_s = GM_INHAND_MLP_SMOKE_EPISODE_LENGTH_S  # smoke 阶段优先快速发现 reset/reward 问题
        self.commands.goal_pose.debug_vis = True  # GUI smoke 需要显示 command-owned 虚拟目标物体


__all__ = [
    "GM_INHAND_MLP_SMOKE_ENVS_PER_HAND",
    "GM_INHAND_MLP_SMOKE_EPISODE_LENGTH_S",
    "GM_INHAND_MLP_SMOKE_HAND_COUNT",
    "GM_INHAND_MLP_SMOKE_NUM_ENVS",
    "GmInHandMlpSmokeEnvCfg",
]
