r"""Termination-related reward placeholders for GM in-hand manipulation."""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv


def termination_penalty_placeholder(env: ManagerBasedRLEnv) -> torch.Tensor:
    r"""Termination penalty 占位项。

    AnyRotate 将 object falling / rotation-axis deviation 等终止条件写成 `r_terminate`。
    在 Isaac Lab ManagerBasedRLEnv 中，更干净的做法通常是：终止逻辑放在
    `terminations.py`，reward 侧只在需要时读取 termination term 的 bool indicator 并乘负权重。

    TODO: 等 `gm/mdp/terminations.py` 的掉落、离手、axis deviation 判据稳定后，再决定是否
    需要显式 penalty reward；不要提前把 termination 语义复制到 reward 里形成双源漂移。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。

    Returns:
        torch.Tensor: 全零 penalty source，形状 `[num_envs]`。
    """

    return torch.zeros(env.num_envs, device=env.device)  # 当前仅占位，不改变训练语义


__all__ = ["termination_penalty_placeholder"]
