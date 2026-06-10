r"""Termination terms for `tasks.gm`."""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv


def object_falling_placeholder(env: ManagerBasedRLEnv, fall_dist: float) -> torch.Tensor:
    r"""Placeholder for object falling termination.

    TODO:
        正式实现应比较 object root position 与 task anchor position 的距离，
        而不是只看世界系 $z$。手内操作中手掌姿态、object spawn 偏置和 play
        可视化姿态都可能改变“掉落”的直观方向。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        fall_dist (float): 允许 object 偏离任务 anchor 的最大距离，单位 m。

    Returns:
        torch.Tensor: bool tensor，形状 `[num_envs]`。
    """

    _ = fall_dist
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


__all__ = [
    "object_falling_placeholder",
]
