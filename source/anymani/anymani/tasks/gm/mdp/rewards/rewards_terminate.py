r"""Termination-related reward placeholders for GM in-hand manipulation.

TODO(tactile rotation impulse semantics):
    新 baseline 的 object-out-of-anchor 与 signed axis-deviation termination 由 termination manager
    定义，reward 侧只消费同一 bool 结果，不能复制判据。termination penalty 是离散 impulse；
    callable 应返回 indicator 除以 `env.step_dt`，使 RewardManager 积分后每次失败贡献固定的
    `-50`，不随 20/30 Hz 控制频率变化。
"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv

from .rewards_common import impulse_to_rate


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


def failure_termination_impulse(
    env: ManagerBasedRLEnv,
    termination_term_names: tuple[str, ...],
) -> torch.Tensor:
    r"""复用 TerminationManager 已计算的 failure bits，返回一次固定 impulse rate。

    `termination_term_names` 应只含 anchor/axis failure，不含 timeout；多个 failure 在同一步
    同时触发时仍只贡献一次 indicator，避免双重惩罚同一个 episode boundary。
    """

    failed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for term_name in termination_term_names:
        failed |= env.termination_manager.get_term(term_name)  # 只消费同一 manager snapshot，不复制判据
    return impulse_to_rate(failed.float(), env.step_dt)


__all__ = ["failure_termination_impulse", "termination_penalty_placeholder"]
