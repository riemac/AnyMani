r"""Stability and regularization reward terms for GM in-hand manipulation."""

from __future__ import annotations

import isaaclab.envs.mdp as isaac_mdp
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .rewards_common import curriculum_gain


def action_l2_curriculum(
    env: ManagerBasedRLEnv,
    lambda_floor: float = 0.0,
    lambda_max: float = 1.0,
) -> torch.Tensor:
    r"""Curriculum-gated action L2 regularizer。

    本项目动作项 `ClampedRelativeJointPositionAction` 已将每步 raw rad delta 通过 `scale=0.1`
    约束在温和范围内，并在下发前 clamp 到 soft joint limits。因而第一版严格模仿 AnyRotate，
    把 action 正则放到 adaptive curriculum 后释放。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        lambda_floor (float): curriculum 早期下限，默认 0.0。
        lambda_max (float): curriculum 完全释放后的上限，默认 1.0。

    Returns:
        torch.Tensor: gated action L2 penalty source，形状 `[num_envs]`；外部配置负权重。
    """

    return isaac_mdp.action_l2(env) * curriculum_gain(env, lambda_floor=lambda_floor, lambda_max=lambda_max)


def action_rate_l2_curriculum(
    env: ManagerBasedRLEnv,
    lambda_floor: float = 0.0,
    lambda_max: float = 1.0,
) -> torch.Tensor:
    r"""Curriculum-gated action-rate L2 regularizer。

    该项惩罚相邻 policy action 的变化率，主要用于抑制高频抖动。由于相对增量动作本身
    已有限幅，默认也放入 curriculum；若训练早期仍出现动作爆炸，可单独把本项
    `lambda_floor` 调到 $0.02\sim0.1$。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        lambda_floor (float): curriculum 早期下限，默认 0.0。
        lambda_max (float): curriculum 完全释放后的上限，默认 1.0。

    Returns:
        torch.Tensor: gated action-rate L2 penalty source，形状 `[num_envs]`。
    """

    return isaac_mdp.action_rate_l2(env) * curriculum_gain(env, lambda_floor=lambda_floor, lambda_max=lambda_max)


def torque_l2_curriculum(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lambda_floor: float = 0.0,
    lambda_max: float = 1.0,
) -> torch.Tensor:
    r"""Curriculum-gated torque L2 penalty source。

    对齐 AnyRotate 的 torque penalty：
    $$
    r_{torque}=\|\tau\|_2^2.
    $$

    本函数返回正值 penalty source，实际惩罚由 `RewardsCfg` 负权重实现。若当前 actuator
    backend 没有暴露 `computed_torque`，返回 0 并保留接口。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        asset_cfg (SceneEntityCfg): robot articulation 配置。
        lambda_floor (float): curriculum 早期下限，默认 0.0。
        lambda_max (float): curriculum 完全释放后的上限，默认 1.0。

    Returns:
        torch.Tensor: gated torque L2 penalty source，形状 `[num_envs]`。
    """

    # 读取 articulation 的 controller torque；不同 actuator backend 可能不存在该字段。
    asset: Articulation = env.scene[asset_cfg.name]
    torque = getattr(asset.data, "computed_torque", None)  # `[B,d]`，控制器计算力矩，单位 N·m
    if torque is None:
        return torch.zeros(env.num_envs, device=env.device)  # 保留接口，不因 backend 差异中断脚手架

    penalty = torch.sum(torque**2, dim=-1)  # `[B]`，$\|\tau\|_2^2$
    return penalty * curriculum_gain(env, lambda_floor=lambda_floor, lambda_max=lambda_max)


__all__ = ["action_l2_curriculum", "action_rate_l2_curriculum", "torque_l2_curriculum"]
