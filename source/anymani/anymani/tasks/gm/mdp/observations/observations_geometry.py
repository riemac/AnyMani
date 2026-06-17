r"""Geometry observation terms for GM in-hand manipulation.

DONE(geometry obs - joint limits): 关节限位作为静态形态量 (morphology feature)。

符号沿用 state obs：$q_i^{\min}, q_i^{\max}$ 为第 $i$ 个关节的 soft 限位 (rad)。

NOTE(路线 B 决策，已与用户对齐): limits 归 geometry / 形态，而非 state obs。
因为 $q_i^{\min}, q_i^{\max}$ 是时间常量，若混进 state obs 并启用 history_length
$H>1$，会在时间窗口里被无意义地重复堆叠。geometry 小组本身倾向交由
`distill` 接管，故 limits 在这里以“不进时间历史的静态特征”形式提供，最终可挂到
joint-centric token 上。

当前提供：

- `q_min, q_max`: `asset.data.soft_joint_pos_limits[:, joint_ids, 0/1]`，单位 rad；
- TODO: 可选派生 margin：
  $$
  \text{margin}_i^{\text{lo}} = q_i - q_i^{\min},\qquad
  \text{margin}_i^{\text{hi}} = q_i^{\max} - q_i.
  $$

NOTE(soft vs hard limits): 必须用 **soft** limits 而非 hard。soft 才是 actuator 实际
clamp、策略真正会撞到的行为边界。

边界：其他静态几何特征，如 link lengths / mount poses、tip mesh 描述符、连杆 mesh
几何特征、palm/global scale 等，由 `distill/rl/geo_obs.py` 接管。本模块只保留浅契约。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_soft_pos_limits(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    r"""读取 runtime soft joint limits $[q^{\min}, q^{\max}]$。

    limits 在语义上属于 geometry / morphology，而非 state；但 teacher RL 应读取
    runtime 的 `soft_joint_pos_limits`，因为它们正是 action clamp 与 actuator 行为
    实际使用的边界：
    $$
    L_i = [q_i^{\min}, q_i^{\max}] \in \mathbb{R}^{2}.
    $$

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        asset_cfg (SceneEntityCfg): robot articulation 与 joint 子集配置。

    Returns:
        torch.Tensor: soft limits，形状 `[num_envs, 2 * num_joints]`，单位 rad。

    NOTE:
        返回时 flatten 最后一维，是为了兼容 `ObsGroup.concatenate_terms=True` 的扁平 obs dict；
        tokenizer 侧若需要 `[B,N_j,2]`，应在 adapter 中按 joint 数 reshape 回结构化形式。
    """

    # 用 soft 而非 hard limits：soft 才是 `ClampedRelativeJointPositionAction` clamp 的同一边界。
    asset: Articulation = env.scene[asset_cfg.name]
    limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, :]  # `[B,N_j,2]`，rad
    return limits.flatten(start_dim=1)  # `[B,2*N_j]`，便于当前 rl_games 扁平 obs 消费


__all__ = ["joint_soft_pos_limits"]
