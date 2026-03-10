# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""LeapHand与动作相关的奖励函数。主要由动作及其导数（如加速度、加加度）决定，用于约束策略的行为流形（Behavior Manifold），使其符合物理可行性或特定风格。

简言之，该奖励文件决定 “怎么做”，是通用（Task-Agnostic）的。

主要包括动作平滑性、能量消耗等。

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def torque_l2_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """关节力矩平方和，用于约束动作的用力大小。"""

    asset: Articulation = env.scene[asset_cfg.name]
    torque = getattr(asset.data, "computed_torque", None)

    if torque is None:
        return torch.zeros(env.num_envs, device=env.device)

    return torch.sum(torque ** 2, dim=-1)

