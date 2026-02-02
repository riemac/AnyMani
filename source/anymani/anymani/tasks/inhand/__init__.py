# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""In-hand manipulation tasks for AnyMani.

This module contains in-hand object manipulation tasks using dexterous hands.

MDP 组件库:
    - inhand_env_cfg: 可复用的 MDP 配置组件（Obs/Actions/Rewards 等）

手型配置:
    - config/leaphand/: LeapHand 配置
    - config/leaphand_round/: 半球指尖 LeapHand 配置
"""

from isaaclab_tasks.utils import import_packages

# 导出 MDP 组件库
from .inhand_env_cfg import (
    # 场景
    InHandObjectSceneCfg,
    TactileSceneCfg,
    # 观测
    JointSpaceObsGroupCfg,
    ProprioceptionObsGroupCfg,
    Se3ObsGroupCfg,
    JointSpaceObservationsCfg,
    Se3ObservationsCfg,
    TactileObsGroupCfg,
    TactileCriticObsGroupCfg,
    RmaPrivInfoObsGroupCfg,
    RmaProprioHistObsGroupCfg,
    # 动作
    JointSpaceActionsCfg,
    Se3ActionsCfg,
    AffineActionsCfg,
    # 奖励
    CommonRewardsCfg,
    Se3RewardsCfg,
    TactileRewardsCfg,
    # 事件
    CommonEventCfg,
    # 终止
    CommonTerminationsCfg,
    # 命令
    ContinuousRotationCommandsCfg,
    # 课程
    EmptyCurriculumCfg,
)

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils", ".mdp"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
