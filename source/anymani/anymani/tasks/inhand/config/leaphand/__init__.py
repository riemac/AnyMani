# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LeapHand configuration for in-hand manipulation tasks.

环境变体:
    - AnyMani-LeapHand-Joint-v0: 关节空间动作（16 维）
    - AnyMani-LeapHand-Tactile-v0: 关节空间 + 触觉观测
"""

import gymnasium as gym

from . import agents

# 注册 gym 环境
from .leaphand_env_cfg import (
    # 训练配置
    LeapHandJointEnvCfg,
    LeapHandTactileEnvCfg,
    # Play 配置
    LeapHandJointEnvCfg_PLAY,
    LeapHandTactileEnvCfg_PLAY,
    # 场景配置
    LeapHandSceneCfg,
    LeapHandTactileSceneCfg,
    LeapHandFullTactileSceneCfg,
    # 观测配置
    TactileObservationsCfg,
)

##
# Register Gym environments.
##

# ===== 关节空间动作（Baseline）=====
gym.register(
    id="AnyMani-LeapHand-Joint-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leaphand_env_cfg:LeapHandJointEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="AnyMani-LeapHand-Joint-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leaphand_env_cfg:LeapHandJointEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

# ===== 触觉传感器（关节空间）=====
gym.register(
    id="AnyMani-LeapHand-Tactile-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leaphand_env_cfg:LeapHandTactileEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_tactile.yaml",
    },
)

gym.register(
    id="AnyMani-LeapHand-Tactile-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leaphand_env_cfg:LeapHandTactileEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_tactile.yaml",
    },
)