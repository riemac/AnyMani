# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LeapHand configuration for in-hand manipulation tasks.

环境变体:
    - AnyMani-LeapHand-Joint-v0: 关节空间动作（16 维）
    - AnyMani-LeapHand-SE3-v0: SE(3) 旋量动作（24 维）
    - AnyMani-LeapHand-Tactile-v0: 关节空间 + 触觉观测
    - AnyMani-LeapHand-SE3-Tactile-v0: SE(3) + 触觉观测
    - AnyMani-LeapHand-Affine-v0: 仿射编队动作（9 维）

说明：LeapHand 的环境配置已从分散的 `inhand_*_env_cfg.py` 整合到 `leaphand_env_cfg.py`。
本模块负责：
1) 重新导出配置类，方便 `from ...config.leaphand import ...`
2) 注册 gym 环境 id -> env_cfg_entry_point
"""

import gymnasium as gym

from . import agents
from .leaphand_env_cfg import (
    LeapHandAffineEnvCfg,
    LeapHandAffineEnvCfg_PLAY,
    LeapHandJointEnvCfg,
    LeapHandJointEnvCfg_PLAY,
    LeapHandSe3EnvCfg,
    LeapHandSe3EnvCfg_PLAY,
    LeapHandSe3TactileEnvCfg,
    LeapHandSe3TactileEnvCfg_PLAY,
    LeapHandTactileEnvCfg,
    LeapHandTactileEnvCfg_PLAY,
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

# ===== SE(3) 动作空间 =====
gym.register(
    id="AnyMani-LeapHand-SE3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leaphand_env_cfg:LeapHandSe3EnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_se3.yaml",
    },
)

gym.register(
    id="AnyMani-LeapHand-SE3-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leaphand_env_cfg:LeapHandSe3EnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_se3.yaml",
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

# ===== SE(3) + 触觉 =====
gym.register(
    id="AnyMani-LeapHand-SE3-Tactile-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leaphand_env_cfg:LeapHandSe3TactileEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_se3_tactile.yaml",
    },
)

gym.register(
    id="AnyMani-LeapHand-SE3-Tactile-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leaphand_env_cfg:LeapHandSe3TactileEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_se3_tactile.yaml",
    },
)

# ===== 仿射编队动作空间 =====
gym.register(
    id="AnyMani-LeapHand-Affine-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leaphand_env_cfg:LeapHandAffineEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="AnyMani-LeapHand-Affine-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leaphand_env_cfg:LeapHandAffineEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

# ===== 兼容旧名称（deprecated，后续版本移除）=====
gym.register(
    id="Template-Leaphand-Rot-Manager-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leaphand_env_cfg:LeapHandJointEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
