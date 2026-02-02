# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LeapHand configuration for in-hand manipulation tasks."""

import gymnasium as gym

from . import agents
from .inhand_base_env_cfg import InHandObjectEnvCfg
from .inhand_se3_env_cfg import InHandse3EnvCfg
from .inhand_affine_env_cfg import InHandAffineEnvCfg
from .inhand_float_env_cfg import InHandFloatEnvCfg
from .inhand_rma_env_cfg import InHandRmaEnvCfg
from .inhand_tactile_env_cfg import InHandTactileEnvCfg
from .inhand_se3_tactile_env_cfg import InHandSe3TactileEnvCfg

##
# Register Gym environments.
##

# Baseline: 关节空间动作
gym.register(
    id="AnyMani-LeapHand-Joint-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.inhand_base_env_cfg:InHandObjectEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

# SE(3) 动作空间
gym.register(
    id="AnyMani-LeapHand-SE3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.inhand_se3_env_cfg:InHandse3EnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_se3.yaml",
    },
)

# 仿射编队动作空间
gym.register(
    id="AnyMani-LeapHand-Affine-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.inhand_affine_env_cfg:InHandAffineEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

# 浮动基座（臂手解耦实验）
gym.register(
    id="AnyMani-LeapHand-Float-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.inhand_float_env_cfg:InHandFloatEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_float.yaml",
    },
)

# RMA (Rapid Motor Adaptation)
gym.register(
    id="AnyMani-LeapHand-RMA-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.inhand_rma_env_cfg:InHandRmaEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_rma.yaml",
    },
)

# 触觉传感器
gym.register(
    id="AnyMani-LeapHand-Tactile-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.inhand_tactile_env_cfg:InHandTactileEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_tactile.yaml",
    },
)

# SE(3) + 触觉
gym.register(
    id="AnyMani-LeapHand-SE3-Tactile-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.inhand_se3_tactile_env_cfg:InHandSe3TactileEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_se3_tactile.yaml",
    },
)


# 兼容旧名称（deprecated，后续版本移除）
gym.register(
    id="Template-Leaphand-Rot-Manager-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.inhand_base_env_cfg:InHandObjectEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)