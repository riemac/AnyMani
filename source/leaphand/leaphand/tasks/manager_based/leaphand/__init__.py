# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LeapHand连续旋转任务 - ManagerBasedRLEnv架构"""

import gymnasium as gym

from . import agents
from .inhand_base_env_cfg import InHandObjectEnvCfg
from .inhand_se3_env_cfg import InHandse3EnvCfg
from .inhand_affine_env_cfg import InHandAffineEnvCfg
from .inhand_float_env_cfg import InHandFloatEnvCfg
from .inhand_rma_env_cfg import InHandRmaEnvCfg
from .inhand_round_base_env_cfg import InHandObjectEnvCfg
from .inhand_tactile_env_cfg import InHandTactileSceneCfg

##
# Register Gym environments.
##

gym.register(
    id="Template-Leaphand-Rot-Manager-v0", # Template可被list_envs.py识别（但不影响环境注册与训练）
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.inhand_base_env_cfg:InHandObjectEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Template-Leaphand-se3-Rot-Manager-v0", # Template可被list_envs.py识别（但不影响环境注册与训练）
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.inhand_se3_env_cfg:InHandse3EnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_se3.yaml",
    },
)

# 仿射编队对比实验环境
gym.register(
    id="Template-Leaphand-Affine-Manager-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.inhand_affine_env_cfg:InHandAffineEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",  # 复用baseline的训练配置
    },
)

# 浮动基座
gym.register(
    id="Template-Leaphand-Float-Manager-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.inhand_float_env_cfg:InHandFloatEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_float.yaml",  # 复用baseline的训练配置
    },
)

# 圆形指尖
gym.register(
    id="Template-Leaphand-RoundTip-Manager-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.inhand_round_base_env_cfg:InHandObjectEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",  # 复用baseline的训练配置
    },
)

# RMA (Rapid Motor Adaptation) 专用环境：与 baseline/float/round-tip 等配置隔离
gym.register(
    id="Template-Leaphand-RMA-Manager-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.inhand_rma_env_cfg:InHandRmaEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_rma.yaml",
    },
)

# 触觉环境
gym.register(
    id="Template-Leaphand-Tactile-Manager-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.inhand_tactile_env_cfg:InHandTactileEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_tactile.yaml",
    },
)