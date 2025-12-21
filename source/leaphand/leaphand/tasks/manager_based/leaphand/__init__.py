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
