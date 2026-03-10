# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LeapHand Round Tip configuration for in-hand manipulation.

This configuration uses LeapHand with hemispherical fingertips.
"""

import gymnasium as gym

from . import agents
from .inhand_round_base_env_cfg import InHandObjectEnvCfg as LeapHandRoundEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="AnyMani-LeapHand-RoundTip-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.inhand_round_base_env_cfg:InHandObjectEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
