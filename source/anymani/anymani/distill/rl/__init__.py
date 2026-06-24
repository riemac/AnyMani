r"""RL training registry for AnyMani distill.

本包只注册 distill 侧的训练任务别名：环境语义仍由 `tasks/gm` 拥有，训练算法、
agent YAML、checkpoint 和日志路径由 `distill` 拥有。当前正式主线是单资产 MLP
MDP probe，用它先验证 generated asset、reset、reward、obs/action 与 PhysX 接触闭环。
"""

from __future__ import annotations

import gymnasium as gym

from . import agents

gym.register(
    id="AnyMani-GM-SingleAsset-MLP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "anymani.tasks.gm.single_asset_env_cfg:GmSingleAssetEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:gm_single_asset_mlp_ppo.yaml",
    },
)

__all__ = []
