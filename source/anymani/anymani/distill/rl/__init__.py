r"""AnyMani distill 的 RL stage registry 与入口 package。

本包注册 distill 侧的训练任务别名，并拥有 ``python -m anymani.distill.rl.train`` 与
``python -m anymani.distill.rl.play`` 入口。环境语义仍由 `tasks/gm` 拥有，训练算法、
agent YAML、checkpoint 和日志路径由 `distill.rl` 拥有。除单资产 MLP probe 外，本 registry
也提供 tactile rotation 的 GRU/TCN 配对训练 alias。

Tactile rotation paired aliases:
    tasks 层注册 `CurrentObs` 与 `History30Obs` 两个环境语义 ID；本 registry 注册两个
    训练 alias：GRU alias 绑定 CurrentObs + GRU YAML，TCN alias 绑定 History30Obs + TCN YAML。
    `GRU` / `TCN` 名称不得进入 tasks 层 ID；环境不拥有网络结构。

    两个 alias 必须共享 seed protocol、4096 env、central critic schema、PPO optimizer、
    `horizon_length=30`、`minibatch_size=30720` 与 reward/ADR contract。YAML 分别维护，避免
    Hydra override 隐藏 temporal encoder 差异。tasks contract tests 固化两个 observation space，
    distill tensor tests 固化网络输入/输出。

未来 per-variant advantage、sampling 与 weighting 算法位于 ``rl/algorithms``；当前仅有
职责 scaffold。共享 adapter/backbone/head 必须来自 ``distill.models``，不得在 rl_games
adapter 内为 SSL/IL 复制另一套科研语义相同的网络。
"""

from __future__ import annotations

import gymnasium as gym

from anymani.tasks.gm.config.heterogeneous_asset import agents as heterogeneous_agents

from . import agents

gym.register(
    id="AnyMani-GM-SingleAsset-MLP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "anymani.tasks.gm.config.single_asset.single_asset_env_cfg:GmSingleAssetEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:gm_single_asset_mlp_ppo.yaml",
    },
)

gym.register(
    id="AnyMani-GM-Leap-MLP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "anymani.tasks.gm.config.leap.leap_env_cfg:GmLeapEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:gm_single_asset_mlp_ppo.yaml",
    },
)

gym.register(
    id="AnyMani-GM-SingleAsset-TactileRotation-GRU-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "anymani.tasks.gm.config.single_asset.tactile_rotation_env_cfg:GmTactileRotationCurrentEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:gm_tactile_rotation_gru_ppo.yaml",
    },
)

gym.register(
    id="AnyMani-GM-SingleAsset-TactileRotation-TCN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "anymani.tasks.gm.config.single_asset.tactile_rotation_env_cfg:GmTactileRotationHistory30EnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:gm_tactile_rotation_tcn_ppo.yaml",
    },
)

gym.register(
    id="AnyMani-GM-Canonical-Unified-PPO-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "anymani.tasks.gm.canonical_unified_env_cfg:CanonicalUnifiedInHandEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:gm_canonical_unified_ppo.yaml",
    },
)

gym.register(
    id="AnyMani-GM-HeterogeneousAsset-TactileRotation-PPO-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "anymani.tasks.gm.config.heterogeneous_asset.tactile_rotation_env_cfg:"
            "HeterogeneousTactileRotationEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:gm_heterogeneous_n000_ppo.yaml",
    },
)

gym.register(
    id="AnyMani-GM-HeterogeneousAsset-N040-History30-PPO-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "anymani.tasks.gm.config.heterogeneous_asset.tactile_rotation_env_cfg:"
            "HeterogeneousN040HistoryTactileRotationEnvCfg"
        ),
        "rl_games_cfg_entry_point": (
            f"{heterogeneous_agents.__name__}:gm_heterogeneous_n040_history30_ppo.yaml"
        ),
    },
)

__all__ = []
