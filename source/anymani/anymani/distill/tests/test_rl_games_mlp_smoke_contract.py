r"""Contract tests for the heterogeneous-hand MLP smoke PPO route.

这些测试不启动 Isaac Sim。它们只锁住本轮 MVP 的训练管线语义：

1. distill 注册了 `AnyMani-GM-Heterogeneous-MLP-Smoke-v0` 这个 rl_games 训练别名；
2. 该别名指向 `tasks/gm` 的异构环境语义和 distill 自己的 MLP YAML；
3. YAML 使用 rl_games 官方 `actor_critic` MLP，而不是 Transformer teacher builder；
4. rollout batch size 与 $3\times100$ envs 的 smoke 目标整除。
"""

from __future__ import annotations

from pathlib import Path

import anymani.distill.rl  # noqa: F401  # 注册 distill-owned Gym task aliases
import anymani.distill.rl.agents as agent_package
import gymnasium as gym
import yaml


def _mlp_agent_cfg() -> dict:
    r"""读取 MLP smoke YAML，避免测试依赖 Hydra / Isaac Sim 启动。"""

    yaml_path = Path(agent_package.__file__).with_name("gm_heterogeneous_mlp_ppo_smoke.yaml")
    with yaml_path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def test_heterogeneous_mlp_smoke_task_alias_points_to_distill_agent() -> None:
    r"""Gym alias 应把异构 env contract 与 MLP agent config 绑在一起。"""

    spec = gym.spec("AnyMani-GM-Heterogeneous-MLP-Smoke-v0")

    assert spec.kwargs["env_cfg_entry_point"].endswith(
        "gm_heterogeneous_mlp_smoke_env_cfg:HeterogeneousMlpSmokeEnvCfg"
    )
    assert spec.kwargs["rl_games_cfg_entry_point"].endswith("agents:gm_heterogeneous_mlp_ppo_smoke.yaml")


def test_heterogeneous_mlp_smoke_yaml_uses_tiny_builtin_mlp() -> None:
    r"""MLP feasibility route 必须使用 rl_games 内置小 MLP，而不是正式 Transformer teacher。"""

    agent_cfg = _mlp_agent_cfg()
    network_cfg = agent_cfg["params"]["network"]
    train_cfg = agent_cfg["params"]["config"]

    assert network_cfg["name"] == "actor_critic"
    assert network_cfg["separate"] is False
    assert network_cfg["mlp"]["units"] == [64, 64]
    assert train_cfg["name"] == "gm_heterogeneous_mlp_smoke_3x100"
    assert train_cfg["max_epochs"] == 2
    assert train_cfg["torch_compile"] is False


def test_heterogeneous_mlp_smoke_rollout_batch_matches_3x100_envs() -> None:
    r"""PPO rollout batch 应能被 minibatch 整除，避免第一步训练卡在 shape 配置。"""

    train_cfg = _mlp_agent_cfg()["params"]["config"]
    num_envs = 3 * 100
    batch_size = num_envs * train_cfg["horizon_length"]

    assert train_cfg["horizon_length"] == 8
    assert batch_size == 2400
    assert train_cfg["minibatch_size"] == 1200
    assert batch_size % train_cfg["minibatch_size"] == 0
