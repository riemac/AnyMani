r"""2048×1/2048×2 heterogeneous PPO YAML 与 central critic 的纯合同。"""

from __future__ import annotations

from pathlib import Path

import anymani.distill.rl.agents as agent_package
import torch
import yaml
from anymani.distill.rl.heterogeneous_masked_ppo import (
    HETEROGENEOUS_CRITIC_OBS_DIM,
    HETEROGENEOUS_MASKED_OBS_DIM,
    HETEROGENEOUS_N000_FRAME_DIM,
    HETEROGENEOUS_ROUTE_DIM,
)
from rl_games.algos_torch import model_builder


def _agent_cfg() -> dict:
    r"""读取 installed-package-compatible heterogeneous rl_games YAML。"""

    yaml_path = Path(agent_package.__file__).with_name("gm_heterogeneous_n000_ppo.yaml")
    with yaml_path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)  # 纯配置解析，不 import IsaacLab/Kit


def test_heterogeneous_actor_and_central_critic_schema() -> None:
    r"""actor 只学习 52D N000 frame，17D routing metadata 与 103D critic 必须显式分离。"""

    params = _agent_cfg()["params"]
    network = params["network"]
    train = params["config"]
    central = train["central_value_config"]

    assert HETEROGENEOUS_N000_FRAME_DIM == 52
    assert HETEROGENEOUS_ROUTE_DIM == 17
    assert HETEROGENEOUS_MASKED_OBS_DIM == 69
    assert HETEROGENEOUS_CRITIC_OBS_DIM == 103
    assert params["algo"]["name"] == "anymani_masked_ppo"
    assert params["model"]["name"] == "anymani_masked_continuous"
    assert network["name"] == "anymani_heterogeneous_n000_masked"
    assert network["heterogeneous_policy"]["geometry_entity_width"] == 128
    assert network["heterogeneous_policy"]["joint_feature_dim"] == 3
    assert network["heterogeneous_policy"]["owner_feature_dim"] == 1
    assert train["normalize_input"] is False  # asset row 不进入 running mean/std
    assert train["use_experimental_cv"] is False
    assert central["network"]["central_value"] is True
    assert central["network"]["mlp"]["units"] == [512, 256, 128]
    assert central["normalize_input"] is True


def test_infrastructure_ppo_budget_divides_2048x1_and_2048x2_batches() -> None:
    r"""同一 preset 必须无需隐式修正即可覆盖 2048 与 4096 environments。"""

    params = _agent_cfg()["params"]
    train = params["config"]
    horizon = train["horizon_length"]  # 每个 env 每轮 rollout transitions 数 $H=16$
    minibatch = train["minibatch_size"]  # actor/critic 共同使用 $M=4096$

    assert horizon == 16
    assert minibatch == 4096
    assert train["mini_epochs"] == 2
    assert train["central_value_config"]["minibatch_size"] == minibatch
    assert train["central_value_config"]["mini_epochs"] == train["mini_epochs"]
    assert (2048 * horizon) % minibatch == 0
    assert (4096 * horizon) % minibatch == 0
    assert float(params["env"]["clip_observations"]) > 2047.0  # asset row 不能被 wrapper clamp


def test_native_central_value_network_consumes_103d_state_and_backpropagates() -> None:
    r"""rl_games 原生 asymmetric critic 必须实际 build/forward/backward `[B,103]`。"""

    central = dict(_agent_cfg()["params"]["config"]["central_value_config"])
    central["model"] = {"name": "central_value"}  # A2CBase.load_networks runtime 注入的同一默认字段
    network = model_builder.ModelBuilder().load(central)
    model = network.build(
        {
            "value_size": 1,
            "input_shape": (HETEROGENEOUS_CRITIC_OBS_DIM,),
            "actions_num": 16,
            "num_agents": 1,
            "num_seqs": 4,
            "normalize_input": True,
            "normalize_value": True,
        }
    )
    output = model({"obs": torch.randn(4, HETEROGENEOUS_CRITIC_OBS_DIM), "is_train": True})
    values = output["values"]  # `[B,1]` central value prediction
    loss = values.square().mean()
    loss.backward()

    assert values.shape == (4, 1)
    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
