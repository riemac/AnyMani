r"""GM tactile rotation GRU/TCN rl_games 配置的纯 YAML contract tests。

这些测试不启动 Isaac Sim。它们把 temporal actor 对照中容易静默分叉的配置锁成可执行
命题：PPO transition budget 相同、GRU 当前帧旁路正确、TCN history 不被误当成 RNN
sequence，以及 privileged central critic 独立训练。
"""

from __future__ import annotations

from pathlib import Path

import anymani.distill.rl.agents as agent_package
import torch
import yaml
from anymani.distill.rl.rl_games_backend import prefer_local_rl_games

prefer_local_rl_games(strict=True)

from rl_games.algos_torch import network_builder  # noqa: E402


def _agent_cfg(filename: str) -> dict:
    r"""读取 distill-owned rl_games agent YAML。

    Args:
        filename (str): `distill/rl/agents` 下的 YAML 文件名。

    Returns:
        dict: `yaml.safe_load` 后的完整 rl_games 配置。
    """

    yaml_path = Path(agent_package.__file__).with_name(filename)  # 与 installed package resource 路径一致
    with yaml_path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)  # 纯配置解析，不 import IsaacLab/Kit


def test_gru_and_tcn_share_ppo_and_central_critic_budget() -> None:
    r"""两条 actor route 只允许 temporal architecture 不同，优化预算必须相同。"""

    gru = _agent_cfg("gm_tactile_rotation_gru_ppo.yaml")["params"]
    tcn = _agent_cfg("gm_tactile_rotation_tcn_ppo.yaml")["params"]

    shared_fields = (
        "horizon_length",
        "minibatch_size",
        "mini_epochs",
        "learning_rate",
        "gamma",
        "tau",
        "e_clip",
        "grad_norm",
        "entropy_coef",
    )
    for field in shared_fields:
        assert gru["config"][field] == tcn["config"][field]  # 同一 PPO sampling/optimizer budget

    assert gru["seed"] == tcn["seed"] == 42
    assert gru["config"]["horizon_length"] == 30
    assert gru["config"]["minibatch_size"] == 30720
    assert 4096 * 30 == 4 * 30720  # 每 iteration 122880 transitions，固定 4 minibatches

    for params in (gru, tcn):
        train_cfg = params["config"]
        central_cfg = train_cfg["central_value_config"]
        assert train_cfg["use_experimental_cv"] is False  # 主 actor value head 不进入 loss
        assert central_cfg["minibatch_size"] == train_cfg["minibatch_size"]
        assert central_cfg["network"]["central_value"] is True
        assert central_cfg["network"]["mlp"]["units"] == [512, 256, 128]
        assert central_cfg["normalize_input"] is True  # privileged state 独立 RMS


def test_gru_yaml_exposes_current_frame_to_policy_mlp() -> None:
    r"""GRU 必须计算 $MLP([h_t,x_t])$，不能把 `concat_input` 误当成 current skip。"""

    params = _agent_cfg("gm_tactile_rotation_gru_ppo.yaml")["params"]
    rnn = params["network"]["rnn"]

    assert params["network"]["name"] == "actor_critic"
    assert rnn == {
        "name": "gru",
        "units": 256,
        "layers": 1,
        "layer_norm": False,
        "before_mlp": True,
        "concat_input": False,
        "concat_output": True,
    }
    assert params["config"]["seq_length"] == 30  # 1.5 s BPTT，不是 env history stack
    assert params["config"]["zero_rnn_on_done"] is True
    assert params["config"]["normalize_input"] is False  # 与 TCN 使用同一物理归一 actor contract


def test_tcn_yaml_uses_env_history_and_registered_builder() -> None:
    r"""TCN route 应消费 `[30,52]` env history，并保持 rl_games 非 RNN。"""

    params = _agent_cfg("gm_tactile_rotation_tcn_ppo.yaml")["params"]
    network = params["network"]
    tcn = network["tactile_tcn"]

    assert network["name"] == "anymani_tactile_tcn"
    assert tcn["frame_dim"] == 52
    assert tcn["latent_dim"] == 64
    assert tcn["hidden_channels"] == [64, 64, 64]
    assert tcn["kernels"] == [9, 5, 5]
    assert tcn["strides"] == [2, 1, 1]
    assert tcn["mlp_units"] == [512, 256, 128]
    assert params["config"]["seq_length"] == 1  # history 属于 observation，不由 rl_games sequence 产生
    assert params["config"]["normalize_input"] is False


def test_native_rl_games_gru_builds_308d_current_frame_skip() -> None:
    r"""本地 rl_games 应把 256D hidden 与当前 52D frame 拼成 308D MLP 输入。

    该测试直接执行当前固定 commit 的 `A2CBuilder`，不只检查 YAML 字段。若未来 rl_games
    改变 `before_mlp/concat_output` 语义，这里会在训练启动前失败。
    """

    params = _agent_cfg("gm_tactile_rotation_gru_ppo.yaml")["params"]
    builder = network_builder.A2CBuilder()  # rl_games 原生 recurrent actor builder
    builder.load(params["network"])  # 注入 `[GRU256 -> concat x_t -> MLP]` 配置
    network = builder.build(
        "tactile_gru_contract",
        actions_num=16,
        input_shape=(52,),
        value_size=1,
        num_seqs=2,
    )

    first_linear = next(module for module in network.actor_mlp.modules() if isinstance(module, torch.nn.Linear))
    assert first_linear.in_features == 256 + 52  # `[h_t,x_t]`，而不是只把 hidden 送入 MLP

    obs = torch.randn(60, 52)  # 2 条完整 30-step sequences，rl_games flatten batch contract
    hidden = (torch.zeros(1, 2, 256),)  # 单层 GRU hidden `[layers,num_seqs,hidden]`
    mu, logstd, value, states = network(
        {"obs": obs, "seq_length": 30, "rnn_states": hidden, "dones": torch.zeros(60, 1)}
    )

    assert mu.shape == (60, 16)
    assert logstd.shape == (60, 16)
    assert value.shape == (60, 1)
    assert isinstance(states, tuple) and states[0].shape == (1, 2, 256)
