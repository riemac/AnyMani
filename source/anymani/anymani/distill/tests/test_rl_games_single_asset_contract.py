r"""Contract tests for the single-asset MLP PPO training route.

这些测试不启动 Isaac Sim。它们只验证 distill 侧的训练管线声明：single-asset MDP
probe 使用独立训练入口、绑定 single-asset env cfg、采用 rl_games 内置 MLP，且默认
2048/4096 env 的 PPO batch 可以被 minibatch 整除。
"""

from __future__ import annotations

from pathlib import Path

import anymani.distill.rl  # noqa: F401  # 注册 distill-owned Gym aliases
import anymani.distill.rl.agents as agent_package
import gymnasium as gym
import yaml

TRAIN_ENTRY_PATH = Path(__file__).resolve().parents[1] / "train_mlp_single_asset.py"
"""单资产 MLP 独立训练入口源码路径。"""


def _single_asset_agent_cfg() -> dict:
    r"""读取 single-asset MLP YAML，不依赖 Hydra / Isaac Sim 启动。"""

    yaml_path = Path(agent_package.__file__).with_name("gm_single_asset_mlp_ppo.yaml")
    with yaml_path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def test_single_asset_mlp_task_alias_points_to_distill_agent() -> None:
    r"""distill Gym alias 应把 single-asset env cfg 与 MLP YAML 绑定。"""

    spec = gym.spec("AnyMani-GM-SingleAsset-MLP-v0")

    assert spec.kwargs["env_cfg_entry_point"].endswith("single_asset_env_cfg:GmSingleAssetEnvCfg")
    assert spec.kwargs["rl_games_cfg_entry_point"].endswith("agents:gm_single_asset_mlp_ppo.yaml")


def test_single_asset_mlp_play_alias_enables_goal_marker_env() -> None:
    r"""回放 checkpoint 时应使用 PLAY env cfg，同时仍加载同一个 MLP PPO YAML。"""

    spec = gym.spec("AnyMani-GM-SingleAsset-MLP-Play-v0")

    assert spec.kwargs["env_cfg_entry_point"].endswith("single_asset_env_cfg:GmSingleAssetEnvCfg_PLAY")
    assert spec.kwargs["rl_games_cfg_entry_point"].endswith("agents:gm_single_asset_mlp_ppo.yaml")


def test_single_asset_mlp_yaml_uses_builtin_mlp_not_transformer() -> None:
    r"""单资产 MDP probe 必须使用 rl_games 内置 MLP，而不是正式 Transformer teacher。"""

    agent_cfg = _single_asset_agent_cfg()
    network_cfg = agent_cfg["params"]["network"]
    train_cfg = agent_cfg["params"]["config"]

    assert network_cfg["name"] == "actor_critic"
    assert network_cfg["mlp"]["units"] == [512, 512, 256]
    assert network_cfg["mlp"]["activation"] == "elu"
    assert train_cfg["name"] == "gm_single_asset_mlp"
    assert train_cfg["horizon_length"] == 32
    assert train_cfg["minibatch_size"] == 16384
    assert train_cfg["torch_compile"] is False


def test_single_asset_mlp_batch_divides_2048_and_4096_envs() -> None:
    r"""默认 2048 与放大 4096 env 的 rollout batch 都应被 minibatch 整除。"""

    train_cfg = _single_asset_agent_cfg()["params"]["config"]
    horizon_length = train_cfg["horizon_length"]
    minibatch_size = train_cfg["minibatch_size"]

    for num_envs in (2048, 4096):
        batch_size = num_envs * horizon_length
        assert batch_size % minibatch_size == 0


def test_train_mlp_single_asset_entry_has_independent_defaults() -> None:
    r"""独立训练入口应默认指向 single-asset task 和 2048 env，而不是 teacher debug。"""

    source = TRAIN_ENTRY_PATH.read_text(encoding="utf-8")

    assert 'DEFAULT_SINGLE_ASSET_TASK = "AnyMani-GM-SingleAsset-MLP-v0"' in source
    assert "DEFAULT_SINGLE_ASSET_NUM_ENVS = 2048" in source
    assert "from isaaclab.app import AppLauncher" in source
    assert "@hydra_task_config(args_cli.task, \"rl_games_cfg_entry_point\")" in source
    assert "import anymani.distill.rl" in source


def test_train_mlp_single_asset_logs_are_anchored_to_anymani_root() -> None:
    r"""训练日志必须落在 `AnyMani/logs`，不能依赖启动命令时的 shell cwd。"""

    source = TRAIN_ENTRY_PATH.read_text(encoding="utf-8")

    assert "from anymani.assets.bank.path_utils import resolve_anymani_root" in source
    assert "ANYMANI_ROOT = resolve_anymani_root()" in source
    assert 'ANYMANI_ROOT / "logs" / "distill" / "rl_games" / config_name' in source
