r"""Contract tests for the unified single-asset MLP PPO training route.

这些测试不启动 Isaac Sim。它们只验证 distill 侧的训练管线声明：single-asset MDP
probe 使用唯一 `anymani.distill.train` 入口、绑定 tasks-owned single-asset env cfg、
采用 rl_games 内置 MLP，且 2048/4096 env 的 PPO batch 都可以被 minibatch 整除。
"""

from __future__ import annotations

from pathlib import Path

import anymani.distill.rl  # noqa: F401  # 注册 distill-owned Gym aliases
import anymani.distill.rl.agents as agent_package
import gymnasium as gym
import pytest
import yaml
from gymnasium.error import NameNotFound

TRAIN_ENTRY_PATH = Path(__file__).resolve().parents[1] / "train.py"
"""统一 RL 训练入口源码路径；当前默认 route 是 single-asset MLP probe。"""

PLAY_ENTRY_PATH = Path(__file__).resolve().parents[1] / "play.py"
"""统一 RL checkpoint 回放入口源码路径；当前默认 route 是 single-asset MLP probe。"""


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


def test_legacy_distill_debug_and_play_aliases_are_removed() -> None:
    r"""旧 MVP/debug/play aliases 应及时出清，避免污染当前训练语义。"""

    removed_aliases = (
        "AnyMani-GM-Teacher-Debug-v0",
        "AnyMani-GM-Teacher-Debug-Play-v0",
        "AnyMani-GM-Heterogeneous-MLP-Smoke-v0",
        "AnyMani-GM-InHand-MLP-Smoke-v0",
        "AnyMani-GM-SingleAsset-MLP-Play-v0",
    )
    for task_id in removed_aliases:
        with pytest.raises(NameNotFound):
            gym.spec(task_id)


def test_single_asset_mlp_yaml_uses_builtin_mlp_not_transformer() -> None:
    r"""单资产 MDP probe 必须使用 rl_games 内置 MLP，而不是正式 Transformer teacher。"""

    agent_cfg = _single_asset_agent_cfg()
    network_cfg = agent_cfg["params"]["network"]
    train_cfg = agent_cfg["params"]["config"]

    assert network_cfg["name"] == "actor_critic"
    assert network_cfg["space"]["continuous"]["sigma_init"]["val"] == -0.5
    assert network_cfg["mlp"]["units"] == [512, 256, 128]
    assert network_cfg["mlp"]["activation"] == "elu"
    assert train_cfg["name"] == "gm_single_asset_mlp"
    assert train_cfg["horizon_length"] == 32
    assert train_cfg["minibatch_size"] == 16384
    assert train_cfg["torch_compile"] is False
    assert train_cfg["rsl_style_console"] is True


def test_single_asset_mlp_batch_divides_2048_and_4096_envs() -> None:
    r"""默认 2048 与放大 4096 env 的 rollout batch 都应被 minibatch 整除。"""

    train_cfg = _single_asset_agent_cfg()["params"]["config"]
    horizon_length = train_cfg["horizon_length"]
    minibatch_size = train_cfg["minibatch_size"]

    for num_envs in (2048, 4096):
        batch_size = num_envs * horizon_length
        assert batch_size % minibatch_size == 0


def test_unified_train_entry_has_single_asset_defaults() -> None:
    r"""统一训练入口应默认指向 single-asset task 和 4096 env，而不是旧 debug route。"""

    source = TRAIN_ENTRY_PATH.read_text(encoding="utf-8")

    assert 'DEFAULT_TASK = "AnyMani-GM-SingleAsset-MLP-v0"' in source
    assert "DEFAULT_NUM_ENVS = 4096" in source
    assert "from isaaclab.app import AppLauncher" in source
    assert "@hydra_task_config(args_cli.task, \"rl_games_cfg_entry_point\")" in source
    assert "import anymani.distill.rl" in source
    assert "register_anymani_rl_games_networks()" in source
    assert "class RslStyleIsaacAlgoObserver" in source
    assert "Runner(_make_isaac_algo_observer(agent_cfg))" in source


def test_unified_train_entry_logs_are_anchored_to_anymani_root() -> None:
    r"""训练日志必须落在 `AnyMani/logs`，不能依赖启动命令时的 shell cwd。"""

    source = TRAIN_ENTRY_PATH.read_text(encoding="utf-8")

    assert "from anymani.assets.bank.path_utils import resolve_anymani_root" in source
    assert "ANYMANI_ROOT = resolve_anymani_root()" in source
    assert 'ANYMANI_ROOT / "logs" / "distill" / "rl_games" / config_name' in source


def test_unified_train_entry_keeps_core_cli_without_full_experiment_stack() -> None:
    r"""当前入口保留核心训练 CLI，不把 wandb/distributed/PBT 重新塞回 first runnable route。"""

    source = TRAIN_ENTRY_PATH.read_text(encoding="utf-8")

    assert 'parser.add_argument("--video"' in source
    assert 'parser.add_argument("--experiment_name"' in source
    assert 'parser.add_argument("--rl_games_strict"' in source
    assert 'parser.add_argument("--distributed"' not in source
    assert "wandb" not in source
    assert "PbtAlgoObserver" not in source


def test_unified_play_entry_loads_distill_checkpoint_from_anymani_logs() -> None:
    r"""统一回放入口应和训练入口共享 task、rl_games backend pinning 与 AnyMani log root。"""

    source = PLAY_ENTRY_PATH.read_text(encoding="utf-8")

    assert 'DEFAULT_TASK = "AnyMani-GM-SingleAsset-MLP-v0"' in source
    assert "DEFAULT_NUM_ENVS = 1" in source
    assert "prefer_local_rl_games" in source
    assert "ANYMANI_ROOT / \"logs\" / \"distill\" / \"rl_games\"" in source
    assert "@hydra_task_config(args_cli.task, \"rl_games_cfg_entry_point\")" in source
    assert "agent.restore(resume_path)" in source
    assert 'parser.add_argument("--checkpoint"' in source
    assert 'parser.add_argument("--run_name"' in source
