r"""MVP80 rl_games alias、预算、学习率与精度配置合同。"""

from __future__ import annotations

from importlib import resources

import anymani.distill.rl  # noqa: F401  # 注册distill-owned training alias
import gymnasium as gym
import yaml
from anymani.distill.rl import agents


def _config() -> dict:
    r"""读取versioned MVP80 agent YAML，不导入Isaac task config。"""

    text = resources.files(agents).joinpath("heterogeneous_palm_rotation_mvp_ppo.yaml").read_text(encoding="utf-8")
    document = yaml.safe_load(text)
    assert isinstance(document, dict)
    return document["params"]


def test_mvp80_alias_binds_task_cfg_and_custom_agent_yaml() -> None:
    r"""训练alias必须绑定palm task与distill-owned custom PPO YAML。"""

    spec = gym.spec("AnyMani-Hetero-Generated-PalmRotation-MVP-RLGames-v0")
    assert spec.kwargs["env_cfg_entry_point"].endswith(":GeneratedPalmRotationMvpEnvCfg")
    assert spec.kwargs["rl_games_cfg_entry_point"].endswith(":heterogeneous_palm_rotation_mvp_ppo.yaml")


def test_mvp80_yaml_locks_stratified_budget_dual_lr_and_fp32_policy() -> None:
    r"""正式YAML固定76,800 batch、16份等量minibatch、5 epochs与三组LR。"""

    params = _config()
    config = params["config"]
    assert params["algo"]["name"] == "anymani_palm_rotation_ppo"
    assert params["model"]["name"] == "anymani_palm_rotation_masked_continuous"
    assert params["network"]["name"] == "anymani_palm_rotation"
    assert params["network"]["palm_rotation"] == {
        "residual_enabled": True,
        "initial_log_std": -0.5,
        "max_log_std": -0.43,
        "base_action_limit": 0.8,
    }
    assert config["num_actors"] == 2560 and config["asset_count"] == 80
    assert config["horizon_length"] == 30
    assert config["num_actors"] * config["horizon_length"] == 76800
    assert config["minibatch_size"] == 4800 and config["mini_epochs"] == 5
    assert config["gradient_accumulation_steps"] == 4
    assert config["learning_rate"] == 3.0e-4
    assert config["adaptive_lr_max"] == config["learning_rate"]
    assert config["residual_learning_rate"] == 1.0e-4
    assert config["critic_learning_rate"] == 5.0e-4
    assert config["mixed_precision"] is False and config["normalize_input"] is False
    assert config["normalize_value"] is True and config["normalize_advantage"] is True
    assert config["gamma"] == 0.99 and config["tau"] == 0.95
    assert config["e_clip"] == 0.2 and config["entropy_coef"] == 0.002
    assert config["kl_threshold"] == 0.01 and config["grad_norm"] == 1.0
    assert config["diagnostics_flush_updates"] == 50
    assert config["gpu_memory_fraction_limit"] == 0.85
    assert config["save_frequency"] == 128 and config["evaluation_frequency"] == 320
