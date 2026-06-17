r"""Contract tests for the heterogeneous-hand MLP smoke PPO route.

这些测试不启动 Isaac Sim。它们只锁住本轮 MVP 的训练管线语义：

1. distill 注册了异构 hand MVP 和 GM in-hand 默认 hand selection 两条 MLP smoke 训练别名；
2. 这些别名指向 `tasks/gm` 环境语义和 distill 自己的 MLP YAML；
3. YAML 使用 rl_games 官方 `actor_critic` MLP，而不是 Transformer teacher builder；
4. rollout batch size 与 $3\times100$ envs 的 smoke 目标整除。
"""

from __future__ import annotations

import ast
from pathlib import Path

import anymani.distill.rl  # noqa: F401  # 注册 distill-owned Gym task aliases
import anymani.distill.rl.agents as agent_package
import gymnasium as gym
import yaml


GM_INHAND_ENV_CFG_PATH = Path(__file__).resolve().parents[2] / "tasks" / "gm" / "inhand_env_cfg.py"
r"""GM in-hand env cfg 源文件路径；测试只做 AST 解析，不 import Isaac Sim runtime。"""


def _mlp_agent_cfg(filename: str = "gm_heterogeneous_mlp_ppo_smoke.yaml") -> dict:
    r"""读取 MLP smoke YAML，避免测试依赖 Hydra / Isaac Sim 启动。

    Args:
        filename (str): `distill/rl/agents` 下的 YAML 文件名。

    Returns:
        dict: YAML 解析后的 rl_games agent cfg。
    """

    yaml_path = Path(agent_package.__file__).with_name(filename)  # distill-owned agent config 文件
    with yaml_path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def _distill_rl_source(filename: str) -> str:
    r"""读取 distill/rl 源码文本，用于检查不启动 Isaac Sim 的声明式可视化 contract。"""

    return Path(agent_package.__file__).parents[1].joinpath(filename).read_text(encoding="utf-8")


def _gm_default_num_envs_from_source() -> int:
    r"""从 `tasks/gm` 源码常量推导当前 GM in-hand 默认并行环境数。"""

    values: dict[str, int] = {}  # 只保存 env-count 相关整数常量，避免 import IsaacLab cfg
    tree = ast.parse(GM_INHAND_ENV_CFG_PATH.read_text(encoding="utf-8"))  # 纯源码 AST，不触发 USD / pxr binding
    target_names = {"GM_DEFAULT_HAND_SAMPLE_COUNT", "GM_DEFAULT_ENVS_PER_HAND", "GM_DEFAULT_NUM_ENVS"}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id  # 模块级常量名
        if name not in target_names:
            continue
        if isinstance(node.value, ast.Constant):
            values[name] = int(node.value.value)  # sample count / env-per-hand 当前 preset
        elif isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Mult):
            left = values[node.value.left.id]  # type: ignore[attr-defined]  # `GM_DEFAULT_HAND_SAMPLE_COUNT`
            right = values[node.value.right.id]  # type: ignore[attr-defined]  # `GM_DEFAULT_ENVS_PER_HAND`
            values[name] = left * right  # 默认总 env 数的源码合同
    return values["GM_DEFAULT_NUM_ENVS"]


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


def test_gm_inhand_mlp_smoke_task_alias_points_to_distill_agent() -> None:
    r"""GM in-hand MLP smoke alias 应绑定到 distill 的简单 MLP YAML。"""

    spec = gym.spec("AnyMani-GM-InHand-MLP-Smoke-v0")

    assert spec.kwargs["env_cfg_entry_point"].endswith("gm_inhand_mlp_smoke_env_cfg:GmInHandMlpSmokeEnvCfg")
    assert spec.kwargs["rl_games_cfg_entry_point"].endswith("agents:gm_inhand_mlp_ppo_smoke.yaml")


def test_gm_inhand_mlp_smoke_enables_command_goal_marker_for_gui_review() -> None:
    r"""MLP smoke GUI 路径应显示 command-owned 虚拟目标物体，便于核对目标姿态。"""

    source = _distill_rl_source("gm_inhand_mlp_smoke_env_cfg.py")  # 纯文本检查，不触发 env 构造

    assert "self.commands.goal_pose.debug_vis = True" in source


def test_gm_inhand_mlp_smoke_yaml_uses_builtin_mlp_not_transformer() -> None:
    r"""GM in-hand smoke 必须先用 rl_games 内置 MLP，不走正式 Transformer teacher。"""

    agent_cfg = _mlp_agent_cfg("gm_inhand_mlp_ppo_smoke.yaml")
    network_cfg = agent_cfg["params"]["network"]
    train_cfg = agent_cfg["params"]["config"]

    assert network_cfg["name"] == "actor_critic"
    assert network_cfg["separate"] is False
    assert network_cfg["mlp"]["units"] == [128, 128]
    assert train_cfg["name"] == "gm_inhand_mlp_smoke_default"
    assert train_cfg["torch_compile"] is False


def test_gm_inhand_mlp_smoke_rollout_batch_matches_current_preset_envs() -> None:
    r"""当前 GM in-hand preset 的 rollout batch 应能被 minibatch 整除。"""

    train_cfg = _mlp_agent_cfg("gm_inhand_mlp_ppo_smoke.yaml")["params"]["config"]
    num_envs = _gm_default_num_envs_from_source()
    batch_size = num_envs * train_cfg["horizon_length"]

    assert train_cfg["horizon_length"] == 4
    assert batch_size > 0
    assert train_cfg["minibatch_size"] <= batch_size
    assert batch_size % train_cfg["minibatch_size"] == 0
