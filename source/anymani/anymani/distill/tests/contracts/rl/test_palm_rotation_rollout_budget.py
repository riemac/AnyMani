r"""PPO连续采样长度的CPU合同：固定B=NH、微批次、预算与默认路径。

从入口提取实际预算函数，避免import入口时启动Isaac Sim。
N512/H60与N1024/H30均应形成30720新样本、3840激活微批和每更新20次参数更新。
rollout H独立于Actor History30和20Hz策略步长；此配置函数只调整采样/优化布局。
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ENTRY = Path(__file__).resolve().parents[3] / "rl/train_palm_rotation_mvp.py"  # 实际训练入口


def configure(*, envs=512, assets=256, horizon=30, updates=500, smoke=False, minibatches=8, accumulation=2):
    r"""执行真实_configure_budget，返回其输出和配置；所有科学参数显式交付。"""
    tree = ast.parse(ENTRY.read_text())  # 仅取AST，不import AppLauncher
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_configure_budget")  # 真实预算实现
    args = SimpleNamespace(  # 与CLI相同的选项命名，seed/物理不在预算函数内改变
        smoke=smoke, rollout_steps=horizon, max_updates=updates, minibatches=minibatches,  # 预算自由度
        gradient_accumulation_steps=accumulation, gradient_probe_frequency=0, full_gradient_shadow_frequency=0,  # 优化步计数
        advantage_normalization_scope="per_asset_rollout", arm="direct_token", history_encoder="tcn",  # 学习方法不变
        torch_compile="default",  # 相同的窄forward编译模式
    )  # 控制变量保持，测试仅改变N与H等预算量
    cfg = {"params": {"config": {"gradient_accumulation_steps": 1, "gamma": .99, "tau": .95},
                      "network": {"palm_rotation": {}}}}  # 真实函数需要的最窄配置接口
    namespace = {"Any": Any, "args_cli": args, "num_envs": envs, "asset_count": assets}  # 显式环境/资产分母
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(ENTRY), "exec"), namespace)  # 编译实际入口函数
    return namespace["_configure_budget"](cfg), cfg  # 保留实际写入的PPO配置


def test_longer_rollout_matches_reference_data_and_optimizer_budget():
    r"""并行副本减半、连续长度加倍时，数据量及优化器预算必须闭合相等。"""
    for envs, horizon in ((1024, 30), (512, 60)):  # R4/H30与R2/H60两个已声明布局
        output, cfg = configure(envs=envs, horizon=horizon)  # 同样500更新、M8/K2/E5
        ppo = cfg["params"]["config"]  # 检查真正交给rl_games的值
        assert output == (horizon, 3840, 500)  # H、激活微批大小、epoch预算
        batch = ppo["num_actors"] * ppo["horizon_length"]  # B=N_env H
        assert batch == 30720 and batch // ppo["asset_count"] == 120  # 每资产等量120新样本
        logical_steps = batch // ppo["minibatch_size"] // ppo["gradient_accumulation_steps"]  # M/K=4
        assert logical_steps * ppo["mini_epochs"] == 20  # 两套优化器各自每更新20步
        assert batch * ppo["max_epochs"] == 15360000  # 独立500-update比较的新交互量
        assert ppo["gamma"] == .99 and ppo["tau"] == .95  # 本次不改变折扣/GAE trace系数
        assert cfg["params"]["network"]["palm_rotation"]["history_encoder"] == "tcn"  # Actor History30路径保持


@pytest.mark.parametrize("envs,updates", [(2560, 391), (1280, 782)])
def test_original_h30_default_budget_is_preserved(envs, updates):
    r"""原MVP默认与fallback保留H30和既有更新预算。"""
    output, cfg = configure(envs=envs, assets=80, updates=None, minibatches=None, accumulation=None)  # 原MVP入口默认值
    assert output == (30, envs * 30 // 16, updates)  # 原16微批次与30.03M量级预算
    assert cfg["params"]["config"]["gradient_accumulation_steps"] == 1  # 沿用配置缺省累积


def test_h60_requires_explicit_update_budget():
    r"""拒绝让旧H30默认epoch公式随H60无声翻倍交互量。"""
    with pytest.raises(ValueError, match="max_updates"):
        configure(horizon=60, updates=None)  # 研究预算应在启动前明确


@pytest.mark.parametrize("horizon", [0, -1])
def test_nonpositive_rollout_length_is_rejected(horizon):
    r"""空采样段不应进入GAE或按资产分层的整数预算。"""
    with pytest.raises(ValueError, match="rollout_steps must be positive"):
        configure(horizon=horizon)  # 在任何整除计算前拒绝


def test_smoke_remains_h4_and_rejects_conflicting_custom_horizon():
    r"""默认smoke仍是4步；显式长rollout不应被smoke静默覆盖。"""
    output, cfg = configure(envs=80, assets=80, updates=None, smoke=True, minibatches=None, accumulation=None)  # 原smoke布局
    assert output == (4, 80, 1) and cfg["params"]["config"]["mini_epochs"] == 1  # 原工程检查成本
    with pytest.raises(ValueError, match="smoke"):
        configure(envs=80, assets=80, horizon=60, smoke=True)  # 两种预算意图相冲突


def test_custom_horizon_retains_per_asset_stratification_gate():
    r"""长度可配置仍须保证每个激活微批具有相同的逐资产样本数。"""
    with pytest.raises(ValueError, match="per-asset"):
        configure(horizon=61)  # R2×61=122，无法分为8份等量资产微批


def test_rollout_cli_default_is_thirty():
    r"""检查CLI真实声明，默认入口不会随新实验改成H60。"""
    tree = ast.parse(ENTRY.read_text())  # Parser声明在AppLauncher之前
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and n.args  # 定位CLI实际声明而非文档描述
             and isinstance(n.args[0], ast.Constant) and n.args[0].value == "--rollout_steps"]
    assert len(calls) == 1  # 有且仅有一个可追踪入口
    default = next(k.value for k in calls[0].keywords if k.arg == "default")  # 默认值而非help文字
    assert isinstance(default, ast.Constant) and default.value == 30  # 与历史H30一致
