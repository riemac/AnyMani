r"""冻结动作对照的CPU合同：原PPO采样一致性、ghost、独立随机流和正式门。

测试直接执行生产PalmRotationMaskedContinuousModel的rollout forward形成参照。
固定a2c输出只消除网络拟合因素；概率变换、随机采样与mask仍走真实生产实现。
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from anymani.distill.rl.runtime.frozen_action_selection import (
    select_frozen_actor_actions,
    validate_frozen_action_mode,
)
from anymani.distill.rl.runtime.palm_rotation_network import PalmRotationMaskedContinuousModel


@pytest.fixture
def policy_parameters():
    r"""两条16槽动作，包含饱和端点、近边界均值及不同ghost分布。"""
    mask = torch.ones(2, 16, dtype=torch.bool)  # 32坐标中4个ghost
    mask[0, -3:] = False  # 第一条有效13关节
    mask[1, 3] = False  # 第二条内部缺失一个关节
    mean = torch.linspace(-1, 1, 32).reshape(2, 16) * mask  # 包含精确±1，检验atanh夹紧合同
    return mean, mask  # FP32中心与同形bool mask


def test_mean_mode_is_exact_identity_and_consumes_no_random_numbers(policy_parameters):
    r"""默认均值路径必须逐值保留动作，包括±1端点，且不采样随机数。"""
    mean, mask = policy_parameters  # 所有有效槽保留原中心
    state = torch.get_rng_state().clone()  # 检查默认路径的随机状态副作用
    action, trace = select_frozen_actor_actions(mean, torch.tensor(-.5), mask)  # 默认mode=mean
    assert action is mean and trace == {}  # 不增加tanh/atanh往返或新的trace字段
    assert torch.equal(state, torch.get_rng_state())  # 不改变后续环境/其他模块随机流


@pytest.mark.parametrize("log_std", [-.5, -2.1, -5.0])
def test_sampling_matches_production_rollout_bitwise_and_isolates_rng(policy_parameters, log_std):
    r"""相同seed下与实际PPO采样逐值一致；当前随机流和ghost不改变动作语义。"""
    mean, mask = policy_parameters  # 均值覆盖非线性tanh区间与边界
    logstd = torch.tensor(log_std, dtype=torch.float32)  # 初期/当前/很窄三种latent探索尺度

    class FixedA2c:
        r"""仅提供a2c已经形成的中心/尺度，生产分布forward仍完整执行。"""

        arm = "direct"  # 要求生产forward交付真实direct side-channel
        last_active_joint_mask = mask  # 分布必须屏蔽这些ghost坐标
        last_direct_mean = mean  # direct动作中心按原接口交付
        last_film_modulation_rms = torch.zeros_like(mean)  # 与概率变换无关的诊断字段

        def __call__(self, _input):
            r"""模拟冻结网络前向返回mu/logstd/value，避免引入另一随机来源。"""
            return mean, logstd.expand_as(mean), torch.zeros(2, 1), None  # 真实forward的四元返回

    cls = PalmRotationMaskedContinuousModel.Network  # 真实训练分布实现
    model = SimpleNamespace(  # 不构造物理环境/优化器/额外模型参数
        a2c_network=FixedA2c(), norm_obs=lambda observation: observation,  # 只替代神经前向，不替代采样算子
        _action_to_latent=cls._action_to_latent,  # 原生产atanh夹紧合同
        _squashed_per_joint_neglogp=cls._squashed_per_joint_neglogp,  # 保留完整rollout密度计算
        denorm_value=lambda value: value,  # 本概率测试没有value坐标变换
    )  # 仅补足production forward访问的外部接口
    with torch.random.fork_rng(devices=[]):  # 单元测试恢复调用方CPU随机状态
        torch.manual_seed(731)  # 生产rollout的全局随机流
        expected = cls.forward(model, {"obs": {}, "is_train": False})  # 真正的Normal.sample路径
    generator = torch.Generator(device="cpu").manual_seed(731)  # 独立但同seed的诊断随机流
    global_state = torch.get_rng_state().clone()  # 外部流不应被诊断采样消耗
    action, trace = select_frozen_actor_actions(mean, logstd, mask, mode="sample", generator=generator)
    torch.testing.assert_close(action, expected["actions"], rtol=0, atol=0)  # 包括±1附近的float32行为
    torch.testing.assert_close(trace["policy_latent_sigma"], expected["sigmas"], rtol=0, atol=0)  # 同一sigma广播
    assert torch.equal(global_state, torch.get_rng_state())  # 后续环境reset的RNG独立
    assert (action[~mask] == 0).all() and torch.isfinite(action).all() and (action.abs() <= 1).all()  # ghost/边界
    torch.testing.assert_close(action, torch.tanh(trace["policy_latent_sample"]) * mask, rtol=0, atol=0)  # 可重建动作
    second, _ = select_frozen_actor_actions(mean, logstd, mask, mode="sample", generator=generator)  # 同一流继续推进
    assert not torch.equal(action[mask], second[mask])  # 禁止每一步重新seed造成固定噪声


@pytest.mark.parametrize("mode,seed", [("sample", None), ("mean", 42), ("sample", -1), ("sample", True), ("unknown", 1)])
def test_ambiguous_action_mode_or_seed_is_rejected(mode, seed):
    r"""随机诊断必须有显式合法seed，均值模式不能声称消费一个未使用的seed。"""
    with pytest.raises(ValueError):
        validate_frozen_action_mode(mode, seed)  # 在创建Isaac环境前拒绝不明确的采样声明


def test_sampler_requires_mode_matched_generator(policy_parameters):
    r"""拒绝漏给随机流或给均值路径附加随机流，避免执行与身份描述不一致。"""
    mean, mask = policy_parameters  # 正常actor输出
    with pytest.raises(ValueError):
        select_frozen_actor_actions(mean, torch.tensor(-.5), mask, mode="sample")  # 不使用隐含全局随机流
    with pytest.raises(ValueError):
        select_frozen_actor_actions(mean, torch.tensor(-.5), mask, generator=torch.Generator())  # mean不消费generator


def test_sample_mode_cannot_enter_formal_reliable_gate():
    r"""读取真实入口的正式门表达式；即使30秒/R16其余条件全满足，sample仍不得正式晋级。"""
    source = Path(__file__).resolve().parents[3] / "rl/evaluate_palm_rotation_mvp.py"  # distill-owned入口
    tree = ast.parse(source.read_text())  # 不import会启动Isaac的入口
    assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)  # 定位实际正式可靠门
                      and any(isinstance(t, ast.Name) and t.id == "reliable_protocol_matched" for t in node.targets))
    expression = compile(ast.Expression(assignment.value), str(source), "eval")  # 实际生产判据
    args = SimpleNamespace(steps=600, num_replicas=16, residual_off=False, tip_only_intervention=False,
                           direct_logit_gain=1.0, action_mode="mean", diagnostic_only=False)  # 正式均值基准
    assert eval(expression, {"args_cli": args, "actor_tip_only": True})  # 正式协议仍可匹配
    args.action_mode = "sample"  # 只改变动作模式
    assert not eval(expression, {"args_cli": args, "actor_tip_only": True})  # 随机诊断不产生正式可靠结论
    cohort_assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)  # 原A80门同样需要模式约束
                             and any(isinstance(t, ast.Name) and t.id == "cohort" for t in node.targets))
    assert isinstance(cohort_assignment.value, ast.IfExp)  # A80的旧cohort门也应显式区分模式
    legacy_gate = compile(ast.Expression(cohort_assignment.value.test), str(source), "eval")  # 真实旧门条件
    assert not eval(legacy_gate, {"args_cli": args, "asset_count": 80})  # sample即使A80也不出正式cohort结论
    args.action_mode, args.diagnostic_only = "mean", True  # 接力诊断以及其不切换参照同样关闭正式门。
    assert not eval(expression, {"args_cli": args, "actor_tip_only": True})
    assert not eval(legacy_gate, {"args_cli": args, "asset_count": 80})
