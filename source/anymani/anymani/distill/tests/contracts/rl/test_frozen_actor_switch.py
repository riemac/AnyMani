r"""接力策略的科学边界：准确交接时刻、状态连续、参数冻结及严格载入。"""

from __future__ import annotations

import pytest
import torch
from anymani.distill.rl.runtime.frozen_actor_switch import FrozenActorSwitch


def make_actor():
    r"""旧策略输出2，接替策略输出3，给交接步一个可精确核对的动作参照。"""
    actor = torch.nn.Linear(2, 1, bias=False).eval()  # 无随机层的最小Actor。
    with torch.no_grad():
        actor.weight.copy_(torch.tensor([[2., 0.]]))  # 输入[1,1]时输出2。
    replacement = {'weight': torch.tensor([[0., 3.]])}  # 同一结构、不同冻结控制映射。
    return actor, replacement


def test_handoff_after_three_actions_preserves_history_targets_and_rng():
    r"""先执行3个旧动作，第4个动作才交接；非零控制状态不能被重置。"""
    actor, replacement = make_actor()
    switch = FrozenActorSwitch(actor, replacement, boundary_step=3)  # 已完成步数，不是1-based动作号。
    state = {'target': torch.tensor([[.2, -.4]]), 'history': torch.arange(30.).reshape(1, 30),
             'object_pose': torch.tensor([1., 2., 3.]), 'active': torch.tensor([True, False])}
    expected_state = {key: value.clone() for key, value in state.items()}  # 模拟连续控制的已有状态。
    rng = torch.get_rng_state().clone()  # 交接不重置或消耗探索/环境随机流。
    outputs = []
    for completed in range(6):
        if completed == 3:
            switch.apply(completed, state)  # 唯一干预是Actor参数替换。
        outputs.append(float(actor(torch.ones(1, 2)).detach()))  # 六个动作的精确时间顺序。
    assert outputs == [2., 2., 2., 3., 3., 3.]
    assert all(torch.equal(value, expected_state[key]) for key, value in state.items())
    assert torch.equal(rng, torch.get_rng_state())  # 包括交接前后所有调用。
    result = switch.finish()  # 验证接替后的Actor仍然冻结。
    assert result['performed'] and result['boundary_step'] == 3 and switch.phase == 1
    assert result['state_continuity_check'] == result['torch_rng_check'] == 'bitwise-equal'


def test_control_condition_reaches_same_boundary_without_replacing_actor():
    r"""全程同策略的参照仍经过边界检查，避免参照遗漏额外的交接调用路径。"""
    actor, _ = make_actor()
    switch = FrozenActorSwitch(actor, None, boundary_step=2)  # 没有接替参数。
    switch.apply(2, {'target': torch.tensor([.3])})
    result = switch.finish()
    assert result['boundary_reached'] and not result['performed'] and switch.phase == 0
    assert float(actor(torch.ones(1, 2)).detach()) == 2.  # 参照动作逐值保持。


def test_wrong_time_duplicate_handoff_and_parameter_drift_are_rejected():
    r"""错误时刻、重复交接、意外参数漂移都必须显式暴露。"""
    actor, replacement = make_actor()
    switch = FrozenActorSwitch(actor, replacement, boundary_step=3)
    with pytest.raises(ValueError, match='boundary'):
        switch.apply(2, {'target': torch.ones(1)})  # 少执行了一步旧动作。
    switch.apply(3, {'target': torch.ones(1)})
    with pytest.raises(ValueError, match='already'):
        switch.apply(3, {'target': torch.ones(1)})  # 不能把同一交接重复应用。
    with torch.no_grad():
        actor.weight.add_(1)  # 模拟未声明的额外训练/权重修改。
    with pytest.raises(RuntimeError, match='Actor'):
        switch.finish()


@pytest.mark.parametrize('replacement', [{}, {'weight': torch.ones(2, 2)}, {'weight': torch.ones(1, 2).double()}])
def test_replacement_contract_is_checked_before_any_mutation(replacement):
    r"""同一接口的参数键、形状与dtype在真正copy之前闭合。"""
    actor, _ = make_actor()
    initial = actor.weight.detach().clone()
    with pytest.raises(ValueError):
        FrozenActorSwitch(actor, replacement, boundary_step=3)
    assert torch.equal(actor.weight, initial)  # 错误checkpoint不能留下部分载入的模型。


def test_model_load_side_effect_on_controller_state_is_detected():
    r"""即使载入hook越界改动控制状态，连续性检查也必须发现。"""
    actor, replacement = make_actor()
    target = torch.tensor([.25])  # 非零旧目标，用于捕获隐式reset。

    def reset_target(_model, _keys):
        r"""模拟越界loader hook，按PyTorch合同不返回值。"""
        target.zero_()  # 不应属于Actor交接的动作。

    actor.register_load_state_dict_post_hook(reset_target)
    switch = FrozenActorSwitch(actor, replacement, boundary_step=3)
    with pytest.raises(RuntimeError, match='continuity'):
        switch.apply(3, {'target': target})
