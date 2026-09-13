r"""逐环境ADR：原25档幅度、独立升级/回退、无高度扰动与状态恢复。"""

import torch

from anymani.tasks.hetero.mdp.adr import ObjectPositionAdrCfg, PerEnvironmentPositionAdr


def test_original_range_resolution_is_independent_of_max_level():
    r"""max=5保留第5/25范围，不能变成全幅1cm。"""
    state = PerEnvironmentPositionAdr(ObjectPositionAdrCfg(enabled=True), 2, 'cpu')
    state.level[:] = torch.tensor([1, 5])
    torch.testing.assert_close(state.half_width, torch.tensor([.0004, .002]))
    offset = state.sample(torch.tensor([0, 1]))
    assert bool((offset[:, 2] == 0).all())
    assert bool((offset.abs() <= state.half_width[:, None]).all())


def test_strong_environment_cannot_raise_weak_environment_level():
    r"""同一资产的不同环境也独立，不由强环境平均值推进弱环境。"""
    state = PerEnvironmentPositionAdr(ObjectPositionAdrCfg(enabled=True), 3, 'cpu')
    ids = torch.tensor([0, 1])
    state.sample(ids)
    for _ in range(3):
        state.observe(ids, torch.tensor([1.5, .1]), torch.tensor([True, False]))
    assert state.level.tolist() == [2, 0, 1]
    assert state.trials.tolist() == [0, 0, 0]
    assert not bool(state.initialized[2])  # 未参与的环境没有虚构回合。
    clone = PerEnvironmentPositionAdr(ObjectPositionAdrCfg(enabled=True), 3, 'cpu')
    clone.load_state_dict(state.state_dict())
    assert torch.equal(clone.level, state.level)
