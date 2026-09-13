r"""全局单头PopArt合同：统计只更新一次，任意预测幅度下保持物理value。"""

from __future__ import annotations

import copy

import pytest
import torch
from anymani.distill.rl.algorithms.popart import PopArtValueNormalizer


def test_popart_preserves_physical_predictions_beyond_old_clip_range() -> None:
    r"""输出保持不能只在[-5,5]内成立；补偿仅作用于最后线性层。"""
    torch.manual_seed(42)
    normalizer = PopArtValueNormalizer()
    head = torch.nn.Linear(3, 1).double()
    features = torch.randn(16, 3, dtype=torch.float64) * 100
    before = normalizer(head(features), denorm=True).detach()
    assert head(features).abs().max() > 5
    normalizer.update_from_returns(torch.tensor([[100.0], [200.0], [-50.0]]), head)
    after = normalizer(head(features), denorm=True).detach()
    torch.testing.assert_close(after, before, atol=1e-10, rtol=1e-10)
    assert normalizer.count.item() == 4


def test_forward_is_read_only_even_when_upstream_switches_to_train() -> None:
    r"""上游prepare_dataset调用train后，values/returns归一化也不能重复更新moments。"""
    normalizer = PopArtValueNormalizer()
    head = torch.nn.Linear(2, 1)
    normalizer.update_from_returns(torch.tensor([[2.0], [4.0]]), head)
    state = copy.deepcopy(normalizer.state_dict())
    normalizer.train()
    normalizer(torch.tensor([[1.0], [3.0]]))
    normalizer(torch.tensor([[2.0], [4.0]]))
    normalizer(torch.tensor([[20.0]]), denorm=True)
    for key, value in state.items():
        torch.testing.assert_close(normalizer.state_dict()[key], value, atol=0, rtol=0)


def test_state_round_trip_and_inverse() -> None:
    r"""统计key保持value_mean_std兼容，正常化/反正常化不引入隐藏裁剪。"""
    normalizer = PopArtValueNormalizer()
    normalizer.update_from_returns(torch.arange(10.0).reshape(-1, 1), torch.nn.Linear(2, 1))
    restored = PopArtValueNormalizer()
    restored.load_state_dict(normalizer.state_dict(), strict=True)
    assert set(restored.state_dict()) == {"running_mean", "running_var", "count"}
    values = torch.tensor([[-200.0], [0.0], [300.0]], dtype=torch.float64)
    torch.testing.assert_close(restored(restored(values), denorm=True), values, atol=1e-10, rtol=1e-10)


def test_bad_returns_do_not_partially_change_head_or_statistics() -> None:
    r"""非法target必须在改动head/moments前被拒绝。"""
    normalizer = PopArtValueNormalizer()
    head = torch.nn.Linear(2, 1)
    old_head = copy.deepcopy(head.state_dict())
    old_stats = copy.deepcopy(normalizer.state_dict())
    with pytest.raises((ValueError, RuntimeError), match="finite"):
        normalizer.update_from_returns(torch.tensor([[float("nan")]]), head)
    for key in old_head:
        torch.testing.assert_close(head.state_dict()[key], old_head[key])
    for key in old_stats:
        torch.testing.assert_close(normalizer.state_dict()[key], old_stats[key])
