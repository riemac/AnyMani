r"""Trainer-owned 显式 minibatch 预算、资产排列、固定评估尾块与 resume 合同。"""

from __future__ import annotations

from collections import Counter

import pytest
from anymani.distill.ssl.runtime.sampling import (
    FixedAssetQSchedule,
    OnlineMinibatchSchedule,
    OnlineSamplingCfg,
)


def test_training_schedule_emits_exact_num_minibatches_and_replays_from_seed() -> None:
    r"""两个同配置日程必须逐项相同，并恰好在显式批数处停止。"""

    config = OnlineSamplingCfg(assets_per_minibatch=2, q_per_asset_per_minibatch=3, seed=29)
    first = OnlineMinibatchSchedule(8, config, num_minibatches=6, max_resident_assets=4)
    replay = OnlineMinibatchSchedule(8, config, num_minibatches=6, max_resident_assets=4)
    first_items = tuple(first.next() for _ in range(6))
    replay_items = tuple(replay.next() for _ in range(6))

    assert first_items == replay_items
    assert all(len(item.asset_indices) == 2 and item.q_per_asset == 3 for item in first_items)
    assert first.complete and first.minibatches_remaining == 0
    with pytest.raises(StopIteration):
        first.next()


def test_first_catalog_cycle_uses_each_asset_once_then_reshuffles() -> None:
    r"""一个完整 catalog 轮次不重复资产，下一批进入新的确定性排列。"""

    config = OnlineSamplingCfg(
        assets_per_minibatch=2,
        q_per_asset_per_minibatch=1,
        shuffle_assets=True,
        seed=41,
    )
    schedule = OnlineMinibatchSchedule(8, config, num_minibatches=5)
    items = tuple(schedule.next() for _ in range(5))
    first_cycle = tuple(index for item in items[:4] for index in item.asset_indices)

    assert Counter(first_cycle) == Counter(range(8))
    assert items[3].q_block_index == 0
    assert items[4].q_block_index == 1
    assert items[4].minibatch_index == 4


def test_training_requires_full_asset_minibatches() -> None:
    r"""训练预算不允许较小尾批隐式改变 $N_{mb}N_{asset}^{mb}$。"""

    config = OnlineSamplingCfg(assets_per_minibatch=3, q_per_asset_per_minibatch=2)
    with pytest.raises(ValueError, match="must be divisible"):
        OnlineMinibatchSchedule(8, config, num_minibatches=1)


def test_schedule_checkpoint_restores_exact_next_minibatch() -> None:
    r"""全局 cursor、当前 permutation 与 session Sobol cursor 共同锁定下一批。"""

    config = OnlineSamplingCfg(assets_per_minibatch=2, q_per_asset_per_minibatch=3, seed=53)
    uninterrupted = OnlineMinibatchSchedule(8, config, num_minibatches=7, max_resident_assets=4)
    for _ in range(5):
        uninterrupted.next()
    state = uninterrupted.state_dict()
    expected = uninterrupted.next()

    resumed = OnlineMinibatchSchedule(8, config, num_minibatches=7, max_resident_assets=4)
    resumed.load_state_dict(state)
    assert resumed.next() == expected


def test_schedule_checkpoint_rejects_budget_drift() -> None:
    r"""恢复时不得把旧游标静默解释到不同的 ``num_minibatches`` 预算。"""

    config = OnlineSamplingCfg(assets_per_minibatch=2, q_per_asset_per_minibatch=1, seed=61)
    source = OnlineMinibatchSchedule(8, config, num_minibatches=4)
    state = source.state_dict()
    changed = OnlineMinibatchSchedule(8, config, num_minibatches=5)

    with pytest.raises(ValueError, match="num_minibatches"):
        changed.load_state_dict(state)


def test_resident_window_contains_complete_training_minibatches() -> None:
    r"""设备窗口只打包完整训练批，容量余数不产生统计尾组。"""

    schedule = OnlineMinibatchSchedule(
        8,
        OnlineSamplingCfg(
            assets_per_minibatch=2,
            q_per_asset_per_minibatch=1,
            shuffle_assets=False,
        ),
        num_minibatches=4,
        max_resident_assets=5,
    )
    items = tuple(schedule.next() for _ in range(4))

    assert tuple(item.resident_asset_indices for item in items) == (
        (0, 1, 2, 3),
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (4, 5, 6, 7),
    )


def test_fixed_evaluation_keeps_real_asset_and_q_tails() -> None:
    r"""固定评估 bank 可保留较短资产尾批和 q 尾块，不污染训练预算。"""

    schedule = FixedAssetQSchedule(
        5,
        q_per_asset=5,
        assets_per_minibatch=2,
        q_per_asset_per_minibatch=2,
        max_resident_assets=4,
    )
    q_count_by_asset: Counter[int] = Counter()
    final_asset_group_sizes: list[int] = []
    identity_order: list[tuple[tuple[int, ...], int]] = []
    while not schedule.complete:
        item = schedule.next()
        final_asset_group_sizes.append(len(item.asset_indices))
        identity_order.append((item.asset_indices, item.q_block_index))
        for asset_index in item.asset_indices:
            q_count_by_asset[asset_index] += item.q_per_asset

    assert q_count_by_asset == Counter({index: 5 for index in range(5)})
    assert final_asset_group_sizes == [2, 2, 2, 2, 2, 2, 1, 1, 1]
    assert identity_order == [
        ((0, 1), 0),
        ((0, 1), 1),
        ((0, 1), 2),
        ((2, 3), 0),
        ((2, 3), 1),
        ((2, 3), 2),
        ((4,), 0),
        ((4,), 1),
        ((4,), 2),
    ]  # 同组 q-block 连续，固定评估无需反复重建同一 source/device state
