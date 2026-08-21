r"""Trainer-owned asset permutation、coverage、tail 与 resume 日程合同。"""

from __future__ import annotations

from collections import Counter

import pytest
from anymani.distill.ssl.runtime.sampling import OnlineMinibatchSchedule, OnlineSamplingCfg


def test_each_q_round_is_one_deterministic_permutation_of_all_assets() -> None:
    r"""每个 epoch 的第一窗覆盖全部资产一次；顺序由 seed/epoch 唯一决定。"""

    config = OnlineSamplingCfg(
        epochs=1,
        q_per_asset_per_epoch=4,
        assets_per_minibatch=3,
        q_per_asset_per_minibatch=2,
        seed=29,
    )
    first = OnlineMinibatchSchedule(7, config)
    replay = OnlineMinibatchSchedule(7, config)
    first_round = tuple(first.next() for _ in range(first.asset_groups_per_round))
    replay_round = tuple(replay.next() for _ in range(replay.asset_groups_per_round))
    flattened = tuple(index for minibatch in first_round for index in minibatch.asset_indices)

    assert first_round == replay_round
    assert Counter(flattened) == Counter(range(7))
    assert len(first_round[-1].asset_indices) == 1


def test_tail_q_block_and_tail_accumulation_do_not_repeat_samples() -> None:
    r"""不能整除的 q coverage 和 accumulation 都以较小真实组结束。"""

    schedule = OnlineMinibatchSchedule(
        5,
        OnlineSamplingCfg(
            epochs=1,
            q_per_asset_per_epoch=5,
            assets_per_minibatch=2,
            q_per_asset_per_minibatch=2,
            seed=41,
        ),
    )
    q_count_by_asset: Counter[int] = Counter()
    update_group_sizes: list[int] = []
    while not schedule.complete:
        accumulation = min(4, schedule.minibatches_remaining_in_epoch)
        update_group_sizes.append(accumulation)
        for _ in range(accumulation):
            minibatch = schedule.next()
            for asset_index in minibatch.asset_indices:
                q_count_by_asset[asset_index] += minibatch.q_per_asset

    assert q_count_by_asset == Counter({index: 5 for index in range(5)})
    assert update_group_sizes == [4, 4, 1]


def test_schedule_checkpoint_restores_the_exact_next_minibatch() -> None:
    r"""cursor、current permutation 与 seed 恢复后，下一个 asset group 必须逐项一致。"""

    config = OnlineSamplingCfg(
        epochs=2,
        q_per_asset_per_epoch=7,
        assets_per_minibatch=2,
        q_per_asset_per_minibatch=3,
        seed=53,
    )
    uninterrupted = OnlineMinibatchSchedule(6, config)
    for _ in range(5):
        uninterrupted.next()
    state = uninterrupted.state_dict()
    expected = uninterrupted.next()

    resumed = OnlineMinibatchSchedule(6, config)
    resumed.load_state_dict(state)
    assert resumed.next() == expected


def test_window_major_schedule_finishes_each_resident_window_before_switching() -> None:
    r"""超过 resident cap 时，同一窗内资产先完成全部 q coverage，再切到下一窗。"""

    schedule = OnlineMinibatchSchedule(
        5,
        OnlineSamplingCfg(
            epochs=1,
            q_per_asset_per_epoch=2,
            assets_per_minibatch=2,
            q_per_asset_per_minibatch=1,
            shuffle_assets=False,
            seed=7,
        ),
        max_resident_assets=4,
    )
    items = tuple(schedule.next() for _ in range(schedule.minibatches_per_epoch))
    windows = tuple(item.resident_asset_indices for item in items)
    assert windows == (
        (0, 1, 2, 3),
        (0, 1, 2, 3),
        (0, 1, 2, 3),
        (0, 1, 2, 3),
        (4,),
        (4,),
    )
    assert tuple(item.asset_indices for item in items) == (
        (0, 1),
        (2, 3),
        (0, 1),
        (2, 3),
        (4,),
        (4,),
    )
    assert tuple(item.q_per_asset for item in items) == (1, 1, 1, 1, 1, 1)
    assert schedule.minibatches_per_epoch == 6


def test_completed_schedule_checkpoint_restores_as_complete() -> None:
    """last checkpoint 的空 permutation 是完成态，不应被解释成不存在的下一 epoch。"""

    config = OnlineSamplingCfg(
        epochs=1,
        q_per_asset_per_epoch=3,
        assets_per_minibatch=2,
        q_per_asset_per_minibatch=2,
        seed=61,
    )
    schedule = OnlineMinibatchSchedule(3, config)
    while not schedule.complete:
        schedule.next()
    state = schedule.state_dict()
    assert state["permutation"] == ()

    resumed = OnlineMinibatchSchedule(3, config)
    resumed.load_state_dict(state)

    assert resumed.complete
    with pytest.raises(StopIteration):
        resumed.next()
