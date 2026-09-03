r"""Per-asset EMA与8-cell median reward-release的纯Torch合同。"""

from __future__ import annotations

import pytest
import torch

from anymani.tasks.hetero.mdp.curriculum_state import (
    HeterogeneousRewardReleaseState,
    even_median,
    release_from_net_turns,
)
from anymani.tasks.hetero.mdp.task_math import active_reference_l2, active_reference_sum


def test_stable_reward_reduction_preserves_n000_and_normalizes_active_dof() -> None:
    r"""16-DoF保持N000 sum/L2，8-DoF同幅关节运动折算到相同参考量级。"""

    values = torch.ones(3, 16, dtype=torch.float32)  # 每个有效关节贡献相同单位penalty
    masks = torch.tensor(
        (
            (True,) * 16,
            (True,) * 8 + (False,) * 8,
            (True,) * 4 + (False,) * 12,
        ),
        dtype=torch.bool,
    )  # 16/8/4 active DoF三种形态
    values[1, 8:] = 1.0e6  # ghost storage污染不应进入任何reward分子
    values[2, 4:] = 1.0e6

    torch.testing.assert_close(active_reference_sum(values, masks), torch.full((3,), 16.0))
    torch.testing.assert_close(active_reference_l2(values, masks), torch.full((3,), 4.0))


def test_stable_reward_reduction_rejects_empty_or_non_bool_masks() -> None:
    r"""无有效关节或数值mask会让$n_i$语义未定义，必须fail closed。"""

    values = torch.ones(1, 16)
    with pytest.raises(ValueError, match="bool active_mask"):
        active_reference_sum(values, torch.ones_like(values))
    with pytest.raises(ValueError, match="at least one active joint"):
        active_reference_l2(values, torch.zeros_like(values, dtype=torch.bool))


def test_even_median_averages_middle_assets() -> None:
    r"""十资产cell必须取第5/6项均值，而不是torch.median的下中位数。"""

    values = torch.arange(10, dtype=torch.float32)
    assert even_median(values).item() == pytest.approx(4.5)
    released = release_from_net_turns(values, release_start_turns=1.0, release_end_turns=2.0)
    assert released.tolist()[:4] == [0.0, 0.0, 1.0, 1.0]


def test_asset_ema_updates_independently_and_cell_median_controls_env_gain() -> None:
    r"""极快单asset不能替整个cell释放reward；env coefficient按asset所属cell广播。"""

    # 与正式MVP相同：8 cells×10 assets；这里每资产只放一个env以隔离scope数学。
    asset_rows = tuple(range(80))
    cell_ids = tuple(cell for cell in range(8) for _ in range(10))
    routing = tuple(range(80))
    state = HeterogeneousRewardReleaseState(
        dataset_rows_by_asset=asset_rows,
        cell_ids_by_asset=cell_ids,
        asset_index_by_env=routing,
        device="cpu",
    )
    turns = torch.zeros(80)
    turns[0] = 4.0  # cell0只有asset0很快，其余9项尚未学习
    turns[10:20] = 2.0  # cell1十项资产都达到release end
    state.update(
        reset_env_ids=torch.arange(80),
        positive_net_turns_by_env=turns,
        ema_alpha=1.0,
        release_start_turns=1.0,
        release_end_turns=2.0,
    )
    assert state.asset_net_turns_ema[0].item() == 4.0
    assert state.asset_candidate_lambda[0].item() == 1.0
    assert state.cell_net_turns_median[:2].tolist() == [0.0, 2.0]
    assert state.cell_lambda[:2].tolist() == [0.0, 1.0]
    assert torch.equal(state.env_lambda[:10], torch.zeros(10))
    assert torch.equal(state.env_lambda[10:20], torch.ones(10))

    # 只更新cell2 asset20，asset21旧EMA保持0；其它cells也不能被partial reset清零。
    before = state.asset_net_turns_ema.clone()
    partial_ids = torch.tensor((20,))
    turns[partial_ids] = 1.5
    state.update(
        reset_env_ids=partial_ids,
        positive_net_turns_by_env=turns,
        ema_alpha=1.0,
        release_start_turns=1.0,
        release_end_turns=2.0,
    )
    assert state.asset_net_turns_ema[20].item() == pytest.approx(1.5)
    assert state.asset_net_turns_ema[21].item() == 0.0
    assert torch.equal(state.asset_net_turns_ema[:20], before[:20])


def test_reward_release_checkpoint_roundtrip_requires_identical_routing() -> None:
    r"""课程resume必须逐值恢复80资产/8-cell状态，并拒绝不同asset顺序。"""

    rows = tuple(range(80))
    cells = tuple(cell for cell in range(8) for _ in range(10))
    routing = tuple(index % 80 for index in range(160))
    source = HeterogeneousRewardReleaseState(
        dataset_rows_by_asset=rows,
        cell_ids_by_asset=cells,
        asset_index_by_env=routing,
        device="cpu",
    )
    turns = torch.linspace(0.0, 3.0, 160)
    source.update(
        reset_env_ids=torch.arange(160),
        positive_net_turns_by_env=turns,
        ema_alpha=0.25,
        release_start_turns=1.0,
        release_end_turns=2.0,
    )
    checkpoint = source.state_dict()
    restored = HeterogeneousRewardReleaseState(
        dataset_rows_by_asset=rows,
        cell_ids_by_asset=cells,
        asset_index_by_env=routing,
        device="cpu",
    )
    restored.load_state_dict(checkpoint)

    for name in (
        "asset_net_turns_ema",
        "asset_episode_updates",
        "asset_candidate_lambda",
        "cell_net_turns_median",
        "cell_lambda",
        "env_lambda",
    ):
        torch.testing.assert_close(getattr(restored, name), getattr(source, name), rtol=0.0, atol=0.0)

    wrong_rows = HeterogeneousRewardReleaseState(
        dataset_rows_by_asset=tuple(reversed(rows)),
        cell_ids_by_asset=cells,
        asset_index_by_env=routing,
        device="cpu",
    )
    with pytest.raises(RuntimeError, match="dataset rows"):
        wrong_rows.load_state_dict(checkpoint)
