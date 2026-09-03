r"""N000-relative单资产、80手cohort与固定三seed能力门合同。"""

from __future__ import annotations

from dataclasses import replace

import pytest
from anymani.distill.diagnostics.evaluation.rl.palm_rotation import (
    PalmRotationReference,
    evaluate_asset,
    evaluate_cohort,
    evaluate_pairs,
    evaluate_seed_confirmation,
    evaluate_trajectory_medians,
)

REFERENCE = PalmRotationReference(goal_count_median=72.0, net_turns_median=6.0)


def _passing_asset(row: int, cell: int):
    r"""构造恰好达到$2/3$参考且满足一圈/方向/command一致性的资产。"""

    return evaluate_asset(
        dataset_row=row,
        cell_id=cell,
        goal_count_median=48.0,
        net_turns_median=4.0,
        absolute_path_turns_median=5.0,
        reference=REFERENCE,
        command_turn_ratio_relative_tolerance=0.10,
    )


def test_asset_gate_uses_minimum_reference_ratio_and_directional_path() -> None:
    r"""Goals达标但net turns不足时取较小ratio；反向/抖动均不能通过。"""

    passing = _passing_asset(0, 0)
    assert passing.passed and passing.score == pytest.approx(2.0 / 3.0)
    assert passing.directional_consistency == pytest.approx(0.8)
    assert passing.command_turn_ratio_relative_error == 0.0

    weak_turns = evaluate_asset(
        dataset_row=1,
        cell_id=0,
        goal_count_median=72.0,
        net_turns_median=2.0,
        absolute_path_turns_median=2.2,
        reference=REFERENCE,
        command_turn_ratio_relative_tolerance=0.10,
    )
    assert weak_turns.score == pytest.approx(1.0 / 3.0) and not weak_turns.passed
    reverse = evaluate_asset(
        dataset_row=2,
        cell_id=0,
        goal_count_median=0.0,
        net_turns_median=-1.0,
        absolute_path_turns_median=2.0,
        reference=REFERENCE,
        command_turn_ratio_relative_tolerance=0.10,
    )
    assert reverse.failure_labels == ("reverse",) and reverse.relative_tier == "le_0"


def test_command_count_must_agree_with_physical_net_turns() -> None:
    r"""即使$S_i$、一圈和方向门都通过，异常subgoal计数仍应拒绝。"""

    inconsistent = evaluate_asset(
        dataset_row=3,
        cell_id=0,
        goal_count_median=60.0,
        net_turns_median=4.0,
        absolute_path_turns_median=4.5,
        reference=REFERENCE,
        command_turn_ratio_relative_tolerance=0.10,
    )
    assert inconsistent.score >= 2.0 / 3.0
    assert inconsistent.command_turn_ratio == pytest.approx(1.25)
    assert inconsistent.command_turn_ratio_relative_error == pytest.approx(0.25)
    assert not inconsistent.passed


def test_cohort_requires_54_total_five_per_cell_and_valid_identity() -> None:
    r"""总数门不能掩盖弱cell，identity/non-finite错误也独立否决整条seed。"""

    assets = [_passing_asset(cell * 10 + index, cell) for cell in range(8) for index in range(10)]
    # 每cell保留前5项通过，共40项：cell门成立但总54项门失败。
    forty = [result if result.dataset_row % 10 < 5 else replace(result, passed=False) for result in assets]
    cohort = evaluate_cohort(seed=42, asset_results=forty, finite_and_identity_valid=True)
    assert cohort.passed_assets == 40 and cohort.passed_by_cell == (5,) * 8 and not cohort.passed

    # 先让54项通过但cell7只有4项，验证per-cell门独立生效。
    selected = {row for row in range(70)} | {70, 71, 72, 73}
    weak_cell = [result if result.dataset_row in selected else replace(result, passed=False) for result in assets]
    cohort = evaluate_cohort(seed=42, asset_results=weak_cell, finite_and_identity_valid=True)
    assert cohort.passed_assets == 74 and cohort.passed_by_cell[-1] == 4 and not cohort.passed

    all_pass = evaluate_cohort(seed=42, asset_results=assets, finite_and_identity_valid=True)
    invalid = evaluate_cohort(seed=42, asset_results=assets, finite_and_identity_valid=False)
    assert all_pass.passed and not invalid.passed


def test_final_confirmation_uses_only_fixed_seeds_and_two_of_three() -> None:
    r"""42/43通过、44失败应满足最终门；任意替补seed应拒绝。"""

    assets = [_passing_asset(cell * 10 + index, cell) for cell in range(8) for index in range(10)]
    seed42 = evaluate_cohort(seed=42, asset_results=assets, finite_and_identity_valid=True)
    seed43 = evaluate_cohort(seed=43, asset_results=assets, finite_and_identity_valid=True)
    seed44 = evaluate_cohort(seed=44, asset_results=assets, finite_and_identity_valid=False)
    assert evaluate_seed_confirmation({42: seed42, 43: seed43, 44: seed44})
    with pytest.raises(ValueError, match="exactly seeds"):
        evaluate_seed_confirmation({42: seed42, 43: seed43, 45: replace(seed44, seed=45)})


def test_trajectory_medians_apply_replica_failure_and_finite_rules() -> None:
    r"""Fixed replicas先按资产取中位数；半数drop否决该资产，任一NaN否决整条seed。"""

    rows = tuple(range(80))
    cells = tuple(cell for cell in range(8) for _ in range(10))
    goals = [[48.0, 48.0] for _ in rows]
    turns = [[4.0, 4.0] for _ in rows]
    paths = [[5.0, 5.0] for _ in rows]
    drops = [[False, False] for _ in rows]
    axes = [[False, False] for _ in rows]
    timeouts = [[True, True] for _ in rows]
    drops[0] = [True, False]  # $1/2$ replicasdrop，按约定形成asset failure

    cohort = evaluate_trajectory_medians(
        seed=42,
        dataset_rows=rows,
        cell_ids=cells,
        goal_counts=goals,
        net_turns=turns,
        absolute_path_turns=paths,
        termination_drop=drops,
        termination_axis=axes,
        termination_timeout=timeouts,
        reference=REFERENCE,
    )
    assert cohort.passed_assets == 79 and cohort.passed
    assert cohort.asset_results[0].drop_failure_rate == pytest.approx(0.5)
    assert cohort.asset_results[0].failure_labels == ("drop",)

    turns[7][1] = float("nan")
    non_finite = evaluate_trajectory_medians(
        seed=42,
        dataset_rows=rows,
        cell_ids=cells,
        goal_counts=goals,
        net_turns=turns,
        absolute_path_turns=paths,
        termination_drop=drops,
        termination_axis=axes,
        termination_timeout=timeouts,
        reference=REFERENCE,
    )
    assert not non_finite.finite_and_identity_valid and not non_finite.passed


def test_pair_diagnostics_do_not_change_asset_cohort_gate() -> None:
    r"""左右pair只报告双过/单侧/双失败与能力差，不追溯改变54/80硬门。"""

    assets = [_passing_asset(cell * 10 + index, cell) for cell in range(8) for index in range(10)]
    assets[1] = replace(assets[1], passed=False)
    pairs = evaluate_pairs(assets, [(2 * index, 2 * index + 1) for index in range(40)])
    assert pairs[0].outcome == "left_only"
    assert sum(pair.outcome == "both_passed" for pair in pairs) == 39
    assert evaluate_cohort(seed=42, asset_results=assets, finite_and_identity_valid=True).passed
