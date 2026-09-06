r"""N000-relative单资产、80手cohort与固定三seed能力门合同。"""

from __future__ import annotations

from dataclasses import replace

import pytest
from anymani.distill.diagnostics.evaluation.rl.palm_rotation import (
    PalmRotationPhysicalAssetResult,
    PalmRotationReference,
    evaluate_asset,
    evaluate_cohort,
    evaluate_pairs,
    evaluate_physical_support_trajectory_medians,
    evaluate_reliable_topology_coverage,
    evaluate_scale_ladder_cohort,
    evaluate_seed_confirmation,
    evaluate_support_trajectory_medians,
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


def test_single_support_trajectory_reduction_does_not_fabricate_cohort_gate() -> None:
    r"""Single closure应形成一个逐资产结果与finite证书，而不要求80-row cell population。"""

    assets, finite = evaluate_support_trajectory_medians(
        dataset_rows=(873,),
        cell_ids=(0,),
        goal_counts=((48.0, 48.0),),
        net_turns=((4.0, 4.0),),
        absolute_path_turns=((5.0, 5.0),),
        termination_drop=((False, False),),
        termination_axis=((False, False),),
        termination_timeout=((True, True),),
        reference=REFERENCE,
    )

    assert finite and len(assets) == 1
    assert assets[0].dataset_row == 873 and assets[0].passed


def test_pair_diagnostics_do_not_change_asset_cohort_gate() -> None:
    r"""左右pair只报告双过/单侧/双失败与能力差，不追溯改变54/80硬门。"""

    assets = [_passing_asset(cell * 10 + index, cell) for cell in range(8) for index in range(10)]
    assets[1] = replace(assets[1], passed=False)
    pairs = evaluate_pairs(assets, [(2 * index, 2 * index + 1) for index in range(40)])
    assert pairs[0].outcome == "left_only"
    assert sum(pair.outcome == "both_passed" for pair in pairs) == 39
    assert evaluate_cohort(seed=42, asset_results=assets, finite_and_identity_valid=True).passed


def _physical_result(row: int, mother: str, *, passed: bool) -> PalmRotationPhysicalAssetResult:
    r"""构造只改变scale-ready布尔值的cohort组合fixture。"""

    return PalmRotationPhysicalAssetResult(
        dataset_row=row,
        cell_id=7,
        mother_id=mother,
        frontier_count_median=48.0 if passed else 12.0,
        max_positive_net_turns_median=4.0 if passed else 1.0,
        net_turns_median=3.0 if passed else 1.0,
        absolute_path_turns_median=3.2 if passed else 2.0,
        directional_consistency=0.9375 if passed else 0.5,
        safe_replica_fraction=1.0,
        replica_count=16,
        finite=True,
        viability_passed=passed,
        scale_ready_passed=passed,
        failure_labels=() if passed else ("net-turns-below-two",),
    )


def test_physical_asset_gate_uses_two_turns_directionality_and_joint_survival() -> None:
    r"""Strict goal不进入输入；恰好12/16安全replicas满足0.75闭边界。"""

    results, finite = evaluate_physical_support_trajectory_medians(
        dataset_rows=(0,),
        cell_ids=(7,),
        mother_ids=("right_t4_i4_m4_r4",),
        frontier_counts=((24.0,) * 16,),
        max_positive_net_turns=((2.1,) * 16,),
        net_turns=((2.0,) * 16,),
        absolute_path_turns=((2.2,) * 16,),
        termination_drop=((False,) * 12 + (True,) * 4,),
        termination_axis=((False,) * 16,),
    )
    result = results[0]
    assert finite and result.scale_ready_passed
    assert result.directional_consistency == pytest.approx(2.0 / 2.2)
    assert result.safe_replica_fraction == 0.75
    assert result.frontier_count_median == 24.0


def test_scale_ladder_requires_asset_and_mother_coverage_independently() -> None:
    r"""A64即使48项通过，少于12条mother达到3/4时仍不能晋级。"""

    # 10条mother全过、4条各2项通过，共48项，但只有10条达到3/4。
    weak_mothers = []
    for mother_index in range(16):
        passing_members = 4 if mother_index < 10 else 2 if mother_index < 14 else 0
        weak_mothers.extend(
            _physical_result(4 * mother_index + member, f"mother-{mother_index}", passed=member < passing_members)
            for member in range(4)
        )
    failed = evaluate_scale_ladder_cohort(weak_mothers, finite_and_identity_valid=True)
    assert failed.scale_ready_assets == 48
    assert failed.mothers_with_three_of_four == 10
    assert not failed.passed

    # 12条mother各4项通过，恰好同时满足48/64与12/16 mother门。
    balanced = [_physical_result(row, f"mother-{row // 4}", passed=row // 4 < 12) for row in range(64)]
    passed = evaluate_scale_ladder_cohort(balanced, finite_and_identity_valid=True)
    assert passed.scale_ready_assets == 48
    assert passed.mothers_with_three_of_four == 12
    assert passed.passed


def test_reliable_coverage_uses_raw_thresholds_and_half_of_each_qualified_topology() -> None:
    r"""新门要求安全75%，且每拓扑至少半数代表通过；同名跨family母体分开计算。"""

    assets = [
        replace(
            _physical_result(row, "same-topology-name", passed=True),
            net_turns_median=1.0,
            directional_consistency=0.7,
            safe_replica_fraction=0.75,
            viability_passed=False,
            scale_ready_passed=False,
        )
        for row in range(8)
    ]  # 恰好命中新闭边界，即使旧布尔标签为false也应按原始指标通过。
    assets[2] = replace(assets[2], safe_replica_fraction=0.625, viability_passed=True)
    assets[3] = replace(assets[3], net_turns_median=0.999)
    assets[5] = replace(assets[5], directional_consistency=0.699)
    assets[6] = replace(assets[6], safe_replica_fraction=0.625)
    assets[7] = replace(assets[7], net_turns_median=0.0)
    topology_ids = ("family-a/same-topology-name",) * 4 + ("family-b/same-topology-name",) * 4
    result = evaluate_reliable_topology_coverage(assets, topology_ids=topology_ids, horizon_s=30.0)
    assert result["asset_count"] == 8
    assert result["passed_asset_count"] == 3
    assert result["passed_asset_rows"] == [0, 1, 4]
    assert result["topology_count"] == 2 and result["passed_topology_count"] == 1
    assert [row["required_asset_count"] for row in result["topology_results"]] == [2, 2]
    assert [row["passed"] for row in result["topology_results"]] == [True, False]


@pytest.mark.parametrize("replicas,horizon", [(1, 30.0), (16, 120.0)])
def test_reliable_coverage_requires_fixed30_r16(replicas: int, horizon: float) -> None:
    r"""单副本或120秒描述性结果不能直接获得30秒R16的入门覆盖。"""

    asset = replace(_physical_result(0, "mother", passed=True), replica_count=replicas)
    with pytest.raises(ValueError, match="30-second R16"):
        evaluate_reliable_topology_coverage((asset,), topology_ids=("leap/mother",), horizon_s=horizon)


def test_reliable_coverage_preserves_invalid_population_and_rejects_duplicate_assets() -> None:
    r"""非有限数据保留完整分母并标无效；重复资产不能增加拓扑代表票数。"""

    asset = _physical_result(0, "mother", passed=True)
    invalid = replace(asset, dataset_row=1, net_turns_median=float("nan"))
    result = evaluate_reliable_topology_coverage((asset, invalid), topology_ids=("leap/mother",) * 2, horizon_s=30.0)
    assert not result["finite"] and result["asset_count"] == 2
    assert result["passed_asset_count"] == result["passed_topology_count"] == 0
    with pytest.raises(ValueError, match="unique assets"):
        evaluate_reliable_topology_coverage((asset, asset), topology_ids=("leap/mother",) * 2, horizon_s=30.0)
