'Reduce fixed palm-rotation trajectories into per-asset capability and failure metrics.'

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PalmRotationReference:
    'Frozen 120-second N000 reference medians for goals and signed turns.'

    goal_count_median: float
    net_turns_median: float

    def __post_init__(self) -> None:
        'Validate the declared contract.'

        if not math.isfinite(self.goal_count_median) or self.goal_count_median <= 0.0:
            raise ValueError("N000 reference goal count must be finite and positive")
        if not math.isfinite(self.net_turns_median) or self.net_turns_median <= 0.0:
            raise ValueError("N000 reference net turns must be finite and positive")

    @property
    def command_turn_ratio(self) -> float:
        'Handle command turn ratio.'

        return self.goal_count_median / (12.0 * self.net_turns_median)


@dataclass(frozen=True)
class PalmRotationAssetResult:
    'Per-asset capability score and fixed-trajectory failures; score and consistency are dimensionless.'

    dataset_row: int  # formal ppo.yaml row
    cell_id: int
    goal_count_median: float
    net_turns_median: float
    absolute_path_turns_median: float
    score: float  # Dimensionless N000-relative capability score.
    directional_consistency: float  # Dimensionless consistency in [0,1].
    command_turn_ratio: float
    command_turn_ratio_relative_error: float
    relative_tier: str
    failure_labels: tuple[str, ...]  # reverse/jitter/drop/axis/value-failure
    passed: bool
    replica_count: int = 1
    drop_failure_rate: float = 0.0
    axis_failure_rate: float = 0.0
    timeout_rate: float = 0.0


@dataclass(frozen=True)
class PalmRotationCohortResult:
    'Contract for PALM rotation cohort result.'

    seed: int
    asset_results: tuple[PalmRotationAssetResult, ...]
    passed_assets: int
    passed_by_cell: tuple[int, ...]  # `[8]`
    finite_and_identity_valid: bool
    passed: bool


@dataclass(frozen=True)
class PalmRotationPairResult:
    'Contract for PALM rotation pair result.'

    pair_index: int
    left_dataset_row: int
    right_dataset_row: int
    left_passed: bool
    right_passed: bool
    outcome: str  # both_passed / left_only / right_only / both_failed
    score_gap_right_minus_left: float
    net_turn_gap_right_minus_left: float


@dataclass(frozen=True)
class PalmRotationPhysicalAssetResult:
    'Contract for PALM rotation physical asset result.'

    dataset_row: int
    cell_id: int
    mother_id: str
    frontier_count_median: float
    max_positive_net_turns_median: float  # units M
    net_turns_median: float
    absolute_path_turns_median: float
    directional_consistency: float
    safe_replica_fraction: float
    replica_count: int
    finite: bool
    viability_passed: bool
    scale_ready_passed: bool
    failure_labels: tuple[str, ...]


@dataclass(frozen=True)
class PalmRotationScaleCohortResult:
    'Contract for PALM rotation scale cohort result.'

    asset_count: int
    required_scale_ready_assets: int
    scale_ready_assets: int
    mother_count: int
    required_mothers_with_three_of_four: int
    mothers_with_three_of_four: int
    scale_ready_by_mother: tuple[tuple[str, int], ...]
    finite_and_identity_valid: bool
    passed: bool


def _relative_tier(score: float) -> str:
    'Assign a tier from a dimensionless N000-relative capability score.'

    if score <= 0.0:
        return "le_0"
    if score < 1.0 / 3.0:
        return "0_to_1_3"
    if score < 1.0 / 2.0:
        return "1_3_to_1_2"
    if score < 2.0 / 3.0:
        return "1_2_to_2_3"
    return "ge_2_3"


def evaluate_asset(
    *,
    dataset_row: int,
    cell_id: int,
    goal_count_median: float,
    net_turns_median: float,
    absolute_path_turns_median: float,
    reference: PalmRotationReference,
    command_turn_ratio_relative_tolerance: float,
    drop_failure: bool = False,
    axis_failure: bool = False,
    value_failure: bool = False,
    replica_count: int = 1,
    drop_failure_rate: float = 0.0,
    axis_failure_rate: float = 0.0,
    timeout_rate: float = 0.0,
) -> PalmRotationAssetResult:
    'Evaluate asset.'

    values = (
        goal_count_median,
        net_turns_median,
        absolute_path_turns_median,
        command_turn_ratio_relative_tolerance,
        drop_failure_rate,
        axis_failure_rate,
        timeout_rate,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("asset capability inputs must be finite")
    if dataset_row < 0 or cell_id not in range(8) or replica_count < 1:
        raise ValueError("asset evaluation requires non-negative row and cell_id in [0,7]")
    if (
        goal_count_median < 0.0
        or absolute_path_turns_median < 0.0
        or command_turn_ratio_relative_tolerance < 0.0
        or any(rate < 0.0 or rate > 1.0 for rate in (drop_failure_rate, axis_failure_rate, timeout_rate))
    ):
        raise ValueError("goal/path/tolerance values must be non-negative")

    score = min(
        goal_count_median / reference.goal_count_median,
        net_turns_median / reference.net_turns_median,
    )  # Dimensionless N000-relative capability score.
    directional_consistency = max(0.0, net_turns_median) / max(
        absolute_path_turns_median,
        float.fromhex("0x1.0p-23"),
    )
    directional_consistency = min(directional_consistency, 1.0)
    if net_turns_median > float.fromhex("0x1.0p-23"):
        command_turn_ratio = goal_count_median / (12.0 * net_turns_median)
        command_turn_ratio_relative_error = abs(command_turn_ratio / reference.command_turn_ratio - 1.0)
    else:
        command_turn_ratio = 0.0
        command_turn_ratio_relative_error = 1.0


    labels: list[str] = []
    if net_turns_median <= 0.0:
        labels.append("reverse")
    elif directional_consistency < 0.7:
        labels.append("jitter")
    if drop_failure:
        labels.append("drop")
    if axis_failure:
        labels.append("axis")
    if value_failure:
        labels.append("value-failure")
    passed = (
        score >= 2.0 / 3.0
        and net_turns_median >= 1.0
        and directional_consistency >= 0.7
        and command_turn_ratio_relative_error <= command_turn_ratio_relative_tolerance
        and not labels
    )
    return PalmRotationAssetResult(
        dataset_row=dataset_row,
        cell_id=cell_id,
        goal_count_median=goal_count_median,
        net_turns_median=net_turns_median,
        absolute_path_turns_median=absolute_path_turns_median,
        score=score,
        directional_consistency=directional_consistency,
        command_turn_ratio=command_turn_ratio,
        command_turn_ratio_relative_error=command_turn_ratio_relative_error,
        relative_tier=_relative_tier(score),
        failure_labels=tuple(labels),
        passed=passed,
        replica_count=int(replica_count),
        drop_failure_rate=float(drop_failure_rate),
        axis_failure_rate=float(axis_failure_rate),
        timeout_rate=float(timeout_rate),
    )


def evaluate_trajectory_medians(
    *,
    seed: int,
    dataset_rows: Sequence[int],
    cell_ids: Sequence[int],
    goal_counts: Sequence[Sequence[float]],
    net_turns: Sequence[Sequence[float]],
    absolute_path_turns: Sequence[Sequence[float]],
    termination_drop: Sequence[Sequence[bool]],
    termination_axis: Sequence[Sequence[bool]],
    termination_timeout: Sequence[Sequence[bool]],
    reference: PalmRotationReference,
    command_turn_ratio_relative_tolerance: float = 0.10,
) -> PalmRotationCohortResult:
    'Evaluate trajectory medians.'

    asset_results, finite_and_identity_valid = evaluate_support_trajectory_medians(
        dataset_rows=dataset_rows,
        cell_ids=cell_ids,
        goal_counts=goal_counts,
        net_turns=net_turns,
        absolute_path_turns=absolute_path_turns,
        termination_drop=termination_drop,
        termination_axis=termination_axis,
        termination_timeout=termination_timeout,
        reference=reference,
        command_turn_ratio_relative_tolerance=command_turn_ratio_relative_tolerance,
    )
    return evaluate_cohort(
        seed=seed,
        asset_results=asset_results,
        finite_and_identity_valid=finite_and_identity_valid,
    )


def evaluate_support_trajectory_medians(
    *,
    dataset_rows: Sequence[int],
    cell_ids: Sequence[int],
    goal_counts: Sequence[Sequence[float]],
    net_turns: Sequence[Sequence[float]],
    absolute_path_turns: Sequence[Sequence[float]],
    termination_drop: Sequence[Sequence[bool]],
    termination_axis: Sequence[Sequence[bool]],
    termination_timeout: Sequence[Sequence[bool]],
    reference: PalmRotationReference,
    command_turn_ratio_relative_tolerance: float = 0.10,
) -> tuple[tuple[PalmRotationAssetResult, ...], bool]:
    'Evaluate support trajectory medians.'

    rows = tuple(int(value) for value in dataset_rows)  # shapes [A]
    cells = tuple(int(value) for value in cell_ids)  # shapes [A]
    matrices = (
        goal_counts,
        net_turns,
        absolute_path_turns,
        termination_drop,
        termination_axis,
        termination_timeout,
    )
    asset_count = len(rows)
    if asset_count < 1 or len(set(rows)) != asset_count or len(cells) != asset_count:
        raise ValueError("trajectory evaluation requires non-empty unique rows aligned with cell labels")
    if any(cell not in range(8) for cell in cells):
        raise ValueError("trajectory evaluation cell labels must lie in [0,7]")
    if any(len(matrix) != asset_count for matrix in matrices):
        raise ValueError("trajectory evaluation matrices must share the selected asset axis")
    replica_counts = {len(row) for matrix in matrices for row in matrix}
    if len(replica_counts) != 1 or not replica_counts or next(iter(replica_counts)) < 1:
        raise ValueError("trajectory evaluation requires one positive shared replica count")
    replica_count = next(iter(replica_counts))


    finite_and_identity_valid = True
    asset_results: list[PalmRotationAssetResult] = []
    for asset_index, (dataset_row, cell_id) in enumerate(zip(rows, cells, strict=True)):
        goal = tuple(float(value) for value in goal_counts[asset_index])
        net = tuple(float(value) for value in net_turns[asset_index])
        path = tuple(float(value) for value in absolute_path_turns[asset_index])
        asset_finite = all(math.isfinite(value) for value in (*goal, *net, *path))
        finite_and_identity_valid &= asset_finite
        if not asset_finite:
            goal = net = path = (0.0,) * replica_count
        drop = tuple(bool(value) for value in termination_drop[asset_index])
        axis = tuple(bool(value) for value in termination_axis[asset_index])
        timeout = tuple(bool(value) for value in termination_timeout[asset_index])
        drop_rate = sum(drop) / replica_count
        axis_rate = sum(axis) / replica_count
        timeout_rate = sum(timeout) / replica_count
        asset_results.append(
            evaluate_asset(
                dataset_row=dataset_row,
                cell_id=cell_id,
                goal_count_median=statistics.median(goal),
                net_turns_median=statistics.median(net),
                absolute_path_turns_median=statistics.median(path),
                reference=reference,
                command_turn_ratio_relative_tolerance=command_turn_ratio_relative_tolerance,
                drop_failure=drop_rate >= 0.5,
                axis_failure=axis_rate >= 0.5,
                replica_count=replica_count,
                drop_failure_rate=drop_rate,
                axis_failure_rate=axis_rate,
                timeout_rate=timeout_rate,
            )
        )
    return tuple(asset_results), bool(finite_and_identity_valid)


def evaluate_physical_support_trajectory_medians(
    *,
    dataset_rows: Sequence[int],
    cell_ids: Sequence[int],
    mother_ids: Sequence[str],
    frontier_counts: Sequence[Sequence[float]],
    max_positive_net_turns: Sequence[Sequence[float]],
    net_turns: Sequence[Sequence[float]],
    absolute_path_turns: Sequence[Sequence[float]],
    termination_drop: Sequence[Sequence[bool]],
    termination_axis: Sequence[Sequence[bool]],
) -> tuple[tuple[PalmRotationPhysicalAssetResult, ...], bool]:
    'Evaluate physical support trajectory medians.'

    rows = tuple(int(value) for value in dataset_rows)  # shapes [A]
    cells = tuple(int(value) for value in cell_ids)
    mothers = tuple(str(value).strip() for value in mother_ids)
    matrices = (
        frontier_counts,
        max_positive_net_turns,
        net_turns,
        absolute_path_turns,
        termination_drop,
        termination_axis,
    )
    asset_count = len(rows)
    if asset_count < 1 or len(set(rows)) != asset_count:
        raise ValueError("physical trajectory evaluation requires non-empty unique asset rows")
    if len(cells) != asset_count or any(cell not in range(8) for cell in cells):
        raise ValueError("physical trajectory evaluation requires one cell_id in [0,7] per asset")
    if len(mothers) != asset_count or any(not mother for mother in mothers):
        raise ValueError("physical trajectory evaluation requires one non-empty mother ID per asset")
    if any(len(matrix) != asset_count for matrix in matrices):
        raise ValueError("physical trajectory matrices must share the selected asset axis")
    replica_counts = {len(row) for matrix in matrices for row in matrix}
    if len(replica_counts) != 1 or not replica_counts or next(iter(replica_counts)) < 1:
        raise ValueError("physical trajectory evaluation requires one positive shared replica count")
    replica_count = next(iter(replica_counts))

    finite_and_identity_valid = True
    results: list[PalmRotationPhysicalAssetResult] = []
    for asset_index, (dataset_row, cell_id, mother_id) in enumerate(zip(rows, cells, mothers, strict=True)):
        frontier = tuple(float(value) for value in frontier_counts[asset_index])
        maximum = tuple(float(value) for value in max_positive_net_turns[asset_index])
        net = tuple(float(value) for value in net_turns[asset_index])
        path = tuple(float(value) for value in absolute_path_turns[asset_index])
        finite = all(math.isfinite(value) for value in (*frontier, *maximum, *net, *path))
        finite_and_identity_valid &= finite
        if not finite:
            frontier = maximum = net = path = (0.0,) * replica_count
        if any(value < 0.0 for value in (*frontier, *maximum, *path)):
            raise ValueError("frontier, maximum-positive and absolute-path values must be non-negative")

        drop = tuple(bool(value) for value in termination_drop[asset_index])
        axis = tuple(bool(value) for value in termination_axis[asset_index])
        safe_fraction = (
            sum(not (drop_bit or axis_bit) for drop_bit, axis_bit in zip(drop, axis, strict=True)) / replica_count
        )
        frontier_median = statistics.median(frontier)
        maximum_median = statistics.median(maximum)
        net_median = statistics.median(net)
        path_median = statistics.median(path)
        directional = min(
            max(0.0, net_median) / max(path_median, float.fromhex("0x1.0p-23")),
            1.0,
        )
        viability = finite and net_median >= 1.0 and directional >= 0.7 and safe_fraction > 0.5
        scale_ready = finite and net_median >= 2.0 and directional >= 0.85 and safe_fraction >= 0.75
        labels: list[str] = []
        if net_median < 2.0:
            labels.append("net-turns-below-two")
        if directional < 0.85:
            labels.append("directional-consistency-below-0p85")
        if safe_fraction < 0.75:
            labels.append("safe-replica-fraction-below-0p75")
        if not finite:
            labels.append("non-finite")
        results.append(
            PalmRotationPhysicalAssetResult(
                dataset_row=dataset_row,
                cell_id=cell_id,
                mother_id=mother_id,
                frontier_count_median=frontier_median,
                max_positive_net_turns_median=maximum_median,
                net_turns_median=net_median,
                absolute_path_turns_median=path_median,
                directional_consistency=directional,
                safe_replica_fraction=safe_fraction,
                replica_count=replica_count,
                finite=finite,
                viability_passed=viability,
                scale_ready_passed=scale_ready,
                failure_labels=tuple(labels),
            )
        )
    return tuple(results), bool(finite_and_identity_valid)


def evaluate_reliable_topology_coverage(
    asset_results: Sequence[PalmRotationPhysicalAssetResult],
    *,
    topology_ids: Sequence[str],
    horizon_s: float,
) -> dict[str, Any]:
    'Measure reliable topology coverage by asset and morphology; position errors use metres.'

    results = tuple(asset_results)
    groups = tuple(topology_ids)
    if not results or len({result.dataset_row for result in results}) != len(results):
        raise ValueError("reliable topology coverage requires non-empty unique assets")
    if len(groups) != len(results) or any(not isinstance(group, str) or not group.strip() for group in groups):
        raise ValueError("reliable topology coverage requires one topology id per asset")
    if not math.isclose(horizon_s, 30.0, rel_tol=0.0, abs_tol=1e-8) or any(
        result.replica_count != 16 for result in results
    ):
        raise ValueError("reliable topology coverage requires the fixed 30-second R16 protocol")


    finite = all(
        result.finite
        and all(
            math.isfinite(value)
            for value in (result.net_turns_median, result.directional_consistency, result.safe_replica_fraction)
        )
        and 0.0 <= result.directional_consistency <= 1.0
        and 0.0 <= result.safe_replica_fraction <= 1.0
        for result in results
    )
    passed = tuple(
        finite
        and result.net_turns_median >= 1.0
        and result.directional_consistency >= 0.7
        and result.safe_replica_fraction >= 0.75
        for result in results
    )
    total_by_topology = Counter(groups)
    passed_by_topology = Counter(group for group, accepted in zip(groups, passed, strict=True) if accepted)
    topology_results = [
        {
            "topology_id": group,
            "asset_count": count,
            "passed_asset_count": passed_by_topology[group],
            "required_asset_count": (count + 1) // 2,
            "passed": passed_by_topology[group] >= (count + 1) // 2,
        }
        for group, count in sorted(total_by_topology.items())
    ]
    return {
        "schema_version": "1.0.0",
        "thresholds": {
            "horizon_s": 30.0,
            "replicas_per_asset": 16,
            "net_turns_min": 1.0,
            "directional_consistency_min": 0.7,
            "safe_replica_fraction_min": 0.75,
            "topology_representative_fraction_min": 0.5,
        },
        "finite": finite,
        "asset_count": len(results),
        "passed_asset_count": sum(passed),
        "passed_asset_rows": [result.dataset_row for result, accepted in zip(results, passed, strict=True) if accepted],
        "topology_count": len(topology_results),
        "passed_topology_count": sum(row["passed"] for row in topology_results),
        "topology_results": topology_results,
    }


def evaluate_scale_ladder_cohort(
    asset_results: Sequence[PalmRotationPhysicalAssetResult],
    *,
    finite_and_identity_valid: bool,
) -> PalmRotationScaleCohortResult:
    'Evaluate scale ladder cohort.'

    results = tuple(asset_results)
    requirements = {16: (12, 0), 64: (48, 12), 128: (96, 24)}
    asset_count = len(results)
    if asset_count not in requirements or len({result.dataset_row for result in results}) != asset_count:
        raise ValueError("scale ladder cohort must contain exactly 16, 64 or 128 unique assets")
    required_assets, required_mothers = requirements[asset_count]
    by_mother: dict[str, list[PalmRotationPhysicalAssetResult]] = {}
    for result in results:
        by_mother.setdefault(result.mother_id, []).append(result)
    if asset_count in (64, 128) and any(len(members) != 4 for members in by_mother.values()):
        raise ValueError("A64/A128 scale cohort requires exactly four members per mother lineage")
    passed_by_mother = tuple(
        sorted(
            (mother_id, sum(member.scale_ready_passed for member in members))
            for mother_id, members in by_mother.items()
        )
    )
    passed_assets = sum(result.scale_ready_passed for result in results)
    passed_mothers = sum(count >= 3 for _, count in passed_by_mother)
    passed = bool(
        finite_and_identity_valid
        and passed_assets >= required_assets
        and (required_mothers == 0 or passed_mothers >= required_mothers)
    )
    return PalmRotationScaleCohortResult(
        asset_count=asset_count,
        required_scale_ready_assets=required_assets,
        scale_ready_assets=passed_assets,
        mother_count=len(by_mother),
        required_mothers_with_three_of_four=required_mothers,
        mothers_with_three_of_four=passed_mothers,
        scale_ready_by_mother=passed_by_mother,
        finite_and_identity_valid=bool(finite_and_identity_valid),
        passed=passed,
    )


def evaluate_pairs(
    asset_results: Sequence[PalmRotationAssetResult],
    pairs: Sequence[tuple[int, int]],
) -> tuple[PalmRotationPairResult, ...]:
    'Evaluate pairs.'

    by_row = {result.dataset_row: result for result in asset_results}
    if len(by_row) != len(asset_results):
        raise ValueError("pair diagnostics require unique asset rows")
    output: list[PalmRotationPairResult] = []
    for pair_index, (left_row, right_row) in enumerate(pairs):
        if left_row not in by_row or right_row not in by_row:
            raise ValueError("pair diagnostics reference a row outside the evaluated cohort")
        left = by_row[left_row]
        right = by_row[right_row]
        outcome = (
            "both_passed"
            if left.passed and right.passed
            else "left_only"
            if left.passed
            else "right_only"
            if right.passed
            else "both_failed"
        )
        output.append(
            PalmRotationPairResult(
                pair_index=pair_index,
                left_dataset_row=left_row,
                right_dataset_row=right_row,
                left_passed=left.passed,
                right_passed=right.passed,
                outcome=outcome,
                score_gap_right_minus_left=right.score - left.score,
                net_turn_gap_right_minus_left=right.net_turns_median - left.net_turns_median,
            )
        )
    return tuple(output)


def evaluate_cohort(
    *,
    seed: int,
    asset_results: Sequence[PalmRotationAssetResult],
    finite_and_identity_valid: bool,
) -> PalmRotationCohortResult:
    'Evaluate cohort.'

    results = tuple(asset_results)
    if len(results) != 80 or len({result.dataset_row for result in results}) != 80:
        raise ValueError("cohort evaluation requires exactly 80 unique assets")
    cell_population = Counter(result.cell_id for result in results)
    if cell_population != Counter({cell: 10 for cell in range(8)}):
        raise ValueError(f"cohort evaluation requires 10 assets per cell, got {dict(cell_population)}")
    passed_assets = sum(result.passed for result in results)
    passed_by_cell = tuple(sum(result.passed for result in results if result.cell_id == cell) for cell in range(8))
    passed = bool(finite_and_identity_valid and passed_assets >= 54 and all(count >= 5 for count in passed_by_cell))
    return PalmRotationCohortResult(
        seed=int(seed),
        asset_results=results,
        passed_assets=passed_assets,
        passed_by_cell=passed_by_cell,
        finite_and_identity_valid=bool(finite_and_identity_valid),
        passed=passed,
    )


def evaluate_seed_confirmation(results_by_seed: Mapping[int, PalmRotationCohortResult]) -> bool:
    'Evaluate seed confirmation.'

    if set(results_by_seed) != {42, 43, 44}:
        raise ValueError("final MVP confirmation requires exactly seeds 42, 43 and 44")
    if any(result.seed != seed for seed, result in results_by_seed.items()):
        raise ValueError("cohort result seed labels disagree with mapping keys")
    return sum(result.passed for result in results_by_seed.values()) >= 2


__all__ = [
    "PalmRotationAssetResult",
    "PalmRotationCohortResult",
    "PalmRotationPairResult",
    "PalmRotationPhysicalAssetResult",
    "PalmRotationReference",
    "PalmRotationScaleCohortResult",
    "evaluate_asset",
    "evaluate_cohort",
    "evaluate_pairs",
    "evaluate_physical_support_trajectory_medians",
    "evaluate_reliable_topology_coverage",
    "evaluate_scale_ladder_cohort",
    "evaluate_seed_confirmation",
    "evaluate_support_trajectory_medians",
    "evaluate_trajectory_medians",
]
