r"""N000-relative MVP80能力门的纯数值定义。

对每个资产先在固定evaluation replicas上取trajectory中位数，再计算：

$$
S_i=\min\left(\frac{G_i}{G_0},\frac{N_i}{N_0}\right),\qquad
C_i=\frac{\max(0,\Psi_i)}{\sum_t|\Delta\psi_{i,t}|+\epsilon}.
$$

其中$G_i$是连续30°目标数，$N_i=\Psi_i/(2\pi)$是实际signed净圈数。单资产还要求
$N_i\ge1$、$C_i\ge0.7$，且$G_i/12$与$N_i$在调用方显式给定的容差内一致。Cohort要求
80项中至少54项通过且8个handedness×tip×thumb cells各至少5/10；最终固定seeds 42/43/44
至少2条独立通过，不以额外seed替换失败seed。
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PalmRotationReference:
    r"""固定scale-1.1、ADR-0、120 s下的accepted N000参考能力。"""

    goal_count_median: float  # $G_0$，连续30°目标数中位数
    net_turns_median: float  # $N_0$，实际signed净圈数中位数

    def __post_init__(self) -> None:
        r"""参考值必须finite且严格为正，否则相对能力比例无定义。"""

        if not math.isfinite(self.goal_count_median) or self.goal_count_median <= 0.0:
            raise ValueError("N000 reference goal count must be finite and positive")
        if not math.isfinite(self.net_turns_median) or self.net_turns_median <= 0.0:
            raise ValueError("N000 reference net turns must be finite and positive")

    @property
    def command_turn_ratio(self) -> float:
        r"""返回N000的$(G_0/12)/N_0$ moving-goal提前触发校准比例。"""

        return self.goal_count_median / (12.0 * self.net_turns_median)


@dataclass(frozen=True)
class PalmRotationAssetResult:
    r"""一个资产的中位数能力、失败标签与最终pass判定。"""

    dataset_row: int  # formal ppo.yaml row
    cell_id: int  # handedness×tip×thumb cell，0..7
    goal_count_median: float  # $G_i$
    net_turns_median: float  # $N_i$
    absolute_path_turns_median: float  # $\sum_t|\Delta\psi_t|/(2\pi)$
    score: float  # $S_i$
    directional_consistency: float  # $C_i\in[0,1]$
    command_turn_ratio: float  # $(G_i/12)/N_i$；正向净圈为0时置0
    command_turn_ratio_relative_error: float  # 相对N000 ratio的无量纲偏差
    relative_tier: str  # N000-relative能力层级
    failure_labels: tuple[str, ...]  # reverse/jitter/drop/axis/value-failure
    passed: bool
    replica_count: int = 1
    drop_failure_rate: float = 0.0
    axis_failure_rate: float = 0.0
    timeout_rate: float = 0.0


@dataclass(frozen=True)
class PalmRotationCohortResult:
    r"""一条training seed的80-asset/8-cell能力门结果。"""

    seed: int
    asset_results: tuple[PalmRotationAssetResult, ...]
    passed_assets: int
    passed_by_cell: tuple[int, ...]  # `[8]`
    finite_and_identity_valid: bool
    passed: bool


@dataclass(frozen=True)
class PalmRotationPairResult:
    r"""一组left/right资产的能力对称性诊断；不参与cohort硬门。"""

    pair_index: int
    left_dataset_row: int
    right_dataset_row: int
    left_passed: bool
    right_passed: bool
    outcome: str  # both_passed / left_only / right_only / both_failed
    score_gap_right_minus_left: float
    net_turn_gap_right_minus_left: float


def _relative_tier(score: float) -> str:
    r"""按计划边界把$S_i$映射到互斥层级。"""

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
    r"""计算一个资产的N000-relative score、方向质量与能力门。

    Moving goal在姿态容差内提前成功，故N000本身通常有$(G_0/12)/N_0\ne1$。一致性检查比较资产
    ratio与N000 ratio的相对偏差；tolerance必须由evaluation protocol显式保存。
    """

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
    )  # $S_i$保留negative reverse结果，不在ratio前clamp
    directional_consistency = max(0.0, net_turns_median) / max(
        absolute_path_turns_median,
        float.fromhex("0x1.0p-23"),
    )  # $C_i$；float32 epsilon只防零路径除法
    directional_consistency = min(directional_consistency, 1.0)  # 数值累计误差不允许产生$C_i>1$
    if net_turns_median > float.fromhex("0x1.0p-23"):
        command_turn_ratio = goal_count_median / (12.0 * net_turns_median)
        command_turn_ratio_relative_error = abs(command_turn_ratio / reference.command_turn_ratio - 1.0)
    else:
        command_turn_ratio = 0.0
        command_turn_ratio_relative_error = 1.0  # 无正向净圈时ratio无定义且必不通过

    # Failure标签区分反向净运动、正向但抖动、物理终止和critic异常；标签不替代数值门。
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
    r"""把固定replicas的first-trajectory数组归约为80项中位数能力门。

    每个资产的$G_i,N_i,\sum_t|\Delta\psi|/(2\pi)$分别沿replica轴取中位数；drop/axis在至少一半
    replicas失败时形成asset failure label。任何非有限trajectory数值都会使整条seed fail closed，而不是在
    ``median``前静默删除。
    """

    rows = tuple(int(value) for value in dataset_rows)
    cells = tuple(int(value) for value in cell_ids)
    matrices = (
        goal_counts,
        net_turns,
        absolute_path_turns,
        termination_drop,
        termination_axis,
        termination_timeout,
    )
    if len(rows) != 80 or len(set(rows)) != 80 or len(cells) != 80:
        raise ValueError("trajectory evaluation requires 80 unique rows and 80 cell labels")
    if any(len(matrix) != 80 for matrix in matrices):
        raise ValueError("trajectory evaluation matrices must share the 80-asset axis")
    replica_counts = {len(row) for matrix in matrices for row in matrix}
    if len(replica_counts) != 1 or not replica_counts or next(iter(replica_counts)) < 1:
        raise ValueError("trajectory evaluation requires one positive shared replica count")
    replica_count = next(iter(replica_counts))

    # 数值矩阵必须完整finite；binary termination matrices必须只含bool/0/1。
    finite_and_identity_valid = True
    asset_results: list[PalmRotationAssetResult] = []
    for asset_index, (dataset_row, cell_id) in enumerate(zip(rows, cells, strict=True)):
        goal = tuple(float(value) for value in goal_counts[asset_index])
        net = tuple(float(value) for value in net_turns[asset_index])
        path = tuple(float(value) for value in absolute_path_turns[asset_index])
        asset_finite = all(math.isfinite(value) for value in (*goal, *net, *path))
        finite_and_identity_valid &= asset_finite
        if not asset_finite:
            goal = net = path = (0.0,) * replica_count  # 保留80项axis并使该资产确定性失败
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
    return evaluate_cohort(
        seed=seed,
        asset_results=asset_results,
        finite_and_identity_valid=finite_and_identity_valid,
    )


def evaluate_pairs(
    asset_results: Sequence[PalmRotationAssetResult],
    pairs: Sequence[tuple[int, int]],
) -> tuple[PalmRotationPairResult, ...]:
    r"""按manifest left/right rows形成不参与硬门的反射一致性诊断。"""

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
    r"""执行单seed的54/80与每cell 5/10 cohort门。"""

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
    r"""要求固定42/43/44中至少两条seed独立通过，不接受替补seed。"""

    if set(results_by_seed) != {42, 43, 44}:
        raise ValueError("final MVP confirmation requires exactly seeds 42, 43 and 44")
    if any(result.seed != seed for seed, result in results_by_seed.items()):
        raise ValueError("cohort result seed labels disagree with mapping keys")
    return sum(result.passed for result in results_by_seed.values()) >= 2


__all__ = [
    "PalmRotationAssetResult",
    "PalmRotationCohortResult",
    "PalmRotationPairResult",
    "PalmRotationReference",
    "evaluate_asset",
    "evaluate_cohort",
    "evaluate_pairs",
    "evaluate_seed_confirmation",
    "evaluate_trajectory_medians",
]
