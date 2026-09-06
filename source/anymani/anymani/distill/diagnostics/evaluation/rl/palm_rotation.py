r"""N000-relative MVP80能力门的纯数值定义。

对每个资产先在固定evaluation replicas上取trajectory中位数，再计算：

$$
S_i=\min\left(\frac{G_i}{G_0},\frac{N_i}{N_0}\right),\qquad
C_i=\frac{\max(0,\Psi_i)}{\sum_t|\Delta\psi_{i,t}|+\epsilon}.
$$

其中$G_i$是完整SO(3) orientation-keypoint与固定position-anchor双门的strict moving-goal命中数，
$N_i=\Psi_i/(2\pi)$是实际signed净圈数。二者分别衡量tracking与物理旋转，不构成恒等的30°计数关系。历史正式资产pass要求
$S_i\ge2/3$、$N_i\ge1$、$C_i\ge0.7$，且$G_i/12$与$N_i$在调用方显式给定的容差内一致；
single debug closure只使用1净圈、0.7方向性及多数replica不发生drop/axis的较低门。Cohort要求80项中
至少54项正式pass且8个handedness×tip×thumb cells各至少5/10；最终固定seeds 42/43/44至少2条独立通过，
不以额外seed替换失败seed。
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PalmRotationReference:
    r"""固定scale-1.1、ADR-0、120 s下的accepted N000参考能力。"""

    goal_count_median: float  # $G_0$，strict full-pose+position goal hits中位数
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
    goal_count_median: float  # $G_i$，strict moving-goal tracking hits
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


@dataclass(frozen=True)
class PalmRotationPhysicalAssetResult:
    r"""一个资产在固定ADR-0 first trajectories上的物理旋转能力。

    ``scale_ready_passed``只读取净圈、方向一致性和drop/axis联合生存率；strict moving-goal hit不进入判定。
    ``viability_passed``保留较低的一圈探索门，不能替代两圈scale-ready门。
    """

    dataset_row: int  # cohort运行时为selection-local index，source-qualified身份由上层artifact保存
    cell_id: int  # handedness×tip×thumb诊断cell，0..7
    mother_id: str  # cohort lock中的mother lineage稳定标签
    frontier_count_median: float  # $K_T$中位数，30°历史物理前沿数
    max_positive_net_turns_median: float  # $M_T/(2\pi)$中位数
    net_turns_median: float  # $\Psi_T/(2\pi)$中位数
    absolute_path_turns_median: float  # $\sum_t|\Delta\psi_t|/(2\pi)$中位数
    directional_consistency: float  # $\max(0,N_i)/(P_i+\epsilon)$
    safe_replica_fraction: float  # 未发生drop且未发生axis failure的联合比例
    replica_count: int
    finite: bool
    viability_passed: bool
    scale_ready_passed: bool
    failure_labels: tuple[str, ...]


@dataclass(frozen=True)
class PalmRotationScaleCohortResult:
    r"""A16/A64/A128固定规模晋级门，不混入strict tracking或人工批准。"""

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

    Moving goal在姿态容差内提前成功，position gate也可与周期姿态窗口形成相位锁定，故通常有
    $(G/12)/N\ne1$。历史一致性检查比较资产ratio与N000 ratio的相对偏差；该量属于strict tracking
    calibration，不是物理净圈真值。Tolerance必须由evaluation protocol显式保存。
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
    r"""归约任意非空single/few/full支持集的fixed first trajectories。

    该函数只形成逐资产N000-relative结果与完整finite证书，不定义支持集级成功比例。正式MVP80调用方仍需
    经过 :func:`evaluate_cohort` 的54/80与8-cell门；single-embodiment closure可独立读取净圈和方向一致性。

    Returns:
        tuple: ``(asset_results, finite_and_identity_valid)``，资产顺序与输入rows一致。
    """

    rows = tuple(int(value) for value in dataset_rows)  # 当前evaluation有序支持轴$[A]$
    cells = tuple(int(value) for value in cell_ids)  # handedness-inclusive cell$[A]$
    matrices = (
        goal_counts,
        net_turns,
        absolute_path_turns,
        termination_drop,
        termination_axis,
        termination_timeout,
    )
    asset_count = len(rows)  # $A=1$ closure或$A=80$正式cohort
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
            goal = net = path = (0.0,) * replica_count  # 保留支持轴并使该资产确定性失败
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
    r"""按资产归约固定replicas，并计算物理viability与scale-ready门。

    对资产$i$，先分别取净圈与绝对路径的trajectory中位数，再定义方向一致性：

    $$
    C_i=\frac{\max(0,\operatorname{med}_r N_{ir})}
    {\max(\operatorname{med}_r P_{ir},\epsilon)}.
    $$

    Scale-ready要求$N_i\ge2$、$C_i\ge0.85$、至少75% replicas同时无drop/axis；viability要求
    $N_i\ge1$、$C_i\ge0.7$、严格多数replicas同时安全。Strict goal count不属于输入，保证tracking与物理能力
    在证据层保持独立。

    Returns:
        tuple: ``(asset_results, finite_and_identity_valid)``，顺序与``dataset_rows``完全一致。
    """

    rows = tuple(int(value) for value in dataset_rows)  # 评估支持轴$[A]$
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
            frontier = maximum = net = path = (0.0,) * replica_count  # 保留支持轴并确定性判失败
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
    r"""按30秒R16的一圈可靠性与拓扑内半数代表，统计共享策略的入门覆盖。

    资产$i$的净圈$N_i$、方向$C_i$和安全比例$S_i$来自既有逐副本归约，判据为：
    $I_i=[N_i\ge1\land C_i\ge0.7\land S_i\ge0.75]$。拓扑$m$的代表集合为$\mathcal A_m$，
    至少$\lceil|\mathcal A_m|/2\rceil$项通过才覆盖该拓扑，4代表时为2项、8代表时为4项。
    ``topology_ids``应包含family/group及mother标签，防止同名跨family母体被合并。

    本函数只做确定性数值归约；调用方另核对TIP-only、首轨迹及无动作/观察干预的评价协议。
    它不读取旧viability/scale-ready布尔值，也不设全部拓扑必须通过的整体门。
    """

    results = tuple(asset_results)
    groups = tuple(topology_ids)  # 每资产一个完整拓扑标签，重复标签表示同拓扑的不同代表。
    if not results or len({result.dataset_row for result in results}) != len(results):
        raise ValueError("reliable topology coverage requires non-empty unique assets")
    if len(groups) != len(results) or any(not isinstance(group, str) or not group.strip() for group in groups):
        raise ValueError("reliable topology coverage requires one topology id per asset")
    if not math.isclose(horizon_s, 30.0, rel_tol=0.0, abs_tol=1e-8) or any(
        result.replica_count != 16 for result in results
    ):
        raise ValueError("reliable topology coverage requires the fixed 30-second R16 protocol")

    # 非有限或非法比例使本次统计无效，完整分母仍保留，不能通过删除坏行提高覆盖。
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
    total_by_topology = Counter(groups)  # 母体代表数，而非simulation replicas数量。
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
    r"""执行固定A16/A64/A128资产数与mother内3/4晋级门。

    A16只要求12/16资产；A64要求48/64且16条mother中至少12条有3/4成员通过；A128要求96/128且
    32条mother中至少24条有3/4成员通过。A64/A128每条mother必须恰含4项，否则cohort recipe本身无效。
    """

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
