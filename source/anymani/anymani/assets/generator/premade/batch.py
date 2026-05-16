"""pre-made 批量枚举与并行调度。

这个模块只关心“如何把 pre-made 离散空间调度成一批样本”，不关心：

- cfg 字段定义
- 单样本 build / mutate / validate / export 的细节
- mutate-only 的 Monte Carlo 采样

这样 `hand_generator.py` 可以把 batch orchestration 与 single-sample pipeline
彻底拆开。
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path
from typing import TYPE_CHECKING

from ..result import HandGenerationResult

if TYPE_CHECKING:
    from ..hand_generator import HandGenerator, HandGeneratorCfg


@dataclass(frozen=True)
class PremadeTask:
    r"""一个可独立调度的 pre-made 离散样本任务。"""

    hand_preset_name: str
    connectivity_preset_name: str
    enumerated: bool


@dataclass
class PremadeWorkerResult:
    r"""worker 返回给主进程的最小结果包。"""

    result: HandGenerationResult | None
    rejection_stage: str | None = None


def build_premade_tasks(generator: "HandGenerator") -> list[PremadeTask]:
    r"""从当前 `HandGenerator` 展开 pre-made 任务表。"""

    tasks: list[PremadeTask] = []
    for hand_preset_name in generator._candidate_hand_preset_names():
        connectivity_names = generator._connectivity_names_for_hand_preset(hand_preset_name=hand_preset_name)
        for connectivity_preset_name in connectivity_names:
            tasks.append(
                PremadeTask(
                    hand_preset_name=hand_preset_name,
                    connectivity_preset_name=connectivity_preset_name,
                    enumerated=True,
                )
            )
    return tasks


def run_premade_serial(generator: "HandGenerator", *, tasks: list[PremadeTask]) -> list[HandGenerationResult]:
    r"""沿用顺序路径执行 pre-made 任务表。"""

    results: list[HandGenerationResult] = []
    success_limit = generator.cfg.max_enumerate
    for task in tasks:
        if success_limit is not None and len(results) >= success_limit:
            break
        result = generator._generate_once(
            hand_preset_name=task.hand_preset_name,
            connectivity_preset_name=task.connectivity_preset_name,
            enumerated=task.enumerated,
        )
        if result is not None:
            results.append(result)
    return results


def run_premade_parallel(generator: "HandGenerator", *, tasks: list[PremadeTask]) -> list[HandGenerationResult]:
    r"""用进程池执行 pre-made 样本级并行。"""

    if not tasks:
        return []

    run_context = generator._ensure_run_context()
    success_limit = generator.cfg.max_enumerate
    worker_count = infer_premade_parallel_worker_count(generator.cfg, task_count=len(tasks))
    if worker_count <= 1:
        return run_premade_serial(generator, tasks=tasks)

    ordered_results: list[HandGenerationResult] = []
    task_cursor = 0
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        while task_cursor < len(tasks):
            if success_limit is None:
                batch_size = len(tasks) - task_cursor
            else:
                remaining_success = success_limit - len(ordered_results)
                if remaining_success <= 0:
                    break
                batch_size = min(len(tasks) - task_cursor, max(remaining_success, worker_count))

            batch_tasks = tasks[task_cursor : task_cursor + batch_size]
            task_cursor += batch_size

            indexed_results: list[tuple[int, PremadeWorkerResult]] = []
            future_to_index = {
                executor.submit(_generate_premade_worker, generator.cfg, run_context.root_dir, task): index
                for index, task in enumerate(batch_tasks)
            }
            for future in as_completed(future_to_index):
                indexed_results.append((future_to_index[future], future.result()))

            for _, worker_result in sorted(indexed_results, key=lambda item: item[0]):
                if success_limit is not None and len(ordered_results) >= success_limit:
                    break
                result = record_premade_worker_result(generator, worker_result)
                if result is not None:
                    ordered_results.append(result)

    run_context.write_summary()
    return ordered_results


def infer_premade_parallel_worker_count(cfg: "HandGeneratorCfg", *, task_count: int) -> int:
    r"""计算 pre-made 样本级并行 worker 数。"""

    if task_count <= 0:
        return 1
    if cfg.premade_parallel_workers is not None:
        return max(1, min(int(cfg.premade_parallel_workers), task_count))
    cpu_count = os.cpu_count() or 2
    inferred_workers = max(cpu_count - 1, 1)  # 给 shell / IDE 留一个核心，避免全机卡死
    return max(1, min(inferred_workers, task_count))


def record_premade_worker_result(
    generator: "HandGenerator",
    worker_result: PremadeWorkerResult,
) -> HandGenerationResult | None:
    r"""把 worker 返回值并入主进程 summary。"""

    if worker_result.result is not None:
        generator._record_generation_success(worker_result.result, write_summary=False)
        return worker_result.result
    generator._record_generation_rejection(
        stage=worker_result.rejection_stage or "premade_worker_rejected",
        write_summary=False,
    )
    return None


def _generate_premade_worker(
    cfg: "HandGeneratorCfg",
    run_root: Path | str,
    task: PremadeTask,
) -> PremadeWorkerResult:
    r"""在独立 worker 中执行一个 pre-made 样本任务。"""

    from ..hand_generator import HandGenerator  # 局部导入，避免主模块导入时形成更长依赖链

    worker_generator = HandGenerator(cfg)  # worker 内部独立持有 runtime façade，避免共享可变状态
    worker_context = worker_generator._make_worker_run_context(Path(run_root))
    result = worker_generator._generate_once(
        hand_preset_name=task.hand_preset_name,
        connectivity_preset_name=task.connectivity_preset_name,
        enumerated=task.enumerated,
        record_summary=False,
    )
    return PremadeWorkerResult(result=result, rejection_stage=worker_context.last_rejection_stage)


__all__ = [
    "PremadeTask",
    "PremadeWorkerResult",
    "build_premade_tasks",
    "infer_premade_parallel_worker_count",
    "record_premade_worker_result",
    "run_premade_parallel",
    "run_premade_serial",
]
