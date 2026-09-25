"Runs pre-made topology tasks serially or with bounded CPU workers."

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..result import HandGenerationResult

if TYPE_CHECKING:
    from ..hand_generator import HandGenerator, HandGeneratorCfg


@dataclass(frozen=True)
class PremadeTask:
    "One topology and handedness combination scheduled for pre-made generation."

    hand_preset_name: str
    connectivity_preset_name: str
    enumerated: bool


@dataclass
class PremadeWorkerResult:
    "Result record with ordered geometry, provenance, and validation status."

    result: HandGenerationResult | None
    rejection_stage: str | None = None
    rejection_error_codes: tuple[str, ...] = ()


def build_premade_tasks(generator: HandGenerator) -> list[PremadeTask]:
    'Builds premade tasks.'

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


def run_premade_serial(generator: HandGenerator, *, tasks: list[PremadeTask]) -> list[HandGenerationResult]:
    'Runs premade serial.'

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


def run_premade_parallel(generator: HandGenerator, *, tasks: list[PremadeTask]) -> list[HandGenerationResult]:
    'Runs premade parallel.'

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


def infer_premade_parallel_worker_count(cfg: HandGeneratorCfg, *, task_count: int) -> int:
    'Infers premade parallel worker count.'

    if task_count <= 0:
        return 1
    if cfg.premade_parallel_workers is not None:
        return max(1, min(int(cfg.premade_parallel_workers), task_count))
    cpu_count = os.cpu_count() or 2
    inferred_workers = max(cpu_count - 1, 1)
    return max(1, min(inferred_workers, task_count))


def record_premade_worker_result(
    generator: HandGenerator,
    worker_result: PremadeWorkerResult,
) -> HandGenerationResult | None:
    'Records premade worker result.'

    if worker_result.result is not None:
        generator._record_generation_success(worker_result.result, write_summary=False)
        return worker_result.result
    generator._record_generation_rejection(
        stage=worker_result.rejection_stage or "premade_worker_rejected",
        error_codes=worker_result.rejection_error_codes,
        write_summary=False,
    )
    return None


def _generate_premade_worker(
    cfg: HandGeneratorCfg,
    run_root: Path | str,
    task: PremadeTask,
) -> PremadeWorkerResult:

    from ..hand_generator import HandGenerator

    worker_generator = HandGenerator(cfg)
    worker_context = worker_generator._make_worker_run_context(Path(run_root))
    result = worker_generator._generate_once(
        hand_preset_name=task.hand_preset_name,
        connectivity_preset_name=task.connectivity_preset_name,
        enumerated=task.enumerated,
        record_summary=False,
    )
    return PremadeWorkerResult(
        result=result,
        rejection_stage=worker_context.last_rejection_stage,
        rejection_error_codes=worker_context.last_rejection_error_codes,
    )


__all__ = [
    "PremadeTask",
    "PremadeWorkerResult",
    "build_premade_tasks",
    "infer_premade_parallel_worker_count",
    "record_premade_worker_result",
    "run_premade_parallel",
    "run_premade_serial",
]
