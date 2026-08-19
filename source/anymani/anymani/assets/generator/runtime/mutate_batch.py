r"""多个 mother variant sets 的 source-level post-mutate 调度。

并行原子是一只 mother 的完整 variant set，而不是单个 variant。这样一个 worker
顺序拥有 source restore、RNG、run summary、shared mesh directory 与 rejection sampling，
不同 worker 只写各自 mother 根，不共享可变 generator 状态。
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..hand_generator import HandGenerator, HandGeneratorCfg, PostMutateSourceCfg


@dataclass(frozen=True)
class PostMutateVariantSetResult:
    r"""一个 source task 的轻量落盘报告。

    worker 不把完整 ``HandCfg`` 列表传回主进程，只返回 dataset builder 需要的 run
    identity、数量与 bundle 路径，避免数万项资产在 process boundary 重复序列化。
    """

    task_id: str
    source_topology_dir: Path
    run_dir: Path
    planned_variants: int
    successful_variants: int
    shortfall: int
    mutation_seed: int
    sidecar_paths: tuple[Path, ...]
    urdf_paths: tuple[Path, ...]
    error: str = ""


def run_post_mutate_source_batch(
    generator: HandGenerator,
    *,
    tasks: tuple[PostMutateSourceCfg, ...],
) -> tuple[PostMutateVariantSetResult, ...]:
    r"""按 cfg 选择串行或 mother-level process parallel 执行 source tasks。"""

    if not tasks:
        return ()
    worker_count = infer_post_mutate_worker_count(generator.cfg, task_count=len(tasks))
    if not generator.cfg.post_mutate_parallel or worker_count <= 1:
        return tuple(_generate_variant_set_safe(generator.cfg, task) for task in tasks)

    indexed: list[tuple[int, PostMutateVariantSetResult]] = []
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_index = {
            executor.submit(_generate_variant_set_safe, generator.cfg, task): index
            for index, task in enumerate(tasks)
        }
        for future in as_completed(future_to_index):
            indexed.append((future_to_index[future], future.result()))
    return tuple(report for _, report in sorted(indexed, key=lambda item: item[0]))


def infer_post_mutate_worker_count(cfg: HandGeneratorCfg, *, task_count: int) -> int:
    r"""计算 conservative mother-level worker 数。

    post-mutate worker 会执行 trimesh、physics closure 与多文件导出，默认上限取 8，
    不直接沿用 pre-made 的 ``cpu_count-1``，避免内存和磁盘并发在未 profile 时失控。
    """

    if task_count <= 0:
        return 1
    if cfg.post_mutate_parallel_workers is not None:
        return max(1, min(int(cfg.post_mutate_parallel_workers), task_count))
    cpu_count = os.cpu_count() or 2
    return max(1, min(cpu_count - 1, 8, task_count))


def _generate_variant_set_worker(
    parent_cfg: HandGeneratorCfg,
    task: PostMutateSourceCfg,
) -> PostMutateVariantSetResult:
    r"""在独立 generator 中执行一个 mother 的完整 variant-set run。"""

    from ..hand_generator import HandGenerator

    child_cfg = parent_cfg.replace(
        source_topology_dir=Path(task.source_topology_dir),
        post_mutate_sources=[],
        n_samples=int(task.n_samples),
        post_mutate_seed=int(task.seed),
        post_mutate_parallel=False,
        post_mutate_parallel_workers=None,
    )
    child_generator = HandGenerator(child_cfg)
    results = list(child_generator.generate_batch())
    context = child_generator._ensure_run_context()
    sidecars = tuple(result.sidecar_path for result in results if result.sidecar_path is not None)
    urdfs = tuple(result.urdf_path for result in results if result.urdf_path is not None)
    successful = len(results)
    return PostMutateVariantSetResult(
        task_id=task.task_id,
        source_topology_dir=Path(task.source_topology_dir),
        run_dir=context.root_dir,
        planned_variants=int(task.n_samples),
        successful_variants=successful,
        shortfall=int(task.n_samples) - successful,
        mutation_seed=int(task.seed),
        sidecar_paths=sidecars,
        urdf_paths=urdfs,
    )


def _generate_variant_set_safe(
    parent_cfg: HandGeneratorCfg,
    task: PostMutateSourceCfg,
) -> PostMutateVariantSetResult:
    r"""把单 task 异常规约成失败报告，使同批其它 mother 仍可完成。"""

    try:
        return _generate_variant_set_worker(parent_cfg, task)
    except Exception as exc:
        return PostMutateVariantSetResult(
            task_id=task.task_id,
            source_topology_dir=Path(task.source_topology_dir),
            run_dir=Path(),
            planned_variants=int(task.n_samples),
            successful_variants=0,
            shortfall=int(task.n_samples),
            mutation_seed=int(task.seed),
            sidecar_paths=(),
            urdf_paths=(),
            error=f"{type(exc).__name__}: {exc}",
        )


__all__ = [
    "PostMutateVariantSetResult",
    "infer_post_mutate_worker_count",
    "run_post_mutate_source_batch",
]
