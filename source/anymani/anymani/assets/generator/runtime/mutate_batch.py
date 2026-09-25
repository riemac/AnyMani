"Runs bounded mutation batches while preserving per-candidate provenance."

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..hand_generator import HandGenerator, HandGeneratorCfg, PostMutateSourceCfg


DATASET_BUILD_ATTEMPT_FILENAME = "DATASET_BUILD_ATTEMPT.yaml"
"Marker that identifies runs owned by one dataset-build invocation."


@dataclass(frozen=True)
class PostMutateVariantSetResult:
    "Result record with ordered geometry, provenance, and validation status."

    task_id: str
    source_topology_dir: Path
    run_dir: Path
    planned_variants: int
    successful_variants: int
    shortfall: int
    mutation_seed: int
    sidecar_paths: tuple[Path, ...]
    urdf_paths: tuple[Path, ...]
    worker_pid: int = 0
    "Stable semantic identifier preserved in the output metadata."

    worker_cuda_initialized: bool = False
    "Whether a worker initialized CUDA; CPU-only dataset workers require false."

    sdf_service_pid: int = 0
    "Stable semantic identifier preserved in the output metadata."

    error: str = ""


def run_post_mutate_source_batch(
    generator: HandGenerator,
    *,
    tasks: tuple[PostMutateSourceCfg, ...],
    on_report: Callable[[PostMutateVariantSetResult], None] | None = None,
) -> tuple[PostMutateVariantSetResult, ...]:
    'Runs post mutate source batch.'

    if not tasks:
        return ()
    worker_count = infer_post_mutate_worker_count(generator.cfg, task_count=len(tasks))
    central_gpu = generator.cfg.post_mutate_sdf_execution == "central_gpu_batch"
    if not central_gpu and (not generator.cfg.post_mutate_parallel or worker_count <= 1):
        reports = []
        for task in tasks:
            report = _generate_variant_set_safe(generator.cfg, task)
            reports.append(report)
            if on_report is not None:
                on_report(report)
        return tuple(reports)

    context = mp.get_context("spawn")
    request_queue = None
    service_process = None
    if central_gpu:
        from ...validator._sdf_service import run_sdf_service

        request_queue = context.Queue()
        startup_receive, startup_send = context.Pipe(duplex=False)
        service_process = context.Process(
            target=run_sdf_service,
            args=(request_queue, startup_send),
            kwargs={"batch_size": worker_count},
            name="anymani-sdf-gpu-service",
        )
        service_process.start()
        startup_send.close()
        if not startup_receive.poll(60.0):
            startup_receive.close()
            service_process.terminate()
            service_process.join(timeout=10.0)
            raise RuntimeError("central GPU SDF service startup timed out")
        startup = startup_receive.recv()
        startup_receive.close()
        if not isinstance(startup, dict) or not startup.get("ok"):
            service_process.join(timeout=5.0)
            error = startup.get("error", "unknown error") if isinstance(startup, dict) else repr(startup)
            raise RuntimeError(f"central GPU SDF service failed during startup: {error}")
        if not service_process.is_alive() or int(startup.get("pid", 0)) != int(service_process.pid or 0):
            service_process.join(timeout=5.0)
            raise RuntimeError("central GPU SDF service exited immediately after startup")

    indexed: list[tuple[int, PostMutateVariantSetResult]] = []
    try:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=_initialize_post_mutate_worker,
            initargs=(request_queue,),
            max_tasks_per_child=1,
        ) as executor:
            future_to_index = {
                executor.submit(_generate_variant_set_safe, generator.cfg, task): index
                for index, task in enumerate(tasks)
            }
            for future in as_completed(future_to_index):
                report = future.result()
                if service_process is not None:
                    report = replace(report, sdf_service_pid=int(service_process.pid or 0))
                indexed.append((future_to_index[future], report))
                if on_report is not None:
                    on_report(report)
    finally:
        if service_process is not None and request_queue is not None:
            from ...validator._sdf_service import stop_sdf_service

            stop_sdf_service(request_queue)
            service_process.join(timeout=30.0)
            if service_process.is_alive():
                service_process.terminate()
                service_process.join(timeout=10.0)
                raise RuntimeError("central GPU SDF service did not stop cleanly")
            if service_process.exitcode != 0:
                raise RuntimeError(f"central GPU SDF service exited with code {service_process.exitcode}")
            request_queue.close()
            request_queue.join_thread()
    return tuple(report for _, report in sorted(indexed, key=lambda item: item[0]))


def _initialize_post_mutate_worker(request_queue) -> None:

    from ...validator._sdf_service import configure_worker_sdf_service

    configure_worker_sdf_service(request_queue)


def infer_post_mutate_worker_count(cfg: HandGeneratorCfg, *, task_count: int) -> int:
    'Infers post mutate worker count.'

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
    context = child_generator._ensure_run_context()
    from .recipe_loader import RecipeLoader

    _write_dataset_build_attempt_marker(
        context.root_dir,
        task=task,
        child_config=RecipeLoader.dump(child_cfg),
    )
    results = list(child_generator.generate_batch())
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
        worker_pid=os.getpid(),
        worker_cuda_initialized=_worker_cuda_is_initialized(),
    )


def _generate_variant_set_safe(
    parent_cfg: HandGeneratorCfg,
    task: PostMutateSourceCfg,
) -> PostMutateVariantSetResult:

    try:
        return _generate_variant_set_worker(parent_cfg, task)
    except Exception as exc:


        from ...validator._sdf_service import CentralSdfServiceError

        if isinstance(exc, CentralSdfServiceError):
            raise
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
            worker_pid=os.getpid(),
            worker_cuda_initialized=_worker_cuda_is_initialized(),
            error=f"{type(exc).__name__}: {exc}",
        )


def _write_dataset_build_attempt_marker(
    run_dir: Path,
    *,
    task: PostMutateSourceCfg,
    child_config: dict[str, Any],
) -> None:

    if not task.build_id:
        return
    if not task.selection_lock_sha256 or not task.generator_config_sha256 or not task.child_config_sha256:
        raise ValueError("dataset build source task has incomplete ownership identity")
    import hashlib

    actual_child_sha256 = hashlib.sha256(
        yaml.safe_dump(child_config, allow_unicode=True, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if actual_child_sha256 != task.child_config_sha256:
        raise ValueError("dataset build child generator config identity drifted before worker execution")
    marker = {
        "schema_version": "1.0.0",
        "build_id": task.build_id,
        "selection_lock_sha256": task.selection_lock_sha256,
        "task_id": task.task_id,
        "attempt_index": task.attempt_index,
        "source_topology_dir": str(Path(task.source_topology_dir).resolve()),
        "seed": task.seed,
        "generator_config_sha256": task.generator_config_sha256,
        "child_config_sha256": actual_child_sha256,
        "created_at": datetime.now(UTC).isoformat(),
    }
    _write_yaml_atomic(run_dir / DATASET_BUILD_ATTEMPT_FILENAME, marker)


def _worker_cuda_is_initialized() -> bool:

    import sys

    torch_module = sys.modules.get("torch")
    return bool(torch_module is not None and torch_module.cuda.is_initialized())


def _write_yaml_atomic(path: Path, document: dict[str, Any]) -> None:

    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(yaml.safe_dump(document, allow_unicode=True, sort_keys=False), encoding="utf-8")
    temporary.replace(path)


__all__ = [
    "DATASET_BUILD_ATTEMPT_FILENAME",
    "PostMutateVariantSetResult",
    "infer_post_mutate_worker_count",
    "run_post_mutate_source_batch",
]
