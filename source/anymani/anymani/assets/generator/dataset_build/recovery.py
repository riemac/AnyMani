r"""中断数据集构建的审计、接管与精确回退。

恢复操作只认 selection lock、build state 与 run-local ownership/config 证据，不按“最新目录”
或模糊时间猜测资产归属。schema 1 legacy state 尚无 ownership marker，因此迁移路径额外要求：
task 正处于 ``running``、没有已登记 attempt/active run、run summary 的完整 child
``HandGeneratorCfg`` 与 lock snapshot + task overrides 逐字段相等，并且 summary 在 state 创建后写入。
任一歧义都会在删除前 fail closed。
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Mapping
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml

from ...bank.path_utils import resolve_bank_path
from ...geometry_identity import geometry_fingerprint_from_sidecar
from .planner import derive_retry_seed
from .schema import DatasetBuildTemplateCfg

RECOVERY_REPORT_SCHEMA_VERSION = "1.0.0"
ATTEMPT_MARKER_FILENAME = "DATASET_BUILD_ATTEMPT.yaml"
_LEGACY_STATE_SCHEMA_VERSION = "1.0.0"


def recover_dataset_build(
    template: DatasetBuildTemplateCfg,
    *,
    lock_path: str | Path,
    state_path: str | Path | None = None,
    strategy: Literal["adopt", "rollback"],
    apply: bool = False,
) -> dict[str, Any]:
    r"""审计并可选执行一次中断 build 的接管或回退。

    ``rollback`` 删除本 invocation 新建的完整 run roots；``adopt`` 仅接管 exact-complete
    且 fingerprint 闭合的 variant sets，部分 run 写 quarantine 后让对应 task 重新生成。
    默认 ``apply=False``，只返回并写出可人工核对的 dry-run report。
    """

    if strategy not in {"adopt", "rollback"}:
        raise ValueError(f"unsupported recovery strategy: {strategy!r}")
    resolved_lock = Path(lock_path).expanduser().resolve()
    resolved_state = (
        Path(state_path).expanduser().resolve()
        if state_path is not None
        else resolved_lock.parent / ".build_state.yaml"
    )
    lock_bytes = resolved_lock.read_bytes()
    lock = yaml.safe_load(lock_bytes) or {}
    state = yaml.safe_load(resolved_state.read_text(encoding="utf-8")) or {}
    if not isinstance(lock, dict) or not isinstance(state, dict):
        raise TypeError("selection lock and build state must be mappings")
    lock_sha256 = hashlib.sha256(lock_bytes).hexdigest()
    if state.get("selection_lock_sha256") != lock_sha256:
        raise ValueError("build state does not match the current selection lock")
    if str(lock.get("template_id")) != template.template_id:
        raise ValueError("selection lock does not match the current dataset template")

    state_schema = str(state.get("schema_version"))
    if state_schema != _LEGACY_STATE_SCHEMA_VERSION:
        raise ValueError(
            f"recovery currently requires legacy state schema {_LEGACY_STATE_SCHEMA_VERSION!r}, got {state_schema!r}"
        )
    candidates = _discover_legacy_candidates(lock, state=state, state_path=resolved_state)
    report = _make_report(
        strategy=strategy,
        apply=apply,
        lock_path=resolved_lock,
        state_path=resolved_state,
        lock_sha256=lock_sha256,
        state_schema=state_schema,
        candidates=candidates,
    )
    if apply:
        if strategy == "rollback":
            _apply_legacy_rollback(candidates, state=state, state_path=resolved_state)
        else:
            _apply_legacy_adopt(candidates, state=state, state_path=resolved_state)
        report["applied_at"] = datetime.now(UTC).isoformat()
    _write_yaml_atomic(resolved_lock.parent / "recovery_report.yaml", report)
    return report


def _discover_legacy_candidates(
    lock: Mapping[str, Any],
    *,
    state: Mapping[str, Any],
    state_path: Path,
) -> list[dict[str, Any]]:
    r"""以完整 task/config 证据发现 schema 1 崩溃前已创建但未回报主进程的 runs。"""

    tasks_by_id = {str(task["task_id"]): task for task in _iter_lock_tasks(lock)}
    raw_states = state.get("tasks")
    if not isinstance(raw_states, Mapping):
        raise TypeError("build state tasks must be a mapping")
    state_started_at = state_path.stat().st_mtime
    candidates: list[dict[str, Any]] = []
    for task_id, raw_task_state in raw_states.items():
        task_state = _mapping(raw_task_state, context=f"state task {task_id}")
        if task_state.get("status") != "running":
            continue
        if task_state.get("attempts") or task_state.get("active_run_dir"):
            raise ValueError(f"legacy running task {task_id!r} already has attempt/active-run provenance")
        task = tasks_by_id.get(str(task_id))
        if task is None:
            raise ValueError(f"state contains task absent from selection lock: {task_id!r}")
        source_dir = _task_source_dir(lock, task)
        expected_seed = derive_retry_seed(
            int(lock["seeds"]["mutation"]),
            str(task["role"]),
            str(task["mother"]["asset_id"]),
            0,
        )
        expected_cfg = _expected_child_config(lock, task, source_dir=source_dir, seed=expected_seed)
        matches: list[dict[str, Any]] = []
        for run_dir in source_dir.iterdir():
            if not run_dir.is_dir() or run_dir.is_symlink() or run_dir.name == "meshes":
                continue
            summary_path = run_dir / "summary.yaml"
            if not summary_path.is_file() or summary_path.stat().st_mtime + 1.0 < state_started_at:
                continue
            summary = yaml.safe_load(summary_path.read_text(encoding="utf-8")) or {}
            if not isinstance(summary, dict) or summary.get("run", {}).get("mode") != "mutate":
                continue
            if summary.get("config") != expected_cfg:
                continue
            matches.append(_classify_run(task, run_dir=run_dir, summary=summary, expected_seed=expected_seed))
        if len(matches) > 1:
            raise ValueError(f"legacy task {task_id!r} has multiple exact recovery candidates")
        candidates.extend(matches)
    return candidates


def _classify_run(
    task: Mapping[str, Any],
    *,
    run_dir: Path,
    summary: Mapping[str, Any],
    expected_seed: int,
) -> dict[str, Any]:
    r"""按 summary 与实际 sidecar 数把一个 owned run 归类为 complete 或 partial。"""

    planned = int(task["variant_count"])
    sampling = _mapping(summary.get("post_mutate_sampling", {}), context=f"sampling summary {run_dir}")
    successful = int(sampling.get("successful_variants", summary.get("stats", {}).get("succeeded", -1)))
    shortfall = int(sampling.get("shortfall", planned - successful))
    sidecars = tuple(sorted(run_dir.glob("*/hand.yaml")))
    complete = successful == planned and shortfall == 0 and len(sidecars) == planned
    return {
        "task_id": str(task["task_id"]),
        "role": str(task["role"]),
        "source_dir": str(run_dir.parent.resolve()),
        "run_dir": str(run_dir.resolve()),
        "seed": expected_seed,
        "planned_variants": planned,
        "successful_variants": successful,
        "shortfall": shortfall,
        "sidecar_count": len(sidecars),
        "classification": "complete" if complete else "partial",
        "include_mother": bool(task["include_mother"]),
    }


def _apply_legacy_rollback(
    candidates: list[dict[str, Any]],
    *,
    state: dict[str, Any],
    state_path: Path,
) -> None:
    r"""删除已审计 run roots，并将 legacy state 恢复到首次 build 前的空状态。"""

    for candidate in candidates:
        run_dir = Path(str(candidate["run_dir"]))
        source_dir = Path(str(candidate["source_dir"]))
        if run_dir.parent != source_dir or not run_dir.is_dir() or run_dir.is_symlink():
            raise ValueError(f"refusing recovery deletion outside audited source boundary: {run_dir}")
    for candidate in candidates:
        shutil.rmtree(Path(str(candidate["run_dir"])))

    task_states = _mapping(state.get("tasks"), context="build state tasks")
    for task_state in task_states.values():
        mutable = _mapping(task_state, context="build task state")
        if mutable.get("status") == "running":
            mutable["status"] = "pending"
    if any(value.get("attempts") or value.get("active_run_dir") for value in task_states.values()):
        _write_yaml_atomic(state_path, state)
    else:
        state_path.unlink(missing_ok=True)


def _apply_legacy_adopt(
    candidates: list[dict[str, Any]],
    *,
    state: dict[str, Any],
    state_path: Path,
) -> None:
    r"""接管 complete runs；partial runs 保留诊断证据并把 task 复位为 pending。"""

    accepted: dict[str, str] = {}
    task_states = _mapping(state.get("tasks"), context="build state tasks")
    for candidate in candidates:
        task_state = _mapping(task_states[candidate["task_id"]], context=f"state task {candidate['task_id']}")
        run_dir = Path(str(candidate["run_dir"]))
        if candidate["classification"] != "complete":
            _write_yaml_atomic(
                run_dir / "QUARANTINED.yaml",
                {"reason": "interrupted_partial_run", "recovery": dict(candidate)},
            )
            task_state["status"] = "pending"
            continue
        sidecars = list(sorted(run_dir.glob("*/hand.yaml")))
        if bool(candidate["include_mother"]):
            sidecars.insert(0, Path(str(candidate["source_dir"])) / "hand.yaml")
        fingerprints = [geometry_fingerprint_from_sidecar(path) for path in sidecars]
        if len(set(fingerprints)) != len(fingerprints):
            raise ValueError(f"cannot adopt run with internal geometry duplicate: {run_dir}")
        for fingerprint in fingerprints:
            if fingerprint in accepted:
                raise ValueError(f"cannot adopt cross-task geometry duplicate with {accepted[fingerprint]}: {run_dir}")
            accepted[fingerprint] = str(candidate["task_id"])
        task_state["attempts"].append(
            {
                "run_dir": str(run_dir),
                "seed": int(candidate["seed"]),
                "planned_variants": int(candidate["planned_variants"]),
                "successful_variants": int(candidate["successful_variants"]),
                "shortfall": 0,
                "error": "",
                "sidecar_paths": [str(path) for path in sidecars[int(bool(candidate["include_mother"])) :]],
                "status": "accepted",
                "reason": "recovered_interrupted_complete_run",
            }
        )
        task_state["status"] = "completed"
        task_state["active_run_dir"] = str(run_dir)
        task_state["geometry_fingerprints"] = fingerprints
    _write_yaml_atomic(state_path, state)


def _make_report(
    *,
    strategy: str,
    apply: bool,
    lock_path: Path,
    state_path: Path,
    lock_sha256: str,
    state_schema: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    r"""构造可保存、可人工核对的 recovery report。"""

    complete = sum(item["classification"] == "complete" for item in candidates)
    partial = len(candidates) - complete
    return {
        "schema_version": RECOVERY_REPORT_SCHEMA_VERSION,
        "strategy": strategy,
        "dry_run": not apply,
        "selection_lock": str(lock_path),
        "selection_lock_sha256": lock_sha256,
        "build_state": str(state_path),
        "build_state_schema_version": state_schema,
        "counts": {
            "run_roots": len(candidates),
            "complete": complete,
            "partial": partial,
            "variant_sidecars": sum(int(item["sidecar_count"]) for item in candidates),
        },
        "runs": candidates,
    }


def _iter_lock_tasks(lock: Mapping[str, Any]) -> list[dict[str, Any]]:
    r"""按 lock role 声明顺序展开 task，并补回 role 字段。"""

    lineages = _mapping(lock.get("lineages"), context="selection lock lineages")
    tasks: list[dict[str, Any]] = []
    for role, raw_tasks in lineages.items():
        if not isinstance(raw_tasks, list):
            raise TypeError(f"selection lock role {role!r} must be a sequence")
        tasks.extend({**_mapping(task, context=f"lock task {role}"), "role": str(role)} for task in raw_tasks)
    return tasks


def _task_source_dir(lock: Mapping[str, Any], task: Mapping[str, Any]) -> Path:
    r"""解析 task mother topology 根，并要求真实目录存在。"""

    inventory = _mapping(lock.get("inventory"), context="selection lock inventory")
    mother = _mapping(task.get("mother"), context=f"task mother {task.get('task_id')}")
    source = resolve_bank_path(str(inventory["run_dir"])) / str(mother["relative_dir"])
    return source.resolve(strict=True)


def _expected_child_config(
    lock: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    source_dir: Path,
    seed: int,
) -> dict[str, Any]:
    r"""从 lock generator snapshot 精确重建 worker 实际写入 summary 的 child cfg。"""

    generator = _mapping(lock.get("generator"), context="selection lock generator")
    snapshot = deepcopy(_mapping(generator.get("config_snapshot"), context="generator config snapshot"))
    snapshot.update(
        {
            "source_topology_dir": str(source_dir),
            "post_mutate_sources": [],
            "n_samples": int(task["variant_count"]),
            "post_mutate_seed": int(seed),
            "post_mutate_parallel": False,
            "post_mutate_parallel_workers": None,
        }
    )
    return snapshot


def _mapping(value: Any, *, context: str) -> dict[str, Any]:
    r"""把 YAML mapping 收窄成可变字典，并在恢复边界拒绝错误容器。"""

    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a mapping")
    return value


def _write_yaml_atomic(path: Path, document: Mapping[str, Any]) -> None:
    r"""以同目录临时文件原子替换 recovery/state YAML。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(yaml.safe_dump(dict(document), allow_unicode=True, sort_keys=False), encoding="utf-8")
    temporary.replace(path)


__all__ = [
    "ATTEMPT_MARKER_FILENAME",
    "RECOVERY_REPORT_SCHEMA_VERSION",
    "recover_dataset_build",
]
