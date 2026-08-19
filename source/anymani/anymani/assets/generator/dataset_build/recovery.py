r"""中断数据集构建的审计、接管与精确回退。

恢复操作只认 selection lock、build state 与 run-local ownership/config 证据，不按“最新目录”
或模糊时间猜测资产归属。schema 2 以 marker 精确授权 rollback/adopt；schema 1 legacy state
尚无 marker，因此迁移路径额外要求 task 正处于 ``running``、没有已登记 attempt/active run、
run summary 的完整 child ``HandGeneratorCfg`` 与 lock snapshot + task overrides 逐字段相等，
并且 summary 在 state 创建后写入。任一歧义都会在删除前 fail closed。
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
from ..runtime.mutate_batch import DATASET_BUILD_ATTEMPT_FILENAME
from .planner import derive_retry_seed
from .schema import DatasetBuildTemplateCfg

RECOVERY_REPORT_SCHEMA_VERSION = "1.0.0"
ATTEMPT_MARKER_FILENAME = DATASET_BUILD_ATTEMPT_FILENAME
_LEGACY_STATE_SCHEMA_VERSION = "1.0.0"
_OWNED_STATE_SCHEMA_VERSION = "2.0.0"


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
    if state_schema == _LEGACY_STATE_SCHEMA_VERSION:
        candidates = _discover_legacy_candidates(lock, state=state, state_path=resolved_state)
    elif state_schema == _OWNED_STATE_SCHEMA_VERSION:
        candidates = _discover_owned_candidates(lock, state=state)
    else:
        raise ValueError(f"unsupported build state schema for recovery: {state_schema!r}")
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
        if state_schema == _LEGACY_STATE_SCHEMA_VERSION and strategy == "rollback":
            _apply_legacy_rollback(candidates, state=state, state_path=resolved_state)
        elif state_schema == _LEGACY_STATE_SCHEMA_VERSION:
            _apply_legacy_adopt(candidates, state=state, state_path=resolved_state)
        elif strategy == "rollback":
            _apply_owned_rollback(candidates, state=state, state_path=resolved_state)
        else:
            _apply_owned_adopt(lock, candidates, state=state, state_path=resolved_state)
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


def _discover_owned_candidates(
    lock: Mapping[str, Any],
    *,
    state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    r"""按 schema-2 marker 发现本 build invocation 精确拥有的 run roots。

    state 中的 attempt 只提供预期身份；删除授权来自 run-local marker。每个 marker 必须同时
    匹配 ``build_id / lock / task / attempt / source / seed / parent cfg / child cfg``，且 run
    必须是 source 的非 symlink 直属子目录。任何一项漂移都 fail closed。
    """

    build_id = str(state.get("build_id", ""))
    lock_sha256 = str(state.get("selection_lock_sha256", ""))
    generator_sha256 = str(state.get("generator_config_sha256", ""))
    if not build_id or not lock_sha256 or not generator_sha256:
        raise ValueError("schema-2 build state has incomplete invocation identity")
    task_states = _mapping(state.get("tasks"), context="build state tasks")
    candidates: list[dict[str, Any]] = []
    for task in _iter_lock_tasks(lock):
        task_id = str(task["task_id"])
        task_state = _mapping(task_states.get(task_id), context=f"state task {task_id}")
        source_dir = _task_source_dir(lock, task)
        attempts = task_state.get("attempts", [])
        if not isinstance(attempts, list):
            raise TypeError(f"state task {task_id!r} attempts must be a sequence")
        attempts_by_index = {
            int(_mapping(attempt, context=f"state attempt {task_id}").get("attempt_index", index)): _mapping(
                attempt,
                context=f"state attempt {task_id}",
            )
            for index, attempt in enumerate(attempts)
        }
        marker_dirs: list[Path] = []
        for child in source_dir.iterdir():
            marker_path = child / ATTEMPT_MARKER_FILENAME
            if child.is_symlink():
                # symlink 绝不能成为删除边界。若其目标 marker 声称属于当前 invocation，
                # recovery 不能静默忽略后重置 state，而应要求人工处理这条歧义路径。
                if marker_path.is_file():
                    linked_marker = yaml.safe_load(marker_path.read_text(encoding="utf-8")) or {}
                    if isinstance(linked_marker, Mapping) and str(linked_marker.get("build_id")) == build_id:
                        raise ValueError(f"owned recovery marker is reachable through a symlink run root: {child}")
                continue
            if child.is_dir() and marker_path.is_file():
                marker_dirs.append(child)
        matched_indices: set[int] = set()
        for run_dir in sorted(marker_dirs):
            marker = _mapping(
                yaml.safe_load((run_dir / ATTEMPT_MARKER_FILENAME).read_text(encoding="utf-8")) or {},
                context=f"ownership marker {run_dir}",
            )
            if str(marker.get("build_id")) != build_id:
                continue
            if str(marker.get("task_id")) != task_id:
                # unseen-variant-set 与 train 可共享同一 mother source；属于另一个 locked task
                # 的 marker 会在该 task 的循环中验证，不能因目录共享被误判为身份冲突。
                if str(marker.get("task_id")) in task_states:
                    continue
                raise ValueError(f"owned marker references an unknown task in source lineage: {run_dir}")
            attempt_index = int(marker.get("attempt_index", -1))
            attempt = attempts_by_index.get(attempt_index)
            if attempt is None or attempt_index in matched_indices:
                raise ValueError(f"owned marker has absent or duplicate state attempt: {run_dir}")
            _validate_owned_marker(
                marker,
                run_dir=run_dir,
                source_dir=source_dir,
                attempt=attempt,
                lock_sha256=lock_sha256,
                generator_sha256=generator_sha256,
            )
            matched_indices.add(attempt_index)
            candidates.append(
                _classify_owned_run(
                    task,
                    run_dir=run_dir,
                    marker=marker,
                    attempt=attempt,
                )
            )
    return candidates


def _validate_owned_marker(
    marker: Mapping[str, Any],
    *,
    run_dir: Path,
    source_dir: Path,
    attempt: Mapping[str, Any],
    lock_sha256: str,
    generator_sha256: str,
) -> None:
    r"""验证 marker、state attempt 与真实路径三方一致。"""

    if run_dir.parent.resolve() != source_dir.resolve() or run_dir.is_symlink():
        raise ValueError(f"owned run is outside its direct source boundary: {run_dir}")
    expected = {
        "schema_version": "1.0.0",
        "selection_lock_sha256": lock_sha256,
        "attempt_index": int(attempt["attempt_index"]),
        "source_topology_dir": str(source_dir.resolve()),
        "seed": int(attempt["seed"]),
        "generator_config_sha256": generator_sha256,
        "child_config_sha256": str(attempt["child_config_sha256"]),
    }
    for field_name, expected_value in expected.items():
        if marker.get(field_name) != expected_value:
            raise ValueError(f"owned marker field {field_name!r} drifted for {run_dir}")


def _classify_owned_run(
    task: Mapping[str, Any],
    *,
    run_dir: Path,
    marker: Mapping[str, Any],
    attempt: Mapping[str, Any],
) -> dict[str, Any]:
    r"""用 summary config hash、seed、quota 与 sidecars 严格分类 schema-2 run。"""

    summary_path = run_dir / "summary.yaml"
    summary = yaml.safe_load(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    if not isinstance(summary, Mapping):
        summary = {}
    raw_config = summary.get("config", {})
    config = _mapping(raw_config, context=f"owned run config {run_dir}") if summary else {}
    config_sha256 = hashlib.sha256(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if summary and (
        summary.get("run", {}).get("mode") != "mutate"
        or config_sha256 != str(marker["child_config_sha256"])
        or int(config.get("post_mutate_seed", -1)) != int(marker["seed"])
        or Path(str(config.get("source_topology_dir", ""))).resolve(strict=False)
        != Path(str(marker["source_topology_dir"])).resolve(strict=False)
    ):
        raise ValueError(f"owned run summary conflicts with marker/config identity: {run_dir}")
    planned = int(task["variant_count"])
    sampling = summary.get("post_mutate_sampling", {}) if summary else {}
    successful = int(sampling.get("successful_variants", summary.get("stats", {}).get("succeeded", 0)))
    shortfall = int(sampling.get("shortfall", planned - successful))
    sidecars = tuple(sorted(run_dir.glob("*/hand.yaml")))
    complete = bool(summary) and successful == planned and shortfall == 0 and len(sidecars) == planned
    return {
        "task_id": str(task["task_id"]),
        "role": str(task["role"]),
        "attempt_index": int(marker["attempt_index"]),
        "source_dir": str(run_dir.parent.resolve()),
        "run_dir": str(run_dir.resolve()),
        "seed": int(marker["seed"]),
        "planned_variants": planned,
        "successful_variants": successful,
        "shortfall": shortfall,
        "sidecar_count": len(sidecars),
        "classification": "complete" if complete else "partial",
        "include_mother": bool(task["include_mother"]),
        "child_config_sha256": str(marker["child_config_sha256"]),
        "state_attempt_status": str(attempt.get("status", "")),
    }


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


def _apply_owned_rollback(
    candidates: list[dict[str, Any]],
    *,
    state: dict[str, Any],
    state_path: Path,
) -> None:
    r"""只删除 marker 精确拥有的 runs，并恢复 invocation baseline task state。"""

    for candidate in candidates:
        run_dir = Path(str(candidate["run_dir"]))
        source_dir = Path(str(candidate["source_dir"]))
        marker_path = run_dir / ATTEMPT_MARKER_FILENAME
        if run_dir.parent != source_dir or not run_dir.is_dir() or run_dir.is_symlink() or not marker_path.is_file():
            raise ValueError(f"refusing owned rollback outside audited source boundary: {run_dir}")
    for candidate in candidates:
        shutil.rmtree(Path(str(candidate["run_dir"])))

    baseline = _mapping(state.get("baseline"), context="schema-2 invocation baseline")
    baseline_tasks = _mapping(baseline.get("tasks"), context="schema-2 baseline tasks")
    state["tasks"] = deepcopy(baseline_tasks)
    state["rolled_back_at"] = datetime.now(UTC).isoformat()
    # build_id 与 baseline 是一次 invocation 的身份；rollback 后保留它会让下一次 build
    # 误把自己视为旧 invocation 的 resume。recovery_report 已保存审计证据，因此这里删除
    # state，让下一次正式 build 创建新的 build_id 与 baseline。
    state_path.unlink(missing_ok=True)


def _apply_owned_adopt(
    lock: Mapping[str, Any],
    candidates: list[dict[str, Any]],
    *,
    state: dict[str, Any],
    state_path: Path,
) -> None:
    r"""按 lock 顺序接管完整 owned runs，并隔离部分 run。

    complete candidate 必须通过 sidecar fingerprint 的 set 内和跨 task 唯一性；同一 task
    最多接管一个 complete run。partial candidate 保留在原目录并写 ``QUARANTINED.yaml``，
    对应 task 回到 pending，后续 build 会以新 retry seed 重新生成整个 variant set。
    """

    tasks_by_id = {str(task["task_id"]): task for task in _iter_lock_tasks(lock)}
    task_states = _mapping(state.get("tasks"), context="build state tasks")
    candidates_by_task: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        candidates_by_task.setdefault(str(candidate["task_id"]), []).append(candidate)

    accepted: dict[str, str] = {}
    for task in _iter_lock_tasks(lock):
        task_id = str(task["task_id"])
        task_state = _mapping(task_states[task_id], context=f"state task {task_id}")
        complete_candidates = [
            candidate
            for candidate in candidates_by_task.get(task_id, ())
            if candidate["classification"] == "complete"
        ]
        if len(complete_candidates) > 1:
            raise ValueError(f"cannot adopt multiple complete owned runs for task {task_id!r}")
        for candidate in candidates_by_task.get(task_id, ()):
            run_dir = Path(str(candidate["run_dir"]))
            attempt = _owned_attempt(task_state, int(candidate["attempt_index"]))
            if candidate["classification"] != "complete":
                _write_yaml_atomic(
                    run_dir / "QUARANTINED.yaml",
                    {"reason": "interrupted_partial_run", "recovery": dict(candidate)},
                )
                attempt["status"] = "quarantined"
                attempt["reason"] = "interrupted_partial_run"
                if task_state.get("status") != "completed":
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
                    raise ValueError(
                        f"cannot adopt cross-task geometry duplicate with {accepted[fingerprint]}: {run_dir}"
                    )
                accepted[fingerprint] = task_id
            variant_sidecars = sidecars[int(bool(candidate["include_mother"])) :]
            attempt.update(
                {
                    "run_dir": str(run_dir),
                    "planned_variants": int(candidate["planned_variants"]),
                    "successful_variants": int(candidate["successful_variants"]),
                    "shortfall": 0,
                    "error": "",
                    "sidecar_paths": [str(path) for path in variant_sidecars],
                    "status": "accepted",
                    "reason": "recovered_interrupted_complete_run",
                }
            )
            task_state["status"] = "completed"
            task_state["active_run_dir"] = str(run_dir)
            task_state["geometry_fingerprints"] = fingerprints
    # candidates 只可能引用 lock 中已验证 tasks；该断言防止未来修改 discovery 后静默丢 task。
    if any(task_id not in tasks_by_id for task_id in candidates_by_task):
        raise ValueError("owned recovery candidate references a task absent from selection lock")
    _write_yaml_atomic(state_path, state)


def _owned_attempt(task_state: Mapping[str, Any], attempt_index: int) -> dict[str, Any]:
    r"""按显式 attempt_index 取回可变 schema-2 attempt。"""

    attempts = task_state.get("attempts", [])
    if not isinstance(attempts, list):
        raise TypeError("schema-2 task attempts must be a sequence")
    matches = [
        _mapping(attempt, context=f"owned attempt {attempt_index}")
        for index, attempt in enumerate(attempts)
        if int(_mapping(attempt, context=f"owned attempt {index}").get("attempt_index", index)) == attempt_index
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one state attempt {attempt_index}, got {len(matches)}")
    return matches[0]


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
