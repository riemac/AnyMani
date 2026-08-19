r"""Selection lock 到 post-mutate variant sets 与最终 SSL/PPO manifests 的执行器。

执行顺序固定为 train、validation、evaluation；同一阶段内的 mother tasks 交给
``HandGenerator.generate_variant_sets`` 并行。每个 task 达到 exact variant quota 且
通过 geometry fingerprint 唯一性后才记为 completed。失败 run 保留 summary 并写入
``QUARANTINED.yaml``，但永不进入最终 dataset manifest。
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from ...bank.path_utils import resolve_bank_path
from ...geometry_identity import geometry_fingerprint_from_sidecar
from ..hand_generator import (
    HandGenerator,
    HandGeneratorCfg,
    PostMutateSourceCfg,
    PostMutateVariantSetResult,
)
from .planner import (
    CanonicalMotherPair,
    MotherInventoryRecord,
    derive_retry_seed,
    select_pair_subset,
)
from .schema import DatasetBuildTemplateCfg

BUILD_STATE_SCHEMA_VERSION = "1.0.0"
"""可恢复 task 状态文件 schema。"""

_STAGES: tuple[tuple[str, ...], ...] = (
    ("train",),
    ("validation.unseen_variant_set", "validation.unseen_mother"),
    ("evaluation.unseen_variant_set", "evaluation.unseen_mother"),
)

RunBatch = Callable[
    [HandGeneratorCfg, tuple[PostMutateSourceCfg, ...], int | None],
    tuple[PostMutateVariantSetResult, ...],
]


def build_dataset_from_lock(
    template: DatasetBuildTemplateCfg,
    *,
    template_sha256: str,
    lock_path: str | Path,
    post_mutate_cfg: HandGeneratorCfg,
    workers: int | None = None,
    resume: bool = True,
    run_batch: RunBatch | None = None,
) -> dict[str, Any]:
    r"""执行 selection lock，成功后发布 ``ssl.yaml`` 与 ``ppo.yaml``。

    Args:
        template (DatasetBuildTemplateCfg): 当前 human-authored 构建模板。
        template_sha256 (str): 当前模板 byte-level identity。
        lock_path (str | Path): 已人工审阅的 selection lock。
        post_mutate_cfg (HandGeneratorCfg): Python 配置模块交付的 mutation recipe。
        workers (int | None): mother-level process workers；``None`` 使用 generator 默认。
        resume (bool): 是否复用已闭合 completed tasks。
        run_batch (RunBatch | None): 测试注入点；生产默认调用 ``HandGenerator``。

    Returns:
        dict[str, Any]: 最终 build report 文档。
    """

    resolved_lock = Path(lock_path).expanduser().resolve()
    lock_bytes = resolved_lock.read_bytes()
    lock = yaml.safe_load(lock_bytes) or {}
    if not isinstance(lock, dict):
        raise TypeError("selection lock must be a mapping")
    _validate_lock(template, template_sha256=template_sha256, lock=lock, post_mutate_cfg=post_mutate_cfg)
    lock_sha256 = hashlib.sha256(lock_bytes).hexdigest()
    output_dir = resolved_lock.parent
    state_path = output_dir / ".build_state.yaml"
    state = _load_or_create_state(lock, lock_sha256=lock_sha256, state_path=state_path, resume=resume)
    runner = run_batch or _run_generator_batch

    # Resume 时按 lock 顺序重建已完成 task 的 fingerprint registry，保证冲突 winner 不随调度改变。
    accepted_fingerprints: dict[str, str] = {}
    for role_names in _STAGES:
        for task in _tasks_for_roles(lock, role_names):
            task_state = state["tasks"][task["task_id"]]
            if task_state.get("status") != "completed":
                continue
            if not _completed_task_is_valid(task, task_state, state=state):
                task_state["status"] = "pending"
                task_state["active_run_dir"] = ""
                continue
            _register_task_fingerprints(task, task_state, accepted_fingerprints)
    _write_yaml_atomic(state_path, state)

    for role_names in _STAGES:
        stage_tasks = _tasks_for_roles(lock, role_names)
        _execute_stage(
            stage_tasks,
            state=state,
            state_path=state_path,
            post_mutate_cfg=post_mutate_cfg,
            workers=workers,
            retry_rounds=template.generation_policy.dataset_retry_rounds,
            mutation_root_seed=template.seeds.mutation,
            accepted_fingerprints=accepted_fingerprints,
            run_batch=runner,
        )

    failed = [task_id for task_id, task_state in state["tasks"].items() if task_state["status"] != "completed"]
    report = _build_report(lock, state=state, failed_task_ids=failed)
    _write_yaml_atomic(output_dir / "build_report.yaml", report)
    if failed:
        for manifest_name in ("ssl.yaml", "ppo.yaml"):
            manifest_path = output_dir / manifest_name
            if manifest_path.exists():
                manifest_path.unlink()
        raise RuntimeError(f"dataset build has incomplete tasks and cannot publish manifests: {tuple(failed)}")

    ssl_manifest = compile_dataset_manifest(template, lock=lock, state=state, ppo=False)
    _write_yaml_atomic(output_dir / "ssl.yaml", ssl_manifest)
    if template.manifests.ppo.enabled:
        ppo_manifest = compile_dataset_manifest(template, lock=lock, state=state, ppo=True)
        _write_yaml_atomic(output_dir / "ppo.yaml", ppo_manifest)
    return report


def compile_dataset_manifest(
    template: DatasetBuildTemplateCfg,
    *,
    lock: Mapping[str, Any],
    state: Mapping[str, Any],
    ppo: bool,
    ppo_pair_keys: set[str] | None = None,
) -> dict[str, Any]:
    r"""把 completed lineage tasks 编译成严格 HandAssetDataset schema 2.0 YAML。"""

    lineages = _lineages_by_role(lock)
    train_lineages = lineages["train"]
    selected_ppo_pairs: set[str] | None = None
    if ppo:
        selected_ppo_pairs = ppo_pair_keys or {str(key) for key in lock.get("ppo_train_pair_keys", ())}
        train_lineages = [
            lineage for lineage in train_lineages if str(lineage["pair_key"]) in selected_ppo_pairs
        ]
    validation = {
        suite: _partition_document(
            _filter_seen_suite_for_ppo(
                lineages[f"validation.{suite}"],
                suite=suite,
                ppo_pair_keys=selected_ppo_pairs,
            ),
            state=state,
        )
        for suite in ("unseen_variant_set", "unseen_mother")
    }
    evaluation = {
        suite: _partition_document(
            _filter_seen_suite_for_ppo(
                lineages[f"evaluation.{suite}"],
                suite=suite,
                ppo_pair_keys=selected_ppo_pairs,
            ),
            state=state,
        )
        for suite in ("unseen_variant_set", "unseen_mother")
    }
    evaluation["official_zero_shot"] = {
        "assets": list(template.partitions.evaluation.official_zero_shot)
    }
    return {
        "schema_version": "2.0.0",
        "default_run_dir": template.inventory.run_dir,
        "train": _partition_document(train_lineages, state=state),
        "validation": validation,
        "evaluation": evaluation,
    }


def derive_ppo_manifest_from_lock(
    template: DatasetBuildTemplateCfg,
    *,
    lock: Mapping[str, Any],
    state: Mapping[str, Any],
    mother_count: int,
    selection_seed: int,
    reuse_ssl_holdouts: bool,
) -> dict[str, Any]:
    r"""不生成新资产，从 completed SSL cohort 派生另一份 PPO manifest。

    ``reuse_ssl_holdouts=True`` 时，两条 SSL seen-mother suites 是 PPO train 的强制
    子集；若关闭，则重新分层选择 PPO train，并自动裁掉 mother 不在 PPO train 的
    unseen-variant records，保持 dataset relation 合法。
    """

    if mother_count < 2 or mother_count % 2 != 0:
        raise ValueError("derived PPO mother_count must be a positive even number")
    train_pairs = _pairs_from_locked_train(lock)
    template_for_selection = replace(
        template,
        seeds=replace(template.seeds, selection=selection_seed),
    )
    mandatory_keys: set[str] = set()
    if reuse_ssl_holdouts:
        by_role = _lineages_by_role(lock)
        mandatory_keys = {
            str(lineage["pair_key"])
            for role in ("validation.unseen_variant_set", "evaluation.unseen_variant_set")
            for lineage in by_role[role]
        }
    target_pairs = mother_count // 2
    if len(mandatory_keys) > target_pairs:
        raise ValueError("derived PPO cohort is smaller than reused SSL seen-mother holdouts")
    mandatory = tuple(pair for pair in train_pairs if pair.pair_key in mandatory_keys)
    remaining = tuple(pair for pair in train_pairs if pair.pair_key not in mandatory_keys)
    extra = select_pair_subset(
        remaining,
        mother_count=2 * (target_pairs - len(mandatory)),
        template=template_for_selection,
        domain=f"derived-ppo/{mother_count}/{selection_seed}",
    )
    selected_keys = {pair.pair_key for pair in (*mandatory, *extra)}
    return compile_dataset_manifest(
        template,
        lock=lock,
        state=state,
        ppo=True,
        ppo_pair_keys=selected_keys,
    )
def _execute_stage(
    tasks: Sequence[dict[str, Any]],
    *,
    state: dict[str, Any],
    state_path: Path,
    post_mutate_cfg: HandGeneratorCfg,
    workers: int | None,
    retry_rounds: int,
    mutation_root_seed: int,
    accepted_fingerprints: dict[str, str],
    run_batch: RunBatch,
) -> None:
    r"""完成一个 outer partition stage；失败 task 以派生 seed 最多补采三轮。"""

    for _ in range(retry_rounds):
        pending = [task for task in tasks if state["tasks"][task["task_id"]]["status"] != "completed"]
        if not pending:
            return
        source_cfgs: list[PostMutateSourceCfg] = []
        for task in pending:
            task_state = state["tasks"][task["task_id"]]
            retry_index = len(task_state["attempts"])
            seed = derive_retry_seed(
                mutation_root_seed,
                str(task["role"]),
                str(task["mother"]["asset_id"]),
                retry_index,
            )
            source_cfgs.append(
                PostMutateSourceCfg(
                    task_id=str(task["task_id"]),
                    source_topology_dir=_source_path(state, task),
                    n_samples=int(task["variant_count"]),
                    seed=seed,
                )
            )
            task_state["status"] = "running"
        _write_yaml_atomic(state_path, state)

        reports = run_batch(post_mutate_cfg, tuple(source_cfgs), workers)
        report_by_task = {report.task_id: report for report in reports}
        for task in pending:
            task_id = str(task["task_id"])
            report = report_by_task.get(task_id)
            if report is None:
                state["tasks"][task_id]["status"] = "failed"
                continue
            _accept_or_quarantine_report(
                task,
                report,
                task_state=state["tasks"][task_id],
                accepted_fingerprints=accepted_fingerprints,
            )
            _write_yaml_atomic(state_path, state)

    for task in tasks:
        task_state = state["tasks"][task["task_id"]]
        if task_state["status"] != "completed":
            task_state["status"] = "failed"
    _write_yaml_atomic(state_path, state)


def _accept_or_quarantine_report(
    task: Mapping[str, Any],
    report: PostMutateVariantSetResult,
    *,
    task_state: dict[str, Any],
    accepted_fingerprints: dict[str, str],
) -> None:
    r"""按 exact quota 与 geometry fingerprint 决定 variant set 是否可发布。"""

    attempt = {
        "run_dir": str(report.run_dir),
        "seed": report.mutation_seed,
        "planned_variants": report.planned_variants,
        "successful_variants": report.successful_variants,
        "shortfall": report.shortfall,
        "error": report.error,
        "sidecar_paths": [str(path) for path in report.sidecar_paths],
        "status": "candidate",
        "reason": "",
    }
    task_state["attempts"].append(attempt)
    if report.error:
        attempt["status"] = "failed"
        attempt["reason"] = f"worker_error:{report.error}"
        task_state["status"] = "pending"
        return
    if report.shortfall != 0 or report.successful_variants != int(task["variant_count"]):
        _quarantine(report.run_dir, reason="variant_shortfall", details=attempt)
        attempt["status"] = "quarantined"
        attempt["reason"] = "variant_shortfall"
        task_state["status"] = "pending"
        return

    local: dict[str, str] = {}
    candidate_paths = list(report.sidecar_paths)
    if bool(task["include_mother"]):
        candidate_paths.insert(0, report.source_topology_dir / "hand.yaml")
    conflict: tuple[str, str] | None = None
    for sidecar_path in candidate_paths:
        fingerprint = geometry_fingerprint_from_sidecar(sidecar_path)
        previous = local.get(fingerprint) or accepted_fingerprints.get(fingerprint)
        if previous is not None:
            conflict = (fingerprint, previous)
            break
        local[fingerprint] = str(sidecar_path)
    if conflict is not None:
        reason = f"geometry_fingerprint_collision:{conflict[0]}:{conflict[1]}"
        _quarantine(report.run_dir, reason=reason, details=attempt)
        attempt["status"] = "quarantined"
        attempt["reason"] = reason
        task_state["status"] = "pending"
        return

    accepted_fingerprints.update({fingerprint: str(task["task_id"]) for fingerprint in local})
    attempt["status"] = "accepted"
    task_state["status"] = "completed"
    task_state["active_run_dir"] = str(report.run_dir)
    task_state["geometry_fingerprints"] = list(local)


def _run_generator_batch(
    cfg: HandGeneratorCfg,
    tasks: tuple[PostMutateSourceCfg, ...],
    workers: int | None,
) -> tuple[PostMutateVariantSetResult, ...]:
    r"""把 locked source tasks lower 到正式多-source HandGenerator façade。"""

    if cfg.mode != "mutate" or cfg.artifact_level != "bundle":
        raise ValueError("dataset build requires POST_MUTATE_CFG mode='mutate', artifact_level='bundle'")
    run_cfg = cfg.replace(
        source_topology_dir=None,
        post_mutate_sources=list(tasks),
        post_mutate_parallel=True,
        post_mutate_parallel_workers=workers,
    )
    return tuple(HandGenerator(run_cfg).generate_variant_sets())


def _validate_lock(
    template: DatasetBuildTemplateCfg,
    *,
    template_sha256: str,
    lock: Mapping[str, Any],
    post_mutate_cfg: HandGeneratorCfg,
) -> None:
    r"""拒绝模板、inventory 或 generator recipe 与 selection lock 漂移。"""

    if str(lock.get("schema_version")) != "1.0.0":
        raise ValueError("selection lock schema must be exactly '1.0.0'")
    if str(lock.get("template_id")) != template.template_id or str(lock.get("template_sha256")) != template_sha256:
        raise ValueError("selection lock does not match current dataset template")
    from ..runtime.recipe_loader import RecipeLoader

    snapshot = RecipeLoader.dump(post_mutate_cfg)
    digest = hashlib.sha256(yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=True).encode()).hexdigest()
    generator = lock.get("generator", {})
    if not isinstance(generator, Mapping) or str(generator.get("config_sha256")) != digest:
        raise ValueError("selection lock generator config hash does not match current POST_MUTATE_CFG")


def _load_or_create_state(
    lock: Mapping[str, Any],
    *,
    lock_sha256: str,
    state_path: Path,
    resume: bool,
) -> dict[str, Any]:
    r"""读取匹配 lock 的 build state，或为全部 tasks 建立 pending 状态。"""

    tasks = [task for role_tasks in _lineages_by_role(lock).values() for task in role_tasks]
    if state_path.exists() and resume:
        state = yaml.safe_load(state_path.read_text(encoding="utf-8")) or {}
        if not isinstance(state, dict) or state.get("selection_lock_sha256") != lock_sha256:
            raise ValueError("build state does not match current selection lock")
        return state
    return {
        "schema_version": BUILD_STATE_SCHEMA_VERSION,
        "selection_lock_sha256": lock_sha256,
        "inventory_run_dir": str(resolve_bank_path(str(lock["inventory"]["run_dir"]))),
        "tasks": {
            str(task["task_id"]): {
                "status": "pending",
                "active_run_dir": "",
                "attempts": [],
                "geometry_fingerprints": [],
            }
            for task in tasks
        },
    }


def _completed_task_is_valid(
    task: Mapping[str, Any],
    task_state: Mapping[str, Any],
    *,
    state: Mapping[str, Any],
) -> bool:
    r"""复核 completed run 的 summary/source/count 和 variant sidecars。"""

    run_dir = Path(str(task_state.get("active_run_dir", "")))
    summary_path = run_dir / "summary.yaml"
    if not summary_path.is_file():
        return False
    summary = yaml.safe_load(summary_path.read_text(encoding="utf-8")) or {}
    if not isinstance(summary, Mapping) or summary.get("run", {}).get("mode") != "mutate":
        return False
    if int(summary.get("stats", {}).get("succeeded", -1)) != int(task["variant_count"]):
        return False
    source = Path(str(summary.get("config", {}).get("source_topology_dir", ""))).resolve(strict=False)
    if source != _source_path(state, task).resolve(strict=False):
        return False
    variant_sidecars = tuple(sorted(run_dir.glob("*/hand.yaml")))
    if len(variant_sidecars) != int(task["variant_count"]):
        return False

    # Resume 不能只信任旧 state 中缓存的 fingerprints：bundle 可能在中断期间被改写、
    # 删除或手工替换。重新读取实际 sidecars，并同时证明 set 内没有重复。
    candidate_paths = list(variant_sidecars)
    if bool(task["include_mother"]):
        candidate_paths.insert(0, _source_path(state, task) / "hand.yaml")
    actual_fingerprints = [geometry_fingerprint_from_sidecar(path) for path in candidate_paths]
    stored_fingerprints = [str(value) for value in task_state.get("geometry_fingerprints", ())]
    return (
        len(set(actual_fingerprints)) == len(actual_fingerprints)
        and len(set(stored_fingerprints)) == len(stored_fingerprints)
        and set(actual_fingerprints) == set(stored_fingerprints)
    )


def _register_task_fingerprints(
    task: Mapping[str, Any],
    task_state: Mapping[str, Any],
    accepted: dict[str, str],
) -> None:
    r"""把 resume task 的已验证 fingerprint 写回全局 registry。"""

    for fingerprint in task_state.get("geometry_fingerprints", ()):
        previous = accepted.get(str(fingerprint))
        if previous is not None:
            raise ValueError(f"completed build state contains duplicate geometry fingerprint: {fingerprint}")
        accepted[str(fingerprint)] = str(task["task_id"])


def _tasks_for_roles(lock: Mapping[str, Any], roles: Sequence[str]) -> list[dict[str, Any]]:
    r"""按 lock 声明顺序返回一个 outer stage 的 source tasks。"""

    by_role = _lineages_by_role(lock)
    return [task for role in roles for task in by_role[role]]


def _lineages_by_role(lock: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    r"""验证并返回 selection lock 的 role -> lineage mapping。"""

    raw = lock.get("lineages")
    if not isinstance(raw, Mapping):
        raise TypeError("selection lock lineages must be a mapping")
    parsed: dict[str, list[dict[str, Any]]] = {}
    for role, tasks in raw.items():
        if not isinstance(tasks, list) or not all(isinstance(task, dict) for task in tasks):
            raise TypeError(f"selection lock role {role!r} must be a sequence of task mappings")
        parsed[str(role)] = [{**task, "role": str(role)} for task in tasks]
    return parsed


def _filter_seen_suite_for_ppo(
    lineages: Sequence[dict[str, Any]],
    *,
    suite: str,
    ppo_pair_keys: set[str] | None,
) -> list[dict[str, Any]]:
    r"""PPO 只裁剪 seen-mother suite；unseen-mother holdout 与 SSL 保持相同。"""

    if ppo_pair_keys is None or suite != "unseen_variant_set":
        return list(lineages)
    return [lineage for lineage in lineages if str(lineage["pair_key"]) in ppo_pair_keys]


def _pairs_from_locked_train(lock: Mapping[str, Any]) -> tuple[CanonicalMotherPair, ...]:
    r"""从 lock 的 train lineages 恢复 canonical pairs，供纯 manifest 派生重采样。"""

    grouped: dict[str, dict[str, MotherInventoryRecord]] = {}
    for lineage in _lineages_by_role(lock)["train"]:
        pair_key = str(lineage["pair_key"])
        mother = _mother_record_from_lock(lineage["mother"])
        grouped.setdefault(pair_key, {})[mother.handedness] = mother
    pairs: list[CanonicalMotherPair] = []
    for pair_key, members in grouped.items():
        if set(members) != {"left", "right"}:
            raise ValueError(f"locked train pair {pair_key!r} is incomplete")
        pairs.append(CanonicalMotherPair(pair_key=pair_key, left=members["left"], right=members["right"]))
    return tuple(pairs)


def _mother_record_from_lock(payload: Mapping[str, Any]) -> MotherInventoryRecord:
    r"""恢复 YAML 将 tuple lowering 成 list 后的 mother inventory record。"""

    return MotherInventoryRecord(
        asset_id=str(payload["asset_id"]),
        relative_dir=str(payload["relative_dir"]),
        collection_kind=cast(Literal["groups", "mixed"], str(payload["collection_kind"])),
        group_name=str(payload["group_name"]),
        mother_name=str(payload["mother_name"]),
        handedness=cast(Literal["left", "right"], str(payload["handedness"])),
        base_family=cast(Literal["allegro", "leap"], str(payload["base_family"])),
        family_composition=cast(
            Literal["single_family", "mixed"], str(payload["family_composition"])
        ),
        macro_family=str(payload["macro_family"]),
        topology_shape=cast(Literal["full", "missing"], str(payload["topology_shape"])),
        missing_slots=tuple(str(slot) for slot in payload.get("missing_slots", ())),
        dof=int(payload["dof"]),
        finger_count=int(payload["finger_count"]),
        slot_family_map=tuple((str(key), str(value)) for key, value in payload.get("slot_family_map", ())),
        selected_slot_recipes=tuple(
            (str(key), str(value)) for key, value in payload.get("selected_slot_recipes", ())
        ),
    )


def _partition_document(lineages: Sequence[Mapping[str, Any]], *, state: Mapping[str, Any]) -> dict[str, Any]:
    r"""按 collection/group/mother 层级编译一个 generated partition。"""

    run_block: dict[str, Any] = {"groups": {}, "mixed": {}}
    for lineage in lineages:
        mother = lineage["mother"]
        collection = str(mother["collection_kind"])
        group = str(mother["group_name"])
        mother_name = str(mother["mother_name"])
        task_state = state["tasks"][lineage["task_id"]]
        variant_sets = []
        if int(lineage["variant_count"]) > 0:
            variant_sets.append(Path(str(task_state["active_run_dir"])).name)
        run_block[collection].setdefault(group, {})[mother_name] = {
            "include_mother": bool(lineage["include_mother"]),
            "variant_sets": variant_sets,
        }
    run_block = {key: value for key, value in run_block.items() if value}
    return {"runs": {"default": run_block}} if run_block else {"runs": {}}


def _source_path(state: Mapping[str, Any], task: Mapping[str, Any]) -> Path:
    r"""由 state inventory root 与 locked relative mother dir 恢复绝对 source path。"""

    return Path(str(state["inventory_run_dir"])) / str(task["mother"]["relative_dir"])


def _quarantine(run_dir: Path, *, reason: str, details: Mapping[str, Any]) -> None:
    r"""在失败 variant-set 根写稳定 quarantine 证据，不删除诊断现场。"""

    _write_yaml_atomic(run_dir / "QUARANTINED.yaml", {"reason": reason, "attempt": dict(details)})


def _build_report(lock: Mapping[str, Any], *, state: Mapping[str, Any], failed_task_ids: Sequence[str]) -> dict[str, Any]:
    r"""汇总 task、attempt、quarantine 与发布状态。"""

    statuses = {task_id: task_state["status"] for task_id, task_state in state["tasks"].items()}
    return {
        "schema_version": "1.0.0",
        "template_id": lock["template_id"],
        "published": not failed_task_ids,
        "failed_task_ids": list(failed_task_ids),
        "status_counts": {
            status: sum(current == status for current in statuses.values())
            for status in sorted(set(statuses.values()))
        },
        "quota_report": deepcopy(lock.get("quota_report", {})),
        "tasks": deepcopy(state["tasks"]),
    }


def _write_yaml_atomic(path: Path, document: Mapping[str, Any]) -> None:
    r"""在同目录临时文件完成 YAML 写入后原子替换目标。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(dict(document), allow_unicode=True, sort_keys=False), encoding="utf-8")
    temporary.replace(path)


__all__ = [
    "BUILD_STATE_SCHEMA_VERSION",
    "build_dataset_from_lock",
    "compile_dataset_manifest",
    "derive_ppo_manifest_from_lock",
]
