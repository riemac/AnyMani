r"""把四个 source-level evaluation strata 注册为可供 cohort API 消费的技术资产池。

本入口接在 ``prepare_evaluation_cohorts.py`` 之后。它只消费已经冻结的 source-selection plan，
不重新选择资产、不读取策略分数，也不修改原始 ``ssl.yaml``、N040 evaluation manifest、teacher
canonical lock 或 Allegro revision bundle。

``HandAssetDataset`` 的稳定 API 要求非空 ``train`` partition，而最终 N040 evaluation manifest
自然位于 ``evaluation.unseen_variant_set`` / ``evaluation.unseen_mother``。因此本脚本为每个 stratum
写一份独立的 schema-2.0 technical manifest：它复制原 N040 source manifest 的 ``default_run_dir`` 和
目标 suite 的 ``runs`` 配置，只保留 pure-right production group，并把这份 manifest 明确标为
``evaluation_only`` 技术注册视图。外部 source lock 的 ``selection`` 同时绑定原 evaluation partition、
原 manifest SHA、suite 内 source row、selection plan SHA 和 ``registration_is_not_policy_or_ssl_training``；
技术 manifest 的 ``train`` 字段绝不能被解释为真实 SSL/PPO train exposure。

默认阶段只发布 source lock 与 reserve 来源映射，状态为 ``pending-canonical``。显式 ``--canonicalize``
才会使用 CPU-safe ``restore_hand_cfg_snapshot`` 与 ``materialize_canonical_artifact`` 生成 candidate
canonical artifacts，并依据 canonical physical hash 做 selected/reserve 补位。显式 ``--audit-exposures``
才会把最终 N040 train source 交给同一 canonical runtime 做完整暴露审计；该阶段可能遍历 8192 assets，
本脚本不擅自启动它。无完整暴露审计或 strict pregrasp 前，任何输出都不会标成 ``strict-passed`` 或
``ready``。

下游调用约定：

1. 先运行本脚本生成每个 stratum 的 ``evaluation_only_manifest.yaml`` 与 ``source.lock.yaml``；
2. primary 审阅 source registration，并在需要时使用 ``--canonicalize`` 生成独立 canonical-final lock；
3. primary 再把 ``*.canonical.lock.yaml`` 交给 ``scripts/research/prepare_cohort_pregrasp_shards.py``，
   由 strict Top-8 搜索/准入产生 catalog；
4. canonical physical isolation 和 pregrasp 未闭合时，保留所有 reserve 与 replacement reason。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

try:
    from scripts.research.family_teacher_distillation.prepare_evaluation_cohorts import (
        PURE_GROUP_BY_FAMILY,
        _display_path,
        _load_document,
        _resolve_path,
        _sha256_file,
        _stable_hash,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    # 直接 ``python scripts/...py`` 时 sys.path 只有脚本目录；补入 AnyMani 根后
    # 仍通过同一 prepare 模块读取常量/路径 helper，不复制第二套选择语义。
    _script_repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_script_repo_root))
    from scripts.research.family_teacher_distillation.prepare_evaluation_cohorts import (
        PURE_GROUP_BY_FAMILY,
        _display_path,
        _load_document,
        _resolve_path,
        _sha256_file,
        _stable_hash,
    )

# 计划与 materialization 分开版本化；source lock 仍由 assets.bank.cohort 负责 schema 1.1/1.2。
MATERIALIZATION_SCHEMA_VERSION = "1.0.0"
TECHNICAL_DATASET_SCHEMA_VERSION = "2.0.0"
CANONICAL_PHYSICAL_ALGORITHM = "canonical-runtime-lowering"


def _stable_digest(value: Any) -> str:
    r"""对 JSON-safe source/identity payload 求稳定 SHA-256，不把文件路径 hash 当物理 hash。"""

    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()  # 仅用于计划/registration 内容身份


def _mapping(document: Any, *, label: str) -> Mapping[str, Any]:
    r"""拒绝错误顶层类型，使技术 manifest 注册不会静默产生空 train。"""

    if not isinstance(document, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return document


def _write_new_or_verify(path: Path, payload: str, *, resume: bool) -> None:
    r"""新建 metadata；resume 时只接受逐字节相同内容，绝不覆盖既有证据。"""

    if path.exists():
        if not resume or path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"output already exists or differs: {path}")
        return  # 相同 bytes 表示可安全复用已完成的 source registration
    path.parent.mkdir(parents=True, exist_ok=True)  # 只创建本次独占 output subtree
    path.write_text(payload, encoding="utf-8")  # source plan/registration 是本入口允许的可审计写入


def _load_selection_plan(path: Path) -> dict[str, Any]:
    r"""加载并检查 selection plan 的最小 schema/状态合同。"""

    document = _mapping(_load_document(path), label="selection plan")
    if document.get("artifact_type") != "anymani.family_teacher_distillation.evaluation_cohort_source_selection":
        raise ValueError("selection plan artifact_type is not the accepted source-selection artifact")
    if document.get("schema_version") != MATERIALIZATION_SCHEMA_VERSION:
        raise ValueError("selection plan schema_version is unsupported")
    if document.get("status") != "pending-canonical":
        raise ValueError("materialization requires an unfinalized pending-canonical selection plan")
    if document.get("policy_results_read") is not False:
        raise ValueError("selection plan must prove that policy results were not read")
    declared_digest = document.get("selection_digest")
    digest_payload = dict(document)
    digest_payload.pop("selection_digest", None)
    recomputed_digest = _stable_hash(
        (json.dumps(digest_payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")),)
    )
    if declared_digest != recomputed_digest:
        raise ValueError("selection plan selection_digest does not match its contents")
    strata = document.get("strata")
    if not isinstance(strata, Mapping) or not strata:
        raise ValueError("selection plan must contain source strata")
    for name, details in strata.items():
        if not isinstance(details, Mapping):
            raise ValueError(f"selection stratum {name!r} must be a mapping")
        for role in ("selected", "reserve", "excluded"):
            if not isinstance(details.get(role), list):
                raise ValueError(f"selection stratum {name!r} lacks list field {role!r}")
    return dict(document)


def _plan_input_path(plan: Mapping[str, Any], key: str, root: Path) -> Path:
    r"""从 plan 的绑定输入恢复绝对路径，避免 materializer 自己重新猜 N040 版本。"""

    inputs = _mapping(plan.get("inputs"), label="selection plan inputs")
    item = _mapping(inputs.get(key), label=f"selection plan inputs.{key}")
    raw_path = item.get("path")
    if not raw_path:
        raise ValueError(f"selection plan input {key!r} has no path")
    path = _resolve_path(Path(str(raw_path)), root)
    if not path.is_file():
        raise FileNotFoundError(f"selection plan input is missing: {path}")
    expected = item.get("sha256")
    if expected and _sha256_file(path) != expected:
        raise ValueError(f"selection plan input SHA changed: {path}")
    return path


def _origin_member(member: Mapping[str, Any], *, root: Path, role: str) -> dict[str, Any]:
    r"""把 plan member 降为 source registration 所需的 origin 坐标，排除策略结果字段。"""

    record = _mapping(member.get("source_record"), label="selection source_record")
    origin = _mapping(member.get("origin_source"), label="selection origin_source")
    source_asset = _mapping(member.get("source_asset"), label="selection source_asset")
    source_files = source_asset.get("source_file_sha256", {})
    if not isinstance(source_files, Mapping):
        raise ValueError("selection source_file_sha256 must be a mapping")
    return {
        "asset_id": str(record.get("asset_id", member.get("asset_id", ""))),
        "role": role,
        "origin_partition": str(origin.get("partition", record.get("partition", ""))),
        "origin_manifest_row": origin.get("manifest_row"),
        "family": record.get("family"),
        "handedness": record.get("handedness"),
        "collection_kind": record.get("collection_kind"),
        "group_name": record.get("group_name"),
        "mother_name": record.get("mother_name"),
        "mother_path": record.get("mother_path"),
        "variant_set": record.get("variant_set", ""),
        "asset_role": record.get("asset_role"),
        "source_asset_path": source_asset.get("path"),
        "source_file_sha256": dict(source_files),
        "source_content_hash": record.get("content_hash"),
        "source_manifest_physical_geometry_hash": record.get("physical_geometry_hash"),
        "pre_revision_source_identity": member.get("pre_revision_source_identity"),
        "source_selection_sort_key": member.get("selection", {}).get("source_sort_key"),
    }


def _stratum_origin_members(
    details: Mapping[str, Any], *, root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    r"""恢复 selected/reserve 的 source lineage 列表；excluded 永不进入技术 train manifest。"""

    selected = [_origin_member(item, root=root, role="selected") for item in details["selected"]]
    reserve = [_origin_member(item, root=root, role="reserve") for item in details["reserve"]]
    return selected, reserve


def build_evaluation_only_manifest(
    source_manifest: Mapping[str, Any],
    *,
    family: str,
    suite: str,
    allowed_bases: Sequence[str],
) -> dict[str, Any]:
    r"""从原 N040 source manifest 复制目标 suite 的 runs，构成 evaluation-only 技术 train 视图。

    技术视图只为 ``write_hand_asset_cohort_lock`` 满足非空 train API；所有 source registration 语义在
    外部 lock.selection 中显式标为非训练。这里仅复制 pure production group 的 right base 配置，混合
    group、left group、validation 和 evaluation 都保持空，避免任何隐式额外暴露。
    """

    source_schema = str(source_manifest.get("schema_version", ""))
    if source_schema != TECHNICAL_DATASET_SCHEMA_VERSION:
        raise ValueError(f"source manifest schema must be {TECHNICAL_DATASET_SCHEMA_VERSION!r}")
    evaluation = _mapping(source_manifest.get("evaluation"), label="source manifest evaluation")
    suite_document = _mapping(evaluation.get(suite), label=f"source manifest evaluation.{suite}")
    runs = _mapping(suite_document.get("runs"), label=f"evaluation.{suite}.runs")
    pure_group = PURE_GROUP_BY_FAMILY[family]
    allowed = {str(base) for base in allowed_bases}  # source row 选择已在 plan 冻结，技术视图不重新抽样
    technical_runs: dict[str, Any] = {}
    for alias, raw_run in runs.items():
        run = _mapping(raw_run, label=f"evaluation.{suite}.runs.{alias}")
        groups = _mapping(run.get("groups", {}), label=f"evaluation.{suite}.runs.{alias}.groups")
        pure_records = _mapping(groups.get(pure_group, {}), label=f"{pure_group} source group")
        filtered_records = {
            str(base): value for base, value in pure_records.items() if str(base) in allowed
        }  # 只保留本 stratum selected/reserve 所属 lineage
        if not filtered_records:
            continue  # 一个 run alias 若没有目标 family，不能写入空 job
        technical_runs[str(alias)] = {
            "run_dir": str(run.get("run_dir", "")),  # 保留原 run_dir，确保 source path 不发生重解释
            "groups": {pure_group: filtered_records},  # mixed 分支故意不注册
        }
    if not technical_runs:
        raise ValueError(f"evaluation.{suite} has no allowed {family}/right pure-group run")
    return {
        "schema_version": TECHNICAL_DATASET_SCHEMA_VERSION,
        "default_run_dir": source_manifest.get("default_run_dir", ""),  # 原始 generated root 真源
        "train": {"runs": technical_runs},  # 仅为 cohort API 注册；外部 selection 明确 evaluation_only
        "validation": {"unseen_variant_set": {"runs": {}}, "unseen_mother": {"runs": {}}},
        "evaluation": {
            "unseen_variant_set": {"runs": {}},
            "unseen_mother": {"runs": {}},
            "official_zero_shot": {"assets": []},
        },
    }


def _registration_document(
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
    eval_manifest_path: Path,
    technical_manifest_path: Path,
    technical_manifest_sha256: str,
    stratum: str,
    source_suite: str,
    selected: Sequence[Mapping[str, Any]],
    reserve: Sequence[Mapping[str, Any]],
    root: Path,
) -> dict[str, Any]:
    r"""构造写入 source lock.selection 的外部 registration contract。"""

    source_records = [
        {
            **_origin_member(member, root=root, role="selected"),
            "technical_manifest_row": None,  # materializer 在 resolve 后补充 source row
        }
        for member in selected
    ]
    reserve_records = [
        {
            **_origin_member(member, root=root, role="reserve"),
            "technical_manifest_row": None,  # reserve 仍保留 source coordinate，便于后续补位
        }
        for member in reserve
    ]
    return {
        "artifact_type": "anymani.family_teacher_distillation.evaluation_only_source_registration",
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "status": "pending-canonical",
        "stratum": stratum,
        "origin_partition": source_suite,
        "origin_manifest_path": _display_path(eval_manifest_path, root),
        "origin_manifest_sha256": _sha256_file(eval_manifest_path),
        "source_selection_plan_path": _display_path(plan_path, root),
        "source_selection_plan_sha256": _sha256_file(plan_path),
        "source_selection_digest": plan.get("selection_digest"),
        "technical_manifest_path": _display_path(technical_manifest_path, root),
        "technical_manifest_sha256": technical_manifest_sha256,
        "registration_mode": "evaluation_only_as_technical_train_for_cohort_api",
        "registration_is_not_policy_or_ssl_training": True,
        "source_lock_canonical_status": "pending-canonical",
        "selected_origin_members": source_records,
        "reserve_origin_members": reserve_records,
    }


def _resolve_technical_partition(technical_manifest_path: Path) -> tuple[Any, dict[str, int], bool]:
    r"""用真实 HandAssetDataset/prepared cache 解析技术 train 视图，并返回 asset_id→technical row。"""

    from anymani.assets.bank.dataset import HandAssetDataset
    from anymani.assets.bank.prepared_train import resolve_prepared_train

    dataset = HandAssetDataset.from_yaml(technical_manifest_path)  # 只解析本 stratum 的 pure-right source pool
    partition, cache_hit = resolve_prepared_train(
        dataset,
        require_geometry_semantics=True,
    )  # 复用现有 typed sidecar/cache，不启动 Isaac
    id_to_row: dict[str, int] = {}
    for row, record in enumerate(partition.records):
        asset_id = record.container.asset_id
        if asset_id in id_to_row:
            raise ValueError(f"technical manifest has duplicate asset ID: {asset_id}")
        id_to_row[asset_id] = row  # writer 后续使用这个 dense technical train row
    return partition, id_to_row, cache_hit


def _bind_registration_rows(
    registration: dict[str, Any],
    *,
    selected: Sequence[Mapping[str, Any]],
    reserve: Sequence[Mapping[str, Any]],
    id_to_row: Mapping[str, int],
) -> dict[str, Any]:
    r"""把 technical manifest row 写回 registration，仍保留 origin suite row 与 source path。"""

    result = json.loads(json.dumps(registration))  # 避免原地修改 plan-derived mapping
    selected_origin = result["selected_origin_members"]
    reserve_origin = result["reserve_origin_members"]
    selected_ids = [str(item["source_record"]["asset_id"]) for item in selected]
    reserve_ids = [str(item["source_record"]["asset_id"]) for item in reserve]
    for item, asset_id in zip(selected_origin, selected_ids, strict=True):
        if asset_id not in id_to_row:
            raise ValueError(f"selected asset is absent from technical manifest: {asset_id}")
        item["technical_manifest_row"] = id_to_row[asset_id]
    for item, asset_id in zip(reserve_origin, reserve_ids, strict=True):
        if asset_id not in id_to_row:
            raise ValueError(f"reserve asset is absent from technical manifest: {asset_id}")
        item["technical_manifest_row"] = id_to_row[asset_id]
    result["technical_asset_count"] = len(id_to_row)
    result["selected_technical_rows"] = [id_to_row[asset_id] for asset_id in selected_ids]
    result["reserve_technical_rows"] = [id_to_row[asset_id] for asset_id in reserve_ids]
    return result


def _verify_resolved_source_members(
    partition: Any,
    *,
    selected: Sequence[Mapping[str, Any]],
    reserve: Sequence[Mapping[str, Any]],
    id_to_row: Mapping[str, int],
    root: Path,
) -> None:
    r"""把 plan 的 source coordinates 与真实 resolver 结果逐项核对后才允许写 source lock。

    ``source_manifest_sha256`` 只绑定 YAML bytes，不能单独证明 bundle 内容仍未变化。这里补查
    resolved asset ID、bundle 路径、hand.yaml/hand.urdf bytes、typed content hash 和 lineage 字段；
    任一变化都 fail closed，避免 registration.json 记录旧 source 而 source lock 已封存新内容。
    """

    for member in [*selected, *reserve]:
        source_record = _mapping(member.get("source_record"), label="selection source_record")
        source_asset = _mapping(member.get("source_asset"), label="selection source_asset")
        asset_id = str(member.get("asset_id", source_record.get("asset_id", "")))
        if asset_id not in id_to_row:
            raise ValueError(f"planned source member is absent from resolved technical partition: {asset_id}")
        record = partition.records[id_to_row[asset_id]]
        container = record.container
        if container.asset_id != asset_id:
            raise ValueError(
                f"resolved asset ID changed at technical row {id_to_row[asset_id]}: "
                f"expected={asset_id}, actual={container.asset_id}"
            )
        expected_bundle = _resolve_path(Path(str(source_asset.get("path", ""))), root)
        actual_bundle = container.urdf_path.parent.resolve(strict=True)
        if actual_bundle != expected_bundle.resolve(strict=True):
            raise ValueError(
                f"source bundle path changed for {asset_id}: expected={expected_bundle}, actual={actual_bundle}"
            )
        expected_files = _mapping(source_asset.get("source_file_sha256"), label="source file SHA map")
        actual_files = {"hand.urdf": container.urdf_path, "hand.yaml": container.sidecar_path}
        for name, expected_sha in expected_files.items():
            if name not in actual_files:
                raise ValueError(f"unsupported source file coordinate {name!r} for {asset_id}")
            actual_sha = _sha256_file(actual_files[name])
            if actual_sha != str(expected_sha):
                raise ValueError(
                    f"source file SHA changed for {asset_id} {name}: expected={expected_sha}, actual={actual_sha}"
                )
        expected_content = str(source_record.get("content_hash") or "")
        if expected_content and record.content_hash != expected_content:
            raise ValueError(
                f"typed source content hash changed for {asset_id}: "
                f"expected={expected_content}, actual={record.content_hash}"
            )
        expected_lineage = {
            "group_name": source_record.get("group_name"),
            "mother_name": source_record.get("mother_name"),
            "mother_path": source_record.get("mother_path"),
            "variant_set": source_record.get("variant_set", ""),
            "asset_role": source_record.get("asset_role"),
        }
        actual_lineage = {
            "group_name": record.provenance.group_name,
            "mother_name": record.provenance.mother_name,
            "mother_path": record.provenance.mother_path,
            "variant_set": record.provenance.variant_set,
            "asset_role": record.provenance.asset_role,
        }
        for field, expected in expected_lineage.items():
            if expected is not None and str(actual_lineage[field]) != str(expected):
                raise ValueError(
                    f"source lineage changed for {asset_id} field={field}: "
                    f"expected={expected}, actual={actual_lineage[field]}"
                )


def _write_source_lock(
    *,
    output_path: Path,
    technical_manifest_path: Path,
    registration: Mapping[str, Any],
    selected_ids: Sequence[str],
    id_to_row: Mapping[str, int],
    resume: bool,
) -> Path:
    r"""调用现有 cohort writer 发布 1.1 source lock，selection 保留 evaluation-only provenance。"""

    from anymani.assets.bank.cohort import load_hand_asset_cohort, write_hand_asset_cohort_lock

    if output_path.exists():
        if not resume:
            raise FileExistsError(output_path)
        loaded = load_hand_asset_cohort(output_path, require_geometry_semantics=True)
        if (
            loaded.selection.get("source_selection_plan_sha256") != registration.get("source_selection_plan_sha256")
            or loaded.selection.get("stratum") != registration.get("stratum")
            or loaded.selection.get("registration_is_not_policy_or_ssl_training") is not True
        ):
            raise ValueError("resume source lock does not match this evaluation-only registration")
        if tuple(member.asset_id for member in loaded.members) != tuple(selected_ids):
            raise ValueError("resume source lock member order differs from this selection")
        return output_path
    coordinates = tuple(("evaluation_only", id_to_row[asset_id]) for asset_id in selected_ids)
    selection = {
        **dict(registration),
        "source_lock_schema": "1.1.0",
        "source_lock_selection_contract": "origin partition/row/path retained; technical train is not exposure",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return write_hand_asset_cohort_lock(
        output_path,
        cohort_id=f"{registration['stratum']}-source-registration",
        source_manifests={"evaluation_only": technical_manifest_path},
        member_coordinates=coordinates,
        selection=selection,
        require_geometry_semantics=True,
    )


def _canonicalize_partition(
    partition: Any,
    *,
    root: Path,
    output_root: Path,
    asset_rows: Mapping[str, int],
    records_by_id: Mapping[str, Mapping[str, Any]],
    limit: int | None = None,
) -> dict[str, dict[str, Any]]:
    r"""使用真实 typed sidecar 恢复与 canonical runtime 相同的 CPU artifact/hash。"""

    from anymani.assets.asset_sidecar import restore_hand_cfg_snapshot
    from anymani.assets.canonical_runtime import (
        CANONICAL_HAND_SCHEMA_V1,
        materialize_canonical_artifact,
        validate_canonical_artifact,
    )

    artifacts: dict[str, dict[str, Any]] = {}  # asset_id -> JSON-safe canonical artifact summary
    records = partition.records[:limit] if limit is not None else partition.records
    for record in records:
        container = record.container
        asset_id = container.asset_id
        if asset_id not in asset_rows or asset_id not in records_by_id:
            raise ValueError(f"canonical record is absent from source selection: {asset_id}")
        raw_hand_cfg = container.sidecar.get("hand_cfg")
        if not isinstance(raw_hand_cfg, dict):
            raise ValueError(f"source sidecar lacks hand_cfg for canonicalization: {asset_id}")
        hand_cfg = restore_hand_cfg_snapshot(raw_hand_cfg)  # sidecar snapshot是 HandCfg 真源，不逆向猜 URDF
        semantics = container.geometry_semantics
        q_home = tuple(semantics.q_home_rad) if semantics is not None else ()  # source home pose，单位 rad
        q_home_joint_names = tuple(semantics.active_joint_names) if semantics is not None else ()
        source_record = records_by_id[asset_id]
        artifact = materialize_canonical_artifact(
            hand_cfg,
            asset_id=asset_id,
            output_root=output_root,
            source_urdf_path=container.urdf_path,
            schema=CANONICAL_HAND_SCHEMA_V1,
            asset_row=asset_rows[asset_id],
            topology=str(source_record.get("topology_key", source_record.get("mother_name", "unknown"))),
            q_home=q_home,
            q_home_joint_names=q_home_joint_names,
        )
        validate_canonical_artifact(artifact, schema=CANONICAL_HAND_SCHEMA_V1)  # hash/URDF/schema闭环
        artifacts[asset_id] = {
            "asset_id": asset_id,
            "technical_manifest_row": asset_rows[asset_id],
            # finalize_hand_asset_cohort_lock 的 configuration 域就是
            # materialize_canonical_artifact 返回的 source_content_hash；保留
            # 两个命名，避免把 canonical physical hash 与 source/config hash 混域。
            "configuration_domain_hash": artifact.source_content_hash,
            "source_content_hash": artifact.source_content_hash,
            "source_urdf_hash": artifact.source_urdf_hash,
            "physical_geometry_hash": artifact.physical_geometry_hash,
            "canonical_schema_digest": artifact.schema_digest,
            "canonical_urdf_hash": artifact.canonical_urdf_hash,
            "canonical_urdf_path": _display_path(Path(artifact.canonical_urdf_path), root),
            "canonical_manifest_path": _display_path(Path(artifact.manifest_path), root),
            "canonical_physical_algorithm": CANONICAL_PHYSICAL_ALGORITHM,
        }
    return artifacts


def _teacher_canonical_hashes(plan: Mapping[str, Any], *, root: Path) -> tuple[set[str], dict[str, Any]]:
    r"""读取 teacher canonical lock 的同域 physical hashes，并拒绝缺 algorithm binding 的旧证据。"""

    inputs = _mapping(plan["inputs"], label="selection plan inputs")
    lock_items = inputs.get("teacher_locks", [])
    if not isinstance(lock_items, list):
        raise ValueError("selection plan teacher_locks must be a list")
    physical: set[str] = set()  # 只收 canonical lock physical hash，不混 N040 source/provider hash
    summaries = []
    for item in lock_items:
        entry = _mapping(item, label="teacher lock descriptor")
        path = _resolve_path(Path(str(entry["path"])), root)
        document = _mapping(_load_document(path), label=f"teacher lock {path}")
        binding = _mapping(document.get("canonical_binding", {}), label=f"teacher lock binding {path}")
        if binding.get("physical_identity_algorithm") != CANONICAL_PHYSICAL_ALGORITHM:
            raise ValueError(f"teacher lock is not bound to {CANONICAL_PHYSICAL_ALGORITHM}: {path}")
        members = document.get("members", [])
        if not isinstance(members, list):
            raise ValueError(f"teacher lock members must be a list: {path}")
        for member in members:
            if not isinstance(member, Mapping) or not member.get("physical_geometry_hash"):
                raise ValueError(f"teacher lock lacks canonical physical hash: {path}")
            physical.add(str(member["physical_geometry_hash"]))
        summaries.append(
            {"path": _display_path(path, root), "member_count": len(members), "sha256": _sha256_file(path)}
        )
    return physical, {"lock_count": len(summaries), "physical_hash_count": len(physical), "locks": summaries}


def _canonical_selection(
    details: Mapping[str, Any],
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
    teacher_hashes: set[str],
    n040_train_hashes: set[str] | None,
    target_count: int,
    n040_train_audit_complete: bool | None = None,
) -> dict[str, Any]:
    r"""按原 selected→reserve 顺序进行同域 canonical collision 检测与 deterministic 补位。"""

    planned_selected = [str(item["asset_id"]) for item in details["selected"]]
    ordered = list(details["selected"]) + list(details["reserve"])  # reserve 只能按冻结顺序补位
    chosen: list[dict[str, Any]] = []
    reserve: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    replacements: list[dict[str, Any]] = []
    used_physical: set[str] = set()  # 同一 candidate stratum 内也禁止 canonical physical duplicate
    protected = set(teacher_hashes)
    protected_sources = ["teacher_canonical_locks"]
    if n040_train_hashes is not None:
        protected.update(n040_train_hashes)
        protected_sources.append("n040_train_canonical_audit")
    for member in ordered:
        asset_id = str(member["asset_id"])
        artifact = artifacts.get(asset_id)
        if artifact is None:
            raise ValueError(f"canonical artifact missing for selected/reserve asset: {asset_id}")
        physical = str(artifact["physical_geometry_hash"])
        reasons: list[str] = []
        if physical in protected:
            reasons.append("canonical_physical_hash_in_protected_exposure")
        if physical in used_physical:
            reasons.append("canonical_physical_hash_duplicate_in_stratum")
        if reasons:
            excluded.append(
                {
                    "asset_id": asset_id,
                    "planned_role": member["selection"]["role"],
                    "physical_geometry_hash": physical,
                    "reasons": reasons,
                }
            )
            if asset_id in planned_selected:
                replacements.append(
                    {
                        "replaced_planned_asset_id": asset_id,
                        "reason": reasons,
                        "replacement_source": "next-reserve-in-original-source-hash-order",
                    }
                )
            continue
        mapped = json.loads(json.dumps(member))  # 不改变 selection plan 原始角色字段
        mapped["canonical_physical_identity"] = {
            "status": "canonicalized",
            "physical_geometry_hash": physical,
            "configuration_domain_hash": artifact["source_content_hash"],
            "canonical_schema_digest": artifact["canonical_schema_digest"],
            "algorithm": CANONICAL_PHYSICAL_ALGORITHM,
        }
        used_physical.add(physical)
        if len(chosen) < target_count:
            mapped["canonical_selection_role"] = "selected"
            chosen.append(mapped)
        else:
            mapped["canonical_selection_role"] = "reserve"
            reserve.append(mapped)
    # ``n040_train_hashes`` 在 bounded-prefix smoke 中也可能非空；只有完整 8192
    # exposure audit 才能让输出进入 audited 状态，避免把前缀核对冒充完整隔离。
    exposure_audit_complete = (
        n040_train_hashes is not None if n040_train_audit_complete is None else n040_train_audit_complete
    )
    if exposure_audit_complete and n040_train_hashes is None:
        raise ValueError("complete N040 exposure audit requires canonical hashes")
    if len(chosen) < target_count:
        status = "canonical-pool-insufficient"
    elif not exposure_audit_complete:
        status = "canonicalized-pending-n040-train-audit"
    else:
        status = "canonicalized-and-exposure-audited-pending-pregrasp"
    return {
        "status": status,
        "target_count": target_count,
        "selected_count": len(chosen),
        "reserve_count": len(reserve),
        "excluded_count": len(excluded),
        "protected_hash_sources": protected_sources,
        "selected": chosen,
        "reserve": reserve,
        "excluded": excluded,
        "replacements": replacements,
        "canonical_physical_hashes_unique": len(used_physical) == len(chosen),
    }


def _write_canonical_lock(
    *,
    source_lock_path: Path,
    canonical_lock_path: Path,
    final_selection: Mapping[str, Any],
    technical_rows: Mapping[str, int],
    artifacts: Mapping[str, Mapping[str, Any]],
    resume: bool,
) -> Path:
    r"""用现有 finalize API 写 schema-1.2 canonical lock；不把它标为 strict pregrasp ready。"""

    from anymani.assets.bank.cohort import finalize_hand_asset_cohort_lock

    if canonical_lock_path.exists():
        if not resume:
            raise FileExistsError(canonical_lock_path)
        from anymani.assets.bank.cohort import load_hand_asset_cohort

        loaded = load_hand_asset_cohort(canonical_lock_path, require_geometry_semantics=True)
        binding = loaded.canonical_binding
        expected_source_sha = _sha256_file(source_lock_path)
        if binding.get("source_lock_sha256") != expected_source_sha:
            raise ValueError("resume canonical lock is bound to another source lock")
        selected_ids = [str(item["asset_id"]) for item in final_selection["selected"]]
        if tuple(member.asset_id for member in loaded.members) != tuple(selected_ids):
            raise ValueError("resume canonical lock member order differs from canonical selection")
        for member, asset_id in zip(loaded.members, selected_ids, strict=True):
            artifact = artifacts[asset_id]
            if (
                member.configuration_domain_hash != artifact["configuration_domain_hash"]
                or member.physical_geometry_hash != artifact["physical_geometry_hash"]
                or member.canonical_schema_digest != artifact["canonical_schema_digest"]
            ):
                raise ValueError(f"resume canonical lock identity differs for {asset_id}")
        return canonical_lock_path  # 重新加载并核对 source/canonical bytes 后才复用
    selected = final_selection["selected"]
    selected_ids = [str(item["asset_id"]) for item in selected]
    canonical_identities = []
    for asset_id in selected_ids:
        artifact = artifacts[asset_id]
        if asset_id not in technical_rows:
            raise ValueError(f"canonical selected asset lacks technical row: {asset_id}")
        canonical_identities.append(
            (
                str(artifact["configuration_domain_hash"]),
                str(artifact["physical_geometry_hash"]),
                str(artifact["canonical_schema_digest"]),
            )
        )
    # finalize API 会重新读取 source lock，确保 source member order 与 canonical identities 完全一致。
    return finalize_hand_asset_cohort_lock(
        source_lock_path,
        canonical_lock_path,
        canonical_identities=tuple(canonical_identities),
        canonical_schema_version="v1",
        require_geometry_semantics=True,
    )


def _materialize_stratum(
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
    source_manifest: Mapping[str, Any],
    eval_manifest_path: Path,
    stratum: str,
    details: Mapping[str, Any],
    output_dir: Path,
    root: Path,
    canonicalize: bool,
    audit_n040_train_hashes: set[str] | None,
    n040_train_audit_complete: bool,
    teacher_hashes: set[str],
    resume: bool,
) -> dict[str, Any]:
    r"""完成一个 stratum 的 technical manifest、source lock 与可选 canonical-final lock。"""

    family = str(details["family"])
    suite = str(details["source_suite"]).removeprefix("evaluation.")
    selected, reserve = _stratum_origin_members(details, root=root)
    allowed_bases = sorted({str(item["mother_name"]) for item in [*selected, *reserve]})
    technical_document = build_evaluation_only_manifest(
        source_manifest,
        family=family,
        suite=suite,
        allowed_bases=allowed_bases,
    )
    stratum_dir = output_dir / stratum
    source_dir = stratum_dir / "source"
    technical_path = source_dir / "evaluation_only_manifest.yaml"
    technical_payload = yaml.safe_dump(technical_document, sort_keys=False, allow_unicode=True)
    _write_new_or_verify(technical_path, technical_payload, resume=resume)
    technical_sha = _sha256_file(technical_path)
    registration = _registration_document(
        plan=plan,
        plan_path=plan_path,
        eval_manifest_path=eval_manifest_path,
        technical_manifest_path=technical_path,
        technical_manifest_sha256=technical_sha,
        stratum=stratum,
        source_suite=str(details["source_suite"]),
        selected=details["selected"],
        reserve=details["reserve"],
        root=root,
    )
    partition, id_to_row, cache_hit = _resolve_technical_partition(technical_path)
    registration = _bind_registration_rows(
        registration,
        selected=details["selected"],
        reserve=details["reserve"],
        id_to_row=id_to_row,
    )
    _verify_resolved_source_members(
        partition,
        selected=details["selected"],
        reserve=details["reserve"],
        id_to_row=id_to_row,
        root=root,
    )
    registration_payload = json.dumps(registration, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    registration_path = source_dir / "registration.json"
    _write_new_or_verify(registration_path, registration_payload, resume=resume)
    selected_ids = [str(item["asset_id"]) for item in details["selected"]]
    source_lock_path = source_dir / "source.lock.yaml"
    source_lock = _write_source_lock(
        output_path=source_lock_path,
        technical_manifest_path=technical_path,
        registration=registration,
        selected_ids=selected_ids,
        id_to_row=id_to_row,
        resume=resume,
    )
    result: dict[str, Any] = {
        "stratum": stratum,
        "source_suite": str(details["source_suite"]),
        "source_pool_count": int(details["source_pool_count"]),
        "eligible_pool_count": int(details["eligible_pool_count"]),
        "planned_selected_count": len(selected_ids),
        "planned_reserve_count": len(details["reserve"]),
        "technical_manifest": {
            "path": _display_path(technical_path, root),
            "sha256": technical_sha,
            "schema_version": TECHNICAL_DATASET_SCHEMA_VERSION,
            "evaluation_only": True,
            "resolved_record_count": len(partition.records),
            "prepared_cache_hit": cache_hit,
        },
        "registration": {
            "path": _display_path(registration_path, root),
            "sha256": _sha256_file(registration_path),
            "registration_is_not_policy_or_ssl_training": True,
        },
        "source_lock": {
            "path": _display_path(source_lock, root),
            "sha256": _sha256_file(source_lock),
            "schema_version": "1.1.0",
        },
        "status": "source-registered-pending-canonical",
    }
    if not canonicalize:
        return result
    materialization_stage = "exposure-audited" if n040_train_audit_complete else "candidate-only"
    stage_suffix = ".exposure-audited" if n040_train_audit_complete else ""
    records_by_id = {
        str(item["source_record"]["asset_id"]): item["source_record"]
        for item in [*details["selected"], *details["reserve"]]
    }
    canonical_output_root = stratum_dir / "canonical_artifacts"
    artifacts = _canonicalize_partition(
        partition,
        root=root,
        output_root=canonical_output_root,
        asset_rows=id_to_row,
        records_by_id=records_by_id,
    )
    final_selection = _canonical_selection(
        details,
        artifacts=artifacts,
        teacher_hashes=teacher_hashes,
        n040_train_hashes=audit_n040_train_hashes,
        target_count=int(details["target_count"]),
        n040_train_audit_complete=n040_train_audit_complete,
    )
    canonical_selection_path = stratum_dir / f"canonical-selection{stage_suffix}.json"
    canonical_selection_document = {
        "artifact_type": "anymani.family_teacher_distillation.canonical_selection_audit",
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "status": final_selection["status"],
        "stratum": stratum,
        "source_lock": _display_path(source_lock, root),
        "source_lock_sha256": _sha256_file(source_lock),
        "selection_plan_sha256": _sha256_file(plan_path),
        "materialization_stage": materialization_stage,
        "registration_is_not_policy_or_ssl_training": True,
        "artifacts": artifacts,
        "selection": final_selection,
    }
    _write_new_or_verify(
        canonical_selection_path,
        json.dumps(canonical_selection_document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        resume=resume,
    )
    final_selected_ids = [str(item["asset_id"]) for item in final_selection["selected"]]
    final_source_lock = source_dir / f"canonical-selected{stage_suffix}.source.lock.yaml"
    final_registration = dict(registration)
    final_registration.update(
        {
            "canonical_selection_path": _display_path(canonical_selection_path, root),
            "canonical_selection_sha256": _sha256_file(canonical_selection_path),
            "canonical_selection_status": final_selection["status"],
            "final_selected_asset_ids": final_selected_ids,
            "replacement_reasons": final_selection["replacements"],
        }
    )
    final_source_lock = _write_source_lock(
        output_path=final_source_lock,
        technical_manifest_path=technical_path,
        registration=final_registration,
        selected_ids=final_selected_ids,
        id_to_row=id_to_row,
        resume=resume,
    )
    canonical_lock_path = stratum_dir / "canonical" / f"cohort{stage_suffix}.canonical.lock.yaml"
    canonical_lock = None
    if final_selection["selected_count"] == int(details["target_count"]):
        canonical_lock = _write_canonical_lock(
            source_lock_path=final_source_lock,
            canonical_lock_path=canonical_lock_path,
            final_selection=final_selection,
            technical_rows=id_to_row,
            artifacts=artifacts,
            resume=resume,
        )
    result.update(
        {
            "materialization_stage": materialization_stage,
            "status": final_selection["status"],
            "canonical_selection": {
                "path": _display_path(canonical_selection_path, root),
                "sha256": _sha256_file(canonical_selection_path),
                "selected_count": final_selection["selected_count"],
                "reserve_count": final_selection["reserve_count"],
                "excluded_count": final_selection["excluded_count"],
                "replacement_count": len(final_selection["replacements"]),
            },
            "canonical_selected_source_lock": {
                "path": _display_path(final_source_lock, root),
                "sha256": _sha256_file(final_source_lock),
                "schema_version": "1.1.0",
            },
            "canonical_lock": (
                {
                    "path": _display_path(canonical_lock, root),
                    "sha256": _sha256_file(canonical_lock),
                    "schema_version": "1.2.0",
                    "strict_pregrasp_status": "pending-pregrasp",
                }
                if canonical_lock is not None
                else None
            ),
        }
    )
    return result


def _audit_only(plan: Mapping[str, Any], *, plan_path: Path, root: Path) -> dict[str, Any]:
    r"""只读检查 source plan 的四 strata/路径/schema，完全不写 technical manifest 或 lock。"""

    strata_summary = {}
    for name, details in plan["strata"].items():
        details = _mapping(details, label=f"stratum {name}")
        selected = details["selected"]
        reserve = details["reserve"]
        strata_summary[name] = {
            "source_suite": details.get("source_suite"),
            "target_count": details.get("target_count"),
            "selected_count": len(selected),
            "reserve_count": len(reserve),
            "excluded_count": len(details["excluded"]),
            "source_paths_present": sum(
                bool(_mapping(item.get("source_asset"), label="source asset").get("path"))
                for item in [*selected, *reserve]
            ),
            "canonical_status": "pending-canonical",
        }
    return {
        "artifact_type": "anymani.family_teacher_distillation.evaluation_cohort_materialization_audit",
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "status": "pending-canonical",
        "audit_only": True,
        "policy_results_read": False,
        "selection_plan": {"path": _display_path(plan_path, root), "sha256": _sha256_file(plan_path)},
        "strata": strata_summary,
        "canonical_physical_hash_comparison": "not-performed-in-audit-only",
    }


def _audit_n040_train(
    plan: Mapping[str, Any],
    *,
    root: Path,
    output_dir: Path,
    limit: int | None = None,
) -> tuple[set[str] | None, dict[str, Any]]:
    r"""对最终 N040 train 做同域 canonical audit；仅由显式 --audit-exposures 触发。"""

    train_manifest_path = _plan_input_path(plan, "n040_train_manifest", root)
    from anymani.assets.bank.dataset import HandAssetDataset
    from anymani.assets.bank.prepared_train import resolve_prepared_train

    dataset = HandAssetDataset.from_yaml(train_manifest_path)
    # bounded smoke 直接走 resolver 的 max_assets 分支；不能先准备完整 8192 再切片，
    # 否则 --audit-exposures-limit 会在表面上有限、实际仍触发长 IO。
    partition, cache_hit = resolve_prepared_train(
        dataset,
        require_geometry_semantics=True,
        max_assets=limit,
    )
    records_by_id = {
        record.container.asset_id: {
            "asset_id": record.container.asset_id,
            "family": record.provenance.group_name,
            "topology_key": record.provenance.mother_name,
        }
        for record in partition.records
    }
    asset_rows = {record.container.asset_id: index for index, record in enumerate(partition.records)}
    artifacts = _canonicalize_partition(
        partition,
        root=root,
        output_root=output_dir / "exposure-audit" / "n040-train-canonical-artifacts",
        asset_rows=asset_rows,
        records_by_id=records_by_id,
    )
    hashes = {str(artifact["physical_geometry_hash"]) for artifact in artifacts.values()}
    complete = limit is None and len(artifacts) == len(partition.records) and len(hashes) == len(artifacts)
    status = "complete" if complete else ("canonical-duplicate" if limit is None else "bounded-prefix")
    audit = {
        "status": status,
        "source_manifest": _display_path(train_manifest_path, root),
        "source_manifest_sha256": _sha256_file(train_manifest_path),
        "resolved_record_count": len(partition.records),
        "canonicalized_record_count": len(artifacts),
        "canonical_physical_hash_count": len(hashes),
        "canonical_physical_hashes_unique": len(hashes) == len(artifacts),
        "prepared_cache_hit": cache_hit,
        "algorithm": CANONICAL_PHYSICAL_ALGORITHM,
    }
    return hashes, audit


def materialize(
    *,
    selection_plan_path: Path,
    output_dir: Path,
    repo_root: Path,
    canonicalize: bool,
    audit_only: bool,
    audit_exposures: bool,
    audit_exposures_limit: int | None,
    resume: bool,
) -> dict[str, Any]:
    r"""执行 CLI 的读、technical registration、候选 canonicalization 和可选 full exposure audit。"""

    if audit_only and (canonicalize or audit_exposures):
        raise ValueError("--audit-only cannot be combined with --canonicalize/--audit-exposures")
    if audit_exposures and not canonicalize:
        raise ValueError("--audit-exposures requires --canonicalize")
    if audit_exposures_limit is not None and audit_exposures_limit < 1:
        raise ValueError("--audit-exposures-limit must be positive")
    root = repo_root.expanduser().resolve(strict=True)  # 统一所有 plan/source/output 路径
    plan_path = _resolve_path(selection_plan_path, root)
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    plan = _load_selection_plan(plan_path)
    if audit_only:
        return _audit_only(plan, plan_path=plan_path, root=root)
    resolved_output = _resolve_path(output_dir, root)
    if resolved_output.exists() and not resume:
        raise FileExistsError(f"output-dir must be new unless --resume is supplied: {resolved_output}")
    resolved_output.mkdir(parents=True, exist_ok=True)
    source_manifest_path = _plan_input_path(plan, "n040_train_manifest", root)
    eval_manifest_path = _plan_input_path(plan, "n040_extended512_eval_manifest", root)
    source_manifest = _mapping(_load_document(source_manifest_path), label="N040 source manifest")
    teacher_hashes, teacher_audit = _teacher_canonical_hashes(plan, root=root)
    n040_train_hashes: set[str] | None = None
    n040_train_audit_complete = False
    n040_audit: dict[str, Any] = {"status": "not-run", "reason": "explicit --audit-exposures not supplied"}
    if audit_exposures:
        n040_train_hashes, n040_audit = _audit_n040_train(
            plan,
            root=root,
            output_dir=resolved_output,
            limit=audit_exposures_limit,
        )
        n040_train_audit_complete = n040_audit["status"] == "complete"
    strata_results = {}
    for stratum, details in plan["strata"].items():
        strata_results[stratum] = _materialize_stratum(
            plan=plan,
            plan_path=plan_path,
            source_manifest=source_manifest,
            eval_manifest_path=eval_manifest_path,
            stratum=stratum,
            details=_mapping(details, label=f"stratum {stratum}"),
            output_dir=resolved_output,
            root=root,
            canonicalize=canonicalize,
            audit_n040_train_hashes=n040_train_hashes,
            n040_train_audit_complete=n040_train_audit_complete,
            teacher_hashes=teacher_hashes,
            resume=resume,
        )
    status = "source-registered-pending-canonical"
    if canonicalize:
        statuses = {str(result["status"]) for result in strata_results.values()}
        status = (
            "canonicalized-and-exposure-audited-pending-pregrasp"
            if audit_exposures and statuses == {"canonicalized-and-exposure-audited-pending-pregrasp"}
            else "canonicalized-pending-exposure-audit"
        )
    summary = {
        "artifact_type": "anymani.family_teacher_distillation.evaluation_cohort_materialization",
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "status": status,
        "policy_results_read": False,
        "selection_plan": {
            "path": _display_path(plan_path, root),
            "sha256": _sha256_file(plan_path),
            "selection_digest": plan.get("selection_digest"),
        },
        "source_manifest": {
            "path": _display_path(source_manifest_path, root),
            "sha256": _sha256_file(source_manifest_path),
        },
        "evaluation_manifest": {
            "path": _display_path(eval_manifest_path, root),
            "sha256": _sha256_file(eval_manifest_path),
        },
        "teacher_canonical_audit": teacher_audit,
        "n040_train_canonical_audit": n040_audit,
        "canonical_physical_algorithm": CANONICAL_PHYSICAL_ALGORITHM,
        "strict_pregrasp_ready": False,
        "strata": strata_results,
        "next_stage": {
            "canonical_lock_paths": {
                stratum: (result["canonical_lock"]["path"] if result.get("canonical_lock") is not None else None)
                for stratum, result in strata_results.items()
            },
            "prepare_command_template": [
                "/home/hac/isaac/IsaacLab/isaaclab.sh",
                "-p",
                "scripts/research/prepare_cohort_pregrasp_shards.py",
                "--cohort-lock",
                "<canonical_lock_paths[stratum]>",
                "--output-dir",
                "<stratum>/pregrasp",
                "--shard-assets",
                "16",
                "--catalog",
                "<stratum>/pregrasp/catalog",
            ],
            "condition": "only after canonical source/exposure audit; this script does not run pregrasp",
        },
    }
    # source、canonical 与 exposure audit 是三个可恢复阶段，各自封存摘要；
    # 进入后续阶段不能覆盖早期证据，也不能因早期摘要已存在而拒绝合法推进。
    summary_name = (
        "materialization-summary.exposure-audited.json" if n040_train_audit_complete
        else "materialization-summary.canonical.json" if canonicalize
        else "materialization-summary.json"
    )
    summary_path = resolved_output / summary_name
    _write_new_or_verify(
        summary_path,
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        resume=resume,
    )
    summary["summary_path"] = _display_path(summary_path, root)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    r"""解析最小 CLI 并打印 source/canonical materialization 摘要。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--audit-only", action="store_true", help="只读审计 plan，不写 technical manifest/source lock")
    parser.add_argument("--canonicalize", action="store_true", help="CPU canonicalize candidate selected+reserve")
    parser.add_argument(
        "--audit-exposures",
        action="store_true",
        help="canonicalize final N040 train exposure too; primary-owned potentially long 8192-asset stage",
    )
    parser.add_argument(
        "--audit-exposures-limit",
        type=int,
        default=None,
        help="bounded prefix for a smoke/audit probe; never represents complete exposure audit",
    )
    parser.add_argument(
        "--resume", action="store_true", help="reuse only byte-identical files in an existing output dir"
    )
    args = parser.parse_args(argv)
    summary = materialize(
        selection_plan_path=args.selection_plan,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
        canonicalize=args.canonicalize,
        audit_only=args.audit_only,
        audit_exposures=args.audit_exposures,
        audit_exposures_limit=args.audit_exposures_limit,
        resume=args.resume,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
