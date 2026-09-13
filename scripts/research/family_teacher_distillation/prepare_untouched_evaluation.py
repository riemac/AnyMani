r"""注册已经冻结的 untouched evaluation source strata。

本入口消费 ``selected-source-assets.json`` 的直接 source records，不重新读取策略结果、
不重新选择 reserve，也不改写 N040 evaluation manifest。每个 stratum 先生成一份标准
schema-2 technical manifest；该 manifest 的 ``train`` 只是为了满足现有
``HandAssetDataset`` / cohort writer 的非空注册接口，外部 registration selection 明确把
它标成 evaluation-only，不能解释为 SSL/PPO 训练暴露。

``--canonicalize`` 是显式的第二阶段：它对已选中的 32 个 source bundles 调用现有
``canonical_runtime.materialize_canonical_artifact`` 与
``finalize_hand_asset_cohort_lock``，不把 reserve 当作替换池，不把初始化失败的 selected
成员静默删除。canonical-final lock 的 strict pregrasp 仍由下游入口完成。

四个输入 strata 的固定科学配额为：

* ``leap_right_variant``：32 个 variant；
* ``allegro_right_variant``：32 个 variant；
* ``leap_right_mother``：8 个 mother + 24 个 variant；
* ``allegro_right_mother``：8 个 mother + 24 个 variant。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

ADAPTER_SCHEMA_VERSION = "1.0.0"
"""适配器 registration/canonical audit 文档的稳定 schema 版本。"""

TECHNICAL_DATASET_SCHEMA_VERSION = "2.0.0"
"""HandAssetDataset 的技术注册 manifest schema。"""

CANONICAL_LOCK_NAME = "cohort.canonical.lock.yaml"
"""canonical-final 文件名；跨集合物理身份比较与 strict pregrasp 仍由下游完成。"""

PURE_GROUP_BY_FAMILY = {
    "leap": "single_palm_leap",
    "allegro": "single_palm_allegro",
}
"""两个允许的 pure-right production group；mixed/left 永不由本入口注册。"""

STRATUM_SPECS: dict[str, dict[str, Any]] = {
    "leap_right_variant": {
        "family": "leap",
        "kind": "variant",
        "selected_count": 32,
        "base_designs": 8,
        "selected_kind_counts": {"variant": 32},
    },
    "allegro_right_variant": {
        "family": "allegro",
        "kind": "variant",
        "selected_count": 32,
        "base_designs": 8,
        "selected_kind_counts": {"variant": 32},
    },
    "leap_right_mother": {
        "family": "leap",
        "kind": "new_base",
        "selected_count": 32,
        "base_designs": 8,
        "selected_kind_counts": {"mother": 8, "variant": 24},
    },
    "allegro_right_mother": {
        "family": "allegro",
        "kind": "new_base",
        "selected_count": 32,
        "base_designs": 8,
        "selected_kind_counts": {"mother": 8, "variant": 24},
    },
}
"""每个 stratum 的 source selection 配额；不允许适配器自行补配额。"""


def _sha256_file(path: Path) -> str:
    r"""按块计算 source metadata 文件的 SHA-256。"""

    digest = hashlib.sha256()  # 文件 bytes 身份，不与 canonical physical hash 混用
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)  # bounded IO，避免把 source sidecar 大对象复制到额外 buffer
    return digest.hexdigest()


def _resolve_path(path: Path, root: Path) -> Path:
    r"""把相对 source/output path 固定到 AnyMani repository root。"""

    candidate = path if path.is_absolute() else root / path
    return candidate.expanduser().resolve(strict=False)


def _display_path(path: Path, root: Path) -> str:
    r"""优先写 root-relative provenance，root 外 source 保留绝对路径。"""

    resolved = path.resolve(strict=False)
    try:
        return resolved.relative_to(root.resolve(strict=False)).as_posix()
    except ValueError:
        return str(resolved)


def _load_document(path: Path) -> Any:
    r"""优先按 JSON 读取 canonical/selection bytes，必要时回退安全 YAML。"""

    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)  # selected list 与 cohort lock 都以 canonical JSON 为主
    except json.JSONDecodeError:
        document = yaml.safe_load(text)  # technical manifest 仍允许普通 YAML 语法
        if document is None:
            raise ValueError(f"metadata document is empty: {path}")
        return document


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    r"""要求 provenance 节点为 mapping，避免错误输入变成空集合。"""

    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _write_new(path: Path, payload: str) -> None:
    r"""只新建当前 invocation 的证据文件，拒绝同名覆盖。"""

    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def _source_asset_path(item: Mapping[str, Any], *, root: Path) -> tuple[Path, Path, Path, str]:
    r"""解析一条 selected/reserve record 的 bundle、mother、run 与 production group。"""

    asset_id = str(item.get("asset_id", "")).strip()
    mother_name = str(item.get("mother_name", "")).strip()
    kind = str(item.get("kind", "")).strip()
    variant_set = str(item.get("variant_set", ""))
    bundle = _resolve_path(Path(str(item.get("bundle_path", ""))), root)
    if not asset_id or not mother_name or kind not in {"mother", "variant"}:
        raise ValueError("source record requires asset_id, mother_name and kind=mother|variant")
    if bundle.name != asset_id:
        if kind == "mother" and bundle.name == mother_name:
            pass  # mother bundle 目录名是 topology/mother_name，不是 asset_id
        else:
            raise ValueError(f"source bundle basename disagrees with asset identity: {bundle}")
    if kind == "mother":
        if variant_set:
            raise ValueError(f"mother source record must have empty variant_set: {asset_id}")
        mother_path = bundle
    else:
        if not variant_set:
            raise ValueError(f"variant source record requires variant_set: {asset_id}")
        mother_path = bundle.parent.parent
        expected_bundle = mother_path / variant_set / asset_id
        if bundle != expected_bundle:
            raise ValueError(f"variant source path does not match mother/variant-set contract: {bundle}")
    if mother_path.name != mother_name:
        raise ValueError(
            f"source mother lineage disagrees with path: declared={mother_name!r}, path={mother_path.name!r}"
        )
    group_name = mother_path.parent.name
    run_root = mother_path.parent.parent
    return bundle, mother_path, run_root, group_name


def _validate_source_record(
    item: Mapping[str, Any],
    *,
    family: str,
    stratum: str,
    root: Path,
) -> dict[str, Any]:
    r"""核对真实 source bundle、sidecar identity、文件 SHA 与 lineage 坐标。"""

    if str(item.get("family", "")) != family or str(item.get("handedness", "")) != "right":
        raise ValueError(f"{stratum} contains a non-{family}-right source record")
    bundle, mother_path, run_root, group_name = _source_asset_path(item, root=root)
    expected_group = PURE_GROUP_BY_FAMILY[family]
    if group_name != expected_group:
        raise ValueError(f"{stratum} source group must be {expected_group!r}, got {group_name!r}")
    hand_yaml = _resolve_path(Path(str(item.get("hand_yaml", ""))), root)
    hand_urdf = _resolve_path(Path(str(item.get("hand_urdf", ""))), root)
    if (hand_yaml, hand_urdf) != (bundle / "hand.yaml", bundle / "hand.urdf"):
        raise ValueError(f"source file paths do not match bundle directory: {bundle}")
    if not hand_yaml.is_file() or not hand_urdf.is_file():
        raise FileNotFoundError(f"selected/reserve source bundle is incomplete: {bundle}")

    # 选择阶段已经封存 source file SHA；适配器重新核对 bytes，禁止把 stale selection 注册成新 cohort。
    declared_files = _mapping(item.get("source_file_sha256"), label=f"{stratum} source_file_sha256")
    actual_files = {"hand.yaml": hand_yaml, "hand.urdf": hand_urdf}
    for name, path in actual_files.items():
        expected = str(declared_files.get(name, ""))
        if not expected or expected != _sha256_file(path):
            raise ValueError(f"{stratum} source file SHA changed for {item.get('asset_id')}: {name}")
    sidecar = _mapping(_load_document(hand_yaml), label=f"source sidecar {hand_yaml}")
    asset_id = str(item["asset_id"])
    if str(sidecar.get("id", "")) != asset_id:
        raise ValueError(f"source sidecar ID disagrees for {asset_id}")
    if str(sidecar.get("family", "")) != family or str(sidecar.get("handedness", "")) != "right":
        raise ValueError(f"source sidecar family/handedness disagrees for {asset_id}")
    if not str(item.get("static_geometry_fingerprint", "")):
        raise ValueError(f"source record lacks static geometry fingerprint: {asset_id}")
    if item.get("declared_fingerprint_matches_current") is not True:
        raise ValueError(f"source record fingerprint audit is not complete: {asset_id}")
    rejection_reasons = item.get("rejection_reasons", [])
    if rejection_reasons:
        raise ValueError(f"selected/reserve source record carries rejection reasons: {asset_id}")
    normalized = dict(item)
    normalized["_bundle_path"] = bundle
    normalized["_mother_path"] = mother_path
    normalized["_run_root"] = run_root
    normalized["_group_name"] = group_name
    return normalized


def _validate_selected_shape(
    *,
    stratum: str,
    details: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    reserve: Sequence[Mapping[str, Any]],
) -> None:
    r"""锁定四格 selected 配额、mother/variant 比例与每格八个 lineage。"""

    spec = STRATUM_SPECS[stratum]
    if len(selected) != spec["selected_count"]:
        raise ValueError(f"{stratum} selected count must be {spec['selected_count']}")
    if int(details.get("nominal_assets", -1)) != spec["selected_count"]:
        raise ValueError(f"{stratum} nominal_assets disagrees with the fixed 32-member contract")
    if int(details.get("base_designs", -1)) != spec["base_designs"]:
        raise ValueError(f"{stratum} must contain exactly eight mother lineages")
    kinds = Counter(str(item.get("kind", "")) for item in selected)
    if dict(kinds) != spec["selected_kind_counts"]:
        raise ValueError(f"{stratum} selected mother/variant proportions are invalid: {dict(kinds)}")
    selected_by_mother: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in selected:
        selected_by_mother[str(item.get("mother_name", ""))].append(item)
    if len(selected_by_mother) != spec["base_designs"] or any(
        len(group) != 4 for group in selected_by_mother.values()
    ):
        raise ValueError(f"{stratum} selected assets must be four per each of eight mother lineages")
    if spec["kind"] == "new_base":
        for mother_name, group in selected_by_mother.items():
            if sum(str(item.get("kind", "")) == "mother" for item in group) != 1:
                raise ValueError(f"{stratum} lineage {mother_name!r} must select exactly one mother")
    else:
        if any(str(item.get("kind", "")) == "mother" for item in selected):
            raise ValueError(f"{stratum} variant-only selection cannot contain a mother")
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    for role, records in (("selected", selected), ("reserve", reserve)):
        for item in records:
            asset_id = str(item.get("asset_id", ""))
            bundle = Path(str(item["_bundle_path"]))
            if asset_id in seen_ids or bundle in seen_paths:
                raise ValueError(f"duplicate source identity across selected/reserve in {stratum}: {asset_id}")
            seen_ids.add(asset_id)
            seen_paths.add(bundle)
            if role == "reserve" and str(item.get("kind", "")) == "mother":
                raise ValueError(f"reserve cannot contain a mother in {stratum}")


def _load_selection(path: Path, *, root: Path) -> tuple[dict[str, Any], str]:
    r"""加载并完整验证 selection file，返回文档与其原始 bytes SHA。"""

    if not path.is_file():
        raise FileNotFoundError(path)
    selection_sha = _sha256_file(path)
    document = dict(_mapping(_load_document(path), label="untouched selection"))
    if document.get("status") != "static-geometry-isolated-pending-canonical-and-pregrasp":
        raise ValueError("selection file is not the pending-canonical untouched holdout artifact")
    if document.get("original_registry_changed") is not False:
        raise ValueError("selection must prove that the original registry was unchanged")
    if document.get("policy_results_read") is not False:
        raise ValueError("selection must prove that policy results were not read")
    strata = _mapping(document.get("strata"), label="untouched selection strata")
    if set(strata) != set(STRATUM_SPECS):
        raise ValueError(f"selection strata must be exactly {tuple(STRATUM_SPECS)}")
    if int(document.get("nominal_assets", -1)) != 128:
        raise ValueError("untouched selection nominal_assets must be 128")
    if document.get("family_assets") != {"leap": 64, "allegro": 64}:
        raise ValueError("untouched selection family_assets must contain 64 Leap and 64 Allegro assets")

    # 选择脚本的上游输入也是 provenance；存在且 SHA 不变才允许继续注册。
    sources = document.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("untouched selection must bind at least one source file")
    for source in sources:
        source = _mapping(source, label="selection source")
        source_path = _resolve_path(Path(str(source.get("path", ""))), root)
        if not source_path.is_file() or _sha256_file(source_path) != str(source.get("sha256", "")):
            raise ValueError(f"selection input source changed or missing: {source_path}")

    all_selected: list[dict[str, Any]] = []
    all_source_ids: set[str] = set()
    all_source_paths: set[Path] = set()
    for stratum, raw_details in strata.items():
        details = _mapping(raw_details, label=f"selection stratum {stratum}")
        selected_raw = details.get("selected")
        reserve_raw = details.get("reserve")
        if not isinstance(selected_raw, list) or not isinstance(reserve_raw, list):
            raise ValueError(f"{stratum} requires list-valued selected and reserve")
        family = STRATUM_SPECS[stratum]["family"]
        selected = [_validate_source_record(_mapping(item, label=f"{stratum}.selected"), family=family, stratum=stratum, root=root) for item in selected_raw]
        reserve = [_validate_source_record(_mapping(item, label=f"{stratum}.reserve"), family=family, stratum=stratum, root=root) for item in reserve_raw]
        _validate_selected_shape(stratum=stratum, details=details, selected=selected, reserve=reserve)
        details_copy = dict(details)
        details_copy["selected"] = selected
        details_copy["reserve"] = reserve
        strata[stratum] = details_copy
        all_selected.extend(selected)
        for item in [*selected, *reserve]:
            asset_id = str(item["asset_id"])
            bundle = Path(item["_bundle_path"])
            if asset_id in all_source_ids or bundle in all_source_paths:
                raise ValueError(f"source identity is duplicated across strata: {asset_id}")
            all_source_ids.add(asset_id)
            all_source_paths.add(bundle)
    document["strata"] = dict(strata)
    return document, selection_sha


def build_technical_manifest(
    strata_details: Mapping[str, Any],
    *,
    family: str,
) -> dict[str, Any]:
    r"""从 selected+reserve 的真实 lineage 构造标准 schema-2 technical manifest。

    manifest 只表达完整 variant-set 的目录原子；selected 个体通过后续 source lock 的
    ``technical_manifest_row`` 精确绑定。此函数从不把 selected/reserve 重新排序为实验轴。
    """

    run_groups: dict[Path, dict[str, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for item in [*strata_details["selected"], *strata_details["reserve"]]:
        run_root = Path(item["_run_root"])
        group_name = str(item["_group_name"])
        mother_name = str(item["mother_name"])
        lineage = run_groups[run_root][group_name].setdefault(
            mother_name,
            {"include_mother": False, "variant_sets": set()},
        )
        if str(item["kind"]) == "mother":
            lineage["include_mother"] = True
        else:
            lineage["variant_sets"].add(str(item["variant_set"]))
    if not run_groups:
        raise ValueError(f"{family} technical manifest has no source lineages")
    runs: dict[str, Any] = {}
    for ordinal, run_root in enumerate(sorted(run_groups, key=str)):
        groups: dict[str, Any] = {}
        for group_name, mothers in run_groups[run_root].items():
            groups[group_name] = {
                mother_name: {
                    "include_mother": bool(lineage["include_mother"]),
                    "variant_sets": sorted(lineage["variant_sets"]),
                }
                for mother_name, lineage in mothers.items()
            }
        runs[f"source_{ordinal:02d}"] = {"run_dir": str(run_root), "groups": groups}
    return {
        "schema_version": TECHNICAL_DATASET_SCHEMA_VERSION,
        "default_run_dir": str(sorted(run_groups, key=str)[0]),
        "train": {"runs": runs},
        "validation": {"unseen_variant_set": {"runs": {}}, "unseen_mother": {"runs": {}}},
        "evaluation": {
            "unseen_variant_set": {"runs": {}},
            "unseen_mother": {"runs": {}},
            "official_zero_shot": {"assets": []},
        },
    }


def _resolve_technical_partition(
    technical_path: Path,
    *,
    expected_items: Sequence[Mapping[str, Any]],
) -> tuple[Any, dict[str, int]]:
    r"""用真实 HandAssetDataset resolver 展开技术 manifest 并冻结 asset_id→dense row。"""

    from anymani.assets.bank.dataset import HandAssetDataset

    dataset = HandAssetDataset.from_yaml(technical_path)
    partition = dataset.resolve_train(require_geometry_semantics=True)
    id_to_row: dict[str, int] = {}
    path_by_id: dict[str, Path] = {}
    for row, record in enumerate(partition.records):
        asset_id = record.container.asset_id
        bundle = record.container.urdf_path.parent.resolve(strict=True)
        if asset_id in id_to_row or bundle in path_by_id.values():
            raise ValueError(f"technical manifest contains duplicate asset identity: {asset_id}")
        id_to_row[asset_id] = row
        path_by_id[asset_id] = bundle
    expected_ids = {str(item["asset_id"]) for item in expected_items}
    missing = sorted(expected_ids - set(id_to_row))
    if missing:
        raise ValueError(f"technical manifest is missing selected/reserve source assets: {missing}")
    # 完整 variant-set 展开可能包含 selection plan 未选中的额外候选（例如静态几何去重后的
    # 未选成员）。这些候选保留在 technical registration 中，不能被本适配器擅自删除；selected
    # source lock 仍只使用下方显式 expected_items 的坐标，因此它们不会进入 evaluation cohort。
    for item in expected_items:
        asset_id = str(item["asset_id"])
        record = partition.records[id_to_row[asset_id]]
        expected_bundle = Path(item["_bundle_path"]).resolve(strict=True)
        actual_bundle = record.container.urdf_path.parent.resolve(strict=True)
        if actual_bundle != expected_bundle:
            raise ValueError(f"technical resolver path changed for {asset_id}")
        provenance = record.provenance
        expected_role = "mother" if str(item["kind"]) == "mother" else "variant"
        if (
            provenance.mother_name != str(item["mother_name"])
            or Path(provenance.mother_path).resolve(strict=True) != Path(item["_mother_path"]).resolve(strict=True)
            or provenance.variant_set != str(item["variant_set"])
            or provenance.asset_role != expected_role
        ):
            raise ValueError(f"technical resolver lineage changed for {asset_id}")
    return partition, id_to_row


def _origin_document(
    item: Mapping[str, Any],
    *,
    role: str,
    selection_index: int,
    technical_row: int,
    root: Path,
) -> dict[str, Any]:
    r"""把 direct selection record 封装为 source lock 可读的 origin provenance。"""

    return {
        "selection_index": selection_index,
        "selection_role": role,
        "technical_manifest_row": technical_row,
        "asset_id": str(item["asset_id"]),
        "family": str(item["family"]),
        "handedness": str(item["handedness"]),
        "kind": str(item["kind"]),
        "asset_role": "mother" if str(item["kind"]) == "mother" else "variant",
        "mother_name": str(item["mother_name"]),
        "mother_path": _display_path(Path(item["_mother_path"]), root),
        "variant_set": str(item["variant_set"]),
        "bundle_path": _display_path(Path(item["_bundle_path"]), root),
        "hand_yaml": _display_path(_resolve_path(Path(str(item["hand_yaml"])), root), root),
        "hand_urdf": _display_path(_resolve_path(Path(str(item["hand_urdf"])), root), root),
        "source_file_sha256": dict(item["source_file_sha256"]),
        "static_geometry_fingerprint": str(item["static_geometry_fingerprint"]),
        "geometry_fingerprint": item.get("geometry_fingerprint"),
        "post_mutate_modes": item.get("post_mutate_modes"),
        "geometry_change_fields": item.get("geometry_change_fields"),
        "source_selection_sort_key": item.get("selection_sort_key"),
    }


def _registration_document(
    *,
    selection_path: Path,
    selection_sha: str,
    stratum: str,
    family: str,
    details: Mapping[str, Any],
    technical_path: Path,
    technical_sha: str,
    id_to_row: Mapping[str, int],
    root: Path,
) -> dict[str, Any]:
    r"""构造独立 registration.json/source-lock.selection contract。"""

    selected = [
        _origin_document(item, role="selected", selection_index=index, technical_row=id_to_row[str(item["asset_id"])], root=root)
        for index, item in enumerate(details["selected"])
    ]
    reserve = [
        _origin_document(item, role="reserve", selection_index=index, technical_row=id_to_row[str(item["asset_id"])], root=root)
        for index, item in enumerate(details["reserve"])
    ]
    return {
        "artifact_type": "anymani.family_teacher_distillation.untouched_evaluation_source_registration",
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "status": "source-registered-pending-canonical",
        "stratum": stratum,
        "family": family,
        "source_selection_path": _display_path(selection_path, root),
        "source_selection_sha256": selection_sha,
        "source_selection_status": "static-geometry-isolated-pending-canonical-and-pregrasp",
        "technical_manifest_path": _display_path(technical_path, root),
        "technical_manifest_sha256": technical_sha,
        "technical_manifest_schema": TECHNICAL_DATASET_SCHEMA_VERSION,
        "registration_mode": "evaluation_only_as_technical_train_for_cohort_api",
        "registration_is_not_policy_or_ssl_training": True,
        "not_n040_evaluation_manifest": True,
        "reserve_is_not_selected": True,
        "selected_count": len(selected),
        "reserve_count": len(reserve),
        "selected_asset_ids": [item["asset_id"] for item in selected],
        "reserve_asset_ids": [item["asset_id"] for item in reserve],
        "selected_origin_members": selected,
        "reserve_origin_members": reserve,
    }


def _write_source_lock(
    *,
    output_path: Path,
    technical_path: Path,
    registration: Mapping[str, Any],
    selected_ids: Sequence[str],
    id_to_row: Mapping[str, int],
) -> Path:
    r"""调用标准 cohort writer 发布 selected-only schema-1.1 source lock。"""

    from anymani.assets.bank.cohort import load_hand_asset_cohort, write_hand_asset_cohort_lock

    selection = {
        **dict(registration),
        "source_lock_schema": "1.1.0",
        "source_lock_selection_contract": "selected origin order is immutable; technical train is registration-only",
    }
    coordinates = tuple(("evaluation_only", id_to_row[str(asset_id)]) for asset_id in selected_ids)
    source_lock = write_hand_asset_cohort_lock(
        output_path,
        cohort_id=f"{registration['stratum']}-untouched-source",
        source_manifests={"evaluation_only": technical_path},
        member_coordinates=coordinates,
        selection=selection,
        require_geometry_semantics=True,
    )
    loaded = load_hand_asset_cohort(source_lock, require_geometry_semantics=True)
    if tuple(member.asset_id for member in loaded.members) != tuple(selected_ids):
        raise AssertionError("source writer changed selected member order")
    return source_lock


def _canonicalize_selected(
    *,
    partition: Any,
    selected: Sequence[Mapping[str, Any]],
    id_to_row: Mapping[str, int],
    output_root: Path,
    root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    r"""仅对 selected 32 调用现有 canonical runtime；reserve 不进入 lowering/替换。"""

    from anymani.assets.asset_sidecar import restore_hand_cfg_snapshot
    from anymani.assets.canonical_runtime import (
        CANONICAL_HAND_SCHEMA_V1,
        materialize_canonical_artifact,
        validate_canonical_artifact,
    )

    artifacts: dict[str, dict[str, Any]] = {}
    for item in selected:
        asset_id = str(item["asset_id"])
        record = partition.records[id_to_row[asset_id]]
        raw_hand_cfg = record.container.sidecar.get("hand_cfg")
        if not isinstance(raw_hand_cfg, dict):
            raise ValueError(f"selected source sidecar lacks hand_cfg: {asset_id}")
        hand_cfg = restore_hand_cfg_snapshot(raw_hand_cfg)  # sidecar 是 typed HandCfg 真源
        semantics = record.container.geometry_semantics
        q_home = tuple(semantics.q_home_rad) if semantics is not None else ()  # canonical startup pose，单位 rad
        q_home_names = tuple(semantics.active_joint_names) if semantics is not None else ()
        artifact = materialize_canonical_artifact(
            hand_cfg,
            asset_id=asset_id,
            output_root=output_root,
            source_urdf_path=record.container.urdf_path,
            schema=CANONICAL_HAND_SCHEMA_V1,
            asset_row=id_to_row[asset_id],
            topology=str(item["mother_name"]),
            q_home=q_home,
            q_home_joint_names=q_home_names,
        )
        validate_canonical_artifact(artifact, schema=CANONICAL_HAND_SCHEMA_V1)
        artifacts[asset_id] = {
            "asset_id": asset_id,
            "technical_manifest_row": id_to_row[asset_id],
            "configuration_domain_hash": artifact.source_content_hash,
            "source_content_hash": artifact.source_content_hash,
            "source_urdf_hash": artifact.source_urdf_hash,
            "physical_geometry_hash": artifact.physical_geometry_hash,
            "canonical_schema_digest": artifact.schema_digest,
            "canonical_urdf_hash": artifact.canonical_urdf_hash,
            "canonical_urdf_path": _display_path(Path(artifact.canonical_urdf_path), root),
            "canonical_manifest_path": _display_path(Path(artifact.manifest_path), root),
            "canonical_physical_algorithm": "canonical-runtime-lowering",
        }
    audit = {
        "status": "canonicalized-selected-only-pending-pregrasp",
        "selected_count": len(selected),
        "reserve_canonicalized": False,
        "canonical_physical_hashes_unique": len(
            {str(item["physical_geometry_hash"]) for item in artifacts.values()}
        )
        == len(artifacts),
        "artifacts": artifacts,
    }
    return artifacts, audit


def _write_canonical_lock(
    *,
    source_lock: Path,
    output_path: Path,
    selected_ids: Sequence[str],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> Path:
    r"""调用标准 canonical-finalizer，拒绝 selected 物理 hash 重复。"""

    from anymani.assets.bank.cohort import finalize_hand_asset_cohort_lock

    identities = tuple(
        (
            str(artifacts[asset_id]["configuration_domain_hash"]),
            str(artifacts[asset_id]["physical_geometry_hash"]),
            str(artifacts[asset_id]["canonical_schema_digest"]),
        )
        for asset_id in selected_ids
    )
    return finalize_hand_asset_cohort_lock(
        source_lock,
        output_path,
        canonical_identities=identities,
        canonical_schema_version="v1",
        require_geometry_semantics=True,
    )


def _materialize_stratum(
    *,
    selection_path: Path,
    selection_sha: str,
    stratum: str,
    details: Mapping[str, Any],
    output_dir: Path,
    root: Path,
    canonicalize: bool,
) -> dict[str, Any]:
    r"""完成单个 stratum 的 technical manifest、source lock 与可选 canonical lock。"""

    family = str(STRATUM_SPECS[stratum]["family"])
    technical_document = build_technical_manifest(details, family=family)
    stratum_dir = output_dir / stratum
    source_dir = stratum_dir / "source"
    technical_path = source_dir / "evaluation_only_manifest.yaml"
    _write_new(technical_path, yaml.safe_dump(technical_document, sort_keys=False, allow_unicode=True))
    technical_sha = _sha256_file(technical_path)
    expected_items = [*details["selected"], *details["reserve"]]
    partition, id_to_row = _resolve_technical_partition(technical_path, expected_items=expected_items)
    registration = _registration_document(
        selection_path=selection_path,
        selection_sha=selection_sha,
        stratum=stratum,
        family=family,
        details=details,
        technical_path=technical_path,
        technical_sha=technical_sha,
        id_to_row=id_to_row,
        root=root,
    )
    registration_path = source_dir / "registration.json"
    _write_new(registration_path, json.dumps(registration, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    selected_ids = [str(item["asset_id"]) for item in details["selected"]]
    source_lock_path = source_dir / "source.lock.yaml"
    source_lock = _write_source_lock(
        output_path=source_lock_path,
        technical_path=technical_path,
        registration=registration,
        selected_ids=selected_ids,
        id_to_row=id_to_row,
    )
    result: dict[str, Any] = {
        "stratum": stratum,
        "family": family,
        "selected_count": len(selected_ids),
        "reserve_count": len(details["reserve"]),
        "selected_asset_ids": selected_ids,
        "selected_kind_counts": dict(Counter(str(item["kind"]) for item in details["selected"])),
        "registration_is_not_policy_or_ssl_training": True,
        "not_n040_evaluation_manifest": True,
        "technical_manifest": {
            "path": _display_path(technical_path, root),
            "sha256": technical_sha,
            "schema_version": TECHNICAL_DATASET_SCHEMA_VERSION,
            "evaluation_only": True,
            "resolved_record_count": len(partition.records),
        },
        "registration": {
            "path": _display_path(registration_path, root),
            "sha256": _sha256_file(registration_path),
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

    artifacts, canonical_audit = _canonicalize_selected(
        partition=partition,
        selected=details["selected"],
        id_to_row=id_to_row,
        output_root=stratum_dir / "canonical_artifacts",
        root=root,
    )
    canonical_selection_path = stratum_dir / "canonical-selection.json"
    canonical_selection_document = {
        "artifact_type": "anymani.family_teacher_distillation.untouched_canonical_selection_audit",
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "status": canonical_audit["status"],
        "stratum": stratum,
        "source_selection_path": _display_path(selection_path, root),
        "source_selection_sha256": selection_sha,
        "source_lock": _display_path(source_lock, root),
        "source_lock_sha256": _sha256_file(source_lock),
        "registration_is_not_policy_or_ssl_training": True,
        "reserve_canonicalized": False,
        "selected_asset_ids": selected_ids,
        "artifacts": artifacts,
    }
    _write_new(
        canonical_selection_path,
        json.dumps(canonical_selection_document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    canonical_path = stratum_dir / "canonical" / CANONICAL_LOCK_NAME
    canonical_lock = _write_canonical_lock(
        source_lock=source_lock,
        output_path=canonical_path,
        selected_ids=selected_ids,
        artifacts=artifacts,
    )
    result.update(
        {
            "status": "canonicalized-pending-pregrasp",
            "canonical_selection": {
                "path": _display_path(canonical_selection_path, root),
                "sha256": _sha256_file(canonical_selection_path),
                "selected_count": len(selected_ids),
                "reserve_canonicalized": False,
            },
            "canonical_lock": {
                "path": _display_path(canonical_lock, root),
                "sha256": _sha256_file(canonical_lock),
                "schema_version": "1.2.0",
                "strict_pregrasp_status": "pending-pregrasp",
            },
        }
    )
    return result


def materialize(
    *,
    selection_path: Path,
    output_dir: Path,
    repo_root: Path,
    canonicalize: bool,
) -> dict[str, Any]:
    r"""执行 untouched selection 的 source registration 与可选 canonical-finalization。"""

    root = repo_root.expanduser().resolve(strict=True)
    resolved_selection = _resolve_path(selection_path, root)
    plan, selection_sha = _load_selection(resolved_selection, root=root)
    resolved_output = _resolve_path(output_dir, root)
    if resolved_output.exists():
        raise FileExistsError(f"output-dir must be a new path: {resolved_output}")
    resolved_output.mkdir(parents=True)
    strata_results: dict[str, Any] = {}
    for stratum in STRATUM_SPECS:
        strata_results[stratum] = _materialize_stratum(
            selection_path=resolved_selection,
            selection_sha=selection_sha,
            stratum=stratum,
            details=_mapping(plan["strata"][stratum], label=f"selection stratum {stratum}"),
            output_dir=resolved_output,
            root=root,
            canonicalize=canonicalize,
        )
    status = "canonicalized-pending-pregrasp" if canonicalize else "source-registered-pending-canonical"
    summary = {
        "artifact_type": "anymani.family_teacher_distillation.untouched_evaluation_materialization",
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "status": status,
        "policy_results_read": False,
        "original_registry_changed": False,
        "not_n040_evaluation_manifest": True,
        "registration_is_not_policy_or_ssl_training": True,
        "selection": {
            "path": _display_path(resolved_selection, root),
            "sha256": selection_sha,
            "status": plan["status"],
        },
        "canonicalize_requested": canonicalize,
        "strict_pregrasp_ready": False,
        "strata": strata_results,
    }
    summary_name = "materialization-summary.canonical.json" if canonicalize else "materialization-summary.json"
    summary_path = resolved_output / summary_name
    _write_new(summary_path, json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    summary["summary_path"] = _display_path(summary_path, root)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    r"""解析 source selection、输出目录、repository root 与显式 canonicalize 开关。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument(
        "--canonicalize",
        action="store_true",
        help="explicitly materialize selected canonical artifacts and finalize a schema-1.2 lock",
    )
    args = parser.parse_args(argv)
    summary = materialize(
        selection_path=args.selection,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
        canonicalize=args.canonicalize,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
