r"""为现有 cohort 发布独立的短指限位修订，保留完整成员轴及原始 bundle。

只为实际发生变化的 Allegro 两关节非拇指生成新 bundle/ID；未修改成员仍引用
原 source manifest 与 source row。新旧一一对应关系、数值差分和父文件 SHA
均写入 revision.json。这里不创建仿真，也不把旧预抓取安全证书重命名为新证书。
输出 source lock 后，调用正式 canonical lowering 和 strict pregrasp 管线重验。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from anymani.assets.asset_revisions import widen_two_joint_allegro_flexion
from anymani.assets.asset_schema_core import JointLimitCfg
from anymani.assets.asset_sidecar import restore_hand_cfg_snapshot
from anymani.assets.bank.cohort import load_hand_asset_cohort, write_hand_asset_cohort_lock
from anymani.assets.bank.dataset import HandAssetDataset
from anymani.assets.bank.hand_bank import HandBank, HandBankCfg
from anymani.assets.bank.hand_container import HandContainerCfg
from anymani.assets.exporter.hand_exporter import HandExporter, HandExporterCfg
from anymani.assets.generator.result import HandGenerationResult

REVISION_RULE = "allegro-exact-two-revolute-mcp2-upper-v1"
"""短指设计的稳定规则身份；数值实现仍唯一归属 ``asset_revisions.py``。"""

REVISION_VARIANT_SET = "joint_limit_revision_v1"
"""修订 bundle 的 deterministic property-revision 目录名。"""

REVISION_SOURCE_ALIAS = "joint_limit_revision"
"""新修订 manifest 在 cohort source mapping 中使用的固定 alias。"""


def _sha(path: Path) -> str:
    """绑定确切父资产字节，禁止只记录容易漂移的路径。"""

    return hashlib.sha256(path.read_bytes()).hexdigest()  # URDF/sidecar/本入口均为小文件


def _json(path: Path, document: dict) -> None:
    """在本次独占目录写可人工核对的 JSON；不覆盖已有同名证据。"""

    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2)  # 清晰保留父子轴与单位
        stream.write("\n")


def _require_allegro_right(hand: Any, *, context: str) -> None:
    r"""验证一份候选 ``HandCfg`` 的 family/handedness 物理前提。

    本发布器的规则只对 Allegro-right 原型有定义；尤其是 registration-only parent
    不能因为没有进入 selected 轴就绕过这道检查。这里不从目录名猜 family，避免把
    错误 prototype 当成 2.23 rad 设计的母体。
    """

    if hand.family != "allegro" or hand.handedness != "right":
        raise ValueError(f"{context} must be Allegro-right")


def _assert_only_declared_limit_changes(original: Any, revised: Any, edits: tuple[Any, ...], *, context: str) -> None:
    r"""逐字段证明 revision 只改了 widening 函数声明的 MCP2 upper。

    先把每个实际 edit 的 ``new_upper_rad`` 恢复为 ``old_upper_rad``，再比较完整
    ``HandCfg.to_dict()``。这样会同时覆盖 palm、mount、几何、惯量、joint properties、
    metadata 与所有未选中 finger，而不是只检查少数限位字段。
    """

    restored = deepcopy(revised)  # 复制 revised，避免验证过程改变即将导出的对象
    changes = {edit.joint_name: edit for edit in edits}  # joint name 在 HandCfg 中必须全局唯一
    for joint in restored.iter_joints():
        if joint.name in changes:
            if not isinstance(joint.limit, JointLimitCfg):
                raise AssertionError(f"{context} revision lost typed limit for {joint.name!r}")
            joint.limit.upper = changes[joint.name].old_upper_rad  # 恢复原 $q_{max}$ 做全快照等价检查
    if restored.to_dict() != original.to_dict():
        raise AssertionError(f"{context} changed undeclared physical fields")


def _revision_identity(
    *,
    parent_id: str,
    parent_sidecar: Path,
    upper_rad: float,
    edits: tuple[Any, ...],
) -> dict[str, Any]:
    r"""构造由真实 parent bytes、规则与数值差分共同绑定的 revision identity。"""

    return {
        "parent_id": parent_id,
        "parent_sidecar_sha256": _sha(parent_sidecar),
        "rule": REVISION_RULE,
        "upper_rad": upper_rad,
        "changes": [asdict(edit) for edit in edits],
    }


def _revision_id(identity: dict[str, Any], *, parent_id: str, edits: tuple[Any, ...]) -> str:
    r"""从 revision identity 派生新 bundle ID；no-op parent 保留其物理 ID。"""

    if not edits:
        return parent_id  # registration parent 允许与原/既有 revision parent 同 ID 同内容
    payload = json.dumps(identity, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _load_registration_parent(
    plan: dict[str, Any],
    *,
    upper_rad: float,
) -> dict[str, Any]:
    r"""从 variant provenance 的真实 ``mother_path`` 物化 registration-only revised mother。

    该母体只服务新 variant set 的 ``source_topology_dir`` 合同，不代表 selected 数据，
    不加入 cohort member axis。解析仍走标准 ``HandBank`` explicit route；因此缺失
    hand bundle、geometry semantics、family 或 handedness 都会在任何输出目录建立前失败。
    """

    provenance = plan["record"].provenance
    raw_mother_path = str(provenance.mother_path).strip()
    if not raw_mother_path:
        raise FileNotFoundError(f"registration mother path is empty for lineage {provenance.mother_name!r}")
    mother_path = Path(raw_mother_path).expanduser().resolve(strict=False)
    if not (mother_path / "hand.urdf").is_file() or not (mother_path / "hand.yaml").is_file():
        raise FileNotFoundError(f"registration mother bundle is incomplete: {mother_path}")

    # HandBank 在显式 single-container 路径上验证 URDF、sidecar、mesh 闭合和 geometry semantics。
    selection = HandBank(
        HandBankCfg(
            source_mode="mixed",
            selection_mode="explicit",
            containers=(HandContainerCfg(path=mother_path, source_kind="generated"),),
            require_geometry_semantics=True,
        )
    ).resolve()
    if len(selection.assets) != 1:
        raise ValueError(f"registration mother resolver returned {len(selection.assets)} assets")
    container = selection.assets[0]
    if container.urdf_path.parent.resolve(strict=False) != mother_path:
        raise ValueError(f"registration mother resolver escaped declared path: {mother_path}")
    if "hand_cfg" not in container.sidecar:
        raise ValueError(f"registration mother sidecar lacks hand_cfg: {container.sidecar_path}")
    mother_hand = restore_hand_cfg_snapshot(container.sidecar["hand_cfg"])
    _require_allegro_right(mother_hand, context=f"registration mother {mother_path}")
    revised, edits = widen_two_joint_allegro_flexion(mother_hand, upper_rad=upper_rad)
    _assert_only_declared_limit_changes(mother_hand, revised, edits, context=f"registration mother {mother_path}")
    identity = _revision_identity(
        parent_id=container.asset_id,
        parent_sidecar=container.sidecar_path,
        upper_rad=upper_rad,
        edits=edits,
    )
    return {
        "mother_path": mother_path,
        "container": container,
        "hand": revised,
        "edits": edits,
        "identity": identity,
        "new_id": _revision_id(identity, parent_id=container.asset_id, edits=edits),
    }


def _revision_extra(
    source_sidecar: dict[str, Any],
    *,
    new_id: str,
    identity: dict[str, Any],
    parent_bundle: Path,
    parent_urdf: Path,
    parent_provenance: dict[str, Any],
    parent_validation: Any,
    registration_only: bool = False,
    registration_lineage: str | None = None,
    registration_source_coordinates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    r"""复制原 sidecar 的研究 provenance，并写入新的 revision 边界证据。

    旧 ``validation``、``geometry_semantics`` 与 ``hand_cfg`` 不直接继承，因为限位是
    canonical identity 的输入；exporter 会从 revised ``HandCfg`` 重建它们。其它字段
    仍完整保留，保证 mutation/source annotations 不因发布而消失。
    """

    extra = deepcopy(source_sidecar)  # 保留原 mutation payload、实验标签及 source ancestry
    for key in ("id", "timestamp", "hand_cfg", "geometry_semantics", "validation"):
        extra.pop(key, None)  # 新物理设计必须重新生成快照和后续安全证书
    extra["id"] = new_id
    revision = identity | {
        "parent_bundle": str(parent_bundle),
        "parent_urdf_sha256": _sha(parent_urdf),
        "parent_provenance": parent_provenance,
        "parent_validation": parent_validation,
        "runtime_validation": "pending-new-canonical-and-strict-pregrasp-evidence",
    }
    if registration_only:
        revision |= {
            "registration_only": True,
            "selected_for_cohort": False,
            "registration_lineage": registration_lineage,
            "registration_source_coordinates": registration_source_coordinates or [],
        }
    extra["asset_revision"] = revision
    return extra


def _export_revision(
    exporter: HandExporter,
    *,
    hand: Any,
    destination: Path,
    sample_id: str,
    metadata: dict[str, Any],
) -> HandGenerationResult:
    r"""通过正式 ``HandExporter`` 写 bundle，并立即回读完整 HandCfg 快照。"""

    result = HandGenerationResult(hand_cfg=hand, metadata=metadata)
    exported = exporter.export(result, destination, sample_id=sample_id, nest_sample_dir=False)
    if exported.errors or result.urdf_path is None or result.sidecar_path is None:
        raise RuntimeError(f"revision export failed: {exported}")
    sidecar = yaml.safe_load(result.sidecar_path.read_text(encoding="utf-8"))
    if restore_hand_cfg_snapshot(sidecar["hand_cfg"]).to_dict() != hand.to_dict():
        raise AssertionError("exporter changed the revised HandCfg snapshot")
    return result


def main() -> None:
    """复制物理设计、导出受影响 bundle，并按父成员顺序冻结新 source lock。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lock", type=Path, required=True)  # 原始已冻结 source/canonical lock
    parser.add_argument("--output-dir", type=Path, required=True)  # 必须是尚未存在的独占 case 子目录
    parser.add_argument("--cohort-id", required=True)  # 新物理设计使用独立实验集合名称
    parser.add_argument("--upper-rad", type=float, default=2.23)  # LEAP 对应屈曲上限锚点，rad
    parser.add_argument("--expected-assets", type=int, default=128)  # 本次完整分母，不通过筛除失败缩小
    parser.add_argument("--expected-modified-assets", type=int, default=None)
    parser.add_argument("--expected-modified-joints", type=int, default=None)
    parser.add_argument("--expected-registration-parents", type=int, default=None)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()  # 绝不对已有目录做重跑覆盖
    if output.exists():
        raise FileExistsError(output)
    parent = load_hand_asset_cohort(args.source_lock)  # 真正 source resolver，拒绝手工猜目录
    if len(parent.members) != args.expected_assets:
        raise ValueError("parent cohort does not preserve the declared complete asset count")

    # 先在内存完成所有选择与物理差分检查，任何语义歧义都在生成目录前拒绝。
    plans = []  # 保持父 cohort 行序，后续 variant ID 排序不能改变这个评价轴
    for row, record in enumerate(parent.partition.records):
        hand = restore_hand_cfg_snapshot(record.container.sidecar["hand_cfg"])
        _require_allegro_right(hand, context="this cohort producer")
        revised, edits = widen_two_joint_allegro_flexion(hand, upper_rad=args.upper_rad)
        _assert_only_declared_limit_changes(hand, revised, edits, context=f"selected asset row {row}")
        source = record.container.urdf_path.parent  # 路径来自 typed container
        identity = _revision_identity(
            parent_id=record.container.asset_id,
            parent_sidecar=source / "hand.yaml",
            upper_rad=args.upper_rad,
            edits=edits,
        )
        new_id = _revision_id(identity, parent_id=record.container.asset_id, edits=edits)
        plans.append(
            {
                "row": row,
                "record": record,
                "hand": revised,
                "edits": edits,
                "new_id": new_id,
                "identity": identity,
                "source": source,
            }
        )
    modified = [plan for plan in plans if plan["edits"]]  # 不修改的 64 项继续共享原始 bundle
    joint_count = sum(len(plan["edits"]) for plan in modified)
    if args.expected_modified_assets is not None and len(modified) != args.expected_modified_assets:
        raise ValueError("modified asset count differs from the preregistered scope")
    if args.expected_modified_joints is not None and joint_count != args.expected_modified_joints:
        raise ValueError("modified joint count differs from the preregistered scope")
    if not modified:
        raise ValueError("no physical change; do not publish an empty revision")

    # 以 mother_name + mother_path 组成 lineage key；同名而指向不同真实母体时必须 fail closed，
    # 否则两个物理 ancestry 会在新 generated 目录中被错误合并。
    plans_by_lineage = defaultdict(list)  # 完整父轴分组，用于检查 selected mother 数量与 source lineage
    for plan in plans:
        provenance = plan["record"].provenance
        plans_by_lineage[(provenance.mother_name, provenance.mother_path)].append(plan)
    lineage_paths: dict[str, str] = {}
    for (mother_name, mother_path), lineage_plans in plans_by_lineage.items():
        prior_path = lineage_paths.setdefault(mother_name, mother_path)
        if prior_path != mother_path:
            raise ValueError(f"mother name maps to multiple provenance paths: {mother_name!r}")
        selected_mothers = [
            plan for plan in lineage_plans if plan["record"].provenance.asset_role == "mother"
        ]
        if len(selected_mothers) > 1:
            raise ValueError(f"lineage {mother_name!r} contains multiple selected mothers")

    # 只为真正发生 limit 变化的成员建立输出 lineage；registration parent 在此阶段预解析，
    # 任何 prototype 缺失或 family/handedness 不符都会在 output.mkdir 前拒绝并保持路径空白。
    modified_by_lineage = defaultdict(list)  # 新 dataset 只列 selected revised variants/mothers
    for plan in modified:
        provenance = plan["record"].provenance
        modified_by_lineage[(provenance.mother_name, provenance.mother_path)].append(plan)
    lineage_info: dict[tuple[str, str], dict[str, Any]] = {}
    registration_plans: dict[tuple[str, str], dict[str, Any]] = {}
    for lineage_key, group in modified_by_lineage.items():
        mother_name, _mother_path = lineage_key
        edited_mothers = [
            plan for plan in group if plan["record"].provenance.asset_role == "mother"
        ]
        if len(edited_mothers) > 1:
            raise ValueError(f"lineage {mother_name!r} contains multiple revised selected mothers")
        selected_mother_plan = edited_mothers[0] if edited_mothers else None
        include_mother = selected_mother_plan is not None
        lineage_info[lineage_key] = {
            "selected_mother_plan": selected_mother_plan,
            "include_mother": include_mother,
        }
        if not include_mother:
            registration_plans[lineage_key] = _load_registration_parent(group[0], upper_rad=args.upper_rad)
    registration_count = len(registration_plans)
    if (
        args.expected_registration_parents is not None
        and registration_count != args.expected_registration_parents
    ):
        raise ValueError("registration parent count differs from the preregistered scope")
    if REVISION_SOURCE_ALIAS in parent.sources:
        raise ValueError(f"source alias {REVISION_SOURCE_ALIAS!r} is already in use")

    output.mkdir(parents=True)  # 该目录从此属于本 invocation；失败后保留可审计的 partial evidence
    run_root = output / "generated"  # 新生成批次，与原 generated run 完全独立
    run_root.mkdir()
    _json(
        output / "invocation.json",
        {
            "status": "exporting",
            "parent_lock": str(parent.lock_path),
            "parent_lock_sha256": parent.lock_sha256,
            "command_arguments": vars(args) | {"source_lock": str(args.source_lock), "output_dir": str(output)},
            "selected_modified_assets": len(modified),
            "registration_parent_count": registration_count,
            "materialized_asset_count": len(modified) + registration_count,
            "created_utc": datetime.now(UTC).isoformat(),
            "producer_sha256": _sha(Path(__file__)),
            "rule_sha256": _sha(Path(__file__).parents[1] / "asset_revisions.py"),
        },
    )
    exporter = HandExporter(HandExporterCfg())  # 复用标准 URDF/geometry sidecar exporter
    lineages = {}  # 新 schema-2 manifest 中只列实际修改的 lineages
    mapping_rows = []  # 用户可按此找回每一个原资产及其新版对应项
    registration_parents: list[dict[str, Any]] = []  # 技术父体单列，绝不混入 selected member 统计
    for (mother, _mother_path), group in modified_by_lineage.items():
        mother_dir = run_root / "single_palm_allegro" / mother
        variant_set = REVISION_VARIANT_SET  # 确定性 property revision，不宣称重新抽样几何
        variant_count = 0

        # variant-only cohort 仍必须让 dataset resolver 看到一个真实且已改造的 source topology。
        # 该 bundle 是 registration-only parent：它不占 selected 轴、不计训练/评测分母，也不写入
        # 新 cohort 的 member coordinates；它仅满足 variant-set summary 的 source_topology_dir 合同。
        info = lineage_info[(mother, _mother_path)]
        if not info["include_mother"]:
            registration = registration_plans[(mother, _mother_path)]
            # source_alias/source_row 必须来自 cohort member，而不是 provenance.run_alias；这里保留
            # 原坐标的显式副本，供审计者证明 registration parent 没有被偷偷当作 selected 数据。
            registration_coordinates = [
                {
                    "source_alias": parent.members[plan["row"]].source_alias,
                    "source_row": parent.members[plan["row"]].source_row,
                }
                for plan in group
            ]
            registration_metadata = _revision_extra(
                registration["container"].sidecar,
                new_id=registration["new_id"],
                identity=registration["identity"],
                parent_bundle=registration["mother_path"],
                parent_urdf=registration["container"].urdf_path,
                parent_provenance={
                    "mother_name": mother,
                    "mother_path": str(registration["mother_path"]),
                    "asset_role": "registration_parent",
                },
                parent_validation=registration["container"].sidecar.get("validation"),
                registration_only=True,
                registration_lineage=mother,
                registration_source_coordinates=registration_coordinates,
            )
            exported_parent = _export_revision(
                exporter,
                hand=registration["hand"],
                destination=mother_dir,
                sample_id=registration["new_id"],
                metadata=registration_metadata,
            )
            registration_parents.append(
                {
                    "lineage": mother,
                    "asset_id": registration["new_id"],
                    "parent_asset_id": registration["container"].asset_id,
                    "parent_bundle": str(registration["mother_path"]),
                    "new_bundle": str(mother_dir),
                    "parent_sidecar_sha256": registration["identity"]["parent_sidecar_sha256"],
                    "parent_urdf_sha256": _sha(registration["container"].urdf_path),
                    "new_urdf_sha256": _sha(exported_parent.urdf_path),
                    "new_sidecar_sha256": _sha(exported_parent.sidecar_path),
                    "identity": registration["identity"],
                    "changes": registration["identity"]["changes"],
                    "registration_only": True,
                    "selected_for_cohort": False,
                    "source_coordinates": registration_coordinates,
                }
            )

        for plan in group:
            record = plan["record"]  # 原始 provenance 保存在下面的显式父记录中
            is_mother = plan is info["selected_mother_plan"]
            destination = mother_dir if is_mother else mother_dir / variant_set / plan["new_id"]
            extra = _revision_extra(
                record.container.sidecar,
                new_id=plan["new_id"],
                identity=plan["identity"],
                parent_bundle=plan["source"],
                parent_urdf=record.container.urdf_path,
                parent_provenance=asdict(record.provenance),
                parent_validation=record.container.sidecar.get("validation"),
            )
            result = _export_revision(
                exporter,
                hand=plan["hand"],
                destination=destination,
                sample_id=plan["new_id"],
                metadata=extra,
            )
            source_member = parent.members[plan["row"]]
            mapping_rows.append(
                {
                    "row": plan["row"],
                    "source_alias": source_member.source_alias,
                    "source_row": source_member.source_row,
                    "parent_asset_id": record.container.asset_id,
                    "asset_id": plan["new_id"],
                    "new_bundle": str(destination),
                    "changes": plan["identity"]["changes"],
                    "parent_provenance": asdict(record.provenance),
                    "new_urdf_sha256": _sha(result.urdf_path),
                    "new_sidecar_sha256": _sha(result.sidecar_path),
                }
            )
            variant_count += int(not is_mother)  # dataset resolver 要核对 direct-child 数量
        lineages[mother] = {
            "include_mother": bool(info["include_mother"]),
            "variant_sets": [variant_set] if variant_count else [],
        }
        if variant_count:
            (mother_dir / variant_set / "summary.yaml").write_text(
                yaml.safe_dump(
                    {
                        "run": {"mode": "mutate", "producer": "deterministic-joint-limit-revision"},
                        "config": {"source_topology_dir": str(mother_dir), "upper_rad": args.upper_rad},
                        "stats": {
                            "succeeded": variant_count,
                            "selected_modified_assets": variant_count,
                            "registration_parent_count": int(not info["include_mother"]),
                        },
                        "semantics": "Each selected original variant is independently revised; original random mutation payload is retained as ancestry.",
                    },
                    sort_keys=False,
                )
            )
    (run_root / "summary.yaml").write_text(
        yaml.safe_dump(
            {
                "run": {"mode": "made", "producer": "deterministic-joint-limit-revision"},
                "parent_cohort": str(parent.lock_path),
                "stats": {
                    "mother_count": len(modified_by_lineage),
                    "asset_count": len(modified),
                    "selected_modified_asset_count": len(modified),
                    "registration_parent_count": registration_count,
                    "materialized_asset_count": len(modified) + registration_count,
                },
            },
            sort_keys=False,
        )
    )
    manifest = output / "revised-assets.yaml"  # 只包含新 bundle，原成员继续引用旧 source manifest
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": "2.0.0",
                "default_run_dir": str(run_root),
                "train": {"runs": {"revision": {"groups": {"single_palm_allegro": lineages}}}},
                "evaluation": {},  # schema-2 要求显式 evaluation 声明；本资产修订不另选评测集
            },
            sort_keys=False,
        )
    )
    partition = HandAssetDataset.from_yaml(manifest).resolve_train(require_geometry_semantics=True)
    by_id = {record.container.asset_id: row for row, record in enumerate(partition.records)}
    if len(by_id) != len(modified):
        raise AssertionError("revised dataset lost or duplicated an asset")
    source_manifests = {alias: source.manifest_path for alias, source in parent.sources.items()}
    source_manifests[REVISION_SOURCE_ALIAS] = manifest
    coordinates = [
        (REVISION_SOURCE_ALIAS, by_id[plan["new_id"]])
        if plan["edits"]
        else (parent.members[plan["row"]].source_alias, parent.members[plan["row"]].source_row)
        for plan in plans
    ]
    source_mapping = [
        {
            "cohort_row": plan["row"],
            "source_alias": parent.members[plan["row"]].source_alias,
            "source_row": parent.members[plan["row"]].source_row,
            "parent_asset_id": parent.members[plan["row"]].asset_id,
            "asset_id": plan["new_id"] if plan["edits"] else parent.members[plan["row"]].asset_id,
            "asset_role": parent.members[plan["row"]].provenance.asset_role,
            "mother_name": parent.members[plan["row"]].provenance.mother_name,
            "mother_path": parent.members[plan["row"]].provenance.mother_path,
            "revised": bool(plan["edits"]),
        }
        for plan in plans
    ]
    lock = write_hand_asset_cohort_lock(
        output / "training.lock.yaml",
        cohort_id=args.cohort_id,
        source_manifests=source_manifests,
        member_coordinates=coordinates,
        selection={
            "algorithm": "same-ordered-cohort-local-joint-limit-revision-v1",
            "parent_lock": str(parent.lock_path),
            "parent_lock_sha256": parent.lock_sha256,
            "upper_rad": args.upper_rad,
            "modified_assets": len(modified),
            "modified_joints": joint_count,
            "selected_modified_assets": len(modified),
            "registration_parent_count": registration_count,
            "materialized_assets": len(modified) + registration_count,
            "source_mapping": source_mapping,
        },
    )
    # 旧文件应始终保持原 bytes；完成生成不表示新版物理或学习验收已经通过。
    if _sha(parent.lock_path) != parent.lock_sha256:
        raise AssertionError("parent lock changed during revision export")
    report = {
        "status": "source-bundles-published-canonical-and-physics-validation-pending",
        "parent_lock": str(parent.lock_path),
        "parent_lock_sha256": parent.lock_sha256,
        "source_lock": str(lock),
        "source_lock_sha256": _sha(lock),
        "assets": len(plans),
        "selected_assets": len(plans),
        "modified_assets": len(modified),
        "selected_modified_assets": len(modified),
        "unchanged_assets": len(plans) - len(modified),
        "modified_joints": joint_count,
        "registration_parent_count": registration_count,
        "materialized_assets": len(modified) + registration_count,
        "registration_parents": registration_parents,
        "source_mapping": source_mapping,
        "mapping": sorted(mapping_rows, key=lambda p: p["row"]),
        "unchanged_rows": [plan["row"] for plan in plans if not plan["edits"]],
    }
    _json(output / "revision.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"mapping", "unchanged_rows"}}, indent=2))


if __name__ == "__main__":
    main()  # CPU 资产生产；canonical lowering / physics / policy 保持各自入口所有权
