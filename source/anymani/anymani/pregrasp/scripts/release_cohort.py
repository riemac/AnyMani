r"""依据已认证初态按固定备用顺位发布训练、开发与终验集合。

角色与备用顺序来自源层预先记录的strata。准入只看完整strict条目，顺序固定为训练、开发、终验；
每条谱系原成员可用时保持，缺失变体取首个可用备用，母体缺失则报告该拓扑尚未就绪。输出复用已验证
canonical父证书，并为每个角色发布独立catalog。检查模式只输出缺口，不产生部分正式发布。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence, Set
from pathlib import Path
from typing import Any

from anymani.assets.bank.cohort import (
    ResolvedHandAssetCohort,
    load_hand_asset_cohort,
    write_hand_asset_cohort_union,
)
from anymani.pregrasp.good_catalog import GoodPregraspCatalog, GoodPregraspKey
from anymani.pregrasp.strict_gate import MVP80_STRICT_GOOD_PREGRASP_GATE


def select_admitted_roles(
    training_by_lineage: Mapping[tuple[str, str], Sequence[str]],
    strata: Sequence[Mapping[str, Any]],
    available_keys: Set[str],
) -> dict[str, Any]:
    r"""只用准入状态完成同谱系补位，保留声明角色的数量和原位置。

    training_by_lineage以(source_alias,mother_path)为键，每组首项为母体。strata包含各谱系预定的
    development/acceptance/reserve来源键顺序。available_keys只表示已恢复且严格通过的完整Top-8。
    返回role_keys、replacements、unavailable及ready；未就绪时保留原缺失位置，发布者仅接受ready=True。
    """

    groups = [(str(row["source_alias"]), str(row["mother_path"])) for row in strata]
    if not strata or len(set(groups)) != len(groups):
        raise ValueError("admission strata must contain unique non-empty lineages")
    if training_by_lineage and set(training_by_lineage) != set(groups):
        raise ValueError("training and role strata must describe the same lineages")
    declared = [key for group in training_by_lineage.values() for key in group]
    declared += [key for row in strata for role in ("development", "acceptance", "reserve") for key in row[role]]
    if len(set(declared)) != len(declared):
        raise ValueError("admission rejects duplicate source keys across declared roles")

    # 各谱系的备用池互不共享；组内顺序固定，避免根据策略成绩或临时偏好决定归属。
    selected: dict[str, list[str]] = {role: [] for role in ("training", "development", "acceptance")}
    replacements: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    remaining: list[str] = []
    used: set[str] = set()
    for row, group in zip(strata, groups, strict=True):
        reserve = tuple(str(key) for key in row["reserve"])
        cursor = 0
        originals = {
            "training": tuple(training_by_lineage.get(group, ())),
            "development": tuple(row["development"]),
            "acceptance": tuple(row["acceptance"]),
        }
        for role, keys in originals.items():
            for slot, original in enumerate(keys):
                chosen = original
                if original not in available_keys:
                    if role == "training" and slot == 0:
                        unavailable.append(
                            {"role": role, "mother_path": group[1], "key": original, "reason": "mother-not-admitted"}
                        )
                    else:
                        while cursor < len(reserve) and (
                            reserve[cursor] not in available_keys or reserve[cursor] in used
                        ):
                            cursor += 1
                        if cursor == len(reserve):
                            unavailable.append(
                                {"role": role, "mother_path": group[1], "key": original, "reason": "reserve-exhausted"}
                            )
                        else:
                            chosen = reserve[cursor]
                            cursor += 1
                            replacements.append({"role": role, "mother_path": group[1], "old": original, "new": chosen})
                selected[role].append(chosen)  # 缺口仍占原声明位置，ready=False时不会发布
                used.add(chosen)
        remaining.extend(key for key in reserve[cursor:] if key not in used)
    return {
        "ready": not unavailable,
        "role_keys": selected,
        "replacements": replacements,
        "unavailable": unavailable,
        "reserve_remaining": remaining,
    }


def _inspection_keys(path: Path, parent: ResolvedHandAssetCohort) -> tuple[GoodPregraspKey, ...]:
    r"""恢复先前正向binding生成的exact keys，并与canonical成员逐项核对。"""

    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("parent_cohort_lock_sha256") != parent.lock_sha256:
        raise ValueError("inspection belongs to another canonical parent")
    keys = tuple(GoodPregraspKey.from_dict(item) for item in document["expected_keys"])
    if len(keys) != len(parent.members):
        raise ValueError("inspection keys must align with the complete parent axis")
    for key, member in zip(keys, parent.members, strict=True):
        if (
            key.asset_id != member.asset_id
            or key.source_content_hash != member.configuration_domain_hash
            or key.physical_geometry_hash != member.physical_geometry_hash
            or key.canonical_schema_digest != member.canonical_schema_digest
        ):
            raise ValueError(f"inspection/canonical identity mismatch for {member.source_key}")
    return keys


def main() -> int:
    r"""检查完整角色准入，显式--publish后才形成最终锁和独立catalog。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-cohort", type=Path, default=None)
    parser.add_argument("--training-inspection", type=Path, default=None)
    parser.add_argument("--role-cohort", type=Path, required=True)
    parser.add_argument("--role-inspection", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cohort-prefix", required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    if (args.training_cohort is None) != (args.training_inspection is None):
        parser.error("training cohort and inspection must be supplied together")
    output = args.output_dir.expanduser().resolve()
    if args.publish and output.exists():
        raise FileExistsError(output)

    # 父对象各加载一次；后续每个角色只复用canonical身份，不再次解析8192项训练源。
    roles = load_hand_asset_cohort(args.role_cohort, require_geometry_semantics=True)
    parents = {"roles": roles}
    inspections = {"roles": _inspection_keys(args.role_inspection, roles)}
    if args.training_cohort is not None:
        training = load_hand_asset_cohort(args.training_cohort, require_geometry_semantics=True)
        parents["training"] = training
        inspections["training"] = _inspection_keys(args.training_inspection, training)
    by_source = {}
    training_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for name, parent in parents.items():
        for index, (member, key) in enumerate(zip(parent.members, inspections[name], strict=True)):
            if member.source_key in by_source:
                raise ValueError("canonical role parents contain duplicate source keys")
            by_source[member.source_key] = (name, index, member, key)
            if name == "training":
                group = (member.source_alias, member.provenance.mother_path)
                if not training_groups[group] and member.provenance.asset_role != "mother":
                    raise ValueError("training lineage must begin with its mother")
                training_groups[group].append(member.source_key)
    if any(len(keys) != 4 for keys in training_groups.values()):
        raise ValueError("training admission requires mother plus three variants")

    # 源层预注册的角色只能指向同一谱系中的真实变体；旧保护物理身份继续保留在最终筛选边界。
    strata = roles.selection["strata"]
    for row in strata:
        for role in ("development", "acceptance", "reserve"):
            for source_key in row[role]:
                if source_key not in by_source:
                    raise ValueError("role pool must include all declared reserve candidates")
                member = by_source[source_key][2]
                if (
                    member.source_alias != row["source_alias"]
                    or member.provenance.mother_path != row["mother_path"]
                    or member.provenance.asset_role != "variant"
                ):
                    raise ValueError("declared role key disagrees with its lineage")
    physical = [value[2].physical_geometry_hash for value in by_source.values()]
    if len(set(physical)) != len(physical):
        raise ValueError("role parents contain duplicate canonical physical identities")
    protected = set(roles.selection.get("protected_canonical_physical_hashes", ()))
    if protected.intersection(member.physical_geometry_hash for member in roles.members):
        raise ValueError("role pool intersects protected physical identities")

    catalog = GoodPregraspCatalog(args.catalog)
    snapshot = {entry.key.digest: entry for entry in catalog.read_entries()}  # 单次index/payload快照
    available = set()
    admitted_entries = {}
    for source_key, (_name, _index, _member, key) in by_source.items():
        entry = snapshot.get(key.digest)
        if entry is not None:
            if entry.key != key:
                raise ValueError("catalog embedded key disagrees with binding inspection")
            MVP80_STRICT_GOOD_PREGRASP_GATE.validate_entry(entry)
            available.add(source_key)
            admitted_entries[source_key] = entry
    result = select_admitted_roles(training_groups, strata, available)
    result.update(
        {
            "catalog_root": str(catalog.root.resolve()),
            "available_member_count": len(available),
            "parent_cohorts": {
                name: {"path": str(parent.lock_path), "sha256": parent.lock_sha256} for name, parent in parents.items()
            },
            "published": {},
        }
    )
    if args.publish and result["ready"]:
        output.mkdir(parents=True, exist_ok=False)
        for role, source_keys in result["role_keys"].items():
            if not source_keys:
                continue
            source_path, canonical_path = output / f"{role}.lock.yaml", output / f"{role}.canonical.lock.yaml"
            target = GoodPregraspCatalog(output / "catalogs" / role)
            write_hand_asset_cohort_union(
                parents,
                source_path,
                canonical_path,
                cohort_id=f"{args.cohort_prefix}-{role}",
                member_coordinates=tuple((by_source[key][0], by_source[key][1]) for key in source_keys),
                selection={
                    "role": role,
                    "algorithm": "strict-admission-reserve-order-v1",
                    "catalog_root": str(target.root),
                    "replacements": [row for row in result["replacements"] if row["role"] == role],
                },
            )
            published = target.publish_many(tuple(admitted_entries[key] for key in source_keys))
            result["published"][role] = {
                "cohort_lock": str(canonical_path),
                "catalog_root": str(target.root),
                "member_count": len(source_keys),
                "entry_count": len(published),
            }
        # 完整发布标记最后写入；调用方只从已完成的release记录进入正式训练/评价。
        temporary = output / ".release.json.tmp"
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(output / "release.json")
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["ready"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
