r"""从既有训练母体系中，冻结开发验证与最终验收的未训练variant候选。

本入口只消费已发布train源，不改上游YAML partition，也不读取策略成绩。
每条lineage排除已训练成员，再以独立seed作无放回随机划分；各lineage使用相同名额。
开发集允许用于方法/检查点选择，最终验收集只用于选定模型的终验。

输出一个source-level候选并集，开发成员在前、终验成员在后，role索引进入selection。
并集只为一次canonical lowering服务，不作为训练或策略评价集合。
完成lowering并核验旧训练/dev/acceptance物理hash隔离后，才用现有subset writer发布两个最终集合。
剩余reserve顺序也记录，物理重复若需替换，只按此顺序处理，不能依据策略表现换掉困难样本。

源content hash隔离不等于physical geometry hash隔离；本脚本明确保留后者为待验证状态。
真正几何解析与source lock发布复用生产API；已有输出拒绝覆盖，文件发布采用同文件系统硬链接。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from anymani.assets.bank.cohort import (
    HandAssetCohortMember,
    load_hand_asset_cohort,
    write_hand_asset_cohort_lock,
)
from anymani.assets.bank.dataset import HandAssetDataset, ResolvedHandAssetPartition
from anymani.assets.bank.prepared_train import resolve_prepared_train


def select_policy_unseen_variant_splits(
    parent_members: Sequence[HandAssetCohortMember],
    train_partitions: Mapping[str, ResolvedHandAssetPartition],
    *,
    seed: int = 20260907,
    development_per_lineage: int = 2,
    acceptance_per_lineage: int = 4,
    excluded_mother_names: Sequence[str] = (),
) -> tuple[dict[str, tuple[tuple[str, int], ...]], dict[str, Any]]:
    r"""在每个已训练LEAP/right母体系内，无放回划分未训练variants。

    Args:
        parent_members: 已由cohort loader验证的训练成员；只按来源坐标识别，不按局部cohort index识别。
        train_partitions: alias到已验证train partition的映射；不同alias的数字row可以相同。
        seed: 数据划分的独立整数seed；不改变Python全局RNG或训练seed。
        development_per_lineage: 每个母体系分配的开发验证variant数，默认2。
        acceptance_per_lineage: 每个母体系分配的最终验收variant数，默认4。
        excluded_mother_names: 不允许出现在父训练集合中的既有研究留出母体。

    Returns:
        三类有序source坐标与JSON-safe选择证据。reserve也是无放回排列，不是补齐后的重复样本。
        此时仅建立source-level隔离，物理等价性留给canonical lowering后的独立验证。
    """
    if not parent_members or not train_partitions:
        raise ValueError("parent members and train partitions must be non-empty")
    if development_per_lineage < 1 or acceptance_per_lineage < 1:
        raise ValueError("each unseen role requires a positive count per lineage")
    if any(partition.name != "train" for partition in train_partitions.values()):
        raise ValueError("unseen variants must be selected only from published train partitions")

    # 来源坐标唯一性先于抽样；母体系的分层键同时包含source alias和真实mother path。
    trained = {(member.source_alias, member.source_row) for member in parent_members}
    if len(trained) != len(parent_members):
        raise ValueError("duplicate training coordinate")
    lineages: dict[tuple[str, str], str] = {}  # (alias, mother_path) -> 人类可读的mother_name
    lineage_sources: dict[str, str] = {}  # 同一母体若被不同alias重复引用，应先解决来源关系
    known_hashes = set()  # 已训练源内容与已纳入候选池的源内容，均不可再次成为独立样本
    for member in parent_members:
        partition = train_partitions.get(member.source_alias)
        if partition is None or not 0 <= member.source_row < len(partition.records):
            raise ValueError(f"invalid training coordinate: {member.source_key}")
        record = partition.records[member.source_row]
        if (
            record.container.asset_id != member.asset_id
            or record.content_hash != member.content_hash
            or record.provenance != member.provenance
        ):
            raise ValueError(f"parent member disagrees with its published source: {member.source_key}")
        provenance = record.provenance
        if provenance.mother_name in excluded_mother_names:
            raise ValueError(f"excluded mother in parent: {provenance.mother_name}")
        if (
            provenance.partition != "train"
            or provenance.group_name != "single_palm_leap"
            or not provenance.mother_name.startswith("right_")
        ):
            raise ValueError("parent must contain published pure LEAP/right train lineages")
        if not record.content_hash:
            raise ValueError("source content identity is required for unseen selection")
        prior_alias = lineage_sources.setdefault(provenance.mother_path, member.source_alias)
        if prior_alias != member.source_alias:
            raise ValueError("one training lineage is represented by multiple source aliases")
        lineages[(member.source_alias, provenance.mother_path)] = provenance.mother_name
        known_hashes.add(record.content_hash)

    # 候选池只含同一lineage的variant。源级重复先排除，之后仍需检验canonical物理重复。
    pools: dict[tuple[str, str], list[tuple[str, int]]] = {key: [] for key in sorted(lineages)}
    excluded_duplicates = []  # 记录因source身份重复而排除的坐标，不把它们称作策略失败
    for alias in sorted(train_partitions):
        for row, record in enumerate(train_partitions[alias].records):
            provenance = record.provenance
            key = (alias, provenance.mother_path)
            if key not in pools or provenance.asset_role != "variant" or (alias, row) in trained:
                continue
            if provenance.partition != "train" or provenance.group_name != "single_palm_leap":
                raise ValueError("candidate provenance disagrees with its training lineage")
            if not record.content_hash:
                raise ValueError("candidate source content identity is missing")
            if record.content_hash in known_hashes:
                excluded_duplicates.append(f"{alias}#{row}")
                continue
            known_hashes.add(record.content_hash)
            pools[key].append((alias, row))

    # 局部RNG只定义数据划分。排序使父成员的排列变化不影响候选池的抽样顺序。
    rng = random.Random(seed)
    selected: dict[str, list[tuple[str, int]]] = {role: [] for role in ("development", "acceptance", "reserve")}
    strata = []  # 每条母体系的完整选择/备用顺序，支持后续物理重复的确定性替换
    required = development_per_lineage + acceptance_per_lineage
    for key, candidates in pools.items():
        if len(candidates) < required:
            raise ValueError(f"not enough unseen variants for {lineages[key]}: {len(candidates)} < {required}")
        ordered = rng.sample(candidates, k=len(candidates))  # 全部剩余候选的无放回排列
        roles = {
            "development": ordered[:development_per_lineage],
            "acceptance": ordered[development_per_lineage:required],
            "reserve": ordered[required:],
        }
        for role, coordinates in roles.items():
            selected[role].extend(coordinates)
        strata.append(
            {
                "source_alias": key[0],
                "mother_path": key[1],
                "mother_name": lineages[key],
                "available_unseen_variants": len(candidates),
                **{role: [f"{alias}#{row}" for alias, row in coordinates] for role, coordinates in roles.items()},
            }
        )
    metadata = {
        "algorithm": "within-lineage-uniform-without-replacement-v1",
        "selection_seed": seed,
        "topology_count": len(lineages),
        "parent_member_count": len(parent_members),
        "development_per_lineage": development_per_lineage,
        "acceptance_per_lineage": acceptance_per_lineage,
        "excluded_mother_names": sorted(excluded_mother_names),
        "excluded_source_duplicate_keys": excluded_duplicates,
        "physical_isolation": "pending-canonical-lowering",
        "strata": strata,
    }
    return {role: tuple(coordinates) for role, coordinates in selected.items()}, metadata


def main() -> None:
    r"""验证父集合及完整source，再发布只供物理准备的候选并集。

    生产writer会重新核验source内容；最后硬链接只允许新建目标文件。
    role索引记录在并集轴上，后续可用write_hand_asset_cohort_subset发布最终dev/acceptance。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort-id", required=True)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--development-per-lineage", type=int, default=2)
    parser.add_argument("--acceptance-per-lineage", type=int, default=4)
    parser.add_argument("--exclude-mother", action="append", default=["right_t3_i3_m4", "right_t4_m4_r3"])
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)  # 冻结集合不由后续重新运行原地覆盖

    # 读取经过schema/source验证的父集合，再获取同一source的完整train库存。
    parent = load_hand_asset_cohort(args.parent, require_geometry_semantics=True)
    partitions = {}
    cache_hits = {}
    for alias, source in parent.sources.items():
        dataset = HandAssetDataset.from_yaml(source.manifest_path)
        if dataset.source_sha256 != source.manifest_sha256:
            raise ValueError(f"parent source manifest changed: {alias}")
        partitions[alias], cache_hits[alias] = resolve_prepared_train(dataset, require_geometry_semantics=True)
    splits, selection = select_policy_unseen_variant_splits(
        parent.members,
        partitions,
        seed=args.seed,
        development_per_lineage=args.development_per_lineage,
        acceptance_per_lineage=args.acceptance_per_lineage,
        excluded_mother_names=args.exclude_mother,
    )
    coordinates = splits["development"] + splits["acceptance"]  # 并集只lower一次，reserve暂不发布
    development_count = len(splits["development"])
    selection.update(
        {
            "purpose": "policy-unseen-variant-candidates-for-physical-preparation-only",
            "parent_cohort_id": parent.cohort_id,
            "parent_lock_path": str(parent.lock_path),
            "parent_lock_sha256": parent.lock_sha256,
            "selector_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "candidate_member_order": "development-first-then-acceptance",
            "role_indices": {
                "development": list(range(development_count)),
                "acceptance": list(range(development_count, len(coordinates))),
            },
        }
    )

    # staging与目标位于同一文件系统。os.link的原子新建语义也保护运行期间并发出现的同名输出。
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".unseen-variants-", dir=output.parent) as temporary:
        staged = write_hand_asset_cohort_lock(
            Path(temporary) / "candidate.lock.yaml",
            cohort_id=args.cohort_id,
            source_manifests={alias: source.manifest_path for alias, source in parent.sources.items()},
            member_coordinates=coordinates,
            selection=selection,
            require_geometry_semantics=True,
        )
        os.link(staged, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "topologies": selection["topology_count"],
                "role_counts": {role: len(values) for role, values in splits.items()},
                "source_cache_hits": cache_hits,
                "physical_isolation": selection["physical_isolation"],
            }
        )
    )


if __name__ == "__main__":
    main()
