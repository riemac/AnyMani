r"""纯家族未见变体的来源排除与角色数量合同，完全使用合成metadata。"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
from anymani.assets.bank.cohort import HandAssetCohortMember
from anymani.assets.bank.dataset import HandAssetProvenance, ResolvedHandAssetPartition, ResolvedHandAssetRecord
from anymani.assets.bank.hand_container import HandContainer


def _inputs(group: str) -> tuple[tuple[HandAssetCohortMember, ...], dict[str, ResolvedHandAssetPartition]]:
    r"""一条母体系含母体及15变体，其中0..3已列入训练；不构造物理证书。"""

    records = []  # 相同来源坐标语义用于LEAP和Allegro，家族由生产组显式给出
    for row in range(16):
        provenance = HandAssetProvenance(
            partition="train",
            run_alias="default",
            run_dir="/source",
            collection_kind="groups",
            group_name=group,
            mother_name="right_t4_i4_m4_r4",
            mother_path=f"/source/{group}/right_t4_i4_m4_r4",
            variant_set="" if row == 0 else "variants",
            asset_role="mother" if row == 0 else "variant",
        )
        records.append(
            ResolvedHandAssetRecord(
                container=cast(HandContainer, SimpleNamespace(asset_id=f"asset-{row}")),
                provenance=provenance,
                content_hash=hashlib.sha256(f"{group}-{row}".encode()).hexdigest(),
            )
        )
    members = tuple(
        HandAssetCohortMember(
            cohort_index=row,
            source_alias="ssl",
            source_row=row,
            asset_id=records[row].container.asset_id,
            content_hash=records[row].content_hash,
            provenance=records[row].provenance,
            mutation_descriptor={},
        )
        for row in range(4)
    )
    return members, {"ssl": ResolvedHandAssetPartition(name="train", records=tuple(records))}


def test_allegro_role_split_preserves_per_lineage_counts() -> None:
    r"""Allegro得到开发2、终验4、备用6；默认LEAP域不静默接受另一族。"""

    from anymani.assets.scripts.unseen_variants import select_policy_unseen_variant_splits

    parent, sources = _inputs("single_palm_allegro")
    split, metadata = select_policy_unseen_variant_splits(parent, sources, production_group="single_palm_allegro")
    assert {role: len(rows) for role, rows in split.items()} == {"development": 2, "acceptance": 4, "reserve": 6}
    assert metadata["production_group"] == "single_palm_allegro"
    with pytest.raises(ValueError, match="parent must contain"):
        select_policy_unseen_variant_splits(parent, sources)


def test_protected_exposure_is_excluded_by_content_across_source_aliases() -> None:
    r"""旧评价成员即使使用另一alias，也不能重新成为新盲测；可只生成终验以复用旧开发集。"""

    from anymani.assets.scripts.unseen_variants import select_policy_unseen_variant_splits

    parent, sources = _inputs("single_palm_leap")
    protected = tuple(
        replace(
            parent[0],
            source_alias="historical",
            source_row=row + 100,
            asset_id=sources["ssl"].records[row].container.asset_id,
            content_hash=sources["ssl"].records[row].content_hash,
            provenance=sources["ssl"].records[row].provenance,
        )
        for row in range(4, 10)
    )  # 已训练4项以外，旧开发/终验另占6项
    split, metadata = select_policy_unseen_variant_splits(
        parent,
        sources,
        excluded_members=protected,
        development_per_lineage=0,
        acceptance_per_lineage=4,
    )
    assert split["development"] == ()
    assert len(split["acceptance"]) == 4 and len(split["reserve"]) == 2
    assert all(row >= 10 for values in split.values() for _alias, row in values)
    assert metadata["protected_member_count"] == 6
    assert metadata["physical_isolation"] == "pending-canonical-lowering"


def test_all_empty_role_request_is_rejected() -> None:
    r"""允许零开发配额不意味着允许没有任何实际评价角色的发布。"""

    from anymani.assets.scripts.unseen_variants import select_policy_unseen_variant_splits

    parent, sources = _inputs("single_palm_leap")
    with pytest.raises(ValueError, match="positive"):
        select_policy_unseen_variant_splits(parent, sources, development_per_lineage=0, acceptance_per_lineage=0)
