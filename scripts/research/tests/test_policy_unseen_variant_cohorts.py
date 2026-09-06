r"""同拓扑policy-unseen划分的纯CPU合同。

这些测试只验证来源坐标、随机划分与数据隔离，不把合成metadata当成几何或PhysX证书。
容器仅需asset_id；真实发布仍通过生产cohort writer解析完整来源。
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from anymani.assets.bank.cohort import HandAssetCohortMember
from anymani.assets.bank.dataset import HandAssetProvenance, ResolvedHandAssetPartition, ResolvedHandAssetRecord
from anymani.assets.bank.hand_container import HandContainer

from scripts.research import build_policy_unseen_variant_cohorts as split_module


def _inputs() -> tuple[tuple[HandAssetCohortMember, ...], dict[str, ResolvedHandAssetPartition]]:
    r"""两个source均使用row0..15，主动覆盖数字row碰撞但来源不同的情况。"""
    sources = {}  # 每个alias一条母体系，母体加15个variant
    parent = []  # 模拟已训练的mother+3，不依赖任何真实手文件
    for alias, mother in (("ppo", "right_t3_i3_m4"), ("ssl", "right_t4_i4_m4_r4")):
        records = []
        for row in range(16):
            provenance = HandAssetProvenance(
                partition="train",  # 只从发布train源划分，角色不改写上游partition
                run_alias="default",
                run_dir=f"/source/{alias}",
                collection_kind="groups",
                group_name="single_palm_leap",
                mother_name=mother,
                mother_path=f"/source/{alias}/{mother}",
                variant_set="" if row == 0 else "mutations",
                asset_role="mother" if row == 0 else "variant",
            )
            container = cast(HandContainer, SimpleNamespace(asset_id=f"{alias}-{row}"))
            digest = hashlib.sha256(f"{alias}-{row}".encode()).hexdigest()  # 源级identity，非物理hash
            records.append(ResolvedHandAssetRecord(container=container, provenance=provenance, content_hash=digest))
        sources[alias] = ResolvedHandAssetPartition(name="train", records=tuple(records))
        for row in (0, 1, 4, 12):
            record = records[row]
            parent.append(
                HandAssetCohortMember(
                    cohort_index=len(parent),
                    source_alias=alias,
                    source_row=row,
                    asset_id=record.container.asset_id,
                    content_hash=record.content_hash,
                    provenance=record.provenance,
                    mutation_descriptor={},  # 本测试不发布lock，不伪造mutation证书
                )
            )
    return tuple(parent), sources


def test_source_qualified_disjoint_splits_and_local_random_stream() -> None:
    r"""每条lineage开发2、终验4、reserve6，集合互不相交且不改变全局RNG。"""
    parent, sources = _inputs()
    state = random.getstate()  # 选择器不能改变调用者的随机过程
    first, metadata = split_module.select_policy_unseen_variant_splits(parent, sources, seed=20260907)
    second, _ = split_module.select_policy_unseen_variant_splits(tuple(reversed(parent)), sources, seed=20260907)
    assert first == second  # 调整父成员排列不改变分层无放回抽样结果
    assert random.getstate() == state
    assert {name: len(values) for name, values in first.items()} == {
        "development": 4,
        "acceptance": 8,
        "reserve": 12,
    }
    selected = set().union(*(set(values) for values in first.values()))
    trained = {(member.source_alias, member.source_row) for member in parent}
    assert len(selected) == 24 and not selected.intersection(trained)
    for name, per_lineage in (("development", 2), ("acceptance", 4), ("reserve", 6)):
        assert [sum(alias == source for alias, _ in first[name]) for source in sources] == [per_lineage, per_lineage]
    assert metadata["physical_isolation"] == "pending-canonical-lowering"
    assert metadata["topology_count"] == 2


def test_different_selection_seed_changes_members() -> None:
    r"""选择seed独立可控，不隐式复用训练种子或按policy表现选成员。"""
    parent, sources = _inputs()
    first, _ = split_module.select_policy_unseen_variant_splits(parent, sources, seed=20260907)
    second, _ = split_module.select_policy_unseen_variant_splits(parent, sources, seed=20260908)
    assert first["acceptance"] != second["acceptance"]


def test_known_training_content_duplicate_is_not_held_out() -> None:
    r"""新row若只有不同坐标而源内容已训练，不获得policy-unseen资格。"""
    parent, sources = _inputs()
    records = list(sources["ppo"].records)
    records[2] = replace(records[2], content_hash=records[0].content_hash)
    sources["ppo"] = replace(sources["ppo"], records=tuple(records))
    selected, metadata = split_module.select_policy_unseen_variant_splits(parent, sources, seed=20260907)
    assert all(("ppo", 2) not in values for values in selected.values())
    assert len(selected["reserve"]) == 11
    assert "ppo#2" in metadata["excluded_source_duplicate_keys"]


def test_insufficient_unseen_variants_fail_closed() -> None:
    r"""要求超过库存时拒绝，而不是减少分母或重复抽样补足数量。"""
    parent, sources = _inputs()
    with pytest.raises(ValueError, match="not enough unseen variants"):
        split_module.select_policy_unseen_variant_splits(parent, sources, seed=20260907, acceptance_per_lineage=11)


def test_duplicate_training_coordinates_are_rejected() -> None:
    r"""训练集合成员唯一性先于任何抽样，避免重复成员导致错误排除。"""
    parent, sources = _inputs()
    with pytest.raises(ValueError, match="duplicate training coordinate"):
        split_module.select_policy_unseen_variant_splits(parent + (parent[0],), sources, seed=20260907)


def test_excluded_lineage_cannot_enter_the_parent() -> None:
    r"""已留出的整条母体系不能靠新的集合名字重新获得训练资格。"""
    parent, sources = _inputs()
    with pytest.raises(ValueError, match="excluded mother"):
        split_module.select_policy_unseen_variant_splits(
            parent, sources, seed=20260907, excluded_mother_names=("right_t3_i3_m4",)
        )


def test_output_overwrite_is_rejected_before_source_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r"""生产writer能替换文件，外层发布入口必须先保护已经冻结的输出。"""
    output = tmp_path / "existing.lock.yaml"
    output.write_text("existing evidence\n")
    monkeypatch.setattr(
        "sys.argv", ["split", "--parent", "unused", "--output", str(output), "--cohort-id", "test-split"]
    )
    with pytest.raises(FileExistsError):
        split_module.main()
    assert output.read_text() == "existing evidence\n"
