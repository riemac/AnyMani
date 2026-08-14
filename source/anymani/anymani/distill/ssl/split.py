r"""Geometry SSL 的 physical-group leakage-safe 资产划分。

路径、sample ID 与 sidecar content hash 只能识别文件或完整声明，不能判断两个资产是否
实现同一物理映射。split 以 `physical_geometry_hash` 为不可拆分 group；limit-only variants
可以有不同 `configuration_domain_hash`，但不能分别进入 train/validation。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class GeometryAssetIdentityRecord:
    r"""split 所需的最小资产身份记录。"""

    asset_id: str  # bank 稳定资产 ID
    path: str  # manifest 可回放 bundle 路径
    content_hash: str  # sidecar 完整声明身份
    physical_geometry_hash: str  # 不可跨 split 的物理映射 group
    configuration_domain_hash: str  # joint names + limits 采样域


@dataclass(frozen=True)
class GroupedGeometryAssetSplit:
    r"""按完整 physical groups 冻结的 train/validation 结果。"""

    train: tuple[GeometryAssetIdentityRecord, ...]  # optimizer assets，含 mother group
    validation: tuple[GeometryAssetIdentityRecord, ...]  # held-out morphology groups
    split_seed: int  # group 排序的复现 seed
    requested_validation_asset_count: int  # 用户目标资产数
    actual_validation_asset_count: int  # group 完整性约束后的实际资产数

    def __post_init__(self) -> None:
        r"""拒绝 asset/content/physical identity 跨 split 泄漏。"""

        train_ids = {record.asset_id for record in self.train}
        validation_ids = {record.asset_id for record in self.validation}
        if train_ids & validation_ids:
            raise ValueError("asset IDs leak across grouped train/validation splits")
        train_content = {record.content_hash for record in self.train}
        validation_content = {record.content_hash for record in self.validation}
        if train_content & validation_content:
            raise ValueError("asset content hashes leak across grouped train/validation splits")
        train_physical = {record.physical_geometry_hash for record in self.train}
        validation_physical = {record.physical_geometry_hash for record in self.validation}
        if train_physical & validation_physical:
            raise ValueError("physical geometry hashes leak across grouped train/validation splits")


def split_geometry_asset_groups(
    records: tuple[GeometryAssetIdentityRecord, ...],
    *,
    mother_asset_id: str,
    validation_asset_count: int,
    split_seed: int,
) -> GroupedGeometryAssetSplit:
    r"""确定性选择最接近目标数量的完整 validation physical groups。

    mother 所在 physical group 固定进入训练。其余 group 先按
    `SHA256(seed || physical_hash)` 排序，再用子集动态规划枚举可达到的资产数；目标数量
    不可达时最小化绝对偏差，平手优先不超过目标，最后按 seeded group 顺序决定。

    Args:
        records (tuple[GeometryAssetIdentityRecord, ...]): family 中全部 mother/variants。
        mother_asset_id (str): 固定进入训练的 mother ID。
        validation_asset_count (int): 期望 held-out asset 数；group 约束可使实际数变化。
        split_seed (int): physical group 确定性排序 seed。

    Returns:
        GroupedGeometryAssetSplit: 完整 train/validation records 与实际数量。
    """

    if not records:
        raise ValueError("grouped geometry split requires at least one asset")
    if validation_asset_count < 0:
        raise ValueError("validation_asset_count must be non-negative")
    if len({record.asset_id for record in records}) != len(records):
        raise ValueError("grouped geometry split asset IDs must be unique")
    if mother_asset_id not in {record.asset_id for record in records}:
        raise ValueError(f"mother asset ID is absent from family records: {mother_asset_id!r}")

    groups: dict[str, list[GeometryAssetIdentityRecord]] = {}
    for record in records:
        groups.setdefault(record.physical_geometry_hash, []).append(record)
    mother_record = next(record for record in records if record.asset_id == mother_asset_id)
    mother_group_hash = mother_record.physical_geometry_hash
    candidate_hashes = sorted(
        (group_hash for group_hash in groups if group_hash != mother_group_hash),
        key=lambda group_hash: _seeded_group_key(split_seed, group_hash),
    )

    # `reachable[count]` 保存按 seeded 顺序选择的 group index tuple；相同 count 的首个
    # realization 即稳定 tie-break，不依赖 dict/path 枚举顺序。
    reachable: dict[int, tuple[int, ...]] = {0: ()}
    for group_index, group_hash in enumerate(candidate_hashes):
        group_size = len(groups[group_hash])
        additions: dict[int, tuple[int, ...]] = {}
        for count, selected in tuple(reachable.items()):
            next_count = count + group_size
            if next_count not in reachable and next_count not in additions:
                additions[next_count] = selected + (group_index,)  # 同 count 保留 seeded 顺序下首个 realization
        reachable.update(additions)
    target = min(validation_asset_count, sum(len(groups[group_hash]) for group_hash in candidate_hashes))
    selected_count = min(
        reachable,
        key=lambda count: (
            abs(count - target),
            count > target,
            reachable[count],
        ),
    )
    selected_indices = set(reachable[selected_count])
    validation_hashes = {
        group_hash for group_index, group_hash in enumerate(candidate_hashes) if group_index in selected_indices
    }
    train = tuple(record for record in records if record.physical_geometry_hash not in validation_hashes)
    validation = tuple(record for record in records if record.physical_geometry_hash in validation_hashes)
    return GroupedGeometryAssetSplit(
        train=train,
        validation=validation,
        split_seed=int(split_seed),
        requested_validation_asset_count=int(validation_asset_count),
        actual_validation_asset_count=len(validation),
    )


def _seeded_group_key(split_seed: int, physical_geometry_hash: str) -> bytes:
    r"""把 split seed 与物理 group hash 混成平台无关的排序键。"""

    return hashlib.sha256(f"{int(split_seed)}:{physical_geometry_hash}".encode("ascii")).digest()


__all__ = [
    "GeometryAssetIdentityRecord",
    "GroupedGeometryAssetSplit",
    "split_geometry_asset_groups",
]
