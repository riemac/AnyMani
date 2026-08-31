r"""异构PPO固定八个morphology cells的唯一分类合同。

Cell只由handedness、active fingertip count与thumb DoF构成：

$$
g=(h,n_{tip},d_{thumb})\in\{L,R\}\times\{3,4\}\times\{3,4\}.
$$

顶层routing `family`只是生成/provenance标签，不能表达mixed composition，因此不得进入cell identity或
actor输入。Active DoF/topology继续作为分层诊断字段，不扩展本八组轴。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from anymani.assets.canonical_runtime import CANONICAL_HAND_SCHEMA_V1, CanonicalHandRouting


@dataclass(frozen=True, order=True)
class MorphologyCell:
    r"""一个固定handedness×TIP-count×thumb-DoF诊断cell。"""

    handedness: str
    tip_count: int
    thumb_dof: int

    def __post_init__(self) -> None:
        r"""拒绝八组支持域外的标签。"""

        if self.handedness not in {"left", "right"}:
            raise ValueError("morphology cell handedness must be left or right")
        if self.tip_count not in {3, 4}:
            raise ValueError("morphology cell tip_count must be 3 or 4")
        if self.thumb_dof not in {3, 4}:
            raise ValueError("morphology cell thumb_dof must be 3 or 4")

    @property
    def cell_id(self) -> int:
        r"""返回稳定`0..7`cell ID；只服务diagnostics/ADR metadata。"""

        handedness_offset = 0 if self.handedness == "left" else 4
        tip_offset = 0 if self.tip_count == 3 else 2
        thumb_offset = 0 if self.thumb_dof == 3 else 1
        return handedness_offset + tip_offset + thumb_offset

    @property
    def label(self) -> str:
        r"""返回TensorBoard/JSONL稳定标签。"""

        return f"{self.handedness}_tips{self.tip_count}_thumb{self.thumb_dof}dof"

    @classmethod
    def all_cells(cls) -> tuple[MorphologyCell, ...]:
        r"""按cell ID顺序枚举固定八组。"""

        return tuple(
            cls(handedness, tip_count, thumb_dof)
            for handedness in ("left", "right")
            for tip_count in (3, 4)
            for thumb_dof in (3, 4)
        )


def morphology_cell_from_routing(routing: CanonicalHandRouting) -> MorphologyCell:
    r"""从canonical routing推导八组cell，忽略binary family标签。

    Args:
        routing (CanonicalHandRouting): 单资产active joint/TIP与handedness真源。

    Returns:
        MorphologyCell: 固定八组之一。
    """

    tip_count = sum(bool(active) for active in routing.active_tip_mask)
    finger_order = CANONICAL_HAND_SCHEMA_V1.physx_finger_order  # index/middle/ring/thumb
    thumb_index = finger_order.index("thumb")
    thumb_slots = tuple(
        depth * len(finger_order) + thumb_index
        for depth in range(CANONICAL_HAND_SCHEMA_V1.max_revolute_per_finger)
    )
    thumb_dof = sum(bool(routing.active_joint_mask[index]) for index in thumb_slots)
    return MorphologyCell(str(routing.handedness), tip_count, thumb_dof)


def balanced_morphology_rows(
    routings: Sequence[CanonicalHandRouting],
    *,
    rows_per_cell: int,
) -> tuple[int, ...]:
    r"""按原dataset row顺序从八组各取固定数量资产。

    该函数只定义诊断/canary selection；formal PPO仍消费完整manifest。选择不读取routing family，且每个cell
    数量不足时fail closed，避免ordered prefix被误称为balanced cohort。

    Args:
        routings (Sequence[CanonicalHandRouting]): formal dataset顺序的routing rows。
        rows_per_cell (int): 每组所需资产数。

    Returns:
        tuple[int, ...]: 按cell ID、组内原asset row排序的selection-local row IDs。
    """

    if rows_per_cell < 1:
        raise ValueError("rows_per_cell must be positive")
    buckets: dict[MorphologyCell, list[int]] = {cell: [] for cell in MorphologyCell.all_cells()}
    for fallback_row, routing in enumerate(routings):
        cell = morphology_cell_from_routing(routing)
        row = int(routing.asset_row if routing.asset_row >= 0 else fallback_row)
        buckets[cell].append(row)
    insufficient = {cell.label: len(rows) for cell, rows in buckets.items() if len(rows) < rows_per_cell}
    if insufficient:
        raise ValueError(f"morphology cells lack requested rows_per_cell={rows_per_cell}: {insufficient}")
    return tuple(row for cell in MorphologyCell.all_cells() for row in sorted(buckets[cell])[:rows_per_cell])


__all__ = ["MorphologyCell", "balanced_morphology_rows", "morphology_cell_from_routing"]
