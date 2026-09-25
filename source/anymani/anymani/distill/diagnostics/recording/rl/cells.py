'Assign assets to the fixed eight morphology cells using handedness, active TIP count, and thumb DoF.'

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from anymani.assets.canonical_runtime import CANONICAL_HAND_SCHEMA_V1, CanonicalHandRouting


@dataclass(frozen=True, order=True)
class MorphologyCell:
    'Contract for morphology cell.'

    handedness: str
    tip_count: int
    thumb_dof: int

    def __post_init__(self) -> None:
        'Validate the declared contract.'

        if self.handedness not in {"left", "right"}:
            raise ValueError("morphology cell handedness must be left or right")
        if self.tip_count not in {3, 4}:
            raise ValueError("morphology cell tip_count must be 3 or 4")
        if self.thumb_dof not in {3, 4}:
            raise ValueError("morphology cell thumb_dof must be 3 or 4")

    @property
    def cell_id(self) -> int:
        'Handle cell id.'

        handedness_offset = 0 if self.handedness == "left" else 4
        tip_offset = 0 if self.tip_count == 3 else 2
        thumb_offset = 0 if self.thumb_dof == 3 else 1
        return handedness_offset + tip_offset + thumb_offset

    @property
    def label(self) -> str:
        'Handle label.'

        return f"{self.handedness}_tips{self.tip_count}_thumb{self.thumb_dof}dof"

    @classmethod
    def all_cells(cls) -> tuple[MorphologyCell, ...]:
        'Handle all cells.'

        return tuple(
            cls(handedness, tip_count, thumb_dof)
            for handedness in ("left", "right")
            for tip_count in (3, 4)
            for thumb_dof in (3, 4)
        )


def morphology_cell_from_routing(routing: CanonicalHandRouting) -> MorphologyCell:
    'Handle morphology cell from routing.'

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
    'Handle balanced morphology rows; shapes [CanonicalHandRouting].'

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
