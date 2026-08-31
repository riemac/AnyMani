"""八个 morphology cells 的唯一分类与 family-independence 合同。"""

from __future__ import annotations

from anymani.assets.canonical_runtime import CanonicalHandRouting
from anymani.distill.diagnostics.recording.rl.cells import (
    MorphologyCell,
    balanced_morphology_rows,
    morphology_cell_from_routing,
)


def _routing(*, handedness: str, tips: int, thumb_dof: int, family: str) -> CanonicalHandRouting:
    r"""构造指定手性、TIP数和拇指DOF的canonical routing。"""

    joint_mask = [False] * 16
    for depth in range(thumb_dof):
        joint_mask[depth * 4 + 3] = True  # thumb在physx finger order中的index=3
    for finger in range(tips - 1):
        for depth in range(3):
            joint_mask[depth * 4 + finger] = True
    tip_mask = tuple([True] * (tips - 1) + [False] * (4 - tips) + [True])
    names = tuple(f"j{index}" for index, active in enumerate(joint_mask) if active)
    return CanonicalHandRouting(
        asset_id=f"{handedness}-{tips}-{thumb_dof}-{family}",
        source_dof_count=sum(joint_mask),
        source_joint_names=names,
        active_joint_names=names,
        active_joint_mask=tuple(joint_mask),
        active_tip_mask=tip_mask,
        source_to_canonical=tuple((name, name) for name in names),
        handedness=handedness,
        family=family,
    )


def test_all_eight_cells_are_unique_and_ignore_binary_routing_family() -> None:
    r"""Cell只能由handedness×TIP count×thumb DoF定义，family/mixed provenance不能泄漏。"""

    cells = set()
    for handedness in ("left", "right"):
        for tips in (3, 4):
            for thumb_dof in (3, 4):
                leap = morphology_cell_from_routing(
                    _routing(handedness=handedness, tips=tips, thumb_dof=thumb_dof, family="leap")
                )
                allegro = morphology_cell_from_routing(
                    _routing(handedness=handedness, tips=tips, thumb_dof=thumb_dof, family="allegro")
                )
                assert leap == allegro
                assert leap.label == f"{handedness}_tips{tips}_thumb{thumb_dof}dof"
                cells.add(leap)
    assert len(cells) == 8
    assert cells == set(MorphologyCell.all_cells())


def test_balanced_selection_takes_equal_rows_from_every_cell() -> None:
    r"""Canary selection不能把manifest ordered prefix误称为八组平衡。"""

    routings = []
    row = 0
    for handedness in ("left", "right"):
        for tips in (3, 4):
            for thumb_dof in (3, 4):
                for family in ("leap", "allegro", "generic"):
                    routing = _routing(handedness=handedness, tips=tips, thumb_dof=thumb_dof, family=family)
                    object.__setattr__(routing, "asset_row", row)
                    routings.append(routing)
                    row += 1

    selected = balanced_morphology_rows(routings, rows_per_cell=2)

    assert len(selected) == 16
    selected_cells = [morphology_cell_from_routing(routings[index]) for index in selected]
    assert all(selected_cells.count(cell) == 2 for cell in MorphologyCell.all_cells())
