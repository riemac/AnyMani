r"""80手MVP代表性pair排序的纯Python合同。"""

from __future__ import annotations

from anymani.assets.bank.representative_selection import (
    SUPPORTED_CELL_VALUES,
    RepresentativeAsset,
    finalize_representative_selection,
    ranked_representative_pairs,
)


def _asset(
    *,
    row: int,
    handedness: str,
    cell: tuple[int, int],
    topology: str,
    family: str,
    descriptor: tuple[float, ...],
) -> RepresentativeAsset:
    r"""构造不依赖真实bundle IO的selection fixture。"""

    tip_count, thumb_dof = cell
    return RepresentativeAsset(
        row=row,
        asset_id=f"asset-{row}",
        geometry_identity=f"geometry-{row}",
        handedness=handedness,
        tip_count=tip_count,
        thumb_dof=thumb_dof,
        active_dof=tip_count * 2 + thumb_dof,
        topology=topology,
        family_signature=family,
        asset_role="mother" if row % 6 == 0 else "variant",
        descriptor=descriptor,
    )


def _balanced_fixture() -> tuple[RepresentativeAsset, ...]:
    r"""为每个cell构造12个topology-paired left/right候选。"""

    assets: list[RepresentativeAsset] = []
    row = 0
    for cell_index, cell in enumerate(SUPPORTED_CELL_VALUES):
        for pair_index in range(12):
            topology = f"t{cell_index}-{pair_index % 6}"
            family = f"index:{'leap' if pair_index % 2 else 'allegro'}|thumb:leap"
            center = float(cell_index * 100 + pair_index)
            assets.append(
                _asset(
                    row=row,
                    handedness="left",
                    cell=cell,
                    topology=topology,
                    family=family,
                    descriptor=(-center, float(pair_index % 3), 1.0),
                )
            )
            row += 1
            assets.append(
                _asset(
                    row=row,
                    handedness="right",
                    cell=cell,
                    topology=topology,
                    family=family,
                    descriptor=(-center + 0.01, float(pair_index % 3), 1.0),
                )
            )
            row += 1
    return tuple(assets)


def test_pair_ranking_is_deterministic_balanced_and_non_reusing() -> None:
    r"""输入顺序不能改变四cell候选序，且每项资产最多进入一个pair。"""

    assets = _balanced_fixture()
    forward = ranked_representative_pairs(assets)
    reverse = ranked_representative_pairs(tuple(reversed(assets)))
    assert tuple(forward) == SUPPORTED_CELL_VALUES
    for cell in SUPPORTED_CELL_VALUES:
        forward_rows = tuple((pair.left.row, pair.right.row) for pair in forward[cell])
        reverse_rows = tuple((pair.left.row, pair.right.row) for pair in reverse[cell])
        assert forward_rows == reverse_rows
        assert len(forward_rows) == 12
        assert len({row for pair in forward_rows for row in pair}) == 24
        assert all(pair.left.handedness == "left" and pair.right.handedness == "right" for pair in forward[cell])
        assert all(pair.left.cell == cell and pair.right.cell == cell for pair in forward[cell])


def test_final_selection_uses_only_complete_pairs_in_candidate_order() -> None:
    r"""Good-pregrasp失败只允许跳过完整pair，不能破坏8-cell左右平衡。"""

    assets = _balanced_fixture()
    ranked = ranked_representative_pairs(assets)
    cells = []
    passed_rows = []
    for cell in SUPPORTED_CELL_VALUES:
        pair_documents = []
        for rank, pair in enumerate(ranked[cell]):
            pair_documents.append(
                {
                    "rank": rank,
                    "topology": pair.topology,
                    "family_signature": pair.family_signature,
                    "left": {"row": pair.left.row},
                    "right": {"row": pair.right.row},
                }
            )
            if rank != 0:
                passed_rows.extend((pair.left.row, pair.right.row))
        cells.append(
            {
                "label": f"tips{cell[0]}_thumb{cell[1]}dof",
                "tip_count": cell[0],
                "thumb_dof": cell[1],
                "candidate_pairs": pair_documents,
            }
        )
    candidate_document = {
        "artifact_type": "anymani.hand_asset_representative_selection",
        "schema_version": "1.0.0",
        "selection_name": "fixture",
        "parent_dataset_path": "ppo.yaml",
        "parent_dataset_sha256": "a" * 64,
        "selection_algorithm": "fixture",
        "pairs_per_cell": 10,
        "cells": cells,
    }
    final = finalize_representative_selection(
        candidate_document,
        passed_rows=passed_rows,
        pregrasp_catalog_root="outputs/pregrasp/catalog",
        pregrasp_summary_paths=("summary.json",),
    )
    assert len(final["selected_rows"]) == 80
    assert len(set(final["selected_rows"])) == 80
    assert len(final["rejected_pairs"]) == 4
    assert all(cell["pairs"][0]["rank"] == 1 for cell in final["cells"])
