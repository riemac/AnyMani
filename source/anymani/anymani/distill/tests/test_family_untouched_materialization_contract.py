r"""untouched evaluation source registration 的最小 CPU contract。

这里不启动 Isaac Sim，也不执行 canonical lowering；只锁定四格配额和 technical
manifest 对 mother/variant lineage 的表达。真实 HandAssetDataset/cohort writer
路径由 case 内 source-registration smoke 验证。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.research.family_teacher_distillation.prepare_untouched_evaluation import (
    _validate_selected_shape,
    build_technical_manifest,
)


def _item(
    *,
    root: Path,
    mother_name: str,
    kind: str,
    variant_set: str,
    index: int,
) -> dict[str, object]:
    """构造已通过 source resolver 的内部 normalized record。"""

    return {
        "asset_id": f"{mother_name}-{index}",
        "mother_name": mother_name,
        "kind": kind,
        "variant_set": variant_set,
        "_run_root": root / "generated",
        "_group_name": "single_palm_allegro",
        "_bundle_path": root / "generated" / "single_palm_allegro" / mother_name / str(index),
    }


def test_technical_manifest_keeps_variant_only_and_mother_lineages_distinct(tmp_path: Path) -> None:
    r"""variant-only lineage 不隐式收录母体，new-base lineage 显式收录母体。"""

    variant_details = {
        "selected": [_item(root=tmp_path, mother_name="right_variant", kind="variant", variant_set="v1", index=1)],
        "reserve": [],
    }
    variant_manifest = build_technical_manifest(variant_details, family="allegro")
    variant_lineage = variant_manifest["train"]["runs"]["source_00"]["groups"]["single_palm_allegro"][
        "right_variant"
    ]
    assert variant_lineage == {"include_mother": False, "variant_sets": ["v1"]}

    mother_details = {
        "selected": [
            _item(root=tmp_path, mother_name="right_mother", kind="mother", variant_set="", index=2),
            _item(root=tmp_path, mother_name="right_mother", kind="variant", variant_set="v1", index=3),
        ],
        "reserve": [],
    }
    mother_manifest = build_technical_manifest(mother_details, family="allegro")
    mother_lineage = mother_manifest["train"]["runs"]["source_00"]["groups"]["single_palm_allegro"][
        "right_mother"
    ]
    assert mother_lineage == {"include_mother": True, "variant_sets": ["v1"]}


def test_selected_shape_rejects_wrong_mother_variant_proportion(tmp_path: Path) -> None:
    r"""new-base selected 轴必须是八个 mother lineage、每格一母三变体。"""

    selected = []
    for base_index in range(8):
        mother_name = f"right_mother_{base_index}"
        selected.append(
            _item(root=tmp_path, mother_name=mother_name, kind="mother", variant_set="", index=base_index * 4)
        )
        selected.extend(
            _item(
                root=tmp_path,
                mother_name=mother_name,
                kind="variant",
                variant_set="v1",
                index=base_index * 4 + offset,
            )
            for offset in (1, 2, 3)
        )
    details = {"nominal_assets": 32, "base_designs": 8}
    _validate_selected_shape(
        stratum="allegro_right_mother",
        details=details,
        selected=selected,
        reserve=[],
    )

    broken = list(selected)
    broken[1] = {**broken[1], "kind": "mother", "variant_set": ""}
    with pytest.raises(ValueError, match="proportions|exactly one mother"):
        _validate_selected_shape(
            stratum="allegro_right_mother",
            details=details,
            selected=broken,
            reserve=[],
        )
