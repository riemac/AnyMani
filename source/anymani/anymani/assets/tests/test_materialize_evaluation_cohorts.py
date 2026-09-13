r"""evaluation-only source registration 与 canonical 补位的纯 CPU 合同测试。

这些测试只覆盖 manifest 视图、provenance 注册和确定性 canonical collision 选择，不调用
``HandAssetDataset.resolve_train``、IsaacLab 或物理仿真。这样既能验证 ``train`` 非空这一现有
cohort API 约束，也能避免把技术注册视图误当成 SSL/PPO 训练暴露。真实 bundle 的 sidecar、URDF
与 canonical lowering 由 primary 在独立输出目录中运行本脚本的 ``--canonicalize`` 阶段验证。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.research.family_teacher_distillation.materialize_evaluation_cohorts import (
    _bind_registration_rows,
    _canonical_selection,
    _registration_document,
    _verify_resolved_source_members,
    build_evaluation_only_manifest,
)


def _digest(value: str) -> str:
    r"""为 fixture 生成合法 SHA-256，确保测试覆盖字段绑定而不是 hash 格式捷径。"""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _member(asset_id: str, *, row: int, role: str) -> dict[str, Any]:
    r"""构造带 origin/source 坐标的最小 plan member，模拟 prepare 脚本的真实输出。"""

    return {
        "asset_id": asset_id,
        "source_record": {
            "asset_id": asset_id,
            "partition": "evaluation.unseen_variant_set",
            "family": "leap",
            "handedness": "right",
            "collection_kind": "groups",
            "group_name": "single_palm_leap",
            "mother_name": "right_fixture",
            "mother_path": "/source/right_fixture",
            "variant_set": "fixture-variants",
            "asset_role": "variant",
            "content_hash": _digest(f"content:{asset_id}"),
            "physical_geometry_hash": _digest(f"source-physical:{asset_id}"),
        },
        "origin_source": {
            "partition": "evaluation.unseen_variant_set",
            "manifest_row": row,
            "asset_id": asset_id,
            "mother_path": "/source/right_fixture",
            "variant_set": "fixture-variants",
        },
        "source_asset": {
            "path": f"/source/right_fixture/fixture-variants/{asset_id}",
            "source_file_sha256": {
                "hand.yaml": _digest(f"yaml:{asset_id}"),
                "hand.urdf": _digest(f"urdf:{asset_id}"),
            },
        },
        "pre_revision_source_identity": {
            "status": "outside-allegro-training-revision",
            "pre_revision_asset_id": asset_id,
            "revised_asset_id": None,
        },
        "selection": {"role": role, "source_sort_key": f"sort:{asset_id}"},
    }


def _artifact(asset_id: str, physical: str) -> dict[str, str]:
    r"""构造 canonical runtime 摘要；physical 字段是唯一参与 collision 的同域身份。"""

    return {
        "asset_id": asset_id,
        "configuration_domain_hash": _digest(f"config:{asset_id}"),
        "source_content_hash": _digest(f"config:{asset_id}"),
        "physical_geometry_hash": physical,
        "canonical_schema_digest": _digest("canonical-schema-v1"),
    }


def test_evaluation_manifest_is_nonempty_technical_train_and_pure_right_only() -> None:
    r"""技术 manifest 复制 suite run/default root，且只留下目标 pure-right group/base。"""

    pure = {
        "include_mother": False,
        "variant_sets": ["evaluation-variants"],
    }
    source = {
        "schema_version": "2.0.0",
        "default_run_dir": "source/generated/final-run",
        "train": {"runs": {"default": {"groups": {}}}},
        "evaluation": {
            "unseen_variant_set": {
                "runs": {
                    "default": {
                        "run_dir": "source/generated/final-run",
                        "groups": {
                            "single_palm_leap": {
                                "right_fixture": pure,
                                "left_fixture": {**pure, "include_mother": True},
                            },
                            "single_palm_allegro": {"right_other": pure},
                        },
                        "mixed": {"mixed_fixture": pure},
                    }
                }
            }
        },
    }

    technical = build_evaluation_only_manifest(
        source,
        family="leap",
        suite="unseen_variant_set",
        allowed_bases=("right_fixture",),
    )

    assert technical["schema_version"] == "2.0.0"
    assert technical["default_run_dir"] == source["default_run_dir"]
    assert technical["train"]["runs"]["default"]["groups"] == {"single_palm_leap": {"right_fixture": pure}}
    assert technical["train"]["runs"]["default"].get("mixed", {}) == {}
    assert technical["validation"]["unseen_variant_set"]["runs"] == {}
    assert technical["evaluation"]["unseen_variant_set"]["runs"] == {}


def test_registration_binds_origin_and_technical_rows_without_policy_fields(tmp_path: Path) -> None:
    r"""registration 同时保留 suite row、文件 SHA、plan SHA 和 technical dense row。"""

    plan_path = tmp_path / "selection-plan.json"
    eval_path = tmp_path / "extended512-eval.yaml"
    technical_path = tmp_path / "evaluation-only.yaml"
    for path, payload in (
        (plan_path, "plan"),
        (eval_path, "evaluation"),
        (technical_path, "technical"),
    ):
        path.write_text(payload, encoding="utf-8")
    selected = [_member("selected-a", row=17, role="selected")]
    reserve = [_member("reserve-b", row=23, role="reserve")]

    registration = _registration_document(
        plan={"selection_digest": _digest("selection")},
        plan_path=plan_path,
        eval_manifest_path=eval_path,
        technical_manifest_path=technical_path,
        technical_manifest_sha256=_digest("technical"),
        stratum="leap_right_variant",
        source_suite="evaluation.unseen_variant_set",
        selected=selected,
        reserve=reserve,
        root=tmp_path,
    )
    bound = _bind_registration_rows(
        registration,
        selected=selected,
        reserve=reserve,
        id_to_row={"selected-a": 3, "reserve-b": 8},
    )

    assert bound["registration_is_not_policy_or_ssl_training"] is True
    assert bound["origin_partition"] == "evaluation.unseen_variant_set"
    assert bound["origin_manifest_sha256"] == _digest("evaluation")
    assert bound["source_selection_plan_sha256"] == _digest("plan")
    assert bound["selected_origin_members"][0]["origin_manifest_row"] == 17
    assert bound["selected_origin_members"][0]["technical_manifest_row"] == 3
    assert bound["selected_origin_members"][0]["pre_revision_source_identity"]["pre_revision_asset_id"] == "selected-a"
    assert bound["reserve_origin_members"][0]["technical_manifest_row"] == 8
    assert "policy_results" not in json.dumps(bound, sort_keys=True)


def test_canonical_selection_replaces_protected_selected_in_frozen_reserve_order() -> None:
    r"""teacher collision 会淘汰原 selected，并按冻结 reserve 顺序补位且保留 replacement reason。"""

    selected = [
        _member("selected-a", row=1, role="selected"),
        _member("selected-protected", row=2, role="selected"),
    ]
    reserve = [
        _member("reserve-c", row=3, role="reserve"),
        _member("reserve-d", row=4, role="reserve"),
    ]
    details = {"selected": selected, "reserve": reserve}
    physical_a = _digest("physical-a")
    physical_protected = _digest("physical-protected")
    physical_c = _digest("physical-c")
    physical_d = _digest("physical-d")
    artifacts = {
        "selected-a": _artifact("selected-a", physical_a),
        "selected-protected": _artifact("selected-protected", physical_protected),
        "reserve-c": _artifact("reserve-c", physical_c),
        "reserve-d": _artifact("reserve-d", physical_d),
    }

    result = _canonical_selection(
        details,
        artifacts=artifacts,
        teacher_hashes={physical_protected},
        n040_train_hashes=set(),
        target_count=2,
    )

    assert result["status"] == "canonicalized-and-exposure-audited-pending-pregrasp"
    assert [member["asset_id"] for member in result["selected"]] == ["selected-a", "reserve-c"]
    assert [member["asset_id"] for member in result["reserve"]] == ["reserve-d"]
    assert result["replacements"][0]["replaced_planned_asset_id"] == "selected-protected"
    assert "canonical_physical_hash_in_protected_exposure" in result["excluded"][0]["reasons"]
    assert (
        result["selected"][0]["canonical_physical_identity"]["configuration_domain_hash"]
        == artifacts["selected-a"]["source_content_hash"]
    )


def test_resolved_source_identity_is_fail_closed_on_bundle_byte_drift(tmp_path: Path) -> None:
    r"""technical row 解析后仍需匹配 plan 的 bundle/source SHA，bytes 漂移必须拒绝注册。"""

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    urdf = bundle / "hand.urdf"
    sidecar = bundle / "hand.yaml"
    urdf.write_text("urdf-v1\n", encoding="utf-8")
    sidecar.write_text("yaml-v1\n", encoding="utf-8")
    member = _member("asset-a", row=11, role="selected")
    member["source_asset"]["path"] = "bundle"
    member["source_asset"]["source_file_sha256"] = {
        "hand.urdf": _digest(urdf.read_text(encoding="utf-8")),
        "hand.yaml": _digest(sidecar.read_text(encoding="utf-8")),
    }
    mother_path = str(tmp_path / "mother")
    member["source_record"]["mother_path"] = mother_path
    resolved_record = SimpleNamespace(
        container=SimpleNamespace(asset_id="asset-a", urdf_path=urdf, sidecar_path=sidecar),
        content_hash=member["source_record"]["content_hash"],
        provenance=SimpleNamespace(
            group_name="single_palm_leap",
            mother_name="right_fixture",
            mother_path=mother_path,
            variant_set="fixture-variants",
            asset_role="variant",
        ),
    )
    partition = SimpleNamespace(records=(resolved_record,))

    _verify_resolved_source_members(
        partition,
        selected=(member,),
        reserve=(),
        id_to_row={"asset-a": 0},
        root=tmp_path,
    )
    sidecar.write_text("yaml-v2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source file SHA changed"):
        _verify_resolved_source_members(
            partition,
            selected=(member,),
            reserve=(),
            id_to_row={"asset-a": 0},
            root=tmp_path,
        )


def test_bounded_exposure_prefix_never_reports_complete_audit() -> None:
    r"""已知前缀 hash 可用于排除，但状态必须继续等待完整 N040 exposure audit。"""

    member = _member("asset-a", row=1, role="selected")
    physical = _digest("physical-a")
    result = _canonical_selection(
        {"selected": [member], "reserve": []},
        artifacts={"asset-a": _artifact("asset-a", physical)},
        teacher_hashes=set(),
        n040_train_hashes=set(),
        n040_train_audit_complete=False,
        target_count=1,
    )

    assert result["status"] == "canonicalized-pending-n040-train-audit"
