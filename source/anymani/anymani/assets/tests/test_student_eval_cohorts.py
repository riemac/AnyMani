r"""共享学生四-strata source-selection 的纯 Python 合同测试。

测试故意只构造小型 YAML/JSON metadata 和文本 bundle，不导入 Isaac、AnyMani task、canonical lowering 或
策略评价器。要证伪的命题是：typed pure-right 过滤不会吸收 mixed/left；variant 与 new-base 的 lineage
规则互不混淆；Allegro revision parent/child 不会被当成两个逻辑资产；固定 source hash 顺序在相同输入上
可重复；所有输出仍明确标记 ``pending-canonical``。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.research.family_teacher_distillation.prepare_evaluation_cohorts import (
    _select_balanced,
    build_source_selection_plan,
)


def _digest(text: str) -> str:
    r"""为合成 metadata 生成合法 SHA-256，避免测试依赖随机 UUID。"""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()  # source identity fixture 的稳定摘要


def _write_source_bundle(path: Path) -> None:
    r"""写入最小 hand bundle，使 source-file SHA 合同可以被真正核对。"""

    path.mkdir(parents=True, exist_ok=True)  # 只创建测试临时目录，不触碰仓库资产树
    (path / "hand.yaml").write_text("family: fixture\n", encoding="utf-8")  # sidecar source evidence
    (path / "hand.urdf").write_text("<robot name='fixture'/>\n", encoding="utf-8")  # URDF source evidence


def _record(
    root: Path,
    *,
    family: str,
    suite: str,
    base: str,
    asset_id: str,
    role: str = "variant",
    collection_kind: str = "groups",
    handedness: str = "right",
) -> dict[str, Any]:
    r"""构造一条与最终 N040 evaluation manifest 同字段语义的 source record。"""

    group = "single_palm_leap" if family == "leap" else "single_palm_allegro"
    mother_path = root / family / base  # fixture mother path保留 family/base lineage
    variant_set = "fixture-variants" if role == "variant" else ""  # mother 直接位于 mother_path
    asset_path = mother_path / variant_set / asset_id if role == "variant" else mother_path
    _write_source_bundle(asset_path)  # 只物化两个小文本文件，测试 source SHA 输出
    return {
        "asset_id": asset_id,
        "content_hash": _digest(f"content:{asset_id}"),
        "physical_geometry_hash": _digest(f"n040-physical:{asset_id}"),
        "configuration_domain_hash": _digest(f"config:{asset_id}"),
        "partition": f"evaluation.{suite}",
        "source_kind": "generated",
        "topology_key": base,
        "family": family,
        "handedness": handedness,
        "joint_count": 16,
        "owner_count": 21,
        "run_alias": "fixture",
        "run_dir": str(root),
        "collection_kind": collection_kind,
        "group_name": group if collection_kind == "groups" else f"mixed_{group}",
        "mother_name": base,
        "mother_path": str(mother_path),
        "variant_set": variant_set,
        "asset_role": role,
    }


def _lock_member(root: Path, *, family: str, base: str, asset_id: str, role: str = "variant") -> dict[str, Any]:
    r"""构造 canonical lock member；canonical physical fields只作 fixture，不参与脚本比较。"""

    group = "single_palm_leap" if family == "leap" else "single_palm_allegro"
    mother_path = root / family / base
    return {
        "asset_id": asset_id,
        "content_hash": _digest(f"teacher-content:{asset_id}"),
        "physical_geometry_hash": _digest(f"canonical-physical:{asset_id}"),
        "configuration_domain_hash": _digest(f"canonical-config:{asset_id}"),
        "canonical_schema_digest": _digest("canonical-schema"),
        "provenance": {
            "partition": "train",
            "run_alias": "fixture",
            "run_dir": str(root),
            "collection_kind": "groups",
            "group_name": group,
            "mother_name": base,
            "mother_path": str(mother_path),
            "variant_set": "teacher-variants" if role == "variant" else "",
            "asset_role": role,
        },
        "mutation_descriptor": {"kind": role},
    }


def _write_lock(path: Path, members: list[dict[str, Any]], cohort_id: str) -> None:
    r"""写入最小 JSON canonical lock，模拟 writer 的 JSON-with-yaml-suffix 产物。"""

    path.write_text(json.dumps({"cohort_id": cohort_id, "members": members}, sort_keys=True), encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, Any]:
    r"""建立四个 strata 共用的 typed manifest、teacher locks、revision map 和 train artifact IDs。"""

    train_groups = {
        "single_palm_leap": {
            "right_l_seen": {"include_mother": True, "variant_sets": ["train"]},
            "right_l_teacher": {"include_mother": True, "variant_sets": ["train"]},
        },
        "single_palm_allegro": {
            "right_a_seen": {"include_mother": True, "variant_sets": ["train"]},
            "right_a_teacher": {"include_mother": True, "variant_sets": ["train"]},
        },
    }
    train_manifest = {"schema_version": "2.0.0", "train": {"runs": {"default": {"groups": train_groups}}}}
    train_path = tmp_path / "ssl.yaml"
    import yaml

    train_path.write_text(yaml.safe_dump(train_manifest, sort_keys=False), encoding="utf-8")

    eval_entries: dict[str, list[dict[str, Any]]] = {"unseen_variant_set": [], "unseen_mother": []}
    # 两个 teacher-seen variant bases 使 round-robin 配额能被直接观察。
    for family, bases in {
        "leap": ["right_l_seen", "right_l_teacher"],
        "allegro": ["right_a_seen", "right_a_teacher"],
    }.items():
        for base in bases:
            for index in range(4):
                eval_entries["unseen_variant_set"].append(
                    _record(
                        tmp_path,
                        family=family,
                        suite="unseen_variant_set",
                        base=base,
                        asset_id=f"{family}-variant-{base}-{index}",
                    )
                )
    # 每个 new-base 组包含一个 mother 和三个 variant；脚本必须 mother-first 再做 hash 选择。
    for family, bases in {
        "leap": ["right_l_new0", "right_l_new1"],
        "allegro": ["right_a_new0", "right_a_new1"],
    }.items():
        for base in bases:
            eval_entries["unseen_mother"].append(
                _record(
                    tmp_path,
                    family=family,
                    suite="unseen_mother",
                    base=base,
                    asset_id=f"{family}-mother-{base}",
                    role="mother",
                )
            )
            for index in range(3):
                eval_entries["unseen_mother"].append(
                    _record(
                        tmp_path,
                        family=family,
                        suite="unseen_mother",
                        base=base,
                        asset_id=f"{family}-mother-variant-{base}-{index}",
                    )
                )
    # 同一 family 字段的 mixed 与 left 行必须在 typed filter 中被排除。
    eval_entries["unseen_variant_set"].append(
        _record(
            tmp_path,
            family="leap",
            suite="unseen_variant_set",
            base="right_mixed_should_drop",
            asset_id="mixed-drop",
            collection_kind="mixed",
        )
    )
    eval_entries["unseen_variant_set"].append(
        _record(
            tmp_path,
            family="leap",
            suite="unseen_variant_set",
            base="left_should_drop",
            asset_id="left-drop",
            handedness="left",
        )
    )
    eval_path = tmp_path / "extended512-eval.yaml"
    eval_path.write_text(
        yaml.safe_dump({"schema_version": "4.0.0", "evaluation": eval_entries}, sort_keys=False), encoding="utf-8"
    )

    teacher_specs: list[tuple[str, Path]] = []
    for label, family, base in (
        ("leap_teacher_train", "leap", "right_l_seen"),
        ("leap_teacher_dev_full", "leap", "right_l_seen"),
        ("leap_teacher_old_final", "leap", "right_l_seen"),
        ("allegro_teacher_train_old", "allegro", "right_a_seen"),
        ("allegro_teacher_train_old_secondary", "allegro", "right_a_teacher"),
        ("allegro_teacher_dev_old", "allegro", "right_a_seen"),
        ("allegro_teacher_old_final", "allegro", "right_a_seen"),
    ):
        lock_path = tmp_path / f"{label}.canonical.lock.yaml"
        _write_lock(lock_path, [_lock_member(tmp_path, family=family, base=base, asset_id=f"teacher-{label}")], label)
        teacher_specs.append((label, lock_path))
    revised_path = tmp_path / "allegro-revised-training.canonical.lock.yaml"
    _write_lock(
        revised_path,
        [_lock_member(tmp_path, family="allegro", base="right_a_seen", asset_id="revised-a")],
        "allegro-revised",
    )
    teacher_specs.append(("allegro_teacher_train_revised", revised_path))

    revision_path = tmp_path / "revision.json"
    revision_path.write_text(
        json.dumps(
            {
                "assets": 2,
                "modified_assets": 1,
                "unchanged_assets": 1,
                "mapping": [{"parent_asset_id": "old-a", "asset_id": "new-a"}],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    artifacts_path = tmp_path / "source_artifacts.jsonl"
    artifacts_path.write_text(
        "\n".join(json.dumps({"asset_id": asset_id}) for asset_id in ("ssl-train-leap", "ssl-train-allegro")) + "\n",
        encoding="utf-8",
    )
    return {
        "train": train_path,
        "evaluation": eval_path,
        "artifacts": artifacts_path,
        "teacher": tuple(teacher_specs),
        "revision": revision_path,
    }


def test_source_plan_is_typed_deterministic_and_pending_canonical(tmp_path: Path) -> None:
    r"""纯-right 过滤、base 资格、source SHA 和 pending-canonical 状态必须同时成立。"""

    fixture = _fixture(tmp_path)
    plan = build_source_selection_plan(
        repo_root=tmp_path,
        ssl_train_manifest=fixture["train"],
        ssl_eval_manifest=fixture["evaluation"],
        ssl_train_artifacts=fixture["artifacts"],
        teacher_locks=fixture["teacher"],
        allegro_revision=fixture["revision"],
        target_count=4,
    )
    repeat = build_source_selection_plan(
        repo_root=tmp_path,
        ssl_train_manifest=fixture["train"],
        ssl_eval_manifest=fixture["evaluation"],
        ssl_train_artifacts=fixture["artifacts"],
        teacher_locks=fixture["teacher"],
        allegro_revision=fixture["revision"],
        target_count=4,
    )

    assert plan["status"] == "pending-canonical"
    assert plan["policy_results_read"] is False
    assert plan["selection_digest"] == repeat["selection_digest"]
    assert plan["strata"]["leap_right_variant"]["source_pool_count"] == 8  # mixed/left fixture rows被排除
    assert plan["strata"]["leap_right_variant"]["selected_count"] == 4
    assert plan["strata"]["allegro_right_variant"]["eligible_base_count"] == 2
    assert plan["strata"]["leap_right_mother"]["mother_role_count_in_source_pool"] == 2
    assert plan["strata"]["allegro_right_mother"]["eligible_pool_count"] == 8
    assert plan["canonical_physical_isolation"]["status"] == "pending-canonical"

    # 每个 selected map保留原 partition/path/variant_set 与两个 source 文件的 SHA。
    member = plan["strata"]["leap_right_mother"]["selected"][0]
    assert member["source_record"]["partition"] == "evaluation.unseen_mother"
    assert set(member["source_asset"]["source_file_sha256"]) == {"hand.yaml", "hand.urdf"}
    assert member["canonical_physical_identity"]["canonical_physical_geometry_hash"] is None


def test_allegro_variant_excludes_unseen_base_and_reports_location(tmp_path: Path) -> None:
    r"""Allegro variant 只接受 teacher-train base；缺失 base 必须留在 excluded 定位中。"""

    fixture = _fixture(tmp_path)
    # 将一个额外的 candidate base 注入 eval 文件；它不在 Allegro teacher train。
    import yaml

    document = yaml.safe_load(fixture["evaluation"].read_text(encoding="utf-8"))
    document["evaluation"]["unseen_variant_set"].extend(
        _record(
            tmp_path,
            family="allegro",
            suite="unseen_variant_set",
            base="right_a_not_teacher_trained",
            asset_id="allegro-not-teacher-0",
        )
        for _ in range(2)
    )
    fixture["evaluation"].write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    plan = build_source_selection_plan(
        repo_root=tmp_path,
        ssl_train_manifest=fixture["train"],
        ssl_eval_manifest=fixture["evaluation"],
        ssl_train_artifacts=fixture["artifacts"],
        teacher_locks=fixture["teacher"],
        allegro_revision=fixture["revision"],
        target_count=4,
    )
    details = plan["strata"]["allegro_right_variant"]
    assert details["source_pool_count"] == 10
    assert details["eligible_pool_count"] == 8
    assert any(
        "base_not_in_teacher_train" in member["selection"]["exclusion_reasons"] for member in details["excluded"]
    )


def test_balanced_quotas_and_new_base_mother_first_are_source_only() -> None:
    r"""七个 Allegro base 的 floor/ceil 配额差为一，新 base 每组首项为 mother。"""

    records = []
    for base_index in range(7):
        base = f"right_base_{base_index}"
        records.append(
            {
                "family": "allegro",
                "mother_name": base,
                "mother_path": f"/source/{base}",
                "asset_id": f"m-{base_index}",
                "asset_role": "mother",
            }
        )
        records.extend(
            {
                "family": "allegro",
                "mother_name": base,
                "mother_path": f"/source/{base}",
                "asset_id": f"v-{base_index}-{variant_index}",
                "asset_role": "variant",
            }
            for variant_index in range(5)
        )
    selected, _reserve, quotas, base_order = _select_balanced(
        records,
        family="allegro",
        kind="new_base",
        target_count=32,
        new_base=True,
    )
    counts = {base: sum(row["mother_name"] == base for row in selected) for base in base_order}
    assert max(counts.values()) - min(counts.values()) <= 1
    assert max(quotas.values()) - min(quotas.values()) <= 1
    for base in base_order:
        selected_for_base = [row for row in selected if row["mother_name"] == base]
        assert selected_for_base[0]["asset_role"] == "mother"
