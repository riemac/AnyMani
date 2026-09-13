r"""限位修订发布器的 variant-only / selected-mother contract tests。

这些测试只启动 Python 资产 schema、标准 exporter、dataset resolver 与 cohort writer。
它们验证的是发布边界：registration-only parent 必须存在以闭合新 variant set，
但不能进入 selected member axis、训练/评测分母或新 manifest 的 ``include_mother``。
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from anymani.assets.asset_schema_core import CollisionGeometryCfg, JointLimitCfg, PoseCfg
from anymani.assets.asset_schema_embodiment import FingerCfg, HandCfg, JointCfg, PalmCfg
from anymani.assets.bank.cohort import load_hand_asset_cohort, write_hand_asset_cohort_lock
from anymani.assets.exporter.hand_exporter import HandExporter, HandExporterCfg
from anymani.assets.generator.result import HandGenerationResult
from anymani.assets.scripts import revise_joint_limits


def _hand(*, name: str, family: str = "allegro", handedness: str = "right") -> HandCfg:
    r"""构造带真实 primitive collision 的最小 Allegro hand fixture。

    ``index`` 恰有 MCP1/MCP2 两个活动关节，因此是本次 2.23 rad 规则的选中轴；
    ``middle`` 有三关节，``thumb`` 即使两关节也必须被规则排除。每根手指保留显式
    fixed TIP，使标准 geometry semantics exporter 不需要测试侧伪造 schema。
    """

    def joint(
        finger: str,
        index: int,
        parent: str,
        child: str,
        *,
        fixed: bool = False,
        upper: float = 1.61,
    ) -> JointCfg:
        """创建一个带 box collision 的活动关节或 TIP。"""

        return JointCfg(
            name=f"{finger}_tip" if fixed else f"{finger}_j{index}",
            parent=parent,
            child=child,
            joint_type="fixed" if fixed else "revolute",
            limit=None if fixed else JointLimitCfg(lower=-0.196, upper=upper, effort=10.0, velocity=3.14),
            origin=PoseCfg(),
            collisions=[
                CollisionGeometryCfg(
                    name=f"{finger}_{index}_collision",
                    geometry={"type": "box", "size": (0.02, 0.02, 0.02)},
                )
            ],
            is_tip=fixed,
        )

    fingers: list[FingerCfg] = []
    for finger_name, active_count in (("index", 2), ("middle", 3), ("thumb", 2)):
        joints: list[JointCfg] = []
        parent = "palm"
        for joint_index in range(active_count):
            suffix = ("mcp1", "mcp2", "pip")[joint_index]
            child = f"{finger_name}_{suffix}"
            joints.append(joint(finger_name, joint_index, parent, child))
            parent = child
        joints.append(joint(finger_name, active_count, parent, f"{finger_name}_tip", fixed=True))
        fingers.append(FingerCfg(name=finger_name, parent_link="palm", joints=joints))

    return HandCfg(
        name=name,
        palm=PalmCfg(
            name="palm",
            origin=PoseCfg(),
            collisions=[
                CollisionGeometryCfg(
                    name="palm_collision",
                    geometry={"type": "box", "size": (0.08, 0.08, 0.02)},
                )
            ],
        ),
        fingers=fingers,
        family=family,
        handedness=handedness,
        metadata={
            "premade_connectivity": {
                "slot_family_map": {"index": "allegro", "middle": "allegro", "thumb": "allegro"}
            }
        },
    )


def _export_hand(path: Path, hand: HandCfg, *, asset_id: str, variant: bool = False) -> None:
    """调用正式 HandExporter 写一个可被 HandBank 解析的 bundle。"""

    metadata = {"id": asset_id}
    if variant:
        metadata.update(
            {
                "post_mutate_samples": {"limit_tweak": {"resolved_self_mode": "identity", "value": 0.0}},
                "source_origin_sample_id": "fixture-mother",
                "source_origin_topology_dir": str(path.parent.parent),
            }
        )
    result = HandGenerationResult(hand_cfg=hand, metadata=metadata)
    exported = HandExporter(HandExporterCfg()).export(result, path, sample_id=asset_id, nest_sample_dir=False)
    assert not exported.errors
    assert result.urdf_path is not None and result.sidecar_path is not None


def _source_manifest(root: Path, mother_name: str) -> tuple[Path, Path, Path]:
    r"""发布 source dataset manifest，并返回其 mother、variant 与 manifest 路径。"""

    generated = root / "generated"
    generated.mkdir(parents=True)
    (generated / "summary.yaml").write_text(yaml.safe_dump({"run": {"mode": "made"}}), encoding="utf-8")
    mother = generated / "single_palm_allegro" / mother_name
    _export_hand(mother, _hand(name="fixture_mother"), asset_id="fixture-mother")
    variant_set = mother / "variants"
    variant = variant_set / "fixture-variant"
    _export_hand(variant, _hand(name="fixture_variant"), asset_id="fixture-variant", variant=True)
    (variant_set / "summary.yaml").write_text(
        yaml.safe_dump(
            {
                "run": {"mode": "mutate"},
                "config": {"source_topology_dir": str(mother)},
                "stats": {"succeeded": 1},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    manifest = root / "dataset.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": "2.0.0",
                "default_run_dir": str(generated),
                "train": {
                    "runs": {
                        "source": {
                            "groups": {
                                "single_palm_allegro": {
                                    mother_name: {"include_mother": True, "variant_sets": ["variants"]}
                                }
                            }
                        }
                    }
                },
                "validation": {"unseen_variant_set": {"runs": {}}, "unseen_mother": {"runs": {}}},
                "evaluation": {
                    "unseen_variant_set": {"runs": {}},
                    "unseen_mother": {"runs": {}},
                    "official_zero_shot": {"assets": []},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return manifest, mother, variant


def _source_lock(root: Path, *, selected_mother: bool) -> tuple[Path, Path, Path, Path]:
    """用标准 cohort writer 选择 variant-only 或 mother+variant 轴。"""

    manifest, mother, variant = _source_manifest(root, "right_fixture")
    source_lock = root / "source.lock.yaml"
    coordinates = (("source", 0), ("source", 1)) if selected_mother else (("source", 1),)
    write_hand_asset_cohort_lock(
        source_lock,
        cohort_id="fixture-parent",
        source_manifests={"source": manifest},
        member_coordinates=coordinates,
        selection={"algorithm": "fixture"},
    )
    return source_lock, mother, variant, manifest


def _run_revision(
    monkeypatch: pytest.MonkeyPatch,
    source_lock: Path,
    output: Path,
    *,
    expected_assets: int,
) -> None:
    """通过真实 CLI main 入口执行一次 CPU-only revision publication。"""

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "revise_joint_limits",
            "--source-lock",
            str(source_lock),
            "--output-dir",
            str(output),
            "--cohort-id",
            "fixture-revision",
            "--expected-assets",
            str(expected_assets),
            "--expected-modified-assets",
            "1" if expected_assets == 1 else "2",
        ],
    )
    revise_joint_limits.main()


def _joint_upper(snapshot: dict, finger_name: str, joint_name: str) -> float:
    """读取 HandCfg snapshot 中指定关节的 upper，供差分断言使用。"""

    for finger in snapshot["fingers"]:
        if finger["name"] == finger_name:
            for joint in finger["joints"]:
                if joint["name"] == joint_name:
                    return float(joint["limit"]["upper"])
    raise AssertionError(f"joint not found: {finger_name}/{joint_name}")


def test_variant_only_publication_materializes_registration_parent_without_exposure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""variant-only 选择保持分母与 source 坐标，并将 registration parent 排除出 selected 轴。"""

    source_lock, mother, variant, _manifest = _source_lock(tmp_path / "variant-only", selected_mother=False)
    original_mother = yaml.safe_load((mother / "hand.yaml").read_text(encoding="utf-8"))
    original_variant = yaml.safe_load((variant / "hand.yaml").read_text(encoding="utf-8"))
    output = tmp_path / "published"
    _run_revision(monkeypatch, source_lock, output, expected_assets=1)

    report = json.loads((output / "revision.json").read_text(encoding="utf-8"))
    assert report["assets"] == report["selected_assets"] == 1
    assert report["modified_assets"] == report["selected_modified_assets"] == 1
    assert report["registration_parent_count"] == 1
    assert report["materialized_assets"] == 2
    assert report["registration_parents"][0]["selected_for_cohort"] is False

    manifest = yaml.safe_load((output / "revised-assets.yaml").read_text(encoding="utf-8"))
    lineage = manifest["train"]["runs"]["revision"]["groups"]["single_palm_allegro"]["right_fixture"]
    assert lineage == {"include_mother": False, "variant_sets": ["joint_limit_revision_v1"]}
    registration_root = output / "generated" / "single_palm_allegro" / "right_fixture"
    assert (registration_root / "hand.urdf").is_file()
    assert (registration_root / "hand.yaml").is_file()
    assert yaml.safe_load((registration_root / "hand.yaml").read_text())["asset_revision"]["registration_only"] is True

    published = load_hand_asset_cohort(output / "training.lock.yaml")
    assert len(published.members) == 1
    assert published.members[0].provenance.asset_role == "variant"
    assert published.members[0].asset_id != original_variant["id"]
    assert all(member.asset_id != yaml.safe_load((registration_root / "hand.yaml").read_text())["id"] for member in published.members)
    assert published.selection["registration_parent_count"] == 1
    assert published.selection["source_mapping"][0]["asset_role"] == "variant"
    assert published.selection["source_mapping"][0]["source_alias"] == "source"
    assert published.selection["source_mapping"][0]["source_row"] == 1

    revised_variant = yaml.safe_load(
        (registration_root / "joint_limit_revision_v1" / published.members[0].asset_id / "hand.yaml").read_text()
    )
    assert _joint_upper(revised_variant["hand_cfg"], "index", "index_j1") == pytest.approx(2.23)
    before = deepcopy(original_variant["hand_cfg"])
    after = deepcopy(revised_variant["hand_cfg"])
    after["fingers"][0]["joints"][1]["limit"]["upper"] = before["fingers"][0]["joints"][1]["limit"]["upper"]
    assert after == before
    assert yaml.safe_load((mother / "hand.yaml").read_text(encoding="utf-8")) == original_mother
    assert yaml.safe_load((variant / "hand.yaml").read_text(encoding="utf-8")) == original_variant


def test_selected_mother_path_keeps_include_mother_and_no_aux_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""旧的 selected mother + variant 路径保持原 manifest 组成与 member 分母。"""

    source_lock, _mother, _variant, _manifest = _source_lock(tmp_path / "selected-mother", selected_mother=True)
    output = tmp_path / "published"
    _run_revision(monkeypatch, source_lock, output, expected_assets=2)

    report = json.loads((output / "revision.json").read_text(encoding="utf-8"))
    assert report["assets"] == report["selected_assets"] == 2
    assert report["modified_assets"] == report["selected_modified_assets"] == 2
    assert report["registration_parent_count"] == 0
    assert report["materialized_assets"] == 2
    manifest = yaml.safe_load((output / "revised-assets.yaml").read_text(encoding="utf-8"))
    lineage = manifest["train"]["runs"]["revision"]["groups"]["single_palm_allegro"]["right_fixture"]
    assert lineage["include_mother"] is True
    assert lineage["variant_sets"] == ["joint_limit_revision_v1"]
    published = load_hand_asset_cohort(output / "training.lock.yaml")
    assert [member.provenance.asset_role for member in published.members] == ["mother", "variant"]
    assert len(published.members) == 2


def test_missing_registration_parent_fails_before_creating_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r"""原 provenance 母体不存在时 fail closed，且不建立新的发布目录。"""

    source_lock, mother, _variant, _manifest = _source_lock(tmp_path / "missing", selected_mother=False)
    mother.rename(mother.with_name("right_fixture_missing"))
    output = tmp_path / "published"
    with pytest.raises((FileNotFoundError, ValueError), match="mother|source topology|dataset"):
        _run_revision(monkeypatch, source_lock, output, expected_assets=1)
    assert not output.exists()


def test_registration_parent_family_mismatch_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r"""registration-only prototype 若 HandCfg family 不是 Allegro 必须拒绝。"""

    source_lock, mother, _variant, _manifest = _source_lock(tmp_path / "wrong-family", selected_mother=False)
    sidecar_path = mother / "hand.yaml"
    sidecar = yaml.safe_load(sidecar_path.read_text(encoding="utf-8"))
    sidecar["hand_cfg"]["family"] = "leap"
    sidecar_path.write_text(yaml.safe_dump(sidecar, sort_keys=False), encoding="utf-8")
    output = tmp_path / "published"
    with pytest.raises(ValueError, match="Allegro-right"):
        _run_revision(monkeypatch, source_lock, output, expected_assets=1)
    assert not output.exists()


def test_existing_output_is_never_overwritten(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r"""发布目标已存在时在任何 source/export 操作前停止，并保留已有证据。"""

    source_lock, _mother, _variant, _manifest = _source_lock(tmp_path / "existing", selected_mother=False)
    output = tmp_path / "published"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _run_revision(monkeypatch, source_lock, output, expected_assets=1)
    assert sentinel.read_text(encoding="utf-8") == "keep"
