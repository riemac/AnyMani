"""历史 generated run 修补 CLI 的审计与删除护栏测试。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

ASSETS_ROOT = Path(__file__).resolve().parents[3]  # `assets/` 子项目根，用于定位真实 CLI 脚本
REPAIR_SCRIPT = ASSETS_ROOT / "scripts" / "repair_generated_run.py"


def _write_fake_run(run_root: Path, *, ambiguous: bool = False) -> None:
    r"""写出一份最小 pre-made 历史 run，覆盖完整、低 DOF 与异族拇指候选。

    Args:
        run_root (Path): 测试 run 根目录。
        ambiguous (bool): 是否额外写出只有 URDF、没有 sidecar 的歧义候选。
    """

    complete = run_root / "single_palm_leap" / "right_t4_i4_m4_r4"
    low_dof = run_root / "single_palm_leap" / "right_t3_i2_m2_r2"
    mixed_both = (
        run_root
        / "mixed"
        / "allegro_single_palm_allegro_leap_thumb_index_middle_ring"
        / "right_leap_t3_allegro_i2_m2_r2"
    )

    # 完整 bundle 必须在 apply 后逐字节保留；测试内容只需满足目录合同。
    (complete / "meshes").mkdir(parents=True)
    (complete / "meshes" / "cs_tip.obj").write_text("# complete mesh\n", encoding="utf-8")
    (complete / "hand.urdf").write_text("<robot name='complete'/>\n", encoding="utf-8")
    (complete / "hand.yaml").write_text("hand_cfg: {}\n", encoding="utf-8")

    # pre-made topology 根可以继续承载独立 post-mutate 时间戳 run；修补工具不得跨 run 清理。
    mutate_run = complete / "2026-06-11_14-20-22"
    mutate_sample = mutate_run / "abcdef12"
    (mutate_run / "meshes").mkdir(parents=True)
    (mutate_run / "meshes" / "shared.obj").write_text("# mutate shared mesh\n", encoding="utf-8")
    (mutate_run / "summary.yaml").write_text("run: {mode: mutate}\n", encoding="utf-8")
    mutate_sample.mkdir(parents=True)
    (mutate_sample / "hand.urdf").write_text("<robot name='mutate'/>\n", encoding="utf-8")
    (mutate_sample / "hand.yaml").write_text("hand_cfg: {}\n", encoding="utf-8")

    # 两个 OBJ-only 候选分别覆盖 low-DOF only 与 palm-thumb mismatch + low-DOF。
    for sample in (low_dof, mixed_both):
        (sample / "meshes").mkdir(parents=True)
        (sample / "meshes" / "cs_tip.obj").write_text("# rejected mesh\n", encoding="utf-8")

    if ambiguous:
        ambiguous_sample = run_root / "single_palm_leap" / "right_t4_i4_m4_r3"
        (ambiguous_sample / "meshes").mkdir(parents=True)
        (ambiguous_sample / "meshes" / "cs_tip.obj").write_text("# ambiguous mesh\n", encoding="utf-8")
        (ambiguous_sample / "hand.urdf").write_text("<robot name='ambiguous'/>\n", encoding="utf-8")

    summary = {
        "run": {
            "timestamp": "2026-06-10_11-30-08",
            "root_dir": str(run_root),
            "mode": "made",
            "artifact_level": "bundle",
        },
        "config": {
            "Validate": {
                "pre_made": {"require_non_thumb_with_min_revolute_dof": 3}
            }
        },
        "stats": {
            "attempted": 3,
            "succeeded": 1,
            "rejected": 2,
            "rejected_by_stage": {"pre_made_validate": 2},
            "by_topology": {"single_palm_leap/right_t4_i4_m4_r4": 1},
            "topology_count": 1,
        },
    }
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "summary.yaml").write_text(
        yaml.safe_dump(summary, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _run_repair(run_root: Path, *extra_args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    r"""通过真实 Python subprocess 调用修补 CLI。"""

    return subprocess.run(
        [sys.executable, str(REPAIR_SCRIPT), "--run-root", str(run_root), *extra_args],
        check=check,
        capture_output=True,
        text=True,
    )


def test_repair_generated_run_dry_run_then_apply_preserves_history_and_complete_bundle(tmp_path):
    r"""dry-run 不改磁盘；apply 删除半成品并给旧 summary 追加可审计维护记录。"""

    run_root = tmp_path / "2026-06-10_11-30-08"
    _write_fake_run(run_root)
    summary_before = (run_root / "summary.yaml").read_bytes()
    complete_urdf = run_root / "single_palm_leap" / "right_t4_i4_m4_r4" / "hand.urdf"
    complete_urdf_before = complete_urdf.read_bytes()
    mutate_summary = complete_urdf.parent / "2026-06-11_14-20-22" / "summary.yaml"
    mutate_summary_before = mutate_summary.read_bytes()

    dry_run = _run_repair(run_root)

    assert "mode=dry-run" in dry_run.stdout
    assert "complete=1 incomplete=2 ambiguous=0" in dry_run.stdout
    assert (run_root / "single_palm_leap" / "right_t3_i2_m2_r2").is_dir()
    assert (run_root / "summary.yaml").read_bytes() == summary_before

    applied = _run_repair(
        run_root,
        "--apply",
        "--expect-complete",
        "1",
        "--expect-incomplete",
        "2",
    )

    assert "mode=apply" in applied.stdout
    assert complete_urdf.read_bytes() == complete_urdf_before
    assert mutate_summary.read_bytes() == mutate_summary_before
    assert not (run_root / "single_palm_leap" / "right_t3_i2_m2_r2").exists()
    assert not (run_root / "mixed" / "allegro_single_palm_allegro_leap_thumb_index_middle_ring").exists()
    assert set(run_root.rglob("*.obj")) == {
        run_root / "single_palm_leap" / "right_t4_i4_m4_r4" / "meshes" / "cs_tip.obj",
        run_root
        / "single_palm_leap"
        / "right_t4_i4_m4_r4"
        / "2026-06-11_14-20-22"
        / "meshes"
        / "shared.obj",
    }

    repaired_summary = yaml.safe_load((run_root / "summary.yaml").read_text(encoding="utf-8"))
    assert repaired_summary["stats"]["attempted"] == 3
    assert repaired_summary["stats"]["succeeded"] == 1
    assert repaired_summary["stats"]["rejected"] == 2
    assert repaired_summary["stats"]["rejected_by_reason"] == {
        "hand.non_thumb_revolute_dof_below_min": 1,
        "hand.non_thumb_revolute_dof_below_min+hand.palm_thumb_family_mismatch": 1,
    }
    assert repaired_summary["maintenance"][-1]["preserved_original_stats"] is True
    assert repaired_summary["maintenance"][-1]["removed_incomplete_bundles"] == 2


def test_repair_generated_run_refuses_ambiguous_candidate_without_mutation(tmp_path):
    r"""只有 URDF 或只有 YAML 的目录不是纯半成品，CLI 必须 fail closed。"""

    run_root = tmp_path / "ambiguous-run"
    _write_fake_run(run_root, ambiguous=True)
    summary_before = (run_root / "summary.yaml").read_bytes()

    completed = _run_repair(
        run_root,
        "--apply",
        "--expect-complete",
        "1",
        "--expect-incomplete",
        "2",
        check=False,
    )

    assert completed.returncode != 0
    assert "ambiguous=1" in completed.stderr
    assert (run_root / "single_palm_leap" / "right_t3_i2_m2_r2").is_dir()
    assert (run_root / "summary.yaml").read_bytes() == summary_before


def test_repair_generated_run_refuses_mismatched_expected_counts(tmp_path):
    r"""显式 apply 计数与审计结果不一致时，任何目录和 summary 都不得改变。"""

    run_root = tmp_path / "count-guard-run"
    _write_fake_run(run_root)
    summary_before = (run_root / "summary.yaml").read_bytes()

    completed = _run_repair(
        run_root,
        "--apply",
        "--expect-complete",
        "1",
        "--expect-incomplete",
        "999",
        check=False,
    )

    assert completed.returncode != 0
    assert "expected incomplete=999, audited incomplete=2" in completed.stderr
    assert (run_root / "single_palm_leap" / "right_t3_i2_m2_r2").is_dir()
    assert (run_root / "summary.yaml").read_bytes() == summary_before
