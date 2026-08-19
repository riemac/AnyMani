r"""HandGenerator 多 mother post-mutate façade、seed 与并行调度合同。"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from anymani.assets.generator.hand_generator import (
    HandGenerator,
    HandGeneratorCfg,
    PostMutateSourceCfg,
)
from anymani.assets.generator.mutate import HandMutatorCfg, MountPerturbCfg


class _BatchMountMutatorCfg(HandMutatorCfg):
    r"""测试用确定性小幅 mount 扰动；module-level class 可安全传入进程 worker。"""

    mount_perturb = MountPerturbCfg(
        self_mode="general",
        pos_radius=0.001,
        rot_radius=0.01,
        distrib="uniform",
        boundary_policy="clip",
    )


@pytest.mark.parametrize("parallel", [False, True])
def test_generate_variant_sets_dispatches_multiple_sources_with_per_task_counts_and_seeds(
    tmp_path: Path,
    parallel: bool,
) -> None:
    r"""一个 façade 应为每个 mother 创建独立 run，并按 task 顺序返回轻量报告。"""

    sources = _make_two_mothers(tmp_path / "premade")
    cfg = HandGeneratorCfg(
        mode="mutate",
        artifact_level="bundle",
        source_topology_dir=None,
        post_mutate_sources=[
            PostMutateSourceCfg(task_id="train:left", source_topology_dir=sources[0], n_samples=2, seed=101),
            PostMutateSourceCfg(task_id="train:right", source_topology_dir=sources[1], n_samples=1, seed=202),
        ],
        post_mutate_parallel=parallel,
        post_mutate_parallel_workers=2,
        Mutate=_BatchMountMutatorCfg(),
        Validate=None,
        Physics=None,
    )

    reports = list(HandGenerator(cfg).generate_variant_sets())

    assert [report.task_id for report in reports] == ["train:left", "train:right"]
    assert [report.planned_variants for report in reports] == [2, 1]
    assert [report.successful_variants for report in reports] == [2, 1]
    assert all(report.shortfall == 0 and report.run_dir.is_dir() for report in reports)
    assert all((report.run_dir / "summary.yaml").is_file() for report in reports)
    assert all(path.is_file() for report in reports for path in report.sidecar_paths)

    summaries = [yaml.safe_load((report.run_dir / "summary.yaml").read_text(encoding="utf-8")) for report in reports]
    assert [summary["config"]["post_mutate_seed"] for summary in summaries] == [101, 202]
    assert [summary["config"]["source_topology_dir"] for summary in summaries] == [str(path) for path in sources]
    assert all(summary["config"]["post_mutate_sources"] == [] for summary in summaries)


def test_post_mutate_source_modes_are_mutually_exclusive(tmp_path: Path) -> None:
    r"""single-source 调试与 multi-source batch 不得同时声明，避免重复写同一 mother。"""

    source = _make_two_mothers(tmp_path / "premade")[0]
    try:
        HandGeneratorCfg(
            mode="mutate",
            source_topology_dir=source,
            post_mutate_sources=[
                PostMutateSourceCfg(task_id="duplicate", source_topology_dir=source, n_samples=1, seed=1)
            ],
            Mutate=_BatchMountMutatorCfg(),
        )
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("mutate cfg accepted simultaneous single- and multi-source inputs")


def _make_two_mothers(output_dir: Path) -> tuple[Path, Path]:
    r"""生成同一 canonical topology 的 left/right pre-made mother bundles。"""

    connectivity = {
        "single_palm_allegro": {
            "thumb": ["allegro_thumb_full"],
            "index": ["allegro_non_thumb_full"],
            "middle": ["allegro_non_thumb_full"],
            "ring": ["allegro_non_thumb_full"],
        }
    }
    results = list(
        HandGenerator(
            HandGeneratorCfg(
                mode="made",
                artifact_level="bundle",
                output_dir=output_dir,
                handedness="all",
                hand_presets=["single_palm_allegro"],
                connectivity_presets=connectivity,
                mixed=False,
                missing=False,
                max_enumerate=2,
                premade_parallel=False,
                Validate=None,
                Physics=None,
            )
        ).generate_batch()
    )
    roots = tuple(result.sidecar_path.parent for result in results if result.sidecar_path is not None)
    assert len(roots) == 2
    return roots
