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
from anymani.assets.generator.mutate import HandMutatorCfg, LinkScaleCfg, MountPerturbCfg
from anymani.assets.geometry_identity import geometry_fingerprint_from_sidecar
from anymani.assets.validator.hand_rules import HandValidatorCfg


class _BatchMountMutatorCfg(HandMutatorCfg):
    r"""测试用确定性小幅 mount 扰动；module-level class 可安全传入进程 worker。"""

    mount_perturb = MountPerturbCfg(
        self_mode="general",
        pos_radius=0.001,
        rot_radius=0.01,
        distrib="uniform",
        boundary_policy="clip",
    )


class _IdentityLinkScaleMutatorCfg(HandMutatorCfg):
    r"""进程 worker 可 pickle 的确定性 mother-geometry no-op。"""

    link_scale = LinkScaleCfg(
        self_mode="identity",
        scale_type="rel",
        link_scale=(1.0, 1.0),
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


@pytest.mark.parametrize("parallel", [False, True])
def test_generate_variant_sets_preserves_per_mother_uniqueness_in_serial_and_parallel(
    tmp_path: Path,
    parallel: bool,
) -> None:
    r"""mother-level 进程并行不共享 registry，但每个 worker 都拒绝自己的 mother no-op。"""

    sources = _make_two_mothers(tmp_path / "premade")
    cfg = HandGeneratorCfg(
        mode="mutate",
        artifact_level="bundle",
        source_topology_dir=None,
        post_mutate_sources=[
            PostMutateSourceCfg(task_id="strict:left", source_topology_dir=sources[0], n_samples=1, seed=101),
            PostMutateSourceCfg(task_id="strict:right", source_topology_dir=sources[1], n_samples=1, seed=202),
        ],
        post_mutate_parallel=parallel,
        post_mutate_parallel_workers=2,
        post_mutate_attempts_per_variant=2,
        post_mutate_require_unique_geometry=True,
        Mutate=_IdentityLinkScaleMutatorCfg(),
        Validate=None,
        Physics=None,
    )

    reports = list(HandGenerator(cfg).generate_variant_sets())
    summaries = [yaml.safe_load((report.run_dir / "summary.yaml").read_text(encoding="utf-8")) for report in reports]

    assert [report.task_id for report in reports] == ["strict:left", "strict:right"]
    assert all(report.successful_variants == 0 and report.shortfall == 1 for report in reports)
    assert all(report.sidecar_paths == () and report.urdf_paths == () for report in reports)
    assert all(
        summary["stats"]["rejected_by_reason"] == {"post_mutate.duplicate_mother_geometry": 2}
        for summary in summaries
    )


def test_central_gpu_variant_sets_match_local_scalar_and_recycle_cpu_workers(tmp_path: Path) -> None:
    r"""固定 seed 下 central batch 与 scalar 接纳序列一致，且每个 mother 使用无 CUDA 的新 worker。"""

    sources = _make_two_mothers(tmp_path / "premade")
    tasks = [
        PostMutateSourceCfg(task_id="parity:left", source_topology_dir=sources[0], n_samples=1, seed=101),
        PostMutateSourceCfg(task_id="parity:right", source_topology_dir=sources[1], n_samples=1, seed=202),
    ]
    validator = HandValidatorCfg(
        post_mutate=HandValidatorCfg.PostMutateCfg(
            dof_min=None,
            finger_count_min=None,
            finger_count_max=None,
            require_thumb=False,
            require_non_thumb_with_min_revolute_dof=None,
            check_finger_spacing=True,
            min_finger_spacing=-1.0,
            sdf_device="cuda",
            sdf_mesh_backend="warp",
        )
    )
    common = dict(
        mode="mutate",
        artifact_level="bundle",
        source_topology_dir=None,
        post_mutate_sources=tasks,
        post_mutate_parallel_workers=2,
        Mutate=_BatchMountMutatorCfg(),
        Validate=validator,
        Physics=None,
    )

    central_reports = list(
        HandGenerator(
            HandGeneratorCfg(
                **common,
                post_mutate_parallel=True,
                post_mutate_sdf_execution="central_gpu_batch",
            )
        ).generate_variant_sets()
    )
    local_reports = list(
        HandGenerator(
            HandGeneratorCfg(
                **common,
                post_mutate_parallel=False,
                post_mutate_sdf_execution="local",
            )
        ).generate_variant_sets()
    )

    central_fingerprints = [
        [geometry_fingerprint_from_sidecar(path) for path in report.sidecar_paths]
        for report in central_reports
    ]
    local_fingerprints = [
        [geometry_fingerprint_from_sidecar(path) for path in report.sidecar_paths]
        for report in local_reports
    ]
    assert central_fingerprints == local_fingerprints
    assert all(report.successful_variants == 1 for report in central_reports)
    assert len({report.worker_pid for report in central_reports}) == len(tasks)
    assert all(not report.worker_cuda_initialized for report in central_reports)
    assert len({report.sdf_service_pid for report in central_reports}) == 1
    assert central_reports[0].sdf_service_pid > 0


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
