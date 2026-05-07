r"""独立 post-mutate 统一配置与 runner 回归测试。"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

from anymani.assets.config import asset_gen_cfg as asset_cfg_module
from anymani.assets.generator.hand_generator import HandGenerator, HandGeneratorCfg
from anymani.assets.generator.mutate import HandMutatorCfg, MountPerturbCfg
from anymani.assets.scripts import generate as generate_module
from anymani.assets.scripts import _asset_generate_runner as runner_module
from anymani.assets.validator.hand_rules import HandValidatorCfg


def _single_full_pool() -> dict[str, dict[str, list[str]]]:
    r"""提供一个只有 canonical full topology 的 pre-made pool。"""

    return {
        "single_palm_allegro": {
            "thumb": ["allegro_thumb_full"],
            "index": ["allegro_non_thumb_full"],
            "middle": ["allegro_non_thumb_full"],
            "ring": ["allegro_non_thumb_full"],
        }
    }


def _make_pre_made_topology_dir(tmp_path) -> tuple[Path, str]:
    r"""先生成一个真实的 pre-made topology 目录，供 mutate-only 回放使用。"""

    result = next(
        HandGenerator(
            HandGeneratorCfg(
                mode="made",
                artifact_level="bundle",
                handedness="left",
                hand_presets=["single_palm_allegro"],
                connectivity_presets=_single_full_pool(),
                mixed=False,
                missing=False,
                output_dir=tmp_path,
                max_enumerate=1,
                recolored="anatomy_v1",
            )
        ).generate_batch()
    )
    assert result.sidecar_path is not None
    return result.sidecar_path.parent.parent, result.sidecar_path.parent.name


def _make_fake_source_sample(sample_dir: Path) -> None:
    sample_dir.mkdir(parents=True)
    (sample_dir / "hand.yaml").write_text("hand_cfg: {}\n", encoding="utf-8")
    (sample_dir / "hand.urdf").write_text("<robot name=\"fake\" />\n", encoding="utf-8")


class DemoMountMutatorCfg(HandMutatorCfg):
    r"""测试用 post-mutate cfg：只启用一个 mount perturb term。"""

    mount = MountPerturbCfg(
        disturb_unit="rad",
        self_mode="general",
        pos_range=(0.001, 0.001),
    )


def test_post_mutate_config_is_direct_hand_generator_cfg():
    r"""配置模块中的 `POST_MUTATE_CFG` 应直接是 `HandGeneratorCfg`。"""

    assert isinstance(asset_cfg_module.POST_MUTATE_CFG, HandGeneratorCfg)
    assert asset_cfg_module.POST_MUTATE_CFG.mode == "mutate"
    assert isinstance(asset_cfg_module.POST_MUTATE_CFG.Validate, HandValidatorCfg)
    assert tuple(name for name, _ in asset_cfg_module.POST_MUTATE_CFG.Mutate.ordered_terms()) == (
        "link_scale",
        "mount_perturb",
        "limit_tweak",
        "tip_replace",
    )


def test_post_mutate_runner_resolves_direct_sample_path(tmp_path):
    source_sample_dir = tmp_path / "right_t4_i4_m4_r4" / "f5d8c069"
    _make_fake_source_sample(source_sample_dir)

    resolved_dir = runner_module.resolve_source_premade_sample_dir(
        source_sample_dir,
        sample_id="f5d8c069",
    )

    assert resolved_dir == source_sample_dir


def test_post_mutate_runner_resolves_topology_path_with_sample_id(tmp_path):
    topology_dir = tmp_path / "right_t4_i4_m4_r4"
    _make_fake_source_sample(topology_dir / "f5d8c069")
    _make_fake_source_sample(topology_dir / "abcd1234")

    resolved_dir = runner_module.resolve_source_premade_sample_dir(
        topology_dir,
        sample_id="f5d8c069",
    )

    assert resolved_dir == topology_dir / "f5d8c069"


def test_post_mutate_runner_plans_nested_and_sibling_run_dirs(tmp_path):
    source_sample_dir = tmp_path / "right_t4_i4_m4_r4" / "f5d8c069"

    nested_dir = runner_module.planned_post_mutate_topology_dir(
        source_sample_dir=source_sample_dir,
        layout="nested",
        run_name="try_small",
    )
    sibling_dir = runner_module.planned_post_mutate_topology_dir(
        source_sample_dir=source_sample_dir,
        layout="sibling",
        run_name="try_small",
    )

    assert nested_dir == source_sample_dir / "try_small"
    assert sibling_dir == source_sample_dir.parent / "f5d8c069_post_mutate" / "try_small"


def test_post_mutate_runner_prepare_overwrite_copies_source_without_run_recursion(tmp_path):
    source_sample_dir = tmp_path / "topology" / "f5d8c069"
    _make_fake_source_sample(source_sample_dir)

    stale_run_dir = source_sample_dir / "try_001"
    stale_run_dir.mkdir(parents=True)
    (stale_run_dir / "stale.txt").write_text("old failed attempt\n", encoding="utf-8")

    preserved_run_dir = source_sample_dir / "try_keep"
    preserved_run_dir.mkdir(parents=True)
    (preserved_run_dir / "keep.txt").write_text("keep me\n", encoding="utf-8")
    legacy_post_mutate_dir = source_sample_dir / "post_mutate"
    legacy_post_mutate_dir.mkdir()
    (legacy_post_mutate_dir / "legacy.txt").write_text("old layout\n", encoding="utf-8")

    run_dir = runner_module.prepare_post_mutate_source_topology(
        source_sample_dir=source_sample_dir,
        layout="nested",
        run_name="try_001",
        run_policy="overwrite",
    )

    staged_sample_dir = run_dir / "f5d8c069"
    assert run_dir == stale_run_dir
    assert not (run_dir / "stale.txt").exists()
    assert (staged_sample_dir / "hand.yaml").is_file()
    assert not (staged_sample_dir / "post_mutate").exists()
    assert not (staged_sample_dir / "try_keep").exists()
    assert (source_sample_dir / "hand.yaml").is_file()
    assert (preserved_run_dir / "keep.txt").is_file()
    assert (legacy_post_mutate_dir / "legacy.txt").is_file()


def test_post_mutate_runner_prepare_new_allocates_suffix_without_touching_existing_runs(tmp_path):
    source_sample_dir = tmp_path / "topology" / "f5d8c069"
    _make_fake_source_sample(source_sample_dir)

    for run_name in ("try_001", "try_001_01"):
        occupied_dir = source_sample_dir / run_name
        occupied_dir.mkdir(parents=True)
        (occupied_dir / "occupied.txt").write_text(run_name, encoding="utf-8")

    run_dir = runner_module.prepare_post_mutate_source_topology(
        source_sample_dir=source_sample_dir,
        layout="nested",
        run_name="try_001",
        run_policy="new",
    )

    assert run_dir == source_sample_dir / "try_001_02"
    assert (run_dir / "f5d8c069" / "hand.yaml").is_file()
    assert (source_sample_dir / "try_001" / "occupied.txt").is_file()
    assert (source_sample_dir / "try_001_01" / "occupied.txt").is_file()


def test_post_mutate_nested_staging_keeps_original_sample_after_generator_run(tmp_path):
    topology_dir, original_sample_name = _make_pre_made_topology_dir(tmp_path)
    source_sample_dir = topology_dir / original_sample_name
    run_dir = runner_module.prepare_post_mutate_source_topology(
        source_sample_dir=source_sample_dir,
        layout="nested",
        run_name="try_nested",
        run_policy="overwrite",
    )

    mutate_cfg = HandGeneratorCfg(
        mode="mutate",
        artifact_level="bundle",
        source_topology_dir=run_dir,
        output_dir=run_dir.parent,
        n_samples=1,
        Mutate=DemoMountMutatorCfg(),
        Validate=None,
        recolored="anatomy_v1",
    )

    results = list(HandGenerator(mutate_cfg).generate_batch())

    assert len(results) == 1
    assert source_sample_dir.is_dir()
    assert (source_sample_dir / "hand.yaml").is_file()
    assert (run_dir / f"{original_sample_name}_origin" / "hand.yaml").is_file()
    assert not (run_dir / original_sample_name).exists()
    assert (run_dir / "summary.yaml").is_file()
    assert results[0].metadata["source_origin_sample_id"] == original_sample_name


def test_independent_post_mutate_renames_origin_and_writes_sibling_variants(tmp_path):
    topology_dir, original_sample_name = _make_pre_made_topology_dir(tmp_path)
    mutate_cfg = HandGeneratorCfg(
        mode="mutate",
        artifact_level="bundle",
        source_topology_dir=topology_dir,
        output_dir=tmp_path,
        n_samples=2,
        Mutate=DemoMountMutatorCfg(),
        Validate=None,
        recolored="anatomy_v1",
    )

    results = list(HandGenerator(mutate_cfg).generate_batch())

    assert len(results) == 2

    origin_dir = topology_dir / f"{original_sample_name}_origin"
    assert origin_dir.is_dir()
    assert not (topology_dir / original_sample_name).exists()
    assert (topology_dir / "summary.yaml").is_file()

    sample_dirs = sorted(path for path in topology_dir.iterdir() if path.is_dir() and (path / "hand.yaml").is_file())
    assert len(sample_dirs) == 3
    assert sum(path.name.endswith("_origin") for path in sample_dirs) == 1

    new_variant_dirs = [path for path in sample_dirs if not path.name.endswith("_origin")]
    assert len(new_variant_dirs) == 2
    assert all((path / "hand.urdf").is_file() for path in new_variant_dirs)
    assert all(result.metadata["source_origin_sample_id"] == original_sample_name for result in results)


def test_unified_generate_runner_accepts_post_mutate_cli(monkeypatch, tmp_path):
    r"""统一 runner 应支持通过 CLI 选择 post-mutate 阶段。"""

    topology_dir, original_sample_name = _make_pre_made_topology_dir(tmp_path)
    custom_module = import_module("anymani.assets.config.asset_gen_cfg")
    monkeypatch.setattr(
        custom_module,
        "POST_MUTATE_CFG",
        HandGeneratorCfg(
            mode="mutate",
            artifact_level="hand_cfg",
            source_topology_dir=Path("__placeholder__"),
            output_dir=Path("__placeholder__"),
            n_samples=1,
            Mutate=DemoMountMutatorCfg(),
            Validate=None,
            recolored="anatomy_v1",
        ),
    )
    monkeypatch.setattr(custom_module, "POST_MUTATE_SOURCE_PREMADE_PATH", topology_dir)
    monkeypatch.setattr(custom_module, "POST_MUTATE_SOURCE_PREMADE_SAMPLE_ID", original_sample_name)
    monkeypatch.setattr(custom_module, "POST_MUTATE_RUN_NAME", "try_cli")
    monkeypatch.setattr(custom_module, "POST_MUTATE_RUN_POLICY", "overwrite")
    monkeypatch.setattr(custom_module, "POST_MUTATE_PRINT_RESULT_LIMIT", 0)
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate.py",
            "--stage",
            "post-mutate",
            "--config-module",
            "anymani.assets.config.asset_gen_cfg",
        ],
    )

    exit_code = generate_module.main()

    assert exit_code == 0
