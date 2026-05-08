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
    return result.sidecar_path.parent, result.metadata["id"]


def _make_fake_topology(topology_dir: Path, *, sample_id: str = "fake1234") -> None:
    topology_dir.mkdir(parents=True)
    (topology_dir / "hand.yaml").write_text(f"id: {sample_id}\nhand_cfg: {{}}\n", encoding="utf-8")
    (topology_dir / "hand.urdf").write_text("<robot name=\"fake\" />\n", encoding="utf-8")


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


def test_post_mutate_runner_resolves_topology_root_path(tmp_path):
    topology_dir = tmp_path / "right_t4_i4_m4_r4"
    _make_fake_topology(topology_dir)

    resolved_dir = runner_module.resolve_source_topology_dir(topology_dir)

    assert resolved_dir == topology_dir


def test_post_mutate_runner_plans_timestamp_run_under_topology_root(tmp_path):
    topology_dir = tmp_path / "right_t4_i4_m4_r4"
    _make_fake_topology(topology_dir)

    run_dir = runner_module.plan_post_mutate_run_dir(topology_dir)

    assert run_dir.parent == topology_dir
    assert run_dir.name


def test_post_mutate_runner_prepare_cfg_keeps_topology_root_and_previews_run_dir(tmp_path):
    topology_dir = tmp_path / "right_t4_i4_m4_r4"
    _make_fake_topology(topology_dir)

    run_cfg = HandGeneratorCfg(
        mode="mutate",
        artifact_level="hand_cfg",
        source_topology_dir=Path("__placeholder__"),
        output_dir=Path("__placeholder__"),
        n_samples=1,
        Mutate=DemoMountMutatorCfg(),
        Validate=None,
    )

    prepared_cfg, resolved_topology_dir, planned_run_dir = runner_module.prepare_post_mutate_run_cfg(
        run_cfg,
        source_path=topology_dir,
    )

    assert prepared_cfg.source_topology_dir == topology_dir
    assert resolved_topology_dir == topology_dir
    assert planned_run_dir.parent == topology_dir


def test_independent_post_mutate_restores_from_topology_root_and_writes_timestamp_hash_runs(tmp_path):
    topology_dir, original_sample_id = _make_pre_made_topology_dir(tmp_path)
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
    assert all(result.metadata["source_origin_sample_id"] == original_sample_id for result in results)
    assert all(result.metadata["source_origin_topology_dir"] == str(topology_dir) for result in results)

    mutate_run_dirs = [path for path in topology_dir.iterdir() if path.is_dir()]
    assert len(mutate_run_dirs) == 1

    mutate_run_dir = mutate_run_dirs[0]
    assert (mutate_run_dir / "summary.yaml").is_file()
    assert all(result.urdf_path is not None and result.urdf_path.parent.parent == mutate_run_dir for result in results)
    assert all((mutate_run_dir / result.metadata["id"] / "hand.yaml").is_file() for result in results)
    assert (topology_dir / "hand.yaml").is_file()
    assert (topology_dir / "hand.urdf").is_file()


def test_unified_generate_runner_accepts_post_mutate_cli(monkeypatch, tmp_path):
    r"""统一 runner 应支持通过 CLI 选择 post-mutate 阶段。"""

    topology_dir, _original_sample_id = _make_pre_made_topology_dir(tmp_path)
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
    monkeypatch.setattr(custom_module, "POST_MUTATE_SOURCE_TOPOLOGY_PATH", topology_dir)
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
