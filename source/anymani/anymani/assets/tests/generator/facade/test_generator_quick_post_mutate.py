r"""独立 post-mutate 统一配置与 runner 回归测试。"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import yaml

from anymani.assets.config import asset_gen_cfg as asset_cfg_module
from anymani.assets.generator.hand_generator import HandGenerator, HandGeneratorCfg
from anymani.assets.generator.mutate import HandMutatorCfg, LimitTweakCfg, MountPerturbCfg, TipReplaceCfg
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


def _assert_urdf_has_no_mount_helper_topology(urdf_path: Path) -> None:
    r"""断言 URDF 文本里不再残留 `mount_link` / `mount_joint` 辅助拓扑。

    这里检查的是整手正式导出的 contract，而不是 palm preview 里的 marker 机制。
    一旦这里再次出现 `*_mount_link`，说明整手导出退回了旧的非官方挂载语义。
    """

    urdf_text = urdf_path.read_text(encoding="utf-8")
    assert "_mount_link" not in urdf_text
    assert "_mount_joint" not in urdf_text


class DemoMountMutatorCfg(HandMutatorCfg):
    r"""测试用 post-mutate cfg：只启用一个 mount perturb term。"""

    mount = MountPerturbCfg(
        self_mode="general",
        pos_radius=0.001,
    )


class DemoQuotaMountMutatorCfg(HandMutatorCfg):
    r"""测试 accepted/output quota 的 post-mutate cfg。"""

    mount_perturb = MountPerturbCfg(
        self_mode={"identity": 0.5, "general": 0.5},
        pos_radius=0.001,
    )


class DemoQuotaLimitMutatorCfg(HandMutatorCfg):
    r"""测试 `limit_tweak.self_mode` accepted/output quota 的 post-mutate cfg。"""

    limit_tweak = LimitTweakCfg(
        self_mode={"identity": 0.5, "homologous_non_thumb": 0.5},
        disturb_object="independent",
        disturb_type="add",
        joint_range=(-0.02, 0.02),
    )


class DemoQuotaMountAndLimitMutatorCfg(HandMutatorCfg):
    r"""测试多 mode-term 同时存在时的边缘 accepted/output 统计。"""

    mount_perturb = MountPerturbCfg(
        self_mode={"identity": 0.5, "general": 0.5},
        pos_radius=0.001,
    )
    limit_tweak = LimitTweakCfg(
        self_mode={"identity": 0.5, "homologous_non_thumb": 0.5},
        disturb_object="independent",
        disturb_type="add",
        joint_range=(-0.02, 0.02),
    )


class DemoQuotaTipReplaceMutatorCfg(HandMutatorCfg):
    r"""测试 `tip_replace.self_mode` accepted/output quota 与 tip_type proposal 统计。"""

    tip_replace = TipReplaceCfg(
        self_mode={"identity": 0.5, "same": 0.5},
        tip_range={"cs": 0.5, "round": 0.5},
        scale=(1.0, 1.0),
    )


class DemoMeshOnlyTipReplaceMutatorCfg(HandMutatorCfg):
    r"""测试导出路径时使用的确定性 custom-mesh tip_replace cfg。"""

    tip_replace = TipReplaceCfg(
        self_mode="same",
        tip_range=["round"],
        scale=(1.0, 1.0),
    )


def test_post_mutate_config_is_direct_hand_generator_cfg():
    r"""配置模块中的 `POST_MUTATE_CFG` 应直接是 `HandGeneratorCfg`。"""

    assert isinstance(asset_cfg_module.POST_MUTATE_CFG, HandGeneratorCfg)
    assert asset_cfg_module.POST_MUTATE_CFG.mode == "mutate"
    assert isinstance(asset_cfg_module.POST_MUTATE_CFG.Validate, HandValidatorCfg)
    assert tuple(name for name, _ in asset_cfg_module.POST_MUTATE_CFG.Mutate.ordered_terms()) == ("tip_replace",)


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
    assert all(
        result.metadata["post_mutate_samples"]["mount_perturb"]["resolved_self_mode"] == "general"
        for result in results
    )

    mutate_run_dirs = [path for path in topology_dir.iterdir() if path.is_dir()]
    assert len(mutate_run_dirs) == 1

    mutate_run_dir = mutate_run_dirs[0]
    assert (mutate_run_dir / "summary.yaml").is_file()
    assert all(result.urdf_path is not None and result.urdf_path.parent.parent == mutate_run_dir for result in results)
    assert all((mutate_run_dir / result.metadata["id"] / "hand.yaml").is_file() for result in results)
    assert (topology_dir / "hand.yaml").is_file()
    assert (topology_dir / "hand.urdf").is_file()

    _assert_urdf_has_no_mount_helper_topology(topology_dir / "hand.urdf")  # pre-made topology 根导出也必须遵守同一语义
    for result in results:
        _assert_urdf_has_no_mount_helper_topology(result.urdf_path)  # mutate-only 派生样本不允许回退到旧 mount helper 拓扑


def test_post_mutate_run_reuses_shared_mesh_directory_for_custom_tip_outputs(tmp_path):
    r"""mutate-only run 应在 run 根目录共享 `meshes/`，而不是每个样本各拷一份。"""

    topology_dir, _original_sample_id = _make_pre_made_topology_dir(tmp_path)
    mutate_cfg = HandGeneratorCfg(
        mode="mutate",
        artifact_level="bundle",
        source_topology_dir=topology_dir,
        output_dir=tmp_path,
        n_samples=2,
        Mutate=DemoMeshOnlyTipReplaceMutatorCfg(),
        Validate=None,
    )

    results = list(HandGenerator(mutate_cfg).generate_batch())
    mutate_run_dir = next(path for path in topology_dir.iterdir() if path.is_dir())
    meshes_dir = mutate_run_dir / "meshes"

    assert meshes_dir.is_dir()
    assert any(meshes_dir.iterdir())
    assert all(not (result.urdf_path.parent / "meshes").exists() for result in results if result.urdf_path is not None)
    assert all(result.urdf_path is not None for result in results)
    for result in results:
        urdf_text = result.urdf_path.read_text(encoding="utf-8")
        assert "<mesh " in urdf_text
        assert "../meshes/" in urdf_text or "meshes/" in urdf_text
        assert "/home/hac/isaac/AnyMani/source/anymani/anymani/assets/custom/tips/" not in urdf_text


def test_post_mutate_self_mode_probability_is_accepted_output_quota(tmp_path):
    r"""self_mode dict 应控制 accepted/output 分布，而不是 proposal prior。"""

    topology_dir, _original_sample_id = _make_pre_made_topology_dir(tmp_path)
    mutate_cfg = HandGeneratorCfg(
        mode="mutate",
        artifact_level="bundle",
        source_topology_dir=topology_dir,
        output_dir=tmp_path,
        n_samples=4,
        Mutate=DemoQuotaMountMutatorCfg(),
        Validate=None,
    )

    results = list(HandGenerator(mutate_cfg).generate_batch())

    modes = [
        result.metadata["post_mutate_samples"]["mount_perturb"]["resolved_self_mode"]
        for result in results
    ]
    mutate_run_dir = next(path for path in topology_dir.iterdir() if path.is_dir())
    summary = yaml.safe_load((mutate_run_dir / "summary.yaml").read_text(encoding="utf-8"))

    assert modes.count("identity") == 2
    assert modes.count("general") == 2
    assert summary["post_mutate_mode_stats"]["mount_perturb"]["identity"]["target_quota"] == 2
    assert summary["post_mutate_mode_stats"]["mount_perturb"]["identity"]["accepted"] == 2
    assert summary["post_mutate_mode_stats"]["mount_perturb"]["general"]["target_quota"] == 2
    assert summary["post_mutate_mode_stats"]["mount_perturb"]["general"]["accepted"] == 2


def test_limit_tweak_self_mode_probability_is_accepted_output_quota(tmp_path):
    r"""`limit_tweak.self_mode` dict 也应控制 accepted/output 分布。"""

    topology_dir, _original_sample_id = _make_pre_made_topology_dir(tmp_path)
    mutate_cfg = HandGeneratorCfg(
        mode="mutate",
        artifact_level="bundle",
        source_topology_dir=topology_dir,
        output_dir=tmp_path,
        n_samples=4,
        Mutate=DemoQuotaLimitMutatorCfg(),
        Validate=None,
    )

    results = list(HandGenerator(mutate_cfg).generate_batch())

    modes = [
        result.metadata["post_mutate_samples"]["limit_tweak"]["resolved_self_mode"]
        for result in results
    ]
    mutate_run_dir = next(path for path in topology_dir.iterdir() if path.is_dir())
    summary = yaml.safe_load((mutate_run_dir / "summary.yaml").read_text(encoding="utf-8"))

    assert modes.count("identity") == 2
    assert modes.count("homologous_non_thumb") == 2
    assert summary["post_mutate_mode_stats"]["limit_tweak"]["identity"]["target_quota"] == 2
    assert summary["post_mutate_mode_stats"]["limit_tweak"]["identity"]["accepted"] == 2
    assert summary["post_mutate_mode_stats"]["limit_tweak"]["homologous_non_thumb"]["target_quota"] == 2
    assert summary["post_mutate_mode_stats"]["limit_tweak"]["homologous_non_thumb"]["accepted"] == 2


def test_multiple_mode_terms_track_marginal_accepted_output_quota(tmp_path):
    r"""多个 mode-term 同时存在时，应各自满足自己的边缘 accepted/output 分布。"""

    topology_dir, _original_sample_id = _make_pre_made_topology_dir(tmp_path)
    mutate_cfg = HandGeneratorCfg(
        mode="mutate",
        artifact_level="bundle",
        source_topology_dir=topology_dir,
        output_dir=tmp_path,
        n_samples=4,
        Mutate=DemoQuotaMountAndLimitMutatorCfg(),
        Validate=None,
    )

    results = list(HandGenerator(mutate_cfg).generate_batch())
    mount_modes = [
        result.metadata["post_mutate_samples"]["mount_perturb"]["resolved_self_mode"]
        for result in results
    ]
    limit_modes = [
        result.metadata["post_mutate_samples"]["limit_tweak"]["resolved_self_mode"]
        for result in results
    ]
    mutate_run_dir = next(path for path in topology_dir.iterdir() if path.is_dir())
    summary = yaml.safe_load((mutate_run_dir / "summary.yaml").read_text(encoding="utf-8"))

    assert mount_modes.count("identity") == 2
    assert mount_modes.count("general") == 2
    assert limit_modes.count("identity") == 2
    assert limit_modes.count("homologous_non_thumb") == 2
    assert summary["post_mutate_mode_stats"]["mount_perturb"]["identity"]["accepted"] == 2
    assert summary["post_mutate_mode_stats"]["mount_perturb"]["general"]["accepted"] == 2
    assert summary["post_mutate_mode_stats"]["limit_tweak"]["identity"]["accepted"] == 2
    assert summary["post_mutate_mode_stats"]["limit_tweak"]["homologous_non_thumb"]["accepted"] == 2


def test_tip_replace_self_mode_probability_is_accepted_output_quota(tmp_path):
    r"""`tip_replace.self_mode` dict 应控制 accepted/output mode 分布。"""

    topology_dir, _original_sample_id = _make_pre_made_topology_dir(tmp_path)
    mutate_cfg = HandGeneratorCfg(
        mode="mutate",
        artifact_level="bundle",
        source_topology_dir=topology_dir,
        output_dir=tmp_path,
        n_samples=4,
        Mutate=DemoQuotaTipReplaceMutatorCfg(),
        Validate=None,
    )

    results = list(HandGenerator(mutate_cfg).generate_batch())

    modes = [
        result.metadata["post_mutate_samples"]["tip_replace"]["resolved_self_mode"]
        for result in results
    ]
    mutate_run_dir = next(path for path in topology_dir.iterdir() if path.is_dir())
    summary = yaml.safe_load((mutate_run_dir / "summary.yaml").read_text(encoding="utf-8"))

    assert modes.count("identity") == 2
    assert modes.count("same") == 2
    assert summary["post_mutate_mode_stats"]["tip_replace"]["identity"]["target_quota"] == 2
    assert summary["post_mutate_mode_stats"]["tip_replace"]["identity"]["accepted"] == 2
    assert summary["post_mutate_mode_stats"]["tip_replace"]["same"]["target_quota"] == 2
    assert summary["post_mutate_mode_stats"]["tip_replace"]["same"]["accepted"] == 2
    assert sum(summary["post_mutate_tip_type_stats"]["proposed"].values()) == 8
    assert sum(summary["post_mutate_tip_type_stats"]["accepted"].values()) == 8


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
