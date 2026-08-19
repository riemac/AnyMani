r"""独立 post-mutate 统一配置与 runner 回归测试。"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import yaml
from anymani.assets.config import asset_gen_cfg as asset_cfg_module
from anymani.assets.generator.hand_generator import HandGenerator, HandGeneratorCfg
from anymani.assets.generator.mutate import HandMutatorCfg, LimitTweakCfg, LinkScaleCfg, MountPerturbCfg, TipReplaceCfg
from anymani.assets.geometry_identity import geometry_fingerprint_from_sidecar
from anymani.assets.scripts import _asset_generate_runner as runner_module
from anymani.assets.scripts import generate as generate_module
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
                recolored="anatomy_soft_v1",
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


def _mutate_run_dirs(topology_dir: Path) -> list[Path]:
    r"""列出 topology 根下真正的 post-mutate timestamp run 目录。

    `cs` single-mesh contract 后，pre-made topology 根会合法持有共享 `meshes/` 目录；
    它不是 mutate run，也不应被这些目录结构测试当作样本运行目录。
    """

    return [path for path in topology_dir.iterdir() if path.is_dir() and path.name != "meshes"]


class DemoMountMutatorCfg(HandMutatorCfg):
    r"""测试用 post-mutate cfg：只启用一个 mount perturb term。"""

    mount = MountPerturbCfg(
        self_mode="general",
        pos_radius=0.001,
    )


class DemoProposalMountMutatorCfg(HandMutatorCfg):
    r"""测试 `self_mode` proposal 概率的 post-mutate cfg。"""

    mount_perturb = MountPerturbCfg(
        self_mode={"identity": 0.5, "general": 0.5},
        pos_radius=0.001,
    )


class DemoProposalLimitMutatorCfg(HandMutatorCfg):
    r"""测试 `limit_tweak.self_mode` proposal 概率的 post-mutate cfg。"""

    limit_tweak = LimitTweakCfg(
        self_mode={"identity": 0.5, "homologous_non_thumb": 0.5},
        disturb_object="independent",
        disturb_type="add",
        joint_range=(-0.02, 0.02),
    )


class DemoProposalMountAndLimitMutatorCfg(HandMutatorCfg):
    r"""测试多个 mode term 在每次候选中独立联合抽样。"""

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


class DemoProposalTipReplaceMutatorCfg(HandMutatorCfg):
    r"""测试 `tip_replace.self_mode` 与 tip type 的 proposal 统计。"""

    tip_replace = TipReplaceCfg(
        self_mode={"identity": 0.5, "same": 0.5},
        tip_range={"cs": 0.5, "round": 0.5},
        scale=(1.0, 1.0),
    )


class DemoProposalLinkScaleMutatorCfg(HandMutatorCfg):
    r"""测试 `link_scale.self_mode` proposal 统计。"""

    link_scale = LinkScaleCfg(
        self_mode={"identity": 0.5, "only_length": 0.5},
        scale_type="rel",
        link_scale=(1.0, 1.0, 1.5, 1.5, 0.5, 0.5),
        distrib="uniform",
    )


class DemoMeshOnlyTipReplaceMutatorCfg(HandMutatorCfg):
    r"""测试导出路径时使用的确定性 custom-mesh tip_replace cfg。"""

    tip_replace = TipReplaceCfg(
        self_mode="same",
        tip_range=["round"],
        scale=(1.0, 1.0),
    )


class DemoIdentityLinkScaleMutatorCfg(HandMutatorCfg):
    r"""显式生成与 mother 静态几何相同的 identity proposal。"""

    link_scale = LinkScaleCfg(
        self_mode="identity",
        scale_type="rel",
        link_scale=(1.0, 1.0),
    )


class DemoFixedLinkScaleMutatorCfg(HandMutatorCfg):
    r"""每次都生成同一份非 mother 几何，用于证伪 variant-set 内重复。"""

    link_scale = LinkScaleCfg(
        self_mode="only_length",
        scale_type="rel",
        link_scale=(1.1, 1.1),
        distrib="uniform",
    )


class DemoFixedLimitMutatorCfg(HandMutatorCfg):
    r"""只改变 joint limits；静态 geometry fingerprint 应仍与 mother 相同。"""

    limit_tweak = LimitTweakCfg(
        self_mode="disturb",
        disturb_object="independent",
        disturb_type="add",
        joint_range=(0.01, 0.01),
    )


def test_post_mutate_config_is_direct_hand_generator_cfg():
    r"""配置模块中的 `POST_MUTATE_CFG` 应直接是 `HandGeneratorCfg`。"""

    assert isinstance(asset_cfg_module.POST_MUTATE_CFG, HandGeneratorCfg)
    assert asset_cfg_module.POST_MUTATE_CFG.mode == "mutate"
    assert isinstance(asset_cfg_module.POST_MUTATE_CFG.Validate, HandValidatorCfg)
    assert asset_cfg_module.POST_MUTATE_CFG.post_mutate_require_unique_geometry is True
    assert asset_cfg_module.POST_MUTATE_CFG.post_mutate_sdf_execution == "central_gpu_batch"
    assert asset_cfg_module.POST_MUTATE_CFG.Validate is not None
    assert asset_cfg_module.POST_MUTATE_CFG.Validate.post_mutate.sdf_device == "cuda"
    assert asset_cfg_module.POST_MUTATE_CFG.Validate.post_mutate.sdf_mesh_backend == "warp"
    term_names = tuple(name for name, _ in asset_cfg_module.POST_MUTATE_CFG.Mutate.ordered_terms())
    assert "tip_replace" in term_names
    assert "link_proximal_overlap" in term_names
    assert term_names.index("link_scale") < term_names.index("link_proximal_overlap")


def test_post_mutate_default_allows_identity_geometry_for_weighted_sampling(tmp_path):
    r"""关闭 strict uniqueness 时保留旧实验语义：identity 可以重复计入成功样本。"""

    topology_dir, _original_sample_id = _make_pre_made_topology_dir(tmp_path)
    cfg = HandGeneratorCfg(
        mode="mutate",
        artifact_level="bundle",
        source_topology_dir=topology_dir,
        n_samples=2,
        post_mutate_attempts_per_variant=1,
        Mutate=DemoIdentityLinkScaleMutatorCfg(),
        Validate=None,
        Physics=None,
    )

    results = list(HandGenerator(cfg).generate_batch())
    summary = yaml.safe_load((_mutate_run_dirs(topology_dir)[0] / "summary.yaml").read_text(encoding="utf-8"))

    assert cfg.post_mutate_require_unique_geometry is False
    assert len(results) == 2
    assert summary["stats"]["succeeded"] == 2
    assert summary["stats"]["rejected"] == 0


def test_post_mutate_strict_uniqueness_rejects_mother_and_limit_only_geometry(tmp_path):
    r"""strict 模式把 identity 与 limit-only 候选都视为 mother geometry no-op。"""

    for mutator_cfg in (DemoIdentityLinkScaleMutatorCfg(), DemoFixedLimitMutatorCfg()):
        topology_dir, _original_sample_id = _make_pre_made_topology_dir(tmp_path / type(mutator_cfg).__name__)
        cfg = HandGeneratorCfg(
            mode="mutate",
            artifact_level="bundle",
            source_topology_dir=topology_dir,
            n_samples=1,
            post_mutate_attempts_per_variant=2,
            post_mutate_require_unique_geometry=True,
            Mutate=mutator_cfg,
            Validate=None,
            Physics=None,
        )

        results = list(HandGenerator(cfg).generate_batch())
        run_dir = _mutate_run_dirs(topology_dir)[0]
        summary = yaml.safe_load((run_dir / "summary.yaml").read_text(encoding="utf-8"))

        assert results == []
        assert summary["stats"]["succeeded"] == 0
        assert summary["stats"]["rejected"] == 2
        assert summary["stats"]["rejected_by_reason"] == {
            "post_mutate.duplicate_mother_geometry": 2,
        }
        assert summary["post_mutate_sampling"]["shortfall"] == 1
        assert all(
            rejection["error_codes"] == ["post_mutate.duplicate_mother_geometry"]
            for rejection in summary["post_mutate_sampling"]["slots"][0]["rejections"]
        )
        assert not tuple(run_dir.glob("*/hand.yaml"))  # duplicate 在 sample bundle 创建前被拒绝


def test_post_mutate_strict_uniqueness_rejects_duplicate_within_variant_set(tmp_path):
    r"""固定非 identity proposal 只允许首个 variant，后续同几何候选逐槽补抽直至 shortfall。"""

    topology_dir, _original_sample_id = _make_pre_made_topology_dir(tmp_path)
    cfg = HandGeneratorCfg(
        mode="mutate",
        artifact_level="bundle",
        source_topology_dir=topology_dir,
        n_samples=2,
        post_mutate_attempts_per_variant=3,
        post_mutate_require_unique_geometry=True,
        Mutate=DemoFixedLinkScaleMutatorCfg(),
        Validate=None,
        Physics=None,
    )

    results = list(HandGenerator(cfg).generate_batch())
    summary = yaml.safe_load((_mutate_run_dirs(topology_dir)[0] / "summary.yaml").read_text(encoding="utf-8"))

    assert len(results) == 1
    assert "geometry_fingerprint" in results[0].metadata
    assert results[0].sidecar_path is not None
    assert geometry_fingerprint_from_sidecar(results[0].sidecar_path) == results[0].metadata["geometry_fingerprint"]
    assert summary["stats"]["succeeded"] == 1
    assert summary["stats"]["rejected"] == 3
    assert summary["stats"]["rejected_by_reason"] == {
        "post_mutate.duplicate_variant_geometry": 3,
    }
    assert summary["post_mutate_sampling"]["successful_variants"] == 1
    assert summary["post_mutate_sampling"]["shortfall"] == 1
    assert summary["post_mutate_sampling"]["slots"][0]["accepted"] is True
    assert summary["post_mutate_sampling"]["slots"][1]["accepted"] is False
    assert len(tuple(_mutate_run_dirs(topology_dir)[0].glob("*/hand.yaml"))) == 1


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
        recolored="anatomy_soft_v1",
    )

    results = list(HandGenerator(mutate_cfg).generate_batch())

    assert len(results) == 2
    assert all(result.metadata["source_origin_sample_id"] == original_sample_id for result in results)
    assert all(result.metadata["source_origin_topology_dir"] == str(topology_dir) for result in results)
    assert all(
        result.metadata["post_mutate_samples"]["mount_perturb"]["resolved_self_mode"] == "general"
        for result in results
    )

    mutate_run_dirs = _mutate_run_dirs(topology_dir)
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
    mutate_run_dir = _mutate_run_dirs(topology_dir)[0]
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


def test_post_mutate_seed_replays_independent_joint_proposals_and_complete_summary(tmp_path):
    r"""同一 seed 应重放完整联合样本，且多个 term 不得按边缘 quota 对齐。"""

    topology_dir, _original_sample_id = _make_pre_made_topology_dir(tmp_path)
    mutate_cfg = HandGeneratorCfg(
        mode="mutate",
        artifact_level="hand_cfg",
        source_topology_dir=topology_dir,
        output_dir=tmp_path,
        n_samples=8,
        post_mutate_seed=20260813,
        Mutate=DemoProposalMountAndLimitMutatorCfg(),
        Validate=None,
        Physics=None,
    )

    first_results = list(HandGenerator(mutate_cfg).generate_batch())
    second_results = list(HandGenerator(mutate_cfg).generate_batch())
    first_samples = [result.metadata["post_mutate_samples"] for result in first_results]
    second_samples = [result.metadata["post_mutate_samples"] for result in second_results]
    joint_modes = {
        (
            sample["mount_perturb"]["resolved_self_mode"],
            sample["limit_tweak"]["resolved_self_mode"],
        )
        for sample in first_samples
    }
    first_run_dir, _second_run_dir = _mutate_run_dirs(topology_dir)
    summary = yaml.safe_load((first_run_dir / "summary.yaml").read_text(encoding="utf-8"))

    assert first_samples == second_samples
    assert len(joint_modes) >= 3  # 旧 quota schedule 只能产生两个彼此对齐的组合
    assert summary["config"]["Mutate"]["mount_perturb"]["self_mode"] == {"identity": 0.5, "general": 0.5}
    assert summary["post_mutate_sampling"]["seed"] == 20260813
    assert summary["post_mutate_sampling"]["planned_variants"] == 8
    assert summary["post_mutate_sampling"]["successful_variants"] == 8
    assert summary["post_mutate_sampling"]["shortfall"] == 0
    assert len(summary["post_mutate_sampling"]["slots"]) == 8
    assert sum(summary["post_mutate_joint_mode_stats"]["proposed"].values()) == 8
    assert summary["post_mutate_joint_mode_stats"]["proposed"] == summary["post_mutate_joint_mode_stats"]["accepted"]
    assert "target_quota" not in str(summary)


def test_tip_replace_records_proposed_and_accepted_type_counts(tmp_path):
    r"""tip type 统计应来自实际 proposal，而不是由目标配额反推。"""

    topology_dir, _original_sample_id = _make_pre_made_topology_dir(tmp_path)
    mutate_cfg = HandGeneratorCfg(
        mode="mutate",
        artifact_level="bundle",
        source_topology_dir=topology_dir,
        output_dir=tmp_path,
        n_samples=4,
        post_mutate_seed=20260813,
        Mutate=DemoProposalTipReplaceMutatorCfg(),
        Validate=None,
    )

    results = list(HandGenerator(mutate_cfg).generate_batch())

    mutate_run_dir = _mutate_run_dirs(topology_dir)[0]
    summary = yaml.safe_load((mutate_run_dir / "summary.yaml").read_text(encoding="utf-8"))

    assert len(results) == 4
    assert summary["post_mutate_tip_type_stats"]["proposed"] == summary["post_mutate_tip_type_stats"]["accepted"]
    assert sum(summary["post_mutate_tip_type_stats"]["accepted"].values()) > 0


def test_post_mutate_failed_slot_stops_at_its_own_attempt_budget(monkeypatch, tmp_path):
    r"""一个槽位失败不应全局补位；每个槽位独立耗尽预算后形成 shortfall。"""

    topology_dir, _original_sample_id = _make_pre_made_topology_dir(tmp_path)
    mutate_cfg = HandGeneratorCfg(
        mode="mutate",
        artifact_level="hand_cfg",
        source_topology_dir=topology_dir,
        output_dir=tmp_path,
        n_samples=2,
        post_mutate_seed=20260813,
        post_mutate_attempts_per_variant=3,
        Mutate=DemoProposalLinkScaleMutatorCfg(),
        Validate=None,
        Physics=None,
    )
    generator = HandGenerator(mutate_cfg)

    def reject_candidate(**_kwargs):
        generator._last_rejection_detail = {
            "stage": "post_mutate_validate",
            "errors": ["synthetic rejection"],
            "error_codes": ["test.synthetic_rejection"],
            "metadata": {},
        }
        generator._record_generation_rejection(
            stage="post_mutate_validate",
            error_codes=("test.synthetic_rejection",),
        )

    monkeypatch.setattr(generator, "_generate_once", reject_candidate)
    results = list(generator.generate_batch())
    mutate_run_dir = _mutate_run_dirs(topology_dir)[0]
    summary = yaml.safe_load((mutate_run_dir / "summary.yaml").read_text(encoding="utf-8"))

    assert results == []
    assert summary["stats"]["attempted"] == 6
    assert summary["post_mutate_sampling"]["successful_variants"] == 0
    assert summary["post_mutate_sampling"]["shortfall"] == 2
    assert [slot["attempts"] for slot in summary["post_mutate_sampling"]["slots"]] == [3, 3]
    assert all(slot["accepted"] is False for slot in summary["post_mutate_sampling"]["slots"])
    assert sum(summary["post_mutate_joint_mode_stats"]["proposed"].values()) == 6
    assert summary["post_mutate_joint_mode_stats"]["accepted"] == {}


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
            recolored="anatomy_soft_v1",
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
