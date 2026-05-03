r"""independent post-mutate quick façade 回归测试。

这里同时覆盖两层语义：

1. `HandGenerator(mode="mutate")` 的旧核心语义仍然是 topology 目录；
2. `quick_post_mutate.py` 给研究者暴露的是更直观的 pre-made sample 目录。

quick 层必须通过 staging 目录把二者接起来，并保证原始 pre-made sample 不被
`*_origin` 重命名污染。这个约束是二次调参时最关键的可逆性保证。
"""

from __future__ import annotations

from pathlib import Path

from assets.generator.hand_generator import HandGenerator, HandGeneratorCfg
from assets.generator.mutate import HandMutatorCfg, MountPerturbCfg, MutatorTerm, ScalarDistributionCfg
import assets.scripts.quick_post_mutate as quick_post_mutate_module
from assets.validator.hand_rules import HandValidatorCfg


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
    r"""创建只用于 quick staging 目录策略测试的轻量 sample 目录。

    这里不需要恢复成真正 `HandCfg`，因为这些测试只检查 quick façade 的目录复制
    与覆盖策略；真正的 `HandCfg` 恢复路径由后面的 generator smoke 测试覆盖。
    """

    sample_dir.mkdir(parents=True)  # 模拟 `.../<topology>/<sample_id>/`
    (sample_dir / "hand.yaml").write_text("hand_cfg: {}\n", encoding="utf-8")  # staging 只要求存在 sidecar
    (sample_dir / "hand.urdf").write_text("<robot name=\"fake\" />\n", encoding="utf-8")  # 模拟 bundle 里的 URDF


def test_quick_post_mutate_run_cfg_is_direct_hand_generator_cfg():
    r"""quick_post_mutate.py 顶部正式入口应直接是 `HandGeneratorCfg`。"""

    assert isinstance(quick_post_mutate_module.RUN_CFG, HandGeneratorCfg)
    assert quick_post_mutate_module.RUN_CFG.mode == "mutate"
    assert quick_post_mutate_module.RUN_CFG.source_topology_dir == quick_post_mutate_module.SOURCE_TOPOLOGY_DIR
    assert quick_post_mutate_module.SOURCE_TOPOLOGY_DIR == (
        quick_post_mutate_module.SOURCE_PREMADE_SAMPLE_DIR
        / quick_post_mutate_module.POST_MUTATE_RUN_NAME
    )
    assert quick_post_mutate_module.POST_MUTATE_CFG.order == quick_post_mutate_module.POST_MUTATE_PATCH_ORDER
    assert isinstance(quick_post_mutate_module.RUN_CFG.Validate, HandValidatorCfg)
    assert tuple(quick_post_mutate_module.RUN_CFG.Mutate.order) == (
        "link_scale",
        "mount_perturb",
        "limit_tweak",
        "tip_replace",
    )


def test_quick_post_mutate_resolves_direct_sample_path(tmp_path):
    r"""用户直接粘贴到 sample 目录时，不需要 topology 层推断。"""

    source_sample_dir = tmp_path / "right_t4_i4_m4_r4" / "f5d8c069"
    _make_fake_source_sample(source_sample_dir)

    resolved_dir = quick_post_mutate_module.resolve_source_premade_sample_dir(
        source_sample_dir,
        sample_id="f5d8c069",
    )

    assert resolved_dir == source_sample_dir


def test_quick_post_mutate_resolves_topology_path_with_sample_id(tmp_path):
    r"""用户粘贴 topology 目录时，可用 sample ID 精确选择父拓扑下的来源样本。"""

    topology_dir = tmp_path / "right_t4_i4_m4_r4"
    _make_fake_source_sample(topology_dir / "f5d8c069")
    _make_fake_source_sample(topology_dir / "abcd1234")

    resolved_dir = quick_post_mutate_module.resolve_source_premade_sample_dir(
        topology_dir,
        sample_id="f5d8c069",
    )

    assert resolved_dir == topology_dir / "f5d8c069"


def test_quick_post_mutate_plans_nested_and_sibling_run_dirs(tmp_path):
    r"""quick façade 应能显式切换 sample 内部 run 和平级目录两种布局。"""

    source_sample_dir = tmp_path / "right_t4_i4_m4_r4" / "f5d8c069"  # 模拟用户最直观会复制的 sample 路径

    nested_dir = quick_post_mutate_module.planned_post_mutate_topology_dir(
        source_sample_dir=source_sample_dir,
        layout="nested",
        run_name="try_small",
    )
    sibling_dir = quick_post_mutate_module.planned_post_mutate_topology_dir(
        source_sample_dir=source_sample_dir,
        layout="sibling",
        run_name="try_small",
    )

    assert nested_dir == source_sample_dir / "try_small"
    assert sibling_dir == source_sample_dir.parent / "f5d8c069_post_mutate" / "try_small"


def test_quick_post_mutate_prepare_overwrite_copies_source_without_run_recursion(tmp_path):
    r"""`overwrite` 应只覆盖当前 run，不把旧 run 目录递归复制进 origin。"""

    source_sample_dir = tmp_path / "topology" / "f5d8c069"
    _make_fake_source_sample(source_sample_dir)

    stale_run_dir = source_sample_dir / "try_001"
    stale_run_dir.mkdir(parents=True)
    (stale_run_dir / "stale.txt").write_text("old failed attempt\n", encoding="utf-8")  # 模拟上次失败调参产物

    preserved_run_dir = source_sample_dir / "try_keep"
    preserved_run_dir.mkdir(parents=True)
    (preserved_run_dir / "keep.txt").write_text("keep me\n", encoding="utf-8")  # 其它 run 不应被覆盖策略误删
    legacy_post_mutate_dir = source_sample_dir / "post_mutate"
    legacy_post_mutate_dir.mkdir()
    (legacy_post_mutate_dir / "legacy.txt").write_text("old layout\n", encoding="utf-8")  # 旧布局目录也不应进入 origin

    run_dir = quick_post_mutate_module.prepare_post_mutate_source_topology(
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


def test_quick_post_mutate_prepare_new_allocates_suffix_without_touching_existing_runs(tmp_path):
    r"""`new` 应自动追加后缀，适合保留多轮人工调参痕迹。"""

    source_sample_dir = tmp_path / "topology" / "f5d8c069"
    _make_fake_source_sample(source_sample_dir)

    for run_name in ("try_001", "try_001_01"):
        occupied_dir = source_sample_dir / run_name
        occupied_dir.mkdir(parents=True)
        (occupied_dir / "occupied.txt").write_text(run_name, encoding="utf-8")

    run_dir = quick_post_mutate_module.prepare_post_mutate_source_topology(
        source_sample_dir=source_sample_dir,
        layout="nested",
        run_name="try_001",
        run_policy="new",
    )

    assert run_dir == source_sample_dir / "try_001_02"
    assert (run_dir / "f5d8c069" / "hand.yaml").is_file()
    assert (source_sample_dir / "try_001" / "occupied.txt").is_file()
    assert (source_sample_dir / "try_001_01" / "occupied.txt").is_file()


def test_quick_post_mutate_nested_staging_keeps_original_sample_after_generator_run(tmp_path):
    r"""nested staging 下 generator 只能重命名复制件，不能重命名原始 sample。"""

    topology_dir, original_sample_name = _make_pre_made_topology_dir(tmp_path)
    source_sample_dir = topology_dir / original_sample_name
    run_dir = quick_post_mutate_module.prepare_post_mutate_source_topology(
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
        Mutate=HandMutatorCfg(
            terms={
                "mount": MutatorTerm(
                    cfg=MountPerturbCfg(
                        target_fingers=("index",),
                        translation_distribution=ScalarDistributionCfg(kind="fixed", value=0.001),
                    )
                )
            },
            order=("mount",),
        ),
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
    r"""mutate-only 应从 topology 目录恢复 `HandCfg`，并把新样本写成 `_origin` 的兄弟目录。"""

    topology_dir, original_sample_name = _make_pre_made_topology_dir(tmp_path)
    mutate_cfg = HandGeneratorCfg(
        mode="mutate",
        artifact_level="bundle",
        source_topology_dir=topology_dir,
        output_dir=tmp_path,
        n_samples=2,
        Mutate=HandMutatorCfg(
            terms={
                "mount": MutatorTerm(
                    cfg=MountPerturbCfg(
                        target_fingers=("index",),
                        translation_distribution=ScalarDistributionCfg(kind="fixed", value=0.001),
                    )
                )
            },
            order=("mount",),
        ),
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
