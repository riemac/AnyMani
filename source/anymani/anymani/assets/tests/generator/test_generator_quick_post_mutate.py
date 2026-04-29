"""independent post-mutate quick façade 回归测试。"""

from __future__ import annotations

import assets.generator.quick_post_mutate as quick_post_mutate_module
from assets.generator.hand_generator import HandGenerator, HandGeneratorCfg
from assets.generator.mutate import HandMutatorCfg, MountPerturbCfg, MutatorTerm, ScalarDistributionCfg
from assets.validator.hand_rules import HandValidatorCfg


def _single_full_pool() -> dict[str, dict[str, list[str]]]:
    """提供一个只有 canonical full topology 的 pre-made pool。"""

    return {
        "single_palm_allegro": {
            "thumb": ["allegro_thumb_full"],
            "index": ["allegro_non_thumb_full"],
            "middle": ["allegro_non_thumb_full"],
            "ring": ["allegro_non_thumb_full"],
        }
    }


def _make_pre_made_topology_dir(tmp_path):
    """先生成一个真实的 pre-made topology 目录，供 mutate-only 回放使用。"""

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


def test_quick_post_mutate_run_cfg_is_direct_hand_generator_cfg():
    """quick_post_mutate.py 顶部正式入口应直接是 `HandGeneratorCfg`。"""

    assert isinstance(quick_post_mutate_module.RUN_CFG, HandGeneratorCfg)
    assert quick_post_mutate_module.RUN_CFG.mode == "mutate"
    assert quick_post_mutate_module.RUN_CFG.source_topology_dir == quick_post_mutate_module.SOURCE_TOPOLOGY_DIR
    assert isinstance(quick_post_mutate_module.RUN_CFG.Validate, HandValidatorCfg)
    assert tuple(quick_post_mutate_module.RUN_CFG.Mutate.order) == (
        "link_scale",
        "mount_perturb",
        "limit_tweak",
        "tip_replace",
    )


def test_independent_post_mutate_renames_origin_and_writes_sibling_variants(tmp_path):
    """mutate-only 应从 topology 目录恢复 `HandCfg`，并把新样本写成 `_origin` 的兄弟目录。"""

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
