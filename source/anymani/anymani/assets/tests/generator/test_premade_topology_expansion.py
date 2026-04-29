"""pre-made mixed / missing topology 扩展测试。

这组测试专门锁住这轮新引入的 topology 主线：

1. slot-level connectivity candidate pool 会触发 missing-finger pre-made 枚举；
2. mixed-family finger 直接在 builder 层生成，而不是依赖 post-mutate `finger_replace`；
3. mixed 产物的递归输出根目录会进入 `generated/<timestamp>/mixed/...`；
4. missing 产物的 metadata 会明确暴露 surviving slots、handedness 与 topology kind。
"""

from __future__ import annotations

from assets.generator.hand_generator import HandGenerator, HandGeneratorCfg
from assets.validator.hand_rules import HandValidatorCfg


def test_generate_batch_enumerates_missing_topologies_when_slot_level_connectivity_pool_is_used(tmp_path):
    r"""slot-level connectivity pool 应把 missing-finger topology 纳入 pre-made 空间。"""

    results = list(
        HandGenerator(
            HandGeneratorCfg(
                mode="made",
                artifact_level="hand_cfg",
                output_dir=tmp_path,
                handedness="right",
                hand_presets=["single_palm_allegro"],
                mixed=False,
                missing=True,
                connectivity_presets={
                    "single_palm_allegro": {
                        "thumb": ["allegro_thumb_full"],
                        "index": ["allegro_non_thumb_full"],
                        "middle": ["allegro_non_thumb_full"],
                        "ring": ["allegro_non_thumb_full"],
                    }
                },
            )
        ).generate_batch()
    )

    missing_results = [result for result in results if result.metadata["topology_kind"] == "missing"]
    assert len(results) == 4  # 1 canonical + 3 missing(one missing slot each)
    assert len(missing_results) == 3
    assert {result.metadata["topology_name"] for result in missing_results} == {
        "right_t4_m4_r4",
        "right_t4_i4_r4",
        "right_t4_i4_m4",
    }
    for result in missing_results:
        assert result.hand_cfg is not None
        assert result.metadata["handedness"] == "right"
        assert result.metadata["topology_group_name"] == "single_palm_allegro"
        assert result.metadata["surviving_slots"] == ["thumb", "index", "middle"] or result.metadata["surviving_slots"] == ["thumb", "index", "ring"] or result.metadata["surviving_slots"] == ["thumb", "middle", "ring"]
        assert len(result.hand_cfg.fingers) == 3
        assert "thumb" in {finger.name for finger in result.hand_cfg.fingers}
    assert len(list(tmp_path.glob("*/summary.yaml"))) == 1


def test_generate_batch_builds_mixed_family_topologies_and_exports_under_mixed_root(tmp_path):
    r"""mixed-family finger 组合应直接作为 pre-made canonical topology 导出。"""

    results = list(
        HandGenerator(
            HandGeneratorCfg(
                mode="made",
                artifact_level="bundle",
                output_dir=tmp_path,
                handedness="right",
                hand_presets=["single_palm_allegro"],
                mixed=True,
                missing=False,
                connectivity_presets={
                    "single_palm_allegro": {
                        "thumb": ["allegro_thumb_full", "leap_thumb_full"],
                        "index": ["allegro_non_thumb_full", "leap_non_thumb_full"],
                        "middle": ["allegro_non_thumb_full"],
                        "ring": ["allegro_non_thumb_full"],
                    }
                },
                Validate=HandValidatorCfg(
                    pre_made=HandValidatorCfg.PreMadeCfg(
                        check_finger_spacing=False,
                        check_palm_thumb_binding=True,
                    )
                ),
            )
        ).generate_batch()
    )

    mixed_results = [result for result in results if result.metadata["topology_kind"] == "mixed"]
    assert mixed_results
    assert all(result.urdf_path is not None and "/mixed/" in result.urdf_path.as_posix() for result in mixed_results)
    assert all(result.urdf_path.parent.parent.name == result.metadata["topology_name"] for result in mixed_results)
    assert all(result.urdf_path.parent.parent.parent.name == result.metadata["topology_group_name"] for result in mixed_results)
    assert all(result.urdf_path.parent.parent.parent.parent.name == "mixed" for result in mixed_results)
    assert all(result.metadata["slot_family_map"]["thumb"] == "allegro" for result in mixed_results)
    assert any(result.metadata["slot_family_map"]["index"] == "leap" for result in mixed_results)
    assert all(result.metadata["topology_anchor"] == "mixed" for result in mixed_results)
    assert len(list(tmp_path.glob("*/summary.yaml"))) == 1
