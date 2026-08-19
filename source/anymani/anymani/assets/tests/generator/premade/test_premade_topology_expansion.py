"""pre-made family composition / missing-slot 正交拓扑扩展测试。

这组测试专门锁住这轮新引入的 topology 主线：

1. slot-level connectivity candidate pool 会触发 missing-finger pre-made 枚举；
2. family composition 与 missing slots 是两个正交轴，mixed 手也能缺失一根 non-thumb；
3. true mixed 只由存活 non-thumb 判定，并且必须同时包含 LEAP 与 Allegro；
4. mixed-family finger 直接在 builder 层生成，而不是依赖 post-mutate `finger_replace`；
5. mixed 产物的递归输出根目录会进入 `generated/<timestamp>/mixed/...`；
6. metadata 会同时暴露 family composition、missing slots、surviving slots 与 handedness。
"""

from __future__ import annotations

from collections import Counter

import yaml
from assets.config.asset_gen_cfg import PRE_MADE_CFG
from assets.generator.hand_generator import HandGenerator, HandGeneratorCfg
from assets.generator.premade.batch import build_premade_tasks
from assets.generator.premade.topology import (
    build_base_hand,
    build_premade_topology_registry,
    extract_premade_topology_metadata,
)
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
        assert (
            result.metadata["surviving_slots"] == ["thumb", "index", "middle"]
            or result.metadata["surviving_slots"] == ["thumb", "index", "ring"]
            or result.metadata["surviving_slots"] == ["thumb", "middle", "ring"]
        )
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
    assert all(result.urdf_path.parent.name == result.metadata["topology_name"] for result in mixed_results)
    assert all(
        result.urdf_path.parent.parent.name == result.metadata["topology_group_name"] for result in mixed_results
    )
    assert all(result.urdf_path.parent.parent.parent.name == "mixed" for result in mixed_results)
    assert all(result.metadata["slot_family_map"]["thumb"] == "allegro" for result in mixed_results)
    assert any(result.metadata["slot_family_map"]["index"] == "leap" for result in mixed_results)
    assert all(result.metadata["topology_anchor"] == "mixed" for result in mixed_results)
    assert len(list(tmp_path.glob("*/summary.yaml"))) == 1


def test_generate_batch_combines_mixed_family_with_missing_slot_under_mixed_root(tmp_path):
    r"""mixed 与 missing 必须形成真实的组合资产，而不是两条互斥枚举支路。

    这里固定 Allegro palm/thumb，并只允许 index 切换到 LEAP。删除 middle 或 ring 后，
    存活 non-thumb 仍分别包含 LEAP 与 Allegro，因此它们是 true mixed；删除 index 后
    只剩 Allegro non-thumb，必须退化为 single-family missing，而不能误报为 mixed。
    """

    results = list(
        HandGenerator(
            HandGeneratorCfg(
                mode="made",
                artifact_level="bundle",
                output_dir=tmp_path,
                handedness="right",
                hand_presets=["single_palm_allegro"],
                mixed=True,
                missing=True,
                connectivity_presets={
                    "single_palm_allegro": {
                        "thumb": ["allegro_thumb_full"],
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

    mixed_missing = [
        result
        for result in results
        if result.metadata["family_composition"] == "mixed" and result.metadata["missing_slots"]
    ]
    assert {tuple(result.metadata["missing_slots"]) for result in mixed_missing} == {("middle",), ("ring",)}
    for result in mixed_missing:
        assert result.urdf_path is not None
        assert result.urdf_path.parent.parent.parent.name == "mixed"
        assert len(result.metadata["surviving_slots"]) == 3
        non_thumb_families = {family for slot, family in result.metadata["slot_family_map"].items() if slot != "thumb"}
        assert non_thumb_families == {"allegro", "leap"}
        missing_slot = result.metadata["missing_slots"][0]
        assert missing_slot not in result.metadata["topology_group_name"]
        assert f"_{missing_slot[0]}" not in result.metadata["topology_name"]

        # 顶层 sidecar 与可恢复 HandCfg snapshot 必须交付同一正交 topology provenance。
        sidecar = yaml.safe_load(result.urdf_path.with_name("hand.yaml").read_text(encoding="utf-8"))
        assert sidecar["family_composition"] == "mixed"
        assert sidecar["missing_slots"] == [missing_slot]
        assert sidecar["hand_cfg"]["metadata"]["premade_connectivity"]["family_composition"] == "mixed"
        assert sidecar["hand_cfg"]["metadata"]["premade_connectivity"]["missing_slots"] == [missing_slot]

        # Run summary 继续沿用 mixed/<composition>/<topology> 三层键，便于 dataset manifest 审计。
        run_root = result.urdf_path.parents[3]
        summary = yaml.safe_load((run_root / "summary.yaml").read_text(encoding="utf-8"))
        topology_key = f"mixed/{result.metadata['topology_group_name']}/{result.metadata['topology_name']}"
        assert summary["stats"]["by_topology"][topology_key] == 1


def test_default_registry_factorizes_family_composition_and_missing_slots():
    r"""默认 registry 应形成 full/missing 与 single/mixed 的受约束笛卡尔积。

    计数均发生在 validator 之前：

    - canonical base：$2$ 个 palm family $\times2$ 个 handedness，共 $4$ 个；
    - single-family missing：每个 base 缺一根 non-thumb，共 $4\times3=12$ 个；
    - full true mixed：三个 non-thumb 同时含两种 family，共 $4\times(2^3-2)=24$ 个；
    - mixed missing：两个 surviving non-thumb 一边一个 family，共 $4\times3\times2=24$ 个；
    - 合计 $64$ 个 topology specs，并展开为 $3256$ 个 connectivity 候选任务。
    """

    registry = build_premade_topology_registry(PRE_MADE_CFG)
    base_specs = [
        spec for spec in registry.values() if spec.family_composition == "single_family" and not spec.missing_slots
    ]
    missing_specs = [
        spec for spec in registry.values() if spec.family_composition == "single_family" and spec.missing_slots
    ]
    full_mixed_specs = [
        spec for spec in registry.values() if spec.family_composition == "mixed" and not spec.missing_slots
    ]
    mixed_missing_specs = [
        spec for spec in registry.values() if spec.family_composition == "mixed" and spec.missing_slots
    ]
    tasks = build_premade_tasks(HandGenerator(PRE_MADE_CFG))

    assert len(registry) == 64
    assert len(base_specs) == 4
    assert len(missing_specs) == 12
    assert len(full_mixed_specs) == 24
    assert len(mixed_missing_specs) == 24
    assert len(tasks) == 3256
    mixed_specs = full_mixed_specs + mixed_missing_specs
    assert all(spec.slot_family_map()["thumb"] == spec.family for spec in mixed_specs)
    assert all(
        {family for slot, family in spec.slot_family_map().items() if slot != "thumb"} == {"allegro", "leap"}
        for spec in mixed_specs
    )
    assert all(len(spec.missing_slots) == 1 and len(spec.surviving_slots) == 3 for spec in mixed_missing_specs)
    assert Counter((spec.family, spec.handedness) for spec in full_mixed_specs) == {
        ("allegro", "left"): 6,
        ("allegro", "right"): 6,
        ("leap", "left"): 6,
        ("leap", "right"): 6,
    }
    assert Counter((spec.family, spec.handedness) for spec in mixed_missing_specs) == {
        ("allegro", "left"): 6,
        ("allegro", "right"): 6,
        ("leap", "left"): 6,
        ("leap", "right"): 6,
    }


def test_legacy_missing_metadata_recovers_orthogonal_axes_without_rewriting_asset() -> None:
    r"""历史 sidecar 的粗粒度 ``missing`` 标签应在内存中恢复正交字段。

    旧 bundle 已持久化并被 dataset 引用，patch 不能要求用户重写资产。恢复时使用
    base preset 的 canonical slots 与 surviving slots 求差，得到缺失的 ring；原始
    ``HandCfg`` 与 metadata dict 均保持不变。
    """

    cfg = HandGeneratorCfg(
        mode="made",
        artifact_level="hand_cfg",
        handedness="right",
        hand_presets=["single_palm_allegro"],
        mixed=True,
        missing=True,
    )
    hand, _ = build_base_hand(cfg, hand_preset_name="single_palm_allegro__right__missing_ring")
    premade_topology = dict(hand.metadata["premade_topology"])
    premade_topology.pop("family_composition")
    premade_topology.pop("missing_slots")
    legacy_hand = hand.replace(metadata={"premade_topology": premade_topology})

    normalized = extract_premade_topology_metadata(
        legacy_hand,
        hand_preset_name="single_palm_allegro__right__missing_ring",
    )

    assert normalized["topology_kind"] == "missing"
    assert normalized["family_composition"] == "single_family"
    assert normalized["missing_slots"] == ["ring"]
    assert "family_composition" not in premade_topology
    assert "missing_slots" not in premade_topology
