"""`finger_replace` 回归测试。

这一组测试把“整根 finger 替换”这条结构级但相对温和的 mutate 路线锁住，
以区别于更重的 `joint_delete + regroup`。
"""

from __future__ import annotations

from assets.builder.hand_builders import HumanLikeHandBuilder, HumanLikeHandBuilderCfg
from assets.generator.hand_generator import HandGenerator, HandGeneratorCfg
from assets.generator.mutate import FingerReplaceCfg, FingerReplaceMutator, HandMutatorCfg
from assets.presets import make_human_like_builder_cfg


def _make_allegro_builder_cfg() -> HumanLikeHandBuilderCfg:
    """构造一份稳定的 Allegro pre-made hand recipe。"""

    return make_human_like_builder_cfg(
        name="allegro_finger_replace_demo",
        family="allegro",
        handedness="right",
        palm_cfg="com_allegro",
        finger_cfg="allegro_non_thumb_v1",
        thumb_cfg="allegro_thumb_v1",
    )


def _build_allegro_hand():
    """构造一份稳定的 Allegro `HandCfg`。"""

    return HumanLikeHandBuilder(_make_allegro_builder_cfg()).build()


def _finger_by_name(hand, finger_name: str):
    """按名字取 finger。"""

    for finger in hand.fingers:
        if finger.name == finger_name:
            return finger
    raise KeyError(finger_name)


def test_finger_replace_mutator_rebuilds_target_slot_from_preset_and_inherits_mount():
    """`finger_replace` 应替换目标 slot 的结构，同时保留原挂载位姿。"""

    hand = _build_allegro_hand()
    before_index = _finger_by_name(hand, "index")
    before_middle = _finger_by_name(hand, "middle")

    mutated = FingerReplaceMutator(
        FingerReplaceCfg(
            target_finger="index",
            strategy="preset",
            replacement_preset_name="leap_non_thumb_v1",
            inherit_mount=True,
        )
    ).mutate(hand)

    assert mutated is not None
    after_index = _finger_by_name(mutated, "index")
    after_middle = _finger_by_name(mutated, "middle")
    assert after_index.mount.pos == before_index.mount.pos
    assert after_index.mount.rpy == before_index.mount.rpy
    assert after_index.joints[0].axis != before_index.joints[0].axis
    assert after_middle.joints[0].axis == before_middle.joints[0].axis


def test_hand_generator_executes_finger_replace_pipeline():
    """`HandGenerator` 在 full 模式下应能执行 `finger_replace`。"""

    baseline = HumanLikeHandBuilder(_make_allegro_builder_cfg()).build()
    result = HandGenerator(
        HandGeneratorCfg(
            mode="full",
            artifact_level="hand_cfg",
            Made=_make_allegro_builder_cfg(),
            Mutate=HandMutatorCfg(
                finger_replace=FingerReplaceCfg(
                    target_finger="index",
                    strategy="preset",
                    replacement_preset_name="leap_non_thumb_v1",
                    inherit_mount=True,
                ),
                order=("finger_replace",),
            ),
        )
    ).generate()

    assert result is not None
    assert result.hand_cfg is not None
    assert _finger_by_name(result.hand_cfg, "index").joints[0].axis != _finger_by_name(baseline, "index").joints[0].axis
