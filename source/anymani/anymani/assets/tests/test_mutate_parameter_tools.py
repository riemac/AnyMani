"""后序 mutate 的参数级工具回归测试。

当前这组测试刻意只覆盖“最不改拓扑”的 slice：

1. `limit_tweak`
2. `mount_perturb`
3. `HandMutator` 的流水线编排
4. `HandGenerator` 对已支持 mutate 工具的调度

这样做的目的，是先把 post-mutate 从“完全不可执行”推进到
“参数级工具已经可用”，而不在同一轮里混入 `joint_delete` / `finger_replace`
 这类结构级重写。
"""

from __future__ import annotations

import math
import random

from assets.builder.hand_builders import HumanLikeHandBuilder, HumanLikeHandBuilderCfg
from assets.builder.palm_builders import ComPalmBuilderCfg
from assets.generator.hand_generator import HandGenerator, HandGeneratorCfg
from assets.generator.mutate import HandMutator, HandMutatorCfg, LimitTweakCfg, LimitTweakMutator, MountPerturbCfg, MountPerturbMutator


def _make_allegro_builder_cfg() -> HumanLikeHandBuilderCfg:
    """构造一份稳定的 Allegro pre-made hand recipe。"""

    return HumanLikeHandBuilderCfg(
        name="allegro_mutate_demo",
        family="allegro",
        handedness="right",
        palm_cfg=ComPalmBuilderCfg(preset="allegro"),
        finger_cfg="allegro_non_thumb_v1",
        thumb_cfg="allegro_thumb_v1",
    )


def _build_allegro_hand():
    """构造一份稳定的整手 `HandCfg`，供 mutate 测试复用。"""

    return HumanLikeHandBuilder(_make_allegro_builder_cfg()).build()


def _joint_by_name(hand, joint_name: str):
    """按名字取 joint，避免测试里反复手写展平查找。"""

    for joint in hand.iter_joints():
        if joint.name == joint_name:
            return joint
    raise KeyError(joint_name)


def _finger_by_name(hand, finger_name: str):
    """按名字取 finger。"""

    for finger in hand.fingers:
        if finger.name == finger_name:
            return finger
    raise KeyError(finger_name)


def test_limit_tweak_mutator_changes_target_joint_and_preserves_valid_interval():
    """`limit_tweak` 应只改目标关节，并保持 `lower < upper`。"""

    random.seed(0)
    hand = _build_allegro_hand()
    before_index = _joint_by_name(hand, "index_j0").limit
    before_middle = _joint_by_name(hand, "middle_j0").limit

    mutated = LimitTweakMutator(
        LimitTweakCfg(
            target_joints=("index_j0",),
            mode="absolute",
            sigma=0.1,
            symmetric=True,
            clip=0.1,
        )
    ).mutate(hand)

    assert mutated is not None
    after_index = _joint_by_name(mutated, "index_j0").limit
    after_middle = _joint_by_name(mutated, "middle_j0").limit
    assert after_index.lower < after_index.upper
    assert not math.isclose(after_index.lower, before_index.lower, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(after_index.lower - before_index.lower, -(after_index.upper - before_index.upper), rel_tol=0.0, abs_tol=1e-12)
    assert after_middle.lower == before_middle.lower
    assert after_middle.upper == before_middle.upper


def test_mount_perturb_mutator_changes_only_target_finger_mount():
    """`mount_perturb` 应只改目标 finger 的挂载位姿，不影响其他 finger。"""

    random.seed(0)
    hand = _build_allegro_hand()
    before_index = _finger_by_name(hand, "index").mount
    before_middle = _finger_by_name(hand, "middle").mount

    mutated = MountPerturbMutator(
        MountPerturbCfg(
            target_fingers=("index",),
            translation_sigma=0.003,
            perturb_rotation=True,
            rotation_sigma=0.05,
        )
    ).mutate(hand)

    assert mutated is not None
    after_index = _finger_by_name(mutated, "index").mount
    after_middle = _finger_by_name(mutated, "middle").mount
    assert after_index.pos != before_index.pos
    assert after_index.rpy != before_index.rpy
    assert after_middle.pos == before_middle.pos
    assert after_middle.rpy == before_middle.rpy


def test_hand_mutator_pipeline_applies_parameter_tools_and_step_validation():
    """`HandMutator` 应能串联参数级工具，并在 step_validate 打开时仍顺利通过。"""

    random.seed(0)
    hand = _build_allegro_hand()

    mutated = HandMutator(
        HandMutatorCfg(
            limit_tweak=LimitTweakCfg(
                target_joints=("index_j0",),
                mode="absolute",
                sigma=0.1,
                symmetric=True,
                clip=0.1,
            ),
            mount_perturb=MountPerturbCfg(
                target_fingers=("index",),
                translation_sigma=0.003,
            ),
            order=("limit_tweak", "mount_perturb"),
            step_validate=True,
        )
    ).mutate(hand)

    assert mutated is not None
    assert _joint_by_name(mutated, "index_j0").limit.lower != _joint_by_name(hand, "index_j0").limit.lower
    assert _finger_by_name(mutated, "index").mount.pos != _finger_by_name(hand, "index").mount.pos


def test_hand_generator_executes_supported_mutate_pipeline():
    """`HandGenerator` 在 full 模式下应能执行已支持的参数级 mutate。"""

    random.seed(0)
    baseline = HumanLikeHandBuilder(_make_allegro_builder_cfg()).build()
    result = HandGenerator(
        HandGeneratorCfg(
            mode="full",
            artifact_level="hand_cfg",
            Made=_make_allegro_builder_cfg(),
            Mutate=HandMutatorCfg(
                mount_perturb=MountPerturbCfg(
                    target_fingers=("index",),
                    translation_sigma=0.003,
                ),
                order=("mount_perturb",),
            ),
        )
    ).generate()

    assert result is not None
    assert result.hand_cfg is not None
    assert _finger_by_name(result.hand_cfg, "index").mount.pos != _finger_by_name(baseline, "index").mount.pos
