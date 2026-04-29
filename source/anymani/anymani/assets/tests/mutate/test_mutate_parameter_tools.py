"""后序 mutate 参数工具与联合采样编排回归测试。"""

from __future__ import annotations

import math

from assets.builder.hand_builders import HumanLikeHandBuilder, HumanLikeHandBuilderCfg
from assets.generator.hand_generator import HandGenerator, HandGeneratorCfg
from assets.generator.mutate import (
    HandMutator,
    HandMutatorCfg,
    LimitTweakCfg,
    LimitTweakMutator,
    MountPerturbCfg,
    MountPerturbMutator,
    MutatorTerm,
    ScalarDistributionCfg,
)
from assets.presets import make_human_like_builder_cfg


def _make_allegro_builder_cfg() -> HumanLikeHandBuilderCfg:
    """构造一份稳定的 Allegro pre-made hand recipe。"""

    return make_human_like_builder_cfg(
        name="allegro_mutate_demo",
        family="allegro",
        handedness="right",
        palm_cfg="com_allegro",
        finger_cfg="allegro_non_thumb_v1",
        thumb_cfg="allegro_thumb_v1",
    )


def _build_allegro_hand():
    """构造一份稳定的整手 `HandCfg`，供 mutate 测试复用。"""

    return HumanLikeHandBuilder(_make_allegro_builder_cfg()).build()


def _joint_by_name(hand, joint_name: str):
    """按名字取 joint。"""

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


class DemoParameterMutatorCfg(HandMutatorCfg):
    """用类属性声明 term，锁住新的 IsaacLab 风格 container 用法。"""

    limit = MutatorTerm(
        cfg=LimitTweakCfg(
            target_joints=("index_j0",),
            mode="absolute",
            symmetric=True,
            clip=0.1,
            delta_distribution=ScalarDistributionCfg(kind="fixed", value=0.05),
        )
    )
    mount = MutatorTerm(
        cfg=MountPerturbCfg(
            target_fingers=("index",),
            perturb_rotation=True,
            translation_distribution=ScalarDistributionCfg(kind="fixed", value=0.001),
            rotation_distribution=ScalarDistributionCfg(kind="fixed", value=0.02),
        )
    )


def test_limit_tweak_mutator_consumes_sampled_values_and_preserves_valid_interval():
    """`limit_tweak` 应只改目标关节，并保持 `lower < upper`。"""

    hand = _build_allegro_hand()
    before_index = _joint_by_name(hand, "index_j0").limit
    before_middle = _joint_by_name(hand, "middle_j0").limit

    mutated = LimitTweakMutator(
        LimitTweakCfg(
            target_joints=("index_j0",),
            mode="absolute",
            symmetric=True,
            clip=0.1,
            delta_distribution=ScalarDistributionCfg(kind="fixed", value=0.05),
        )
    ).mutate(hand, sampled_params={"index_j0::delta": 0.05})

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

    hand = _build_allegro_hand()
    before_index = _finger_by_name(hand, "index").mount
    before_middle = _finger_by_name(hand, "middle").mount

    mutated = MountPerturbMutator(
        MountPerturbCfg(
            target_fingers=("index",),
            perturb_rotation=True,
            translation_distribution=ScalarDistributionCfg(kind="fixed", value=0.001),
            rotation_distribution=ScalarDistributionCfg(kind="fixed", value=0.02),
        )
    ).mutate(
        hand,
        sampled_params={
            "index::tx": 0.001,
            "index::ty": 0.001,
            "index::tz": 0.001,
            "index::rr": 0.02,
            "index::rp": 0.02,
            "index::ry": 0.02,
        },
    )

    assert mutated is not None
    after_index = _finger_by_name(mutated, "index").mount
    after_middle = _finger_by_name(mutated, "middle").mount
    assert after_index.pos != before_index.pos
    assert after_index.rpy != before_index.rpy
    assert after_middle.pos == before_middle.pos
    assert after_middle.rpy == before_middle.rpy


def test_hand_mutator_pipeline_accepts_declared_terms_and_step_validation():
    """`HandMutatorCfg` 应按声明顺序解析 term，并接受上游采样值。"""

    hand = _build_allegro_hand()
    cfg = DemoParameterMutatorCfg(order=("limit", "mount"), step_validate=True)

    mutated = HandMutator(cfg).mutate(
        hand,
        sampled_params={
            "limit": {"index_j0::delta": 0.05},
            "mount": {
                "index::tx": 0.001,
                "index::ty": 0.001,
                "index::tz": 0.001,
                "index::rr": 0.02,
                "index::rp": 0.02,
                "index::ry": 0.02,
            },
        },
    )

    assert mutated is not None
    assert [name for name, _ in cfg.ordered_terms()] == ["limit", "mount"]
    assert _joint_by_name(mutated, "index_j0").limit.lower != _joint_by_name(hand, "index_j0").limit.lower
    assert _finger_by_name(mutated, "index").mount.pos != _finger_by_name(hand, "index").mount.pos


def test_hand_generator_executes_term_container_and_emits_samples_per_topology():
    """`HandGenerator` full 模式应对每个 pre-made topology 发出 `n_samples` 个 post-mutate 样本。"""

    result_list = list(
        HandGenerator(
            HandGeneratorCfg(
                mode="full",
                artifact_level="hand_cfg",
                handedness="right",
                hand_presets=["single_palm_allegro"],
                connectivity_presets=_single_full_pool(),
                mixed=False,
                missing=False,
                max_enumerate=3,
                n_samples=3,
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
            )
        ).generate_batch()
    )

    assert len(result_list) == 3
    assert all(result.hand_cfg is not None for result in result_list)
    assert all(result.metadata["topology_name"] == "right_t4_i4_m4_r4" for result in result_list)
    assert all("post_mutate_samples" in result.metadata for result in result_list)
