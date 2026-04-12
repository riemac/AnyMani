"""`joint_delete` 回归测试。

这里锁住的是当前首轮已经实现的保守语义：

1. 显式删除指定 joint
2. 删除后 finger 链重新接通
3. `merge` 会把被删 joint 的几何并入上一保留容器
4. `HandGenerator` 已能在 full 模式下执行这条结构级 mutate
"""

from __future__ import annotations

from assets.builder.hand_builders import HumanLikeHandBuilder, HumanLikeHandBuilderCfg
from assets.builder.palm_builders import ComPalmBuilderCfg
from assets.generator.hand_generator import HandGenerator, HandGeneratorCfg
from assets.generator.mutate import HandMutatorCfg, JointDeleteCfg, JointDeleteMutator


def _make_allegro_builder_cfg() -> HumanLikeHandBuilderCfg:
    """构造一份稳定的 Allegro pre-made hand recipe。"""

    return HumanLikeHandBuilderCfg(
        name="allegro_joint_delete_demo",
        family="allegro",
        handedness="right",
        palm_cfg=ComPalmBuilderCfg(preset="allegro"),
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


def test_joint_delete_mutator_relinks_chain_and_merges_deleted_geometry():
    """`joint_delete` 应删除目标 joint、重接链，并把几何并到上一保留容器。"""

    hand = _build_allegro_hand()
    before_index = _finger_by_name(hand, "index")
    before_collision_count = len(before_index.joints[1].collisions)

    mutated = JointDeleteMutator(
        JointDeleteCfg(
            target_finger="index",
            deleted_joints=("index_j2",),
            regroup_strategy="merge",
        )
    ).mutate(hand)

    assert mutated is not None
    after_index = _finger_by_name(mutated, "index")
    assert [joint.name for joint in after_index.joints] == ["index_j0", "index_j1", "index_j3", "index_tip"]
    assert all(current.parent == previous.child for previous, current in zip(after_index.joints[:-1], after_index.joints[1:]))
    assert len(after_index.joints[1].collisions) > before_collision_count
    assert mutated.dof_count == hand.dof_count - 1


def test_hand_generator_executes_joint_delete_pipeline():
    """`HandGenerator` 在 full 模式下应能执行 `joint_delete`。"""

    result = HandGenerator(
        HandGeneratorCfg(
            mode="full",
            artifact_level="hand_cfg",
            Made=_make_allegro_builder_cfg(),
            Mutate=HandMutatorCfg(
                joint_delete=JointDeleteCfg(
                    target_finger="index",
                    deleted_joints=("index_j2",),
                    regroup_strategy="merge",
                ),
                order=("joint_delete",),
            ),
        )
    ).generate()

    assert result is not None
    assert result.hand_cfg is not None
    assert result.hand_cfg.dof_count == 15
    assert [joint.name for joint in _finger_by_name(result.hand_cfg, "index").joints] == ["index_j0", "index_j1", "index_j3", "index_tip"]
