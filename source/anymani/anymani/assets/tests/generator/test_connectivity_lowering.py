r"""pre-made connectivity lowering 回归测试。

这里锁住的是当前 `JointDeleteMutator` 作为 connectivity lower 执行器的两类核心语义：

1. 显式删除指定 joint；
2. 删除后 finger 链重新接通；
3. `merge` 会把被删 joint 的几何并入上一保留容器；
4. `drop` 会按“配置项消失后的物理缩短”语义重连 surviving 链；
5. 这条工具服务 pre-made connectivity lower，而不是 post-mutate 容器。

# NOTE:
第 4 条正是这轮新增的关键保护：
当删除中间 joint / child-link 时，剩余的 distal joint / tip 不应继续保留
被删段累计长度，而应接回“第一段被删 joint 的挂接位姿”。
"""

from __future__ import annotations

from assets.builder.hand_builders import HumanLikeHandBuilder, HumanLikeHandBuilderCfg
from assets.generator.premade.connectivity_lowering import JointDeleteCfg, JointDeleteMutator
from assets.presets import make_human_like_builder_cfg


def _make_allegro_builder_cfg() -> HumanLikeHandBuilderCfg:
    """构造一份稳定的 Allegro pre-made hand recipe。"""

    return make_human_like_builder_cfg(
        name="allegro_joint_delete_demo",
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


def test_joint_delete_mutator_relinks_chain_and_merges_deleted_geometry():
    """`joint_delete` 应删除目标 joint、重接链、压紧 joint 名，并把几何并到上一保留容器。"""

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
    assert [joint.name for joint in after_index.joints] == ["index_j0", "index_j1", "index_j2", "index_tip"]
    assert [joint.child for joint in after_index.joints] == ["index_mcp1", "index_mcp2", "index_dip", "index_tip"]
    assert all(current.parent == previous.child for previous, current in zip(after_index.joints[:-1], after_index.joints[1:]))
    assert len(after_index.joints[1].collisions) > before_collision_count
    assert mutated.dof_count == hand.dof_count - 1


def test_joint_delete_drop_relinks_tip_to_first_deleted_joint_origin():
    r"""`drop` 应按物理缩短语义把 tip 接回第一段被删 joint 的挂接位姿。

    当前用 Allegro index 测这个最清楚：

    - 原链：`j0 -> j1 -> j2 -> j3 -> tip`
    - 删除：`j2, j3`
    - 目标：`tip` 直接接到 `j1` 后面

    这里最关键的不是“链还连着”这么弱的条件，而是：
    `tip.origin` 必须等于原始 `j2.origin`，而不是 `j2 + j3 + tip` 的累计位姿。
    """

    hand = _build_allegro_hand()
    before_index = _finger_by_name(hand, "index")
    original_first_deleted_origin = before_index.joints[2].origin.copy()  # `j2.origin`：删除序列中的第一段挂接位姿
    original_tip_origin = before_index.joints[-1].origin.copy()  # 原 tip 仍挂在 `j3` 之后，只用于防止误回退

    mutated = JointDeleteMutator(
        JointDeleteCfg(
            target_finger="index",
            deleted_joints=("index_j2", "index_j3"),
            regroup_strategy="drop",
            respect_preset=False,
        )
    ).mutate(hand)

    assert mutated is not None
    after_index = _finger_by_name(mutated, "index")
    assert [joint.name for joint in after_index.joints] == ["index_j0", "index_j1", "index_tip"]
    assert all(current.parent == previous.child for previous, current in zip(after_index.joints[:-1], after_index.joints[1:]))
    assert after_index.joints[-1].parent == after_index.joints[1].child
    assert after_index.joints[-1].origin.pos == original_first_deleted_origin.pos
    assert after_index.joints[-1].origin.rpy == original_first_deleted_origin.rpy
    assert after_index.joints[-1].origin.pos != original_tip_origin.pos


def test_joint_delete_drop_reanchors_remaining_chain_to_mount_when_root_joint_is_deleted():
    r"""删除指根 joint 时，剩余链的第一个 surviving joint 必须贴回 finger 挂载点。

    虽然正式合法 pre-made 变体一般不会删指根 joint，
    但 `JointDeleteMutator` 作为通用工具，仍必须保证：

    - 若删除第一段运动关节；
    - 下一段 surviving joint 会重新挂到 `finger.parent_link`；
    - 且其 origin 应等于“第一段被删 joint 的挂接位姿”。

    对 Allegro 而言，`j0.origin = 0`，因此删除 `j0` 后，
    新的首 joint `j1` 应直接贴回 `palm` 挂载点。
    """

    hand = _build_allegro_hand()
    before_index = _finger_by_name(hand, "index")
    original_root_origin = before_index.joints[0].origin.copy()  # Allegro `j0` 默认就是 finger 挂载点自身

    mutated = JointDeleteMutator(
        JointDeleteCfg(
            target_finger="index",
            deleted_joints=("index_j0",),
            regroup_strategy="drop",
            respect_preset=False,
        )
    ).mutate(hand)

    assert mutated is not None
    after_index = _finger_by_name(mutated, "index")
    assert [joint.name for joint in after_index.joints] == ["index_j0", "index_j1", "index_j2", "index_tip"]
    assert [joint.child for joint in after_index.joints] == ["index_mcp2", "index_pip", "index_dip", "index_tip"]
    assert after_index.joints[0].parent == after_index.parent_link
    assert after_index.joints[0].origin.pos == original_root_origin.pos
    assert after_index.joints[0].origin.rpy == original_root_origin.rpy
