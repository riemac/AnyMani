"""pre-made builder 主链测试。

这组测试的目标不是把所有参数空间一次测完，而是先把首轮最关键的
pre-made 闭环骨架锁住：

1. primitive joint builder 能 lower 出稳定的 `JointCfg`
2. regular finger builder 能产出合法 finger 链
3. palm builder 能产出可被 hand builder / exporter 消费的 `PalmCfg`
4. human-like hand builder 能把 palm + fingers 装配成 `HandCfg`

一旦这些测试稳定，后续再围绕 exporter / generator 继续往下补纵向测试。
"""

from __future__ import annotations

import math

from assets.builder.hand_builders import HumanLikeHandBuilder, HumanLikeHandBuilderCfg
from assets.builder.joint_builders_primitive import PrimJointBuilderCfg
from assets.builder.palm_builders import ComPalmBuilder, ComPalmBuilderCfg, SinglePalmBuilder, SinglePalmBuilderCfg
from assets.presets import get_finger_builder_preset, make_human_like_builder_cfg


def _make_allegro_hand_cfg() -> HumanLikeHandBuilderCfg:
    """构造一份稳定的 Allegro 风格 hand builder cfg。

    这里统一走：
    - `com_allegro` palm
    - `allegro_non_thumb_v1` 非拇指 preset
    - `allegro_thumb_v1` 拇指 preset

    这样可以把 hand-level mount preset、finger preset 和 palm preset
    一次性串起来，作为首轮纵向闭环的最小锚点。
    """

    return make_human_like_builder_cfg(
        name="allegro_demo",
        family="allegro",
        handedness="right",
        palm_cfg="com_allegro",
        finger_cfg="allegro_non_thumb_v1",
        thumb_cfg="allegro_thumb_v1",
    )


def test_primitive_joint_builder_builds_box_link():
    """box primitive 应能稳定 lower 成 joint-centric link 描述。"""

    cfg = PrimJointBuilderCfg(
        name="index_j0",
        parent="palm",
        child="index_link_0",
        joint_type="revolute",
        axis=(0.0, 1.0, 0.0),
        limit=(-1.2, 1.0),
        mesh={"type": "box", "length": 0.03, "width": 0.015, "height": 0.02},
    )

    joint = cfg.class_type(cfg).build()

    # 先锁住最核心的 schema lower 结果：名字、拓扑、几何类型、正质量。
    assert joint.name == "index_j0"
    assert joint.parent == "palm"
    assert joint.child == "index_link_0"
    assert joint.collisions[0].geometry.kind == "box"
    assert joint.visuals[0].geometry.kind == "box"
    assert joint.inertial.mass > 0.0


def test_primitive_joint_builder_builds_composite_tip():
    """复合 primitive tip 应保留 `fixed + is_tip + 两个几何体` 语义。"""

    cfg = PrimJointBuilderCfg(
        name="index_tip",
        parent="index_link_3",
        child="index_tip_link",
        joint_type="fixed",
        axis=(0.0, 0.0, 0.0),
        limit=None,
        mesh={"type": "cs", "radius": 0.01, "height": 0.012},
        is_tip=True,
    )

    joint = cfg.class_type(cfg).build()

    assert joint.joint_type == "fixed"
    assert joint.is_tip is True
    assert len(joint.collisions) == 2
    assert {collision.geometry.kind for collision in joint.collisions} == {"cylinder", "sphere"}


def test_regular_finger_preset_builds_allegro_chain():
    """Allegro 非拇指 preset 应生成 4 个运动关节 + 1 个 fixed tip joint。"""

    cfg = get_finger_builder_preset("allegro_non_thumb_v1").replace(name="index", parent_link="palm")
    finger = cfg.class_type(cfg).build()

    # 这里锁的是 current contract，而不是某个几何数值微调：
    # - 4 个 revolute + 1 个 fixed tip
    # - 命名和 parent/child 串联关系稳定
    assert finger.name == "index"
    assert len(finger.joints) == 5
    assert finger.dof_count == 4
    assert finger.joints[-1].joint_type == "fixed"
    assert finger.joints[-1].is_tip is True
    assert finger.joints[0].parent == "palm"
    assert all(current.parent == previous.child for previous, current in zip(finger.joints[:-1], finger.joints[1:]))


def test_regular_thumb_preset_builds_thumb_chain():
    """thumb preset 至少要锁住 CMC 特例链没有破坏整体串联结构。"""

    cfg = get_finger_builder_preset("allegro_thumb_v1").replace(name="thumb", parent_link="palm")
    finger = cfg.class_type(cfg).build()

    assert finger.name == "thumb"
    assert len(finger.joints) == 5
    assert finger.dof_count == 4
    assert finger.joints[-1].is_tip is True
    # 第二个关节是 thumb 链里最容易被特殊坐标约定影响的点；
    # 这里至少锁住它不是退化零偏移。
    assert not math.isclose(sum(abs(value) for value in finger.joints[1].origin.pos), 0.0, abs_tol=1e-9)


def test_single_palm_builder_builds_box_palm():
    """single palm 路线应产出可消费的 box palm。"""

    cfg = SinglePalmBuilderCfg(shape="box", width=0.10, length=0.08, height=0.04)
    palm = SinglePalmBuilder(cfg).build()

    assert palm.name == "palm"
    assert palm.collisions[0].geometry.kind == "box"
    assert palm.visuals[0].geometry.kind == "box"
    assert palm.metadata["shape"] == "box"
    # box palm 的质心当前约定在 `(0, length/2, 0)`。
    assert palm.inertial.origin.pos == (0.0, 0.04, 0.0)


def test_com_palm_builder_exposes_mount_metadata():
    """复合 palm 除几何本体外，还必须暴露 hand-level mount preset 元数据。"""

    palm = ComPalmBuilder(ComPalmBuilderCfg(preset="allegro")).build()

    assert len(palm.collisions) == 3
    assert palm.metadata["mount_preset"] == "allegro"
    assert set(palm.metadata["finger_mounts"]) == {"index", "middle", "ring", "thumb"}


def test_human_like_hand_builder_assembles_allegro_hand():
    """human-like hand builder 应把 palm、非拇指和 thumb 正确装配为整手。"""

    cfg = _make_allegro_hand_cfg()
    hand = HumanLikeHandBuilder(cfg).build()

    assert hand.name == "allegro_demo"
    assert hand.family == "allegro"
    assert hand.handedness == "right"
    assert [finger.name for finger in hand.fingers] == ["index", "middle", "ring", "thumb"]
    assert hand.dof_count == 16
    # index 的挂载点来自 Allegro mount preset，而不是零位 fallback。
    assert math.isclose(hand.fingers[0].mount.pos[1], 0.0435, rel_tol=0.0, abs_tol=1e-6)
