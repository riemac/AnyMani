r"""official joint physical profile 回归测试。

这组测试锁住的是 pre-made 运行时的新物理契约：

1. `physical_presets.py` 中的数值来自官方 LEAP / Allegro URDF；
2. `get_finger_builder_preset(...)` 返回时已经注入 profile；
3. 绑定锚点是 child link 的 anatomy 语义，而不是会被 delete 后重排的 joint 名。
"""

from __future__ import annotations

import math

from assets.builder.hand_builders import HumanLikeHandBuilder
from assets.generator._connectivity_lowering import JointDeleteCfg, JointDeleteMutator
from assets.presets import get_finger_builder_preset, get_finger_physical_profile, make_human_like_builder_cfg


def _finger_by_name(hand, finger_name: str):
    r"""按 finger 名取 `FingerCfg`。"""

    for finger in hand.fingers:
        if finger.name == finger_name:
            return finger
    raise KeyError(finger_name)


def test_leap_physical_profile_keeps_official_limit_and_friction_values():
    r"""LEAP profile 应按真实串联语义保留官方物理属性。

    LEAP 官方 URDF 的 non-thumb joint 编号有一个很容易踩坑的地方：
    `0/4/8` 在文件里排在 `1/5/9` 前面，但真实 parent-child 串联上
    `1/5/9` 才是从 palm 接到 `mcp_joint*` 的近掌 slot。
    """

    profile = get_finger_physical_profile("leap_non_thumb_v1")
    assert [item.child_suffix for item in profile] == ["mcp1", "mcp2", "pip", "dip"]
    assert profile[0].source_joints == ("1", "5", "9")
    assert profile[1].source_joints == ("0", "4", "8")
    assert math.isclose(profile[0].limit.lower, -0.314, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(profile[0].limit.upper, 2.23, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(profile[0].limit.effort, 0.95, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(profile[0].limit.velocity, 8.48, rel_tol=0.0, abs_tol=1e-12)
    assert profile[0].friction == 0.0


def test_finger_preset_injects_physical_profile_before_build():
    r"""finger preset 返回的 builder cfg 应已携带 official joint physical profile。"""

    cfg = get_finger_builder_preset("leap_non_thumb_v1").replace(name="index", parent_link="palm")
    finger = cfg.class_type(cfg).build()
    joint_by_child = {joint.child.removeprefix("index_"): joint for joint in finger.joints if joint.joint_type == "revolute"}

    assert math.isclose(joint_by_child["mcp1"].limit.lower, -0.314, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(joint_by_child["mcp1"].limit.upper, 2.23, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(joint_by_child["mcp2"].limit.lower, -1.047, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(joint_by_child["mcp2"].limit.upper, 1.047, rel_tol=0.0, abs_tol=1e-12)
    assert joint_by_child["mcp1"].joint_properties.friction == 0.0


def test_connectivity_lowering_preserves_physical_profile_by_child_link_semantics():
    r"""joint delete 后，surviving child link 的物理属性不应按重排后的 `j*` 漂移。"""

    hand = HumanLikeHandBuilder(
        make_human_like_builder_cfg(
            name="leap_physical_delete_demo",
            family="leap",
            handedness="right",
            palm_cfg="single_box_leap",
            finger_cfg="leap_non_thumb_v1",
            thumb_cfg="leap_thumb_v1",
        )
    ).build()

    mutated = JointDeleteMutator(
        JointDeleteCfg(
            target_finger="index",
            deleted_joints=("index_j1",),
            regroup_strategy="drop",
            respect_preset=False,
        )
    ).mutate(hand)

    assert mutated is not None
    after_index = _finger_by_name(mutated, "index")
    surviving = {joint.child.removeprefix("index_"): joint for joint in after_index.joints if joint.joint_type == "revolute"}
    assert [joint.name for joint in after_index.joints if joint.joint_type == "revolute"] == ["index_j0", "index_j1", "index_j2"]
    assert math.isclose(surviving["pip"].limit.lower, -0.506, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(surviving["dip"].limit.upper, 2.042, rel_tol=0.0, abs_tol=1e-12)
