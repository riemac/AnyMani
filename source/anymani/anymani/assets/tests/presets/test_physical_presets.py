r"""official joint physical profile 回归测试。

这组测试锁住的是 pre-made 运行时的新物理契约：

1. `physical_presets.py` 中的数值来自官方 LEAP / Allegro URDF；
2. `get_finger_builder_preset(...)` 返回时已经注入 profile；
3. 绑定锚点是 child link 的 anatomy 语义，而不是会被 delete 后重排的 joint 名。
"""

from __future__ import annotations

import math
from pathlib import Path

from assets.builder.hand_builders import HumanLikeHandBuilder
from assets.generator.premade.connectivity_lowering import JointDeleteCfg, JointDeleteMutator
from assets.presets._physical_profile_extractor import extract_profile
from assets.presets import get_finger_builder_preset, get_finger_physical_profile, make_human_like_builder_cfg


REPO_ROOT = Path(__file__).resolve().parents[6]
OFFICIAL_URDF_PATHS = {
    "leap": REPO_ROOT / "source" / "anymani" / "assets" / "hands" / "leap_hand" / "leap_hand_right.urdf",
    "allegro": REPO_ROOT / "source" / "anymani" / "assets" / "hands" / "allegro_hand" / "allegro_hand_right.urdf",
}
PROFILE_CASES = (
    (
        "leap_non_thumb_v1",
        "leap",
        {
            "mcp1": ("1", "5", "9"),
            "mcp2": ("0", "4", "8"),
            "pip": ("2", "6", "10"),
            "dip": ("3", "7", "11"),
        },
    ),
    (
        "leap_thumb_v1",
        "leap",
        {
            "cmc1": ("12",),
            "cmc2": ("13",),
            "mcp": ("14",),
            "dip": ("15",),
        },
    ),
    (
        "allegro_non_thumb_v1",
        "allegro",
        {
            "mcp1": ("joint_0.0", "joint_4.0", "joint_8.0"),
            "mcp2": ("joint_1.0", "joint_5.0", "joint_9.0"),
            "pip": ("joint_2.0", "joint_6.0", "joint_10.0"),
            "dip": ("joint_3.0", "joint_7.0", "joint_11.0"),
        },
    ),
    (
        "allegro_thumb_v1",
        "allegro",
        {
            "cmc1": ("joint_12.0",),
            "cmc2": ("joint_13.0",),
            "mcp": ("joint_14.0",),
            "dip": ("joint_15.0",),
        },
    ),
)


def _finger_by_name(hand, finger_name: str):
    r"""按 finger 名取 `FingerCfg`。"""

    for finger in hand.fingers:
        if finger.name == finger_name:
            return finger
    raise KeyError(finger_name)


def test_all_physical_profiles_match_official_urdf_slots():
    r"""全部 Python profile 都应逐槽对齐官方 URDF source joints。

    这个测试不是运行时依赖路径；它只在测试阶段调用离线 extractor，用来防止
    `joint 名字排序` 和 `真实 parent-child 串联顺序` 再次混淆。尤其 LEAP
    non-thumb 中 `mcp1 <- joint 1/5/9`、`mcp2 <- joint 0/4/8` 这一点必须锁死。
    """

    for preset_name, family, mapping in PROFILE_CASES:
        extracted = extract_profile(OFFICIAL_URDF_PATHS[family], mapping)
        profile_by_suffix = {item.child_suffix: item for item in get_finger_physical_profile(preset_name)}
        assert set(profile_by_suffix) == set(mapping)
        for child_suffix, source_joints in mapping.items():
            profile_item = profile_by_suffix[child_suffix]
            extracted_item = extracted[child_suffix][0]
            assert profile_item.source_joints == source_joints
            assert math.isclose(profile_item.limit.lower, extracted_item.lower, rel_tol=0.0, abs_tol=1e-12)
            assert math.isclose(profile_item.limit.upper, extracted_item.upper, rel_tol=0.0, abs_tol=1e-12)
            assert math.isclose(profile_item.limit.effort, extracted_item.effort, rel_tol=0.0, abs_tol=1e-12)
            assert math.isclose(profile_item.limit.velocity, extracted_item.velocity, rel_tol=0.0, abs_tol=1e-12)
            assert profile_item.friction == extracted_item.friction


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
