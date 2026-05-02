"""validator error / warning 契约测试。

这组测试专门回答一个问题：在 pre-made 首轮闭环里，validator 到底把哪些问题
视为 warning，哪些在 strict 模式下必须升级成 error。

这类测试的价值不在于“多测几个 if 分支”，而在于给后续的层次化枚举和自动筛选
固定住一个稳定语义边界，避免后面出现“同一个产物在不同轮实验里被不同标准判掉”。
"""

from __future__ import annotations

from assets.builder.hand_builders import HumanLikeHandBuilder, HumanLikeHandBuilderCfg
from assets.builder.joint_builders_primitive import PrimJointBuilderCfg
from assets.generator._connectivity_lowering import JointDeleteCfg, JointDeleteMutator
from assets.presets import get_finger_builder_preset, make_human_like_builder_cfg
from assets.validator.finger_rules import FingerValidator, FingerValidatorCfg
from assets.validator.hand_rules import HandValidator, HandValidatorCfg
from assets.validator.joint_rules import JointValidator, JointValidatorCfg


def _build_allegro_hand():
    """构造一份稳定的 Allegro 整手，用于 hand-level validator 测试。"""

    cfg = make_human_like_builder_cfg(
        name="allegro_demo",
        family="allegro",
        handedness="right",
        palm_cfg="com_allegro",
        finger_cfg="allegro_non_thumb_v1",
        thumb_cfg="allegro_thumb_v1",
    )
    return HumanLikeHandBuilder(cfg).build()


def test_joint_validator_allows_zero_origin_when_metadata_explicitly_marks_it():
    """root joint 的零位 origin 在显式豁免时不应被误判成退化 link。"""

    cfg = PrimJointBuilderCfg(
        name="index_j0",
        parent="palm",
        child="index_link_0",
        joint_type="revolute",
        origin=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        axis=(0.0, 1.0, 0.0),
        limit=(-1.0, 1.0),
        mesh={"type": "box", "length": 0.03, "width": 0.015, "height": 0.02},
        metadata={"allow_zero_origin": True},
    )

    result = JointValidator(JointValidatorCfg()).validate(cfg.class_type(cfg).build())

    assert result.passed is True
    assert result.warnings == []


def test_joint_validator_upgrades_large_limit_warning_in_strict_mode():
    """limit range 超过建议值时，默认是 warning，strict 下要升级成 error。"""

    cfg = PrimJointBuilderCfg(
        name="index_j1",
        parent="index_link_0",
        child="index_link_1",
        joint_type="revolute",
        origin=(0.0, 0.03, 0.0, 0.0, 0.0, 0.0),
        axis=(1.0, 0.0, 0.0),
        limit=(-4.0, 4.0),
        mesh={"type": "box", "length": 0.03, "width": 0.015, "height": 0.02},
    )
    joint = cfg.class_type(cfg).build()

    relaxed = JointValidator(JointValidatorCfg(strict=False)).validate(joint)
    strict = JointValidator(JointValidatorCfg(strict=True)).validate(joint)

    assert relaxed.passed is True
    assert any("limit range" in warning for warning in relaxed.warnings)
    assert strict.passed is False
    assert any("limit range" in error for error in strict.errors)


def test_finger_validator_treats_missing_tip_flag_as_warning():
    """tip 唯一性在当前阶段是软约束，应先记 warning 而不是直接拒绝。"""

    finger_cfg = get_finger_builder_preset("allegro_non_thumb_v1").replace(name="index", parent_link="palm")
    finger = finger_cfg.class_type(finger_cfg).build()
    broken_tip_finger = finger.replace(
        joints=[*finger.joints[:-1], finger.joints[-1].replace(is_tip=False)]
    )

    result = FingerValidator(FingerValidatorCfg()).validate(broken_tip_finger)

    assert result.passed is True
    assert any("expected exactly 1 tip joint" in warning for warning in result.warnings)


def test_hand_validator_upgrades_dof_warning_to_error_in_strict_mode():
    """hand-level 的 warning 在 strict 模式下必须升级，便于枚举筛选直接拒绝。"""

    hand = _build_allegro_hand()
    relaxed = HandValidator(
        HandValidatorCfg(
            post_mutate=HandValidatorCfg.PostMutateCfg(
                dof_max=8,
                check_finger_spacing=False,
                strict=False,
            )
        )
    ).validate(hand)
    strict = HandValidator(
        HandValidatorCfg(
            post_mutate=HandValidatorCfg.PostMutateCfg(
                dof_max=8,
                check_finger_spacing=False,
                strict=True,
            )
        )
    ).validate(hand)

    assert relaxed.passed is True
    assert any("dof" in warning for warning in relaxed.warnings)
    assert strict.passed is False
    assert any("dof" in error for error in strict.errors)


def test_hand_validator_rejects_missing_thumb_under_current_premade_contract():
    """当前 pre-made 范围内，缺拇指 topology 必须直接被 hand-level validator 拒绝。"""

    hand = HumanLikeHandBuilder(
        make_human_like_builder_cfg(
            name="no_thumb_demo",
            family="allegro",
            handedness="right",
            palm_cfg="com_allegro",
            finger_cfg="allegro_non_thumb_v1",
            thumb_cfg="allegro_thumb_v1",
        ).replace(thumb_cfg=None)
    ).build()

    result = HandValidator(
        HandValidatorCfg(
            pre_made=HandValidatorCfg.PreMadeCfg(
                check_finger_spacing=False,
            )
        )
    ).validate_pre_made(hand)

    assert result.passed is False
    assert any("missing required thumb" in error for error in result.errors)


def test_hand_validator_rejects_when_all_non_thumb_fingers_drop_below_three_revolute_dof():
    """若所有 non-thumb 都被裁到 <3 DOF，则当前科研边界下应直接拒绝。"""

    hand = _build_allegro_hand()
    for finger_name in ("index", "middle", "ring"):
        mutated = JointDeleteMutator(
            JointDeleteCfg(
                target_finger=finger_name,
                deleted_joints=(f"{finger_name}_j2", f"{finger_name}_j3"),
                regroup_strategy="drop",
                respect_preset=False,
            )
        ).mutate(hand)
        assert mutated is not None
        hand = mutated

    result = HandValidator(
        HandValidatorCfg(
            pre_made=HandValidatorCfg.PreMadeCfg(
                check_finger_spacing=False,
            )
        )
    ).validate_pre_made(hand)

    assert result.passed is False
    assert any("at least one non-thumb finger" in error for error in result.errors)


def test_hand_validator_pre_made_rejects_palm_thumb_family_mismatch():
    """pre-made mixed 中，thumb family 必须与 palm family 一致。"""

    hand = _build_allegro_hand().replace(
        metadata={
            "premade_topology": {
                "base_hand_preset": "single_palm_allegro",
                "handedness": "right",
                "topology_kind": "mixed",
                "topology_anchor": "mixed",
                "topology_name": "right_leap_t4_allegro_i4_m4_r4",
                "surviving_slots": ["thumb", "index", "middle", "ring"],
                "slot_finger_presets": {
                    "thumb": "leap_thumb_v1",
                    "index": "allegro_non_thumb_v1",
                    "middle": "allegro_non_thumb_v1",
                    "ring": "allegro_non_thumb_v1",
                },
                "slot_family_map": {
                    "thumb": "leap",
                    "index": "allegro",
                    "middle": "allegro",
                    "ring": "allegro",
                },
            }
        }
    )

    result = HandValidator(HandValidatorCfg()).validate_pre_made(hand)

    assert result.passed is False
    assert any("thumb family" in error for error in result.errors)
