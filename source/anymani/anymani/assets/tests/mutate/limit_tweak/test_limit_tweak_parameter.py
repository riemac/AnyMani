"""`limit_tweak` 参数与 mode 合同回归测试。"""

from __future__ import annotations

import math
from itertools import count
from unittest.mock import patch

import pytest

from assets.builder.hand_builders import HumanLikeHandBuilder, HumanLikeHandBuilderCfg
from assets.generator.mutate import LimitTweakCfg, LimitTweakMutator
from assets.presets import make_human_like_builder_cfg


def _make_allegro_builder_cfg() -> HumanLikeHandBuilderCfg:
    """构造一份稳定的 Allegro pre-made hand recipe。"""

    return make_human_like_builder_cfg(
        name="allegro_limit_tweak_demo",
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


def test_limit_tweak_mutator_consumes_sampled_values_and_preserves_valid_interval():
    """`limit_tweak` 应消费外部采样值，并保持 `lower < upper`。"""

    hand = _build_allegro_hand()
    before_index = _joint_by_name(hand, "index_j0").limit
    before_middle = _joint_by_name(hand, "middle_j0").limit

    mutated = LimitTweakMutator(
        LimitTweakCfg(
            disturb_object="shared",
            disturb_type="add",
            joint_range=(0.05, 0.05),
            clip={"abs": 0.1},
        )
    ).mutate(hand, sampled_params={"index_j0": 0.05})

    assert mutated is not None
    after_index = _joint_by_name(mutated, "index_j0").limit
    after_middle = _joint_by_name(mutated, "middle_j0").limit
    assert after_index.lower < after_index.upper
    assert not math.isclose(after_index.lower, before_index.lower, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(after_index.lower - before_index.lower, after_index.upper - before_index.upper, rel_tol=0.0, abs_tol=1e-12)
    assert after_middle.lower == before_middle.lower
    assert after_middle.upper == before_middle.upper


def test_limit_tweak_independent_upper_samples_do_not_share_last_joint_closure():
    r"""independent 模式下，每个 joint 的 upper 必须消费自己的采样值。"""

    hand = _build_allegro_hand()
    before_index = _joint_by_name(hand, "index_j0").limit
    before_middle = _joint_by_name(hand, "middle_j0").limit

    mutated = LimitTweakMutator(
        LimitTweakCfg(
            disturb_object="independent",
            disturb_type="add",
            joint_range=(-0.1, 0.1),
            clip={"abs": 0.1},
        )
    ).mutate(
        hand,
        sampled_params={
            "index_j0::lower": 0.01,
            "index_j0::upper": -0.02,
            "middle_j0::lower": -0.03,
            "middle_j0::upper": 0.04,
        },
    )

    assert mutated is not None
    after_index = _joint_by_name(mutated, "index_j0").limit
    after_middle = _joint_by_name(mutated, "middle_j0").limit
    assert math.isclose(after_index.lower - before_index.lower, 0.01, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(after_index.upper - before_index.upper, -0.02, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(after_middle.lower - before_middle.lower, -0.03, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(after_middle.upper - before_middle.upper, 0.04, rel_tol=0.0, abs_tol=1e-12)


def test_limit_tweak_identity_mode_is_explicit_noop_and_keeps_provenance():
    r"""`identity` 应显式保留为 accepted/output 锚点样本，而不是隐式“不采样”。"""

    hand = _build_allegro_hand()
    before_index = _joint_by_name(hand, "index_j0").limit

    mutated = LimitTweakMutator(
        LimitTweakCfg(
            self_mode="identity",
            joint_range=None,
        )
    ).mutate(
        hand,
        sampled_params={"sample": {"resolved_self_mode": "identity"}},
    )

    assert mutated is not None
    after_index = _joint_by_name(mutated, "index_j0").limit
    assert after_index.lower == before_index.lower
    assert after_index.upper == before_index.upper
    assert mutated.metadata["post_mutate_samples"]["limit_tweak"]["resolved_self_mode"] == "identity"


def test_limit_tweak_homologous_non_thumb_groups_by_family_and_joint_semantic():
    r"""`homologous_non_thumb` 应按 `(family, semantic)` 共享 non-thumb 扰动，thumb 独立。"""

    hand = _build_allegro_hand()
    hand.metadata["premade_topology"] = {
        "slot_family_map": {
            "thumb": "allegro",
            "index": "allegro",
            "middle": "allegro",
            "ring": "leap",
        }
    }

    sequence = count(start=1)

    def _fake_sampler(*_args, **_kwargs):
        return lambda: next(sequence) / 100.0

    with patch("assets.generator.mutate.limit_tweak._make_range_sampler", side_effect=_fake_sampler):
        sample = LimitTweakMutator(
            LimitTweakCfg(
                self_mode="homologous_non_thumb",
                disturb_object="independent",
                disturb_type="add",
                joint_range=(-0.2, 0.2),
            )
        ).sample_one_for_mode(hand, resolved_mode="homologous_non_thumb")

    joint_deltas = sample["joint_deltas"]
    assert joint_deltas["index_j0"] == joint_deltas["middle_j0"]
    assert joint_deltas["index_j0"] != joint_deltas["ring_j0"]
    assert joint_deltas["thumb_j0"] != joint_deltas["thumb_j1"]
    assert sample["homologous_groups"]["allegro:mcp1"]["joint_names"] == ["index_j0", "middle_j0"]
    assert sample["homologous_groups"]["leap:mcp1"]["joint_names"] == ["ring_j0"]


def test_limit_tweak_homologous_non_thumb_requires_slot_family_map():
    r"""缺少 `slot_family_map` 时必须 fail-hard，不能静默退回独立扰动。"""

    hand = _build_allegro_hand()
    hand.metadata = {}

    with pytest.raises(ValueError, match="slot_family_map"):
        LimitTweakMutator(
            LimitTweakCfg(
                self_mode="homologous_non_thumb",
                disturb_type="add",
                joint_range=(-0.1, 0.1),
            )
        ).sample_one_for_mode(hand, resolved_mode="homologous_non_thumb")
