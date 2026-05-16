"""`tip_replace` 几何与 embodied tip spec 回归测试。"""

from __future__ import annotations

import math

from assets.builder.hand_builders import HumanLikeHandBuilder, HumanLikeHandBuilderCfg
from assets.generator.mutate import TipReplaceCfg, TipReplaceMutator
from assets.presets import make_human_like_builder_cfg


def _make_allegro_builder_cfg() -> HumanLikeHandBuilderCfg:
    """构造一份稳定的 Allegro pre-made hand recipe。"""

    return make_human_like_builder_cfg(
        name="allegro_tip_replace_demo",
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


def test_tip_replace_identity_records_sample_without_changing_tip_geometry():
    r"""`identity` mode 应显式记录 provenance，但不改变 tip embodiment。"""

    hand = _build_allegro_hand()
    before_tip_joint = _finger_by_name(hand, "index").tip_joint
    mutated = TipReplaceMutator(TipReplaceCfg(self_mode="identity")).mutate(hand)

    assert mutated is not None
    tip_joint = _finger_by_name(mutated, "index").tip_joint
    assert tip_joint.collisions[0].geometry.kind == before_tip_joint.collisions[0].geometry.kind
    assert tip_joint.visuals[0].geometry.kind == before_tip_joint.visuals[0].geometry.kind
    assert mutated.metadata["post_mutate_samples"]["tip_replace"]["resolved_self_mode"] == "identity"


def test_tip_replace_same_broadcasts_one_custom_tip_spec_to_all_fingers():
    r"""`same` mode 应把同一个 tip_type 和连续参数广播到所有目标 finger。"""

    hand = _build_allegro_hand()
    sample = {
        "resolved_self_mode": "same",
        "finger_specs": {
            finger.name: {"tip_type": "round", "scale": 1.05}
            for finger in hand.fingers
        },
    }

    mutated = TipReplaceMutator(
        TipReplaceCfg(
            self_mode="same",
            tip_range=["round"],
            scale=(1.05, 1.05),
        )
    ).mutate(hand, sampled_params={"sample": sample})

    assert mutated is not None
    tip_samples = mutated.metadata["post_mutate_samples"]["tip_replace"]["finger_specs"]
    assert {payload["tip_type"] for payload in tip_samples.values()} == {"round"}
    assert {collision.geometry.kind for collision in _finger_by_name(mutated, "index").tip_joint.collisions} == {"mesh"}
    assert _finger_by_name(mutated, "index").tip_joint.metadata["post_mutate_tip_type"] == "round"
    assert _finger_by_name(mutated, "middle").tip_joint.metadata["post_mutate_tip_scale"] == 1.05


def test_tip_replace_general_allows_per_finger_tip_types():
    r"""`general` mode 下每根 finger 可拥有不同 tip_type。"""

    hand = _build_allegro_hand()
    sample = {
        "resolved_self_mode": "general",
        "finger_specs": {
            "thumb": {"tip_type": "cs", "scale": 1.0, "radius": 0.012, "height": 0.012, "cs_ratio": 1.0},
            "index": {"tip_type": "round", "scale": 1.0},
            "middle": {"tip_type": "wedge", "scale": 1.0},
            "ring": {"tip_type": "thinner", "scale": 1.0},
        },
    }

    mutated = TipReplaceMutator(
        TipReplaceCfg(
            self_mode="general",
            tip_range=["cs", "round", "wedge", "thinner"],
            scale=(1.0, 1.0),
        )
    ).mutate(hand, sampled_params={"sample": sample})

    assert mutated is not None
    assert _finger_by_name(mutated, "thumb").tip_joint.collisions[0].geometry.kind == "cylinder"
    assert _finger_by_name(mutated, "index").tip_joint.collisions[0].geometry.kind == "mesh"
    assert _finger_by_name(mutated, "middle").tip_joint.metadata["post_mutate_tip_type"] == "wedge"
    assert _finger_by_name(mutated, "ring").tip_joint.metadata["post_mutate_tip_type"] == "thinner"


def test_tip_replace_applies_thumb_functional_phase_for_custom_mesh_tip():
    r"""post-mutate 把 thumb 从 `cs` 换到 custom mesh 时，也必须保留 thumb 功能相位。"""

    hand = _build_allegro_hand()
    sample = {
        "resolved_self_mode": "general",
        "finger_specs": {
            "thumb": {"tip_type": "leap_cube", "scale": 1.0},
            "index": {"tip_type": "leap_cube", "scale": 1.0},
            "middle": {"tip_type": "cs", "scale": 1.0, "radius": 0.012, "height": 0.012, "cs_ratio": 1.0},
            "ring": {"tip_type": "cs", "scale": 1.0, "radius": 0.012, "height": 0.012, "cs_ratio": 1.0},
        },
    }

    mutated = TipReplaceMutator(
        TipReplaceCfg(
            self_mode="general",
            tip_range=["cs", "leap_cube"],
            scale=(1.0, 1.0),
        )
    ).mutate(hand, sampled_params={"sample": sample})

    assert mutated is not None
    thumb_tip = _finger_by_name(mutated, "thumb").tip_joint
    index_tip = _finger_by_name(mutated, "index").tip_joint
    assert thumb_tip.collisions[0].geometry.kind == "mesh"
    assert index_tip.collisions[0].geometry.kind == "mesh"
    assert math.isclose(thumb_tip.collisions[0].origin.rpy[1], -math.pi, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(index_tip.collisions[0].origin.rpy[1], -math.pi / 2.0, rel_tol=0.0, abs_tol=1e-12)
    assert thumb_tip.metadata["thumb_functional_tip_phase_rpy"] == (0.0, -math.pi / 2.0, 0.0)
    assert thumb_tip.metadata["mesh_origin_rpy"] == thumb_tip.collisions[0].origin.rpy


def test_tip_replace_cs_ratio_keeps_radius_and_changes_height():
    r"""`cs_ratio` 应固定半径 $r$，按 $\lambda=h/r$ 改写圆柱高度。"""

    hand = _build_allegro_hand()
    before_tip = _finger_by_name(hand, "index").tip_joint
    before_radius = before_tip.collisions[0].geometry.radius
    sample = {
        "resolved_self_mode": "same",
        "finger_specs": {
            finger.name: {
                "tip_type": "cs",
                "scale": 1.0,
                "radius": before_radius,
                "height": before_radius * 1.5,
                "cs_ratio": 1.5,
            }
            for finger in hand.fingers
        },
    }

    mutated = TipReplaceMutator(
        TipReplaceCfg(
            self_mode="same",
            tip_range=["cs"],
            scale=(1.0, 1.0),
            cs_ratio={"abs": (1.5, 1.5)},
        )
    ).mutate(hand, sampled_params={"sample": sample})

    assert mutated is not None
    tip_joint = _finger_by_name(mutated, "index").tip_joint
    assert math.isclose(tip_joint.collisions[0].geometry.radius, before_radius, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(tip_joint.collisions[0].geometry.length, before_radius * 1.5, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(tip_joint.collisions[1].origin.pos[1], before_radius * 1.5, rel_tol=0.0, abs_tol=1e-12)
    assert tip_joint.inertial is not None
    assert tip_joint.inertial.mass > 0.0
    assert tip_joint.inertial.inertia.ixx > 0.0
