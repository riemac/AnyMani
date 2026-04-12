import pytest

from source.anymani.anymani.assets.builder.finger_buiders import (
    AllegroFingerBuilderCfg,
    FINGER_PRESET_REGISTRY,
    LeapFingerBuilderCfg,
    RegularFingerBuilder,
    RegularThumbBuilderCfg,
    get_finger_builder_preset,
)
from source.anymani.anymani.assets.builder.hand_builders import HumanLikeHandBuilder, HumanLikeHandBuilderCfg
from source.anymani.anymani.assets.builder.palm_builders import (
    ComPalmBuilderCfg,
    PALM_PRESET_REGISTRY,
    SinglePalmBuilder,
    SinglePalmBuilderCfg,
)
from source.anymani.anymani.assets.asset_schema_core import PoseCfg


def test_single_palm_box_uses_bottom_center_frame():
    palm = SinglePalmBuilder(
        SinglePalmBuilderCfg(shape="box", width=0.12, length=0.08, height=0.046)
    ).build()

    collision = palm.collisions[0]
    assert collision.geometry.kind == "box"
    assert collision.origin.pos == (0.0, 0.04, 0.0)


def test_allegro_cfg_normalizes_offsets_and_tip_units():
    cfg = AllegroFingerBuilderCfg(name="index")
    assert cfg._mesh_offsets_6d[2][1] == -0.006
    assert cfg.tip["radius"] == 0.012
    assert cfg.tip["height"] == 0.01


def test_finger_preset_registry_exposes_named_builder_cfgs():
    assert "allegro_non_thumb_v1" in FINGER_PRESET_REGISTRY
    preset = get_finger_builder_preset("allegro_non_thumb_v1")
    assert isinstance(preset, AllegroFingerBuilderCfg)
    assert preset.name == "index"


def test_palm_preset_registry_exposes_named_factories():
    palm_cfg = PALM_PRESET_REGISTRY["com_allegro"]()
    assert isinstance(palm_cfg, ComPalmBuilderCfg)
    assert palm_cfg.preset == "allegro"


def test_regular_thumb_cfg_keeps_cmc1_joint_center_semantics():
    cfg = RegularThumbBuilderCfg(name="thumb")
    assert cfg.mesh_shape[0]["center_on_joint"] is True
    assert cfg._mesh_offsets_6d[0][1] == pytest.approx(0.009)
    assert cfg._mesh_offsets_6d[0][2] == pytest.approx(0.0145)


def test_allegro_finger_builds_chain_with_fixed_tip():
    finger = RegularFingerBuilder(AllegroFingerBuilderCfg(name="index")).build()

    assert finger.name == "index"
    assert len(finger.joints) == 5
    assert finger.joints[-1].joint_type == "fixed"
    assert finger.joints[-1].is_tip is True
    assert finger.joints[1].origin.pos[1] == pytest.approx(0.018)


def test_leap_finger_inserts_fixed_part_before_first_revolute_joint():
    finger = RegularFingerBuilder(LeapFingerBuilderCfg(name="index")).build()

    assert finger.joints[0].origin.pos == pytest.approx((0.0, 0.013, 0.0))
    assert finger.joints[-1].is_tip is True


def test_human_like_builder_prefers_explicit_mounts_over_preset_mounts():
    explicit_mount = PoseCfg(pos=(0.1, 0.2, 0.3), rpy=(0.0, 0.1, 0.2))
    hand = HumanLikeHandBuilder(
        HumanLikeHandBuilderCfg(
            name="demo",
            family="allegro",
            palm_cfg=ComPalmBuilderCfg(preset="allegro"),
            finger_cfg=AllegroFingerBuilderCfg(),
            thumb_cfg=RegularThumbBuilderCfg(),
            mounts={"index": explicit_mount},
        )
    ).build()

    index_finger = next(finger for finger in hand.fingers if finger.name == "index")
    thumb_finger = next(finger for finger in hand.fingers if finger.name == "thumb")
    assert index_finger.mount.pos == explicit_mount.pos
    assert index_finger.mount.rpy == explicit_mount.rpy
    assert thumb_finger.mount.pos == (-0.0182, 0.019333, -0.045987)


def test_human_like_builder_accepts_string_finger_presets():
    hand = HumanLikeHandBuilder(
        HumanLikeHandBuilderCfg(
            name="demo",
            family="allegro",
            palm_cfg=ComPalmBuilderCfg(preset="allegro"),
            finger_cfg="allegro_non_thumb_v1",
            thumb_cfg="allegro_thumb_v1",
        )
    ).build()
    assert [finger.name for finger in hand.fingers] == ["index", "middle", "ring", "thumb"]
