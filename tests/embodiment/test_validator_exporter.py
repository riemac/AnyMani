from pathlib import Path

import yaml

from source.anymani.anymani.assets.builder.finger_buiders import AllegroFingerBuilderCfg, RegularThumbBuilderCfg
from source.anymani.anymani.assets.builder.hand_builders import HumanLikeHandBuilder, HumanLikeHandBuilderCfg
from source.anymani.anymani.assets.builder.palm_builders import ComPalmBuilderCfg
from source.anymani.anymani.assets.exporter.hand_exporter import HandExporter, HandExporterCfg
from source.anymani.anymani.assets.exporter.sidecar import SidecarCfg, SidecarExporter
from source.anymani.anymani.assets.exporter.urdf_writer import UrdfWriter, UrdfWriterCfg
from source.anymani.anymani.assets.generator.hand_generator import HandGenerationResult
from source.anymani.anymani.assets.validator import FingerValidator, FingerValidatorCfg, HandValidator, HandValidatorCfg


def _build_demo_hand():
    return HumanLikeHandBuilder(
        HumanLikeHandBuilderCfg(
            name="demo",
            family="allegro",
            palm_cfg=ComPalmBuilderCfg(preset="allegro"),
            finger_cfg=AllegroFingerBuilderCfg(),
            thumb_cfg=RegularThumbBuilderCfg(),
        )
    ).build()


def test_hand_validator_accepts_normal_premade_hand():
    result = HandValidator(HandValidatorCfg()).validate(_build_demo_hand())
    assert result.passed is True
    assert result.errors == []


def test_finger_validator_flags_missing_tip():
    finger = _build_demo_hand().fingers[0]
    broken = finger.replace(joints=[joint.replace(is_tip=False) for joint in finger.joints])

    result = FingerValidator(FingerValidatorCfg()).validate(broken)
    assert any("expected exactly 1 tip joint" in warning for warning in result.warnings)


def test_hand_validator_warns_for_close_mounts():
    hand = _build_demo_hand()
    close_fingers = [
        hand.fingers[0].replace(mount=(0.0, 0.0, 0.0)),
        hand.fingers[1].replace(mount=(0.001, 0.0, 0.0)),
        *hand.fingers[2:],
    ]
    crowded = hand.replace(fingers=close_fingers)

    result = HandValidator(HandValidatorCfg()).validate(crowded)
    assert any("finger spacing" in warning for warning in result.warnings)


def test_urdf_writer_includes_mount_link_and_limit_tags():
    urdf = UrdfWriter(UrdfWriterCfg()).to_urdf_string(_build_demo_hand())

    assert "<robot name=\"demo\">" in urdf
    assert "index_mount_link" in urdf
    assert "<limit " in urdf
    assert "<collision" in urdf


def test_sidecar_exporter_writes_provenance(tmp_path):
    out_dir = tmp_path / "bundle"
    result = SidecarExporter(SidecarCfg(experiment_tag="unit-test")).export(
        _build_demo_hand(),
        out_dir,
        extra={"id": "demo-id", "recipe_hash": "abc123", "seed": 7},
    )
    sidecar = yaml.safe_load(result.written[0].read_text(encoding="utf-8"))

    assert sidecar["id"] == "demo-id"
    assert sidecar["provenance"]["recipe_hash"] == "abc123"
    assert sidecar["provenance"]["experiment_tag"] == "unit-test"


def test_hand_exporter_bundle_writes_all_artifacts(tmp_path):
    result = HandGenerationResult(hand_cfg=_build_demo_hand(), metadata={"id": "sample"})
    export_result = HandExporter(HandExporterCfg(artifact_level="bundle")).export(result, tmp_path)

    written_names = {path.name for path in export_result.written}
    assert {"hand.urdf", "hand.yaml", "tree.txt", "tree.mmd"} <= written_names
    assert result.urdf_path is not None
    assert result.sidecar_path is not None
