from source.anymani.anymani.assets.builder.finger_buiders import AllegroFingerBuilderCfg, RegularThumbBuilderCfg
from source.anymani.anymani.assets.builder.hand_builders import HumanLikeHandBuilderCfg
from source.anymani.anymani.assets.builder.palm_builders import ComPalmBuilderCfg
from source.anymani.anymani.assets.generator.hand_generator import HandGenerator, HandGeneratorCfg


def _generator_cfg(*, artifact_level: str) -> HandGeneratorCfg:
    return HandGeneratorCfg(
        mode="made",
        artifact_level=artifact_level,
        Made=HumanLikeHandBuilderCfg(
            name="demo",
            family="allegro",
            palm_cfg=ComPalmBuilderCfg(preset="allegro"),
            finger_cfg=AllegroFingerBuilderCfg(),
            thumb_cfg=RegularThumbBuilderCfg(),
        ),
    )


def test_generator_returns_hand_cfg_in_lightweight_mode():
    result = HandGenerator(_generator_cfg(artifact_level="hand_cfg")).generate()
    assert result is not None
    assert result.hand_cfg is not None
    assert result.urdf_path is None


def test_generator_bundle_mode_writes_outputs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    result = HandGenerator(_generator_cfg(artifact_level="bundle")).generate()

    assert result is not None
    assert result.urdf_path is not None
    assert result.sidecar_path is not None
    assert result.urdf_path.exists()
    assert result.sidecar_path.exists()
