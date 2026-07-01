from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[6]
INHAND_ENV_CFG = REPO_ROOT / "source/anymani/anymani/tasks/inhand/inhand_env_cfg.py"
LEAPHAND_ADR_CFG = REPO_ROOT / "source/anymani/anymani/tasks/inhand/config/leaphand/leaphand_adr_env_cfg.py"
GENERATED_ADR_CFG = (
    REPO_ROOT
    / "source/anymani/anymani/tasks/inhand/config/generated_right_t4_i4_m4_r4/generated_right_t4_i4_m4_r4_adr_env_cfg.py"
)
ROUND_BASE_CFG = REPO_ROOT / "source/anymani/anymani/tasks/inhand/config/leaphand_round/inhand_round_base_env_cfg.py"

GM_SKY_HDR_SUFFIX = "/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"


def _read(path: Path) -> str:
    r"""Read a source file for text-level visual-scene contract checks."""

    return path.read_text(encoding="utf-8")


def test_inhand_base_scene_uses_gm_style_sky_light_constants() -> None:
    r"""The inhand shared scene anchor should expose the GM-style sky light constants."""

    source = _read(INHAND_ENV_CFG)

    assert "INHAND_CLEAR_SKY_TEXTURE_FILE" in source
    assert "INHAND_CLEAR_SKY_LIGHT_INTENSITY = 750.0" in source
    assert GM_SKY_HDR_SUFFIX in source
    assert 'prim_path="/World/skyLight"' in source
    assert "texture_file=INHAND_CLEAR_SKY_TEXTURE_FILE" in source


def test_inhand_special_scenes_reuse_gm_style_sky_light() -> None:
    r"""Custom scenes that do not rely purely on the shared base should still use the same skyLight/HDR."""

    for path in (LEAPHAND_ADR_CFG, GENERATED_ADR_CFG, ROUND_BASE_CFG):
        source = _read(path)
        assert 'prim_path="/World/skyLight"' in source, path
        assert "INHAND_CLEAR_SKY_LIGHT_INTENSITY" in source, path
        assert "INHAND_CLEAR_SKY_TEXTURE_FILE" in source, path


def test_generated_official_adr_restores_hand_visual_materials() -> None:
    r"""Generated official-ADR should enable visual material restore for correct hand colors."""

    source = _read(GENERATED_ADR_CFG)

    assert "restore_visual_materials=True" in source
    assert "restore_visual_materials=False" not in source
