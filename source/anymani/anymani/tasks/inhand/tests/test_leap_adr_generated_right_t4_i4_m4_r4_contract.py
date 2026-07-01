from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[6]
GENERATED_CFG = (
    REPO_ROOT
    / "source/anymani/anymani/tasks/inhand/config/generated_right_t4_i4_m4_r4/"
    / "generated_right_t4_i4_m4_r4_adr_env_cfg.py"
)
GENERATED_REGISTER = REPO_ROOT / "source/anymani/anymani/tasks/inhand/config/generated_right_t4_i4_m4_r4/__init__.py"
LEAPHAND_USD_REGISTER = REPO_ROOT / "source/anymani/anymani/tasks/inhand/config/leaphand/__init__.py"
LEAPHAND_URDF_REGISTER = REPO_ROOT / "source/anymani/anymani/tasks/inhand/config/leaphand_urdf/__init__.py"

def _read(path: Path) -> str:
    r"""Read a source file for text-level contract assertions."""

    return path.read_text(encoding="utf-8")


def test_generated_env_registers_new_non_tactile_ids_without_overwriting_existing_ids() -> None:
    r"""Generated probe must be a new official-ADR id and must not rename accepted USD/URDF ids."""

    generated_register = _read(GENERATED_REGISTER)
    usd_register = _read(LEAPHAND_USD_REGISTER)
    urdf_register = _read(LEAPHAND_URDF_REGISTER)

    assert "AnyMani-LeapHand-ADR-Generated-right_t4_i4_m4_r4-v0" in generated_register
    assert "AnyMani-LeapHand-ADR-Generated-right_t4_i4_m4_r4-Play-v0" in generated_register
    assert "Tactile" not in generated_register
    assert "rl_games_ppo_cfg_official_adr.yaml" in generated_register

    assert "AnyMani-LeapHand-Tactile-ADR-v0" in usd_register
    assert "AnyMani-LeapHand-Tactile-ADR-URDF-v0" in urdf_register


def test_generated_env_reuses_n010_official_adr_mdp_and_replaces_asset_specific_terms() -> None:
    r"""Generated cfg should inherit official ADR semantics rather than GM single-asset MDP."""

    source = _read(GENERATED_CFG)

    assert "LeapHandTactileADREnvCfg" in source
    assert "LeapHandOfficialADRCommandsCfg" in source
    assert "LeapHandOfficialADRRewardsCfg" in source
    assert "LeapHandOfficialADRTerminationsCfg" in source
    assert "LeapHandOfficialADRCurriculumCfg" in source
    assert "OfficialADRTargetJointPositionActionCfg" in source
    assert "scale=1.0 / 24.0" in source

    assert "RelativeJointPositionActionCfg" not in source
    assert "scale=0.1" not in source
    assert "GmSingleAssetRewardsCfg" not in source
    assert "ReorientCommandCfg" not in source


def test_generated_env_consumes_expected_bundle_and_latest_preset() -> None:
    r"""The env must lock to the accepted generated bundle and latest preset path.

    Preset YAML files are intentionally ignored by the AnyMani code repo because they
    are local calibration artifacts. This contract therefore checks the code-level
    binding to the expected preset path, but does not require the ignored YAML file
    to be present in a fresh clone.
    """

    source = _read(GENERATED_CFG)

    assert "right_t4_i4_m4_r4" in source
    assert "GENERATED_RIGHT_T4_I4_M4_R4_BUNDLE_ID" in source
    assert "asset_preset_path(\"generated_asset\", \"right_t4_i4_m4_r4\")" in source
    assert "expected_hand_source=\"generated_bundle\"" in source
    assert "expected_hand_ref_contains=\"right_t4_i4_m4_r4\"" in source


def test_generated_object_source_scale_and_pose_are_the_current_dexcube_preset() -> None:
    r"""Object init must come from the accepted DexCube 1.2 generated preset path."""

    source = _read(GENERATED_CFG)

    assert "dex_cube_usd" in source
    assert "DexCube/dex_cube_instanceable.usd" in source
    assert "scale=GENERATED_RIGHT_T4_I4_M4_R4_OBJECT_SCALE" in source
    assert "pos=GENERATED_RIGHT_T4_I4_M4_R4_GRASP_PRESET.object_pos_cfg" in source
    assert "rot=GENERATED_RIGHT_T4_I4_M4_R4_GRASP_PRESET.object_rot_wxyz" in source
    assert "GENERATED_RIGHT_T4_I4_M4_R4_OBJECT_SOURCE" in source
    assert "GENERATED_RIGHT_T4_I4_M4_R4_OBJECT_SCALE" in source


def test_generated_official_slot_joint_order_and_pregrasp_vector_share_one_order() -> None:
    r"""Action, observation, reset and pre-grasp vector must use the same official-slot order."""

    source = _read(GENERATED_CFG)
    expected_order_block = '''GENERATED_OFFICIAL_SLOT_JOINT_ORDER = (
    "index_j0",
    "thumb_j0",
    "middle_j0",
    "ring_j0",
    "index_j1",
    "thumb_j1",
    "middle_j1",
    "ring_j1",
    "index_j2",
    "thumb_j2",
    "middle_j2",
    "ring_j2",
    "index_j3",
    "thumb_j3",
    "middle_j3",
    "ring_j3",
)'''

    assert expected_order_block in source
    assert "joint_names=list(GENERATED_OFFICIAL_SLOT_JOINT_ORDER)" in source
    assert "pregrasp_joint_pos=GENERATED_RIGHT_T4_I4_M4_R4_PREGRASP_VECTOR" in source
    assert "_require_joint_values_by_order" in source
    assert "preserve_order=True" in source


def test_generated_structural_collision_filter_uses_pairwise_filtered_pairs_rule() -> None:
    r"""Generated probe must reuse the GM pairwise rule: palm-finger and same-finger filtered."""

    source = _read(GENERATED_CFG)

    assert "apply_generated_structural_collision_filter" in source
    assert "mode=\"prestartup\"" in source
    assert "palm_link_name" in source
    assert "finger_link_chains" in source
    assert "\"filter_palm_finger\": True" in source
    assert "\"filter_same_finger\": True" in source
    assert "PhysicsCollisionGroup" not in source
