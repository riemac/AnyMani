from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[6]
GENERATED_ADR_CFG = (
    REPO_ROOT
    / "source/anymani/anymani/tasks/inhand/config/generated_right_t4_i4_m4_r4/"
    / "generated_right_t4_i4_m4_r4_adr_env_cfg.py"
)
GENERATED_RAW_CFG = (
    REPO_ROOT
    / "source/anymani/anymani/tasks/inhand/config/generated_right_t4_i4_m4_r4/"
    / "generated_raw_action_env_cfg.py"
)
GENERATED_EMA_ABSOLUTE_CFG = (
    REPO_ROOT
    / "source/anymani/anymani/tasks/inhand/config/generated_right_t4_i4_m4_r4/"
    / "generated_ema_absolute_env_cfg.py"
)
GENERATED_REGISTER = REPO_ROOT / "source/anymani/anymani/tasks/inhand/config/generated_right_t4_i4_m4_r4/__init__.py"
REWARDS_FILE = REPO_ROOT / "source/anymani/anymani/tasks/inhand/mdp/rewards.py"
GM_ACTION_FILE = REPO_ROOT / "source/anymani/anymani/tasks/gm/mdp/actions/adr_joint_actions.py"


def _read(path: Path) -> str:
    r"""Read a source file for text-level contract assertions."""

    return path.read_text(encoding="utf-8")


def test_n031_registers_train_and_play_env_ids() -> None:
    r"""N031 must register explicit NoDtReward train/play ids next to N030 generated official-ADR."""

    register_source = _read(GENERATED_REGISTER)

    assert "AnyMani-LeapHand-ADR-Generated-right_t4_i4_m4_r4-NoDtReward-v0" in register_source
    assert "AnyMani-LeapHand-ADR-Generated-right_t4_i4_m4_r4-NoDtReward-Play-v0" in register_source
    assert "LeapHandADRGeneratedRightT4I4M4R4NoDtRewardEnvCfg" in register_source
    assert "LeapHandADRGeneratedRightT4I4M4R4NoDtRewardEnvCfg_PLAY" in register_source


def test_n031_cfg_only_switches_combined_reward_dt_alignment() -> None:
    r"""N031 should keep N030 semantics and only flip OfficialLeapReward's dt switch."""

    source = _read(GENERATED_ADR_CFG)

    assert "GeneratedRightT4I4M4R4OfficialADRNoDtRewardsCfg" in source
    assert '"divide_by_step_dt": False' in source
    assert "LeapHandADRGeneratedRightT4I4M4R4NoDtRewardEnvCfg" in source
    assert "LeapHandADRGeneratedRightT4I4M4R4NoDtRewardEnvCfg_PLAY" in source


def test_n040_registers_train_and_play_env_ids() -> None:
    r"""N040 must register explicit RawDelta train/play ids for direct comparison against N030/N031."""

    register_source = _read(GENERATED_REGISTER)

    assert "AnyMani-LeapHand-ADR-Generated-right_t4_i4_m4_r4-RawDelta-v0" in register_source
    assert "AnyMani-LeapHand-ADR-Generated-right_t4_i4_m4_r4-RawDelta-Play-v0" in register_source
    assert "LeapHandADRGeneratedRightT4I4M4R4RawDeltaEnvCfg" in register_source
    assert "LeapHandADRGeneratedRightT4I4M4R4RawDeltaEnvCfg_PLAY" in register_source


def test_n040_cfg_replaces_only_action_and_actor_obs_semantics() -> None:
    r"""N040 should keep official reward/command/termination/ADR but switch to raw actor obs and raw-relative action."""

    source = _read(GENERATED_RAW_CFG)

    assert "raw_policy_frame" in source
    assert "ADRRelativeJointPositionActionCfg" in source
    assert 'reference="current"' in source
    assert "RAW_DELTA_ACTION_SCALE_RAD = 1.0 / 24.0" in source
    assert '"joint_scale_rad": 3.141592653589793' in source
    assert "LeapHandOfficialADRRewardsCfg" in source
    assert "LeapHandOfficialADRCommandsCfg" in source
    assert "LeapHandOfficialADRTerminationsCfg" in source
    assert "LeapHandOfficialADRCurriculumCfg" in source


def test_rewards_file_exposes_combined_and_split_official_terms() -> None:
    r"""The new rewards.py must expose one combined reward and fully independent split terms."""

    source = _read(REWARDS_FILE)

    assert "class OfficialLeapReward(ManagerTermBase)" in source
    assert "official_goal_distance" in source
    assert "official_orientation" in source
    assert "official_action_l2" in source
    assert "official_pregrasp_l2" in source
    assert "official_success_bonus" in source
    assert "official_fall_penalty" in source
    assert "official_z_spin_bonus" in source
    assert "divide_by_step_dt" in source
    assert "OfficialRewardState" not in source


def test_gm_declarative_adr_actions_expose_shared_runtime_contract() -> None:
    r"""N4x actor obs and official reward depend on a shared action runtime contract."""

    source = _read(GM_ACTION_FILE)

    assert "class ADRJointAction(JointAction)" in source
    assert "class ADRRelativeJointPositionAction(ADRJointAction)" in source
    assert "class ADREMAJointPositionToLimitsAction(ADRJointAction)" in source
    assert "class ADRRelativeJointPositionActionCfg" in source
    assert "class ADREMAJointPositionToLimitsActionCfg" in source
    assert "def current_targets" in source
    assert "def executed_actions" in source
    assert "pregrasp_joint_pos" in source
    assert "compute_leap_adr_latency_steps" in source
    assert "compute_relative_joint_command" in source
    assert "compute_ema_joint_command" in source


def test_n041_registers_ema_absolute_train_and_play_env_ids() -> None:
    r"""N041 must register explicit EMAAbsolute train/play ids next to N030/N040."""

    register_source = _read(GENERATED_REGISTER)

    assert "AnyMani-LeapHand-ADR-Generated-right_t4_i4_m4_r4-EMAAbsolute-v0" in register_source
    assert "AnyMani-LeapHand-ADR-Generated-right_t4_i4_m4_r4-EMAAbsolute-Play-v0" in register_source
    assert "LeapHandADRGeneratedRightT4I4M4R4EMAAbsoluteEnvCfg" in register_source
    assert "LeapHandADRGeneratedRightT4I4M4R4EMAAbsoluteEnvCfg_PLAY" in register_source


def test_n041_cfg_replaces_only_action_law_and_keeps_official_obs() -> None:
    r"""N041 should compare EMA absolute against N030 without introducing raw obs changes."""

    source = _read(GENERATED_EMA_ABSOLUTE_CFG)

    assert "ADREMAJointPositionToLimitsActionCfg" in source
    assert 'reference="target"' in source
    assert "EMA_ABSOLUTE_ALPHA = 1.0 / 24.0" in source
    assert "GeneratedRightT4I4M4R4OfficialADRObservationsCfg" in source
    assert "raw_policy_frame" not in source
    assert "joint_scale_rad" not in source
    assert "LeapHandOfficialADRRewardsCfg" in source
    assert "LeapHandOfficialADRCommandsCfg" in source
    assert "LeapHandOfficialADRTerminationsCfg" in source
    assert "LeapHandOfficialADRCurriculumCfg" in source
