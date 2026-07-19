r"""Generated single-asset in-hand variants 的静态 contract tests。

测试文件按稳定的环境/算法语义组织，而不按 N031/N040 等阶段性实验编号分类。实验编号只允许作为
docstring 中的注释性 provenance；它不进入 AnyMani 的文件分类、公共符号、import 路径或运行时
contract。函数名描述真正被证伪的 reward、action、observation 或注册语义，使下游实验记录可以引用
AnyMani，而 AnyMani 本身不反向依赖实验记录目录。
"""

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
GENERATED_RAW_OBSERVATION_CFG = (
    REPO_ROOT
    / "source/anymani/anymani/tasks/inhand/config/generated_right_t4_i4_m4_r4/"
    / "generated_raw_observation_env_cfg.py"
)
GENERATED_POLICY_STEP_TARGET_CFG = (
    REPO_ROOT
    / "source/anymani/anymani/tasks/inhand/config/generated_right_t4_i4_m4_r4/"
    / "generated_policy_step_target_env_cfg.py"
)
GENERATED_REGISTER = REPO_ROOT / "source/anymani/anymani/tasks/inhand/config/generated_right_t4_i4_m4_r4/__init__.py"
REWARDS_FILE = REPO_ROOT / "source/anymani/anymani/tasks/inhand/mdp/rewards.py"
LEAPHAND_OFFICIAL_ADR_CFG = REPO_ROOT / "source/anymani/anymani/tasks/inhand/config/leaphand/leaphand_adr_env_cfg.py"
LEGACY_REWARD_FILES = (
    REPO_ROOT / "source/anymani/anymani/tasks/inhand/mdp/rewards_action.py",
    REPO_ROOT / "source/anymani/anymani/tasks/inhand/mdp/rewards_task.py",
)
GM_ACTION_FILE = REPO_ROOT / "source/anymani/anymani/tasks/gm/mdp/actions/adr_joint_actions.py"
INHAND_ACTION_FILE = REPO_ROOT / "source/anymani/anymani/tasks/inhand/mdp/actions/adr_relative_action.py"
OBSERVATIONS_FILE = REPO_ROOT / "source/anymani/anymani/tasks/inhand/mdp/observations.py"


def _read(path: Path) -> str:
    r"""Read a source file for text-level contract assertions."""

    return path.read_text(encoding="utf-8")


def test_no_dt_reward_variant_registers_train_and_play_env_ids() -> None:
    r"""NoDtReward（N031）必须在 generated baseline（N030）旁注册独立 train/play ids。"""

    register_source = _read(GENERATED_REGISTER)

    assert "AnyMani-LeapHand-ADR-Generated-right_t4_i4_m4_r4-NoDtReward-v0" in register_source
    assert "AnyMani-LeapHand-ADR-Generated-right_t4_i4_m4_r4-NoDtReward-Play-v0" in register_source
    assert "LeapHandADRGeneratedRightT4I4M4R4NoDtRewardEnvCfg" in register_source
    assert "LeapHandADRGeneratedRightT4I4M4R4NoDtRewardEnvCfg_PLAY" in register_source


def test_no_dt_reward_variant_only_switches_combined_reward_dt_alignment() -> None:
    r"""NoDtReward（N031）应保持 N030 语义，只翻转 `OfficialLeapReward` 的 dt 开关。"""

    source = _read(GENERATED_ADR_CFG)

    assert "GeneratedRightT4I4M4R4OfficialADRNoDtRewardsCfg" in source
    assert '"divide_by_step_dt": False' in source
    assert "LeapHandADRGeneratedRightT4I4M4R4NoDtRewardEnvCfg" in source
    assert "LeapHandADRGeneratedRightT4I4M4R4NoDtRewardEnvCfg_PLAY" in source


def test_raw_delta_variant_registers_train_and_play_env_ids() -> None:
    r"""RawDelta（N040）必须注册独立 train/play ids，供 N030/N031 直接对照。"""

    register_source = _read(GENERATED_REGISTER)

    assert "AnyMani-LeapHand-ADR-Generated-right_t4_i4_m4_r4-RawDelta-v0" in register_source
    assert "AnyMani-LeapHand-ADR-Generated-right_t4_i4_m4_r4-RawDelta-Play-v0" in register_source
    assert "LeapHandADRGeneratedRightT4I4M4R4RawDeltaEnvCfg" in register_source
    assert "LeapHandADRGeneratedRightT4I4M4R4RawDeltaEnvCfg_PLAY" in register_source


def test_raw_delta_variant_replaces_only_action_and_actor_obs_semantics() -> None:
    r"""RawDelta（N040）只切换 raw actor obs 与 current-relative action，其余 official MDP 保持不变。"""

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


def test_official_reward_cfg_pins_direct_step_formula_and_dt_alignment() -> None:
    r"""Official ADR 主线必须显式固定七项系数，并抵消 RewardManager 的 policy-step dt。"""

    source = _read(LEAPHAND_OFFICIAL_ADR_CFG)

    assert "func=official_rewards.OfficialLeapReward" in source
    assert '"dist_reward_scale": -10.0' in source
    assert '"rot_reward_scale": 1.0' in source
    assert '"rot_eps": 0.1' in source
    assert '"action_penalty_scale": -0.0002' in source
    assert '"pose_diff_penalty_scale": -0.3' in source
    assert '"success_tolerance": 0.2' in source
    assert '"position_success_threshold": 0.025' in source
    assert '"reach_goal_bonus": 250.0' in source
    assert '"fall_dist": 0.07' in source
    assert '"fall_penalty": -10.0' in source
    assert '"divide_by_step_dt": True' in source


def test_legacy_reward_modules_remain_removed() -> None:
    r"""旧 action/task reward 聚合模块不得重新进入主线，避免两套 official 公式并存。"""

    assert all(not path.exists() for path in LEGACY_REWARD_FILES)


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


def test_ema_absolute_variant_registers_train_and_play_env_ids() -> None:
    r"""EMAAbsolute（N041）必须在 N030/N040 旁注册独立 train/play ids。"""

    register_source = _read(GENERATED_REGISTER)

    assert "AnyMani-LeapHand-ADR-Generated-right_t4_i4_m4_r4-EMAAbsolute-v0" in register_source
    assert "AnyMani-LeapHand-ADR-Generated-right_t4_i4_m4_r4-EMAAbsolute-Play-v0" in register_source
    assert "LeapHandADRGeneratedRightT4I4M4R4EMAAbsoluteEnvCfg" in register_source
    assert "LeapHandADRGeneratedRightT4I4M4R4EMAAbsoluteEnvCfg_PLAY" in register_source


def test_ema_absolute_variant_replaces_only_action_law_and_keeps_official_obs() -> None:
    r"""EMAAbsolute（N041）只替换 action law，不把 raw observation 混入 N030 对照。"""

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


def test_raw_observation_variants_register_train_and_play_env_ids() -> None:
    r"""RawRadObs（N050）与 UnitRawObs（N051）必须注册独立 train/play ids。"""

    register_source = _read(GENERATED_REGISTER)

    assert "AnyMani-LeapHand-ADR-Generated-right_t4_i4_m4_r4-RawRadObs-v0" in register_source
    assert "AnyMani-LeapHand-ADR-Generated-right_t4_i4_m4_r4-RawRadObs-Play-v0" in register_source
    assert "LeapHandADRGeneratedRightT4I4M4R4RawRadObsEnvCfg" in register_source
    assert "LeapHandADRGeneratedRightT4I4M4R4RawRadObsEnvCfg_PLAY" in register_source

    assert "AnyMani-LeapHand-ADR-Generated-right_t4_i4_m4_r4-UnitRawObs-v0" in register_source
    assert "AnyMani-LeapHand-ADR-Generated-right_t4_i4_m4_r4-UnitRawObs-Play-v0" in register_source
    assert "LeapHandADRGeneratedRightT4I4M4R4UnitRawObsEnvCfg" in register_source
    assert "LeapHandADRGeneratedRightT4I4M4R4UnitRawObsEnvCfg_PLAY" in register_source


def test_raw_observation_variants_only_replace_actor_obs_and_keep_baseline_mdp() -> None:
    r"""N050/N051 只隔离 observation scaling，不改变 N030 action/reward/ADR 语义。"""

    source = _read(GENERATED_RAW_OBSERVATION_CFG)

    assert "official_policy_frame_raw_rad" in source
    assert "raw_policy_frame" in source
    assert "UNIT_RAW_OBS_JOINT_SCALE_RAD = 3.141592653589793" in source
    assert "GeneratedRightT4I4M4R4OfficialADRActionsCfg" in source
    assert "GeneratedRightT4I4M4R4OfficialADREventCfg" in source
    assert "GeneratedRightT4I4M4R4OfficialADRSceneCfg" in source
    assert "LeapHandOfficialADRRewardsCfg" in source
    assert "LeapHandOfficialADRCommandsCfg" in source
    assert "LeapHandOfficialADRTerminationsCfg" in source
    assert "LeapHandOfficialADRCurriculumCfg" in source
    assert "history_length=3" in source
    assert "flatten_history_dim=True" in source
    assert "ADRRelativeJointPositionActionCfg" not in source
    assert "ADREMAJointPositionToLimitsActionCfg" not in source


def test_raw_rad_observation_keeps_target_buffer_in_rad_units() -> None:
    r"""RawRadObs（N050）必须保留 raw-rad target buffer，不能静默换成 `last_action`。"""

    source = _read(OBSERVATIONS_FILE)

    assert "def official_policy_frame_raw_rad" in source
    assert "torch.cat((joint_pos, action_term.current_targets), dim=-1).clone()" in source
    assert "official_policy_frame_raw_rad expects action term" in source
    assert (
        "last_action"
        not in source[source.index("def official_policy_frame_raw_rad") : source.index("def raw_policy_frame")]
    )


def test_policy_step_target_variant_registers_semantic_train_and_play_ids() -> None:
    r"""PolicyStepTarget 必须使用语义化 public ids，Research 编号不能进入代码接口。"""

    register_source = _read(GENERATED_REGISTER)

    assert "AnyMani-LeapHand-ADR-Generated-right_t4_i4_m4_r4-PolicyStepTarget-v0" in register_source
    assert "AnyMani-LeapHand-ADR-Generated-right_t4_i4_m4_r4-PolicyStepTarget-Play-v0" in register_source
    assert "LeapHandADRGeneratedRightT4I4M4R4PolicyStepTargetEnvCfg" in register_source
    assert "LeapHandADRGeneratedRightT4I4M4R4PolicyStepTargetEnvCfg_PLAY" in register_source
    assert "N052" not in register_source


def test_policy_step_target_variant_inherits_unit_raw_obs_and_only_overrides_actions() -> None:
    r"""PolicyStepTarget 必须冻结 UnitRawObs 父 MDP，只替换 action group。"""

    source = _read(GENERATED_POLICY_STEP_TARGET_CFG)

    assert "LeapHandADRGeneratedRightT4I4M4R4UnitRawObsEnvCfg" in source
    assert "PolicyStepADRTargetJointPositionActionCfg" in source
    assert "POLICY_STEP_TARGET_SCALE_RAD = 1.0 / 24.0" in source
    assert "actions: GeneratedRightT4I4M4R4PolicyStepTargetActionsCfg" in source
    assert "LeapHandOfficialADRRewardsCfg" not in source
    assert "LeapHandOfficialADRCommandsCfg" not in source
    assert "LeapHandOfficialADRTerminationsCfg" not in source
    assert "LeapHandOfficialADRCurriculumCfg" not in source
    assert "N052" not in source


def test_policy_step_action_updates_in_process_and_apply_is_idempotent_hold() -> None:
    r"""Target recurrence 必须位于 `process_actions()`；`apply_actions()` 只能下发 target。"""

    source = _read(INHAND_ACTION_FILE)
    class_start = source.index("class PolicyStepADRTargetJointPositionAction(")
    cfg_start = source.index("class PolicyStepADRTargetJointPositionActionCfg(")
    class_source = source[class_start:cfg_start]
    process_start = class_source.index("def process_actions")
    apply_start = class_source.index("def apply_actions")
    process_source = class_source[process_start:apply_start]
    apply_source = class_source[apply_start:]

    assert "compute_official_target_update" in process_source
    assert "self._current_targets[:] = next_targets" in process_source
    assert "self._previous_targets[:] = next_targets" in process_source
    assert "set_joint_position_target" in apply_source
    assert "compute_official_target_update" not in apply_source
    assert "_previous_targets" not in apply_source
    assert "_current_targets[:]" not in apply_source
