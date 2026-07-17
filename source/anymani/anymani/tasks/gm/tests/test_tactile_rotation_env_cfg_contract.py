r"""Tactile rotation executable cfg 与 tasks/distill registry 的纯 declaration contracts。"""

from __future__ import annotations

import ast
from pathlib import Path

import anymani.distill.rl  # noqa: F401  # 注册 distill-owned GRU/TCN aliases
import anymani.tasks.gm  # noqa: F401  # 注册 tasks-owned observation variants
import gymnasium as gym

CFG_PATH = Path(__file__).resolve().parents[1] / "config" / "single_asset" / "tactile_rotation_env_cfg.py"


def _source() -> str:
    r"""读取 executable tactile cfg source，不 import Isaac/Kit runtime。"""

    return CFG_PATH.read_text(encoding="utf-8")


def _class_node(name: str) -> ast.ClassDef:
    r"""按类名解析 config declaration。"""

    for node in ast.parse(_source()).body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"Class {name!r} not found in {CFG_PATH}")


def test_tasks_registry_uses_observation_semantics_not_network_names() -> None:
    r"""tasks IDs 只区分 CurrentObs/History30Obs，并各有 Play variant。"""

    expected = {
        "AnyMani-GM-SingleAsset-TactileRotation-CurrentObs-v0": "GmTactileRotationCurrentEnvCfg",
        "AnyMani-GM-SingleAsset-TactileRotation-CurrentObs-Play-v0": "GmTactileRotationCurrentEnvCfg_PLAY",
        "AnyMani-GM-SingleAsset-TactileRotation-History30Obs-v0": "GmTactileRotationHistory30EnvCfg",
        "AnyMani-GM-SingleAsset-TactileRotation-History30Obs-Play-v0": "GmTactileRotationHistory30EnvCfg_PLAY",
    }
    for task_id, cfg_name in expected.items():
        entry_point = gym.spec(task_id).kwargs["env_cfg_entry_point"]
        assert entry_point.endswith(f"tactile_rotation_env_cfg:{cfg_name}")
        assert "GRU" not in task_id and "TCN" not in task_id and "N052" not in task_id


def test_distill_aliases_bind_current_to_gru_and_history_to_tcn() -> None:
    r"""Network architecture 名只属于 distill alias，并分别选择正确 env/YAML。"""

    gru = gym.spec("AnyMani-GM-SingleAsset-TactileRotation-GRU-v0").kwargs
    tcn = gym.spec("AnyMani-GM-SingleAsset-TactileRotation-TCN-v0").kwargs

    assert gru["env_cfg_entry_point"].endswith("GmTactileRotationCurrentEnvCfg")
    assert gru["rl_games_cfg_entry_point"].endswith(":gm_tactile_rotation_gru_ppo.yaml")
    assert tcn["env_cfg_entry_point"].endswith("GmTactileRotationHistory30EnvCfg")
    assert tcn["rl_games_cfg_entry_point"].endswith(":gm_tactile_rotation_tcn_ppo.yaml")


def test_history_variant_only_overrides_policy_observations() -> None:
    r"""History30 env 必须继承同一 base assembly，只覆盖 observations field。"""

    history_class = _class_node("GmTactileRotationHistory30EnvCfg")
    assert [base.id for base in history_class.bases if isinstance(base, ast.Name)] == ["GmTactileRotationCurrentEnvCfg"]
    assigned_names = {
        node.target.id for node in history_class.body if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert assigned_names == {"observations"}

    source = _source()
    assert "frame = _policy_frame_term(history_length=30, flatten_history_dim=False)" in source
    assert "func=gm_mdp.tactile_rotation_policy_frame" in source
    assert "func=gm_mdp.tactile_rotation_critic_state" in source


def test_cfg_locks_physics_reward_contact_and_adr_contracts() -> None:
    r"""高风险物理/MDP 数值必须在 executable cfg 中显式出现。"""

    source = _source()
    assert "self.decimation = 6" in source
    assert "self.sim.dt = 1.0 / 120.0" in source
    assert "self.episode_length_s = 120.0" in source
    assert "num_envs=4096" in source
    assert "scale=(1.2, 1.2, 1.2)" in source
    assert '"scale_range": (1.1, 1.25)' in source
    assert "func=gm_mdp.randomize_object_com_from_default_and_record" in source
    assert "func=gm_mdp.PolicyStepADRTargetJointPositionActionCfg" not in source  # cfg constructor，不是 callable
    assert "gm_mdp.PolicyStepADRTargetJointPositionActionCfg(" in source
    assert "scale=1.0 / 24.0" in source
    assert '"min_contacts": 2' in source
    assert "weight=0.1" in source
    assert "weight=-0.2" in source
    assert '"fall_dist": 0.07' in source
    assert '"max_angle_deg": 45.0' in source
    assert '"threshold_turns_per_s": 0.08' in source
    assert '"min_reset_checks_for_increase": 960' in source
    assert "anymani.tasks.inhand" not in source  # 用户选择 GM-owned action/ADR，无父包循环依赖


def test_bad_contact_and_material_roles_exclude_neutral_palm() -> None:
    r"""Bad contact 使用 19 finger non-tips；palm 只以独立 support role 进入 critic/material state。"""

    source = _source()
    assert "TACTILE_FINGER_NON_TIP_SENSOR_NAMES = GM_SINGLE_ASSET_CONTACT_LAYOUT.finger_non_tip_sensor_names" in source
    assert "TACTILE_PALM_SENSOR_NAME = GM_SINGLE_ASSET_CONTACT_LAYOUT.palm_sensor_name" in source
    assert "func=gm_mdp.tactile_bad_finger_non_tip_contact" in source
    assert '"adr_state_field": "hand_contact_material"' in source
    assert "list(GM_SINGLE_ASSET_CONTACT_LAYOUT.all_link_names)" in source
