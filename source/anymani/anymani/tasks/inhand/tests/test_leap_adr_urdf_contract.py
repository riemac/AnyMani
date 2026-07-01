from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[6]
LEAPHAND_USD_CFG = REPO_ROOT / "source/anymani/anymani/tasks/inhand/config/leaphand/leaphand_adr_env_cfg.py"
LEAPHAND_USD_REGISTER = REPO_ROOT / "source/anymani/anymani/tasks/inhand/config/leaphand/__init__.py"
LEAPHAND_URDF_CFG = REPO_ROOT / "source/anymani/anymani/tasks/inhand/config/leaphand_urdf/leaphand_urdf_adr_env_cfg.py"
LEAPHAND_URDF_REGISTER = REPO_ROOT / "source/anymani/anymani/tasks/inhand/config/leaphand_urdf/__init__.py"
LEAPHAND_URDF_ROBOT = REPO_ROOT / "source/anymani/anymani/robots/leap_urdf.py"


def _read(path: Path) -> str:
    r"""Read a source file for a pure text-level contract assertion."""

    return path.read_text(encoding="utf-8")


def test_urdf_env_registers_as_separate_official_adr_backend() -> None:
    r"""URDF backend must be a new Gym id, not a mutation of the accepted N010 id."""

    register_source = _read(LEAPHAND_URDF_REGISTER)
    usd_register_source = _read(LEAPHAND_USD_REGISTER)

    assert "AnyMani-LeapHand-Tactile-ADR-URDF-v0" in register_source
    assert "AnyMani-LeapHand-Tactile-ADR-URDF-Play-v0" in register_source
    assert "leaphand_urdf_adr_env_cfg:LeapHandTactileADRURDFEnvCfg" in register_source
    assert "leaphand_urdf_adr_env_cfg:LeapHandTactileADRURDFEnvCfg_PLAY" in register_source
    assert "rl_games_ppo_cfg_official_adr.yaml" in register_source

    assert "AnyMani-LeapHand-Tactile-ADR-v0" in usd_register_source
    assert "leaphand_adr_env_cfg:LeapHandTactileADREnvCfg" in usd_register_source


def test_urdf_env_reuses_n010_mdp_and_replaces_only_robot_backend() -> None:
    r"""URDF variant should inherit N010 MDP terms and override only scene.robot."""

    cfg_source = _read(LEAPHAND_URDF_CFG)

    assert "from anymani.robots.leap_urdf import LEAP_HAND_URDF_CFG" in cfg_source
    assert "LeapHandOfficialADRSceneCfg" in cfg_source
    assert "LeapHandTactileADREnvCfg" in cfg_source
    assert "class LeapHandOfficialADRURDFSceneCfg(LeapHandOfficialADRSceneCfg)" in cfg_source
    assert "class LeapHandTactileADRURDFEnvCfg(LeapHandTactileADREnvCfg)" in cfg_source
    assert "robot: ArticulationCfg = LEAP_HAND_URDF_CFG.replace" in cfg_source

    assert "pos=(0.0, 0.0, 0.5)" in cfg_source
    assert "rot=(0.5, 0.5, -0.5, 0.5)" in cfg_source
    assert "joint_pos=OFFICIAL_PREGRASP_BY_NAME" in cfg_source
    assert "joint_vel={\"a_.*\": 0.0}" in cfg_source


def test_official_object_contact_basin_remains_env_frame_seed() -> None:
    r"""Both USD and URDF official-aligned envs must use the official object pose in `{e}`."""

    usd_cfg_source = _read(LEAPHAND_USD_CFG)
    urdf_cfg_source = _read(LEAPHAND_URDF_CFG)

    assert "init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.1, 0.56), rot=(1.0, 0.0, 0.0, 0.0))" in usd_cfg_source
    assert "object 初始位姿与 N010 official-aligned USD baseline 保持一致" in urdf_cfg_source
    assert "$p_o^e=(0,-0.1,0.56)$" in urdf_cfg_source


def test_urdf_robot_keeps_n010_actuator_control_envelope() -> None:
    r"""The raw URDF backend experiment keeps N010 actuator control semantics."""

    robot_source = _read(LEAPHAND_URDF_ROBOT)

    assert "effort_limit_sim=0.5" in robot_source
    assert "velocity_limit_sim=100.0" in robot_source
    assert "stiffness=3.0" in robot_source
    assert "damping=0.1" in robot_source
    assert "armature=0.001" in robot_source


def test_urdf_play_variant_matches_n010_visualization_contract() -> None:
    r"""URDF play env should keep the N010 debug settings while changing only backend."""

    cfg_source = _read(LEAPHAND_URDF_CFG)

    assert "class LeapHandTactileADRURDFEnvCfg_PLAY(LeapHandTactileADRURDFEnvCfg)" in cfg_source
    assert "self.scene.num_envs = 50" in cfg_source
    assert "self.commands.goal_pose.debug_vis = True" in cfg_source
