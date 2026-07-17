r"""Tactile episode diagnostics 的 policy-step accumulation 与 partial-reset contracts。"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_diagnostics_module():
    r"""以 ADR/contact stubs 加载 diagnostics，不启动 Isaac Sim。"""

    module_path = Path(__file__).resolve().parents[1] / "mdp" / "tactile_diagnostics_state.py"
    package_names = ("anymani", "anymani.tasks", "anymani.tasks.gm", "anymani.tasks.gm.mdp")
    replacements: dict[str, types.ModuleType] = {}
    for package_name in package_names:
        package = types.ModuleType(package_name)
        package.__path__ = []  # type: ignore[attr-defined]
        replacements[package_name] = package

    adr_stub = types.ModuleType("anymani.tasks.gm.mdp.adr_state")
    adr_stub.ADR_STATE_SLICES = {
        "scale": slice(0, 1),
        "mass": slice(1, 2),
        "com": slice(2, 5),
        "object_material": slice(5, 8),
        "hand_contact_material": slice(8, 11),
        "stiffness": slice(11, 27),
        "damping": slice(27, 43),
        "action_noise": slice(43, 44),
        "latency_steps": slice(44, 45),
        "wrench_gate": slice(45, 46),
        "max_acceleration": slice(46, 47),
        "fraction": slice(47, 48),
    }
    adr_stub.get_gm_adr_state = lambda env: env.adr_state
    contact_stub = types.ModuleType("anymani.tasks.gm.mdp.tactile_contact_state")
    contact_stub.get_tactile_contact_state = lambda env, *_args, **_kwargs: env.contact
    replacements.update(
        {
            "anymani.tasks.gm.mdp.adr_state": adr_stub,
            "anymani.tasks.gm.mdp.tactile_contact_state": contact_stub,
        }
    )
    previous = {name: sys.modules.get(name) for name in replacements}
    try:
        sys.modules.update(replacements)
        spec = importlib.util.spec_from_file_location("anymani.tasks.gm.mdp.tactile_diagnostics_contract", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load diagnostics module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


def _fake_env() -> SimpleNamespace:
    r"""构造 2-env action/contact/ADR/termination runtime surface。"""

    action_term = SimpleNamespace(
        raw_actions=torch.zeros(2, 16),
        executed_actions=torch.zeros(2, 16),
        current_targets=torch.zeros(2, 16),
    )
    adr_values = torch.zeros(2, 48)
    adr_values[:, 0] = 1.2  # actual scale
    adr_values[:, 1] = torch.tensor([0.2, 0.3])  # kg
    adr_values[:, 5:8] = torch.tensor([0.8, 0.7, 0.1])  # object material
    adr_values[:, 8:11] = torch.tensor([1.0, 0.9, 0.05])  # hand material
    adr_values[:, 11:27] = 3.0  # Kp
    adr_values[:, 27:43] = 0.1  # Kd
    contact = SimpleNamespace(
        tip_bits=torch.tensor([[True, True, False, False], [True, False, False, False]]),
        finger_non_tip_bits=torch.zeros(2, 19, dtype=torch.bool),
        palm_bits=torch.tensor([[True], [False]]),
        tip_force_ema=torch.tensor([[1.0, 2.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        palm_force_ema=torch.tensor([[3.0], [0.0]]),
    )
    term_values = {
        "object_out_of_anchor": torch.tensor([True, False]),
        "goal_axis_misaligned": torch.tensor([False, False]),
        "time_out": torch.tensor([False, True]),
    }
    return SimpleNamespace(
        num_envs=2,
        device="cpu",
        step_dt=0.05,
        common_step_counter=0,
        episode_length_buf=torch.tensor([10, 20]),
        leap_adr_episode_lengths=torch.tensor([400, 800]),
        action_manager=SimpleNamespace(get_term=lambda _name: action_term),
        termination_manager=SimpleNamespace(active_terms=list(term_values), get_term=lambda name: term_values[name]),
        adr_state=SimpleNamespace(values=adr_values),
        contact=contact,
    )


def test_diagnostics_accumulate_rates_and_reset_only_selected_envs() -> None:
    r"""Rate/occupancy 应按 episode policy-step 平均，partial reset 不得清未结束 env。"""

    module = _load_diagnostics_module()
    env = _fake_env()
    diagnostics = module.GmTactileEpisodeDiagnostics(
        env,
        fingertip_sensor_names=("t0", "t1", "t2", "t3"),
        finger_non_tip_sensor_names=tuple(f"n{i}" for i in range(19)),
        palm_sensor_name="palm",
    )
    action_term = env.action_manager.get_term("hand_joint_pos")
    diagnostics.reset(env, torch.tensor([0, 1]))

    # 第一步 policy action 从 0 跳到 0.1，故 RMS rate 为 $0.1/0.05=2\ s^{-1}$。
    action_term.raw_actions[:] = 0.1
    action_term.executed_actions[:] = 0.05
    action_term.current_targets[:] = 0.01
    env.common_step_counter = 1
    diagnostics.ensure_updated(
        env,
        axis_w=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        axis_speed=torch.tensor([0.5, -0.25]),
        object_ang_vel_w=torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, -1.0]]),
        joint_position=torch.zeros(2, 16),
        position_error=torch.tensor([0.01, 0.02]),
        orientation_keypoint_error=torch.tensor([0.03, 0.04]),
    )

    assert diagnostics.metrics["action/policy_delta_rms_per_s"][0].item() == 2.0
    assert diagnostics.metrics["contact/tip_active_count_mean"][0].item() == 2.0
    assert diagnostics.metrics["contact/palm_occupancy_fraction"][0].item() == 1.0
    torch.testing.assert_close(
        diagnostics.metrics["rotation/off_axis_ang_vel_rms_rad_s"][0], torch.tensor(5.0).sqrt()
    )

    # 同一 stamp 重入必须幂等，不能把 episode count 加倍。
    diagnostics.ensure_updated(
        env,
        axis_w=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        axis_speed=torch.tensor([0.5, -0.25]),
        object_ang_vel_w=torch.zeros(2, 3),
        joint_position=torch.zeros(2, 16),
        position_error=torch.zeros(2),
        orientation_keypoint_error=torch.zeros(2),
    )
    assert diagnostics.step_count.tolist() == [1.0, 1.0]

    diagnostics.capture_terminal(env, torch.tensor([0]))
    assert diagnostics.metrics["task/episode_duration_s"][0].item() == 0.5
    assert diagnostics.metrics["termination/object_out_of_anchor_fraction"][0].item() == 1.0
    torch.testing.assert_close(diagnostics.metrics["adr/actual_object_mass_kg"][0], torch.tensor(0.2))

    env0_metric_before_reset = diagnostics.metrics["rotation/axis_speed_abs_mean_rad_s"][0].item()
    env1_metric_before_reset = diagnostics.metrics["rotation/axis_speed_abs_mean_rad_s"][1].item()
    diagnostics.reset(env, torch.tensor([0]))
    assert env0_metric_before_reset > 0.0 and diagnostics.metrics["rotation/axis_speed_abs_mean_rad_s"][0] == 0.0
    assert diagnostics.metrics["rotation/axis_speed_abs_mean_rad_s"][1] == env1_metric_before_reset
