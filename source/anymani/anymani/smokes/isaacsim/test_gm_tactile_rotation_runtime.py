r"""Isaac Sim runtime smoke for GM palm-supported tactile rotation。

默认验证 CurrentObs；设置
`ANYMANI_GM_TACTILE_SMOKE_TASK=AnyMani-GM-SingleAsset-TactileRotation-History30Obs-v0`
可在独立 Kit 进程验证 TCN history route。

```bash
timeout --kill-after=20s 300s /home/hac/isaac/IsaacLab/isaaclab.sh -p -m pytest \
  source/anymani/anymani/smokes/isaacsim/test_gm_tactile_rotation_runtime.py -q -s
```
"""

from __future__ import annotations

# ruff: noqa: I001
# AppLauncher 必须先于 gym/任务/pxr runtime imports。

import os
import traceback
from collections.abc import Mapping
from typing import Any

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
import pytest
import torch
import isaaclab.utils.math as math_utils
from isaaclab_tasks.utils import parse_env_cfg

import anymani.tasks.gm  # noqa: F401
from anymani.tasks.gm.config.single_asset.tactile_rotation_env_cfg import (
    TACTILE_FINGER_NON_TIP_SENSOR_NAMES,
    TACTILE_JOINT_CFG,
    TACTILE_PALM_SENSOR_NAME,
    TACTILE_TIP_SENSOR_NAMES,
)
from anymani.tasks.gm import mdp as gm_mdp

CURRENT_TASK_ID = "AnyMani-GM-SingleAsset-TactileRotation-CurrentObs-v0"
HISTORY_TASK_ID = "AnyMani-GM-SingleAsset-TactileRotation-History30Obs-v0"
TASK_ID = os.environ.get("ANYMANI_GM_TACTILE_SMOKE_TASK", CURRENT_TASK_ID)
SMOKE_NUM_ENVS = 2


def teardown_module() -> None:
    r"""关闭 Kit app，避免 smoke 后遗留 Isaac Sim 进程。"""

    simulation_app.close()


@pytest.mark.isaacsim
def test_gm_tactile_rotation_runtime_contract() -> None:
    r"""验证新 MDP 在真实 PhysX/ContactSensor/Manager lifecycle 下 reset 与 step。"""

    assert TASK_ID in (CURRENT_TASK_ID, HISTORY_TASK_ID)
    env_cfg = parse_env_cfg(TASK_ID, device="cuda:0", num_envs=SMOKE_NUM_ENVS)
    env: Any = None  # Gym 动态注册 task 的具体 ManagerBasedRLEnv 类型无法由静态泛型推断
    try:
        env = gym.make(TASK_ID, cfg=env_cfg)
        runtime_env = env.unwrapped
        runtime_env.sim._app_control_on_stop_handle = None
        obs, _ = env.reset()
        _assert_observation_contract(obs, expect_reset_prefix=True)
        _assert_adr_runtime(runtime_env)
        _assert_body_yaw_reset(runtime_env)

        action_term = runtime_env.action_manager.get_term("hand_joint_pos")
        runtime_env.leap_adr_action_noise = 1.0  # global curriculum 发布值故意与 ongoing episode actual 不同
        runtime_env._gm_adr_state.values[:, 43] = 0.0  # ongoing episode $\sigma_a=0$；同时验证 action 读取 per-env state
        action_term._latency_steps.zero_()
        action_term._action_history.zero_()
        targets_before = action_term.current_targets.clone()
        joint_mid = 0.5 * (action_term._joint_lower + action_term._joint_upper)
        actions = torch.where(joint_mid >= targets_before, 0.25, -0.25)
        expected_targets = torch.clamp(
            targets_before + actions / 24.0,
            min=action_term._joint_lower,
            max=action_term._joint_upper,
        )

        obs, reward, terminated, truncated, _ = env.step(actions)
        _assert_observation_contract(obs, expect_reset_prefix=False)
        assert reward.shape == terminated.shape == truncated.shape == (SMOKE_NUM_ENVS,)
        assert torch.isfinite(reward).all()
        assert not torch.any(terminated | truncated), "first controlled step should remain inside tactile task basin"
        torch.testing.assert_close(action_term.executed_actions, actions, rtol=0.0, atol=1.0e-6)
        torch.testing.assert_close(action_term.current_targets, expected_targets, rtol=0.0, atol=1.0e-6)

        held_targets = action_term.current_targets.clone()
        for _ in range(runtime_env.cfg.decimation + 1):
            action_term.apply_actions()
        torch.testing.assert_close(action_term.current_targets, held_targets, rtol=0.0, atol=0.0)

        _assert_shared_state_lifecycle(runtime_env)
        _assert_diagnostics_runtime(runtime_env)
        latest_frame = torch.cat(
            (
                gm_mdp.joint_pos_raw(runtime_env, TACTILE_JOINT_CFG) / torch.pi,
                gm_mdp.joint_target(runtime_env, "hand_joint_pos") / torch.pi,
                gm_mdp.last_action(runtime_env, "hand_joint_pos"),
                gm_mdp.tip_contact_bits_ema(
                    runtime_env,
                    TACTILE_TIP_SENSOR_NAMES,
                    TACTILE_FINGER_NON_TIP_SENSOR_NAMES,
                    TACTILE_PALM_SENSOR_NAME,
                ),
            ),
            dim=-1,
        )  # `[B,52]`，与正式 semantic ObsTerms 的声明顺序一致
        observed_latest = obs["policy"] if TASK_ID == CURRENT_TASK_ID else obs["policy"][:, -1]
        torch.testing.assert_close(observed_latest, latest_frame, rtol=0.0, atol=1.0e-6)

        if TASK_ID == HISTORY_TASK_ID:
            # 只结束 env 0：其历史应重建 reset prefix，env 1 的 causal history 不得被 collateral reset。
            runtime_env.episode_length_buf[0] = runtime_env.leap_adr_episode_lengths[0] - 2
            runtime_env.episode_length_buf[1] = 0
            partial_obs, _, partial_terminated, partial_truncated, partial_info = env.step(torch.zeros_like(actions))
            assert partial_truncated.tolist() == [True, False]
            assert not torch.any(partial_terminated)
            env0_prefix = partial_obs["policy"][0, :1].expand_as(partial_obs["policy"][0])
            torch.testing.assert_close(partial_obs["policy"][0], env0_prefix, rtol=0.0, atol=0.0)
            assert not torch.allclose(
                partial_obs["policy"][1], partial_obs["policy"][1, :1].expand_as(partial_obs["policy"][1])
            )
            _assert_episode_diagnostic_extras(partial_info)

        # 把两个 env 推到各自 sampled horizon 前一帧；本 step 应统一 timeout、flush 并 reset。
        runtime_env.episode_length_buf[:] = runtime_env.leap_adr_episode_lengths - 2
        _, _, terminated, truncated, info = env.step(torch.zeros_like(actions))
        assert torch.all(truncated) and not torch.any(terminated)
        _assert_episode_diagnostic_extras(info)
    except BaseException:
        traceback.print_exc()  # SimulationApp.close 会提前终止 pytest reporter，先保留真实 failure stack
        raise
    finally:
        if env is not None:
            env.close()


def _assert_observation_contract(obs: Mapping[str, torch.Tensor], *, expect_reset_prefix: bool) -> None:
    r"""检查 actor/central critic shape、history reset prefix 与 finite values。"""

    assert set(obs) >= {"policy", "critic"}
    expected_policy_shape = (SMOKE_NUM_ENVS, 52) if TASK_ID == CURRENT_TASK_ID else (SMOKE_NUM_ENVS, 30, 52)
    assert obs["policy"].shape == expected_policy_shape
    assert obs["critic"].shape == (SMOKE_NUM_ENVS, 152)
    assert torch.isfinite(obs["policy"]).all() and torch.isfinite(obs["critic"]).all()
    if TASK_ID == HISTORY_TASK_ID and expect_reset_prefix:
        repeated_first_frame = obs["policy"][:, :1].expand_as(obs["policy"])
        torch.testing.assert_close(obs["policy"], repeated_first_frame, rtol=0.0, atol=0.0)


def _assert_adr_runtime(runtime_env) -> None:
    r"""核对 sampled horizon、48D state 与 mass/COM PhysX actual values。"""

    horizon_s = runtime_env.leap_adr_episode_lengths.float() * runtime_env.step_dt
    assert torch.all((horizon_s >= 20.0) & (horizon_s <= 120.0))
    assert runtime_env.leap_adr_increment == 1  # 第 0 档在首次 reset check 自动 bootstrap 到第 1 档
    state = runtime_env._gm_adr_state.values
    assert state.shape == (SMOKE_NUM_ENVS, 48)
    assert torch.isfinite(state).all()
    assert torch.all((state[:, 0] >= 1.1) & (state[:, 0] <= 1.25))  # actual prestartup isotropic scale

    object_asset = runtime_env.scene["object"]
    actual_mass = object_asset.root_physx_view.get_masses().float().mean(dim=-1).to(runtime_env.device)
    torch.testing.assert_close(state[:, 1], actual_mass, rtol=0.0, atol=1.0e-6)
    actual_com = object_asset.root_physx_view.get_coms().cpu()
    default_com = runtime_env._gm_default_object_coms_cpu
    if actual_com.ndim == 2:
        actual_offset = (actual_com[:, :3] - default_com[:, :3]).to(runtime_env.device)
    else:
        actual_offset = (actual_com[:, :, :3] - default_com[:, :, :3]).mean(dim=1).to(runtime_env.device)
    torch.testing.assert_close(state[:, 2:5], actual_offset, rtol=0.0, atol=1.0e-6)
    assert torch.all(torch.abs(state[:, 2:5]) <= 0.01 + 1.0e-7)


def _assert_body_yaw_reset(runtime_env) -> None:
    r"""验证 reset 是 default pose 的 object-body yaw 右乘，且支撑面法向与 hand `+z_h` 对齐。"""

    object_asset = runtime_env.scene["object"]
    current_quat_w = object_asset.data.root_quat_w
    default_quat_w = object_asset.data.default_root_state[:, 3:7]
    relative_body_quat = math_utils.quat_mul(math_utils.quat_inv(default_quat_w), current_quat_w)  # $q_0^{-1}q$
    relative_rotvec = math_utils.axis_angle_from_quat(relative_body_quat)  # 应严格平行于 body $z_o$
    torch.testing.assert_close(relative_rotvec[:, :2], torch.zeros_like(relative_rotvec[:, :2]), atol=1.0e-5, rtol=0.0)
    assert torch.all(torch.abs(relative_rotvec[:, 2]) <= runtime_env.leap_adr_object_body_yaw + 1.0e-5)

    object_z_w = math_utils.quat_apply(
        current_quat_w,
        torch.tensor([0.0, 0.0, 1.0], device=runtime_env.device).expand(SMOKE_NUM_ENVS, -1),
    )
    command = runtime_env.command_manager.get_term("goal_pose")
    command.ensure_post_physics_progress_updated(runtime_env)
    alignment = torch.sum(object_z_w * command.axis_w, dim=-1)  # $z_o^{w\mathsf T}z_h^w$
    assert torch.all(alignment > 0.999), f"object/hand support normals are not aligned: {alignment.tolist()}"


def _assert_shared_state_lifecycle(runtime_env) -> None:
    r"""Contact/command 在同 step 幂等；partial contact reset 当前 stamp 保持零。"""

    contact = runtime_env._gm_tactile_contact_state
    command = runtime_env.command_manager.get_term("goal_pose")
    assert torch.all(contact.last_update_step == runtime_env.common_step_counter)
    assert torch.all(command.last_progress_step == runtime_env.common_step_counter)
    contact_before = contact.force_ema.clone()
    net_rotation_before = command.net_rotation_rad.clone()
    contact.ensure_updated(runtime_env)
    command.ensure_post_physics_progress_updated(runtime_env)
    torch.testing.assert_close(contact.force_ema, contact_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(command.net_rotation_rad, net_rotation_before, rtol=0.0, atol=0.0)

    contact.reset(runtime_env, torch.tensor([0], device=runtime_env.device))
    contact.ensure_updated(runtime_env)
    assert torch.count_nonzero(contact.force_ema[0]) == 0  # 同 stamp 不从 stale ContactSensor 重填
    torch.testing.assert_close(contact.force_ema[1], contact_before[1], rtol=0.0, atol=0.0)


def _assert_diagnostics_runtime(runtime_env) -> None:
    r"""检查 diagnostics 已按当前 policy stamp 更新一次，且在线 summary 全部 finite。"""

    command = runtime_env.command_manager.get_term("goal_pose")
    diagnostics = command.diagnostics
    assert diagnostics is not None
    assert torch.all(diagnostics.step_count >= 1.0)
    assert torch.all(diagnostics.last_update_step == runtime_env.common_step_counter)
    for name, value in diagnostics.metrics.items():
        assert torch.isfinite(value).all(), f"non-finite online diagnostic metric: {name}={value}"


def _assert_episode_diagnostic_extras(info: Mapping[str, Any]) -> None:
    r"""Forced timeout 必须一次性输出带单位的核心 episode metrics，且所有标量 finite。"""

    assert "log" in info
    episode_log = info["log"]
    required_metric_suffixes = (
        "rotation/axis_speed_mean_rad_s",
        "rotation/axis_speed_abs_mean_rad_s",
        "rotation/off_axis_ang_vel_rms_rad_s",
        "pose/anchor_distance_mean_m",
        "pose/anchor_distance_max_m",
        "pose/orientation_keypoint_error_mean_m",
        "action/policy_delta_rms_per_s",
        "action/executed_delta_rms_per_s",
        "action/target_delta_rms_rad_s",
        "action/target_tracking_error_rms_rad",
        "contact/tip_active_count_mean",
        "contact/palm_occupancy_fraction",
        "contact/finger_non_tip_occupancy_fraction",
        "contact/tip_force_ema_mean_N",
        "contact/palm_force_ema_mean_N",
        "task/episode_duration_s",
        "task/sampled_horizon_s",
        "termination/time_out_fraction",
        "adr/actual_object_mass_kg",
        "adr/actual_com_offset_norm_m",
        "adr/actual_joint_stiffness_mean",
        "adr/actual_joint_damping_mean",
        "adr/actual_action_noise_std",
        "adr/actual_latency_steps_mean",
        "adr/actual_wrench_gate_fraction",
        "adr/actual_max_linear_acceleration_m_s2",
    )
    for suffix in required_metric_suffixes:
        key = f"Metrics/goal_pose/{suffix}"
        assert key in episode_log, f"missing episode diagnostic key: {key}"
        assert torch.isfinite(torch.as_tensor(episode_log[key])).all(), f"non-finite episode diagnostic: {key}"
    assert episode_log["Metrics/goal_pose/termination/time_out_fraction"] == 1.0
