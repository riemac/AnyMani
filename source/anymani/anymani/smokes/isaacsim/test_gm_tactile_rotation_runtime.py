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
from isaaclab_tasks.utils import parse_env_cfg

import anymani.tasks.gm  # noqa: F401
from anymani.tasks.gm.config.single_asset.tactile_rotation_env_cfg import (
    TACTILE_FINGER_NON_TIP_SENSOR_NAMES,
    TACTILE_JOINT_CFG,
    TACTILE_PALM_SENSOR_NAME,
    TACTILE_TIP_SENSOR_NAMES,
)
from anymani.tasks.gm.mdp.observations.observations_tactile import tactile_rotation_policy_frame

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

        action_term = runtime_env.action_manager.get_term("hand_joint_pos")
        runtime_env.leap_adr_action_noise = 0.0  # smoke 隔离 target recurrence，不改变训练 cfg
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
        latest_frame = tactile_rotation_policy_frame(
            runtime_env,
            TACTILE_TIP_SENSOR_NAMES,
            TACTILE_FINGER_NON_TIP_SENSOR_NAMES,
            TACTILE_PALM_SENSOR_NAME,
            robot_cfg=TACTILE_JOINT_CFG,
        )
        observed_latest = obs["policy"] if TASK_ID == CURRENT_TASK_ID else obs["policy"][:, -1]
        torch.testing.assert_close(observed_latest, latest_frame, rtol=0.0, atol=1.0e-6)
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
