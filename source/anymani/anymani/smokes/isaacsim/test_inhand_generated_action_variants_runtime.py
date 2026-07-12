r"""IsaacSim smoke for generated in-hand action/observation variants.

本文件是显式 runtime smoke，不属于默认 `pytest` contract suite。它验证 generated variants 的
最低运行时闭环：Gym 注册、ManagerBasedRLEnv 创建、reset/step、policy obs 维度、action term runtime
contract，以及 PolicyStepTarget 的 target-buffer lifecycle。

运行命令：

```bash
cd /home/hac/isaac/AnyMani
source /home/hac/isaac/env_isaaclab/bin/activate
timeout --kill-after=20s 240s /home/hac/isaac/IsaacLab/isaaclab.sh -p -m pytest \
  source/anymani/anymani/smokes/isaacsim/test_inhand_generated_action_variants_runtime.py -q -s
```

若要检查其它 generated variant，用环境变量覆盖 task id；每次只创建一个 IsaacSim env，
避免同一 Kit 进程内连续创建/销毁多个 ManagerBasedRLEnv 引入不稳定因素。
"""

from __future__ import annotations

# ruff: noqa: I001
# IsaacLab smoke 必须先启动 AppLauncher，再 import gym / 任务注册模块。

from collections.abc import Mapping
import os

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
import pytest
import torch
from isaaclab_tasks.utils import parse_env_cfg

import anymani.tasks.inhand.config.generated_right_t4_i4_m4_r4  # noqa: F401  # 注册 generated N03x/N04x envs

SMOKE_NUM_ENVS = 1
r"""generated-hand action smoke 的最小并行数；1 env 足以验证 reset/step runtime contract。"""

SMOKE_STEPS = 2
r"""短步数覆盖 action process/apply、ADR latency buffer 与 current target 更新。"""

POLICY_STEP_TARGET_TASK_ID = "AnyMani-LeapHand-ADR-Generated-right_t4_i4_m4_r4-PolicyStepTarget-v0"
r"""需要验证单次 policy-step target recurrence 的语义化 task id。"""

DEFAULT_TASK_ID = POLICY_STEP_TARGET_TASK_ID
r"""默认 smoke 覆盖新增 PolicyStepTarget；环境变量仍可切换到其它 generated variant。"""

TASK_ID = os.environ.get(
    "ANYMANI_INHAND_GENERATED_SMOKE_TASK",
    DEFAULT_TASK_ID,
)
r"""本次 smoke 实际检查的 generated env id。"""


def teardown_module() -> None:
    r"""关闭 IsaacSim app，避免 smoke 结束后遗留 Kit 进程。"""

    simulation_app.close()


@pytest.mark.isaacsim
def test_generated_variant_reset_step_and_action_lifecycle() -> None:
    r"""验证 generated action / observation variants 的最低 runtime contract。

    当前 generated action/observation variants 的 policy observation 都应保持 96D。具体 task id 由
    `ANYMANI_INHAND_GENERATED_SMOKE_TASK` 控制；默认检查 PolicyStepTarget：

    $$
    \dim(o^\pi)=96,
    $$

    action term 必须暴露当前 PD target：

    $$
    u_t=\texttt{current\_targets}\in\mathbb R^{1\times16}.
    $$
    """

    # 训练配置默认 4096 env；smoke 压到 1 env，验证语义而非吞吐。
    env_cfg = parse_env_cfg(TASK_ID, device="cuda:0", num_envs=SMOKE_NUM_ENVS)  # 小规模 runtime cfg。
    env = None
    try:
        # `gym.make` 触发 generated hand spawn、structural collision filter、manager prepare 与 reset hooks。
        env = gym.make(TASK_ID, cfg=env_cfg)
        runtime_env = env.unwrapped  # ManagerBasedRLEnv；action manager 与 ADR runtime 挂在该对象上。
        runtime_env.sim._app_control_on_stop_handle = None  # smoke 退出时避免 Kit timeline 二次接管。

        # reset 后 obs/action term 必须已经建立，且 target buffer 应与 reset joint pose 对齐。
        obs, _ = env.reset()
        _assert_policy_obs_contract(obs, TASK_ID)
        _assert_action_runtime_contract(runtime_env, TASK_ID)

        # PolicyStepTarget 使用非零动作验证一次 policy step 只产生一次 $a_t^{exec}/24$ target increment。
        if TASK_ID == POLICY_STEP_TARGET_TASK_ID:
            obs, reward, terminated, truncated = _step_and_assert_policy_step_target(env, runtime_env)
            _assert_policy_obs_contract(obs, TASK_ID)
            _assert_action_runtime_contract(runtime_env, TASK_ID)
            assert reward.shape == (SMOKE_NUM_ENVS,), f"{TASK_ID} reward shape mismatch"
            assert terminated.shape == truncated.shape == reward.shape, f"{TASK_ID} done/reward shape mismatch"
            assert torch.isfinite(reward).all(), f"{TASK_ID} non-finite reward"

        # 其它 variant 仍使用零动作做最低 reset/step smoke，不对其 action law 作额外归因。
        zero_actions = torch.zeros(env.action_space.shape, device=runtime_env.device)  # $a_t=0$，形状 `[1,16]`。
        remaining_steps = SMOKE_STEPS - int(TASK_ID == POLICY_STEP_TARGET_TASK_ID)
        for step_id in range(remaining_steps):
            obs, reward, terminated, truncated, _ = env.step(zero_actions)
            _assert_policy_obs_contract(obs, TASK_ID)
            _assert_action_runtime_contract(runtime_env, TASK_ID)
            assert reward.shape == (SMOKE_NUM_ENVS,), f"{TASK_ID} reward shape mismatch at step {step_id}"
            assert terminated.shape == truncated.shape == reward.shape, f"{TASK_ID} done/reward shape mismatch"
            assert torch.isfinite(reward).all(), f"{TASK_ID} non-finite reward at step {step_id}"
    finally:
        if env is not None:
            env.close()


def _step_and_assert_policy_step_target(env, runtime_env):
    r"""验证单次 target recurrence、重复 apply 幂等性与 partial reset。

    Returns:
        tuple: 一次 `env.step()` 返回的 observation、reward、terminated 与 truncated。
    """

    action_term = runtime_env.action_manager.get_term("hand_joint_pos")  # PolicyStep target-buffer action runtime。

    # 关闭 action noise，并把 latency/history 置零，使 $a_t^{exec}=a_t$，隔离 lifecycle 本身。
    runtime_env.leap_adr_action_noise = 0.0  # 训练仍使用 ADR；仅 smoke 临时固定 noise=0。
    action_term._latency_steps.zero_()  # 每关节 latency index $\ell=0$。
    action_term._action_history.zero_()  # 清除 reset 前后可能残留的 noisy action history。

    targets_before = action_term.current_targets.clone()  # $u_t$，单位 rad，形状 `[1,16]`。
    joint_mid = 0.5 * (action_term._joint_lower + action_term._joint_upper)  # soft-limit 中点，避免 clipping。
    actions = torch.where(joint_mid >= targets_before, 0.5, -0.5)  # 朝中点移动的 $a_t\in\{-0.5,0.5\}$。
    expected_targets = torch.clamp(
        targets_before + actions / 24.0,
        min=action_term._joint_lower,
        max=action_term._joint_upper,
    )  # 单次 policy-step recurrence $u_{t+1}=clip(u_t+a_t/24)$。

    obs, reward, terminated, truncated, _ = env.step(actions)
    torch.testing.assert_close(action_term.executed_actions, actions, rtol=0.0, atol=1.0e-6)
    torch.testing.assert_close(action_term.current_targets, expected_targets, rtol=0.0, atol=1.0e-6)

    # 模拟 decimation loop 的重复 apply；幂等 hold 不能再次推进 target accumulator。
    held_targets = action_term.current_targets.clone()  # 已提交的 $u_{t+1}$。
    for _ in range(runtime_env.cfg.decimation + 1):
        action_term.apply_actions()
    torch.testing.assert_close(action_term.current_targets, held_targets, rtol=0.0, atol=0.0)

    # Partial reset 必须恢复指定 env 的 target/history，不影响 action runtime shape。
    env_ids = torch.tensor([0], device=runtime_env.device, dtype=torch.long)
    action_term.reset(env_ids)
    if hasattr(runtime_env, "leap_official_reset_joint_pos"):
        reset_targets = runtime_env.leap_official_reset_joint_pos[env_ids]
    else:
        reset_targets = action_term._asset.data.default_joint_pos[env_ids][:, action_term._joint_ids]
    torch.testing.assert_close(action_term.current_targets[env_ids], reset_targets, rtol=0.0, atol=1.0e-6)
    assert torch.count_nonzero(action_term._action_history[env_ids]) == 0

    return obs, reward, terminated, truncated


def _assert_policy_obs_contract(obs: Mapping[str, torch.Tensor], task_id: str) -> None:
    r"""检查 generated variant policy observation 仍是 96D actor-facing tensor。"""

    assert "policy" in obs, f"{task_id} missing policy observation group"
    assert obs["policy"].shape == (SMOKE_NUM_ENVS, 96), f"{task_id} policy obs shape: {obs['policy'].shape}"
    assert torch.isfinite(obs["policy"]).all(), f"{task_id} policy obs contains non-finite values"


def _assert_action_runtime_contract(runtime_env, task_id: str) -> None:
    r"""检查 action term 对 obs/reward 暴露的共享 runtime contract。"""

    assert runtime_env.action_manager.total_action_dim == 16  # $\dim(a)=16$，保持 generated policy head 维度。
    action_term = runtime_env.action_manager.get_term("hand_joint_pos")  # generated variant 唯一 hand action term。
    assert action_term.current_targets.shape == (SMOKE_NUM_ENVS, 16), f"{task_id} current_targets shape mismatch"
    assert action_term.executed_actions.shape == (SMOKE_NUM_ENVS, 16), f"{task_id} executed_actions shape mismatch"
    assert action_term.pregrasp_targets.shape == (SMOKE_NUM_ENVS, 16), f"{task_id} pregrasp_targets shape mismatch"
    assert torch.isfinite(action_term.current_targets).all(), f"{task_id} current_targets contains non-finite values"
    assert torch.isfinite(action_term.executed_actions).all(), f"{task_id} executed_actions contains non-finite values"
    assert torch.isfinite(action_term.pregrasp_targets).all(), f"{task_id} pregrasp_targets contains non-finite values"
