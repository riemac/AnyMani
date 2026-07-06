r"""IsaacSim smoke for generated N041 EMAAbsolute action variant.

本文件是显式 runtime smoke，不属于默认 `pytest` contract suite。它验证 N041 的最低运行时
闭环：Gym 注册、ManagerBasedRLEnv 创建、reset/step、policy obs 维度、EMA action term runtime contract，
以及 `current_targets` 是否在 step 后保持 finite。

运行命令：

```bash
cd /home/hac/isaac/AnyMani
source /home/hac/isaac/env_isaaclab/bin/activate
timeout --kill-after=20s 240s /home/hac/isaac/IsaacLab/isaaclab.sh -p -m pytest \
  source/anymani/anymani/smokes/isaacsim/test_inhand_generated_n4x_action_runtime.py -q -s
```
"""

from __future__ import annotations

# ruff: noqa: I001
# IsaacLab smoke 必须先启动 AppLauncher，再 import gym / 任务注册模块。

from collections.abc import Mapping

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

TASK_ID = "AnyMani-LeapHand-ADR-Generated-right_t4_i4_m4_r4-EMAAbsolute-v0"
r"""本 smoke 覆盖的 N041 EMAAbsolute env id。"""


def teardown_module() -> None:
    r"""关闭 IsaacSim app，避免 smoke 结束后遗留 Kit 进程。"""

    simulation_app.close()


@pytest.mark.isaacsim
def test_generated_n41_ema_absolute_resets_steps_and_exposes_targets() -> None:
    r"""验证 N041 generated EMAAbsolute action variant 的最低 runtime contract。

    N041 的 policy observation 应保持 N030 official 96D：

    $$
    \dim(o^\pi)=96,
    $$

    action term 必须暴露：

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
        _assert_policy_obs_contract(obs)
        _assert_action_runtime_contract(runtime_env)

        # 用零动作避免随机接触扰动主导 smoke；本测试只证伪 action law runtime 崩溃。
        zero_actions = torch.zeros(env.action_space.shape, device=runtime_env.device)  # $a_t=0$，形状 $[1,16]$。
        for step_id in range(SMOKE_STEPS):
            obs, reward, terminated, truncated, _ = env.step(zero_actions)
            _assert_policy_obs_contract(obs)
            _assert_action_runtime_contract(runtime_env)
            assert reward.shape == (SMOKE_NUM_ENVS,), f"N041 reward shape mismatch at step {step_id}"
            assert terminated.shape == truncated.shape == reward.shape, "N041 done/reward shape mismatch"
            assert torch.isfinite(reward).all(), f"N041 non-finite reward at step {step_id}"
    finally:
        if env is not None:
            env.close()


def _assert_policy_obs_contract(obs: Mapping[str, torch.Tensor]) -> None:
    r"""检查 N041 policy observation 仍是 96D actor-facing tensor。"""

    assert "policy" in obs, "N041 missing policy observation group"
    assert obs["policy"].shape == (SMOKE_NUM_ENVS, 96), f"N041 policy obs shape: {obs['policy'].shape}"
    assert torch.isfinite(obs["policy"]).all(), "N041 policy obs contains non-finite values"


def _assert_action_runtime_contract(runtime_env) -> None:
    r"""检查 action term 对 obs/reward 暴露的共享 runtime contract。"""

    assert runtime_env.action_manager.total_action_dim == 16  # $\dim(a)=16$，保持 N030/N040 policy head 维度。
    action_term = runtime_env.action_manager.get_term("hand_joint_pos")  # N4x 唯一 hand action term。
    assert action_term.current_targets.shape == (SMOKE_NUM_ENVS, 16), "N041 current_targets shape mismatch"
    assert action_term.executed_actions.shape == (SMOKE_NUM_ENVS, 16), "N041 executed_actions shape mismatch"
    assert action_term.pregrasp_targets.shape == (SMOKE_NUM_ENVS, 16), "N041 pregrasp_targets shape mismatch"
    assert torch.isfinite(action_term.current_targets).all(), "N041 current_targets contains non-finite values"
    assert torch.isfinite(action_term.executed_actions).all(), "N041 executed_actions contains non-finite values"
    assert torch.isfinite(action_term.pregrasp_targets).all(), "N041 pregrasp_targets contains non-finite values"
