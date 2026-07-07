r"""IsaacSim smoke for generated N041 action and N05x observation variants.

本文件是显式 runtime smoke，不属于默认 `pytest` contract suite。它验证 generated N4x/N5x 的
最低运行时闭环：Gym 注册、ManagerBasedRLEnv 创建、reset/step、policy obs 维度、action term runtime
contract，以及 `current_targets` 是否在 step 后保持 finite。

运行命令：

```bash
cd /home/hac/isaac/AnyMani
source /home/hac/isaac/env_isaaclab/bin/activate
timeout --kill-after=20s 240s /home/hac/isaac/IsaacLab/isaaclab.sh -p -m pytest \
  source/anymani/anymani/smokes/isaacsim/test_inhand_generated_n4x_action_runtime.py -q -s
```

若要检查指定 N05x observation variant，用环境变量覆盖 task id；每次只创建一个 IsaacSim env，
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

DEFAULT_TASK_ID = "AnyMani-LeapHand-ADR-Generated-right_t4_i4_m4_r4-EMAAbsolute-v0"
r"""默认 smoke 仍覆盖既有 N041 EMAAbsolute action variant。"""

TASK_ID = os.environ.get(
    "ANYMANI_INHAND_GENERATED_SMOKE_TASK",
    DEFAULT_TASK_ID,
)
r"""本次 smoke 实际检查的 generated env id；可由环境变量覆盖到 N050/N051。"""


def teardown_module() -> None:
    r"""关闭 IsaacSim app，避免 smoke 结束后遗留 Kit 进程。"""

    simulation_app.close()


@pytest.mark.isaacsim
def test_generated_n4x_n5x_variant_reset_step_and_exposes_targets() -> None:
    r"""验证 generated action / observation variants 的最低 runtime contract。

    N041 / N050 / N051 的 policy observation 都应保持 96D。具体检查哪个 task id 由
    `ANYMANI_INHAND_GENERATED_SMOKE_TASK` 控制，默认检查 N041：

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

        # 用零动作避免随机接触扰动主导 smoke；本测试只证伪 action law runtime 崩溃。
        zero_actions = torch.zeros(env.action_space.shape, device=runtime_env.device)  # $a_t=0$，形状 $[1,16]$。
        for step_id in range(SMOKE_STEPS):
            obs, reward, terminated, truncated, _ = env.step(zero_actions)
            _assert_policy_obs_contract(obs, TASK_ID)
            _assert_action_runtime_contract(runtime_env, TASK_ID)
            assert reward.shape == (SMOKE_NUM_ENVS,), f"{TASK_ID} reward shape mismatch at step {step_id}"
            assert terminated.shape == truncated.shape == reward.shape, f"{TASK_ID} done/reward shape mismatch"
            assert torch.isfinite(reward).all(), f"{TASK_ID} non-finite reward at step {step_id}"
    finally:
        if env is not None:
            env.close()


def _assert_policy_obs_contract(obs: Mapping[str, torch.Tensor], task_id: str) -> None:
    r"""检查 generated variant policy observation 仍是 96D actor-facing tensor。"""

    assert "policy" in obs, f"{task_id} missing policy observation group"
    assert obs["policy"].shape == (SMOKE_NUM_ENVS, 96), f"{task_id} policy obs shape: {obs['policy'].shape}"
    assert torch.isfinite(obs["policy"]).all(), f"{task_id} policy obs contains non-finite values"


def _assert_action_runtime_contract(runtime_env, task_id: str) -> None:
    r"""检查 action term 对 obs/reward 暴露的共享 runtime contract。"""

    assert runtime_env.action_manager.total_action_dim == 16  # $\dim(a)=16$，保持 N030/N040 policy head 维度。
    action_term = runtime_env.action_manager.get_term("hand_joint_pos")  # N4x/N5x 唯一 hand action term。
    assert action_term.current_targets.shape == (SMOKE_NUM_ENVS, 16), f"{task_id} current_targets shape mismatch"
    assert action_term.executed_actions.shape == (SMOKE_NUM_ENVS, 16), f"{task_id} executed_actions shape mismatch"
    assert action_term.pregrasp_targets.shape == (SMOKE_NUM_ENVS, 16), f"{task_id} pregrasp_targets shape mismatch"
    assert torch.isfinite(action_term.current_targets).all(), f"{task_id} current_targets contains non-finite values"
    assert torch.isfinite(action_term.executed_actions).all(), f"{task_id} executed_actions contains non-finite values"
    assert torch.isfinite(action_term.pregrasp_targets).all(), f"{task_id} pregrasp_targets contains non-finite values"
