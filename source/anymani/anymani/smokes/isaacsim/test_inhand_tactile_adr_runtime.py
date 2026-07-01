r"""IsaacSim smoke for `AnyMani-LeapHand-Tactile-ADR-v0`.

本文件是显式 runtime smoke，不属于默认 `pytest` contract suite。它验证 ADR env 的最低运行时
闭环：Gym 注册、ManagerBasedRLEnv 创建、reset/step、obs/action 维度、随机 horizon buffer、
ADR curriculum 日志状态与 action/material/wrench runtime hook 不崩溃。

运行命令：

```bash
cd /home/hac/isaac/AnyMani
source /home/hac/isaac/env_isaaclab/bin/activate
timeout --kill-after=20s 240s /home/hac/isaac/IsaacLab/isaaclab.sh -p -m pytest \
  source/anymani/anymani/smokes/isaacsim/test_inhand_tactile_adr_runtime.py -q -s
```
"""

from __future__ import annotations

# ruff: noqa: I001
# IsaacLab smoke 必须先启动 AppLauncher，再 import gym/pxr/任务注册模块。

from collections.abc import Mapping
from typing import Any

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
import pytest
import torch
from isaaclab_tasks.utils import parse_env_cfg

import anymani.tasks.inhand.config.leaphand  # noqa: F401  # 显式注册 LeapHand inhand envs

TASK_ID = "AnyMani-LeapHand-Tactile-ADR-v0"
r"""当前 smoke 验证的新 ADR 环境 id；不能退化为 baseline `Tactile-v0`。"""

SMOKE_NUM_ENVS = 2
r"""runtime smoke 最小并行数；两个 env 足够验证 per-env horizon buffer。"""

SMOKE_STEPS = 4
r"""短步数覆盖 action noise/latency、interval manager 初始化和 reset 后第一批 step。"""


def teardown_module() -> None:
    r"""关闭 IsaacSim app，避免 smoke 结束后遗留 Kit 进程。"""

    simulation_app.close()


@pytest.mark.isaacsim
def test_tactile_adr_env_resets_steps_and_preserves_contract() -> None:
    r"""验证 tactile ADR env 的最低运行时 contract。

    科研语义：ADR 是 N000 tactile baseline 的随机化/课程分支，不应改变 actor/critic/action
    的学习接口。因此本 smoke 的关键断言是：
    $$
    \dim(o^\pi)=43,\qquad \dim(o^V)=62,\qquad \dim(a)=16.
    $$
    同时，官方随机 horizon 必须满足：
    $$
    T_i\in[20s,120s].
    $$
    """

    # 训练 cfg 默认 4096 env；smoke 显式压到 2 env，避免把 runtime contract 变成压力测试。
    env_cfg = parse_env_cfg(TASK_ID, device="cuda:0", num_envs=SMOKE_NUM_ENVS)
    env = None
    try:
        # `gym.make` 触发 scene creation、manager prepare、startup/prestartup events 和 sim reset。
        env = gym.make(TASK_ID, cfg=env_cfg)
        runtime_env = env.unwrapped  # ManagerBasedRLEnv；ADR buffers 和 manager state 挂在该对象上。
        runtime_env.sim._app_control_on_stop_handle = None  # smoke 退出时避免 Kit timeline 二次接管。

        # reset 后应建立 policy/critic observation tree，并在 reset event 中采样 per-env horizon。
        obs, _ = env.reset()
        _assert_obs_contract(obs)
        _assert_adr_runtime_buffers(runtime_env)

        # 用 action_space 的 batch shape 生成 $a\in[-1,1]^{N\times16}$，覆盖 action term 和 step pipeline。
        for step_id in range(SMOKE_STEPS):
            actions = 2.0 * torch.rand(env.action_space.shape, device=runtime_env.device) - 1.0
            obs, reward, terminated, truncated, _ = env.step(actions)
            _assert_obs_contract(obs)
            assert reward.shape == (SMOKE_NUM_ENVS,), f"reward shape mismatch at step {step_id}: {reward.shape}"
            assert terminated.shape == truncated.shape == reward.shape, "done/reward batch shapes must agree"
            assert torch.isfinite(reward).all(), f"non-finite reward at step {step_id}"
    finally:
        if env is not None:
            env.close()


def _assert_obs_contract(obs: Mapping[str, torch.Tensor]) -> None:
    r"""检查 tactile ADR 与 N000 baseline 保持相同观测维度。"""

    assert set(obs.keys()) >= {"policy", "critic"}, f"unexpected obs groups: {tuple(obs.keys())}"
    assert obs["policy"].shape == (SMOKE_NUM_ENVS, 43), f"policy obs must be (N,43), got {obs['policy'].shape}"
    assert obs["critic"].shape == (SMOKE_NUM_ENVS, 62), f"critic obs must be (N,62), got {obs['critic'].shape}"
    _assert_finite_tree("obs", obs)


def _assert_adr_runtime_buffers(runtime_env) -> None:
    r"""检查 ADR scheduler/event/action runtime buffer 已建立且数值范围正确。"""

    assert runtime_env.action_manager.total_action_dim == 16  # $\dim(a)=16$，不改变 policy head。
    assert hasattr(runtime_env, "leap_adr_episode_lengths"), "missing randomized horizon buffer"
    horizon_s = runtime_env.leap_adr_episode_lengths.float() * runtime_env.step_dt  # $T_i$，单位秒。
    assert torch.all(horizon_s >= 20.0), f"ADR horizon below 20s: {horizon_s}"
    assert torch.all(horizon_s <= 120.0), f"ADR horizon above 120s: {horizon_s}"
    assert hasattr(runtime_env, "leap_adr_increment"), "missing ADR global increment k"
    assert runtime_env.leap_adr_increment == 0  # 初始 smoke 未达到 rotation-rate trigger，应仍在第 0 档。
    action_term = runtime_env.action_manager.get_term("hand_joint_pos")
    assert action_term.raw_actions.shape == (SMOKE_NUM_ENVS, 16)  # last_action 观测来源仍是 raw policy action。


def _assert_finite_tree(name: str, value: Any) -> None:
    r"""递归检查 observation tree 中所有 tensor 是否 finite。"""

    if isinstance(value, Mapping):
        for child_name, child_value in value.items():
            _assert_finite_tree(f"{name}.{child_name}", child_value)
        return
    if torch.is_tensor(value):
        assert torch.isfinite(value).all(), f"non-finite tensor in {name}"
