r"""Pure contract tests for GM reward helper resolution.

这些测试不启动 Isaac Sim，只验证 `rewards_common.py` 对 command term 的读取契约：

1. goal quaternion 必须来自 command term 的 canonical buffer，例如 `goal_quat_w`；
2. reward helper 不允许再从 policy-facing `get_command()` tensor 反推 quaternion，
   因为 `command_output` 现在可以被配置为省略 quaternion、改 frame，或重排字段。
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REWARDS_COMMON_PATH = Path(__file__).resolve().parents[1] / "mdp" / "rewards" / "rewards_common.py"
r"""被测试的 reward helper 源文件路径；path import 可避免触发完整 Isaac runtime。"""


def _load_rewards_common_module() -> types.ModuleType:
    r"""加载 `rewards_common.py`，并 stub 掉最小 Isaac 依赖。"""

    math_stub = types.ModuleType("isaaclab.utils.math")

    envs_stub = types.ModuleType("isaaclab.envs")
    envs_stub.ManagerBasedRLEnv = object

    replacements = {
        "isaaclab": types.ModuleType("isaaclab"),
        "isaaclab.envs": envs_stub,
        "isaaclab.utils": types.ModuleType("isaaclab.utils"),
        "isaaclab.utils.math": math_stub,
    }
    previous = {name: sys.modules.get(name) for name in replacements}
    try:
        sys.modules.update(replacements)
        spec = importlib.util.spec_from_file_location("_gm_rewards_common_for_test", REWARDS_COMMON_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


def test_resolve_goal_quat_w_reads_canonical_term_buffer() -> None:
    r"""`goal_quat_w` 应优先来自 command term buffer，而不是 `get_command()`。"""

    module = _load_rewards_common_module()

    expected_goal = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)  # `[B,4]`，测试用目标 quaternion
    command_term = SimpleNamespace(goal_quat_w=expected_goal)  # canonical goal state buffer
    command_manager = SimpleNamespace(
        get_term=lambda _name: command_term,
        get_command=lambda _name: (_ for _ in ()).throw(AssertionError("resolve_goal_quat_w 不应读取 get_command()")),
    )
    env = SimpleNamespace(command_manager=command_manager)

    goal = module.resolve_goal_quat_w(env, "goal_pose")

    assert torch.allclose(goal, expected_goal)


def test_resolve_goal_quat_w_rejects_policy_tensor_fallback() -> None:
    r"""当 term 不显式暴露目标 quaternion 时，reward helper 必须失败，而不是猜 policy-facing tensor。"""

    module = _load_rewards_common_module()

    command_term = SimpleNamespace()  # 故意不提供 `goal_quat_w` / `quat_command_w`
    command_manager = SimpleNamespace(
        get_term=lambda _name: command_term,
        get_command=lambda _name: torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=torch.float32),  # 即使给 tensor，也不应被读取
    )
    env = SimpleNamespace(command_manager=command_manager)

    with pytest.raises(RuntimeError, match="must expose `goal_quat_w` / `quat_command_w`"):
        module.resolve_goal_quat_w(env, "goal_pose")
