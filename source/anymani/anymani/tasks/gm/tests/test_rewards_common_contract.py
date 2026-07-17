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

    def matrix_from_quat(quat: torch.Tensor) -> torch.Tensor:
        r"""测试用 `(w,x,y,z)` quaternion 到 rotation matrix 转换。"""

        quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True)
        w, x, y, z = quat.unbind(dim=-1)
        return torch.stack(
            (
                1 - 2 * (y * y + z * z),
                2 * (x * y - w * z),
                2 * (x * z + w * y),
                2 * (x * y + w * z),
                1 - 2 * (x * x + z * z),
                2 * (y * z - w * x),
                2 * (x * z - w * y),
                2 * (y * z + w * x),
                1 - 2 * (x * x + y * y),
            ),
            dim=-1,
        ).reshape(-1, 3, 3)

    math_stub.matrix_from_quat = matrix_from_quat

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


def test_full_pose_keypoint_reward_is_one_at_goal_and_monotonic_in_translation() -> None:
    r"""归一化 full-pose kernel 在目标处为 1，并随纯平移误差严格下降。"""

    module = _load_rewards_common_module()
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    goal_pos = torch.zeros(1, 3)
    rewards = []
    for offset in (0.0, 0.01, 0.02):
        current_pos = torch.tensor([[offset, 0.0, 0.0]])
        distances = module.full_pose_keypoint_distances(current_pos, identity, goal_pos, identity, radius=0.05)
        rewards.append(module.normalized_keypoint_kernel(distances).item())

    assert rewards[0] == pytest.approx(1.0)
    assert rewards[0] > rewards[1] > rewards[2]


def test_impulse_rate_integrates_identically_at_20_and_30_hz() -> None:
    r"""一次 rotation/goal/termination impulse 的积分不得依赖 policy frequency。"""

    module = _load_rewards_common_module()
    impulse = torch.tensor([0.025, 1.0])
    rate_20_hz = module.impulse_to_rate(impulse, step_dt=0.05)
    rate_30_hz = module.impulse_to_rate(impulse, step_dt=1.0 / 30.0)

    assert torch.allclose(rate_20_hz * 0.05, impulse)
    assert torch.allclose(rate_30_hz * (1.0 / 30.0), impulse)
