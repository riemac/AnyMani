r"""直接执行生产进展奖励，核验截断角度与20 Hz奖励率的物理量纲。"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import torch


def test_progress_clip_changes_only_saturated_increments_and_preserves_reverse_penalty() -> None:
    r"""低速斜率不变，0.025→0.04 rad仅扩展饱和区间；正反向仍为奇对称。"""

    path = Path(__file__).resolve().parents[1] / "mdp" / "rewards.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "signed_rotation_progress_rate"
    )
    namespace = {"torch": torch, "get_rotation_command": lambda env, _: env.command}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    delta = torch.tensor((-0.06, -0.03, -0.01, 0.0, 0.01, 0.03, 0.06))
    env = SimpleNamespace(step_dt=0.05, command=SimpleNamespace(delta_psi=delta))
    reward = namespace["signed_rotation_progress_rate"]
    original = reward(env, "goal_pose", clip_rad_per_step=0.025)
    extended = reward(env, "goal_pose", clip_rad_per_step=0.04)
    torch.testing.assert_close(original, torch.tensor((-0.5, -0.5, -0.2, 0.0, 0.2, 0.5, 0.5)))
    torch.testing.assert_close(extended, torch.tensor((-0.8, -0.6, -0.2, 0.0, 0.2, 0.6, 0.8)))
    assert torch.equal(extended, -extended.flip(0))
    assert torch.equal(original[2:5], extended[2:5])
