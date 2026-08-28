r"""canonical mask/reset 的纯 Torch 合同测试。

测试不启动 Isaac Sim；它们只锁定 [env,joint] mask 的数学语义，使 ghost slot 不会在
动作、reset 或按 DOF 归一化的正则项中重新获得隐含权重。
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "mdp" / "canonical_runtime.py"
EVENTS_MODULE_PATH = Path(__file__).resolve().parents[1] / "mdp" / "events.py"


def _module():
    r"""按文件加载纯 tensor module，避免触发 gm 的 IsaacLab registry。"""

    spec = importlib.util.spec_from_file_location("gm_canonical_runtime_contract", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mask_broadcast_and_action_projection_are_exact() -> None:
    r"""一维 routing 可广播到 batch，任意大小 ghost action 必须为 0。"""

    module = _module()
    mask = module.normalize_active_joint_mask([True, False, True, False], batch_size=2, dof=4)
    action = torch.tensor([[1.0, 1.0e9, -2.0, -1.0e9], [0.5, -3.0, 2.0, 4.0]])

    assert mask.shape == (2, 4)
    torch.testing.assert_close(
        module.mask_action(action, mask), torch.tensor([[1.0, 0.0, -2.0, 0.0], [0.5, 0.0, 2.0, 0.0]])
    )


def test_reset_pose_uses_active_midpoint_and_zeroes_ghosts() -> None:
    r"""active reset 取 per-env limit midpoint，ghost 不受 source limit 数值影响。"""

    module = _module()
    limits = torch.tensor(
        [
            [[-1.0, 1.0], [0.2, 1.2], [-2.0, 2.0]],
            [[-0.5, 0.5], [0.3, 0.7], [-1.0, 3.0]],
        ]
    )
    mask = torch.tensor([[True, False, True], [False, True, False]])

    reset = module.canonical_reset_pose(limits, mask)
    torch.testing.assert_close(reset, torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0]]))


def test_masked_mean_does_not_depend_on_ghost_magnitude() -> None:
    r"""inactive slot 的任意大正则值不能污染 active-joint 平均量。"""

    module = _module()
    mask = torch.tensor([[True, False, True], [True, True, False]])
    values = torch.tensor([[2.0, 1.0e12, 4.0], [2.0, 4.0, 1.0e12]])

    torch.testing.assert_close(module.masked_mean(values, mask), torch.tensor([3.0, 3.0]))


def test_install_state_records_asset_rows_without_normalization() -> None:
    r"""asset_row 是离散 evidence routing，不应被转换成可归一化连续量。"""

    module = _module()
    env = SimpleNamespace(num_envs=2, device=torch.device("cpu"))
    mask = module.install_canonical_runtime_state(env, [[True, False], [False, True]], asset_rows=[7, 11], dof=2)

    torch.testing.assert_close(mask, torch.tensor([[True, False], [False, True]]))
    assert env._anymani_canonical_asset_row.dtype == torch.long
    assert env._anymani_canonical_asset_row.tolist() == [7, 11]


def test_invalid_mask_shape_fails_closed() -> None:
    r"""不能通过 silent padding 把非 canonical 动作维度混入 runtime。"""

    module = _module()
    with pytest.raises(ValueError, match="must have 4 joints"):
        module.normalize_active_joint_mask([True, False], batch_size=1, dof=4)


def test_round_robin_routing_matches_multi_asset_spawner_assignment() -> None:
    r"""第 b 个 env 必须消费 b mod R 的 asset mask/evidence row，不能按连续块分组。"""

    module = _module()
    mask_rows = torch.zeros(5, 16, dtype=torch.bool)
    mask_rows[:, 0] = True
    mask_rows[1, 1] = True
    mask_rows[2, 2] = True
    mask_rows[3, 3] = True
    mask_rows[4, 4] = True

    masks, rows = module.expand_round_robin_routing(mask_rows, [10, 11, 12, 13, 14], num_envs=12)

    assert rows.tolist() == [10, 11, 12, 13, 14, 10, 11, 12, 13, 14, 10, 11]
    torch.testing.assert_close(masks[7], mask_rows[2])  # env 7 -> prototype 2
    assert masks[:, 0].all()


def test_ghost_startup_locks_position_without_zero_velocity_braking_constraint() -> None:
    r"""Ghost 位置可精确锁零，但 ``max_velocity=0`` 会改变 active articulation dynamics。"""

    tree = ast.parse(EVENTS_MODULE_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lock_canonical_ghost_joint_limits"
    )
    called_methods = {
        node.func.attr
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert "write_joint_position_limit_to_sim" in called_methods
    assert "write_joint_velocity_limit_to_sim" not in called_methods
