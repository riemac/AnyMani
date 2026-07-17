r"""Pure tensor tests for GM proprioception / action observations.

这些测试不启动 Isaac Sim，只用 fake env 锁定 state/action obs 的科研语义：
`gm_mdp.joint_pos_limit_normalized` 与 `gm_mdp.last_action` 是 IsaacLab 官方子集
的等价 wrapper；`gm_mdp.last_processed_action` 则继续表达 AnyMani raw-rad
动作语义。三者必须同时存在，避免 LEAP 对照实验在改命名空间时悄悄改变量纲。
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

OBS_STATE_PATH = Path(__file__).resolve().parents[1] / "mdp" / "observations" / "observations_state.py"
r"""被测试的 state obs 源文件路径；用 path import 避免触发完整 Isaac runtime。"""


class _SceneEntityCfgStub:
    r"""最小 SceneEntityCfg stub，保留 observation 函数读取的 `name` 与 `joint_ids` 字段。"""

    def __init__(self, name: str, joint_ids: list[int] | None = None):
        r"""保存 scene asset 名称与 joint 子集索引。"""

        self.name = name  # scene 字典 key，例如 `"robot"`
        self.joint_ids = joint_ids  # joint 子集索引；真实 IsaacLab 会在 manager 初始化时解析


def _load_observations_state_module() -> types.ModuleType:
    r"""加载 `observations_state.py`，并 stub 掉 IsaacLab runtime 依赖。"""

    assets_stub = types.ModuleType("isaaclab.assets")
    assets_stub.Articulation = object

    managers_stub = types.ModuleType("isaaclab.managers")
    managers_stub.SceneEntityCfg = _SceneEntityCfgStub

    replacements = {
        "isaaclab": types.ModuleType("isaaclab"),
        "isaaclab.assets": assets_stub,
        "isaaclab.managers": managers_stub,
    }
    previous = {name: sys.modules.get(name) for name in replacements}  # 保存原模块，避免污染其他测试
    try:
        sys.modules.update(replacements)
        spec = importlib.util.spec_from_file_location("_gm_observations_state_for_test", OBS_STATE_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_joint_pos_limit_normalized_matches_isaaclab_scale_transform_semantics() -> None:
    r"""GM wrapper 应保持 IsaacLab 官方 soft-limit 归一化公式，不改成 raw rad。"""

    module = _load_observations_state_module()
    asset_cfg = _SceneEntityCfgStub("robot", joint_ids=[0, 2])  # 只观测第 0 / 2 个关节，测试 joint_ids 子集语义
    robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=torch.tensor([[0.5, 0.0, 1.5]], dtype=torch.float32),  # $q$，单位 rad，形状 [B,3]
            soft_joint_pos_limits=torch.tensor(
                [[[-1.0, 1.0], [0.0, 1.0], [1.0, 3.0]]], dtype=torch.float32
            ),  # $[q^{\min},q^{\max}]$，单位 rad，形状 [B,3,2]
        )
    )
    env = SimpleNamespace(scene={"robot": robot})  # 最小 ManagerBased env stub

    q_norm = module.joint_pos_limit_normalized(env, asset_cfg=asset_cfg)  # `[B,2]`，无量纲

    assert torch.allclose(q_norm, torch.tensor([[0.5, -0.5]], dtype=torch.float32))


def test_last_action_and_last_processed_action_remain_distinct_contracts() -> None:
    r"""raw action wrapper 与 processed action wrapper 必须保持两个不同量纲的实验语义。"""

    module = _load_observations_state_module()
    action_term = SimpleNamespace(
        raw_actions=torch.tensor([[1.0, -1.0]], dtype=torch.float32),  # policy raw output，无量纲
        processed_actions=torch.tensor([[0.1, -0.1]], dtype=torch.float32),  # scale 后 $\Delta q$，单位 rad
        current_targets=torch.tensor([[0.4, 0.8]], dtype=torch.float32),  # recurrent target $u_t$，单位 rad
    )
    action_manager = SimpleNamespace(
        action=torch.tensor([[1.0, -1.0, 0.5]], dtype=torch.float32),  # manager 拼接后的完整 raw action
        get_term=lambda name: action_term,
    )
    env = SimpleNamespace(action_manager=action_manager)  # 最小 ManagerBased env stub

    assert torch.allclose(module.last_action(env), torch.tensor([[1.0, -1.0, 0.5]], dtype=torch.float32))
    assert torch.allclose(module.last_action(env, action_name="hand_joint_pos"), action_term.raw_actions)
    assert torch.allclose(module.last_processed_action(env, action_name="hand_joint_pos"), action_term.processed_actions)
    assert torch.allclose(module.joint_target(env, action_name="hand_joint_pos"), action_term.current_targets)


def test_raw_joint_state_terms_share_resolved_joint_order() -> None:
    r"""$q$ 与 $\dot q$ 必须复用同一个 resolved joint subset/order。"""

    module = _load_observations_state_module()
    asset_cfg = _SceneEntityCfgStub("robot", joint_ids=[2, 0])
    robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=torch.tensor([[0.1, 0.2, 0.3]]),
            joint_vel=torch.tensor([[1.0, 2.0, 3.0]]),
        )
    )
    env = SimpleNamespace(scene={"robot": robot})

    torch.testing.assert_close(module.joint_pos_raw(env, asset_cfg), torch.tensor([[0.3, 0.1]]))
    torch.testing.assert_close(module.joint_vel_raw(env, asset_cfg), torch.tensor([[3.0, 1.0]]))
