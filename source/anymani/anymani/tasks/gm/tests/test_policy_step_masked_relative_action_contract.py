r"""policy-step masked relative action 的纯张量与配置合同。"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import torch


def _load_action_module():
    r"""用极小 IsaacLab stub 加载目标 action 文件，不启动 Kit。"""

    module_path = Path(__file__).resolve().parents[1] / "mdp" / "actions" / "policy_step_masked_relative_action.py"
    spec = importlib.util.spec_from_file_location("gm_policy_step_masked_action_under_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load action module from {module_path}")

    class ActionTerm:
        r"""``class_type`` 类型占位。"""

    class RelativeJointPositionAction:
        r"""父类占位；纯 helper/config 测试不执行 runtime action methods。"""

        def __init__(self, cfg, env):
            self.cfg = cfg
            self.env = env

    @dataclass
    class RelativeJointPositionActionCfg:
        r"""真实 cfg 所需字段的最小 dataclass。"""

        asset_name: str = ""
        joint_names: list[str] | None = None
        class_type: type | None = None
        scale: float = 1.0
        clip: dict[str, tuple[float, float]] | None = None
        preserve_order: bool = False
        use_zero_offset: bool = False

    def configclass(cls):
        return dataclass(cls)

    module_names = {
        "isaaclab": types.ModuleType("isaaclab"),
        "isaaclab.envs": types.ModuleType("isaaclab.envs"),
        "isaaclab.envs.mdp": types.ModuleType("isaaclab.envs.mdp"),
        "isaaclab.envs.mdp.actions": types.ModuleType("isaaclab.envs.mdp.actions"),
        "isaaclab.envs.mdp.actions.actions_cfg": types.ModuleType("isaaclab.envs.mdp.actions.actions_cfg"),
        "isaaclab.envs.mdp.actions.joint_actions": types.ModuleType("isaaclab.envs.mdp.actions.joint_actions"),
        "isaaclab.managers": types.ModuleType("isaaclab.managers"),
        "isaaclab.managers.action_manager": types.ModuleType("isaaclab.managers.action_manager"),
        "isaaclab.utils": types.ModuleType("isaaclab.utils"),
    }
    for name, stub in module_names.items():
        if name.rsplit(".", 1)[-1] not in {"actions_cfg", "joint_actions", "action_manager", "utils"}:
            stub.__path__ = []
    module_names["isaaclab.envs.mdp.actions.actions_cfg"].RelativeJointPositionActionCfg = (
        RelativeJointPositionActionCfg
    )
    module_names["isaaclab.envs.mdp.actions.joint_actions"].RelativeJointPositionAction = RelativeJointPositionAction
    module_names["isaaclab.managers.action_manager"].ActionTerm = ActionTerm
    module_names["isaaclab.utils"].configclass = configclass
    previous = {name: sys.modules.get(name) for name in module_names}
    sys.modules.update(module_names)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
    return module


def test_target_transition_masks_ghost_and_clamps_active_joint() -> None:
    r"""target accumulator 只推进 active joint，并投影到 soft limits。"""

    module = _load_action_module()
    previous = torch.tensor([[0.2, 9.0, -0.4]])
    delta = torch.tensor([[0.1, 100.0, -0.2]])
    lower = torch.tensor([[-1.0, -1.0, -0.5]])
    upper = torch.tensor([[1.0, 1.0, 0.5]])
    mask = torch.tensor([[True, False, True]])

    target = module.compute_policy_step_masked_relative_target(previous, delta, lower, upper, mask)

    assert torch.allclose(target, torch.tensor([[0.3, 0.0, -0.5]]))


def test_target_transition_rejects_shape_or_mask_dtype_mismatch() -> None:
    r"""mask/limit 轴不一致时应在进入 PhysX 前 fail closed。"""

    module = _load_action_module()
    values = torch.zeros(2, 3)
    try:
        module.compute_policy_step_masked_relative_target(values, values[:, :2], values, values, values.bool())
    except ValueError:
        pass
    else:
        raise AssertionError("shape mismatch must be rejected")

    try:
        module.compute_policy_step_masked_relative_target(values, values, values, values, values)
    except TypeError:
        pass
    else:
        raise AssertionError("non-bool mask must be rejected")


def test_cfg_binds_policy_step_action_and_infra_defaults() -> None:
    r"""实例 cfg 必须保留真实 class_type、0.1-rad scale 与 canonical order。"""

    module = _load_action_module()
    cfg = module.PolicyStepMaskedRelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"])

    assert cfg.class_type is module.PolicyStepMaskedRelativeJointPositionAction
    assert cfg.scale == 0.1
    assert cfg.preserve_order is True
    assert cfg.use_zero_offset is True


def test_reset_joint_selection_uses_outer_env_joint_axes() -> None:
    r"""tensor env IDs 与 16 joint IDs 必须产生 `[K,16]`，不能触发 pairwise broadcast。"""

    module = _load_action_module()
    values = torch.arange(4 * 20, dtype=torch.float32).reshape(4, 20)
    env_ids = torch.tensor([1, 3])
    joint_ids = torch.arange(2, 18)

    selected = module._select_joint_rows(values, env_ids, joint_ids)

    assert selected.shape == (2, 16)
    torch.testing.assert_close(selected, values[env_ids][:, joint_ids])


def test_action_class_exposes_n000_executed_action_contract() -> None:
    r"""N000 diagnostics 需要实际无量纲 action property；infra route 不得缺失该接口。"""

    module = _load_action_module()

    assert isinstance(module.PolicyStepMaskedRelativeJointPositionAction.executed_actions, property)
