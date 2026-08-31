r"""`TactileRotationCommand` 的纯 SO(3) 与 policy-step lifecycle contracts。

不实例化 Isaac Sim。测试直接构造 tensor-only command，验证有向增量、quaternion 符号不变、
同 step 幂等、partial reset，以及“reward 先看到 success pulse、command hook 后推进 goal”。
"""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch


class _PxrTypeStub(types.ModuleType):
    r"""允许 IsaacLab math package import 阶段访问 USD 类型名。"""

    def __getattr__(self, name: str) -> type:
        r"""为任意 USD symbol 返回 dummy Python type。"""

        return type(name, (), {})


def _install_import_stubs() -> dict[str, types.ModuleType | None]:
    r"""安装加载 command 文件所需的最小 pxr/omni/manager stubs。"""

    pxr_stub = types.ModuleType("pxr")
    pxr_stub.Usd = _PxrTypeStub("Usd")
    pxr_stub.UsdGeom = _PxrTypeStub("UsdGeom")
    omni_stub = types.ModuleType("omni")
    omni_stub.kit = types.ModuleType("omni.kit")
    omni_stub.kit.app = types.ModuleType("omni.kit.app")
    omni_stub.timeline = types.ModuleType("omni.timeline")
    managers_stub = types.ModuleType("isaaclab.managers")
    managers_stub.CommandTerm = type("CommandTerm", (), {})
    replacements = {
        "pxr": pxr_stub,
        "pxr.Usd": pxr_stub.Usd,
        "pxr.UsdGeom": pxr_stub.UsdGeom,
        "omni": omni_stub,
        "omni.kit": omni_stub.kit,
        "omni.kit.app": omni_stub.kit.app,
        "omni.timeline": omni_stub.timeline,
        "isaaclab.managers": managers_stub,
    }
    previous = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)
    return previous


def _restore_modules(previous: dict[str, types.ModuleType | None]) -> None:
    r"""恢复临时 import stubs。"""

    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _load_command_module():
    r"""从文件加载 tactile command，避免触发 `gm.mdp` package-wide imports。"""

    module_path = Path(__file__).resolve().parents[1] / "mdp" / "commands" / "tactile_rotation_command.py"
    spec = importlib.util.spec_from_file_location("gm_tactile_rotation_command_under_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load tactile rotation command from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_previous_modules = _install_import_stubs()
try:
    import isaaclab.utils.math as math_utils  # noqa: E402

    _command_module = _load_command_module()
finally:
    _restore_modules(_previous_modules)

TactileRotationCommand = _command_module.TactileRotationCommand
projected_space_rotation_delta = _command_module.projected_space_rotation_delta


def _identity_quat(batch: int) -> torch.Tensor:
    r"""构造 `(w,x,y,z)` identity quaternion batch。"""

    quat = torch.zeros(batch, 4)
    quat[:, 0] = 1.0
    return quat


def _z_quat(angles: torch.Tensor) -> torch.Tensor:
    r"""构造绕 world z 轴旋转的 quaternion batch。"""

    axis = torch.tensor([0.0, 0.0, 1.0]).repeat(angles.numel(), 1)
    return math_utils.quat_from_angle_axis(angles, axis)


def _fake_command(batch: int = 2):
    r"""构造绕过 IsaacLab constructor、但覆盖 progress/goal methods 所需字段的 command。"""

    command = TactileRotationCommand.__new__(TactileRotationCommand)
    command.num_envs = batch
    command.device = "cpu"
    command.cfg = SimpleNamespace(
        subgoal_angle=math.pi / 6,
        make_quat_unique=False,
        keypoint_radius=0.05,
        orientation_keypoint_success_threshold=0.005,
        position_success_threshold=0.025,
        speed_ema_time_constant_s=0.25,
        resampling_time_range=(1.0e6, 1.0e6),
    )
    command._env = SimpleNamespace(common_step_counter=0, step_dt=0.05)
    command.semantic_R_ha = torch.eye(3)
    command.robot = SimpleNamespace(data=SimpleNamespace(root_quat_w=_identity_quat(batch)))
    command.object = SimpleNamespace(
        data=SimpleNamespace(root_quat_w=_identity_quat(batch), root_pos_w=torch.zeros(batch, 3))
    )
    command.axis_h = torch.tensor([0.0, 0.0, 1.0]).repeat(batch, 1)
    command.axis_w = command.axis_h.clone()
    command.goal_quat_w = _identity_quat(batch)
    command.error_so3_h = torch.zeros(batch, 3)
    command.position_anchor_w = torch.zeros(batch, 3)
    command.previous_quat_w = _identity_quat(batch)
    command.has_previous = torch.ones(batch, dtype=torch.bool)
    command.last_progress_step = torch.zeros(batch, dtype=torch.long)  # reset stamp 0 已处理
    command.delta_psi = torch.zeros(batch)
    command.net_rotation_rad = torch.zeros(batch)
    command.net_rotation_turns = torch.zeros(batch)
    command.axis_speed = torch.zeros(batch)
    command.axis_speed_ema = torch.zeros(batch)
    command.orientation_keypoint_error = torch.zeros(batch)
    command.position_error = torch.zeros(batch)
    command.goal_normal_alignment = torch.ones(batch)
    command.goal_success_pulse = torch.zeros(batch, dtype=torch.bool)
    command.goal_success_count = torch.zeros(batch)
    command.command_counter = torch.zeros(batch, dtype=torch.long)
    command.time_left = torch.zeros(batch)
    command._resample_command(torch.arange(batch))
    return command


def test_projected_delta_is_signed_and_quaternion_sign_invariant() -> None:
    r"""正/反转必须分别给正/负增量，且 $q\sim-q$ 不得改变结果。"""

    previous = _identity_quat(2)
    current = _z_quat(torch.tensor([0.2, -0.2]))
    axis = torch.tensor([0.0, 0.0, 1.0]).repeat(2, 1)

    delta = projected_space_rotation_delta(previous, current, axis)
    sign_flipped_delta = projected_space_rotation_delta(-previous, -current, axis)

    assert torch.allclose(delta, torch.tensor([0.2, -0.2]), atol=1.0e-5)
    assert torch.allclose(sign_flipped_delta, delta, atol=1.0e-6)


def test_progress_refresh_is_immediate_and_idempotent_per_step() -> None:
    r"""physics 后首个 consumer 当步看到 delta；同 stamp 第二次读取不重复累计。"""

    command = _fake_command()
    command._env.common_step_counter = 1
    command.object.data.root_quat_w = _z_quat(torch.tensor([0.1, -0.1]))

    command.ensure_post_physics_progress_updated()
    first_net = command.net_rotation_rad.clone()
    command.ensure_post_physics_progress_updated()  # 同一个 step stamp 必须 no-op

    assert torch.allclose(command.delta_psi, torch.tensor([0.1, -0.1]), atol=1.0e-5)
    assert torch.allclose(first_net, torch.tensor([0.1, -0.1]), atol=1.0e-5)
    assert torch.allclose(command.net_rotation_rad, first_net)
    assert torch.allclose(command.net_rotation_turns, torch.tensor([0.1 / (2 * math.pi), 0.0]), atol=1.0e-6)
    assert torch.allclose(command.axis_speed, torch.tensor([2.0, -2.0]), atol=1.0e-4)


def test_morphology_cell_extras_preserve_reset_subset_group_means() -> None:
    r"""Command super-reset标量化前必须按cell保留per-env episode evidence。"""

    command = _fake_command(batch=4)
    command._env._anymani_morphology_cell_id = torch.tensor([0, 0, 1, 1])
    command.metrics = {
        "goal_success_count": torch.tensor([1.0, 3.0, 2.0, 6.0]),
        "net_rotation_turns": torch.tensor([0.1, 0.3, 0.2, 0.6]),
        "position_error": torch.zeros(4),
        "contact/tip_active_count_mean": torch.ones(4),
        "contact/finger_non_tip_occupancy_fraction": torch.zeros(4),
        "termination/object_out_of_anchor_fraction": torch.zeros(4),
    }

    extras = command._morphology_cell_extras(torch.arange(4))

    assert extras["cell/left_tips3_thumb3dof/episode_count"] == 2.0
    assert extras["cell/left_tips3_thumb3dof/goal_success_count"] == 2.0
    assert extras["cell/left_tips3_thumb4dof/goal_success_count"] == 4.0
    assert "cell/left_tips4_thumb3dof/goal_success_count" not in extras  # 无样本cell不伪造零值


def test_partial_reset_preserves_other_env_and_blocks_same_stamp_delta() -> None:
    r"""partial reset 只清目标 env，并使 reset observation 不累计同 stamp 伪旋转。"""

    command = _fake_command()
    command._env.common_step_counter = 1
    command.object.data.root_quat_w = _z_quat(torch.tensor([0.2, 0.2]))
    command.ensure_post_physics_progress_updated()

    command._capture_reset_state(torch.tensor([0]))
    assert torch.allclose(command.net_rotation_rad, torch.tensor([0.0, 0.2]), atol=1.0e-5)

    command.object.data.root_quat_w[0] = _z_quat(torch.tensor([0.4]))[0]
    command.ensure_post_physics_progress_updated()  # stamp 1：env 0 reset 已盖戳，env 1 也已更新
    assert torch.allclose(command.net_rotation_rad, torch.tensor([0.0, 0.2]), atol=1.0e-5)


def test_success_pulse_precedes_goal_advance() -> None:
    r"""Reward 可先消费成功 pulse；随后 command hook 才生成相对当前 pose 的下一目标。"""

    command = _fake_command()
    old_goal = command.goal_quat_w.clone()
    command.object.data.root_quat_w = old_goal.clone()  # 精确到达第一个 30 degree subgoal
    command._env.common_step_counter = 1

    command.ensure_post_physics_progress_updated()
    assert torch.all(command.goal_success_pulse)  # reward 阶段应观察到 1-step impulse
    assert torch.allclose(command.goal_quat_w, old_goal)  # ensure 只刷新状态，不提前换 goal

    command._update_command()
    assert torch.allclose(command.goal_success_count, torch.ones(2))
    assert not torch.any(command.goal_success_pulse)
    assert not torch.allclose(command.goal_quat_w, old_goal)  # command hook 从已达到的 current pose 再前推 30 degree
    expected_next_goal = _z_quat(torch.full((2,), math.pi / 3))  # identity -> 30° -> 60°
    assert torch.allclose(command.goal_quat_w, expected_next_goal, atol=1.0e-5)
