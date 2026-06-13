r"""Pure tensor contract tests for `ReorientCommand`.

这些测试不实例化 IsaacLab `ManagerBasedRLEnv`，也不启动 Isaac Sim。它们通过
`ReorientCommand.__new__` 构造一个带最小 buffer 的对象，直接测试 command term
最核心的 SO(3) 数学语义：

$$
R_g = \exp([\hat\omega]\theta)R_o,
\qquad
\phi_e = \log(R_gR_o^{-1}).
$$

这样可以在真正跑物理 smoke 前，先防止 axis/frame/error buffer 的符号或形状写错。
"""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


class _PxrTypeStub(types.ModuleType):
    r"""允许 `Usd.Prim` 这类类型注解在纯 pytest 进程中解析。"""

    def __getattr__(self, name: str) -> type:
        r"""返回一个 dummy 类型，避免 import 阶段访问真实 USD C++ binding。"""

        return type(name, (), {})


pxr_stub = types.ModuleType("pxr")
pxr_stub.Usd = _PxrTypeStub("Usd")
pxr_stub.UsdGeom = _PxrTypeStub("UsdGeom")
sys.modules.setdefault("pxr", pxr_stub)
sys.modules.setdefault("pxr.Usd", pxr_stub.Usd)
sys.modules.setdefault("pxr.UsdGeom", pxr_stub.UsdGeom)

omni_stub = types.ModuleType("omni")
omni_stub.kit = types.ModuleType("omni.kit")
omni_stub.kit.app = types.ModuleType("omni.kit.app")
omni_stub.timeline = types.ModuleType("omni.timeline")
sys.modules.setdefault("omni", omni_stub)
sys.modules.setdefault("omni.kit", omni_stub.kit)
sys.modules.setdefault("omni.kit.app", omni_stub.kit.app)
sys.modules.setdefault("omni.timeline", omni_stub.timeline)

managers_stub = types.ModuleType("isaaclab.managers")
managers_stub.CommandTerm = type("CommandTerm", (), {})
sys.modules.setdefault("isaaclab.managers", managers_stub)

import isaaclab.utils.math as math_utils  # noqa: E402


def _load_reorient_command_class() -> type:
    r"""直接从文件加载 `ReorientCommand`，避免触发 `anymani.tasks` 自动注册。"""

    module_path = (
        Path(__file__).resolve().parents[2]
        / "source"
        / "anymani"
        / "anymani"
        / "tasks"
        / "gm"
        / "mdp"
        / "commands"
        / "reorient_command.py"
    )
    spec = importlib.util.spec_from_file_location("gm_reorient_command_under_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load ReorientCommand module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ReorientCommand


ReorientCommand = _load_reorient_command_class()


def _identity_quat(batch: int) -> torch.Tensor:
    r"""构造 IsaacLab `(w,x,y,z)` 单位四元数 batch。"""

    quat = torch.zeros(batch, 4)  # `[B,4]`，四元数 buffer
    quat[:, 0] = 1.0  # `w=1` 表示单位旋转
    return quat


def _make_fake_command(num_envs: int = 2, theta: float = math.pi / 2) -> ReorientCommand:
    r"""构造一个只含张量 buffer 的 `ReorientCommand` 测试对象。

    Args:
        num_envs (int): fake vectorized env 数。
        theta (float): 固定 subgoal 旋转角，单位 rad。

    Returns:
        ReorientCommand: 未调用 IsaacLab constructor、但拥有测试所需字段的对象。
    """

    cmd = ReorientCommand.__new__(ReorientCommand)  # 绕开 CommandTerm.__init__，只测纯张量逻辑
    cmd.num_envs = num_envs  # `[B]` buffer 的 batch 维度
    cmd.device = "cpu"  # 纯 CPU 测试，不依赖 CUDA / Isaac Sim
    cmd.cfg = SimpleNamespace(
        axis_mode="fixed",  # 固定 hand z 轴，便于解析期望值
        axis_resample_mode="subgoal",  # 每个成功 subgoal 后允许重新采样目标
        fixed_axis_h=(0.0, 0.0, 1.0),  # $\hat\omega^h=e_z$
        theta_range=(theta, theta),  # 固定角度，使测试确定性
        make_quat_unique=False,  # 不额外折叠 quaternion 符号，便于直接比较
        orientation_success_threshold=1.0e-3,  # 到达目标时触发 success
        keypoint_radius=0.05,  # AnyRotate 六轴 keypoint 半径锚点
        resampling_time_range=(1.0e6, 1.0e6),  # 与真实 cfg 一致，禁用时间驱动重采样
    )
    cmd.semantic_R_ha = torch.eye(3)  # generated asset 第一版默认 `{a}` 与 `{h}` 对齐
    cmd.robot = SimpleNamespace(data=SimpleNamespace(root_quat_w=_identity_quat(num_envs)))  # hand root 姿态
    cmd.object = SimpleNamespace(data=SimpleNamespace(root_quat_w=_identity_quat(num_envs)))  # object 当前姿态

    cmd.axis_h = torch.zeros(num_envs, 3)  # `[B,3]`，hand frame axis
    cmd.axis_e = torch.zeros(num_envs, 3)  # `[B,3]`，env/world frame axis
    cmd.theta = torch.zeros(num_envs)  # `[B]`，subgoal 角度 rad
    cmd.goal_quat_w = _identity_quat(num_envs)  # `[B,4]`，目标姿态
    cmd.error_so3_e = torch.zeros(num_envs, 3)  # `[B,3]`，world rotvec error
    cmd.error_so3_h = torch.zeros(num_envs, 3)  # `[B,3]`，hand rotvec error
    cmd.goal_success_count = torch.zeros(num_envs)  # `[B]`，episode success count
    cmd.axis_progress = torch.zeros(num_envs)  # `[B]`，episode target progress rad
    cmd.command_counter = torch.zeros(num_envs, dtype=torch.long)  # IsaacLab command counter buffer
    cmd.time_left = torch.zeros(num_envs)  # IsaacLab resampling time buffer
    cmd.metrics = {
        "orientation_error": torch.zeros(num_envs),
        "keypoint_error": torch.zeros(num_envs),
        "goal_success_count": cmd.goal_success_count,
        "axis_progress": cmd.axis_progress,
    }
    return cmd


def test_reorient_command_resample_populates_axis_goal_and_error() -> None:
    r"""采样 subgoal 后，axis/goal/error buffer 应符合 $R_g=\exp([z]\theta)R_o$。"""

    cmd = _make_fake_command(theta=math.pi / 2)  # 90 degree around hand z axis
    env_ids = torch.tensor([0, 1], dtype=torch.long)  # 同时采样两个 fake env
    cmd._resample_command(env_ids)

    expected_axis = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])  # $e_z^h=e_z^e$
    expected_quat = math_utils.quat_from_angle_axis(torch.full((2,), math.pi / 2), expected_axis)
    expected_error = torch.tensor([[0.0, 0.0, math.pi / 2], [0.0, 0.0, math.pi / 2]])

    assert torch.allclose(cmd.axis_h, expected_axis, atol=1.0e-6)  # policy-facing axis
    assert torch.allclose(cmd.axis_e, expected_axis, atol=1.0e-6)  # reward-facing axis
    assert torch.allclose(cmd.goal_quat_w, expected_quat, atol=1.0e-6)  # 目标姿态 quaternion
    assert torch.allclose(cmd.error_so3_h, expected_error, atol=1.0e-5)  # command 后三维 rotvec
    assert cmd.command.shape == (2, 6)  # `[axis_h,error_so3_h]`


def test_reorient_command_success_updates_progress_and_next_goal() -> None:
    r"""到达目标姿态后，应增加 success/progress 并生成下一 subgoal。"""

    cmd = _make_fake_command(theta=math.pi / 4)  # 45 degree subgoal，便于区分 progress
    cmd._resample_command(torch.tensor([0, 1], dtype=torch.long))
    cmd.object.data.root_quat_w = cmd.goal_quat_w.clone()  # 模拟 object 已达到当前目标姿态

    cmd._update_metrics()
    assert torch.all(cmd.metrics["orientation_error"] < 1.0e-5)  # 到达目标时 SO(3) error 应近 0

    old_goal = cmd.goal_quat_w.clone()
    cmd._update_command()

    assert torch.allclose(cmd.goal_success_count, torch.ones(2))  # 每个 env 成功一次
    assert torch.allclose(cmd.axis_progress, torch.full((2,), math.pi / 4))  # 进度累计当前 subgoal 角度
    assert not torch.allclose(cmd.goal_quat_w, old_goal)  # success 后生成下一目标，而非停在旧目标
    assert torch.allclose(torch.linalg.norm(cmd.error_so3_h, dim=-1), torch.full((2,), math.pi / 4), atol=1.0e-5)


def test_reorient_command_rejects_zero_fixed_axis() -> None:
    r"""固定轴为零向量时必须显式失败，避免 command 变成无方向目标。"""

    cmd = _make_fake_command(theta=math.pi / 4)
    cmd.cfg.fixed_axis_h = (0.0, 0.0, 0.0)  # 非法轴，无法归一化
    with pytest.raises(ValueError, match="non-zero"):
        cmd._resample_command(torch.tensor([0], dtype=torch.long))
