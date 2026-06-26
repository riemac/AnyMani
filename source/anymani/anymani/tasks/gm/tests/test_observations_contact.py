r"""Pure tensor tests for GM contact observations.

这些测试不启动 Isaac Sim，只用 fake env 和 fake contact sensor 验证 contact force
obs 的坐标系语义：对 policy/critic 暴露的是 hand semantic frame `{h}` 下的力，
而不是 world frame `{w}` 或 fingertip local frame `{S_k}`。
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

OBS_CONTACT_PATH = Path(__file__).resolve().parents[1] / "mdp" / "observations" / "observations_contact.py"
r"""被测试的 contact obs 源文件路径；用 path import 避免触发完整 Isaac runtime。"""


class _SceneEntityCfgStub:
    r"""最小 SceneEntityCfg stub，只保留 observation 函数读取的 `name` 字段。"""

    def __init__(self, name: str):
        r"""保存 scene asset 名称。"""

        self.name = name  # scene 字典 key，例如 `"robot"`


def _quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    r"""测试用 quaternion inverse rotation，匹配 IsaacLab `(w,x,y,z)` 约定。"""

    xyz = quat[:, 1:]  # `[B,3]`，四元数虚部
    t = xyz.cross(vec, dim=-1) * 2.0  # IsaacLab `quat_apply_inverse` 的中间项
    return vec - quat[:, 0:1] * t + xyz.cross(t, dim=-1)


def _load_observations_contact_module(force_by_sensor: dict[str, torch.Tensor]) -> types.ModuleType:
    r"""加载 `observations_contact.py`，并 stub 掉 IsaacLab 与 contact sensor 依赖。"""

    math_stub = types.ModuleType("isaaclab.utils.math")
    math_stub.quat_apply_inverse = _quat_apply_inverse

    assets_stub = types.ModuleType("isaaclab.assets")
    assets_stub.Articulation = object

    managers_stub = types.ModuleType("isaaclab.managers")
    managers_stub.SceneEntityCfg = _SceneEntityCfgStub

    contact_sensors_stub = types.ModuleType("anymani.tasks.gm.contact_sensors")
    contact_sensors_stub.sensor_total_force_w = lambda _env, sensor_name: force_by_sensor[sensor_name]

    replacements = {
        "isaaclab": types.ModuleType("isaaclab"),
        "isaaclab.assets": assets_stub,
        "isaaclab.managers": managers_stub,
        "isaaclab.utils": types.ModuleType("isaaclab.utils"),
        "isaaclab.utils.math": math_stub,
        "anymani.tasks.gm.contact_sensors": contact_sensors_stub,
    }
    previous = {name: sys.modules.get(name) for name in replacements}  # 保存原模块，避免污染其他测试
    try:
        sys.modules.update(replacements)
        spec = importlib.util.spec_from_file_location(
            "anymani.tasks.gm.mdp.observations._contact_for_test", OBS_CONTACT_PATH
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        module.__package__ = "anymani.tasks.gm.mdp.observations"
        spec.loader.exec_module(module)
        return module
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _fake_env_with_robot_quat(root_quat_w: torch.Tensor) -> SimpleNamespace:
    r"""构造只包含 robot root orientation 的 fake env。"""

    robot = SimpleNamespace(data=SimpleNamespace(root_quat_w=root_quat_w))  # hand root orientation $R_{wa}$
    return SimpleNamespace(device="cpu", scene={"robot": robot})


def test_fingertip_contact_force_identity_hand_matches_world_force() -> None:
    r"""identity hand frame 下，hand-frame contact force 应等于 world-frame force。"""

    module = _load_observations_contact_module({"tip": torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)})
    env = _fake_env_with_robot_quat(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32))

    force_h = module.fingertip_contact_force(env, ("tip",))  # `[1,3]`，默认 `frame="h"`，单位 N

    assert torch.allclose(force_h, torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32))


def test_fingertip_contact_force_rotates_world_force_into_hand_frame() -> None:
    r"""hand 绕 world z 轴 +90 度时，world x 方向力在 `{h}` 下应为负 y。"""

    half_sqrt2 = 0.70710678118  # $\sqrt{2}/2$，+90deg yaw quaternion 的 w/z 分量
    root_quat_w = torch.tensor([[half_sqrt2, 0.0, 0.0, half_sqrt2]], dtype=torch.float32)  # $R_{wa}=R_z(90^\circ)$
    module = _load_observations_contact_module({"tip": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)})
    env = _fake_env_with_robot_quat(root_quat_w)

    force_h = module.fingertip_contact_force(env, ("tip",))  # $F^h=R_{ha}R_{aw}F^w$

    assert torch.allclose(force_h, torch.tensor([[0.0, -1.0, 0.0]], dtype=torch.float32), atol=1.0e-6)


def test_fingertip_contact_force_e_keeps_env_frame_force() -> None:
    r"""`frame="e"` 应保留 env/world 轴向，服务 hand-frame force 的对照消融。"""

    half_sqrt2 = 0.70710678118  # $\sqrt{2}/2$，+90deg yaw quaternion 的 w/z 分量
    root_quat_w = torch.tensor([[half_sqrt2, 0.0, 0.0, half_sqrt2]], dtype=torch.float32)  # $R_{wa}=R_z(90^\circ)$
    module = _load_observations_contact_module({"tip": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)})
    env = _fake_env_with_robot_quat(root_quat_w)

    force_e = module.fingertip_contact_force(env, ("tip",), frame="e")  # `[B,3]`，不做 $R_{aw}$ 旋转

    assert torch.allclose(force_e, torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32), atol=1.0e-6)
