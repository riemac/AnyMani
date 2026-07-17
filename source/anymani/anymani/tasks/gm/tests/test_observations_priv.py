r"""Pure tensor tests for GM privileged object observations.

这些测试不启动 Isaac Sim，只构造最小 fake env，核对 object pose obs 的坐标系语义：
teacher policy 读取的不是 world-frame 绝对 pose，而是 hand semantic frame `{h}` 下的
相对 object pose。
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

OBS_PRIV_PATH = Path(__file__).resolve().parents[1] / "mdp" / "observations" / "observations_priv.py"
r"""被测试的 privileged obs 源文件路径；用 path import 避免触发完整 Isaac runtime。"""


class _SceneEntityCfgStub:
    r"""最小 SceneEntityCfg stub，只保留 observation 函数读取的 `name` 字段。"""

    def __init__(self, name: str):
        r"""保存 scene asset 名称。"""

        self.name = name  # scene 字典 key，例如 `"object"` / `"robot"`


class _FakeScene(dict):
    r"""带 `env_origins` 属性的最小 scene stub。

    `ManagerBasedRLEnv.scene` 运行时既支持 `scene["robot"]` 取 asset，也暴露
    `scene.env_origins`。通用 `object_pos` 在 `reference="env"` 时会读取该字段，
    因此测试 stub 也要保留这个 contract。
    """

    env_origins: torch.Tensor

    def __init__(self, *args, env_origins: torch.Tensor):
        r"""保存 asset 字典与 env origin buffer。"""

        super().__init__(*args)  # 继承 dict 的 `scene[name]` 行为
        self.env_origins = env_origins  # `[B,3]`，env-local origin 在 world 中的位置，单位 m


def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    r"""测试用 quaternion forward rotation，匹配 IsaacLab `(w,x,y,z)` 约定。"""

    xyz = quat[:, 1:]  # `[B,3]`，四元数虚部
    t = xyz.cross(vec, dim=-1) * 2.0  # IsaacLab `quat_apply` 的中间项
    return vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)


def _quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    r"""测试用 quaternion inverse rotation，匹配 IsaacLab `(w,x,y,z)` 约定。"""

    xyz = quat[:, 1:]  # `[B,3]`，四元数虚部
    t = xyz.cross(vec, dim=-1) * 2.0  # IsaacLab `quat_apply_inverse` 的中间项
    return vec - quat[:, 0:1] * t + xyz.cross(t, dim=-1)


def _matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    r"""测试用 quaternion-to-matrix，实现 observation 纯函数需要的最小数学依赖。"""

    w, x, y, z = torch.unbind(quaternions, dim=-1)  # IsaacLab quaternion order `(w,x,y,z)`
    two_s = 2.0 / (quaternions * quaternions).sum(dim=-1)  # 单位 quat 时等于 2
    matrix = torch.stack(
        (
            1 - two_s * (y * y + z * z),
            two_s * (x * y - z * w),
            two_s * (x * z + y * w),
            two_s * (x * y + z * w),
            1 - two_s * (x * x + z * z),
            two_s * (y * z - x * w),
            two_s * (x * z - y * w),
            two_s * (y * z + x * w),
            1 - two_s * (x * x + y * y),
        ),
        dim=-1,
    )
    return matrix.reshape(quaternions.shape[:-1] + (3, 3))  # `[B,3,3]`


def _load_observations_priv_module() -> types.ModuleType:
    r"""加载 `observations_priv.py`，并用 stub 避免依赖真实 Isaac / USD binding。"""

    math_stub = types.ModuleType("isaaclab.utils.math")
    math_stub.quat_apply = _quat_apply
    math_stub.quat_apply_inverse = _quat_apply_inverse
    math_stub.matrix_from_quat = _matrix_from_quat

    assets_stub = types.ModuleType("isaaclab.assets")
    assets_stub.Articulation = object
    assets_stub.RigidObject = object

    managers_stub = types.ModuleType("isaaclab.managers")
    managers_stub.SceneEntityCfg = _SceneEntityCfgStub

    package_names = (
        "anymani",
        "anymani.tasks",
        "anymani.tasks.gm",
        "anymani.tasks.gm.mdp",
        "anymani.tasks.gm.mdp.observations",
        "anymani.tasks.gm.mdp.commands",
    )
    package_stubs = {}
    for package_name in package_names:
        package = types.ModuleType(package_name)
        package.__path__ = []  # type: ignore[attr-defined]
        package_stubs[package_name] = package
    adr_stub = types.ModuleType("anymani.tasks.gm.mdp.adr_state")
    adr_stub.get_gm_adr_state = lambda *_args, **_kwargs: None
    command_stub = types.ModuleType("anymani.tasks.gm.mdp.commands.tactile_rotation_command")
    command_stub.ensure_post_physics_progress_updated = lambda *_args, **_kwargs: None
    replacements = {
        **package_stubs,
        "isaaclab": types.ModuleType("isaaclab"),
        "isaaclab.assets": assets_stub,
        "isaaclab.managers": managers_stub,
        "isaaclab.utils": types.ModuleType("isaaclab.utils"),
        "isaaclab.utils.math": math_stub,
        "anymani.tasks.gm.mdp.adr_state": adr_stub,
        "anymani.tasks.gm.mdp.commands.tactile_rotation_command": command_stub,
    }
    previous = {name: sys.modules.get(name) for name in replacements}  # 保存原模块，避免污染其他测试
    try:
        sys.modules.update(replacements)
        spec = importlib.util.spec_from_file_location(
            "anymani.tasks.gm.mdp.observations._priv_for_test", OBS_PRIV_PATH
        )
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


def _identity_quat(count: int) -> torch.Tensor:
    r"""构造 IsaacLab `(w,x,y,z)` 单位四元数。

    Args:
        count (int): batch/env 数。

    Returns:
        torch.Tensor: 单位四元数，形状 `[count, 4]`。
    """

    quat = torch.zeros(count, 4, dtype=torch.float32)  # `[B,4]`，IsaacLab quaternion order
    quat[:, 0] = 1.0  # 实部 $w=1$，表示 identity rotation
    return quat


def _fake_env_for_object_pose() -> SimpleNamespace:
    r"""构造只包含 object / robot root pose 的 fake ManagerBased env。

    Returns:
        SimpleNamespace: 拥有 `device` 与 `scene` 字段的最小 fake env。
    """

    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=torch.tensor([[0.0, 0.0, 0.5]], dtype=torch.float32),  # hand anchor $p_a^w$
            root_quat_w=_identity_quat(1),  # $R_{wa}=I$
        )
    )
    obj = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=torch.tensor([[0.02, 0.08, 0.56]], dtype=torch.float32),  # calibrated object $p_o^w$
            root_quat_w=_identity_quat(1),  # $R_{wo}=I$
        )
    )
    scene = _FakeScene({"robot": robot, "object": obj}, env_origins=torch.zeros(1, 3, dtype=torch.float32))
    return SimpleNamespace(device="cpu", num_envs=1, scene=scene)


def test_object_pos_matches_calibrated_contact_basin_under_identity_hand() -> None:
    r"""identity hand pose 下，world pose 应还原为标定台导出的 $p_o^h$。"""

    env = _fake_env_for_object_pose()
    module = _load_observations_priv_module()

    pos_h = module.object_pos(env)  # `[1,3]`，单位 m，默认 `frame="h", reference="hand"`

    assert torch.allclose(pos_h, torch.tensor([[0.02, 0.08, 0.06]], dtype=torch.float32))


def test_object_pos_adds_nonzero_semantic_hand_translation() -> None:
    r"""非零 $p_{ha}$ 必须平移 hand-frame object position，不能再隐式假设 `{a}` 原点等于 `{h}` 原点。"""

    env = _fake_env_for_object_pose()
    module = _load_observations_priv_module()

    pos_h = module.object_pos(env, semantic_p_ha=(1.0, 2.0, 3.0))  # $p_o^h=p_{ha}+R_{ha}p_o^a$

    assert torch.allclose(pos_h, torch.tensor([[1.02, 2.08, 3.06]], dtype=torch.float32))


def test_object_orientation_rot6d_uses_first_two_rotation_columns() -> None:
    r"""identity object orientation 下，6D 表示应为 $[e_x,e_y]$ 的列向量拼接。"""

    env = _fake_env_for_object_pose()
    module = _load_observations_priv_module()

    rot6d_h = module.object_orientation(env)  # `[1,6]`，默认 `frame="h", representation="rot6d"`

    expected = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
    assert torch.allclose(rot6d_h, expected)
