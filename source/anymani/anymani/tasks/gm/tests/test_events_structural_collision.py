r"""Pure contract tests for generated structural collision filtering.

这些测试不启动 Isaac Sim，也不 author USD stage。它们只锁住 AnyMani generated hand
结构碰撞过滤的 pair-level 科研语义：finger-palm 与 same-finger 内部碰撞被过滤，
不同 fingers 之间的碰撞保留。
"""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

EVENTS_PATH = Path(__file__).resolve().parents[1] / "mdp" / "events.py"
r"""被测试的 events 源文件路径；用 path import 避免触发 `anymani.tasks.gm.mdp` 全量导入。"""


def _load_events_module() -> types.ModuleType:
    r"""加载 `events.py`，并 stub 掉本测试不需要的 IsaacLab runtime 类型。"""

    isaaclab = types.ModuleType("isaaclab")
    assets = types.ModuleType("isaaclab.assets")
    envs = types.ModuleType("isaaclab.envs")
    env_mdp = types.ModuleType("isaaclab.envs.mdp")
    managers = types.ModuleType("isaaclab.managers")
    utils = types.ModuleType("isaaclab.utils")
    math_utils = types.ModuleType("isaaclab.utils.math")
    adr_state = types.ModuleType("anymani.tasks.gm.mdp.adr_state")
    tactile_state = types.ModuleType("anymani.tasks.gm.mdp.tactile_contact_state")
    envs.__path__ = []  # type: ignore[attr-defined]  # 允许解析 `isaaclab.envs.mdp`
    utils.__path__ = []  # type: ignore[attr-defined]
    dummy_randomizer = type("DummyRandomizer", (), {})
    env_mdp.randomize_rigid_body_mass = dummy_randomizer
    env_mdp.randomize_actuator_gains = dummy_randomizer
    env_mdp.randomize_rigid_body_material = dummy_randomizer
    env_mdp.randomize_rigid_body_scale = lambda *_args, **_kwargs: None
    assets.Articulation = object
    assets.RigidObject = object
    envs.ManagerBasedRLEnv = object
    managers.SceneEntityCfg = lambda name, **kwargs: types.SimpleNamespace(name=name, **kwargs)
    math_utils.quat_from_angle_axis = lambda angle, axis: torch.cat(
        (torch.cos(angle / 2.0).unsqueeze(-1), axis * torch.sin(angle / 2.0).unsqueeze(-1)), dim=-1
    )
    math_utils.quat_mul = _quat_mul
    adr_state.ADR_STATE_SLICES = {"max_acceleration": slice(46, 47)}
    adr_state.get_gm_adr_state = lambda env, *_args, **_kwargs: env._adr_state
    tactile_state.reset_tactile_contact_state = lambda *_args, **_kwargs: None
    replacement_modules = {
        "isaaclab": isaaclab,
        "isaaclab.assets": assets,
        "isaaclab.envs": envs,
        "isaaclab.envs.mdp": env_mdp,
        "isaaclab.managers": managers,
        "isaaclab.utils": utils,
        "isaaclab.utils.math": math_utils,
        "anymani.tasks.gm.mdp.adr_state": adr_state,
        "anymani.tasks.gm.mdp.tactile_contact_state": tactile_state,
    }
    previous = {name: sys.modules.get(name) for name in replacement_modules}
    sys.modules.update(
        replacement_modules
    )
    try:
        spec = importlib.util.spec_from_file_location("anymani.tasks.gm.mdp.events_for_test", EVENTS_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module  # dataclass string annotation resolution 需要模块已注册
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop("anymani.tasks.gm.mdp.events_for_test", None)
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


def test_generated_structural_collision_filter_keeps_cross_finger_collisions() -> None:
    r"""结构过滤应只移除 palm-finger 与 same-finger pair，不移除 cross-finger pair。"""

    module = _load_events_module()
    pairs = set(
        module.generated_structural_collision_filter_pairs(
            palm_link_name="palm",
            finger_link_chains=(
                ("index_root", "index_tip"),
                ("thumb_root", "thumb_tip"),
            ),
        )
    )

    assert tuple(sorted(("palm", "index_root"))) in pairs
    assert tuple(sorted(("index_root", "index_tip"))) in pairs
    assert tuple(sorted(("thumb_root", "thumb_tip"))) in pairs
    assert tuple(sorted(("index_tip", "thumb_tip"))) not in pairs
    assert tuple(sorted(("index_root", "thumb_root"))) not in pairs


def test_generated_structural_collision_filter_uses_filtered_pairs_api() -> None:
    r"""runtime authoring 应使用 pairwise `FilteredPairsAPI`，不要回退到 collision group。"""

    source = EVENTS_PATH.read_text(encoding="utf-8")

    assert "FilteredPairsAPI.Apply" in source
    assert '"PhysicsCollisionGroup"' not in source
    assert "CollectionAPI:colliders" not in source
    assert "CreateFilteredPairsRel" in source


def test_body_yaw_reset_right_multiplies_non_identity_default_orientation() -> None:
    r"""Body yaw 必须右乘默认姿态；非单位四元数可直接证伪 world-frame 左乘。"""

    module = _load_events_module()
    half_sqrt_two = math.sqrt(0.5)  # 默认姿态为绕 world x 轴 $90^\circ$
    default_quat = torch.tensor([[half_sqrt_two, half_sqrt_two, 0.0, 0.0]])  # `(w,x,y,z)`
    yaw = torch.tensor([math.pi / 2.0])  # 绕 object body $z_o$ 轴 $90^\circ$

    reset_quat = module.compose_body_yaw_reset_quaternion(default_quat, yaw)

    expected_right_product = torch.tensor([[0.5, 0.5, -0.5, 0.5]])  # $q_{wo,0}\otimes q_z$
    world_left_product = torch.tensor([[0.5, 0.5, 0.5, 0.5]])  # $q_z\otimes q_{wo,0}$，错误 frame
    torch.testing.assert_close(reset_quat, expected_right_product)
    assert not torch.allclose(reset_quat, world_left_product)


def test_episode_reset_snapshots_global_adr_level_only_for_reset_envs() -> None:
    r"""Curriculum 发布值只应在 episode reset 时写入对应 env 的 actual ADR state。"""

    module = _load_events_module()
    writes: dict[str, tuple[float, torch.Tensor]] = {}

    class _State:
        def set(self, _env, field: str, value: float, env_ids: torch.Tensor) -> None:
            writes[field] = (float(value), env_ids.clone())

    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        max_episode_length=2400,
        step_dt=0.05,
        leap_adr_action_noise=0.14,
        leap_adr_max_linear_accel=1.4,
        leap_adr_fraction=0.4,
        _adr_state=_State(),
    )

    module.reset_adr_episode_length(env, torch.tensor([1]), min_episode_length_s=20.0)

    assert set(writes) == {"action_noise", "max_acceleration", "fraction"}
    assert writes["action_noise"][0] == 0.14
    assert writes["max_acceleration"][0] == 1.4
    assert writes["fraction"][0] == 0.4
    assert all(torch.equal(ids, torch.tensor([1])) for _, ids in writes.values())
    assert env.leap_adr_episode_lengths[0].item() == env.max_episode_length
    assert 400 <= env.leap_adr_episode_lengths[1].item() <= env.max_episode_length


def _quat_mul(quat_1: torch.Tensor, quat_2: torch.Tensor) -> torch.Tensor:
    r"""Hamilton product stub，输入输出均为 `(w,x,y,z)`。"""

    w1, x1, y1, z1 = quat_1.unbind(-1)
    w2, x2, y2, z2 = quat_2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )
