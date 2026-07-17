r"""共享 tactile contact state 的纯 tensor/lifecycle contract tests。

测试使用 fake ContactSensor data，不启动 Isaac Sim。重点证伪三类高风险错误：把方向相反的
contact pairs 先求和、同一 policy step 重复累计 EMA、partial reset 后 observation 读回上一
episode stale contact。
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
from anymani.tasks.gm.contact_sensors import sensor_contact_magnitude


def _load_contact_state_getter():
    r"""直接加载目标文件，避免 `gm.mdp.__init__` 在纯 pytest 中触发 Kit/pxr imports。"""

    module_path = Path(__file__).resolve().parents[1] / "mdp" / "tactile_contact_state.py"
    module_name = "anymani.tasks.gm.mdp.tactile_contact_state_contract"
    mdp_package_name = "anymani.tasks.gm.mdp"
    previous_package = sys.modules.get(mdp_package_name)
    mdp_package = types.ModuleType(mdp_package_name)
    mdp_package.__path__ = [str(module_path.parent)]  # type: ignore[attr-defined]  # 允许 `..contact_sensors` 相对 import
    sys.modules[mdp_package_name] = mdp_package
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load tactile contact state from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if previous_package is None:
            sys.modules.pop(mdp_package_name, None)
        else:
            sys.modules[mdp_package_name] = previous_package
    return module.get_tactile_contact_state


get_tactile_contact_state = _load_contact_state_getter()


def _sensor(force_matrix_w: torch.Tensor) -> SimpleNamespace:
    r"""构造只暴露 `force_matrix_w` 的 fake object-filtered ContactSensor。"""

    return SimpleNamespace(data=SimpleNamespace(force_matrix_w=force_matrix_w, friction_forces_w=None))


def _force(batch: int, magnitude: float = 0.0) -> torch.Tensor:
    r"""构造 `[B,body=1,filter=1,xyz]` 的确定性 x 向接触力。"""

    force = torch.zeros(batch, 1, 1, 3)
    force[..., 0] = float(magnitude)
    return force


def _fake_env(batch: int = 2) -> SimpleNamespace:
    r"""构造 tactile singleton 需要的最小 vectorized env。"""

    names = ("tip_0", "tip_1", "non_tip_0", "palm")
    return SimpleNamespace(
        num_envs=batch,
        device="cpu",
        common_step_counter=0,
        scene={name: _sensor(_force(batch)) for name in names},
    )


def _state(env: SimpleNamespace):
    r"""按 2-tip/1-non-tip/1-palm schema 取得测试 singleton。"""

    return get_tactile_contact_state(
        env,
        fingertip_sensor_names=("tip_0", "tip_1"),
        finger_non_tip_sensor_names=("non_tip_0",),
        palm_sensor_name="palm",
        ema_alpha=0.5,
        force_threshold=0.25,
    )


def test_sensor_contact_magnitude_takes_pair_max_before_aggregation() -> None:
    r"""方向相反的 3 N pairs 仍应报告 3 N，而不是向量相消为 0 N。"""

    pair_forces = torch.tensor([[[[3.0, 0.0, 0.0], [-3.0, 0.0, 0.0]]]])  # `[1,1,2,3]`
    env = SimpleNamespace(scene={"tip": _sensor(pair_forces)})

    assert torch.allclose(sensor_contact_magnitude(env, "tip"), torch.tensor([3.0]))


def test_contact_state_updates_ema_once_per_policy_step() -> None:
    r"""同一 `common_step_counter` 的多个 consumer 不得重复应用 EMA。"""

    env = _fake_env()
    env.scene["tip_0"].data.force_matrix_w = _force(2, 1.0)  # 首次更新：EMA 从 0 到 0.5 N
    state = _state(env)

    assert torch.allclose(state.tip_force_ema[:, 0], torch.full((2,), 0.5))
    assert torch.all(state.tip_bits[:, 0])  # $0.5>0.25$ N

    env.scene["tip_0"].data.force_matrix_w = _force(2, 0.0)  # 同 stamp 改 raw data，仍应复用原快照
    same_state = _state(env)
    assert same_state is state  # actor/reward/critic 读取同一 owner
    assert torch.allclose(state.tip_force_ema[:, 0], torch.full((2,), 0.5))

    env.common_step_counter += 1  # 下一 policy step 才允许再次更新
    _state(env)
    assert torch.allclose(state.tip_force_ema[:, 0], torch.full((2,), 0.25))
    assert not torch.any(state.tip_bits[:, 0])  # baseline 使用严格 `>0.25`，等于阈值不算接触


def test_partial_reset_zeros_only_selected_env_for_current_stamp() -> None:
    r"""partial reset 后本 stamp 保持零；未 reset env 保留刚计算的 episode state。"""

    env = _fake_env()
    env.scene["palm"].data.force_matrix_w = _force(2, 2.0)  # step 0 后 palm EMA 为 1 N
    state = _state(env)
    state.reset(env, torch.tensor([0]))  # env 0 开始新 episode；env 1 仍在旧 episode

    assert torch.allclose(state.palm_force_ema[:, 0], torch.tensor([0.0, 1.0]))
    assert torch.equal(state.palm_bits[:, 0], torch.tensor([False, True]))

    env.scene["palm"].data.force_matrix_w = _force(2, 4.0)
    _state(env)  # 同一 stamp 的 post-reset observation 不得立即把 stale sensor 重新写进 env 0
    assert torch.allclose(state.palm_force_ema[:, 0], torch.tensor([0.0, 1.0]))

    env.common_step_counter += 1
    _state(env)  # 经下一次 physics/policy step，两个 env 都从各自 episode state 正常更新
    assert torch.allclose(state.palm_force_ema[:, 0], torch.tensor([2.0, 2.5]))
