r"""48D actual ADR privileged state 的纯 tensor schema tests。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load_adr_state_module():
    r"""直接加载 pure-torch ADR state 文件，避免 task registry imports。"""

    module_path = Path(__file__).resolve().parents[1] / "mdp" / "adr_state.py"
    spec = importlib.util.spec_from_file_location("gm_adr_state_contract", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load GM ADR state from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_adr_state_schema_is_exactly_48d_without_overlap_or_gaps() -> None:
    r"""所有命名 field width 之和必须为 48，且每个 index 恰好属于一个 field。"""

    module = _load_adr_state_module()
    covered = []
    for field_slice in module.ADR_STATE_SLICES.values():
        covered.extend(range(field_slice.start, field_slice.stop))

    assert module.ADR_STATE_DIM == 48
    assert sorted(covered) == list(range(48))
    assert len(covered) == len(set(covered))


def test_adr_state_partial_updates_preserve_other_envs_and_features() -> None:
    r"""事件只写自己的命名 slice/partial env，不能污染其他实际 ADR feature。"""

    module = _load_adr_state_module()
    env = SimpleNamespace(num_envs=3, device="cpu", leap_adr_fraction=0.0)
    state = module.GmADRState(env)

    state.set(env, "com", torch.tensor([[0.01, -0.01, 0.005]]), env_ids=torch.tensor([1]))
    state.set(env, "stiffness", torch.arange(16, dtype=torch.float32), env_ids=torch.tensor([1, 2]))
    state.set(env, "latency_steps", torch.tensor([2.0, 3.0]), env_ids=torch.tensor([0, 2]))
    state.set(env, "action_noise", torch.tensor([0.11, 0.17]), env_ids=torch.tensor([0, 2]))
    state.set(env, "max_acceleration", torch.tensor([0.6, 1.2]), env_ids=torch.tensor([0, 2]))
    state.set(env, "fraction", torch.tensor([0.04, 0.20]), env_ids=torch.tensor([0, 2]))

    assert state.values.shape == (3, 48)
    assert torch.allclose(state.values[1, module.ADR_STATE_SLICES["com"]], torch.tensor([0.01, -0.01, 0.005]))
    assert torch.count_nonzero(state.values[0, module.ADR_STATE_SLICES["com"]]) == 0
    assert torch.allclose(state.values[2, module.ADR_STATE_SLICES["stiffness"]], torch.arange(16).float())
    assert torch.allclose(state.values[:, module.ADR_STATE_SLICES["latency_steps"]].flatten(), torch.tensor([2.0, 0.0, 3.0]))
    assert torch.allclose(state.values[:, module.ADR_STATE_SLICES["action_noise"]].flatten(), torch.tensor([0.11, 0.0, 0.17]))
    assert torch.allclose(
        state.values[:, module.ADR_STATE_SLICES["max_acceleration"]].flatten(), torch.tensor([0.6, 0.0, 1.2])
    )
    assert torch.allclose(state.values[:, module.ADR_STATE_SLICES["fraction"]].flatten(), torch.tensor([0.04, 0.0, 0.20]))


def test_adr_state_rejects_non_16d_gain_vector() -> None:
    r"""Kp/Kd schema 必须保持 16D canonical joint order。"""

    module = _load_adr_state_module()
    env = SimpleNamespace(num_envs=1, device="cpu")
    state = module.GmADRState(env)

    with pytest.raises(ValueError, match="expects shape"):
        state.set(env, "damping", torch.zeros(15))
