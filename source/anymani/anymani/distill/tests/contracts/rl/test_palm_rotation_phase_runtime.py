"""内部时钟只读取物理回合计数；不在模型重放时自增，也不进入 N040 输入。"""

from types import SimpleNamespace

import pytest
import torch
from anymani.distill.rl.runtime.palm_rotation_network import PalmRotationRlGamesNetwork
from anymani.distill.rl.runtime.palm_rotation_phase import (
    audit_phase_rollout,
    normalize_phase_period_steps,
    phase_clock_from_episode_steps,
)
from anymani.distill.rl.runtime.palm_rotation_vecenv import PalmRotationRlGamesVecEnv


@pytest.mark.parametrize("value", [0, 1, -2, True, 2.5, float("nan")])
def test_phase_period_rejects_non_integer_or_degenerate_period(value) -> None:
    with pytest.raises(ValueError, match="period"):
        normalize_phase_period_steps(value)


def test_phase_uses_integer_period_and_is_readonly() -> None:
    steps = torch.tensor([0, 1, 42, 43, 86, 10_000_043], dtype=torch.long)
    original = steps.clone()
    phase = phase_clock_from_episode_steps(steps, period_steps=43)
    expected_angle = torch.tensor([0, 1, 42, 0, 0, 10_000_043 % 43], dtype=torch.float64) * (2 * torch.pi / 43)
    expected = torch.stack((expected_angle.sin(), expected_angle.cos()), dim=-1).float()
    torch.testing.assert_close(phase, expected, atol=3e-7, rtol=0)
    assert torch.equal(phase[[0, 3, 4]], torch.tensor([[0.0, 1.0]]).expand(3, 2))
    assert torch.equal(steps, original)
    assert torch.equal(phase_clock_from_episode_steps(steps, period_steps=43), phase)
    assert phase.dtype == torch.float32 and phase.shape == (6, 2)
    assert normalize_phase_period_steps(None) is None


def test_stored_phase_rollout_matches_pre_action_resets_and_rejects_shift() -> None:
    steps = torch.tensor([[0, 1, 2, 3], [0, 1, 0, 1]], dtype=torch.long)
    phase = phase_clock_from_episode_steps(steps.flatten(), period_steps=43).reshape(2, 4, 2)
    resets = torch.tensor([[True, False, False, False], [True, False, True, False]])
    report = audit_phase_rollout(phase, resets, period_steps=43)
    assert report["status"] == "passed" and report["maximum_abs_error"] == 0
    with pytest.raises(RuntimeError, match="phase"):
        audit_phase_rollout(phase.roll(1, dims=1), resets, period_steps=43)
    with pytest.raises(RuntimeError, match="phase"):
        audit_phase_rollout(torch.zeros_like(phase), resets, period_steps=43)


class _RawEnv:
    def __init__(self) -> None:
        self.unwrapped = self
        self.device = "cpu"
        self.num_envs = 2
        self.single_action_space = SimpleNamespace(shape=(16,))
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        masks = {
            "jnt_valid": torch.ones(2, 16, dtype=torch.bool),
            "tip_valid": torch.ones(2, 4, dtype=torch.bool),
            "owner_valid": torch.ones(2, 21, dtype=torch.bool),
        }
        self.observation = {
            "policy": {
                **masks,
                "jnt_current": torch.zeros(2, 16, 5),
                "jnt_history": torch.zeros(2, 30, 16, 5),
                "jnt_limits": torch.tensor([-1.0, 1.0]).expand(2, 16, 2),
                "owner_contact": torch.zeros(2, 21, 1),
            },
            "critic": {
                **masks,
                "jnt_state": torch.zeros(2, 16, 4),
                "owner_contact": torch.zeros(2, 21, 2),
                "obj": torch.zeros(2, 1, 15),
                "task": torch.zeros(2, 1, 8),
                "reward_release": torch.ones(2, 1),
            },
        }

    def reset(self):
        self.episode_length_buf.zero_()
        return self.observation, {}

    def step(self, _actions):
        self.episode_length_buf += 1
        self.episode_length_buf[1] = 0  # 模拟 Isaac 在返回 observation 前仅重置 done 行。
        return self.observation, torch.zeros(2), torch.tensor([False, True]), torch.zeros(2, dtype=torch.bool), {}


class _Geometry:
    def __init__(self) -> None:
        self.resolve_call_count = 0

    def to(self, _device):
        return self

    def resolve(self, _prototype, observation):
        self.resolve_call_count += 1
        assert not hasattr(observation, "phase_clock"), "N040-facing dataclass gained clock input"
        return SimpleNamespace(
            tokens=torch.zeros(2, 21, 128),
            shortest_path=torch.zeros(2, 21, 21, dtype=torch.long),
            parent_direction=torch.zeros(2, 21, 21, dtype=torch.long),
            child_direction=torch.zeros(2, 21, 21, dtype=torch.long),
        )


@pytest.mark.parametrize("period", [None, 43])
def test_vecenv_clock_tracks_physical_reset_and_stays_out_of_env_state(period) -> None:
    raw, provider = _RawEnv(), _Geometry()
    wrapper = PalmRotationRlGamesVecEnv(
        raw,
        geometry_provider=provider,
        prototype_index=torch.tensor([0, 1]),
        rl_device="cpu",
        clip_observations=100.0,
        clip_actions=1.0,
        phase_period_steps=period,
    )
    wrapper._record_rollout_step = lambda _reward: None  # 本合同隔离时钟 transport，不伪造任务 reward 记录。
    first = wrapper.reset()["obs"]
    assert ("phase_clock" in first) == (period is not None)
    assert ("phase_clock" in wrapper.observation_space.spaces) == (period is not None)
    if period is not None:
        assert torch.equal(first["phase_clock"], torch.tensor([[0.0, 1.0], [0.0, 1.0]]))
    second, _, _, _ = wrapper.step(torch.zeros(2, 16))
    if period is not None:
        torch.testing.assert_close(
            second["obs"]["phase_clock"], phase_clock_from_episode_steps(torch.tensor([1, 0]), period_steps=43)
        )
    assert provider.resolve_call_count == 2
    saved = wrapper.get_env_state()
    assert "phase_clock" not in saved and "episode_length_buf" not in saved
    wrapper.reset()
    wrapper.set_env_state(saved)
    assert torch.equal(raw.episode_length_buf, torch.zeros(2, dtype=torch.long))


def test_network_phase_zero_preserves_outputs_and_has_explicit_optimizer_group() -> None:
    raw, provider = _RawEnv(), _Geometry()
    wrapper = PalmRotationRlGamesVecEnv(
        raw,
        geometry_provider=provider,
        prototype_index=torch.tensor([0, 1]),
        rl_device="cpu",
        clip_observations=100.0,
        clip_actions=1.0,
        phase_period_steps=43,
    )
    observation = wrapper.reset()["obs"]
    shapes = {name: tuple(space.shape) for name, space in wrapper.observation_space.spaces.items()}
    config = {
        "palm_rotation": {"arm": "direct_token", "phase_period_steps": 43},
        "anymani_identity": {"identity_digest": "phase-unit-test"},
    }
    enabled = PalmRotationRlGamesNetwork(config, actions_num=16, input_shape=shapes)
    old_shapes = {k: v for k, v in shapes.items() if k != "phase_clock"}
    disabled = PalmRotationRlGamesNetwork(
        {**config, "palm_rotation": {"arm": "direct_token"}}, actions_num=16, input_shape=old_shapes
    )
    mismatch = enabled.load_state_dict(disabled.state_dict(), strict=False)
    assert set(mismatch.missing_keys) == {
        "package.actor.phase_contextual_adapter.weight",
        "package.critic.phase_readout_adapter.weight",
    }
    assert not mismatch.unexpected_keys
    old_observation = {k: v for k, v in observation.items() if k != "phase_clock"}
    reference = disabled({"obs": old_observation})[:3]
    for phase in [torch.tensor([[0.0, 1.0], [1.0, 0.0]]), torch.tensor([[-1.0, 0.0], [0.0, -1.0]])]:
        observed = enabled({"obs": {**observation, "phase_clock": phase}})[:3]
        assert all(torch.equal(left, right) for left, right in zip(reference, observed, strict=True))
    base, contextual = enabled.actor_parameter_groups()
    phase_parameter = enabled.package.actor.phase_contextual_adapter.weight
    assert id(phase_parameter) in {id(p) for p in contextual}
    assert id(phase_parameter) not in {id(p) for p in base}
