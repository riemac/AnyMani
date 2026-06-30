from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("pxr", reason="inhand ADR contract imports IsaacLab modules that require USD pxr bindings")

from anymani.tasks.inhand.mdp.actions.adr_relative_action import compute_leap_adr_latency_steps
from anymani.tasks.inhand.mdp.curriculums import LeapADRGlobalScheduler, leap_adr_interpolate
from anymani.tasks.inhand.mdp.events import resample_adr_material_buckets


class _FakeCommandTerm:
    def __init__(self, successes: torch.Tensor):
        self.metrics = {"consecutive_success": successes}


class _FakeCommandManager:
    def __init__(self, successes: torch.Tensor):
        self._term = _FakeCommandTerm(successes)

    def get_term(self, name: str):
        assert name == "goal_pose"
        return self._term


class _FakeEventManager:
    def get_term_cfg(self, term_name: str):
        raise ValueError(term_name)

    def set_term_cfg(self, term_name: str, term_cfg):
        raise AssertionError("contract scheduler test should not mutate absent event terms")


class _FakeMaterialTerm:
    def __init__(self):
        self.material_buckets = torch.zeros(8, 3)
        self.called = False
        self._adr_bucket_signature = None

    def __call__(self, env, env_ids, **params):
        self.called = True
        self.last_env_ids = env_ids
        self.last_params = params


class _FakeMaterialEventManager:
    def __init__(self, term):
        self.term_cfg = SimpleNamespace(
            func=term,
            params={"num_buckets": 8, "make_consistent": True},
        )

    def get_term_cfg(self, term_name: str):
        assert term_name == "randomized_object_friction"
        return self.term_cfg


def test_leap_adr_interpolate_nested_structure():
    initial = {"mass": (1.0, 1.0), "material": {"static": (1.0, 1.0), "restitution": (0.0, 0.0)}}
    final = {"mass": (0.9, 1.3), "material": {"static": (0.3, 1.5), "restitution": (0.0, 0.5)}}

    out = leap_adr_interpolate(initial, final, 0.5)

    assert out["mass"] == pytest.approx((0.95, 1.15))
    assert out["material"]["static"] == pytest.approx((0.65, 1.25))
    assert out["material"]["restitution"] == pytest.approx((0.0, 0.25))


def _make_scheduler_env(successes: torch.Tensor):
    env = SimpleNamespace()
    env.num_envs = successes.numel()
    env.device = "cpu"
    env.step_dt = 1.0 / 30.0
    env.command_manager = _FakeCommandManager(successes)
    env.event_manager = _FakeEventManager()
    env.leap_adr_episode_lengths = torch.full((successes.numel(),), int(70.0 / env.step_dt), dtype=torch.long)
    return env


def test_leap_adr_scheduler_promotes_and_resets_ema():
    env = _make_scheduler_env(torch.full((4,), 200.0))
    scheduler = LeapADRGlobalScheduler(SimpleNamespace(params={}), env)

    state = scheduler(
        env,
        torch.arange(4),
        min_steps_for_dr_change=0,
        ema_alpha=1.0,
    )

    assert state["increment"] == 1
    assert env.leap_adr_increment == 1
    assert state["ema_success"].item() == pytest.approx(0.0)
    assert env.leap_adr_action_noise > 0.1


def test_leap_adr_scheduler_respects_reset_check_cooldown():
    env = _make_scheduler_env(torch.full((4,), 200.0))
    scheduler = LeapADRGlobalScheduler(SimpleNamespace(params={}), env)

    state = scheduler(
        env,
        torch.arange(4),
        min_steps_for_dr_change=10,
        ema_alpha=1.0,
    )

    assert state["increment"] == 0
    assert state["reset_checks_since_increase"] == 1
    assert state["adr_criteria"].item() > 0.15


def test_resample_adr_material_buckets_updates_bucket_tensor():
    term = _FakeMaterialTerm()
    env = SimpleNamespace()
    env.event_manager = _FakeMaterialEventManager(term)
    env.leap_adr_object_material_ranges = {
        "static": (0.3, 1.5),
        "dynamic": (0.3, 1.5),
        "restitution": (0.0, 0.5),
    }

    resample_adr_material_buckets(
        env,
        torch.tensor([0, 1]),
        term_name="randomized_object_friction",
        range_attr="leap_adr_object_material_ranges",
    )

    assert term.called
    assert term.material_buckets.shape == (8, 3)
    assert torch.all(term.material_buckets[:, 1] <= term.material_buckets[:, 0])
    assert torch.all((term.material_buckets[:, 2] >= 0.0) & (term.material_buckets[:, 2] <= 0.5))


def test_resample_adr_material_buckets_reuses_signature_cache():
    term = _FakeMaterialTerm()
    env = SimpleNamespace()
    env.event_manager = _FakeMaterialEventManager(term)
    env.leap_adr_object_material_ranges = {
        "static": (0.3, 1.5),
        "dynamic": (0.3, 1.5),
        "restitution": (0.0, 0.5),
    }

    resample_adr_material_buckets(
        env,
        torch.tensor([0, 1]),
        term_name="randomized_object_friction",
        range_attr="leap_adr_object_material_ranges",
    )
    first = term.material_buckets.clone()

    resample_adr_material_buckets(
        env,
        torch.tensor([0, 1]),
        term_name="randomized_object_friction",
        range_attr="leap_adr_object_material_ranges",
    )

    assert torch.equal(term.material_buckets, first)


def test_action_latency_index_matches_leap_formula():
    r"""动作延迟索引必须对应 $\ell=\max(0,\lfloor h(k)-r\rfloor)$。"""

    random_subtraction = torch.tensor([[0], [1], [0], [1]])

    latency = compute_leap_adr_latency_steps(1.5, random_subtraction, max_latency=3)

    assert torch.equal(latency.squeeze(-1), torch.tensor([1, 0, 1, 0]))
