r"""真实生产PPO目标的全资产梯度合同，固定MC entropy样本，不启动Isaac。"""

from __future__ import annotations

import math

import pytest
import torch
from anymani.distill.rl.algorithms.task_gradients import per_asset_ppo_gradients
from anymani.distill.rl.palm_rotation_ppo import PalmRotationMaskedContinuousModel, PalmRotationRlGamesBuilder
from anymani.distill.rl.structured_masked_distribution import masked_bound_loss
from anymani.distill.tests.contracts.rl.test_palm_rotation_ppo import _input_shapes, _observation
from rl_games.common import common_losses


@pytest.mark.parametrize("sigma_mode", ["global", "conditional"])
@pytest.mark.parametrize("rejected_action_weight", [0.0, 0.05, 5.0])
@pytest.mark.parametrize("phase_enabled", [False, True])
def test_functional_task_gradients_match_production_graph(monkeypatch, sigma_mode, rejected_action_weight, phase_enabled) -> None:
    r"""两种active masks、非连续资产顺序和相同随机熵下，比较所有参数而非仅head。"""
    torch.set_num_threads(2)
    torch.manual_seed(42)
    obs = {key: value.repeat_interleave(2, 0) for key, value in _observation().items()}
    if phase_enabled:
        obs["phase_clock"] = torch.tensor([[0., 1.], [1., 0.], [0., -1.], [-1., 0.]])
    order = torch.tensor([2, 0, 3, 1])
    obs = {key: value[order] for key, value in obs.items()}
    obs["prototype_index"] = torch.tensor([[1], [0], [1], [0]], dtype=torch.int16)
    if rejected_action_weight:
        # 同一正向动作在asset0的上限会被拒绝，在asset1的下限仍可执行。
        upper = obs["prototype_index"].reshape(-1) == 0
        obs["actor_jnt_current"][..., 1] = torch.where(
            upper[:, None], obs["actor_jnt_limits"][..., 1], obs["actor_jnt_limits"][..., 0]
        )
        obs["actor_owner_contact"].zero_()
    builder = PalmRotationRlGamesBuilder()
    builder.load(
        {
            "palm_rotation": {"arm": "direct_token", "history_encoder": "tcn", "sigma_mode": sigma_mode,
                              **({"phase_period_steps": 43} if phase_enabled else {})},
            "anymani_identity": {"identity_digest": "test"},
        }
    )
    model = PalmRotationMaskedContinuousModel(builder).build(
        {
            "actions_num": 16,
            "input_shape": {**_input_shapes(), **({"phase_clock": (2,)} if phase_enabled else {})},
            "value_size": 1,
            "normalize_input": False,
            "normalize_value": False,
        }
    )
    package = model.a2c_network.package
    if phase_enabled:
        with torch.no_grad():
            package.actor.phase_contextual_adapter.weight.normal_(0., .02)
            package.critic.phase_readout_adapter.weight.normal_(0., .02)
    if rejected_action_weight:
        with torch.no_grad():
            package.actor.direct_head[-1].bias.fill_(0.7)
    actions = torch.randn(4, 16).tanh() * obs["jnt_valid"]
    noise = torch.randn_like(actions)
    monkeypatch.setattr(
        torch.distributions.Normal, "rsample", lambda self, sample_shape=torch.Size(): self.loc + self.scale * noise
    )
    result = model({"is_train": True, "obs": dict(obs), "prev_actions": actions})
    data = {
        "actions": actions,
        "old_logp_actions": result["prev_neglogp"].detach() + 0.02,
        "advantages": torch.tensor([1.0, -0.4, -0.2, 0.8]),
        "old_values": result["values"].detach() + 0.01,
        "returns": torch.tensor([[1.0], [-1.0], [0.5], [0.2]]),
    }
    actor_objective = common_losses.actor_loss(
        data["old_logp_actions"], result["prev_neglogp"], data["advantages"], True, 0.2
    )
    actor_objective = (
        actor_objective - 0.002 * result["entropy"] + 1e-4 * masked_bound_loss(result["mus"], obs["jnt_valid"])
    )
    if rejected_action_weight:
        # 独立按目标空间公式构造参考，不复用生产helper的action-space实现。
        u = obs["actor_jnt_current"][..., 1]
        limits = obs["actor_jnt_limits"]
        target = torch.clamp(u + result["mus"] / (24 * math.pi), limits[..., 0], limits[..., 1])
        rejected = result["mus"] - 24 * math.pi * (target - u)
        cost = (rejected.square() * obs["jnt_valid"]).sum(-1) / obs["jnt_valid"].sum(-1)
        assert bool((cost[upper] > 0.1).all())
        actor_objective = actor_objective + rejected_action_weight * cost
    critic_objective = 2.0 * common_losses.critic_loss(
        model, data["old_values"], result["values"], 0.2, data["returns"], True
    ).reshape(-1)
    labels = obs["prototype_index"].reshape(-1)
    references = []
    for module, objective in ((package.actor, actor_objective), (package.critic, critic_objective)):
        params = dict(module.named_parameters())
        rows = [
            torch.autograd.grad(
                objective[labels == task].mean(), tuple(params.values()), retain_graph=True, allow_unused=True
            )
            for task in range(2)
        ]
        references.append(
            {
                name: torch.stack([torch.zeros_like(p) if row[i] is None else row[i] for row in rows])
                for i, (name, p) in enumerate(params.items())
            }
        )
    for chunk in (1, 2):
        actor, critic, auxiliary = per_asset_ppo_gradients(
            package, obs, data, asset_count=2, entropy_noise=noise, chunk_size=chunk,
            rejected_action_weight=rejected_action_weight,
        )
        for actual, expected in zip((actor, critic), references):
            assert actual.keys() == expected.keys()
            for name in expected:
                assert not actual[name].requires_grad, "task gradients must not retain higher-order graphs"
                torch.testing.assert_close(actual[name], expected[name], rtol=3e-4, atol=3e-6, msg=name)
        torch.testing.assert_close(auxiliary["mus"], result["mus"], rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(auxiliary["entropy"], result["entropy"], rtol=1e-5, atol=1e-7)
    assert all(parameter.grad is None for parameter in package.parameters())


def test_invalid_masks_are_checked_before_vectorization() -> None:
    r"""向量化计算路径不能绕过生产mask合同。"""
    from anymani.distill.models.palm_rotation_policy import PalmRotationActorCritic

    obs = _observation()
    obs["owner_valid"][0, 0] = False
    data = {"actions": torch.zeros(2, 16)}
    with pytest.raises(RuntimeError, match="owner masks"):
        per_asset_ppo_gradients(
            PalmRotationActorCritic(arm="direct_token"), obs, data, asset_count=2, entropy_noise=torch.zeros(2, 16)
        )
