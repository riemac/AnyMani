r"""MVP80 rl_games structured network、privilege边界与分层minibatch合同。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from anymani.distill.rl.masked_ppo import AnyManiMaskedContinuousModel
from anymani.distill.rl.palm_rotation_ppo import (
    PalmRotationMaskedContinuousModel,
    PalmRotationPpoAgent,
    PalmRotationRlGamesBuilder,
    PalmRotationRlGamesNetwork,
    bounded_adaptive_learning_rate,
    stratified_asset_permutation,
)
from anymani.distill.rl.runtime.palm_rotation_vecenv import (
    PALM_ROTATION_BOOL_SHAPES,
    PALM_ROTATION_FLOAT_SHAPES,
    PALM_ROTATION_INT16_SHAPES,
)


def _input_shapes() -> dict[str, tuple[int, ...]]:
    r"""返回transport与network共同冻结的sample-level Dict ABI。"""

    return {**PALM_ROTATION_FLOAT_SHAPES, **PALM_ROTATION_BOOL_SHAPES, **PALM_ROTATION_INT16_SHAPES}


def _network(*, residual_enabled: bool = True) -> PalmRotationRlGamesNetwork:
    r"""构造不依赖Isaac/真实N040的FP32 actor/critic network。"""

    builder = PalmRotationRlGamesBuilder()
    builder.load(
        {
            "palm_rotation": {
                "residual_enabled": residual_enabled,
                "initial_log_std": -0.5,
                "max_log_std": -0.43,
                "base_action_limit": 0.8,
            },
            "anymani_identity": {"identity_digest": "contract-test"},
        }
    )
    return builder.build(
        "a2c",
        actions_num=16,
        input_shape=_input_shapes(),
        value_size=1,
        num_seqs=2,
    )


def _observation(batch: int = 2) -> dict[str, torch.Tensor]:
    r"""构造含12/9 active DoF与完整privileged blocks的synthetic experience batch。"""

    torch.manual_seed(20260902)
    joint_valid = torch.tensor(
        [
            [True] * 12 + [False] * 4,
            [True, False, True, True] * 2 + [True, False, False, True] + [False] * 4,
        ],
        dtype=torch.bool,
    )[:batch]  # 两种variable-cardinality masks
    tip_valid = torch.tensor([[True, True, True, True], [True, False, True, True]], dtype=torch.bool)[:batch]
    owner_valid = torch.cat((torch.ones(batch, 1, dtype=torch.bool), joint_valid, tip_valid), dim=1)  # `[B,21]`
    graph = torch.zeros(batch, 21, 21, dtype=torch.int16)  # 合法relation bucket 0
    return {
        "actor_jnt_current": torch.randn(batch, 16, 5),
        "actor_jnt_history": torch.randn(batch, 30, 16, 5),
        "actor_jnt_limits": torch.stack((-torch.ones(batch, 16), torch.ones(batch, 16)), dim=-1),
        "actor_owner_contact": torch.randint(0, 2, (batch, 21, 1)).float(),
        "critic_jnt_state": torch.randn(batch, 16, 4),
        "critic_owner_contact": torch.rand(batch, 21, 2),
        "critic_obj": torch.randn(batch, 1, 15),
        "critic_task": torch.randn(batch, 1, 8),
        "critic_reward_release": torch.rand(batch, 1),
        "jnt_valid": joint_valid,
        "tip_valid": tip_valid,
        "owner_valid": owner_valid,
        "geometry_tokens": torch.randn(batch, 21, 128),
        "shortest_path": graph.clone(),
        "parent_direction": graph.clone(),
        "child_direction": graph.clone(),
        "prototype_index": torch.arange(batch, dtype=torch.int16).unsqueeze(-1),
    }


def test_stratified_permutation_balances_every_asset_in_every_minibatch() -> None:
    r"""80 assets×12 samples切成4份后，每份必须逐资产恰含3项。"""

    labels = torch.arange(80).repeat_interleave(12)  # 960 synthetic transitions
    generator = torch.Generator().manual_seed(42)
    permutation = stratified_asset_permutation(
        labels,
        asset_count=80,
        minibatch_count=4,
        generator=generator,
    )
    assert torch.equal(torch.sort(permutation).values, torch.arange(labels.numel()))
    for minibatch in permutation.reshape(4, -1):
        counts = torch.bincount(labels[minibatch], minlength=80)
        assert torch.equal(counts, torch.full((80,), 3, dtype=torch.long))


def test_adaptive_lr_can_decrease_or_recover_but_never_exceed_anchor() -> None:
    r"""低KL反复乘1.5只能恢复到$3e-4$，不能进入rl_games默认$1e-2$上限。"""

    assert bounded_adaptive_learning_rate(1.0e-4, 3.0e-4) == 1.0e-4
    assert bounded_adaptive_learning_rate(4.5e-4, 3.0e-4) == 3.0e-4


def test_custom_train_wrapper_preserves_actor_group_lr_ratio_at_step_time() -> None:
    r"""Custom wrapper不得执行upstream逐microbatch的all-groups=`last_lr`覆盖。"""

    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD(
        [
            {"params": [parameter], "lr": 3.0e-4, "name": "actor_base"},
            {"params": [], "lr": 1.0e-4, "name": "actor_global_residual"},
        ]
    )
    fake = SimpleNamespace(
        optimizer=optimizer,
        last_lr=3.0e-4,
        _residual_lr_ratio=1.0 / 3.0,
        train_result=("sentinel",),
        set_train=lambda: None,
        calc_gradients=lambda _batch: None,
    )
    result = PalmRotationPpoAgent.train_actor_critic(fake, {})  # type: ignore[arg-type]
    assert result == ("sentinel",)
    assert [group["lr"] for group in optimizer.param_groups] == [3.0e-4, 1.0e-4]
    PalmRotationPpoAgent._assert_actor_learning_rate_ratio(fake)  # type: ignore[arg-type]
    optimizer.param_groups[1]["lr"] = 3.0e-4
    with pytest.raises(RuntimeError, match="residual LR ratio"):
        PalmRotationPpoAgent._assert_actor_learning_rate_ratio(fake)  # type: ignore[arg-type]


def test_network_blocks_critic_privilege_and_prototype_from_actor() -> None:
    r"""改变object/task/asset label可改变value，但不能改变actor mean。"""

    network = _network()
    observation = _observation()
    baseline_mu, baseline_logstd, baseline_value, _ = network({"obs": observation})
    changed = {key: value.clone() for key, value in observation.items()}
    changed["critic_obj"].add_(100.0)  # actor禁止读取object pose/velocity
    changed["critic_task"].mul_(-7.0)  # actor禁止读取goal/progress
    changed["prototype_index"] = changed["prototype_index"].flip(0)  # asset index只服务routing/sampling certificate
    changed_mu, changed_logstd, changed_value, _ = network({"obs": changed})

    torch.testing.assert_close(changed_mu, baseline_mu, rtol=0.0, atol=0.0)
    torch.testing.assert_close(changed_logstd, baseline_logstd, rtol=0.0, atol=0.0)
    assert not torch.equal(changed_value, baseline_value)


def test_network_has_disjoint_actor_groups_critic_and_masked_probability() -> None:
    r"""Base/residual/critic参数互斥，custom Normal只对active joints计概率。"""

    network = _network()
    base_parameters, residual_parameters = network.actor_parameter_groups()
    base_ids = {id(parameter) for parameter in base_parameters}
    residual_ids = {id(parameter) for parameter in residual_parameters}
    critic_ids = {id(parameter) for parameter in network.package.critic.parameters()}
    assert base_ids.isdisjoint(residual_ids)
    assert base_ids.isdisjoint(critic_ids)
    assert residual_ids.isdisjoint(critic_ids)

    # custom model contract验证ghost action任意污染不改变negative log-prob。
    builder = PalmRotationRlGamesBuilder()
    builder.load(
        {
            "palm_rotation": {
                "residual_enabled": True,
                "initial_log_std": -0.5,
                "max_log_std": -0.43,
                "base_action_limit": 0.8,
            },
            "anymani_identity": {"identity_digest": "contract-test"},
        }
    )
    model = AnyManiMaskedContinuousModel(builder).build(
        {
            "actions_num": 16,
            "input_shape": _input_shapes(),
            "value_size": 1,
            "normalize_input": False,
            "normalize_value": False,
        }
    )
    observation = _observation()
    actions = torch.randn(2, 16)
    poisoned = actions.clone()
    poisoned[~observation["jnt_valid"]] = 1.0e12
    baseline = model({"obs": observation, "prev_actions": actions, "is_train": True})
    changed = model({"obs": observation, "prev_actions": poisoned, "is_train": True})
    torch.testing.assert_close(changed["prev_neglogp"], baseline["prev_neglogp"])
    torch.testing.assert_close(changed["entropy"], baseline["entropy"])


def test_critic_backward_does_not_create_actor_gradients() -> None:
    r"""Privileged value loss只产生$\theta^c$梯度，actor参数必须保持None。"""

    network = _network()
    _, _, value, _ = network({"obs": _observation()})
    value.square().mean().backward()
    assert all(parameter.grad is None for parameter in network.package.actor.parameters())
    assert any(parameter.grad is not None for parameter in network.package.critic.parameters())


def test_palm_rotation_model_exposes_detached_bounded_residual() -> None:
    r"""Rollout model必须把`[B,16]` residual作为无梯度buffer side-channel交付。"""

    builder = PalmRotationRlGamesBuilder()
    builder.load(
        {
            "palm_rotation": {
                "residual_enabled": True,
                "initial_log_std": -0.5,
                "max_log_std": -0.43,
                "base_action_limit": 0.8,
            },
            "anymani_identity": {"identity_digest": "contract-test"},
        }
    )
    model = PalmRotationMaskedContinuousModel(builder).build(
        {
            "actions_num": 16,
            "input_shape": _input_shapes(),
            "value_size": 1,
            "normalize_input": False,
            "normalize_value": False,
        }
    )
    output = model({"obs": _observation(), "prev_actions": None, "is_train": False})
    actions = output["actions"]
    residual = output["residuals"]
    film = output["film_modulations"]
    assert isinstance(residual, torch.Tensor)
    assert residual.shape == (2, 16) and not residual.requires_grad
    assert float(residual.abs().max().item()) <= 0.2
    assert isinstance(film, torch.Tensor) and film.shape == (2, 16) and not film.requires_grad
    torch.testing.assert_close(film, torch.zeros_like(film), rtol=0.0, atol=0.0)
    assert isinstance(actions, torch.Tensor) and bool((actions.abs() < 1.0).all())
    assert bool(torch.isfinite(output["neglogpacs"]).all())

    training = model({"obs": _observation(), "prev_actions": actions, "is_train": True})
    assert bool(torch.isfinite(training["prev_neglogp"]).all())
    assert bool(torch.isfinite(training["entropy"]).all())


def test_squashed_policy_kl_is_zero_for_identical_bounded_means_and_ignores_ghost() -> None:
    r"""Tanh双射下相同latent Normal的KL为0，ghost action mean任意污染不进入归约。"""

    current_mean = torch.tensor([[0.25, -0.75, 0.0]])
    old_mean = current_mean.clone()
    sigma = torch.full_like(current_mean, 0.65)
    active = torch.tensor([[True, True, False]])
    baseline = PalmRotationPpoAgent.masked_policy_kl(current_mean, sigma, old_mean, sigma, active)
    poisoned = current_mean.clone()
    poisoned[:, 2] = 1.0
    changed = PalmRotationPpoAgent.masked_policy_kl(poisoned, sigma, old_mean, sigma, active)
    torch.testing.assert_close(baseline, changed, rtol=0.0, atol=0.0)
    assert abs(float(baseline.item())) < 1.0e-4
