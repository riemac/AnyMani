r"""MVP80 rl_games structured network、privilege边界与分层minibatch合同。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from anymani.distill.rl.masked_ppo import AnyManiMaskedContinuousModel
from anymani.distill.rl.palm_rotation_ppo import (
    PalmRotationMaskedContinuousModel,
    PalmRotationPpoAgent,
    PalmRotationPpoDiagnostics,
    PalmRotationRlGamesBuilder,
    PalmRotationRlGamesNetwork,
    bounded_adaptive_learning_rate,
    denormalize_value_readonly,
    normalize_advantages_per_asset,
    rollout_policy_mechanism_metrics,
    stratified_asset_permutation,
    validate_gradient_probe_compile_compatibility,
)
from anymani.distill.rl.runtime.palm_rotation_vecenv import (
    PALM_ROTATION_BOOL_SHAPES,
    PALM_ROTATION_FLOAT_SHAPES,
    PALM_ROTATION_INT16_SHAPES,
)


def _input_shapes() -> dict[str, tuple[int, ...]]:
    r"""返回transport与network共同冻结的sample-level Dict ABI。"""

    return {**PALM_ROTATION_FLOAT_SHAPES, **PALM_ROTATION_BOOL_SHAPES, **PALM_ROTATION_INT16_SHAPES}


def _network(*, arm: str = "residual", compile_mode: str | None = None) -> PalmRotationRlGamesNetwork:
    r"""构造不依赖Isaac/真实N040的FP32 actor/critic network。"""

    builder = PalmRotationRlGamesBuilder()
    builder.load(
        {
            "palm_rotation": {
                "arm": arm,
                "initial_log_std": -0.5,
                "max_log_std": -0.43,
                "base_action_limit": 0.8,
                "compile_mode": compile_mode,
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


def test_compile_mode_wraps_only_bound_forwards_and_preserves_checkpoint_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""项目compile不得把rl_games model包装成`_orig_mod`，optimizer/state_dict仍拥有原参数。"""

    compiled: list[tuple[object, str]] = []

    def fake_compile(function: object, *, mode: str) -> object:
        r"""记录窄compile调用并原样返回bound method。"""

        compiled.append((function, mode))
        return function

    monkeypatch.setattr(torch, "compile", fake_compile)
    network = _network(compile_mode="default")

    assert len(compiled) == 2 and [mode for _, mode in compiled] == ["default", "default"]
    assert all("_orig_mod" not in key for key in network.state_dict())
    assert network._actor_forward == network.package.actor.forward
    assert network._critic_forward == network.package.critic.forward


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


def test_per_asset_advantage_normalization_removes_independent_affine_scales() -> None:
    r"""每资产full-rollout moments应消除独立正缩放/平移，同时保留样本对应关系。"""

    labels = torch.tensor((0, 1, 0, 1, 0, 1, 0, 1))
    advantages = torch.tensor((-1.0, 70.0, 0.0, 90.0, 1.0, 110.0, 2.0, 130.0))
    normalized, means, standard_deviations = normalize_advantages_per_asset(
        advantages,
        labels,
        asset_count=2,
    )

    torch.testing.assert_close(means, torch.tensor((0.5, 100.0)))
    torch.testing.assert_close(standard_deviations[1], 20.0 * standard_deviations[0])
    for asset_index in range(2):
        asset_values = normalized[labels == asset_index]
        torch.testing.assert_close(asset_values.mean(), torch.tensor(0.0), atol=1.0e-7, rtol=0.0)
        torch.testing.assert_close(asset_values.std(), torch.tensor(1.0), atol=1.0e-6, rtol=0.0)

    transformed = torch.where(labels == 0, advantages * 17.0 + 3.0, advantages * 0.2 - 11.0)
    transformed_normalized, _transformed_means, _transformed_stds = normalize_advantages_per_asset(
        transformed,
        labels,
        asset_count=2,
    )
    torch.testing.assert_close(transformed_normalized, normalized, atol=2.0e-6, rtol=0.0)


def test_per_asset_advantage_normalization_rejects_incomplete_asset_statistics() -> None:
    r"""每个资产至少两个样本才能形成无偏rollout标准差；标签轴必须与advantage逐项对应。"""

    with pytest.raises(ValueError, match="at least two"):
        normalize_advantages_per_asset(torch.tensor((0.0, 1.0, 2.0)), torch.tensor((0, 0, 1)), asset_count=2)
    with pytest.raises(ValueError, match="align"):
        normalize_advantages_per_asset(torch.tensor((0.0, 1.0)), torch.tensor((0,)), asset_count=1)


def test_adaptive_lr_can_decrease_or_recover_but_never_exceed_anchor() -> None:
    r"""低KL反复乘1.5只能恢复到$3e-4$，不能进入rl_games默认$1e-2$上限。"""

    assert bounded_adaptive_learning_rate(1.0e-4, 3.0e-4) == 1.0e-4
    assert bounded_adaptive_learning_rate(4.5e-4, 3.0e-4) == 3.0e-4


def test_compiled_forward_rejects_retain_graph_gradient_probe_before_training() -> None:
    r"""Compile可用于普通训练，head-gradient probe只能走已验证的eager backward。"""

    validate_gradient_probe_compile_compatibility("default", 0)
    validate_gradient_probe_compile_compatibility(None, 500)
    validate_gradient_probe_compile_compatibility(None, 0, 1)
    with pytest.raises(ValueError, match="requires eager"):
        validate_gradient_probe_compile_compatibility("default", 500)
    with pytest.raises(ValueError, match="requires eager"):
        validate_gradient_probe_compile_compatibility("default", 0, 1)


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
        _secondary_lr_ratio=1.0 / 3.0,
        _secondary_group_name="actor_global_residual",
        train_result=("sentinel",),
        set_train=lambda: None,
        calc_gradients=lambda _batch: None,
    )
    result = PalmRotationPpoAgent.train_actor_critic(fake, {})  # type: ignore[arg-type]
    assert result == ("sentinel",)
    assert [group["lr"] for group in optimizer.param_groups] == [3.0e-4, 1.0e-4]
    PalmRotationPpoAgent._assert_actor_learning_rate_ratio(fake)  # type: ignore[arg-type]
    optimizer.param_groups[1]["lr"] = 3.0e-4
    with pytest.raises(RuntimeError, match="contextual LR ratio"):
        PalmRotationPpoAgent._assert_actor_learning_rate_ratio(fake)  # type: ignore[arg-type]


def test_custom_diagnostics_defers_device_transfer_until_publish_boundary() -> None:
    r"""Microbatch诊断应保留device scalars，避免每份explained-variance与clip统计同步host。"""

    diagnostics = PalmRotationPpoDiagnostics()
    batch = {
        "values": torch.tensor([[0.0], [1.0]]),
        "returns": torch.tensor([[0.5], [1.5]]),
        "new_neglogp": torch.tensor([0.0, 0.3]),
        "old_neglogp": torch.tensor([0.0, 0.0]),
        "masks": None,
    }
    diagnostics.mini_batch(SimpleNamespace(), batch, 0.2, 0)

    assert len(diagnostics.exp_vars) == 1 and diagnostics.exp_vars[0].device == batch["values"].device
    assert len(diagnostics.clip_fracs) == 1 and diagnostics.clip_fracs[0].device == batch["values"].device
    assert diagnostics.exp_vars[0].grad_fn is None and diagnostics.clip_fracs[0].grad_fn is None


def test_diagnostic_denormalization_is_read_only_while_model_trains() -> None:
    r"""物理value诊断不得让train-mode ``RunningMeanStd``吸收normalized critic prediction。"""

    builder = PalmRotationRlGamesBuilder()
    builder.load(
        {
            "palm_rotation": {
                "arm": "residual",
                "initial_log_std": -0.5,
                "max_log_std": -0.43,
                "base_action_limit": 0.8,
            },
            "anymani_identity": {"identity_digest": "normalizer-purity-contract"},
        }
    )
    model = PalmRotationMaskedContinuousModel(builder).build(
        {
            "actions_num": 16,
            "input_shape": _input_shapes(),
            "value_size": 1,
            "normalize_input": False,
            "normalize_value": True,
        }
    )
    model.train()  # 复现PPO microbatch；只允许diagnostic helper暂时切换value normalizer
    normalizer = model.value_mean_std
    with torch.no_grad():
        normalizer.running_mean.copy_(torch.tensor((1.25,), dtype=torch.float64))
        normalizer.running_var.copy_(torch.tensor((2.25,), dtype=torch.float64))
        normalizer.count.copy_(torch.tensor(37.0, dtype=torch.float64))
    before = tuple(
        value.detach().clone() for value in (normalizer.running_mean, normalizer.running_var, normalizer.count)
    )
    normalized_value = torch.tensor(((-1.0,), (0.5,), (2.0,)))

    physical_value = denormalize_value_readonly(model, normalized_value)

    after = (normalizer.running_mean, normalizer.running_var, normalizer.count)
    for expected_state, actual_state in zip(before, after, strict=True):
        torch.testing.assert_close(actual_state, expected_state, rtol=0.0, atol=0.0)
    expected_physical = torch.sqrt(before[1].float() + normalizer.epsilon) * normalized_value + before[0].float()
    torch.testing.assert_close(physical_value, expected_physical)
    assert model.training and normalizer.training  # helper必须恢复调用前train/eval生命周期


def test_head_gradient_probe_separates_per_asset_objectives_without_parameter_grad_side_effect() -> None:
    r"""Probe用autograd.grad形成$[A,P]$，不得写入主optimizer的``parameter.grad``。"""

    parameter = torch.nn.Parameter(torch.tensor((1.0, -2.0)))
    features = torch.tensor(((1.0, 0.0), (3.0, 0.0), (0.0, 2.0), (0.0, 4.0)))
    labels = torch.tensor((0, 0, 1, 1))
    prediction = features @ parameter
    objective = prediction.square()
    gradients = PalmRotationPpoAgent._per_asset_gradient_matrix(
        objective,
        labels,
        (parameter,),
        asset_count=2,
    )

    # Asset0: mean((x*w)^2)在w=(1,-2)处梯度为(10,0)；asset1为(0,-40)。
    torch.testing.assert_close(gradients, torch.tensor(((10.0, 0.0), (0.0, -40.0))))
    assert parameter.grad is None


def test_optimizer_scalar_drain_transfers_only_update_level_means() -> None:
    r"""80份GPU scalar之和只在update drain转成六个Python值，且保持原归约分母。"""

    fake = SimpleNamespace(
        _optimizer_step_count=20,
        _optimizer_microbatch_count=80,
        _gradient_microbatch_index=80,
        _gradient_accumulation_steps=4,
        _optimizer_scalar_sums={
            "actor_loss": torch.tensor(160.0),
            "critic_loss": torch.tensor(240.0),
            "entropy": torch.tensor(320.0),
            "policy_sigma": torch.tensor(40.0),
            "actor_grad_norm": torch.tensor(10.0),
            "critic_grad_norm": torch.tensor(30.0),
        },
    )
    result = PalmRotationPpoAgent._drain_optimizer_scalars(fake)  # type: ignore[arg-type]

    assert result["actor_loss"] == 2.0 and result["critic_loss"] == 3.0
    assert result["entropy"] == 4.0 and result["policy_sigma"] == 0.5
    assert result["actor_grad_norm"] == 0.5 and result["critic_grad_norm"] == 1.5
    assert result["optimizer_microbatches"] == 80.0 and result["optimizer_steps"] == 20.0


def test_extracted_gradient_probes_publish_without_updating_parameters(tmp_path: Path) -> None:
    r"""迁移后的探针既要能写完整证据，也不能写主grad或改变Actor参数。"""

    from anymani.distill.rl.runtime import palm_rotation_probes

    actor = torch.nn.Linear(2, 1)
    critic = torch.nn.Linear(2, 1)
    features = torch.tensor(((1.0, 0.0), (3.0, 0.0), (0.0, 2.0), (0.0, 4.0)))
    labels = torch.tensor((0, 0, 1, 1))  # 每资产恰有两个独立half样本
    actor_objective = actor(features).squeeze(-1).square()
    critic_objective = critic(features).squeeze(-1).square()
    before = tuple(parameter.detach().clone() for parameter in actor.parameters())
    fake = SimpleNamespace(
        asset_count=2,
        epoch_num=123,
        experiment_dir=str(tmp_path),
        model=SimpleNamespace(a2c_network=SimpleNamespace(package=SimpleNamespace(actor=actor))),
        _gradient_probe_parameters=lambda: (tuple(actor.parameters()), tuple(critic.parameters())),
        _per_asset_gradient_matrix=PalmRotationPpoAgent._per_asset_gradient_matrix,
    )  # 不提供optimizer；探针若尝试更新参数应直接失败

    palm_rotation_probes.run_gradient_probe(
        fake, actor_objective=actor_objective, critic_objective=critic_objective, labels=labels
    )  # type: ignore[arg-type]
    palm_rotation_probes.run_full_actor_gradient_shadow(
        fake,
        global_objective=actor_objective,
        per_asset_objective=actor_objective * torch.tensor((1.0, 1.0, 2.0, 2.0)),
        labels=labels,
        replica_halves=torch.tensor((0, 1, 0, 1)),
    )  # type: ignore[arg-type]

    assert (tmp_path / "gradient_probes/update_000123.npz").is_file()
    summary = json.loads((tmp_path / "full_gradient_shadows/update_000123.json").read_text())
    assert summary["update"] == 123 and summary["asset_count"] == 2
    assert set(summary["scopes"]) == {"global", "per_asset_rollout"}
    for parameter, reference in zip(actor.parameters(), before, strict=True):
        assert torch.equal(parameter, reference) and parameter.grad is None


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
                "arm": "residual",
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
                "arm": "residual",
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


@pytest.mark.parametrize("arm", ("direct", "direct_token"))
def test_direct_model_exposes_direct_mean_without_residual_side_channel(arm: str) -> None:
    r"""Direct rollout只交付完整authority mean，不能伪造base/residual分解。"""

    builder = PalmRotationRlGamesBuilder()
    builder.load(
        {
            "palm_rotation": {
                "arm": arm,
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
    assert "residuals" not in output
    assert isinstance(output["direct_means"], torch.Tensor)
    torch.testing.assert_close(output["direct_means"], output["mus"], rtol=0.0, atol=0.0)


def test_direct_mechanism_metrics_use_immutable_rollout_mean_across_mini_epochs() -> None:
    r"""Direct机制恒等式必须锚定rollout mean，不能误用mini-epoch间会原地更新的KL参考。"""

    rollout_mean = torch.tensor(((0.25, -0.5), (0.75, 0.0)))  # $\mu^{rollout}$，采样动作时的冻结均值
    direct_mean = rollout_mean.clone()  # Direct定义要求$\mu^{direct}=\mu^{rollout}$逐位成立
    active_mask = torch.ones_like(rollout_mean, dtype=torch.bool)  # 两个样本的两个关节均参与统计
    mutable_kl_reference = rollout_mean.clone()  # 模拟rl_games dataset中的`mu`存储
    frozen_rollout_mean = rollout_mean.detach().clone()  # 新合同要求独立storage，跨mini-epoch保持不变

    mutable_kl_reference.add_(0.125)  # `update_mu_sigma`会在第一轮后原地写入当前策略mean
    metrics = rollout_policy_mechanism_metrics(
        frozen_rollout_mean,
        direct_mean,
        active_mask,
        actor_arm="direct",
    )

    torch.testing.assert_close(frozen_rollout_mean, rollout_mean, rtol=0.0, atol=0.0)
    assert not torch.equal(mutable_kl_reference, frozen_rollout_mean)
    expected_rms = torch.sqrt(rollout_mean.square().mean(dim=-1))  # 每样本active-DoF RMS
    torch.testing.assert_close(metrics["policy_mean_rms"], expected_rms)
    torch.testing.assert_close(metrics["direct_mean_rms"], expected_rms)

    with pytest.raises(RuntimeError, match="direct rollout side-channel"):
        rollout_policy_mechanism_metrics(mutable_kl_reference, direct_mean, active_mask, actor_arm="direct")

    token_metrics = rollout_policy_mechanism_metrics(
        frozen_rollout_mean,
        direct_mean,
        active_mask,
        actor_arm="direct_token",
    )
    torch.testing.assert_close(token_metrics["direct_mean_rms"], expected_rms)


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
