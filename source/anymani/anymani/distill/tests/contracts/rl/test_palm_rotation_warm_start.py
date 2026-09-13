r"""Actor-only checkpoint迁移的namespace、兼容性与重置边界合同。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from anymani.distill.models.palm_rotation_policy import PalmRotationActorCritic
from anymani.distill.rl.runtime.palm_rotation_warm_start import (
    ACTOR_CHECKPOINT_PREFIX,
    inspect_actor_init_checkpoint,
    inspect_resumed_actor_warm_start,
    load_actor_init_checkpoint,
    load_critic_init_checkpoint,
    should_load_actor_init_checkpoint,
)

RETAINED_SHA = "a" * 64


def _provider(sha: str = RETAINED_SHA) -> dict:
    r"""构造只保留retained artifact身份的最小provider fixture。"""

    return {"retained_artifact": {"sha256": sha}}


def _checkpoint(path: Path, *, arm: str = "residual", history: str = "tcn") -> tuple[Path, PalmRotationActorCritic]:
    r"""保存含Actor、无关Critic与optimizer哨兵的完整checkpoint fixture。"""

    package = PalmRotationActorCritic(arm=arm)  # type: ignore[arg-type]
    with torch.no_grad():
        for parameter_index, parameter in enumerate(package.actor.parameters()):
            parameter.fill_(0.001 * (parameter_index + 1))  # 每个tensor可逐项识别的确定性source值
    model = {
        **{f"{ACTOR_CHECKPOINT_PREFIX}{key}": value.clone() for key, value in package.actor.state_dict().items()},
        "a2c_network.package.critic.sentinel": torch.tensor(123.0),
    }
    torch.save(
        {
            "model": model,
            "optimizer": {"must_not_load": True},
            "anymani_critic_optimizer": {"must_not_load": True},
            "anymani_identity": {
                "identity_schema_version": "3.0.0",
                "identity_digest": "b" * 64,
                "task_id": "source-task",
                "task_contract": {"primary_success": "historical-positive-net-rotation-frontier"},
                "policy": {"arm": arm},
                "training": {
                    "history_encoder": history,
                    "cohort_id": "source-cohort",
                    "cohort_lock_sha256": "c" * 64,
                },
                "geometry_provider": _provider(),
            },
        },
        path,
    )
    return path, package


def test_actor_only_warm_start_loads_exact_namespace_and_leaves_critic_fresh(tmp_path: Path) -> None:
    r"""Source Actor逐tensor恢复；target Critic保持初始化值，证据显式列出全部重置组件。"""

    checkpoint, source = _checkpoint(tmp_path / "source.pth")
    target = PalmRotationActorCritic(arm="residual")
    critic_before = {key: value.clone() for key, value in target.critic.state_dict().items()}
    evidence = inspect_actor_init_checkpoint(
        checkpoint,
        target_arm="residual",
        target_history_encoder="tcn",
        target_provider_identity=_provider(),
    )
    loaded = load_actor_init_checkpoint(
        target.actor,
        checkpoint,
        expected_checkpoint_sha256=evidence["checkpoint_sha256"],
    )

    assert len(loaded) == evidence["loaded_tensor_count"]
    for key, value in source.actor.state_dict().items():
        torch.testing.assert_close(target.actor.state_dict()[key], value, rtol=0.0, atol=0.0)
    for key, value in critic_before.items():
        torch.testing.assert_close(target.critic.state_dict()[key], value, rtol=0.0, atol=0.0)
    assert set(evidence["reset_components"]) >= {
        "critic",
        "actor_optimizer",
        "critic_optimizer",
        "value_normalizer",
        "reward_curriculum",
        "adr_state",
    }


def test_actor_init_sigma_override_changes_only_exploration_parameter(tmp_path: Path) -> None:
    r"""显式sigma在权重加载后生效，动作中心的全部权重与source逐值一致。"""

    checkpoint, source = _checkpoint(tmp_path / "source.pth")
    target = PalmRotationActorCritic(arm="residual")
    evidence = inspect_actor_init_checkpoint(
        checkpoint,
        target_arm="residual",
        target_history_encoder="tcn",
        target_provider_identity=_provider(),
        actor_init_sigma=0.15,
    )
    load_actor_init_checkpoint(
        target.actor,
        checkpoint,
        expected_checkpoint_sha256=evidence["checkpoint_sha256"],
        actor_init_sigma=0.15,
    )
    assert evidence["actor_init_sigma"] == 0.15
    assert "actor_exploration" in evidence["reset_components"]
    assert target.actor.global_log_std.exp().item() == pytest.approx(0.15)
    for key, value in source.actor.state_dict().items():
        if key != "global_log_std":
            torch.testing.assert_close(target.actor.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("sigma", [0.0, -0.1, float("nan"), float("inf")])
def test_actor_init_rejects_invalid_sigma_before_loading_source(tmp_path: Path, sigma: float) -> None:
    r"""非法分布尺度在读取checkpoint之前拒绝，避免昂贵初始化后才发现参数错误。"""

    with pytest.raises(ValueError, match="sigma"):
        inspect_actor_init_checkpoint(
            tmp_path / "not-loaded.pth",
            target_arm="residual",
            target_history_encoder="tcn",
            target_provider_identity=_provider(),
            actor_init_sigma=sigma,
        )


def test_actor_init_sigma_ceiling_rejection_preserves_target_parameters(tmp_path: Path) -> None:
    r"""超过目标模型探索上界的请求明确拒绝，而非静默clip或先写入部分source参数。"""

    checkpoint, _source = _checkpoint(tmp_path / "source.pth")
    target = PalmRotationActorCritic(arm="residual")
    before = {key: value.clone() for key, value in target.actor.state_dict().items()}
    evidence = inspect_actor_init_checkpoint(
        checkpoint,
        target_arm="residual",
        target_history_encoder="tcn",
        target_provider_identity=_provider(),
    )
    with pytest.raises(ValueError, match="ceiling"):
        load_actor_init_checkpoint(
            target.actor,
            checkpoint,
            expected_checkpoint_sha256=evidence["checkpoint_sha256"],
            actor_init_sigma=0.8,
        )
    for key, value in before.items():
        torch.testing.assert_close(target.actor.state_dict()[key], value, rtol=0, atol=0)


def test_actor_only_warm_start_rejects_arm_history_and_retained_encoder_mismatch(tmp_path: Path) -> None:
    r"""Shape可能偶合也不能跨arm、History归纳偏置或N040 artifact静默迁移。"""

    checkpoint, _source = _checkpoint(tmp_path / "source.pth")
    common = {"path": checkpoint, "target_provider_identity": _provider()}
    with pytest.raises(ValueError, match="arm mismatch"):
        inspect_actor_init_checkpoint(**common, target_arm="direct", target_history_encoder="tcn")
    with pytest.raises(ValueError, match="History30 encoder mismatch"):
        inspect_actor_init_checkpoint(**common, target_arm="residual", target_history_encoder="raw_stack")
    with pytest.raises(ValueError, match="different retained N040"):
        inspect_actor_init_checkpoint(
            checkpoint,
            target_arm="residual",
            target_history_encoder="tcn",
            target_provider_identity=_provider("d" * 64),
        )


def test_full_resume_preserves_actor_warm_start_lineage_without_reloading_parent(tmp_path: Path) -> None:
    r"""Target checkpoint独立携带初始化血缘；full resume不需要再次访问source Actor。"""

    checkpoint, _source = _checkpoint(tmp_path / "target.pth")
    document = torch.load(checkpoint, map_location="cpu", weights_only=False)
    evidence = {
        "schema_version": "1.0.0",
        "checkpoint_path": "deleted-parent.pth",
        "checkpoint_sha256": "d" * 64,
        "reset_components": ["critic", "actor_optimizer"],
    }
    document["anymani_identity"]["training"]["actor_warm_start"] = evidence
    torch.save(document, checkpoint)

    assert inspect_resumed_actor_warm_start(checkpoint) == evidence


def test_actor_init_mode_distinguishes_fresh_and_full_resume() -> None:
    r"""只有fresh warm-start读取parent；full resume始终等待target checkpoint统一恢复。"""

    evidence = {"schema_version": "1.0.0"}
    assert should_load_actor_init_checkpoint(
        actor_init_path="parent.pth", warm_start=evidence, full_checkpoint_resume=False
    )
    assert not should_load_actor_init_checkpoint(actor_init_path="", warm_start=evidence, full_checkpoint_resume=True)
    assert not should_load_actor_init_checkpoint(actor_init_path="", warm_start=None, full_checkpoint_resume=True)
    with pytest.raises(ValueError, match="jointly present"):
        should_load_actor_init_checkpoint(actor_init_path="", warm_start=evidence, full_checkpoint_resume=False)
    with pytest.raises(ValueError, match="cannot also"):
        should_load_actor_init_checkpoint(
            actor_init_path="parent.pth", warm_start=evidence, full_checkpoint_resume=True
        )


def test_optional_critic_transfer_retains_value_units_but_not_optimizer_or_actor(tmp_path: Path) -> None:
    r"""Critic权重与value统计成对继承，计数不是fresh的1；该调用不写Actor或optimizer。"""

    path, source = _checkpoint(tmp_path / "critic-source.pth")
    document = torch.load(path, map_location="cpu", weights_only=False)
    document["anymani_identity"]["task_contract"]["critic_task_state"] = (
        "axis-goal-error-max-positive-net-and-current-net"
    )
    document["model"].pop("a2c_network.package.critic.sentinel")
    document["model"].update(
        {f"a2c_network.package.critic.{key}": value.clone() for key, value in source.critic.state_dict().items()}
    )
    normalizer = torch.nn.Module()  # 只需验证state_dict语义，不启动任何仿真或PPO更新
    for key, value in (("running_mean", 12.0), ("running_var", 25.0), ("count", 10001.0)):
        normalizer.register_buffer(key, torch.tensor(value, dtype=torch.float64))
        document["model"][f"value_mean_std.{key}"] = torch.tensor(value, dtype=torch.float64)
    torch.save(document, path)
    target = PalmRotationActorCritic(arm="residual")
    actor_before = {key: value.clone() for key, value in target.actor.state_dict().items()}
    for buffer in normalizer.buffers():
        buffer.zero_()
    model = SimpleNamespace(
        a2c_network=SimpleNamespace(package=target), normalize_value=True, value_mean_std=normalizer
    )
    evidence = inspect_actor_init_checkpoint(
        path,
        target_arm="residual",
        target_history_encoder="tcn",
        target_provider_identity=_provider(),
        initialize_critic=True,
    )
    assert evidence["value_normalizer_initial_count"] == 10001.0
    assert "critic" not in evidence["reset_components"] and "value_normalizer" not in evidence["reset_components"]
    assert {"actor_optimizer", "critic_optimizer"} <= set(evidence["reset_components"])
    load_critic_init_checkpoint(model, path, expected_checkpoint_sha256=evidence["checkpoint_sha256"])
    for key, value in source.critic.state_dict().items():
        torch.testing.assert_close(target.critic.state_dict()[key], value, rtol=0, atol=0)
    for key, value in actor_before.items():
        torch.testing.assert_close(target.actor.state_dict()[key], value, rtol=0, atol=0)
    assert normalizer.state_dict()["count"].item() == 10001.0
    assert normalizer.state_dict()["running_var"].item() == 25.0
    model.normalize_value = False
    with pytest.raises(ValueError, match="value normalization"):
        load_critic_init_checkpoint(model, path, expected_checkpoint_sha256=evidence["checkpoint_sha256"])


def test_critic_transfer_rejects_same_shape_with_different_privileged_semantics(tmp_path: Path) -> None:
    r"""Actor可迁移不代表Critic状态语义可迁移。"""

    path, _ = _checkpoint(tmp_path / "source.pth")
    with pytest.raises(ValueError, match="privileged task semantics"):
        inspect_actor_init_checkpoint(
            path,
            target_arm="residual",
            target_history_encoder="tcn",
            target_provider_identity=_provider(),
            initialize_critic=True,
        )


def test_recovery_exploration_warm_start_requires_explicit_adaptation(tmp_path: Path) -> None:
    r"""参数形状相同也不能隐式更换探索规则；显式分支同时记录源和目标。"""
    path, source = _checkpoint(tmp_path / "source.pth")
    kwargs = dict(target_arm="residual", target_history_encoder="tcn", target_provider_identity=_provider())
    with pytest.raises(ValueError, match="recovery"):
        inspect_actor_init_checkpoint(path, **kwargs, target_recovery_sigma_floor=0.6)
    evidence = inspect_actor_init_checkpoint(
        path, **kwargs, target_recovery_sigma_floor=0.6, allow_recovery_exploration_adaptation=True
    )
    adaptation = evidence["recovery_exploration_adaptation"]
    assert adaptation["source_sigma_floor"] is None
    assert adaptation["target_sigma_floor"] == 0.6
    assert evidence["loaded_tensor_count"] == len(source.actor.state_dict())
    document = torch.load(path, weights_only=False)
    document["anymani_identity"]["training"]["recovery_sigma_floor"] = 0.6
    torch.save(document, path)
    with pytest.raises(ValueError, match="recovery"):
        inspect_actor_init_checkpoint(path, **kwargs)
    same_rule = inspect_actor_init_checkpoint(path, **kwargs, target_recovery_sigma_floor=0.6)
    assert "recovery_exploration_adaptation" not in same_rule


def test_phase_clock_actor_migration_zero_initializes_only_new_adapter(tmp_path: Path) -> None:
    r"""旧Direct Actor迁移到43步phase clock时只新增零权重，source均值权重逐值不变。"""

    path, source = _checkpoint(tmp_path / "direct-source.pth", arm="direct_token")
    target = PalmRotationActorCritic(arm="direct_token")
    target.actor.phase_contextual_adapter = torch.nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        target.actor.phase_contextual_adapter.weight.fill_(9.0)
    evidence = inspect_actor_init_checkpoint(
        path,
        target_arm="direct_token",
        target_history_encoder="tcn",
        target_provider_identity=_provider(),
        target_phase_period_steps=43,
        allow_phase_clock_adaptation=True,
    )
    adaptation = evidence["phase_clock_adaptation"]
    assert adaptation["source_period_steps"] is None
    assert adaptation["target_period_steps"] == 43
    assert adaptation["new_actor_key"] == "phase_contextual_adapter.weight"
    assert adaptation["new_critic_key"] == "phase_readout_adapter.weight"
    assert adaptation["zero_initialized"] is True
    loaded = load_actor_init_checkpoint(
        target.actor,
        path,
        expected_checkpoint_sha256=evidence["checkpoint_sha256"],
        allow_phase_clock_adaptation=True,
    )
    assert len(loaded) == evidence["loaded_tensor_count"] == len(source.actor.state_dict())
    assert torch.count_nonzero(target.actor.phase_contextual_adapter.weight) == 0
    for key, value in source.actor.state_dict().items():
        torch.testing.assert_close(target.actor.state_dict()[key], value, rtol=0, atol=0)


def test_phase_clock_actor_migration_requires_explicit_declaration(tmp_path: Path) -> None:
    r"""周期变化不能靠shape相同的旧路径静默通过。"""

    path, _source = _checkpoint(tmp_path / "direct-source.pth", arm="direct_token")
    with pytest.raises(ValueError, match="explicit adaptation"):
        inspect_actor_init_checkpoint(
            path,
            target_arm="direct_token",
            target_history_encoder="tcn",
            target_provider_identity=_provider(),
            target_phase_period_steps=43,
        )


@pytest.mark.parametrize("mutation", ("missing_old", "unexpected_old"))
def test_phase_clock_migration_does_not_swallow_other_actor_key_differences(tmp_path: Path, mutation: str) -> None:
    r"""phase例外只覆盖唯一新adapter，旧权重缺失或source多键仍严格失败。"""

    path, _source = _checkpoint(tmp_path / f"direct-{mutation}.pth", arm="direct_token")
    document = torch.load(path, map_location="cpu", weights_only=False)
    actor_keys = [key for key in document["model"] if key.startswith(ACTOR_CHECKPOINT_PREFIX)]
    if mutation == "missing_old":
        document["model"].pop(actor_keys[0])
    else:
        document["model"][f"{ACTOR_CHECKPOINT_PREFIX}unexpected.weight"] = torch.ones(1)
    torch.save(document, path)
    evidence = inspect_actor_init_checkpoint(
        path,
        target_arm="direct_token",
        target_history_encoder="tcn",
        target_provider_identity=_provider(),
        target_phase_period_steps=43,
        allow_phase_clock_adaptation=True,
    )
    target = PalmRotationActorCritic(arm="direct_token")
    target.actor.phase_contextual_adapter = torch.nn.Linear(3, 1, bias=False)
    with pytest.raises(ValueError, match="state mismatch"):
        load_actor_init_checkpoint(
            target.actor,
            path,
            expected_checkpoint_sha256=evidence["checkpoint_sha256"],
            allow_phase_clock_adaptation=True,
        )


def test_phase_clock_critic_migration_zero_initializes_adapter_and_copies_rms(tmp_path: Path) -> None:
    r"""Critic phase readout的唯一缺项置零，source Critic与value RMS仍严格复制。"""

    path, source = _checkpoint(tmp_path / "direct-critic-source.pth", arm="direct_token")
    document = torch.load(path, map_location="cpu", weights_only=False)
    document["anymani_identity"]["task_contract"]["critic_task_state"] = (
        "axis-goal-error-max-positive-net-and-current-net"
    )
    document["model"].pop("a2c_network.package.critic.sentinel")
    document["model"].update(
        {f"a2c_network.package.critic.{key}": value.clone() for key, value in source.critic.state_dict().items()}
    )
    for key, value in (("running_mean", 12.0), ("running_var", 25.0), ("count", 10001.0)):
        document["model"][f"value_mean_std.{key}"] = torch.tensor(value, dtype=torch.float64)
    torch.save(document, path)

    target = PalmRotationActorCritic(arm="direct_token")
    target.critic.phase_readout_adapter = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        target.critic.phase_readout_adapter.weight.fill_(8.0)
    normalizer = torch.nn.Module()
    for key, value in (("running_mean", 0.0), ("running_var", 1.0), ("count", 1.0)):
        normalizer.register_buffer(key, torch.tensor(value, dtype=torch.float64))
    model = SimpleNamespace(
        a2c_network=SimpleNamespace(package=target), normalize_value=True, value_mean_std=normalizer
    )
    evidence = inspect_actor_init_checkpoint(
        path,
        target_arm="direct_token",
        target_history_encoder="tcn",
        target_provider_identity=_provider(),
        initialize_critic=True,
        target_phase_period_steps=43,
        allow_phase_clock_adaptation=True,
    )
    load_critic_init_checkpoint(
        model,
        path,
        expected_checkpoint_sha256=evidence["checkpoint_sha256"],
        allow_phase_clock_adaptation=True,
    )
    assert torch.count_nonzero(target.critic.phase_readout_adapter.weight) == 0
    for key, value in source.critic.state_dict().items():
        torch.testing.assert_close(target.critic.state_dict()[key], value, rtol=0, atol=0)
    assert normalizer.state_dict()["count"].item() == 10001.0
    assert normalizer.state_dict()["running_var"].item() == 25.0
