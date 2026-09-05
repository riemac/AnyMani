r"""Actor-only checkpoint迁移的namespace、兼容性与重置边界合同。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from anymani.distill.models.palm_rotation_policy import PalmRotationActorCritic
from anymani.distill.rl.runtime.palm_rotation_warm_start import (
    ACTOR_CHECKPOINT_PREFIX,
    inspect_actor_init_checkpoint,
    inspect_resumed_actor_warm_start,
    load_actor_init_checkpoint,
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
    assert not should_load_actor_init_checkpoint(
        actor_init_path="", warm_start=evidence, full_checkpoint_resume=True
    )
    assert not should_load_actor_init_checkpoint(actor_init_path="", warm_start=None, full_checkpoint_resume=True)
    with pytest.raises(ValueError, match="jointly present"):
        should_load_actor_init_checkpoint(actor_init_path="", warm_start=evidence, full_checkpoint_resume=False)
    with pytest.raises(ValueError, match="cannot also"):
        should_load_actor_init_checkpoint(
            actor_init_path="parent.pth", warm_start=evidence, full_checkpoint_resume=True
        )
