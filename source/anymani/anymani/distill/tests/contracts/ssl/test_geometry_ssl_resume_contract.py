r"""Geometry SSL pure-pretrain resume 的科学配置与数据 lineage 合同。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from anymani.distill.ssl.config_store import compose_pretrain_cfg
from anymani.distill.ssl.experiment import EmbodimentPretrainCfg, resolved_config_dict
from anymani.distill.ssl.pretrain import _build_parser, _config_overrides
from anymani.distill.ssl.runtime.checkpointing import (
    require_resume_metadata_identity,
    require_resume_scientific_config,
    resume_scientific_config,
)
from anymani.distill.ssl.runtime.run import PretrainRun, PretrainRunCfg

pytestmark = pytest.mark.contract


def test_resume_allows_only_run_location_fields_to_change() -> None:
    r"""新 output/run/checkpoint 路径不改变科学过程，应允许 resume。"""

    checkpoint_config = _config()
    current = replace(
        checkpoint_config,
        run=replace(
            checkpoint_config.run,
            output_dir="/tmp/new-run-root",
            experiment_name="resumed-run",
            resume_checkpoint="/tmp/source/checkpoints/epoch_000003.pt",
        ),
    )

    require_resume_scientific_config(current, resolved_config_dict(checkpoint_config))


def test_completed_budget_extension_allows_only_a_strict_max_epoch_increase() -> None:
    r"""显式 extension 只放行总 epoch 上界增加，其他科学字段继续逐值相同。"""

    checkpoint_config = _config()
    current = replace(
        checkpoint_config,
        trainer=replace(checkpoint_config.trainer, max_epochs=checkpoint_config.trainer.max_epochs + 32),
        run=replace(
            checkpoint_config.run,
            resume_checkpoint="/tmp/completed/checkpoints/last.pt",
            extend_completed_run=True,
        ),
    )
    require_resume_scientific_config(
        current,
        resolved_config_dict(checkpoint_config),
        allow_completed_budget_extension=True,
    )

    changed_optimizer = replace(
        current,
        trainer=replace(
            current.trainer,
            optimizer=replace(current.trainer.optimizer, learning_rate=1.0e-4),
        ),
    )
    with pytest.raises(ValueError, match="resume scientific config mismatch"):
        require_resume_scientific_config(
            changed_optimizer,
            resolved_config_dict(checkpoint_config),
            allow_completed_budget_extension=True,
        )


def test_completed_budget_extension_rejects_equal_or_smaller_budget() -> None:
    r"""Extension flag 不能伪装成普通 recovery，也不能截短 checkpoint 声明预算。"""

    checkpoint_config = _config()
    for max_epochs in (checkpoint_config.trainer.max_epochs, checkpoint_config.trainer.max_epochs - 32):
        current = replace(
            checkpoint_config,
            trainer=replace(checkpoint_config.trainer, max_epochs=max_epochs),
            run=replace(
                checkpoint_config.run,
                resume_checkpoint="/tmp/completed/checkpoints/last.pt",
                extend_completed_run=True,
            ),
        )
        with pytest.raises(ValueError, match="strictly increase max_epochs"):
            require_resume_scientific_config(
                current,
                resolved_config_dict(checkpoint_config),
                allow_completed_budget_extension=True,
            )


def test_completed_extension_can_supersede_only_snapshot_content_hash() -> None:
    r"""源码迁移授权可接受同一 snapshot 的纯内容 SHA变化，不可换名、模块或路径。"""

    checkpoint_config = _config()
    current_config = replace(
        checkpoint_config,
        trainer=replace(checkpoint_config.trainer, max_epochs=checkpoint_config.trainer.max_epochs + 32),
        run=replace(
            checkpoint_config.run,
            resume_checkpoint="/tmp/completed/checkpoints/last.pt",
            extend_completed_run=True,
        ),
    )
    checkpoint_resolved = resolved_config_dict(checkpoint_config)
    current_resolved = resolved_config_dict(current_config)
    checkpoint_resolved["experiment_identity"] = {
        "name": "same",
        "module": "same.module",
        "path": "/same/snapshot.py",
        "sha256": "old",
    }
    current_resolved["experiment_identity"] = {
        "name": "same",
        "module": "same.module",
        "path": "/same/snapshot.py",
        "sha256": "new",
    }
    with pytest.raises(ValueError, match="experiment_identity"):
        require_resume_scientific_config(
            current_resolved,
            checkpoint_resolved,
            allow_completed_budget_extension=True,
        )
    require_resume_scientific_config(
        current_resolved,
        checkpoint_resolved,
        allow_completed_budget_extension=True,
        allow_experiment_identity_change=True,
    )
    current_resolved["experiment_identity"]["name"] = "changed"
    with pytest.raises(ValueError, match="may change only snapshot sha256"):
        require_resume_scientific_config(
            current_resolved,
            checkpoint_resolved,
            allow_completed_budget_extension=True,
            allow_experiment_identity_change=True,
        )


@pytest.mark.parametrize("section", ["query", "max_epochs", "num_minibatches", "mini_epochs", "seed", "fairgrad"])
def test_resume_rejects_query_or_training_budget_drift(section: str) -> None:
    r"""query 测度、新数据批数、复用次数或根 seed 改变都不是同一训练轨迹。"""

    checkpoint_config = _config()
    if section == "query":
        current = replace(
            checkpoint_config,
            method=replace(
                checkpoint_config.method,
                representation=replace(
                    checkpoint_config.method.representation,
                    query=replace(checkpoint_config.method.representation.query, shell_offset_max_m=0.003),
                ),
            ),
        )
    elif section == "max_epochs":
        current = replace(checkpoint_config, trainer=replace(checkpoint_config.trainer, max_epochs=64))
    elif section == "num_minibatches":
        current = replace(
            checkpoint_config,
            trainer=replace(checkpoint_config.trainer, num_minibatches=64),
        )
    elif section == "mini_epochs":
        current = replace(checkpoint_config, trainer=replace(checkpoint_config.trainer, mini_epochs=3))
    elif section == "seed":
        current = replace(checkpoint_config, run=replace(checkpoint_config.run, seed=19))
    else:
        current = replace(
            checkpoint_config,
            method=replace(
                checkpoint_config.method,
                fairgrad=replace(checkpoint_config.method.fairgrad, near_opposition_tolerance=1.0e-5),
            ),
        )

    with pytest.raises(ValueError, match="resume scientific config mismatch"):
        require_resume_scientific_config(current, resolved_config_dict(checkpoint_config))


def test_resume_rejects_superseded_extra_objective_config() -> None:
    r"""旧完整检查点多出的损失字段改变科学身份，不能恢复到双目标训练。"""

    current = _config()
    checkpoint_resolved = resolved_config_dict(current)
    objectives = checkpoint_resolved["method"]["objectives"]
    objectives["sobolev"] = {"weight": 1.0}
    objectives["chain"] = {"weight": 1.0}

    with pytest.raises(ValueError, match="resume scientific config mismatch"):
        require_resume_scientific_config(current, checkpoint_resolved)


@pytest.mark.parametrize(
    "field",
    [
        "code_revision",
        "package_version",
        "objective_formula",
        "fairgrad_formula",
        "parameter_partition",
        "source_artifact",
        "worktree_fingerprint",
    ],
)
def test_resume_rejects_code_formula_source_or_worktree_drift(field: str) -> None:
    """Resolved config 未变化时，实际实现与 source producer 漂移仍必须 fail closed。"""

    metadata = {
        "code_revision": "revision-a",
        "package_version": "0.7.5",
        "geometry_semantics_schema": "5.0.0",
        "declared_objective": {"density": 1.0, "kappa": 1.0},
        "objective_formula": {"density": "raw-mse", "kappa": "scaled-mse"},
        "fairgrad_formula": {"alpha": 1.0},
        "parameter_partition": {"shared_encoder": 10},
        "source_artifact": {"mode": "readonly", "producer": "sm_120"},
        "worktree_dirty": True,
        "worktree_fingerprint": "fingerprint-a",
    }
    changed = dict(metadata)
    changed[field] = "changed"
    with pytest.raises(ValueError, match="metadata identity mismatch"):
        require_resume_metadata_identity(changed, metadata)


def test_resume_can_explicitly_accept_validated_dirty_worktree_change() -> None:
    r"""源码修复经真实 sanity 验证后，可放行 fingerprint 变化但仍保留其他 identity gates。"""

    metadata = {
        "code_revision": "revision-a",
        "package_version": "0.7.5",
        "geometry_semantics_schema": "5.0.0",
        "declared_objective": {"density": 1.0, "kappa": 1.0},
        "objective_formula": {"density": "raw-mse", "kappa": "scaled-mse"},
        "fairgrad_formula": {"alpha": 1.0},
        "parameter_partition": {"shared_encoder": 10},
        "source_artifact": {"mode": "readonly", "producer": "sm_120"},
        "worktree_dirty": True,
        "worktree_fingerprint": "fingerprint-a",
    }
    changed = dict(
        metadata,
        code_revision="revision-after-validated-fix",
        worktree_fingerprint="fingerprint-after-validated-fix",
    )

    require_resume_metadata_identity(changed, metadata, allow_worktree_change=True)

    changed["fairgrad_formula"] = {"alpha": 2.0}
    with pytest.raises(ValueError, match="fairgrad_formula"):
        require_resume_metadata_identity(changed, metadata, allow_worktree_change=True)


def test_completed_extension_can_explicitly_migrate_package_lineage_only() -> None:
    r"""跨 release 的 matched extension 可迁移 package/code identity，公式与 source 仍严格冻结。"""

    metadata = {
        "code_revision": "old-revision",
        "package_version": "0.7.5",
        "geometry_semantics_schema": "5.0.0",
        "declared_objective": {"density": 1.0, "kappa": 1.0},
        "objective_formula": {"density": "raw-mse", "kappa": "scaled-mse"},
        "fairgrad_formula": {"alpha": 1.0},
        "parameter_partition": {"shared_encoder": 10},
        "source_artifact": {"mode": "readonly", "producer": "sm_120"},
        "worktree_dirty": True,
        "worktree_fingerprint": "old-fingerprint",
    }
    current = dict(
        metadata,
        code_revision="extension-revision",
        package_version="0.8.1",
        worktree_fingerprint="extension-fingerprint",
    )
    require_resume_metadata_identity(
        current,
        metadata,
        allow_worktree_change=True,
        extension_source_package_version="0.7.5",
    )
    with pytest.raises(ValueError, match="explicit extension source package"):
        require_resume_metadata_identity(
            current,
            metadata,
            allow_worktree_change=True,
            extension_source_package_version="0.7.4",
        )
    current["source_artifact"] = {"mode": "readonly", "producer": "changed"}
    with pytest.raises(ValueError, match="source_artifact"):
        require_resume_metadata_identity(
            current,
            metadata,
            allow_worktree_change=True,
            extension_source_package_version="0.7.5",
        )


def test_resume_worktree_change_still_requires_dirty_state_match() -> None:
    r"""显式源码修复授权不能把 clean checkpoint 恢复到 dirty worktree。"""

    metadata = {
        "code_revision": "revision-a",
        "package_version": "0.7.5",
        "geometry_semantics_schema": "5.0.0",
        "declared_objective": {"density": 1.0, "kappa": 1.0},
        "objective_formula": {"density": "raw-mse", "kappa": "scaled-mse"},
        "fairgrad_formula": {"alpha": 1.0},
        "parameter_partition": {"shared_encoder": 10},
        "source_artifact": {"mode": "readonly", "producer": "sm_120"},
        "worktree_dirty": False,
        "worktree_fingerprint": "",
    }
    changed = dict(metadata, worktree_dirty=True, worktree_fingerprint="fingerprint-after-fix")

    with pytest.raises(ValueError, match="worktree_dirty"):
        require_resume_metadata_identity(changed, metadata, allow_worktree_change=True)


def test_allow_worktree_change_is_runtime_resume_policy_not_scientific_identity() -> None:
    r"""CLI 的源码修复授权只控制 resume gate，不改变冻结的实验配置。"""

    args = _build_parser().parse_args(["--allow-worktree-change"])
    overrides = _config_overrides(args)
    assert "run.allow_worktree_change=True" in overrides

    baseline = resolved_config_dict(compose_pretrain_cfg())
    changed = dict(baseline)
    changed["run"] = dict(baseline["run"], allow_worktree_change=True)
    assert resume_scientific_config(changed) == resume_scientific_config(baseline)


def test_completed_extension_is_explicit_runtime_policy_not_scientific_identity() -> None:
    r"""CLI extension flag 只授权已完成前缀迁移，不混入其余科学配置比较。"""

    args = _build_parser().parse_args(
        ["--extend-completed-run", "--extension-source-package-version", "0.7.5"]
    )
    overrides = _config_overrides(args)
    assert "run.extend_completed_run=True" in overrides
    assert "run.extension_source_package_version=0.7.5" in overrides
    baseline = resolved_config_dict(compose_pretrain_cfg())
    changed = dict(baseline)
    changed["run"] = dict(baseline["run"], extend_completed_run=True)
    assert resume_scientific_config(changed) == resume_scientific_config(baseline)


def test_completed_extension_creates_child_but_recovers_incomplete_child_in_place(tmp_path: Path) -> None:
    r"""COMPLETE source 不可写；extension child 的 recovery checkpoint 必须回到自身目录。"""

    source = tmp_path / "source"
    source_checkpoint = source / "checkpoints" / "last.pt"
    source_checkpoint.parent.mkdir(parents=True)
    source_checkpoint.write_bytes(b"checkpoint")
    (source / "COMPLETE").write_text("complete\n", encoding="ascii")
    child_root = tmp_path / "children"
    initial = PretrainRun(
        PretrainRunCfg(
            output_dir=str(child_root),
            experiment_name="extension",
            resume_checkpoint=str(source_checkpoint),
            extend_completed_run=True,
            source_cache_mode="off",
        )
    ).prepare_output_dir()
    assert initial != source
    assert initial.parent == child_root / "extension"
    assert (initial / "INCOMPLETE").is_file()

    recovery = initial / "checkpoints" / "recovery.pt"
    recovery.parent.mkdir(parents=True, exist_ok=True)
    recovery.write_bytes(b"recovery")
    resumed = PretrainRun(
        PretrainRunCfg(
            output_dir=str(child_root),
            experiment_name="extension",
            resume_checkpoint=str(recovery),
            extend_completed_run=True,
            source_cache_mode="off",
        )
    ).prepare_output_dir()
    assert resumed == initial


def _config() -> EmbodimentPretrainCfg:
    """从 ConfigStore 恢复 schema 9 实验，避免测试自己复制 concrete defaults。"""

    return compose_pretrain_cfg()
