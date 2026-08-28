r"""Geometry SSL pure-pretrain resume 的科学配置与数据 lineage 合同。"""

from __future__ import annotations

from dataclasses import replace

import pytest
from anymani.distill.ssl.config_store import compose_pretrain_cfg
from anymani.distill.ssl.experiment import EmbodimentPretrainCfg, resolved_config_dict
from anymani.distill.ssl.pretrain import _build_parser, _config_overrides
from anymani.distill.ssl.runtime.checkpointing import (
    require_resume_metadata_identity,
    require_resume_scientific_config,
    resume_scientific_config,
)

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
    changed = dict(metadata, worktree_fingerprint="fingerprint-after-validated-fix")

    require_resume_metadata_identity(changed, metadata, allow_worktree_change=True)

    changed["fairgrad_formula"] = {"alpha": 2.0}
    with pytest.raises(ValueError, match="fairgrad_formula"):
        require_resume_metadata_identity(changed, metadata, allow_worktree_change=True)


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


def _config() -> EmbodimentPretrainCfg:
    """从 ConfigStore 恢复 schema 9 实验，避免测试自己复制 concrete defaults。"""

    return compose_pretrain_cfg()
