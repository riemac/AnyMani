r"""Geometry SSL resume 的科学配置与 checkpoint-selection lineage 合同。"""

from __future__ import annotations

from dataclasses import replace

import pytest
from anymani.distill.ssl.config_store import compose_pretrain_cfg
from anymani.distill.ssl.experiment import EmbodimentPretrainCfg, resolved_config_dict
from anymani.distill.ssl.runtime.checkpointing import (
    require_resume_calibration_hash,
    require_resume_scientific_config,
    restore_validation_selection_state,
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


@pytest.mark.parametrize("section", ["query", "max_epochs", "num_minibatches", "mini_epochs", "seed"])
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
    else:
        current = replace(checkpoint_config, run=replace(checkpoint_config.run, seed=19))

    with pytest.raises(ValueError, match="resume scientific config mismatch"):
        require_resume_scientific_config(current, resolved_config_dict(checkpoint_config))


def test_resume_rejects_superseded_five_term_objective_config() -> None:
    r"""旧完整检查点多出的损失字段改变科学身份，不能恢复到三项训练。"""

    current = _config()
    checkpoint_resolved = resolved_config_dict(current)
    objectives = checkpoint_resolved["method"]["objectives"]
    objectives["sobolev"] = {"weight": 1.0}
    objectives["chain"] = {"weight": 1.0}

    with pytest.raises(ValueError, match="resume scientific config mismatch"):
        require_resume_scientific_config(current, checkpoint_resolved)


def test_resume_rejects_calibration_artifact_content_drift() -> None:
    r"""artifact 路径不变但内容变化时，resume 仍须按 checkpoint 保存的 SHA-256 拒绝。"""

    with pytest.raises(ValueError, match="calibration artifact hash"):
        require_resume_calibration_hash("new-hash", {"calibration_artifact_hash": "checkpoint-hash"})


def test_resume_restores_initial_baseline_best_score_and_history() -> None:
    r"""中断后 checkpoint selection 必须沿用初始化分母与 historical best。"""

    initial = {
        "unseen_variant_set": {"density": 0.2, "kappa": 0.01, "derived_field": 2.0},
        "unseen_mother": {"density": 0.3, "kappa": 0.02, "derived_field": 2.5},
    }
    history = [{"epoch": 8, "score": 0.8, "metrics": {"unseen_variant_set": {"density": 0.1}}}]

    restored = restore_validation_selection_state(
        {
            "initial_validation_metrics": initial,
            "best_validation_score": 0.8,
            "selection_history": history,
        }
    )

    assert restored == (initial, None, 0.8, history)


def test_resume_rejects_best_score_without_selection_history() -> None:
    r"""历史 best 没有对应评估轨迹时，不得静默继承。"""

    with pytest.raises(ValueError, match="inconsistent"):
        restore_validation_selection_state(
            {
                "initial_validation_metrics": {
                    "unseen_variant_set": {"density": 0.2, "kappa": 0.01, "derived_field": 2.0},
                    "unseen_mother": {"density": 0.3, "kappa": 0.02, "derived_field": 2.5},
                },
                "best_validation_score": 0.8,
                "selection_history": [],
            }
        )


def _config() -> EmbodimentPretrainCfg:
    """从 ConfigStore 恢复 schema 6 实验，避免测试自己复制 concrete defaults。"""

    return compose_pretrain_cfg()
