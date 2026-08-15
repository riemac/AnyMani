r"""Geometry SSL resume 的科学配置与 checkpoint-selection lineage 合同。"""

from __future__ import annotations

from dataclasses import replace

import pytest
from anymani.distill.ssl.config import GeometrySSLExperimentCfg, resolved_config_dict
from anymani.distill.ssl.runtime.checkpointing import (
    require_resume_scientific_config,
    restore_validation_selection_state,
)

pytestmark = pytest.mark.contract


def test_resume_allows_only_run_location_fields_to_change() -> None:
    r"""新 output/run/checkpoint 路径不改变科学过程，应允许 resume。"""

    checkpoint_config = GeometrySSLExperimentCfg()
    current = replace(
        checkpoint_config,
        run=replace(
            checkpoint_config.run,
            output_dir="/tmp/new-run-root",
            experiment_name="resumed-run",
            resume_checkpoint="/tmp/source/checkpoints/step_00000003.pt",
        ),
    )

    require_resume_scientific_config(current, resolved_config_dict(checkpoint_config))


@pytest.mark.parametrize("section", ["query", "coverage"])
def test_resume_rejects_query_or_q_budget_drift(section: str) -> None:
    r"""query 测度或每资产 q coverage 改变都不是同一训练轨迹。"""

    checkpoint_config = GeometrySSLExperimentCfg()
    if section == "query":
        current = replace(
            checkpoint_config,
            representation=replace(
                checkpoint_config.representation,
                query=replace(checkpoint_config.representation.query, shell_offset_max_m=0.003),
            ),
        )
    else:
        current = replace(
            checkpoint_config,
            protocol=replace(
                checkpoint_config.protocol,
                coverage=replace(checkpoint_config.protocol.coverage, q_per_asset_per_epoch=128),
            ),
        )

    with pytest.raises(ValueError, match="resume scientific config mismatch"):
        require_resume_scientific_config(current, resolved_config_dict(checkpoint_config))


def test_resume_restores_initial_baseline_best_score_and_history() -> None:
    r"""中断后 checkpoint selection 必须沿用初始化分母与 historical best。"""

    initial = {"density": 0.2, "kappa": 0.01, "derived_field": 2.0}
    history = [{"step": 4, "score": 0.8, "metrics": {"density": 0.1}}]

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
                "initial_validation_metrics": {"density": 0.2, "kappa": 0.01, "derived_field": 2.0},
                "best_validation_score": 0.8,
                "selection_history": [],
            }
        )
