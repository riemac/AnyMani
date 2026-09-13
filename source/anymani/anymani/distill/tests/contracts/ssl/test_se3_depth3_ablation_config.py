r"""N040三层backbone受控容量消融的实验身份与唯一变量合同。"""

from __future__ import annotations

from dataclasses import replace

import torch
from anymani.distill.models.input_adapters.se3_invariant_encoder import SE3InvariantGeometryEncoder
from anymani.distill.ssl.config_store import compose_evaluation_cfg, compose_pretrain_cfg
from anymani.distill.ssl.experiments import available_experiments

BASE_CONFIG = "geometry_ssl_density_material_jacobian_se3_v0_8_1"
DEPTH3_CONFIG = "geometry_ssl_density_material_jacobian_se3_depth3_v0_8_1"


def test_depth3_ablation_is_registered_and_changes_only_depth_at_matched_budget() -> None:
    r"""三层候选保持N040科学合同，只改变block数并匹配16-cycle有效预算。"""

    assert DEPTH3_CONFIG in available_experiments()
    baseline = compose_pretrain_cfg(config_ref=BASE_CONFIG)
    candidate = compose_pretrain_cfg(config_ref=DEPTH3_CONFIG)

    # 原N040 headline通过completed-run extension达到512 epochs；候选从头使用同一有效预算。
    assert candidate.data == baseline.data
    assert candidate.method == replace(
        baseline.method,
        model=replace(
            baseline.method.model,
            encoder=replace(
                baseline.method.model.encoder,
                backbone=replace(baseline.method.model.encoder.backbone, layers=3),
            ),
        ),
    )
    assert candidate.trainer == replace(baseline.trainer, max_epochs=512)
    assert candidate.run.seed == baseline.run.seed
    assert candidate.run.source_cache_root == baseline.run.source_cache_root
    assert candidate.run.source_cache_mode == baseline.run.source_cache_mode


def test_depth3_evaluation_preserves_canonical_population_and_analyses() -> None:
    r"""容量消融使用与四层N040相同的held-out资产、q-bank和干预测度。"""

    baseline = compose_evaluation_cfg(config_ref=BASE_CONFIG)
    candidate = compose_evaluation_cfg(config_ref=DEPTH3_CONFIG)

    assert candidate.data == baseline.data
    assert candidate.evaluation == baseline.evaluation
    assert candidate.method.model.encoder.backbone.layers == 3
    assert candidate.method == replace(
        baseline.method,
        model=replace(
            baseline.method.model,
            encoder=replace(
                baseline.method.model.encoder,
                backbone=replace(baseline.method.model.encoder.backbone, layers=3),
            ),
        ),
    )


def test_depth3_encoder_parameter_count_matches_one_removed_block() -> None:
    r"""4→3层应恰好删除一个132,480参数的Pre-LN attention/FFN block。"""

    baseline = compose_pretrain_cfg(config_ref=BASE_CONFIG)
    candidate = compose_pretrain_cfg(config_ref=DEPTH3_CONFIG)
    baseline_encoder = SE3InvariantGeometryEncoder(baseline.method.model.encoder)
    candidate_encoder = SE3InvariantGeometryEncoder(candidate.method.model.encoder)
    baseline_parameters = sum(parameter.numel() for parameter in baseline_encoder.parameters())
    candidate_parameters = sum(parameter.numel() for parameter in candidate_encoder.parameters())

    assert baseline_parameters == 582_343
    assert candidate_parameters == 449_863
    assert baseline_parameters - candidate_parameters == 132_480
    assert next(candidate_encoder.parameters()).dtype == torch.float32
