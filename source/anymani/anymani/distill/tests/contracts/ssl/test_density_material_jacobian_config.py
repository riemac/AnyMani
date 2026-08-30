r"""Density + relational Material Jacobian v0.8.0 快照合同。"""

from __future__ import annotations

from anymani.distill.methods.density_material_jacobian import DensityMaterialJacobianMethodCfg
from anymani.distill.ssl.config_store import compose_evaluation_cfg, compose_pretrain_cfg


def test_v080_snapshot_binds_independent_method_and_density_gamma_terms() -> None:
    r"""新快照不得回落到旧 κ method/objective。"""

    config = compose_pretrain_cfg(config_ref="geometry_ssl_density_material_jacobian_v0_8_0")
    assert isinstance(config.method, DensityMaterialJacobianMethodCfg)
    assert tuple(config.method.objectives.enabled()) == ("density", "material_jacobian")
    assert config.method.representation.source.anchors.bank_size == 8
    assert config.method.material_sampling.train_active_per_joint == 2
    assert config.method.material_sampling.train_zero_per_joint == 1
    assert config.method.model.encoder.backbone.hidden_width == 128
    assert config.method.entity_permutation.enabled is True
    assert config.method.joint_sign_rewrite.probability == 0.20
    assert config.trainer.emit_compression_basis is False


def test_v080_evaluation_preserves_canonical_density_measure() -> None:
    r"""训练可优化，但 canonical evaluation 保持 4/16/64 mm 与 64 q/asset。"""

    config = compose_evaluation_cfg(config_ref="geometry_ssl_density_material_jacobian_v0_8_0")
    assert tuple(config.method.representation.field.fixed_bandwidths_m) == (0.004, 0.016, 0.064)
    assert config.evaluation.q_per_asset == 64
    assert config.run.seed == 20260830
    assert config.evaluation.final_ablations == (
        "query_only",
        "same_asset_q_shuffle",
        "cross_asset_shuffle",
        "joint_token_shuffle",
    )
