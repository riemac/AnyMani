r"""训练资产固定 calibration 的数值与 artifact 合同。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from anymani.distill.models.geometry_ssl import GeometrySSLModel, GeometrySSLModelConfig
from anymani.distill.models.input_adapters.geometry import GeometryEncoderConfig
from anymani.distill.objectives.representations.field_reconstruction import GeometrySSLObjective
from anymani.distill.ssl.calibration import calibrate_geometry_ssl_weights


def test_calibration_uses_density_as_reference_and_writes_frozen_clipped_weights(tmp_path: Path) -> None:
    """固定 synthetic terms 的梯度量级应产生可审计 median、reference 与裁剪权重。"""

    model = GeometrySSLModel(
        GeometrySSLModelConfig(
            encoder=GeometryEncoderConfig(
                relation_width=8,
                home_width=8,
                screw_width=8,
                hidden_width=16,
                zero_order_width=8,
                first_order_width=4,
                transformer_layers=1,
                attention_heads=4,
                feedforward_width=24,
                dropout=0.0,
            ),
            decoder_hidden_width=8,
            decoder_residual_blocks=1,
        )
    )
    parameter = model.encoder.screw_projection.weight
    batches = tuple(SimpleNamespace(scale=float(index + 1)) for index in range(8))

    def forward_terms(current_model, _objective, batch):
        base = current_model.encoder.screw_projection.weight.square().mean()
        return SimpleNamespace(
            density=base,
            kappa=base * 0.1 * batch.scale,
            derived_field=base * 10.0,
            sobolev=base * 0.01,
            chain=base * 100.0,
            paired=base * 0.001,
        )

    path = tmp_path / "loss_calibration.yaml"
    weights = calibrate_geometry_ssl_weights(
        model,
        GeometrySSLObjective,
        batches,
        forward_terms,
        output_path=path,
        min_weight=0.01,
        max_weight=10.0,
    )

    assert path.is_file()
    assert weights.density == pytest.approx(1.0)
    assert weights.derived_field == pytest.approx(0.1, rel=1.0e-5)
    assert weights.sobolev == pytest.approx(10.0, rel=1.0e-5)
    assert weights.chain == pytest.approx(0.01, rel=1.0e-5)
    assert weights.paired == pytest.approx(10.0)
    assert parameter.grad is not None
