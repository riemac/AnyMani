"""完整 resume 与 retained-only transfer 的 checkpoint 合同。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from anymani.distill.models.backbones.geometry_transformer import GraphBiasedTransformerCfg
from anymani.distill.models.decoders.representations.implicit_field import (
    DistanceSensitivityDecoderCfg,
    GeometrySSLDecoderCfg,
    ScalarSigmaFiLMDensityDecoderCfg,
)
from anymani.distill.models.geometry_ssl import GeometrySSLModel, GeometrySSLModelCfg
from anymani.distill.models.input_adapters.geometry import (
    GeometryEncoderCfg,
    GeometryLatentHeadsCfg,
    ImplicitGeometryEncoder,
    SO2AnchorFrontendCfg,
)
from anymani.distill.ssl.checkpoint import (
    GeometrySSLCheckpointMetadata,
    load_geometry_ssl_checkpoint,
    load_geometry_ssl_runtime_state,
    load_retained_geometry_encoder,
    save_geometry_ssl_checkpoint,
)


def _config() -> GeometrySSLModelCfg:
    """返回快速 checkpoint round-trip 所需的小网络。"""

    return GeometrySSLModelCfg(
        encoder=GeometryEncoderCfg(
            frontend=SO2AnchorFrontendCfg(relation_width=8, home_width=8, screw_width=8),
            backbone=GraphBiasedTransformerCfg(
                hidden_width=16,
                layers=1,
                attention_heads=4,
                feedforward_width=24,
                dropout=0.0,
            ),
            heads=GeometryLatentHeadsCfg(zero_order_width=12, first_order_width=8),
        ),
        ssl_decoders=GeometrySSLDecoderCfg(
            density=ScalarSigmaFiLMDensityDecoderCfg(hidden_width=16, residual_blocks=1),
            sensitivity=DistanceSensitivityDecoderCfg(coefficient_hidden_width=16),
        ),
    )


def test_checkpoint_resumes_full_state_and_transfers_only_encoder(tmp_path: Path) -> None:
    """SSL resume 包含 decoder/optimizer，PPO 初始化只读取 encoder namespace。"""

    torch.manual_seed(47)
    config = _config()
    model = GeometrySSLModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    synthetic_loss = sum(parameter.square().mean() for parameter in model.parameters())
    synthetic_loss.backward()
    optimizer.step()
    path = tmp_path / "step_000003.pt"
    metadata = GeometrySSLCheckpointMetadata(
        code_revision="test-revision",
        package_version="test-version",
        geometry_semantics_schema="1.0.0",
        asset_manifest={"train": [{"asset_id": "synthetic", "content_hash": "abc"}]},
        resolved_config={"model": {"sigma_reference_m": 0.016}},
        calibrated_objective={"density": 1.0},
    )
    save_geometry_ssl_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        step=3,
        metadata=metadata,
        runtime_state={"epoch": 2, "block_index": 5, "asset_ids": ("synthetic",)},
    )

    resumed = GeometrySSLModel(config)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1.0e-3)
    step, loaded_metadata = load_geometry_ssl_checkpoint(
        path,
        model=resumed,
        optimizer=resumed_optimizer,
    )
    for key, value in model.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[key], value)
    assert step == 3
    assert loaded_metadata["code_revision"] == "test-revision"
    assert resumed_optimizer.state_dict()["state"]
    assert load_geometry_ssl_runtime_state(path)["epoch"] == 2

    transferred = ImplicitGeometryEncoder(config.encoder)
    report = load_retained_geometry_encoder(path, encoder=transferred)
    for key, value in model.encoder.state_dict().items():
        torch.testing.assert_close(transferred.state_dict()[key], value)
    assert report.missing_keys == ()
    assert report.unexpected_keys == ()


def test_checkpoint_schema_1_x_is_rejected_without_compatibility_guessing(tmp_path: Path) -> None:
    """schema 2.0.0 不得把旧 experiment/objective payload 猜测迁移为当前语义。"""

    path = tmp_path / "legacy.pt"
    torch.save({"schema_version": "1.0.0"}, path)

    with pytest.raises(ValueError, match="unsupported geometry SSL checkpoint schema='1.0.0'"):
        load_geometry_ssl_checkpoint(path, model=GeometrySSLModel(_config()))
