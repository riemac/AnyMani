"""完整 resume 与 retained-only transfer 的 checkpoint 合同。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from anymani.distill.methods.contracts import FeatureSpec
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
    load_retained_geometry_artifact,
    save_geometry_ssl_checkpoint,
    save_retained_geometry_artifact,
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
    """full checkpoint 恢复 decoder/optimizer；PPO 只读取独立 encoder artifact。"""

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
        declared_objective={"density": 1.0, "kappa": 1.0, "derived_field": 1.0, "sobolev": 1.0, "chain": 1.0},
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
    with pytest.raises(ValueError, match="artifact type"):
        load_retained_geometry_artifact(path, encoder=ImplicitGeometryEncoder(config.encoder))

    artifact = tmp_path / "retained_artifact.pt"
    save_retained_geometry_artifact(
        artifact,
        model=model,
        feature_spec=FeatureSpec(zero_order_width=12, first_order_width=8),
        metadata=metadata,
        source_checkpoint=path,
    )
    retained_encoder = ImplicitGeometryEncoder(config.encoder)
    artifact_report = load_retained_geometry_artifact(artifact, encoder=retained_encoder)
    assert artifact_report.missing_keys == ()
    assert artifact_report.unexpected_keys == ()
    artifact_payload = torch.load(artifact, map_location="cpu", weights_only=True)
    assert "optimizer_state" not in artifact_payload
    assert "runtime_state" not in artifact_payload
    assert "model_state" not in artifact_payload
    assert "decoder" not in str(artifact_payload)
    assert "objective" not in str(artifact_payload)


@pytest.mark.parametrize("schema_version", ["1.0.0", "2.0.0"])
def test_legacy_checkpoint_is_rejected_without_compatibility_guessing(tmp_path: Path, schema_version: str) -> None:
    """schema 1/2 不得把旧 experiment/objective payload 猜测迁移为当前 schema 4 语义。"""

    path = tmp_path / "legacy.pt"
    torch.save({"schema_version": schema_version}, path)

    with pytest.raises(ValueError, match=f"unsupported geometry SSL checkpoint schema='{schema_version}'"):
        load_geometry_ssl_checkpoint(path, model=GeometrySSLModel(_config()))
