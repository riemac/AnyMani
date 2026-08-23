"""通用 full resume 与 Method-owned retained transfer 的 checkpoint 合同。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from anymani.distill.methods.multi_anchor_gaussian_implicit_field import (
    MultiAnchorGaussianMethod,
    MultiAnchorGaussianMethodCfg,
    load_retained_geometry_artifact,
)
from anymani.distill.models.backbones.geometry_transformer import GraphBiasedTransformerCfg
from anymani.distill.models.decoders.representations.implicit_field import (
    DistanceSensitivityDecoderCfg,
    GeometrySSLDecoderCfg,
    ScalarSigmaFiLMDensityDecoderCfg,
)
from anymani.distill.models.geometry_ssl import GeometrySSLModelCfg
from anymani.distill.models.input_adapters.geometry import (
    GeometryEncoderCfg,
    GeometryLatentHeadsCfg,
    ImplicitGeometryEncoder,
    SO2AnchorFrontendCfg,
)
from anymani.distill.ssl.checkpoint import (
    PretrainCheckpointMetadata,
    load_pretrain_checkpoint,
    save_pretrain_checkpoint,
    save_retained_artifact,
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
    """通用容器恢复 Method/optimizer；standalone artifact 只迁移 encoder。"""

    torch.manual_seed(47)
    model_config = _config()
    method = MultiAnchorGaussianMethod(MultiAnchorGaussianMethodCfg(model=model_config))
    method.initialize_model(device=torch.device("cpu"), dtype=torch.float32)
    optimizer = torch.optim.AdamW(method.parameters(), lr=1.0e-3)
    synthetic_loss = sum(parameter.square().mean() for parameter in method.parameters())
    synthetic_loss.backward()
    optimizer.step()
    path = tmp_path / "step_000003.pt"
    metadata = PretrainCheckpointMetadata(
        code_revision="test-revision",
        package_version="test-version",
        geometry_semantics_schema="1.0.0",
        asset_manifest={"train": [{"asset_id": "synthetic", "content_hash": "abc"}]},
        resolved_config={"method": {"name": "multi_anchor"}},
        declared_objective={"density": 1.0, "kappa": 1.0, "derived_field": 1.0, "sobolev": 1.0, "chain": 1.0},
    )
    save_pretrain_checkpoint(
        path,
        method_state=method.training_state_dict(),
        optimizer_state=optimizer.state_dict(),
        step=3,
        metadata=metadata,
        trainer_state={"minibatch_cursor": 5, "forward_index": 25},
    )

    resumed = MultiAnchorGaussianMethod(MultiAnchorGaussianMethodCfg(model=model_config))
    resumed.initialize_model(device=torch.device("cpu"), dtype=torch.float32)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1.0e-3)
    payload = load_pretrain_checkpoint(path)
    resumed.load_training_state_dict(payload["method_state"])
    resumed_optimizer.load_state_dict(payload["optimizer_state"])
    for key, value in method.training_state_dict().items():
        torch.testing.assert_close(resumed.training_state_dict()[key], value)
    assert payload["step"] == 3
    assert payload["metadata"]["code_revision"] == "test-revision"
    assert payload["trainer_state"]["minibatch_cursor"] == 5
    assert resumed_optimizer.state_dict()["state"]
    with pytest.raises(ValueError, match="artifact schema"):
        load_retained_geometry_artifact(path, encoder=ImplicitGeometryEncoder(model_config.encoder))

    artifact = tmp_path / "retained_artifact.pt"
    retained_payload = method.retained_artifact_payload(
        metadata=asdict(metadata),
        source_checkpoint=path,
    )
    save_retained_artifact(artifact, retained_payload)
    report = load_retained_geometry_artifact(artifact, encoder=ImplicitGeometryEncoder(model_config.encoder))
    assert report.missing_keys == ()
    assert report.unexpected_keys == ()
    artifact_payload = torch.load(artifact, map_location="cpu", weights_only=True)
    assert "optimizer_state" not in artifact_payload
    assert "trainer_state" not in artifact_payload
    assert "method_state" not in artifact_payload
    assert "decoder" not in str(artifact_payload)
    assert "objective" not in str(artifact_payload)


@pytest.mark.parametrize("schema_version", ["1.0.0", "2.0.0", "4.0.0"])
def test_legacy_checkpoint_is_rejected_without_compatibility_guessing(tmp_path: Path, schema_version: str) -> None:
    """旧 payload 不得被猜测迁移为当前 Method/Trainer state 容器。"""

    path = tmp_path / "legacy.pt"
    torch.save({"schema_version": schema_version}, path)
    with pytest.raises(ValueError, match="unsupported pretraining checkpoint schema"):
        load_pretrain_checkpoint(path)
