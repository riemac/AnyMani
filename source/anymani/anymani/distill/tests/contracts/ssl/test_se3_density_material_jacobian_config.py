r"""N040 v0.8.1 snapshot 与 independent runtime 类型合同。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from anymani.distill.methods.density_material_jacobian import SE3DensityMaterialJacobianMethodCfg
from anymani.distill.methods.density_material_jacobian.artifact import load_se3_retained_encoder_artifact
from anymani.distill.models.input_adapters.se3_invariant_encoder import (
    SE3InvariantGeometryEncoder,
    SE3InvariantGeometryEncoderCfg,
)
from anymani.distill.ssl.config_store import compose_evaluation_cfg, compose_pretrain_cfg


def test_n040_snapshot_uses_independent_se3_encoder_and_method() -> None:
    r"""N040 不得回落到 legacy SO2-named encoder config。"""

    config = compose_pretrain_cfg(config_ref="geometry_ssl_density_material_jacobian_se3_v0_8_1")
    assert isinstance(config.method, SE3DensityMaterialJacobianMethodCfg)
    assert isinstance(config.method.model.encoder, SE3InvariantGeometryEncoderCfg)
    assert config.method.se3_coordinate_rewrite.probability == 1.0
    assert config.method.se3_coordinate_rewrite.translation_half_extent_m == 0.05
    assert tuple(config.method.objectives.enabled()) == ("density", "material_jacobian")
    assert config.trainer.max_epochs == 384


def test_n040_evaluation_preserves_n031_canonical_measure() -> None:
    r"""N040 只改变 coordinate representation，不改变 held-out density/Gamma measure。"""

    config = compose_evaluation_cfg(config_ref="geometry_ssl_density_material_jacobian_se3_v0_8_1")
    assert tuple(config.method.representation.field.fixed_bandwidths_m) == (0.004, 0.016, 0.064)
    assert config.evaluation.q_per_asset == 64
    assert config.run.seed == 20260830


def test_n040_retained_artifact_declares_se3_encoder_identity(tmp_path: Path) -> None:
    r"""同形参数不能掩盖 frontend 数学；schema-5 必须显式声明 `se3_invariant`。"""

    config = compose_pretrain_cfg(config_ref="geometry_ssl_density_material_jacobian_se3_v0_8_1")
    method = config.method.runtime_type(config.method)
    method.initialize_model(device=torch.device("cpu"), dtype=torch.float32)
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"lineage-anchor")
    payload = method.retained_artifact_payload(
        metadata={
            "resolved_config": {"trainer": {"execution": {"parameter_dtype": "float32"}}},
            "source_artifact": {},
        },
        source_checkpoint=checkpoint,
    )
    assert payload["retained_model_config"]["encoder_type"] == "se3_invariant"
    assert "proper-SE(3)-invariant" in payload["input_contract"]["frame"]
    assert all(name.startswith("encoder.") for name in payload["retained_state"])


def test_n040_retained_artifact_strict_loader_reconstructs_encoder(tmp_path: Path) -> None:
    r"""PPO consumer 必须从 artifact 自描述 config 重建 N040，并先核对完整文件 SHA。"""

    config = compose_pretrain_cfg(config_ref="geometry_ssl_density_material_jacobian_se3_v0_8_1")
    method = config.method.runtime_type(config.method)
    method.initialize_model(device=torch.device("cpu"), dtype=torch.float32)
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"lineage-anchor")
    artifact = tmp_path / "retained_encoder.pt"
    torch.save(
        method.retained_artifact_payload(
            metadata={
                "resolved_config": {"trainer": {"execution": {"parameter_dtype": "float32"}}},
                "source_artifact": {},
            },
            source_checkpoint=checkpoint,
        ),
        artifact,
    )
    expected_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()

    loaded = load_se3_retained_encoder_artifact(artifact, expected_sha256=expected_sha256)

    assert isinstance(loaded.encoder, SE3InvariantGeometryEncoder)
    assert loaded.encoder.se3_config == config.method.model.encoder
    assert loaded.load_report.missing_keys == ()
    assert loaded.load_report.unexpected_keys == ()
    assert loaded.artifact_sha256 == expected_sha256
    assert loaded.feature_spec["entity_width"] == 128
    assert sum(parameter.numel() for parameter in loaded.encoder.parameters()) == 582_343

    try:
        load_se3_retained_encoder_artifact(artifact, expected_sha256="0" * 64)
    except ValueError as exc:
        assert "SHA-256" in str(exc)
    else:
        raise AssertionError("retained artifact with the wrong SHA-256 must be rejected")
