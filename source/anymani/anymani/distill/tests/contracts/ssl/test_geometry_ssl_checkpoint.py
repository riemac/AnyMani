"""通用 full resume 与 Method-owned retained transfer 的 checkpoint 合同。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from anymani.assets.asset_schema_geometry import SEMANTICS_SCHEMA_VERSION
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
from anymani.distill.ssl.runtime import post_training as post_training_runtime
from anymani.distill.ssl.runtime.post_training import (
    _checkpoint_identity,
    _require_checkpoint_for_stage,
    _require_independent_output_dir,
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
    path = tmp_path / "epoch_000003.pt"
    metadata = PretrainCheckpointMetadata(
        code_revision="test-revision",
        package_version="test-version",
        geometry_semantics_schema="1.0.0",
        dataset_identity={
            "schema_version": "1.0.0",
            "source_sha256": "synthetic-dataset-sha",
            "train_asset_count": 1,
            "train_asset_axis_sha256": "abc",
        },
        resolved_config={"method": {"name": "multi_anchor"}},
        declared_objective={"density": 1.0, "kappa": 1.0, "derived_field": 1.0},
    )
    save_pretrain_checkpoint(
        path,
        method_state=method.training_state_dict(),
        optimizer_state=optimizer.state_dict(),
        epoch=3,
        optimizer_update=12,
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
    assert payload["epoch"] == 3
    assert payload["optimizer_update"] == 12
    assert "step" not in payload
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
    assert artifact_payload["lineage"]["dataset_identity"] == dict(metadata.dataset_identity)


@pytest.mark.parametrize("schema_version", ["1.0.0", "2.0.0", "4.0.0", "5.0.0", "6.0.0"])
def test_legacy_checkpoint_is_rejected_without_compatibility_guessing(tmp_path: Path, schema_version: str) -> None:
    """旧 payload 不得被猜测迁移为当前 Method/Trainer state 容器。"""

    path = tmp_path / "legacy.pt"
    torch.save({"schema_version": schema_version}, path)
    with pytest.raises(ValueError, match="unsupported pretraining checkpoint schema"):
        load_pretrain_checkpoint(path)


def _post_training_payload() -> dict[str, object]:
    r"""构造只覆盖事后 lineage gate 的最小 schema-7 payload。"""

    return {
        "metadata": {
            "dataset_identity": {"source_sha256": "dataset", "train_asset_axis_sha256": "axis"},
            "resolved_config": {
                "data": {"manifest": "ssl.yaml"},
                "method": {"name": "multi_anchor"},
                "trainer": {"max_epochs": 32},
                "run": {"seed": 17},
            },
            "declared_objective": {"density": 1.0, "kappa": 20.0, "derived_field": 0.01},
            "calibration_artifact_hash": "calibration",
            "code_revision": "revision-a",
            "package_version": "0.7.1",
            "geometry_semantics_schema": SEMANTICS_SCHEMA_VERSION,
            "worktree_dirty": True,
            "worktree_fingerprint": "worktree-a",
        },
        "method_state": {"parameter": torch.tensor(1.0)},
        "optimizer_state": {"state": {}},
        "trainer_state": {},
        "epoch": 1,
        "optimizer_update": 4,
    }


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("code_revision", "revision-b"),
        ("package_version", "0.7.2"),
        ("geometry_semantics_schema", "changed-semantics"),
        ("worktree_fingerprint", "worktree-b"),
    ],
)
def test_post_training_lineage_distinguishes_code_and_geometry_identity(field: str, changed: object) -> None:
    r"""相同 data/method 不能掩盖代码、package、几何语义或 dirty-worktree 漂移。"""

    baseline = _post_training_payload()
    candidate = deepcopy(baseline)
    candidate["metadata"][field] = changed  # type: ignore[index]

    assert _checkpoint_identity(candidate) != _checkpoint_identity(baseline)


def test_post_training_output_cannot_be_nested_in_source_run(tmp_path: Path) -> None:
    r"""validation/evaluation 的任何输出都不得创建在输入 checkpoint 所属 run 内。"""

    checkpoint = tmp_path / "train-run" / "checkpoints" / "epoch_000004.pt"
    with pytest.raises(ValueError, match="outside every source checkpoint run"):
        _require_independent_output_dir(tmp_path / "train-run" / "validation", (checkpoint,))

    independent = tmp_path / "validation-run"
    _require_independent_output_dir(independent, (checkpoint,))
    assert not independent.exists()  # safety gate 只检查路径，不应提前创建 artifact 目录


def test_post_training_checkpoint_preflight_keeps_full_state_on_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""候选预检不得把 Method/AdamW full state 批量驻留到 GPU。"""

    checkpoint = tmp_path / "epoch_000004.pt"
    checkpoint.write_bytes(b"synthetic")
    payload = _post_training_payload()
    locations: list[object] = []

    def fake_load(path: Path, *, map_location: object) -> dict[str, object]:
        assert path == checkpoint
        locations.append(map_location)
        return payload

    monkeypatch.setattr(post_training_runtime, "load_pretrain_checkpoint", fake_load)
    loaded = _require_checkpoint_for_stage(
        checkpoint,
        dataset_identity=payload["metadata"]["dataset_identity"],  # type: ignore[index]
        current_data={"manifest": "ssl.yaml"},
        current_method={"name": "multi_anchor"},
        seed=17,
    )

    assert loaded is payload
    assert locations == ["cpu"]
