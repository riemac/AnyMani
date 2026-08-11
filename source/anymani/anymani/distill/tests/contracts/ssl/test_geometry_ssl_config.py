"""Resolved experiment 与 asset split leakage 合同。"""

from __future__ import annotations

from dataclasses import replace

import pytest
from anymani.distill.models.geometry_ssl import GeometrySSLModelConfig
from anymani.distill.ssl.config import (
    GeometrySSLAssetCfg,
    GeometrySSLAssetManifest,
    GeometrySSLExperimentCfg,
    GeometrySSLTrainLoopCfg,
    experiment_config_from_dict,
    resolved_config_dict,
)


def test_omegaconf_payload_round_trip_rebuilds_validated_dataclasses() -> None:
    """CLI 可变 payload 必须重建冻结科研合同，且不改变数值。"""

    original = GeometrySSLExperimentCfg(
        assets=GeometrySSLAssetCfg(train_paths=("/generated/train",)),
    )
    rebuilt = experiment_config_from_dict(resolved_config_dict(original))

    assert rebuilt == original
    assert isinstance(rebuilt.assets.train_paths, tuple)
    assert isinstance(rebuilt.target.bandwidths_m, tuple)


def test_model_and_target_bandwidth_axes_must_close() -> None:
    """decoder 通道数不能与物理 target 带宽数静默错位。"""

    with pytest.raises(ValueError, match="bandwidth_count"):
        GeometrySSLExperimentCfg(model=replace(GeometrySSLModelConfig(), bandwidth_count=3))


@pytest.mark.parametrize("device", ["cpu", "cuda:not-an-index"])
def test_warp_training_config_rejects_non_cuda_device(device: str) -> None:
    """resolved 配置不得接受 Warp 在线 teacher 无法执行的 device。"""

    with pytest.raises(ValueError, match="device.*cuda"):
        GeometrySSLTrainLoopCfg(device=device)


def test_warp_training_config_rejects_float64() -> None:
    """Warp PyTorch bridge 主路径只接受 CUDA float32，不声明伪 float64 路线。"""

    with pytest.raises(ValueError, match="dtype.*float32"):
        GeometrySSLTrainLoopCfg(dtype="float64")


def test_asset_manifest_rejects_content_hash_leakage_across_splits() -> None:
    """路径或 ID 不同但静态语义内容相同的资产仍视为 leakage。"""

    train = ({"asset_id": "train", "content_hash": "same"},)
    validation = ({"asset_id": "renamed", "content_hash": "same"},)
    with pytest.raises(ValueError, match="content hashes leak"):
        GeometrySSLAssetManifest("1.0.0", train, validation, ())


def test_official_path_cannot_overlap_generated_train_or_validation() -> None:
    """official evaluation 配置在 bank resolve 前也必须与 generated splits 隔离。"""

    with pytest.raises(ValueError, match="paths must be disjoint"):
        GeometrySSLAssetCfg(train_paths=("/same",), official_evaluation_paths=("/same",))
