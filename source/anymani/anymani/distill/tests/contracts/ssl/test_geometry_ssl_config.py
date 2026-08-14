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
from anymani.distill.ssl.split import GeometryAssetIdentityRecord, split_geometry_asset_groups


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


def test_asset_manifest_rejects_physical_geometry_leakage_across_splits() -> None:
    r"""不同 limits/content 但同一物理映射的资产仍不得跨 split。"""

    train = ({"asset_id": "train", "content_hash": "content-a", "physical_geometry_hash": "same"},)
    validation = ({"asset_id": "limit-only", "content_hash": "content-b", "physical_geometry_hash": "same"},)
    with pytest.raises(ValueError, match="physical geometry hashes leak"):
        GeometrySSLAssetManifest("1.0.0", train, validation, ())


def test_official_path_cannot_overlap_generated_train_or_validation() -> None:
    """official evaluation 配置在 bank resolve 前也必须与 generated splits 隔离。"""

    with pytest.raises(ValueError, match="paths must be disjoint"):
        GeometrySSLAssetCfg(train_paths=("/same",), official_evaluation_paths=("/same",))


def test_grouped_split_is_deterministic_keeps_mother_group_in_train_and_prevents_leakage() -> None:
    r"""同 physical hash 的 limit-only 资产必须整组移动，mother 所在组固定训练。"""

    records = (
        GeometryAssetIdentityRecord("mother", "/family/mother", "content-m", "physical-a", "domain-a"),
        GeometryAssetIdentityRecord("limit-only", "/family/limit", "content-l", "physical-a", "domain-b"),
        GeometryAssetIdentityRecord("shape-b", "/family/b", "content-b", "physical-b", "domain-a"),
        GeometryAssetIdentityRecord("shape-c", "/family/c", "content-c", "physical-c", "domain-a"),
        GeometryAssetIdentityRecord("shape-d", "/family/d", "content-d", "physical-d", "domain-a"),
    )

    first = split_geometry_asset_groups(
        records,
        mother_asset_id="mother",
        validation_asset_count=2,
        split_seed=20260813,
    )
    second = split_geometry_asset_groups(
        records,
        mother_asset_id="mother",
        validation_asset_count=2,
        split_seed=20260813,
    )

    assert first == second
    assert {record.asset_id for record in first.train} >= {"mother", "limit-only"}
    assert {record.physical_geometry_hash for record in first.train}.isdisjoint(
        record.physical_geometry_hash for record in first.validation
    )
    assert len(first.validation) == 2
