"""Resolved experiment 与 asset split leakage 合同。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from anymani.distill.representations.geometry import GeometryRepresentationCfg
from anymani.distill.representations.sources.collision_geometry import (
    AnchorSamples,
    HomeSurfaceSamples,
    OwnerGeometryCache,
)
from anymani.distill.representations.targets.geometry_field import GaussianProximityFieldCfg
from anymani.distill.ssl.config import (
    GeometrySSLAssetCfg,
    GeometrySSLAssetManifest,
    GeometrySSLExperimentCfg,
    GeometrySSLTrainerCfg,
    derive_geometry_ssl_training_budget,
    experiment_config_from_dict,
    resolved_config_dict,
)
from anymani.distill.ssl.experiments import CanonicalResidualFamilyCfg
from anymani.distill.ssl.runtime import GeometrySSLExperiment
from anymani.distill.ssl.runtime.assets import anchor_realization_record, home_surface_realization_record
from anymani.distill.ssl.split import GeometryAssetIdentityRecord, split_geometry_asset_groups
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf


def test_omegaconf_payload_round_trip_rebuilds_validated_dataclasses() -> None:
    """CLI 可变 payload 必须重建冻结科研合同，且不改变数值。"""

    original = GeometrySSLExperimentCfg(
        assets=GeometrySSLAssetCfg(train_paths=("/generated/train",)),
    )
    rebuilt = experiment_config_from_dict(resolved_config_dict(original))

    assert rebuilt == original
    assert isinstance(rebuilt.assets.train_paths, tuple)
    assert isinstance(rebuilt.representation.field.bandwidth_centers_m, tuple)
    assert isinstance(rebuilt.representation.field.validation_bandwidths_m, tuple)


def test_concrete_canonical_class_composes_trainer_yaml_and_cli_override() -> None:
    """Hydra group 必须进入具体 experiment，且无需 builder function 或 experiment registry。"""

    import anymani.distill.ssl.pretrain  # noqa: F401  # import 时只注册 ConfigStore，不执行 run

    assert CanonicalResidualFamilyCfg.defaults == ({"trainer": "single_gpu_16gb"}, "_self_")
    with initialize_config_module(config_module="anymani.distill.presets.ssl", version_base="1.3"):
        composed = compose(
            config_name="geometry_ssl_canonical_residual_family",
            overrides=["trainer.learning_rate=0.0007"],
        )
    payload = OmegaConf.to_container(composed, resolve=True)
    assert isinstance(payload, dict)
    rebuilt = experiment_config_from_dict(payload)

    assert rebuilt.schema_version == "2.0.0"
    assert rebuilt.trainer.learning_rate == pytest.approx(7.0e-4)
    assert rebuilt.model.encoder.backbone.layers == 2
    assert resolved_config_dict(rebuilt) == payload


def test_experiment_constructor_has_no_filesystem_or_cuda_side_effect(tmp_path: Path) -> None:
    """配置对象到 runtime object 的构造边界不得提前创建 run 或 materialize source。"""

    output_dir = tmp_path / "not-created-until-run"
    experiment = GeometrySSLExperiment(GeometrySSLExperimentCfg(), output_dir=output_dir)

    assert experiment.config.schema_version == "2.0.0"
    assert experiment.output_dir == output_dir
    assert not output_dir.exists()


def test_legacy_resolved_config_fails_with_an_explicit_contract_error() -> None:
    """旧 fixed-bandwidth/AABB 配置不做兼容迁移，但必须给出可审计的 fail-closed 原因。"""

    payload = resolved_config_dict(GeometrySSLExperimentCfg())
    payload["schema_version"] = "1.1.0"

    with pytest.raises(ValueError, match="schema must be exactly 2.0.0"):
        experiment_config_from_dict(payload)


def test_anchor_realization_fingerprint_covers_points_parameters_and_version() -> None:
    """resume manifest 必须区分实际点集、采样 seed、物理尺度和算法版本。"""

    anchors = AnchorSamples(
        anchors_hand_m=np.asarray([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], dtype=np.float64),
        finger_names=("index", "index"),
        seed_ids=("seed/index", "seed/index"),
        surface_mask=np.asarray([True, False]),
        radial_support_radius_m=0.05,
        radial_decay_scale_m=0.025,
        surface_fraction=0.5,
        sampling_seed=7,
        algorithm_version="palm-seed-radial-gaussian-fps-v1",
    )
    baseline = anchor_realization_record(anchors)
    repeated = anchor_realization_record(anchors)
    changed_seed = anchor_realization_record(replace(anchors, sampling_seed=8))
    changed_points = anchor_realization_record(
        replace(anchors, anchors_hand_m=anchors.anchors_hand_m + np.asarray([1.0e-4, 0.0, 0.0]))
    )

    assert baseline == repeated
    assert len(baseline["anchor_realization_hash"]) == 64
    assert changed_seed["anchor_realization_hash"] != baseline["anchor_realization_hash"]
    assert changed_points["anchor_realization_hash"] != baseline["anchor_realization_hash"]


def test_home_surface_fingerprint_covers_retained_points_and_surface_backend() -> None:
    """manifest 必须冻结 retained home points 及其 Boolean/surface sampling 生产语义。"""

    samples = HomeSurfaceSamples(
        owner_ids=("palm",),
        points_owner_local_m=np.asarray([[[0.0, 0.0, 0.0]]], dtype=np.float64),
        face_indices=np.asarray([[3]], dtype=np.int64),
        barycentric=np.asarray([[[0.2, 0.3, 0.5]]], dtype=np.float64),
        sampling_seed=11,
        oversample_factor=8,
    )
    cache = OwnerGeometryCache(
        asset_id="asset",
        asset_content_hash="content",
        boolean_backend="manifold3d",
        records=(),
        surface_geometry_hash="surface-hash",
        surface_processing_version="owner-surface-v2",
    )
    baseline = home_surface_realization_record(samples, cache)
    changed = home_surface_realization_record(
        replace(samples, points_owner_local_m=samples.points_owner_local_m + 1.0e-4),
        cache,
    )

    assert len(baseline["home_surface_realization_hash"]) == 64
    assert baseline["surface_query_sampling_version"] == "owner-triangle-area-barycentric-v1"
    assert baseline["boolean_backend"] == "manifold3d"
    assert changed["home_surface_realization_hash"] != baseline["home_surface_realization_hash"]


def test_model_does_not_freeze_target_sigma_sample_count() -> None:
    """sigma 数量属于 target 数据轴，改变中心数不应重建或拒绝 scalar decoder。"""

    config = GeometrySSLExperimentCfg(
        representation=GeometryRepresentationCfg(
            field=GaussianProximityFieldCfg(bandwidth_centers_m=(0.004, 0.008, 0.016, 0.032, 0.064))
        )
    )
    assert len(config.representation.field.bandwidth_centers_m) == 5


def test_canonical_17_asset_budget_reports_actual_tail_group_and_updates() -> None:
    """17 项训练 split 的尾组应保留真实样本数，而不是用名义 batch_size 掩盖。"""

    budget = derive_geometry_ssl_training_budget(GeometrySSLExperimentCfg(), train_asset_count=17)

    assert budget.microbatches_per_epoch == 1152
    assert budget.optimizer_updates_per_epoch == 288
    assert budget.total_optimizer_updates == 5760
    assert budget.total_q_samples == 87040
    assert budget.nominal_microbatch_q == 4
    assert budget.nominal_effective_q == 16
    assert budget.mean_effective_q == pytest.approx(15.11111111111111)


@pytest.mark.parametrize("device", ["cpu", "cuda:not-an-index"])
def test_warp_training_config_rejects_non_cuda_device(device: str) -> None:
    """resolved 配置不得接受 Warp 在线 teacher 无法执行的 device。"""

    with pytest.raises(ValueError, match="device.*cuda"):
        GeometrySSLTrainerCfg(device=device)


def test_warp_training_config_rejects_float64() -> None:
    """Warp PyTorch bridge 主路径只接受 CUDA float32，不声明伪 float64 路线。"""

    with pytest.raises(ValueError, match="dtype.*float32"):
        GeometrySSLTrainerCfg(dtype="float64")


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
