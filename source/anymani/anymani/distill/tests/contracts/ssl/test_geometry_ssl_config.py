"""Resolved experiment 与 asset split leakage 合同。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from anymani.assets.bank import (
    HandAssetDatasetCfg,
    HandAssetLineageCfg,
    HandAssetPartitionCfg,
    HandAssetRunCfg,
    ResolvedHandAssetDataset,
    ResolvedHandAssetPartition,
)
from anymani.distill.representations.geometry import GeometryRepresentationCfg
from anymani.distill.representations.sources.collision_geometry import (
    AnchorSamples,
    HomeSurfaceSamples,
    OwnerGeometryCache,
)
from anymani.distill.representations.targets.geometry_field import GaussianProximityFieldCfg
from anymani.distill.ssl.config import (
    GeometrySSLAssetManifest,
    GeometrySSLExperimentCfg,
    GeometrySSLTrainerCfg,
    derive_geometry_ssl_training_budget,
    experiment_config_from_dict,
    resolved_config_dict,
    write_resolved_experiment_files,
)
from anymani.distill.ssl.experiments import CanonicalResidualFamilyCfg
from anymani.distill.ssl.runtime import GeometrySSLExperiment
from anymani.distill.ssl.runtime.assets import anchor_realization_record, home_surface_realization_record
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf


def test_omegaconf_payload_round_trip_rebuilds_validated_dataclasses() -> None:
    """CLI 可变 payload 必须重建冻结科研合同，且不改变数值。"""

    original = GeometrySSLExperimentCfg(
        asset_dataset_manifest="source/anymani/anymani/assets/datasets/canonical_cross_mother_v1.yaml",
    )
    rebuilt = experiment_config_from_dict(resolved_config_dict(original))

    assert rebuilt == original
    assert rebuilt.asset_dataset_manifest.endswith("canonical_cross_mother_v1.yaml")
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
    assert rebuilt.asset_dataset_manifest.endswith("canonical_cross_mother_v1.yaml")
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


def test_resolved_artifacts_copy_dataset_and_embed_its_content_hash(tmp_path: Path) -> None:
    r"""run artifact 必须同时保留输入 YAML、解析内容与 expanded physical manifest。"""

    dataset_path = tmp_path / "dataset.yaml"
    dataset_path.write_text("schema_version: 1.0.0\n", encoding="utf-8")  # 原始 bytes 应逐字复制
    dataset_config = HandAssetDatasetCfg(
        default_run_dir="/generated",
        train=HandAssetPartitionCfg(
            runs={
                "default": HandAssetRunCfg(
                    groups={"single_palm_leap": {"right_t4": HandAssetLineageCfg(include_mother=True)}}
                )
            }
        ),
    )
    empty_train = ResolvedHandAssetPartition(name="train", records=())
    empty_validation = ResolvedHandAssetPartition(name="validation", records=())
    dataset = ResolvedHandAssetDataset(
        source_path=dataset_path,
        source_sha256="dataset-sha256",
        config=dataset_config,
        train=empty_train,
        validation=empty_validation,
        evaluation={},
    )
    manifest = GeometrySSLAssetManifest(
        schema_version="2.0.0",
        dataset_source_path=str(dataset_path),
        dataset_source_sha256=dataset.source_sha256,
        train=(),
        validation=(),
        evaluation={},
    )
    output_dir = tmp_path / "run"

    write_resolved_experiment_files(
        output_dir,
        config=GeometrySSLExperimentCfg(asset_dataset_manifest=str(dataset_path)),
        dataset=dataset,
        manifest=manifest,
    )

    resolved = yaml.safe_load((output_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    expanded = yaml.safe_load((output_dir / "asset_manifest.yaml").read_text(encoding="utf-8"))
    assert (output_dir / "asset_dataset.yaml").read_bytes() == dataset_path.read_bytes()
    assert resolved["resolved_asset_dataset"]["source_sha256"] == "dataset-sha256"
    assert resolved["resolved_asset_dataset"]["config"]["train"]["runs"]["default"]["groups"]
    assert expanded["dataset_source_sha256"] == "dataset-sha256"


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


def test_canonical_45_asset_budget_reports_actual_tail_group_and_updates() -> None:
    """45 项 train partition 的尾组应保留真实样本数，而不是用名义 batch_size 掩盖。"""

    budget = derive_geometry_ssl_training_budget(GeometrySSLExperimentCfg(), train_asset_count=45)

    assert budget.microbatches_per_epoch == 2944
    assert budget.optimizer_updates_per_epoch == 736
    assert budget.total_optimizer_updates == 14720
    assert budget.total_q_samples == 230400
    assert budget.nominal_microbatch_q == 4
    assert budget.nominal_effective_q == 16
    assert budget.mean_effective_q == pytest.approx(15.652173913043478)


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
        GeometrySSLAssetManifest(
            schema_version="2.0.0",
            dataset_source_path="/dataset.yaml",
            dataset_source_sha256="dataset-hash",
            train=train,
            validation=validation,
            evaluation={},
        )


def test_asset_manifest_rejects_physical_geometry_leakage_across_splits() -> None:
    r"""不同 limits/content 但同一物理映射的资产仍不得跨 split。"""

    train = ({"asset_id": "train", "content_hash": "content-a", "physical_geometry_hash": "same"},)
    validation = ({"asset_id": "limit-only", "content_hash": "content-b", "physical_geometry_hash": "same"},)
    with pytest.raises(ValueError, match="physical geometry hashes leak"):
        GeometrySSLAssetManifest(
            schema_version="2.0.0",
            dataset_source_path="/dataset.yaml",
            dataset_source_sha256="dataset-hash",
            train=train,
            validation=validation,
            evaluation={},
        )
