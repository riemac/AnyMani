"""Schema 3 Hydra composition、在线预算与 physical realization fingerprint 合同。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from anymani.distill.models.geometry_ssl import GeometrySSLModel
from anymani.distill.representations.sources.collision_geometry import (
    AnchorSamples,
    HomeSurfaceSamples,
    OwnerGeometryCache,
)
from anymani.distill.ssl.data import HandAssetCatalogCfg
from anymani.distill.ssl.experiment import EmbodimentPretrain, EmbodimentPretrainCfg, resolved_config_dict
from anymani.distill.ssl.runtime.assets import (
    anchor_realization_record,
    home_surface_realization_record,
    validate_asset_manifest_isolation,
)
from anymani.distill.ssl.runtime.pretrainer import EmbodimentPretrainTrainerCfg
from anymani.distill.ssl.runtime.sampling import OnlineMinibatchSchedule, OnlineSamplingCfg
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

pytestmark = pytest.mark.contract


def _compose() -> EmbodimentPretrainCfg:
    """恢复正式 canonical YAML，不调用训练副作用。"""

    import anymani.distill.ssl.pretrain  # noqa: F401  # import only registers Hydra schemas

    with initialize_config_module(config_module="anymani.distill.presets.ssl", version_base="1.3"):
        composed = compose(config_name="canonical_multi_anchor_gaussian")
    resolved = OmegaConf.to_object(composed)
    assert isinstance(resolved, EmbodimentPretrainCfg)
    return resolved


def test_hydra_recovers_all_concrete_roles_and_objective_terms() -> None:
    """根五 role 与六个 objective term 必须保持 concrete dataclass 类型。"""

    config = _compose()
    config.validate_composed()
    assert isinstance(config.data, HandAssetCatalogCfg)
    assert type(config.method).__name__ == "MultiAnchorGaussianMethodCfg"
    assert type(config.trainer).__name__ == "EmbodimentPretrainTrainerCfg"
    assert type(config.method.representation).__name__ == "GeometryRepresentationCfg"
    assert type(config.method.model).__name__ == "GeometrySSLModelCfg"
    assert set(config.method.objectives) == {
        "density",
        "kappa",
        "derived_field",
        "sobolev",
        "chain",
        "paired",
    }
    assert all(type(term).__name__.endswith("ObjectiveTermCfg") for term in config.method.objectives.values())


def test_hydra_cli_override_changes_local_cfg_without_central_parser() -> None:
    """Hydra override 应直接改变局部 optimizer cfg，并保留 concrete root。"""

    import anymani.distill.ssl.pretrain  # noqa: F401

    with initialize_config_module(config_module="anymani.distill.presets.ssl", version_base="1.3"):
        composed = compose(
            config_name="canonical_multi_anchor_gaussian",
            overrides=["trainer.optimizer.learning_rate=0.0007"],
        )
    config = OmegaConf.to_object(composed)
    assert isinstance(config, EmbodimentPretrainCfg)
    assert config.trainer.optimizer.learning_rate == pytest.approx(7.0e-4)
    assert resolved_config_dict(config)["schema_version"] == "3.0.0"


def test_experiment_constructor_has_no_filesystem_or_cuda_side_effect(tmp_path) -> None:
    """完整 resolved config 到 façade 的构造不得提前创建 run 或 materialize source。"""

    output_dir = tmp_path / "not-created-until-run"
    experiment = EmbodimentPretrain(_compose(), output_dir=output_dir)

    assert experiment.config.schema_version == "3.0.0"
    assert experiment.output_dir == output_dir
    assert not output_dir.exists()


def test_schema_one_and_two_are_fail_closed() -> None:
    """旧配置不通过 alias 或 parser 猜测进入 schema 3。"""

    config = _compose()
    for version in ("1.0.0", "2.0.0"):
        with pytest.raises(ValueError, match="schema must be exactly 3.0.0"):
            replace(config, schema_version=version).validate_composed()


def test_model_does_not_freeze_target_sigma_sample_count() -> None:
    """sigma 数量属于 target 数据轴，改变中心数不应重建 scalar decoder。"""

    config = _compose()
    representation = replace(
        config.method.representation,
        field=replace(config.method.representation.field, bandwidth_centers_m=(0.004, 0.008, 0.016, 0.032, 0.064)),
    )
    method = replace(config.method, representation=representation)
    model = GeometrySSLModel(method.model)
    assert len(representation.field.bandwidth_centers_m) == 5
    assert model.density_decoder.output.out_features == 1


def test_canonical_45_asset_budget_reports_actual_tail_group_and_updates() -> None:
    """45 项 train partition 的尾组保留真实长度，并给出 proposal 中的预算锚点。"""

    sampling = OnlineSamplingCfg(
        epochs=20,
        q_per_asset_per_epoch=256,
        assets_per_minibatch=2,
        q_per_asset_per_minibatch=2,
        seed=20260813,
    )
    schedule = OnlineMinibatchSchedule(45, sampling)
    assert schedule.minibatches_per_epoch == 2944
    assert schedule.minibatches_per_epoch * sampling.epochs == 58880
    assert 45 * sampling.q_per_asset_per_epoch * sampling.epochs == 230400
    assert 2944 // 4 == 736
    assert pytest.approx(15.652173913043478) == 230400 / 14720
    for _ in range(schedule.minibatches_per_epoch - 1):
        schedule.next()
    tail = schedule.next()
    assert len(tail.asset_indices) == 1


@pytest.mark.parametrize("device", ["cpu", "cuda:not-an-index"])
def test_warp_training_config_rejects_non_cuda_device(device: str) -> None:
    """online Warp teacher 不接受 CPU 或非法 CUDA index。"""

    with pytest.raises(ValueError, match="device.*cuda"):
        EmbodimentPretrainTrainerCfg(device=device)


def test_warp_training_config_rejects_float64() -> None:
    """正式 Warp bridge 只接受 CUDA float32。"""

    with pytest.raises(ValueError, match="dtype.*float32"):
        EmbodimentPretrainTrainerCfg(dtype="float64")


def test_hand_catalog_rejects_missing_manifest_without_io() -> None:
    """data role 必须显式绑定 assets dataset YAML。"""

    with pytest.raises(ValueError, match="requires one dataset manifest"):
        HandAssetCatalogCfg()


@pytest.mark.parametrize(
    ("identity_name", "error"),
    [("content_hash", "content hashes leak"), ("physical_geometry_hash", "physical geometry hashes leak")],
)
def test_expanded_manifest_rejects_identity_leakage(identity_name: str, error: str) -> None:
    r"""路径/ID 或 limits 不同也不能掩盖 content/physical mapping 的跨 role 重复。"""

    train = {"asset_id": "train", "content_hash": "content-a", "physical_geometry_hash": "physical-a"}
    validation = {
        "asset_id": "renamed",
        "content_hash": "content-b",
        "physical_geometry_hash": "physical-b",
    }
    validation[identity_name] = train[identity_name]
    with pytest.raises(ValueError, match=error):
        validate_asset_manifest_isolation({"train": [train], "validation": [validation], "evaluation": {}})


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
    assert baseline == anchor_realization_record(anchors)
    assert len(baseline["anchor_realization_hash"]) == 64
    assert baseline != anchor_realization_record(replace(anchors, sampling_seed=8))
    assert baseline != anchor_realization_record(
        replace(anchors, anchors_hand_m=anchors.anchors_hand_m + np.asarray([1.0e-4, 0.0, 0.0]))
    )


def test_home_surface_fingerprint_covers_retained_points_and_surface_backend() -> None:
    """manifest 必须冻结 retained home points 及 Boolean/surface sampling 生产语义。"""

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
    assert baseline["home_surface_realization_hash"] != changed["home_surface_realization_hash"]
