"""Schema 7 Hydra composition、纯训练预算与 physical realization fingerprint 合同。"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from anymani.distill.methods.multi_anchor_gaussian_implicit_field.provenance import (
    anchor_realization_record,
    home_surface_realization_record,
    validate_asset_manifest_isolation,
)
from anymani.distill.models.geometry_ssl import GeometrySSLModel
from anymani.distill.representations.sources.collision_geometry import (
    AnchorSamples,
    HomeSurfaceSamples,
    OwnerGeometryCache,
)
from anymani.distill.ssl.config_store import compose_evaluation_cfg, compose_pretrain_cfg, compose_validation_cfg
from anymani.distill.ssl.data import HandAssetCatalogCfg
from anymani.distill.ssl.data import hand_assets as hand_assets_module
from anymani.distill.ssl.data.hand_assets import EmbodimentCatalog, HandAssetCatalog, _prune_catalog_cache
from anymani.distill.ssl.experiment import EmbodimentPretrain, EmbodimentPretrainCfg, resolved_config_dict
from anymani.distill.ssl.pretrain import _build_parser, _config_overrides
from anymani.distill.ssl.runtime.post_training import normalized_validation_score, selection_baseline
from anymani.distill.ssl.runtime.pretrainer import EmbodimentPretrainTrainerCfg

pytestmark = pytest.mark.contract


def _compose() -> EmbodimentPretrainCfg:
    """从 ConfigStore 恢复正式 Python 实验，不调用训练副作用。"""

    return compose_pretrain_cfg()


def test_hydra_recovers_all_concrete_roles_and_objective_terms() -> None:
    """根四 role、Trainer 阶段协议与 rho/kappa objective 必须保持 concrete dataclass 类型。"""

    config = _compose()
    config.validate_composed()
    assert isinstance(config.data, HandAssetCatalogCfg)
    assert config.data.manifest == (
        "source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1/ssl.yaml"
    )
    assert config.data.expected_sha256 == "f1398417888e7c237cbb2583dcf8e9cd10bef7fee792b307c67dfa74fb6e0698"
    assert type(config.method).__name__ == "MultiAnchorGaussianMethodCfg"
    assert type(config.trainer).__name__ == "EmbodimentPretrainTrainerCfg"
    assert not hasattr(config, "evaluation")
    assert config.trainer.max_epochs == 32
    assert config.trainer.num_minibatches == 4
    assert config.trainer.mini_epochs == 1
    assert config.trainer.microbatch_size == 64
    assert not hasattr(config.trainer, "validation")
    assert not hasattr(config.trainer, "final_evaluation")
    assert config.trainer.checkpoint_every_epochs == 1
    assert not hasattr(config.trainer, "gradient_accumulation_steps")
    assert config.trainer.sampling.assets_per_minibatch == 64
    assert config.trainer.sampling.q_per_asset_per_minibatch == 8
    assert type(config.method.representation).__name__ == "GeometryRepresentationCfg"
    assert type(config.method.model).__name__ == "GeometrySSLModelCfg"
    assert set(config.method.objectives.enabled()) == {
        "density",
        "kappa",
    }
    assert config.method.objectives.density.weight == pytest.approx(1.0)
    assert config.method.objectives.kappa.weight == pytest.approx(1.0)
    assert not hasattr(config.method.objectives, "derived_field")
    assert not hasattr(config.method.objectives, "sobolev")
    assert not hasattr(config.method.objectives, "chain")
    assert config.method.representation.source.anchors.bank_size == 8
    assert tuple(config.method.representation.field.validation_bandwidths_m) == (0.004, 0.016, 0.064)
    assert not hasattr(config.method.representation, "layout")
    assert "paired" not in config.method.objectives.enabled()
    assert all(term.qualified_func_name().endswith(f"{name}_objective") for name, term in config.method.objectives.enabled().items())


def test_hydra_cli_override_changes_local_cfg_without_central_parser() -> None:
    """Hydra override 应直接改变局部 optimizer cfg，并保留 concrete root。"""

    config = compose_pretrain_cfg(["trainer.optimizer.learning_rate=0.0007"])
    assert config.trainer.optimizer.learning_rate == pytest.approx(7.0e-4)
    assert resolved_config_dict(config)["schema_version"] == "7.0.0"


def test_post_training_configs_are_independent_from_trainer() -> None:
    r"""validation/evaluation 应各自组合 data/method/stage/run，不回填 Trainer 字段。"""

    validation = compose_validation_cfg()
    evaluation = compose_evaluation_cfg()

    assert validation.schema_version == "1.0.0"
    assert evaluation.schema_version == "1.0.0"
    assert not hasattr(validation, "trainer")
    assert not hasattr(evaluation, "trainer")
    assert validation.validation.selection_metrics == ("density", "kappa")
    assert evaluation.evaluation.final_ablations[-1] == "joint_token_shuffle"
    assert validation.data == evaluation.data == _compose().data
    assert validation.method == evaluation.method == _compose().method


def test_flat_cli_flags_compose_one_run_without_exposing_config_paths() -> None:
    r"""普通 ``--flag value`` 入口应完整装配预实验预算，并让统一 seed 覆盖两个随机域。"""

    args = _build_parser().parse_args(
        (
            "--phase",
            "calibrate_objectives",
            "--max_epochs",
            "8",
            "--num_minibatches",
            "2",
            "--assets_per_minibatch",
            "64",
            "--q_per_asset_per_minibatch",
            "8",
            "--mini_epochs",
            "1",
            "--microbatch_size",
            "64",
            "--seed",
            "42",
            "--experiment_name",
            "objective_probe_seed42",
        )
    )
    config = compose_pretrain_cfg(_config_overrides(args))

    assert config.run.phase == "calibrate_objectives"
    assert config.run.experiment_name == "objective_probe_seed42"
    assert config.run.seed == config.trainer.sampling.seed == 42
    assert config.trainer.max_epochs == 8
    assert config.trainer.num_minibatches == 2
    assert config.trainer.sampling.assets_per_minibatch == 64
    assert config.trainer.sampling.q_per_asset_per_minibatch == 8
    assert config.trainer.mini_epochs == 1
    assert config.trainer.microbatch_size == 64


def test_experiment_constructor_has_no_filesystem_or_cuda_side_effect(tmp_path) -> None:
    """完整 resolved config 到 façade 的构造不得提前创建 run 或 materialize source。"""

    output_dir = tmp_path / "not-created-until-run"
    experiment = EmbodimentPretrain(_compose(), output_dir=output_dir)

    assert experiment.config.schema_version == "7.0.0"
    assert experiment.output_dir == output_dir
    assert not output_dir.exists()


def test_old_schemas_are_fail_closed() -> None:
    """旧配置不通过 alias 或 parser 猜测进入 schema 7。"""

    config = _compose()
    for version in ("1.0.0", "2.0.0", "6.0.0"):
        with pytest.raises(ValueError, match="schema must be exactly 7.0.0"):
            replace(config, schema_version=version).validate_composed()


def test_model_does_not_freeze_target_sigma_sample_count() -> None:
    """sigma 数量属于 target 数据轴，改变中心数不应重建 scalar decoder。"""

    config = _compose()
    representation = replace(
        config.method.representation,
        field=replace(
            config.method.representation.field,
            bandwidth_centers_m=(0.004, 0.008, 0.016, 0.032, 0.064),
            validation_bandwidths_m=(0.004, 0.008, 0.016, 0.032, 0.064),
        ),
    )
    model = GeometrySSLModel(config.method.model)
    assert len(representation.field.bandwidth_centers_m) == 5
    assert model.density_decoder.output.out_features == 1


def test_trainer_exposes_epoch_minibatch_and_microbatch_contract() -> None:
    """Trainer 显式声明训练回合、每回合新批数、复用遍数与显存切片。"""

    trainer = _compose().trainer
    assert trainer.max_epochs == 32
    assert trainer.num_minibatches == 4
    assert trainer.mini_epochs == 1
    assert trainer.microbatch_size == 64
    assert trainer.max_epochs * trainer.num_minibatches * 64 * 8 == 65_536
    assert trainer.max_epochs * trainer.mini_epochs * trainer.num_minibatches == 128
    assert not hasattr(trainer, "calibration_epochs")
    assert not hasattr(trainer, "pretrain_epochs")
    assert not hasattr(trainer, "gradient_accumulation_steps")
    assert not hasattr(trainer, "run_safety_step_limit")
    assert not hasattr(trainer.sampling, "q_per_asset_per_epoch")


@pytest.mark.parametrize("microbatch_size", [63, 60, 1024])
def test_trainer_rejects_microbatch_that_breaks_minibatch_or_asset_q_block(microbatch_size: int) -> None:
    r"""microbatch 必须整除 512-pair minibatch，并包含整数个完整 8-q 资产块。"""

    with pytest.raises(ValueError, match="microbatch"):
        EmbodimentPretrainTrainerCfg(microbatch_size=microbatch_size)


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


def test_training_dataset_identity_freezes_ordered_asset_and_content_axis() -> None:
    r"""轻量 checkpoint identity 必须检测资产替换和重排，但不触发 physical source 物化。"""

    records = (
        SimpleNamespace(container=SimpleNamespace(asset_id="asset-a"), content_hash="content-a"),
        SimpleNamespace(container=SimpleNamespace(asset_id="asset-b"), content_hash="content-b"),
    )
    dataset = SimpleNamespace(source_sha256="dataset-sha", train=SimpleNamespace(records=records))
    baseline = EmbodimentCatalog(dataset).training_dataset_identity()
    reordered_dataset = SimpleNamespace(
        source_sha256="dataset-sha",
        train=SimpleNamespace(records=tuple(reversed(records))),
    )

    assert baseline["schema_version"] == "1.0.0"
    assert baseline["source_sha256"] == "dataset-sha"
    assert baseline["train_asset_count"] == 2
    assert len(str(baseline["train_asset_axis_sha256"])) == 64
    assert EmbodimentCatalog(reordered_dataset).training_dataset_identity() != baseline


def test_catalog_cache_prunes_old_indexes_and_only_stale_temporary_files(tmp_path: Path) -> None:
    r"""slim catalog 可跨进程保留，但完整索引总量和中断临时文件必须有界。"""

    root = tmp_path / "asset_catalog"
    root.mkdir()
    oldest = root / "oldest.strict.pkl"
    middle = root / "middle.strict.pkl"
    current = root / "current.strict.pkl"
    for path in (oldest, middle, current):
        path.write_bytes(b"12345678")  # 每项 8 B，24 B 总量超过测试预算 16 B
    os.utime(oldest, ns=(1, 1))
    os.utime(middle, ns=(2, 2))
    os.utime(current, ns=(3, 3))
    stale_temporary = root / ".orphan.strict.pkl.stale"
    fresh_temporary = root / ".active.strict.pkl.fresh"
    stale_temporary.write_bytes(b"partial")
    fresh_temporary.write_bytes(b"partial")
    os.utime(stale_temporary, (1.0, 1.0))
    os.utime(fresh_temporary, (90_000.0, 90_000.0))

    _prune_catalog_cache(root, keep=current, max_bytes=16, now=100_000.0)

    assert not oldest.exists()  # 最旧完整索引先驱逐，使完整 pickle 总量回到 16 B
    assert middle.exists() and current.exists()  # 当前项受保护，另一项仍在预算内
    assert not stale_temporary.exists()  # 超过 24 h 的中断写入可安全回收
    assert fresh_temporary.exists()  # 仍可能属于并行进程的临时文件不得删除


def test_catalog_cache_miss_reloads_slim_dataset_after_releasing_full_heap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    r"""首次解析与 cache hit 必须返回同一 slim 对象语义，不能让 cold run 常驻完整 sidecar。"""

    full_dataset = type("FullDataset", (), {"source_sha256": "dataset-sha"})()
    slim_dataset = type("SlimDataset", (), {"source_sha256": "dataset-sha"})()
    parsed = type(
        "ParsedDataset",
        (),
        {
            "source_path": tmp_path / "ssl.yaml",
            "source_sha256": "dataset-sha",
            "resolve": lambda _self, **_kwargs: full_dataset,
        },
    )()
    monkeypatch.setattr(hand_assets_module.HandAssetDataset, "from_yaml", lambda _manifest: parsed)
    monkeypatch.setattr(hand_assets_module, "_cache_file", lambda *_args, **_kwargs: tmp_path / "catalog.pkl")
    load_results = iter((None, slim_dataset))  # 首次 miss；写盘并 trim full heap 后必须再次加载 slim
    monkeypatch.setattr(hand_assets_module, "_load_cached_dataset", lambda *_args, **_kwargs: next(load_results))
    written: list[object] = []
    monkeypatch.setattr(hand_assets_module, "_write_cached_dataset", lambda _path, dataset, **_kwargs: written.append(dataset))

    catalog = HandAssetCatalog(HandAssetCatalogCfg(manifest=str(tmp_path / "ssl.yaml"))).resolve()

    assert written == [full_dataset]  # writer 只短暂消费完整解析结果，不把它交给训练
    assert catalog.dataset is slim_dataset  # cold miss 与下一进程 pickle hit 使用同一精简数据轴


def test_validation_selection_weights_named_suites_equally() -> None:
    r"""checkpoint score 应先在 suite 内归一化，再对两条泛化轴等权平均。

    该合同与每条 suite 的资产数量无关；否则把 validation-unseen-mother 扩容会
    隐式改变 checkpoint objective，而不是只提高同一指标的统计精度。
    """

    metrics = ("density", "kappa")
    baseline = selection_baseline(
        {
            "unseen_variant_set": {"density": 9.0, "kappa": 9.0},
            "unseen_mother": {"density": 9.0, "kappa": 9.0},
        },
        metrics,
        teacher_baselines={"density": 1.0, "kappa": 2.0},
    )
    score = normalized_validation_score(
        {
            "unseen_variant_set": {"density": 0.5, "kappa": 1.0},
            "unseen_mother": {"density": 1.0, "kappa": 2.0},
        },
        baseline,
        metrics,
    )

    assert score == pytest.approx(0.75)


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
        validate_asset_manifest_isolation(
            {
                "train": [train],
                "validation": {"unseen_variant_set": [validation], "unseen_mother": []},
                "evaluation": {},
            }
        )


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
