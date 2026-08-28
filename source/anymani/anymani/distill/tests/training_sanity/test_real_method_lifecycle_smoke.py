r"""真实 generated assets 上的正式 Method lifecycle 最小 smoke。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from anymani.assets.bank import HandContainer, HandContainerCfg
from anymani.distill.methods.multi_anchor_gaussian_implicit_field import (
    MultiAnchorGaussianMethod,
    MultiAnchorGaussianMethodCfg,
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
    SO2AnchorFrontendCfg,
)
from anymani.distill.representations.geometry import GeometryRepresentationCfg
from anymani.distill.representations.queries.spatial_sampling import SpatialQuerySamplerCfg
from anymani.distill.representations.sources.geometry_source import AnchorBankCfg, GeometrySourceCfg
from anymani.distill.representations.targets.geometry_field import GeometryFieldTargetCfg
from anymani.distill.ssl.data import HandAssetCatalogCfg
from anymani.distill.ssl.experiment import EmbodimentPretrainCfg, resolved_config_dict
from anymani.distill.ssl.post_training import (
    EmbodimentEvaluationCfg,
    EvaluationCfg,
    EvaluationRun,
    EvaluationRunCfg,
    resolved_post_training_config_dict,
)
from anymani.distill.ssl.runtime.lifecycle import fit_embodiment_pretrain
from anymani.distill.ssl.runtime.post_training import evaluate_checkpoint
from anymani.distill.ssl.runtime.pretrainer import (
    AdamWCfg,
    EmbodimentPretrainTrainer,
    EmbodimentPretrainTrainerCfg,
)
from anymani.distill.ssl.runtime.run import PretrainRun, PretrainRunCfg
from anymani.distill.ssl.runtime.sampling import OnlineSamplingCfg

pytestmark = pytest.mark.training_sanity

_ROOT = Path(__file__).resolve().parents[3] / "assets" / "generated" / "2026-08-19_15-10-48"
_PATHS = {
    "train-a": _ROOT / "single_palm_allegro" / "left_t3_i3_m2_r2",
    "train-b": _ROOT / "single_palm_allegro" / "left_t3_i3_m3_r2",
    "evaluation-variant": (
        _ROOT / "single_palm_allegro" / "left_t3_i3_m2_r2" / "2026-08-20_05-01-58" / "018e25d8"
    ),
    "evaluation-mother": _ROOT / "single_palm_allegro" / "left_t3_i4_m2_r3",
}
_requires_assets = pytest.mark.skipif(
    not all(path.is_dir() for path in _PATHS.values()),
    reason="formal generated asset bank is not available locally",
)


class _Dataset:
    source_path = Path("real-method-lifecycle-smoke.yaml")
    source_sha256 = "real-method-lifecycle-smoke"

    def __init__(self, train, evaluation) -> None:
        self.train = SimpleNamespace(records=tuple(SimpleNamespace(container=item, provenance={}) for item in train))
        self.evaluation = {
            name: SimpleNamespace(records=tuple(SimpleNamespace(container=item, provenance={}) for item in assets))
            for name, assets in evaluation.items()
        }

    @staticmethod
    def config_dict() -> dict[str, str]:
        return {"schema_version": "smoke"}


class _Data:
    def __init__(self, catalog) -> None:
        self.catalog = catalog

    def resolve_train(self):
        return self.catalog

    def resolve_evaluation(self):
        return self.catalog


def _container(path: Path) -> HandContainer:
    return HandContainer.from_cfg(HandContainerCfg(path=path), require_geometry_semantics=True)


def _catalog():
    containers = {name: _container(path) for name, path in _PATHS.items()}
    train = (containers["train-a"], containers["train-b"])
    evaluation = {
        "unseen_variant_set": (containers["evaluation-variant"],),
        "unseen_mother": (containers["evaluation-mother"],),
        "official_zero_shot": (),
    }
    return SimpleNamespace(
        dataset=_Dataset(train, evaluation),
        train=train,
        evaluation=evaluation,
        training_dataset_identity=lambda: {
            "schema_version": "1.0.0",
            "source_sha256": "real-method-lifecycle-smoke",
            "train_asset_count": 2,
            "train_asset_axis_sha256": "real-method-lifecycle-smoke-axis",
        },
    )


def _method_cfg() -> MultiAnchorGaussianMethodCfg:
    return MultiAnchorGaussianMethodCfg(
        representation=GeometryRepresentationCfg(
            source=GeometrySourceCfg(
                home_points_per_owner=8,
                home_surface_oversample_factor=2,
                anchors=AnchorBankCfg(bank_size=1, anchors_per_finger=2),
            ),
            query=SpatialQuerySamplerCfg(query_count=8),
            target=GeometryFieldTargetCfg(
                train_active_per_joint=1,
                train_zero_per_joint=1,
                fixed_active_per_joint=4,
                fixed_zero_per_joint=4,
            ),
        ),
        model=GeometrySSLModelCfg(
            encoder=GeometryEncoderCfg(
                frontend=SO2AnchorFrontendCfg(relation_width=16, home_width=16, screw_width=12),
                backbone=GraphBiasedTransformerCfg(
                    hidden_width=128,  # 事后 PCA smoke 需要 canonical Z width 与 32/64/96/128 ranks
                    layers=1,
                    attention_heads=4,
                    feedforward_width=128,
                    dropout=0.0,
                ),
            ),
            ssl_decoders=GeometrySSLDecoderCfg(
                density=ScalarSigmaFiLMDensityDecoderCfg(hidden_width=32, residual_blocks=1),
                sensitivity=DistanceSensitivityDecoderCfg(hidden_width=32, residual_blocks=2),
            ),
        ),
    )


def _trainer_cfg(*, max_epochs: int = 1, checkpoint_every_epochs: int = 1) -> EmbodimentPretrainTrainerCfg:
    return EmbodimentPretrainTrainerCfg(
        sampling=OnlineSamplingCfg(
            assets_per_minibatch=2,
            q_per_asset_per_minibatch=1,
            shuffle_assets=False,
            seed=71,
        ),
        max_epochs=max_epochs,
        num_minibatches=1,
        mini_epochs=1,
        microbatch_size=2,
        optimizer=AdamWCfg(learning_rate=1.0e-3, weight_decay=0.0),
        checkpoint_every_epochs=checkpoint_every_epochs,
    )


@_requires_assets
@pytest.mark.skipif(not torch.cuda.is_available(), reason="real Method lifecycle requires CUDA/Warp")
def test_real_method_pretrain_and_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实 teacher 完成 run-local baseline、FairGrad update 与显式 held-out evaluation。"""

    from anymani.distill.ssl.runtime import lifecycle

    monkeypatch.setattr(lifecycle, "_worktree_fingerprint", lambda: (False, ""))
    catalog = _catalog()
    data_cfg = HandAssetCatalogCfg(manifest="real-method-lifecycle-smoke.yaml")
    method_cfg = _method_cfg()
    trainer_cfg = _trainer_cfg()
    source_cache_root = tmp_path / "source-cache"
    preparer = MultiAnchorGaussianMethod(method_cfg)
    preparer.configure_source_artifacts(
        root=str(source_cache_root),
        mode="read-write",
        dataset_manifest_sha256=str(catalog.dataset.source_sha256),
        producer_device="cuda:0",
    )
    try:
        preparer.prepare(catalog, role="train", device=torch.device("cuda:0"), dtype=torch.float32)
        source_summary = preparer.prepare_source_artifacts(
            device=torch.device("cuda:0"),
            dtype=torch.float32,
        )
    finally:
        preparer.close()
    assert source_summary["base_count"] == 2
    assert source_summary["anchor_shard_count"] == 2
    pretrain_run_cfg = PretrainRunCfg(
        seed=71,
        deterministic_algorithms=False,
        source_cache_root=str(source_cache_root),
        source_cache_mode="readonly",
    )
    pretrain_root = EmbodimentPretrainCfg(
        data=data_cfg,
        method=method_cfg,
        trainer=trainer_cfg,
        run=pretrain_run_cfg,
    )
    pretrain_dir = tmp_path / "pretrain"
    fit_embodiment_pretrain(
        trainer=EmbodimentPretrainTrainer(trainer_cfg),
        data=_Data(catalog),
        method=MultiAnchorGaussianMethod(method_cfg),
        run=PretrainRun(pretrain_run_cfg),
        output_dir_override=pretrain_dir,
        resolved_config=resolved_config_dict(pretrain_root),
    )

    assert (pretrain_dir / "run_teacher_baselines.yaml").is_file()
    assert (pretrain_dir / "metrics_finalized.jsonl").is_file()
    assert (pretrain_dir / "train_dense_update_00000001.npz").is_file()
    assert (pretrain_dir / "checkpoints" / "epoch_000000.pt").is_file()
    assert (pretrain_dir / "checkpoints" / "epoch_000001.pt").is_file()
    assert (pretrain_dir / "checkpoints" / "last.pt").is_file()
    assert (pretrain_dir / "retained_encoder.pt").is_file()
    assert not (pretrain_dir / "checkpoints" / "best.pt").exists()

    evaluation_cfg = EvaluationCfg(
        q_per_asset=1,
        assets_per_minibatch=1,
        q_per_asset_per_minibatch=1,
        bootstrap_replicates=2,
        max_resident_assets=2,
        source_cache_root=str(source_cache_root),
        source_cache_mode="read-write",  # 首次 held-out evaluation 只补建自己的 role index/objects
    )
    evaluation_run_cfg = EvaluationRunCfg(
        checkpoint=str(pretrain_dir / "checkpoints" / "last.pt"),
        seed=71,
        deterministic_algorithms=False,
    )
    evaluation_root = EmbodimentEvaluationCfg(
        data=data_cfg,
        method=method_cfg,
        evaluation=evaluation_cfg,
        run=evaluation_run_cfg,
    )
    evaluation_dir = tmp_path / "evaluation"
    evaluate_checkpoint(
        data=_Data(catalog),
        method=MultiAnchorGaussianMethod(method_cfg),
        config=evaluation_cfg,
        run=EvaluationRun(evaluation_run_cfg),
        output_dir_override=evaluation_dir,
        resolved_config=resolved_post_training_config_dict(evaluation_root),
    )
    assert (evaluation_dir / "evaluation.yaml").is_file()
    evaluation = yaml.safe_load((evaluation_dir / "evaluation.yaml").read_text(encoding="utf-8"))
    assert evaluation["analyses"] == []  # 默认 explicit evaluation 只运行 core，不偷跑昂贵 ablation
    core_suites = 0
    for suite in evaluation["suites"].values():
        if "strata" not in suite:  # official_zero_shot 可显式为空
            continue
        core_suites += 1
        assert "joint_sign_observable_audit" not in suite["strata"]  # joint-sign 双前向只属于显式 ablations
        assert suite["ablations"] is None
    assert core_suites == 2
    assert not (evaluation_dir / "training_morphology_q_bank.yaml").exists()
    assert not (evaluation_dir / "retained_encoder.pt").exists()


@_requires_assets
@pytest.mark.skipif(not torch.cuda.is_available(), reason="compiled latent diagnostic requires CUDA/Warp")
def test_real_method_compiled_latent_diagnostic_at_epoch_four(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""真实 BF16/compile 生命周期必须在第 4 个 epoch 闭合 unified-$Z$ 诊断分支。

    该测试使用与真实 Method lifecycle 相同的 generated assets、source artifact 生产和 compiled
    learned model，但把训练预算压缩为 4 epochs × 1 minibatch。第 4 个 epoch 的最后一个 update
    会触发 `collect_z_gradients=True`，从而覆盖正式 256-epoch run 中之前未被 1-epoch smoke 覆盖的
    latent diagnostic backward 路径。该诊断只检查 $\partial L/\partial Z$ 证据，不改变参数 update。
    """

    from anymani.distill.ssl.runtime import lifecycle

    monkeypatch.setattr(lifecycle, "_worktree_fingerprint", lambda: (False, ""))
    catalog = _catalog()
    data_cfg = HandAssetCatalogCfg(manifest="real-method-lifecycle-smoke.yaml")
    method_cfg = _method_cfg()
    trainer_cfg = _trainer_cfg(max_epochs=4, checkpoint_every_epochs=4)
    source_cache_root = tmp_path / "source-cache"

    # 先在同一 source root 发布 tiny generated-assets smoke 所需的 canonical base 与 anchor shard，
    # 再以 readonly 身份交给 lifecycle；这复现正式 run 的 auto-prepare -> readonly ownership 边界。
    preparer = MultiAnchorGaussianMethod(method_cfg)
    preparer.configure_source_artifacts(
        root=str(source_cache_root),
        mode="read-write",
        dataset_manifest_sha256=str(catalog.dataset.source_sha256),
        producer_device="cuda:0",
    )
    try:
        preparer.prepare(catalog, role="train", device=torch.device("cuda:0"), dtype=torch.float32)
        source_summary = preparer.prepare_source_artifacts(device=torch.device("cuda:0"), dtype=torch.float32)
    finally:
        preparer.close()
    assert source_summary["base_count"] == 2
    assert source_summary["anchor_shard_count"] == 2

    # lifecycle 从 epoch 0 开始执行四个真实 optimizer updates，第 4 个 update 产生 Z-gradient evidence。
    pretrain_run_cfg = PretrainRunCfg(
        seed=71,
        deterministic_algorithms=False,
        source_cache_root=str(source_cache_root),
        source_cache_mode="readonly",
    )
    pretrain_root = EmbodimentPretrainCfg(
        data=data_cfg,
        method=method_cfg,
        trainer=trainer_cfg,
        run=pretrain_run_cfg,
    )
    pretrain_dir = tmp_path / "compiled-latent-diagnostic"
    fit_embodiment_pretrain(
        trainer=EmbodimentPretrainTrainer(trainer_cfg),
        data=_Data(catalog),
        method=MultiAnchorGaussianMethod(method_cfg),
        run=PretrainRun(pretrain_run_cfg),
        output_dir_override=pretrain_dir,
        resolved_config=resolved_config_dict(pretrain_root),
    )

    metrics = [
        json.loads(line)
        for line in (pretrain_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    diagnostic_updates = [
        entry
        for entry in metrics
        if entry["epoch"] == 4 and "raw/rho_norm" in entry["z_gradient_evidence"]
    ]
    assert len(diagnostic_updates) == 1
    assert diagnostic_updates[0]["z_gradient_evidence"]["raw/rho_norm"] > 0.0
    assert diagnostic_updates[0]["z_gradient_evidence"]["raw/kappa_norm"] > 0.0
    assert (pretrain_dir / "checkpoints" / "epoch_000004.pt").is_file()
    assert (pretrain_dir / "COMPLETE").is_file()
