r"""真实 generated assets 上的正式 Method lifecycle 最小 smoke。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
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
    GeometryLatentHeadsCfg,
    SO2AnchorFrontendCfg,
)
from anymani.distill.representations.geometry import GeometryRepresentationCfg
from anymani.distill.representations.queries.spatial_sampling import SpatialQuerySamplerCfg
from anymani.distill.representations.sources.geometry_source import AnchorBankCfg, GeometrySourceCfg
from anymani.distill.representations.targets.geometry_field import GeometryFieldTargetCfg
from anymani.distill.ssl.data import HandAssetCatalogCfg
from anymani.distill.ssl.experiment import EmbodimentPretrainCfg, resolved_config_dict
from anymani.distill.ssl.runtime.lifecycle import fit_embodiment_pretrain
from anymani.distill.ssl.runtime.pretrainer import (
    AdamWCfg,
    EmbodimentPretrainTrainer,
    EmbodimentPretrainTrainerCfg,
    FinalEvaluationCfg,
    ValidationCfg,
)
from anymani.distill.ssl.runtime.run import PretrainRun, PretrainRunCfg
from anymani.distill.ssl.runtime.sampling import OnlineSamplingCfg

pytestmark = pytest.mark.training_sanity

_ROOT = Path(__file__).resolve().parents[3] / "assets" / "generated" / "2026-08-19_15-10-48"
_PATHS = {
    "train-a": _ROOT / "single_palm_allegro" / "left_t3_i3_m2_r2",
    "train-b": _ROOT / "single_palm_allegro" / "left_t3_i3_m3_r2",
    "validation-variant": (
        _ROOT / "single_palm_allegro" / "left_t3_i3_m3_r2" / "2026-08-20_04-46-25" / "4cbf4162"
    ),
    "validation-mother": _ROOT / "single_palm_allegro" / "left_t3_i4_m2_r2",
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

    def __init__(self, train, validation, evaluation) -> None:
        self.train = SimpleNamespace(records=tuple(SimpleNamespace(container=item, provenance={}) for item in train))
        self.validation = {
            name: SimpleNamespace(records=tuple(SimpleNamespace(container=item, provenance={}) for item in assets))
            for name, assets in validation.items()
        }
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

    def resolve(self):
        return self.catalog


def _container(path: Path) -> HandContainer:
    return HandContainer.from_cfg(HandContainerCfg(path=path), require_geometry_semantics=True)


def _catalog():
    containers = {name: _container(path) for name, path in _PATHS.items()}
    train = (containers["train-a"], containers["train-b"])
    validation = {
        "unseen_variant_set": (containers["validation-variant"],),
        "unseen_mother": (containers["validation-mother"],),
    }
    evaluation = {
        "unseen_variant_set": (containers["evaluation-variant"],),
        "unseen_mother": (containers["evaluation-mother"],),
        "official_zero_shot": (),
    }
    return SimpleNamespace(
        dataset=_Dataset(train, validation, evaluation),
        train=train,
        validation=validation,
        evaluation=evaluation,
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
                validation_active_per_joint=4,
                validation_zero_per_joint=4,
            ),
        ),
        model=GeometrySSLModelCfg(
            encoder=GeometryEncoderCfg(
                frontend=SO2AnchorFrontendCfg(relation_width=16, home_width=16, screw_width=12),
                backbone=GraphBiasedTransformerCfg(
                    hidden_width=32,
                    layers=1,
                    attention_heads=4,
                    feedforward_width=64,
                    dropout=0.0,
                ),
                heads=GeometryLatentHeadsCfg(zero_order_width=24, first_order_width=12),
            ),
            ssl_decoders=GeometrySSLDecoderCfg(
                density=ScalarSigmaFiLMDensityDecoderCfg(hidden_width=32, residual_blocks=1),
                sensitivity=DistanceSensitivityDecoderCfg(coefficient_hidden_width=32),
            ),
        ),
    )


def _trainer_cfg() -> EmbodimentPretrainTrainerCfg:
    return EmbodimentPretrainTrainerCfg(
        sampling=OnlineSamplingCfg(
            assets_per_minibatch=2,
            q_per_asset_per_minibatch=1,
            shuffle_assets=False,
            seed=71,
        ),
        num_minibatches=1,
        mini_epochs=1,
        validation=ValidationCfg(
            q_per_asset=1,
            assets_per_minibatch=1,
            q_per_asset_per_minibatch=1,
            every_optimizer_updates=1,
        ),
        final_evaluation=FinalEvaluationCfg(
            q_per_asset=1,
            assets_per_minibatch=1,
            q_per_asset_per_minibatch=1,
            bootstrap_replicates=2,
        ),
        optimizer=AdamWCfg(learning_rate=1.0e-3, weight_decay=0.0),
        gradient_accumulation_steps=1,
        max_resident_assets=2,
        checkpoint_every_updates=1,
        run_safety_step_limit=4,
    )


@_requires_assets
@pytest.mark.skipif(not torch.cuda.is_available(), reason="real Method lifecycle requires CUDA/Warp")
def test_real_method_calibration_pretrain_checkpoint_and_final_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实 teacher 完成 calibration、一次 update、best reload、retained export 与 unseen evaluation。"""

    from anymani.distill.ssl.runtime import lifecycle

    monkeypatch.setattr(lifecycle, "_worktree_fingerprint", lambda: (False, ""))
    catalog = _catalog()
    data_cfg = HandAssetCatalogCfg(manifest="real-method-lifecycle-smoke.yaml")
    method_cfg = _method_cfg()
    trainer_cfg = _trainer_cfg()
    calibration_run_cfg = PretrainRunCfg(seed=71, phase="calibrate_objectives", deterministic_algorithms=False)
    calibration_root = EmbodimentPretrainCfg(
        data=data_cfg,
        method=method_cfg,
        trainer=trainer_cfg,
        run=calibration_run_cfg,
    )
    calibration_dir = tmp_path / "calibration"
    fit_embodiment_pretrain(
        trainer=EmbodimentPretrainTrainer(trainer_cfg),
        data=_Data(catalog),
        method=MultiAnchorGaussianMethod(method_cfg),
        run=PretrainRun(calibration_run_cfg),
        output_dir_override=calibration_dir,
        resolved_config=resolved_config_dict(calibration_root),
    )

    artifact = calibration_dir / "loss_calibration.yaml"
    pretrain_run_cfg = PretrainRunCfg(
        seed=71,
        phase="pretrain",
        calibration_artifact=str(artifact),
        deterministic_algorithms=False,
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

    assert artifact.is_file()
    assert (pretrain_dir / "checkpoints" / "best.pt").is_file()
    assert (pretrain_dir / "retained_artifact.pt").is_file()
    assert (pretrain_dir / "final_evaluation.yaml").is_file()
