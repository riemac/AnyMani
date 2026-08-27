"""真实 mother 固定在线 batch 的显式 tiny-overfit sanity。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from anymani.assets.bank.hand_bank import HandBank, HandBankCfg
from anymani.distill.methods.multi_anchor_gaussian_implicit_field.batch import (
    attach_static_evidence,
    pad_online_geometry_samples,
)
from anymani.distill.methods.multi_anchor_gaussian_implicit_field.config import MultiAnchorGaussianObjectivesCfg
from anymani.distill.methods.multi_anchor_gaussian_implicit_field.context import MultiAnchorObjectiveContext
from anymani.distill.methods.multi_anchor_gaussian_implicit_field.objectives import (
    evaluate_objectives,
)
from anymani.distill.methods.multi_anchor_gaussian_implicit_field.state_measure import SobolJointSampler
from anymani.distill.models.backbones.geometry_transformer import GraphBiasedTransformerCfg
from anymani.distill.models.decoders.representations.implicit_field import (
    DistanceSensitivityDecoderCfg,
    GeometrySSLDecoderCfg,
    ScalarSigmaFiLMDensityDecoderCfg,
)
from anymani.distill.models.geometry_ssl import GeometrySSLModel, GeometrySSLModelCfg
from anymani.distill.models.input_adapters.geometry import (
    GeometryEncoderCfg,
    GeometryPaddingCfg,
    SO2AnchorFrontendCfg,
)
from anymani.distill.representations.geometry import GeometryRepresentation, GeometryRepresentationCfg
from anymani.distill.representations.queries.spatial_sampling import SpatialQuerySamplerCfg
from anymani.distill.representations.sources.geometry_source import AnchorBankCfg, GeometrySourceCfg
from anymani.distill.representations.targets.geometry_field import GeometryFieldTargetCfg

pytestmark = pytest.mark.training_sanity

MOTHER = (
    Path(__file__).resolve().parents[3]
    / "assets"
    / "generated"
    / "2026-08-19_15-10-48"
    / "single_palm_allegro"
    / "left_t3_i3_m2_r2"
)
_requires_local_mother = pytest.mark.skipif(
    not MOTHER.is_dir(),
    reason="generated LEAP mother asset is a local research artifact",
)


def _prediction_and_losses(model: GeometrySSLModel, batch):
    """对固定 teacher batch 返回当前 raw density 与 scaled-kappa 训练目标。"""

    q = batch.q.detach()
    prediction = model(
        q,
        batch.evidence,
        batch.queries.query_points_h,
        batch.field_targets.bandwidths,
        owner_index=batch.sensitivity_targets.owner_index,
        query_index=batch.sensitivity_targets.query_index,
        joint_index=batch.sensitivity_targets.joint_index,
    )
    objectives = evaluate_objectives(
        MultiAnchorObjectiveContext(prediction=prediction, batch=batch),
        MultiAnchorGaussianObjectivesCfg(),
    )
    density = objectives["density"].metrics["loss"]
    kappa = objectives["kappa"].metrics["loss"]
    return prediction, density, kappa


def _diagnostic_values(model: GeometrySSLModel, batch) -> dict[str, float]:
    """分别观察 density、active κ 与 structural-zero false positive。"""

    prediction, density, kappa = _prediction_and_losses(model, batch)
    targets = batch.sensitivity_targets
    active = targets.valid_mask & targets.active_mask
    structural_zero = targets.valid_mask & ~targets.active_mask
    return {
        "density": float(density.detach()),
        "kappa": float(kappa.detach()),
        "active_kappa": float((prediction.kappa[active] - targets.kappa[active]).square().mean().detach()),
        "zero_false_positive": float(prediction.kappa[structural_zero].square().mean().detach()),
    }


@_requires_local_mother
def test_real_mother_fixed_batch_loss_decreases() -> None:
    """验证双项 method objective 能在一份真实几何 batch 上被优化，而非只完成 backward。"""

    if not torch.cuda.is_available():
        pytest.skip("real mother online teacher requires CUDA Warp")
    torch.manual_seed(53)
    container = HandBank(
        HandBankCfg(
            source_mode="post_mutate",
            selection_mode="explicit",
            containers=(MOTHER,),
            require_geometry_semantics=True,
        )
    ).resolve().assets[0]
    query_config = SpatialQuerySamplerCfg(query_count=8)  # 最小合法 4/2/2 分层，shell 保持内外严格各半
    representation = GeometryRepresentation(
        GeometryRepresentationCfg(
            source=GeometrySourceCfg(
                home_points_per_owner=8,
                anchors=AnchorBankCfg(bank_size=1, anchors_per_finger=2),
            ),
            query=query_config,
            target=GeometryFieldTargetCfg(train_active_per_joint=2, train_zero_per_joint=1),
        )
    )
    source = representation.materialize_source(container)
    state = representation.to_device(source, device="cuda:0", dtype=torch.float32)
    q = SobolJointSampler(source.spec_cpu, seed=53).draw(1, device="cuda:0", dtype=torch.float32)
    physical = representation.sample(state, q, sampling_seed=53, q_index=torch.zeros(1, dtype=torch.long), anchor_index=0)
    batch = pad_online_geometry_samples(
        list(
            attach_static_evidence(
                physical,
                source=source,
                spec=state.spec,
                anchors=source.anchor_bank[0],
                device="cuda:0",
                dtype=torch.float32,
            )
        ),
        padding=GeometryPaddingCfg(),
    )
    model = GeometrySSLModel(
        GeometrySSLModelCfg(
            encoder=GeometryEncoderCfg(
                frontend=SO2AnchorFrontendCfg(relation_width=16, home_width=16, screw_width=12),
                backbone=GraphBiasedTransformerCfg(
                    hidden_width=32,
                    layers=1,
                    attention_heads=4,
                    feedforward_width=64,
                    dropout=0.0,
                ),
            ),
            ssl_decoders=GeometrySSLDecoderCfg(
                density=ScalarSigmaFiLMDensityDecoderCfg(hidden_width=32, residual_blocks=1),
                sensitivity=DistanceSensitivityDecoderCfg(hidden_width=32, residual_blocks=2),
            ),
        )
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3)
    initial = _diagnostic_values(model, batch)
    joint_projection_received_gradient = False

    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        _, density_loss, kappa_loss = _prediction_and_losses(model, batch)
        loss = density_loss + kappa_loss
        loss.backward()
        joint_gradient = model.sensitivity_decoder.joint_projection.weight.grad
        joint_projection_received_gradient |= bool(
            joint_gradient is not None and torch.count_nonzero(joint_gradient).detach().cpu() > 0
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
    final = _diagnostic_values(model, batch)

    assert joint_projection_received_gradient, "kappa JOINT projection never received a non-zero gradient"
    assert final["density"] < 0.5 * initial["density"], (initial, final)
    assert final["active_kappa"] < 0.5 * initial["active_kappa"], (initial, final)
    assert final["zero_false_positive"] < 0.5 * initial["zero_false_positive"], (initial, final)
