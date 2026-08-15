"""真实 mother 固定在线 batch 的显式 tiny-overfit sanity。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from anymani.assets.bank.hand_bank import HandBank, HandBankCfg
from anymani.distill.models.backbones.geometry_transformer import GraphBiasedTransformerCfg
from anymani.distill.models.decoders.representations.implicit_field import (
    DistanceSensitivityDecoderCfg,
    GeometrySSLDecoderCfg,
    ScalarSigmaFiLMDensityDecoderCfg,
)
from anymani.distill.models.geometry_ssl import GeometrySSLModel, GeometrySSLModelCfg
from anymani.distill.models.input_adapters.geometry import (
    GeometryEncoderCfg,
    GeometryLatentHeadsCfg,
    GeometryPaddingCfg,
    SO2AnchorFrontendCfg,
)
from anymani.distill.objectives.representations.field_reconstruction import (
    GeometryFieldObjective,
    GeometryFieldObjectiveCfg,
)
from anymani.distill.representations.geometry import (
    GeometryRepresentation,
    GeometryRepresentationCfg,
    OnlineGeometryBatcher,
)
from anymani.distill.representations.queries.spatial_sampling import SpatialQuerySamplerCfg
from anymani.distill.representations.sources.geometry_source import GeometrySourceCfg
from anymani.distill.representations.targets.geometry_field import GeometryFieldTargetCfg

pytestmark = pytest.mark.training_sanity

MOTHER = (
    Path(__file__).resolve().parents[3]
    / "assets"
    / "generated"
    / "2026-08-12_18-16-48"
    / "single_palm_leap"
    / "right_t4_i4_m4_r4"
)
_requires_local_mother = pytest.mark.skipif(
    not MOTHER.is_dir(),
    reason="generated LEAP mother asset is a local research artifact",
)


def _loss(
    model: GeometrySSLModel,
    objective: GeometryFieldObjective,
    batch,
) -> torch.Tensor:
    """对固定 teacher batch 重新建立物理 q Sobolev 图。"""

    q = batch.q.detach().requires_grad_(True)
    prediction = model(
        q,
        batch.evidence,
        batch.queries.query_points_h,
        batch.field_targets.bandwidths,
        owner_index=batch.sensitivity_targets.owner_index,
        query_index=batch.sensitivity_targets.query_index,
        joint_index=batch.sensitivity_targets.joint_index,
    )
    return objective(
        q=q,
        density_prediction=prediction.density,
        kappa_prediction=prediction.kappa,
        field_targets=batch.field_targets,
        sensitivity_targets=batch.sensitivity_targets,
    ).total


@_requires_local_mother
def test_real_mother_fixed_batch_loss_decreases() -> None:
    """验证完整六项联合目标能在一份真实几何 batch 上被优化，而非只完成 backward。"""

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
            source=GeometrySourceCfg(home_points_per_owner=8, anchors_per_finger=2),
            query=query_config,
            target=GeometryFieldTargetCfg(edges_per_owner=1),
        )
    )
    source = representation.materialize_source(container)
    state = representation.to_device(source, device="cuda:0", dtype=torch.float32)
    batch = OnlineGeometryBatcher(
        [state],
        seed=53,
        query_config=query_config,
        target_config=representation.config.target,
        padding=GeometryPaddingCfg(),
    ).sample(batch_size=1, step=0)
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
                heads=GeometryLatentHeadsCfg(zero_order_width=24, first_order_width=12),
            ),
            ssl_decoders=GeometrySSLDecoderCfg(
                density=ScalarSigmaFiLMDensityDecoderCfg(hidden_width=32, residual_blocks=1),
                sensitivity=DistanceSensitivityDecoderCfg(coefficient_hidden_width=32),
            ),
        )
    ).cuda()
    objective = GeometryFieldObjective(GeometryFieldObjectiveCfg())
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3)
    initial = float(_loss(model, objective, batch).detach())

    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, objective, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
    final = float(_loss(model, objective, batch).detach())

    assert final < 0.75 * initial, f"fixed-batch SSL loss did not decrease enough: initial={initial}, final={final}"
