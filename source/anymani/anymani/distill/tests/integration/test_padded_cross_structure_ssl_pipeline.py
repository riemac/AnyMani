"""不同 JOINT/owner 数手型共享一次前向的 synthetic 闭环。"""

from __future__ import annotations

import torch
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
    StaticGeometryEvidence,
)
from anymani.distill.objectives.representations.field_reconstruction import (
    GeometryFieldObjective,
    GeometryFieldObjectiveCfg,
)
from anymani.distill.representations.geometry import OnlineGeometrySample, pad_online_geometry_samples
from anymani.distill.representations.queries.spatial_sampling import SpatialQueryBatch
from anymani.distill.representations.targets.field_samples import FieldTargetBatch, SensitivityTargetBatch


def _evidence(joint_count: int) -> StaticGeometryEvidence:
    """构造 PALM–JOINT×N–TIP chain，不引入资产命名捷径。"""

    owner_count = joint_count + 2
    positions = torch.arange(owner_count, dtype=torch.float64) * 0.03
    surface = torch.stack(
        (
            torch.stack((positions, torch.full_like(positions, -0.01), torch.zeros_like(positions)), dim=-1),
            torch.stack((positions, torch.full_like(positions, 0.01), torch.zeros_like(positions)), dim=-1),
        ),
        dim=1,
    )
    distance = torch.abs(torch.arange(owner_count)[:, None] - torch.arange(owner_count)[None, :])
    parent = torch.where(
        torch.arange(owner_count)[:, None] >= torch.arange(owner_count)[None, :], distance, torch.full_like(distance, 4)
    ).clamp(max=4)
    child = parent.transpose(0, 1).contiguous()
    screws = torch.zeros(joint_count, 6, dtype=torch.float64)
    screws[:, 2] = 1.0
    screws[:, 4] = -positions[1 : joint_count + 1]
    return StaticGeometryEvidence(
        anchors=torch.tensor([[-0.03, -0.02, 0.0], [0.03, 0.02, 0.0]], dtype=torch.float64),
        home_surface_points=surface,
        home_surface_mask=torch.ones(owner_count, 2, dtype=torch.bool),
        palm_normal=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        space_screws=screws,
        q_home=torch.zeros(joint_count, dtype=torch.float64),
        entity_role=torch.tensor([0, *([1] * joint_count), 2], dtype=torch.long),
        entity_joint_index=torch.tensor([-1, *range(joint_count), -1], dtype=torch.long),
        joint_entity_index=torch.arange(1, joint_count + 1, dtype=torch.long),
        shortest_path=distance.clamp(max=4).to(torch.long),
        parent_direction=parent.to(torch.long),
        child_direction=child.to(torch.long),
    )


def _sample(joint_count: int, asset_id: str) -> OnlineGeometrySample:
    """构造一份可由 mask 完整审计的单样本 query/target。"""

    owner_count = joint_count + 2
    query_count = 4
    bandwidths = torch.tensor([0.01, 0.03], dtype=torch.float64)
    q = torch.linspace(-0.2, 0.3, joint_count, dtype=torch.float64).unsqueeze(0)
    query_points = torch.zeros(1, owner_count, query_count, 3, dtype=torch.float64)
    query_points[..., 0] = torch.linspace(-0.05, 0.08, query_count, dtype=torch.float64)
    stratum = torch.tensor([0, 0, 1, 2], dtype=torch.long).reshape(1, 1, query_count).expand(1, owner_count, -1)
    queries = SpatialQueryBatch(
        query_points,
        stratum,
        torch.full_like(stratum, -1),
        torch.full_like(stratum, -1),
    )
    distance = torch.full((1, owner_count, query_count), 0.02, dtype=torch.float64)
    density = torch.exp(-0.5 * (distance.unsqueeze(-1) / bandwidths).square())
    field = FieldTargetBatch(
        query_points=query_points,
        query_stratum=stratum,
        distance=distance,
        density=density,
        valid_mask=torch.ones_like(distance, dtype=torch.bool),
        owner_role=torch.tensor([0, *([1] * joint_count), 2], dtype=torch.long),
        bandwidths=bandwidths,
        provenance={"frame": "h", "length_unit": "m"},
    )
    owner_index = torch.arange(1, joint_count + 1, dtype=torch.long)
    sensitivity = SensitivityTargetBatch(
        owner_index=owner_index,
        query_index=torch.arange(joint_count, dtype=torch.long) % query_count,
        joint_index=torch.arange(joint_count, dtype=torch.long),
        ancestor_mask=torch.ones(joint_count, dtype=torch.bool),
        closest_point=torch.zeros(1, joint_count, 3, dtype=torch.float64),
        closest_source=torch.zeros(1, joint_count, dtype=torch.long),
        uniqueness_margin=torch.ones(1, joint_count, dtype=torch.float64),
        kappa=torch.zeros(1, joint_count, dtype=torch.float64),
        field_sensitivity=torch.zeros(1, joint_count, bandwidths.numel(), dtype=torch.float64),
        valid_mask=torch.ones(1, joint_count, dtype=torch.bool),
        provenance={"frame": "h", "distance_unit": "m", "joint_unit": "rad"},
    )
    return OnlineGeometrySample(asset_id, q, _evidence(joint_count), queries, field, sensitivity)


def test_padded_cross_structure_model_objective_and_backward() -> None:
    """1-DOF 与 2-DOF 手型共享 forward/loss，padding 不进入有效归一化。"""

    torch.manual_seed(41)
    padding = GeometryPaddingCfg(max_joint_count=20, max_tip_count=5, max_graph_distance=4)
    batch = pad_online_geometry_samples([_sample(1, "one"), _sample(2, "two")], padding=padding)
    q = batch.q.detach().requires_grad_(True)
    model = GeometrySSLModel(
        GeometrySSLModelCfg(
            encoder=GeometryEncoderCfg(
                frontend=SO2AnchorFrontendCfg(
                    relation_width=16,
                    home_width=16,
                    screw_width=12,
                    length_scale_m=0.1,
                ),
                backbone=GraphBiasedTransformerCfg(
                    hidden_width=32,
                    layers=1,
                    attention_heads=4,
                    feedforward_width=64,
                    dropout=0.0,
                    max_graph_distance=4,
                ),
                heads=GeometryLatentHeadsCfg(zero_order_width=24, first_order_width=12),
            ),
            ssl_decoders=GeometrySSLDecoderCfg(
                density=ScalarSigmaFiLMDensityDecoderCfg(hidden_width=32, residual_blocks=1),
                sensitivity=DistanceSensitivityDecoderCfg(coefficient_hidden_width=32),
            ),
        ),
    ).to(dtype=torch.float64)
    prediction = model(
        q,
        batch.evidence,
        batch.queries.query_points_h,
        batch.field_targets.bandwidths,
        owner_index=batch.sensitivity_targets.owner_index,
        query_index=batch.sensitivity_targets.query_index,
        joint_index=batch.sensitivity_targets.joint_index,
    )
    terms = GeometryFieldObjective(GeometryFieldObjectiveCfg())(
        q=q,
        density_prediction=prediction.density,
        kappa_prediction=prediction.kappa,
        field_targets=batch.field_targets,
        sensitivity_targets=batch.sensitivity_targets,
    )
    terms.total.backward()

    assert batch.q.shape == (2, 20)
    assert batch.evidence.entity_valid_mask.shape == (2, 26)
    assert batch.sensitivity_targets.valid_mask.tolist() == [[True, False], [True, True]]
    assert torch.count_nonzero(prediction.latents.zero_order[0, 3:]) == 0
    assert torch.count_nonzero(prediction.latents.first_order[0, 1:]) == 0
    assert torch.isfinite(terms.total)
    assert all(parameter.grad is not None for parameter in model.parameters())
