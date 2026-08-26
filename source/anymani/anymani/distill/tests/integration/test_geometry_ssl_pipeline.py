from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from anymani.distill.methods.contracts import MethodStep
from anymani.distill.methods.multi_anchor_gaussian_implicit_field.config import MultiAnchorGaussianObjectivesCfg
from anymani.distill.methods.multi_anchor_gaussian_implicit_field.context import MultiAnchorObjectiveContext
from anymani.distill.methods.multi_anchor_gaussian_implicit_field.objectives import (
    evaluate_objectives,
    reduce_method_steps,
)
from anymani.distill.models.backbones.geometry_transformer import GraphBiasedTransformerCfg
from anymani.distill.models.decoders.representations.implicit_field import (
    ConditionalDensityDecoder,
    DistanceSensitivityDecoder,
    DistanceSensitivityDecoderCfg,
    ScalarSigmaFiLMDensityDecoderCfg,
)
from anymani.distill.models.input_adapters.geometry import (
    GeometryEncoderCfg,
    ImplicitGeometryEncoder,
    SO2AnchorFrontendCfg,
    StaticGeometryEvidence,
)
from anymani.distill.representations.fields.density import gaussian_density_from_distance
from anymani.distill.representations.targets.field_samples import FieldTargetBatch, QueryStratum, SensitivityTargetBatch

pytestmark = pytest.mark.contract


def test_synthetic_geometry_ssl_forward_two_objectives_and_backward() -> None:
    r"""闭合 unified encoder → density/κ readers → baseline-normalized 双损失 → 普通参数反向。"""

    torch.manual_seed(23)
    dtype = torch.float64
    evidence = StaticGeometryEvidence(
        anchors=torch.tensor(
            [[-0.04, -0.03, 0.0], [-0.02, 0.04, 0.002], [0.03, 0.03, -0.002], [0.05, -0.02, 0.001]],
            dtype=dtype,
        ),
        home_surface_points=torch.tensor(
            [
                [[-0.04, -0.03, 0.0], [-0.04, 0.03, 0.0], [0.04, -0.03, 0.0], [0.04, 0.03, 0.0]],
                [[0.02, -0.01, 0.01], [0.05, -0.01, 0.01], [0.02, 0.01, 0.01], [0.05, 0.01, 0.01]],
                [[0.06, -0.01, 0.012], [0.08, -0.01, 0.012], [0.06, 0.01, 0.012], [0.08, 0.01, 0.012]],
            ],
            dtype=dtype,
        ),
        home_surface_mask=torch.ones(3, 4, dtype=torch.bool),
        palm_normal=torch.tensor([0.0, 0.0, 1.0], dtype=dtype),
        space_screws=torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=dtype),
        q_home=torch.zeros(1, dtype=dtype),
        entity_role=torch.tensor([0, 1, 2], dtype=torch.long),
        entity_joint_index=torch.tensor([-1, 0, -1], dtype=torch.long),
        joint_entity_index=torch.tensor([1], dtype=torch.long),
        shortest_path=torch.tensor([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=torch.long),
        parent_direction=torch.tensor([[0, 1, 2], [0, 0, 1], [0, 0, 0]], dtype=torch.long),
        child_direction=torch.tensor([[0, 0, 0], [1, 0, 0], [2, 1, 0]], dtype=torch.long),
    )
    encoder_config = GeometryEncoderCfg(
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
    )
    encoder = ImplicitGeometryEncoder(encoder_config).to(dtype=dtype)
    density_decoder = ConditionalDensityDecoder(
        ScalarSigmaFiLMDensityDecoderCfg(hidden_width=32, residual_blocks=2),
        entity_width=32,
        query_width=16,
    ).to(dtype=dtype)
    sensitivity_decoder = DistanceSensitivityDecoder(
        DistanceSensitivityDecoderCfg(hidden_width=32, residual_blocks=2),
        entity_width=32,
        query_width=16,
    )
    sensitivity_decoder = sensitivity_decoder.to(dtype=dtype)

    batch_size, owner_count, query_count = 2, 3, 4
    q = torch.tensor([[0.17], [-0.29]], dtype=dtype)
    query_points = torch.tensor(
        [
            [
                [[0.00, 0.00, 0.04], [0.03, 0.01, 0.02], [0.06, 0.00, 0.01], [0.08, -0.01, 0.02]],
                [[0.00, 0.00, 0.04], [0.03, 0.01, 0.02], [0.06, 0.00, 0.01], [0.08, -0.01, 0.02]],
                [[0.00, 0.00, 0.04], [0.03, 0.01, 0.02], [0.06, 0.00, 0.01], [0.08, -0.01, 0.02]],
            ]
        ],
        dtype=dtype,
    ).expand(batch_size, owner_count, query_count, 3).clone()
    latents = encoder(q, evidence)
    query_features = encoder.encode_points(query_points, evidence)  # 固定 `{h}` query 只作为 readout condition
    bandwidths = torch.tensor([0.012, 0.032], dtype=dtype)
    density_prediction = density_decoder(latents.entities, query_features, bandwidths)
    owner_index = torch.tensor([1, 2], dtype=torch.long)
    query_index = torch.tensor([1, 2], dtype=torch.long)
    joint_index = torch.tensor([0, 0], dtype=torch.long)
    kappa_prediction = sensitivity_decoder(
        latents.entities[:, owner_index],
        latents.entities[:, evidence.joint_entity_index[joint_index]],
        query_features[:, owner_index, query_index],
    )

    distance = torch.full((batch_size, owner_count, query_count), 0.02, dtype=dtype)
    density_target = gaussian_density_from_distance(distance, bandwidths)
    field_targets = FieldTargetBatch(
        query_points=query_points.detach(),
        query_stratum=torch.tensor(
            [[QueryStratum.WORKSPACE, QueryStratum.WORKSPACE, QueryStratum.OWNER_SHELL, QueryStratum.ADJACENT]]
        )
        .expand(batch_size, owner_count, query_count)
        .clone(),
        distance=distance,
        density=density_target,
        valid_mask=torch.ones(batch_size, owner_count, query_count, dtype=torch.bool),
        owner_role=evidence.entity_role,
        bandwidths=bandwidths,
        provenance={"frame": "h", "length_unit": "m"},
    )
    sensitivity_targets = SensitivityTargetBatch(
        owner_index=owner_index,
        query_index=query_index,
        joint_index=joint_index,
        ancestor_mask=torch.ones(2, dtype=torch.bool),
        active_mask=torch.ones(2, dtype=torch.bool),
        closest_point=torch.zeros(batch_size, 2, 3, dtype=dtype),
        closest_source=torch.zeros(batch_size, 2, dtype=torch.long),
        uniqueness_margin=torch.full((batch_size, 2), 0.004, dtype=dtype),
        kappa=torch.tensor([[0.03, -0.01], [0.02, -0.015]], dtype=dtype),
        field_sensitivity=torch.tensor(
            [[[-0.2, -0.1], [0.05, 0.02]], [[-0.15, -0.08], [0.04, 0.01]]], dtype=dtype
        ),
        valid_mask=torch.ones(batch_size, 2, dtype=torch.bool),
    )

    context = MultiAnchorObjectiveContext(
        prediction=SimpleNamespace(density=density_prediction, kappa=kappa_prediction),
        batch=SimpleNamespace(field_targets=field_targets, sensitivity_targets=sensitivity_targets),
    )
    objectives_cfg = MultiAnchorGaussianObjectivesCfg()
    update = reduce_method_steps(
        (MethodStep(objectives=evaluate_objectives(context, objectives_cfg), sample_count=batch_size),),
        objectives_cfg,
        {"density": 1.0, "kappa": 1.0},
    )
    update.loss.backward()

    assert density_prediction.shape == (batch_size, owner_count, query_count, 2)
    assert kappa_prediction.shape == (batch_size, 2)
    assert torch.isfinite(update.loss)
    assert q.grad is None
    trainable_gradients = [parameter.grad for parameter in encoder.parameters() if parameter.requires_grad]
    assert any(gradient is not None and torch.isfinite(gradient).all() for gradient in trainable_gradients)
