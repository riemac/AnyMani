"""真实 mother cache -> Warp teacher -> SSL model -> 五项目标 -> backward 集成合同。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from anymani.assets.bank import HandContainer, HandContainerCfg
from anymani.distill.models.geometry_ssl import GeometrySSLModel, GeometrySSLModelConfig
from anymani.distill.models.input_adapters.geometry import GeometryEncoderConfig, build_static_geometry_evidence
from anymani.distill.objectives.representations.field_reconstruction import GeometrySSLObjective, GeometrySSLWeights
from anymani.distill.representations.queries.spatial_sampling import (
    SpatialQuerySamplerCfg,
    materialize_owner_surface_sampling_cache,
    sample_spatial_queries,
)
from anymani.distill.representations.targets.geometry_field import generate_geometry_field_targets
from anymani.robots.geometry_kinematics import lower_hand_geometry_semantics
from anymani.robots.owner_geometry import (
    materialize_owner_geometry_cache,
    materialize_warp_owner_geometry_cache,
    sample_owner_home_surfaces,
    sample_palm_anchor_supports,
)

pytestmark = pytest.mark.contract

_MOTHER_ROOT = (
    Path(__file__).resolve().parents[3]
    / "assets"
    / "generated"
    / "2026-06-10_11-30-08"
    / "single_palm_leap"
    / "right_t4_i4_m4_r4"
)
_requires_local_mother = pytest.mark.skipif(
    not _MOTHER_ROOT.is_dir(),
    reason="generated LEAP mother asset is a local research artifact",
)


@_requires_local_mother
@pytest.mark.skipif(not torch.cuda.is_available(), reason="real geometry SSL integration requires CUDA/Warp")
def test_real_mother_geometry_ssl_forward_objective_and_backward() -> None:
    """首个真实资产闭环必须保持 retained/disposable 生命周期和 q JVP 梯度。"""

    torch.manual_seed(47)
    container = HandContainer.from_cfg(
        HandContainerCfg(path=_MOTHER_ROOT),
        require_geometry_semantics=True,
    )
    assert container.geometry_semantics is not None
    semantics = container.geometry_semantics
    spec_cpu = lower_hand_geometry_semantics(semantics)
    geometry_cache = materialize_owner_geometry_cache(container, spec_cpu)
    home_surface = sample_owner_home_surfaces(geometry_cache, points_per_owner=64, sampling_seed=53)
    anchors = sample_palm_anchor_supports(
        geometry_cache,
        semantics,
        spec_cpu,
        anchors_per_finger=10,
        sampling_seed=59,
    )
    query_config = SpatialQuerySamplerCfg(query_count=64)
    warp_cache = materialize_warp_owner_geometry_cache(geometry_cache, device="cuda:0")
    spec = spec_cpu.to(device="cuda:0", dtype=torch.float32)
    surface_sampling = materialize_owner_surface_sampling_cache(
        geometry_cache, device="cuda:0", dtype=torch.float32
    )
    evidence = build_static_geometry_evidence(
        semantics,
        spec,
        home_surface,
        anchors,
        device="cuda:0",
        dtype=torch.float32,
    )
    q = spec.q_home.unsqueeze(0).clone().requires_grad_(True)
    queries = sample_spatial_queries(
        q.detach(),
        spec,
        surface_sampling,
        evidence.anchors,
        config=query_config,
        sampling_seed=67,
    )
    field_targets, sensitivity_targets = generate_geometry_field_targets(
        q.detach(),
        spec,
        geometry_cache,
        warp_cache,
        queries,
        edge_sampling_seed=71,
    )

    encoder_config = GeometryEncoderConfig(
        relation_width=16,
        home_width=16,
        screw_width=12,
        hidden_width=32,
        zero_order_width=24,
        first_order_width=12,
        transformer_layers=1,
        attention_heads=4,
        feedforward_width=64,
        dropout=0.0,
        max_graph_distance=8,
    )
    model = GeometrySSLModel(
        GeometrySSLModelConfig(
            encoder=encoder_config,
            decoder_hidden_width=32,
            decoder_residual_blocks=2,
        )
    ).to(device="cuda:0", dtype=torch.float32)
    prediction = model(
        q,
        evidence,
        queries.query_points_h,
        field_targets.bandwidths,
        sensitivity_targets.owner_index,
        sensitivity_targets.query_index,
        sensitivity_targets.joint_index,
    )
    terms = GeometrySSLObjective(GeometrySSLWeights())(
        q=q,
        density_prediction=prediction.density,
        kappa_prediction=prediction.kappa,
        field_targets=field_targets,
        sensitivity_targets=sensitivity_targets,
    )
    terms.total.backward()
    torch.cuda.synchronize()

    assert prediction.latents.zero_order.shape == (1, 21, 24)
    assert prediction.latents.first_order.shape == (1, 16, 12)
    assert prediction.density.shape == (1, 21, 64, 3)
    assert prediction.kappa.shape == (1, 42)
    assert torch.isfinite(terms.total)
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert any(parameter.grad is not None for parameter in model.encoder.parameters())
    retained_keys = model.retained_state_dict()
    assert retained_keys and all(key.startswith("encoder.") for key in retained_keys)
    assert not any("decoder" in key for key in retained_keys)
