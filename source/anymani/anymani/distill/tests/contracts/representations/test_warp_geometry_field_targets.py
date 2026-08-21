"""Warp 在线 d/ρ/κ/g 教师与有限差分合同。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from anymani.assets.bank import HandContainer, HandContainerCfg
from anymani.distill.representations.queries.spatial_sampling import (
    SpatialQuerySamplerCfg,
    materialize_owner_surface_sampling_cache,
    sample_spatial_queries,
)
from anymani.distill.representations.sources.collision_geometry import (
    materialize_owner_geometry_cache,
    materialize_warp_owner_geometry_cache,
    sample_palm_anchor_supports,
)
from anymani.distill.representations.sources.kinematics import forward_owner_transforms, lower_hand_geometry_semantics
from anymani.distill.representations.targets.geometry_field import generate_geometry_field_targets
from anymani.distill.representations.targets.warp_surface import query_owner_surfaces_warp

pytestmark = pytest.mark.contract

_MOTHER_ROOT = (
    Path(__file__).resolve().parents[4]
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
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Warp CUDA target contract requires an NVIDIA GPU")
def test_warp_geometry_targets_close_density_chain_and_kappa_finite_difference() -> None:
    """完整 teacher 必须保持 query/owner/edge 轴，并让 κ 对齐固定 query 的距离差分。"""

    container = HandContainer.from_cfg(
        HandContainerCfg(path=_MOTHER_ROOT),
        require_geometry_semantics=True,
    )
    assert container.geometry_semantics is not None
    spec_cpu = lower_hand_geometry_semantics(container.geometry_semantics)
    geometry_cache = materialize_owner_geometry_cache(container, spec_cpu)
    query_config = SpatialQuerySamplerCfg(query_count=64)
    anchors = sample_palm_anchor_supports(
        geometry_cache,
        container.geometry_semantics,
        spec_cpu,
        anchors_per_finger=10,
        sampling_seed=37,
    )
    warp_cache = materialize_warp_owner_geometry_cache(geometry_cache, device="cuda:0")
    spec = spec_cpu.to(device="cuda:0", dtype=torch.float32)
    surface_sampling = materialize_owner_surface_sampling_cache(
        geometry_cache, device="cuda:0", dtype=torch.float32
    )
    anchors_hand_m = torch.as_tensor(anchors.anchors_hand_m, device="cuda:0", dtype=torch.float32)
    q = spec.q_home.unsqueeze(0)
    queries = sample_spatial_queries(
        q,
        spec,
        surface_sampling,
        anchors_hand_m,
        config=query_config,
        sampling_seed=41,
    )

    field, sensitivity = generate_geometry_field_targets(
        q,
        spec,
        geometry_cache,
        warp_cache,
        queries,
        edge_sampling_seed=43,
    )
    torch.cuda.synchronize()

    assert field.distance.shape == (1, 21, 64)
    assert field.density.shape == (1, 21, 64, 3)
    assert sensitivity.kappa.shape == (1, 32)
    assert sensitivity.field_sensitivity.shape == (1, 32, 3)
    assert torch.count_nonzero(sensitivity.kappa[:, ~sensitivity.ancestor_mask]) == 0
    assert torch.count_nonzero(sensitivity.field_sensitivity[:, ~sensitivity.ancestor_mask]) == 0
    assert sensitivity.provenance["global_second_nearest_margin"] == "not_materialized"

    candidate = torch.where(sensitivity.valid_mask[0] & sensitivity.ancestor_mask)[0]
    assert len(candidate) > 0, "mother owner-shell queries should provide at least one locally smooth ancestor edge"
    edge = int(candidate[0])
    owner_index = int(sensitivity.owner_index[edge])
    query_index = int(sensitivity.query_index[edge])
    joint_index = int(sensitivity.joint_index[edge])
    epsilon = 1.0e-4
    q_plus = q.clone()
    q_minus = q.clone()
    q_plus[:, joint_index] += epsilon
    q_minus[:, joint_index] -= epsilon
    plus = query_owner_surfaces_warp(
        queries.query_points_h,
        forward_owner_transforms(spec, q_plus),
        warp_cache,
    )
    minus = query_owner_surfaces_warp(
        queries.query_points_h,
        forward_owner_transforms(spec, q_minus),
        warp_cache,
    )
    torch.cuda.synchronize()
    finite_difference = (
        plus.distance_m[0, owner_index, query_index] - minus.distance_m[0, owner_index, query_index]
    ) / (2.0 * epsilon)
    torch.testing.assert_close(
        sensitivity.kappa[0, edge],
        finite_difference,
        atol=3.0e-3,
        rtol=3.0e-2,
    )


@_requires_local_mother
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Warp CUDA target contract requires an NVIDIA GPU")
def test_fixed_workspace_separates_static_palm_from_q_sensitive_joint_owner() -> None:
    """同一 `{h}` workspace realization 下，PALM distance 静止而动态 owner 必须响应 q。"""

    container = HandContainer.from_cfg(
        HandContainerCfg(path=_MOTHER_ROOT),
        require_geometry_semantics=True,
    )
    assert container.geometry_semantics is not None
    spec_cpu = lower_hand_geometry_semantics(container.geometry_semantics)
    geometry_cache = materialize_owner_geometry_cache(container, spec_cpu)
    anchors = sample_palm_anchor_supports(
        geometry_cache,
        container.geometry_semantics,
        spec_cpu,
        anchors_per_finger=10,
        sampling_seed=47,
    )
    spec = spec_cpu.to(device="cuda:0", dtype=torch.float32)
    surface_sampling = materialize_owner_surface_sampling_cache(
        geometry_cache, device="cuda:0", dtype=torch.float32
    )
    warp_cache = materialize_warp_owner_geometry_cache(geometry_cache, device="cuda:0")
    q = spec.q_home.repeat(2, 1)
    q[1, 0] += 0.25
    config = SpatialQuerySamplerCfg(query_count=64)
    queries = sample_spatial_queries(
        q,
        spec,
        surface_sampling,
        torch.as_tensor(anchors.anchors_hand_m, device="cuda:0", dtype=torch.float32),
        config=config,
        sampling_seed=53,
    )
    field, _ = generate_geometry_field_targets(
        q,
        spec,
        geometry_cache,
        warp_cache,
        queries,
        edge_sampling_seed=59,
    )
    workspace_count = config.stratum_counts[0]
    palm_index = spec.owner_ids.index("palm")
    dynamic_owner = int(torch.where(spec.owner_ancestor_mask[:, 0])[0][0])

    torch.testing.assert_close(
        queries.query_points_h[0, palm_index, :workspace_count],
        queries.query_points_h[1, dynamic_owner, :workspace_count],
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        field.distance[0, palm_index, :workspace_count],
        field.distance[1, palm_index, :workspace_count],
        atol=2.0e-7,
        rtol=2.0e-6,
    )
    assert torch.max(
        torch.abs(
            field.distance[0, dynamic_owner, :workspace_count]
            - field.distance[1, dynamic_owner, :workspace_count]
        )
    ) > 1.0e-5
    torch.testing.assert_close(field.bandwidths[0], field.bandwidths[1], atol=0.0, rtol=0.0)
