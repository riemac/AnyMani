"""Warp 在线 d/ρ/κ/g 教师与有限差分合同。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from anymani.assets.bank import HandContainer, HandContainerCfg
from anymani.distill.representations.queries.spatial_sampling import (
    SpatialQuerySamplerCfg,
    build_workspace_query_bank,
    sample_spatial_queries,
)
from anymani.distill.representations.targets.geometry_field import generate_geometry_field_targets
from anymani.distill.representations.targets.warp_surface import query_owner_surfaces_warp
from anymani.robots.geometry_kinematics import forward_owner_transforms, lower_hand_geometry_semantics
from anymani.robots.owner_geometry import (
    materialize_owner_geometry_cache,
    materialize_warp_owner_geometry_cache,
    sample_owner_home_surfaces,
)

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
    home_surface = sample_owner_home_surfaces(geometry_cache, points_per_owner=64, sampling_seed=31)
    query_config = SpatialQuerySamplerCfg(query_count=64)
    workspace = build_workspace_query_bank(
        geometry_cache,
        spec_cpu,
        home_surface,
        query_count=query_config.stratum_counts[0],
        sampling_seed=37,
    )
    warp_cache = materialize_warp_owner_geometry_cache(geometry_cache, device="cuda:0")
    spec = spec_cpu.to(device="cuda:0", dtype=torch.float32)
    q = spec.q_home.unsqueeze(0)
    queries = sample_spatial_queries(
        q,
        spec,
        geometry_cache,
        home_surface,
        workspace,
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
    assert field.density.shape == (1, 21, 64, 4)
    assert sensitivity.kappa.shape == (1, 42)
    assert sensitivity.field_sensitivity.shape == (1, 42, 4)
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
