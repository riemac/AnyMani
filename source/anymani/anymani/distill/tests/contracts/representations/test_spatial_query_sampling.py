"""50/25/25 GPU query mixture、固定 workspace 与邻接 provenance 合同。"""

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
from anymani.distill.representations.targets.field_samples import QueryStratum
from anymani.robots.geometry_kinematics import lower_hand_geometry_semantics
from anymani.robots.owner_geometry import materialize_owner_geometry_cache, sample_owner_home_surfaces

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
def test_query_sampler_preserves_fixed_workspace_and_exact_stratum_counts() -> None:
    """workspace 跨 q/owner 复用，shell/adjacent 随位姿变化且来源不进入坐标。"""

    container = HandContainer.from_cfg(
        HandContainerCfg(path=_MOTHER_ROOT),
        require_geometry_semantics=True,
    )
    assert container.geometry_semantics is not None
    spec = lower_hand_geometry_semantics(container.geometry_semantics)
    cache = materialize_owner_geometry_cache(container, spec)
    home_surface = sample_owner_home_surfaces(cache, points_per_owner=64, sampling_seed=7)
    config = SpatialQuerySamplerCfg(query_count=64)
    workspace = build_workspace_query_bank(
        cache,
        spec,
        home_surface,
        query_count=config.stratum_counts[0],
        sampling_seed=11,
    )

    q = spec.q_home.repeat(2, 1)
    q[1, 0] += 0.25
    q.requires_grad_(True)
    queries = sample_spatial_queries(
        q,
        spec,
        cache,
        home_surface,
        workspace,
        config=config,
        sampling_seed=13,
    )
    repeated = sample_spatial_queries(
        q,
        spec,
        cache,
        home_surface,
        workspace,
        config=config,
        sampling_seed=13,
    )

    workspace_count, shell_count, adjacent_count = config.stratum_counts
    assert (workspace_count, shell_count, adjacent_count) == (32, 16, 16)
    assert queries.query_points_h.shape == (2, 21, 64, 3)
    assert not queries.query_points_h.requires_grad
    assert torch.equal(queries.query_points_h, repeated.query_points_h)
    assert torch.equal(queries.query_stratum, repeated.query_stratum)

    workspace_queries = queries.query_points_h[:, :, :workspace_count]
    assert torch.equal(workspace_queries[0, 0], workspace_queries[1, -1])
    assert torch.all(queries.query_stratum[:, :, :workspace_count] == int(QueryStratum.WORKSPACE))
    assert torch.all(
        queries.query_stratum[:, :, workspace_count : workspace_count + shell_count]
        == int(QueryStratum.OWNER_SHELL)
    )
    assert torch.all(queries.query_stratum[:, :, -adjacent_count:] == int(QueryStratum.ADJACENT))

    index_tip = spec.owner_ids.index("tip/index")
    shell_slice = slice(workspace_count, workspace_count + shell_count)
    assert not torch.equal(
        queries.query_points_h[0, index_tip, shell_slice],
        queries.query_points_h[1, index_tip, shell_slice],
    )
    assert spec.owner_graph_shortest is not None
    adjacent_owner = queries.adjacent_owner_index[:, :, -adjacent_count:]
    for owner_index in range(len(spec.owner_ids)):
        assert torch.all(spec.owner_graph_shortest[owner_index, adjacent_owner[:, owner_index]] == 1)
