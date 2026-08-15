"""50/25/25 GPU query mixture、固定 workspace 与邻接 provenance 合同。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from anymani.assets.bank import HandContainer, HandContainerCfg
from anymani.distill.representations.queries.spatial_sampling import (
    OwnerSurfaceSamplingCache,
    SpatialQuerySamplerCfg,
    _sample_adjacent_queries,
    _sample_anchor_workspace_queries,
    _sample_current_owner_surface,
    _sample_owner_shell_queries,
    materialize_owner_surface_sampling_cache,
    sample_spatial_queries,
)
from anymani.distill.representations.sources.collision_geometry import (
    materialize_owner_geometry_cache,
    sample_palm_anchor_supports,
)
from anymani.distill.representations.sources.kinematics import lower_hand_geometry_semantics
from anymani.distill.representations.targets.field_samples import QueryStratum

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


def test_query_config_rejects_an_unbalanced_odd_shell_count() -> None:
    """50/50 shell 语义要求 $N_S$ 为偶数，不能把单个 shell slot 静默放到同一侧。"""

    with pytest.raises(ValueError, match="owner-shell query count must be even"):
        SpatialQuerySamplerCfg(query_count=4)


def test_workspace_offsets_follow_uniform_ball_volume_and_uniform_anchor_routing() -> None:
    """大样本 oracle 验证 anchor categorical 与 $r=R_WU^{1/3}$ 的三维体积测度。"""

    anchors = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    generator = torch.Generator().manual_seed(19)
    workspace, anchor_index = _sample_anchor_workspace_queries(
        anchors,
        batch_size=2,
        owner_count=3,
        workspace_count=20_000,
        radius_m=0.05,
        generator=generator,
    )
    realization = workspace[0, 0]
    offset = realization - anchors.index_select(0, anchor_index[0, 0])
    normalized_volume = (torch.linalg.vector_norm(offset, dim=-1) / 0.05).pow(3)
    anchor_fraction = (anchor_index[0, 0] == 0).to(torch.float64).mean()

    torch.testing.assert_close(workspace[0, 0], workspace[1, 2], atol=0.0, rtol=0.0)
    assert torch.max(torch.linalg.vector_norm(offset, dim=-1)) <= 0.05
    assert abs(float(normalized_volume.mean()) - 0.5) < 0.01
    assert abs(float(anchor_fraction) - 0.5) < 0.01


def test_surface_sampler_is_area_weighted_barycentric_and_rigid_equivariant() -> None:
    """连续 surface proposal 必须服从 triangle area，并与 owner 刚体变换严格等价。"""

    triangles = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[10.0, 0.0, 0.0], [14.0, 0.0, 0.0], [10.0, 1.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    cache = OwnerSurfaceSamplingCache(
        triangles_owner_local_m=(triangles,),
        face_normals_owner_local=(torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float64),),
        face_area_cdf=(torch.tensor([0.2, 1.0], dtype=torch.float64),),
    )
    identity = torch.eye(4, dtype=torch.float64).reshape(1, 1, 4, 4)
    transformed = identity.clone()
    transformed[0, 0, :3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    transformed[0, 0, :3, 3] = torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64)
    points, normals = _sample_current_owner_surface(
        identity,
        cache,
        sample_count=20_000,
        generator=torch.Generator().manual_seed(23),
    )
    moved_points, moved_normals = _sample_current_owner_surface(
        transformed,
        cache,
        sample_count=20_000,
        generator=torch.Generator().manual_seed(23),
    )
    expected_points = torch.einsum("ij,bgnj->bgni", transformed[0, 0, :3, :3], points)
    expected_points = expected_points + transformed[0, 0, :3, 3]
    expected_normals = torch.einsum("ij,bgnj->bgni", transformed[0, 0, :3, :3], normals)

    large_face_fraction = (points[0, 0, :, 0] > 5.0).to(torch.float64).mean()
    assert abs(float(large_face_fraction) - 0.8) < 0.01
    torch.testing.assert_close(moved_points, expected_points, atol=0.0, rtol=0.0)
    torch.testing.assert_close(moved_normals, expected_normals, atol=0.0, rtol=0.0)


def test_owner_shell_uses_declared_inside_outside_normal_offsets() -> None:
    """平面 oracle 上 shell 的前后两半必须精确对应负/正 face-normal 偏移。"""

    cache = OwnerSurfaceSamplingCache(
        triangles_owner_local_m=(
            torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float64),
        ),
        face_normals_owner_local=(torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64),),
        face_area_cdf=(torch.ones(1, dtype=torch.float64),),
    )
    config = SpatialQuerySamplerCfg(query_count=64)
    shell = _sample_owner_shell_queries(
        torch.eye(4, dtype=torch.float64).reshape(1, 1, 4, 4),
        cache,
        shell_count=2_000,
        config=config,
        generator=torch.Generator().manual_seed(29),
    )[0, 0]
    inside_z, outside_z = shell[:1_000, 2], shell[1_000:, 2]

    assert torch.all(inside_z <= -config.shell_offset_min_m)
    assert torch.all(inside_z >= -config.shell_offset_max_m)
    assert torch.all(outside_z >= config.shell_offset_min_m)
    assert torch.all(outside_z <= config.shell_offset_max_m)


def test_adjacent_queries_reject_non_neighbors_and_stay_in_middle_pair_segment() -> None:
    """两平行 owner oracle 上 neighbor routing 唯一，插值必须落在 pair 连线的 25%--75%。"""

    triangles = (
        torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], dtype=torch.float64),
        torch.tensor([[[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]], dtype=torch.float64),
    )
    cache = OwnerSurfaceSamplingCache(
        triangles_owner_local_m=triangles,
        face_normals_owner_local=(
            torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
            torch.tensor([[-1.0, 0.0, 0.0]], dtype=torch.float64),
        ),
        face_area_cdf=(torch.ones(1, dtype=torch.float64), torch.ones(1, dtype=torch.float64)),
    )
    transforms = torch.eye(4, dtype=torch.float64).reshape(1, 1, 4, 4).expand(1, 2, -1, -1).clone()
    queries, selected = _sample_adjacent_queries(
        transforms,
        cache,
        SimpleNamespace(owner_graph_shortest=torch.tensor([[0, 1], [1, 0]], dtype=torch.long)),
        adjacent_count=2_000,
        candidate_count=4,
        generator=torch.Generator().manual_seed(37),
    )

    assert torch.all(selected[:, 0] == 1)
    assert torch.all(selected[:, 1] == 0)
    assert torch.all(queries[..., 0] >= 0.25)
    assert torch.all(queries[..., 0] <= 0.75)


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
    config = SpatialQuerySamplerCfg(query_count=64)
    anchors = sample_palm_anchor_supports(
        cache,
        container.geometry_semantics,
        spec,
        anchors_per_finger=10,
        sampling_seed=11,
    )
    surface_sampling = materialize_owner_surface_sampling_cache(cache, device="cpu", dtype=spec.q_home.dtype)
    anchors_hand_m = torch.as_tensor(anchors.anchors_hand_m, dtype=spec.q_home.dtype)

    q = spec.q_home.repeat(2, 1)
    q[1, 0] += 0.25
    q.requires_grad_(True)
    queries = sample_spatial_queries(
        q,
        spec,
        surface_sampling,
        anchors_hand_m,
        config=config,
        sampling_seed=13,
    )
    repeated = sample_spatial_queries(
        q,
        spec,
        surface_sampling,
        anchors_hand_m,
        config=config,
        sampling_seed=13,
    )
    resampled = sample_spatial_queries(
        q,
        spec,
        surface_sampling,
        anchors_hand_m,
        config=config,
        sampling_seed=14,
    )

    workspace_count, shell_count, adjacent_count = config.stratum_counts
    assert (workspace_count, shell_count, adjacent_count) == (32, 16, 16)
    assert queries.query_points_h.shape == (2, 21, 64, 3)
    assert not queries.query_points_h.requires_grad
    assert torch.equal(queries.query_points_h, repeated.query_points_h)
    assert torch.equal(queries.query_stratum, repeated.query_stratum)
    assert torch.equal(queries.workspace_anchor_index, repeated.workspace_anchor_index)
    assert not torch.equal(
        queries.query_points_h[:, :, : config.stratum_counts[0]],
        resampled.query_points_h[:, :, : config.stratum_counts[0]],
    )

    workspace_queries = queries.query_points_h[:, :, :workspace_count]
    assert torch.equal(workspace_queries[0, 0], workspace_queries[1, -1])
    assert torch.all(queries.workspace_anchor_index[:, :, :workspace_count] >= 0)
    assert torch.all(queries.workspace_anchor_index[:, :, :workspace_count] < len(anchors.anchors_hand_m))
    assert torch.all(queries.workspace_anchor_index[:, :, workspace_count:] == -1)
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
    palm_index = spec.owner_ids.index("palm")
    expected_palm_neighbors = set(torch.where(spec.owner_graph_shortest[palm_index] == 1)[0].tolist())
    assert set(adjacent_owner[:, palm_index].reshape(-1).tolist()) == expected_palm_neighbors
