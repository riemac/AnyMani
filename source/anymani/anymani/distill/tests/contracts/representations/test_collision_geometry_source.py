"""representation source owner-local collision union 与表面采样合同。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import trimesh
from anymani.assets.bank import HandContainer, HandContainerCfg
from anymani.distill.representations.sources.cache import GeometrySourceArena
from anymani.distill.representations.sources.collision_geometry import (
    GeometryIdentity,
    OwnerGeometryCache,
    OwnerSurfaceRecord,
    _radial_decay_candidates,
    geometry_identity,
    materialize_owner_geometry_cache,
    materialize_warp_owner_geometry_cache,
    prepare_warp_surface_view,
    release_warp_owner_geometry_cache,
    sample_owner_home_surfaces,
    sample_palm_anchor_supports,
    strict_owner_union,
    warp_owner_geometry_cache_stats,
)
from anymani.distill.representations.sources.geometry_source import AnchorBankCfg, GeometrySource, GeometrySourceCfg
from anymani.distill.representations.sources.kinematics import lower_hand_geometry_semantics

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
def test_mother_owner_geometry_materializes_closed_union_and_reproducible_surface_samples() -> None:
    """mother 每个 owner 必须是真实体积，home 点只在 boundary 且固定复现。"""

    container = HandContainer.from_cfg(
        HandContainerCfg(path=_MOTHER_ROOT),
        require_geometry_semantics=True,
    )
    assert container.geometry_semantics is not None
    spec = lower_hand_geometry_semantics(container.geometry_semantics, dtype=torch.float64)
    cache = materialize_owner_geometry_cache(container, spec)

    assert len(cache.records) == 21
    assert cache.asset_content_hash == container.geometry_semantics.content_hash
    assert all(record.surface_mesh.is_volume for record in cache.records)
    assert all(record.solid_mesh is not None for record in cache.records)
    assert sum(record.boolean_applied for record in cache.records) >= 1

    samples = sample_owner_home_surfaces(cache, points_per_owner=16, sampling_seed=17)
    repeated = sample_owner_home_surfaces(cache, points_per_owner=16, sampling_seed=17)
    assert samples.points_owner_local_m.shape == (21, 16, 3)
    assert np.array_equal(samples.points_owner_local_m, repeated.points_owner_local_m)
    assert np.array_equal(samples.face_indices, repeated.face_indices)
    for record, points in zip(cache.records, samples.points_owner_local_m):
        distances = trimesh.proximity.signed_distance(record.surface_mesh, points)
        assert np.max(np.abs(distances)) < 1.0e-7

    anchors = sample_palm_anchor_supports(
        cache,
        container.geometry_semantics,
        spec,
        anchors_per_finger=10,
        sampling_seed=23,
        radial_support_radius_m=0.05,
    )
    repeated_anchors = sample_palm_anchor_supports(
        cache,
        container.geometry_semantics,
        spec,
        anchors_per_finger=10,
        sampling_seed=23,
        radial_support_radius_m=0.05,
    )
    assert anchors.anchors_hand_m.shape == (40, 3)
    assert anchors.surface_mask.sum() == 20
    assert anchors.radial_decay_scale_m == pytest.approx(0.025)
    assert anchors.algorithm_version == "palm-seed-radial-gaussian-fps-v1"
    assert np.array_equal(anchors.anchors_hand_m, repeated_anchors.anchors_hand_m)
    assert len(set(anchors.finger_names)) == 4  # 仅 provenance；网络仍读取统一 40-anchor 集合
    hand_rotation = np.asarray(container.geometry_semantics.asset_to_hand_rotation, dtype=np.float64).reshape(3, 3)
    hand_translation = np.asarray(container.geometry_semantics.asset_to_hand_translation_m, dtype=np.float64)
    for seed in container.geometry_semantics.anchor_seeds:
        seed_hand = hand_rotation @ np.asarray(seed.position_a_m, dtype=np.float64) + hand_translation
        seed_mask = np.asarray(anchors.seed_ids) == seed.seed_id
        seed_points = anchors.anchors_hand_m[seed_mask]
        radius = np.linalg.norm(seed_points - seed_hand, axis=-1)
        pairwise = np.linalg.norm(seed_points[:, None, :] - seed_points[None, :, :], axis=-1)
        pairwise += np.eye(len(seed_points))
        assert np.max(radius) <= anchors.radial_support_radius_m + 1.0e-12
        assert np.min(pairwise) > 1.0e-6


@_requires_local_mother
def test_source_arena_release_and_rebuild_preserve_exact_cpu_realization() -> None:
    r"""同一发布资产在 arena clear 后重建，静态物理真值必须逐元素一致。"""

    container = HandContainer.from_cfg(
        HandContainerCfg(path=_MOTHER_ROOT),
        require_geometry_semantics=True,
    )
    config = GeometrySourceCfg(
        home_points_per_owner=8,
        home_surface_oversample_factor=2,
        static_sampling_seed=17,
        anchors=AnchorBankCfg(bank_size=2, anchors_per_finger=2),
    )
    arena = GeometrySourceArena(max_entries=1, max_bytes=512 * 1024 * 1024)
    first = arena.load_or_create(
        container,
        config=config,
        materialize=lambda: GeometrySource.materialize(container, config=config),
    )
    arena.clear()  # 模拟 subwindow/run teardown 后只保留发布资产真源
    second = arena.load_or_create(
        container,
        config=config,
        materialize=lambda: GeometrySource.materialize(container, config=config),
    )

    assert first.identity == second.identity  # physical/configuration SHA-256 完全一致
    assert first.geometry_cache.surface_geometry_hash == second.geometry_cache.surface_geometry_hash
    assert np.array_equal(first.home_surface.points_owner_local_m, second.home_surface.points_owner_local_m)
    assert np.array_equal(first.home_surface.face_indices, second.home_surface.face_indices)
    assert all(
        np.array_equal(first_bank.anchors_hand_m, second_bank.anchors_hand_m)
        for first_bank, second_bank in zip(first.anchor_bank, second.anchor_bank)
    )
    assert all(
        np.array_equal(first_record.surface_mesh.vertices, second_record.surface_mesh.vertices)
        and np.array_equal(first_record.surface_mesh.faces, second_record.surface_mesh.faces)
        for first_record, second_record in zip(first.geometry_cache.records, second.geometry_cache.records)
    )


def test_anchor_radial_rejection_matches_declared_truncated_gaussian() -> None:
    r"""等量 proposal 的接受率必须服从 $\exp[-r^2/(2\tau_a^2)]$，而非硬球均匀采样。"""

    scale = 0.025
    count = 20_000
    radii = np.asarray([0.0, scale, 2.0 * scale], dtype=np.float64)
    points = np.concatenate(
        tuple(np.tile(np.asarray([[radius, 0.0, 0.0]]), (count, 1)) for radius in radii),
        axis=0,
    )
    accepted = _radial_decay_candidates(points, np.zeros(3), scale, seed=31)
    actual = np.asarray([np.count_nonzero(np.isclose(accepted[:, 0], radius)) / count for radius in radii])
    expected = np.exp(-0.5 * (radii / scale) ** 2)

    np.testing.assert_allclose(actual, expected, atol=0.012, rtol=0.0)


def test_strict_union_removes_buried_faces_without_filling_open_groove() -> None:
    """Boolean union 删除内部面，但不会用 convex hull 填掉向外开放的凹槽。"""

    left = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    right = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    right.apply_translation((0.75, 0.0, 0.0))
    union = strict_owner_union([left, right], owner_id="joint/test")
    assert union.is_volume
    assert union.volume == pytest.approx(1.75, rel=1.0e-5)
    # Boolean 会把相交面重新三角剖分，因此 face 数不必单调减少；物理表面积应删除埋藏面。
    assert union.area < left.area + right.area

    wall_a = trimesh.creation.box(extents=(0.2, 1.0, 1.0))
    wall_b = trimesh.creation.box(extents=(0.2, 1.0, 1.0))
    wall_a.apply_translation((-0.6, 0.0, 0.0))
    wall_b.apply_translation((0.6, 0.0, 0.0))
    groove = strict_owner_union([wall_a, wall_b], owner_id="palm/groove")
    assert groove.is_volume
    assert not bool(groove.contains(np.asarray([[0.0, 0.0, 0.0]]))[0])


def test_open_owner_surface_supports_home_points_but_rejects_interior_anchors() -> None:
    r"""开放三角表面可定义 UDF 边界，但不能伪装成 palm solid。"""

    surface = trimesh.Trimesh(
        vertices=np.asarray([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]]),
        faces=np.asarray([[0, 1, 2], [0, 2, 3]]),
        process=False,
    )
    record = OwnerSurfaceRecord(
        owner_id="palm",
        owner_index=0,
        role="palm",
        finger_name=None,
        component_ids=("palm/open_surface",),
        surface_mesh=surface,
        solid_mesh=None,
        boolean_applied=False,
    )
    cache = OwnerGeometryCache("open-palm", "open-palm-hash", "manifold-unused", (record,))
    home = sample_owner_home_surfaces(cache, points_per_owner=8, sampling_seed=17)

    assert home.points_owner_local_m.shape == (1, 8, 3)
    assert np.max(np.abs(home.points_owner_local_m[..., 2])) < 1.0e-12

    semantics = SimpleNamespace(
        owners=(SimpleNamespace(owner_id="palm", owner_index=0),),
        anchor_seeds=(SimpleNamespace(seed_id="seed/index", finger_name="index", position_a_m=(0.0, 0.0, 0.0)),),
        asset_to_hand_rotation=np.eye(3),
        asset_to_hand_translation_m=np.zeros(3),
    )
    spec = SimpleNamespace(
        owner_ids=("palm",),
        owner_home_transforms=torch.eye(4, dtype=torch.float64).unsqueeze(0),
    )
    surface_only = sample_palm_anchor_supports(
        cache,
        semantics,
        spec,
        anchors_per_finger=4,
        sampling_seed=23,
        radial_support_radius_m=2.0,
        surface_fraction=1.0,
    )
    assert surface_only.surface_mask.all()

    with pytest.raises(ValueError, match="solid_mesh"):
        sample_palm_anchor_supports(
            cache,
            semantics,
            spec,
            anchors_per_finger=4,
            sampling_seed=23,
            radial_support_radius_m=2.0,
            surface_fraction=0.5,
        )


def test_mesh_component_welds_stl_vertices_before_solid_classification(tmp_path: Path) -> None:
    r"""STL 的逐面重复顶点应确定性焊接，几何闭合体不得被误判成开放表面。"""

    mesh_path = tmp_path / "closed_box.stl"
    trimesh.creation.box(extents=(0.1, 0.2, 0.3)).export(mesh_path)
    component = SimpleNamespace(
        component_id="palm/mesh",
        owner_id="palm",
        geometry_kind="mesh",
        geometry_payload={"file_path": str(mesh_path), "scale": (1.0, 1.0, 1.0)},
    )
    semantics = SimpleNamespace(
        owners=(SimpleNamespace(owner_id="palm", owner_index=0, role="palm", component_ids=("palm/mesh",)),),
        components=(component,),
        content_hash="welded-box",
    )
    container = SimpleNamespace(asset_id="welded-box", geometry_semantics=semantics, mesh_refs=())
    spec = SimpleNamespace(
        owner_ids=("palm",),
        component_owner_local_transforms=torch.eye(4, dtype=torch.float64).unsqueeze(0),
    )

    cache = materialize_owner_geometry_cache(container, spec)

    assert cache.records[0].surface_mesh.is_watertight
    assert cache.records[0].solid_mesh is not None
    assert cache.records[0].solid_mesh.is_volume


@_requires_local_mother
def test_physical_identity_excludes_joint_limits_but_configuration_domain_includes_them() -> None:
    r"""limit-only variants 应共享物理映射组，但保留不同构型采样域身份。"""

    container = HandContainer.from_cfg(
        HandContainerCfg(path=_MOTHER_ROOT),
        require_geometry_semantics=True,
    )
    assert container.geometry_semantics is not None
    spec = lower_hand_geometry_semantics(container.geometry_semantics, dtype=torch.float64)
    cache = materialize_owner_geometry_cache(container, spec)
    original = geometry_identity(container.geometry_semantics, spec, cache)
    shifted_limits = spec.joint_limits + torch.tensor((-0.01, 0.02), dtype=torch.float64)
    changed_spec = spec.__class__(
        **{
            **spec.__dict__,
            "joint_limits": shifted_limits,
        }
    )
    changed = geometry_identity(container.geometry_semantics, changed_spec, cache)

    assert isinstance(original, GeometryIdentity)
    assert original.physical_geometry_hash == changed.physical_geometry_hash
    assert original.configuration_domain_hash != changed.configuration_domain_hash


@_requires_local_mother
def test_physical_identity_changes_when_owner_surface_changes() -> None:
    r"""owner-local 物理表面变化必须形成新的 leakage group。"""

    container = HandContainer.from_cfg(
        HandContainerCfg(path=_MOTHER_ROOT),
        require_geometry_semantics=True,
    )
    assert container.geometry_semantics is not None
    spec = lower_hand_geometry_semantics(container.geometry_semantics, dtype=torch.float64)
    cache = materialize_owner_geometry_cache(container, spec)
    original = geometry_identity(container.geometry_semantics, spec, cache)
    changed_cache = OwnerGeometryCache(
        asset_id=cache.asset_id,
        asset_content_hash=cache.asset_content_hash,
        boolean_backend=cache.boolean_backend,
        records=cache.records,
        surface_geometry_hash="0" * 64,
        surface_processing_version=cache.surface_processing_version,
    )

    changed = geometry_identity(container.geometry_semantics, spec, changed_cache)

    assert original.physical_geometry_hash != changed.physical_geometry_hash


def test_float32_surface_view_filters_negligible_collapsed_faces_and_enforces_area_budget() -> None:
    r"""只允许删除 float32 中坍缩且总面积可忽略的三角面。"""

    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0 + 1.0e-8, 0.0, 0.0],
            [1.0, 1.0e-8, 0.0],
        ],
        dtype=np.float64,
    )
    surface = trimesh.Trimesh(vertices=vertices, faces=np.asarray([[0, 1, 2], [3, 4, 5]]), process=False)

    view = prepare_warp_surface_view(surface, owner_id="tip/index", max_area_loss_fraction=1.0e-8)

    assert view.faces.shape == (1, 3)
    assert view.audit.input_face_count == 2
    assert view.audit.removed_face_count == 1
    assert view.audit.removed_area_fraction < 1.0e-8

    tiny_only = trimesh.Trimesh(vertices=vertices[3:], faces=np.asarray([[0, 1, 2]]), process=False)
    with pytest.raises(ValueError, match="area-loss budget"):
        prepare_warp_surface_view(tiny_only, owner_id="tip/index", max_area_loss_fraction=1.0e-8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Warp CUDA cache contract requires an NVIDIA GPU")
def test_warp_owner_cache_lease_release_evicts_global_resident_reference() -> None:
    r"""resident window 释放最后一个 lease 后，同一 key 必须从全局缓存驱逐。"""

    baseline = warp_owner_geometry_cache_stats()  # 其他 contract 可能已持有独立资产 cache
    box = trimesh.creation.box(extents=(0.1, 0.2, 0.3))
    record = OwnerSurfaceRecord(
        owner_id="palm",
        owner_index=0,
        role="palm",
        finger_name=None,
        component_ids=("palm/box",),
        surface_mesh=box,
        solid_mesh=box.copy(),
        boolean_applied=False,
    )
    geometry_cache = OwnerGeometryCache(
        asset_id="lease-test",
        asset_content_hash="lease-test-content",
        boolean_backend="not-used",
        records=(record,),
        surface_geometry_hash="lease-test-unique-surface",
    )
    first = materialize_warp_owner_geometry_cache(geometry_cache, device="cuda:0")
    second = materialize_warp_owner_geometry_cache(geometry_cache, device="cuda:0")

    assert first is second
    assert warp_owner_geometry_cache_stats()["lease_count"] == baseline["lease_count"] + 2
    assert not release_warp_owner_geometry_cache(first)
    assert warp_owner_geometry_cache_stats()["lease_count"] == baseline["lease_count"] + 1
    assert release_warp_owner_geometry_cache(second)
    assert warp_owner_geometry_cache_stats() == baseline
