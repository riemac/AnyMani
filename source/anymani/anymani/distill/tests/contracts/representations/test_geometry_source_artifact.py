"""静态 geometry source base/anchor-shard 的无 pickle round-trip 与损坏检测合同。"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import trimesh
from anymani.distill.representations.sources.anchor_sampling import (
    AnchorClassificationStats,
    AnchorRealization,
    AnchorSamples,
    _anchor_realization_hash,
)
from anymani.distill.representations.sources.artifacts import GeometrySourceArtifactStore, source_artifact_key
from anymani.distill.representations.sources.cache import geometry_source_array_nbytes
from anymani.distill.representations.sources.collision_geometry import (
    GeometryIdentity,
    HomeSurfaceSamples,
    OwnerGeometryCache,
    OwnerSurfaceRecord,
    OwnerSurfaceSamplingArrays,
    WarpSurfaceAudit,
    WarpSurfaceView,
)
from anymani.distill.representations.sources.geometry_source import GeometrySourceCfg, GeometrySourceCore
from anymani.distill.representations.sources.kinematics import EmbodimentGeometrySpec

pytestmark = pytest.mark.contract


def test_artifact_key_changes_with_urdf_and_mesh_bytes(tmp_path: Path) -> None:
    """资产 ID/content_hash 未更新时，真实几何字节变化仍必须隔离 artifact。"""

    urdf = tmp_path / "hand.urdf"
    mesh = tmp_path / "owner.obj"
    urdf.write_bytes(b"<robot name='a'/>")
    mesh.write_bytes(b"v 0 0 0\n")
    container = SimpleNamespace(
        asset_id="same-id",
        geometry_semantics=SimpleNamespace(content_hash="same-content-hash"),
        urdf_path=urdf,
        mesh_refs=(SimpleNamespace(virtual_path="meshes/owner.obj", real_path=mesh),),
    )
    config = GeometrySourceCfg()
    first = source_artifact_key(container, config)
    mesh.write_bytes(b"v 1 0 0\n")
    second = source_artifact_key(container, config)
    assert first != second
    urdf.write_bytes(b"<robot name='b'/>")
    assert second != source_artifact_key(container, config)


def _core() -> tuple[SimpleNamespace, GeometrySourceCore, GeometrySourceCfg]:
    """构造一项单 owner/单 JOINT 的纯 CPU source，不依赖资产文件或 Warp。"""

    semantics = SimpleNamespace(content_hash="content-sha")
    container = SimpleNamespace(asset_id="artifact-hand", geometry_semantics=semantics)
    spec = EmbodimentGeometrySpec(
        space_screws=torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
        q_home=torch.zeros(1, dtype=torch.float64),
        owner_home_transforms=torch.eye(4, dtype=torch.float64).unsqueeze(0),
        owner_ancestor_mask=torch.tensor([[True]]),
        joint_ancestor_mask=torch.tensor([[False]]),
        joint_limits=torch.tensor([[-1.0, 1.0]], dtype=torch.float64),
        owner_parent_indices=torch.tensor([-1]),
        owner_graph_shortest=torch.zeros(1, 1, dtype=torch.long),
        owner_graph_parent=torch.zeros(1, 1, dtype=torch.long),
        owner_graph_child=torch.zeros(1, 1, dtype=torch.long),
        component_owner_indices=torch.tensor([0]),
        component_owner_local_transforms=torch.eye(4, dtype=torch.float64).unsqueeze(0),
        owner_ids=("palm",),
        joint_names=("joint-0",),
        owner_roles=("palm",),
        owner_finger_names=(None,),
        owner_joint_indices=(-1,),
    )
    box = trimesh.creation.box(extents=(0.1, 0.08, 0.04))
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
    cache = OwnerGeometryCache(
        asset_id=container.asset_id,
        asset_content_hash=semantics.content_hash,
        boolean_backend="synthetic",
        records=(record,),
        surface_geometry_hash="surface-sha",
    )
    points = np.asarray([[[0.05, 0.0, 0.0], [-0.05, 0.0, 0.0]]], dtype=np.float64)
    home = HomeSurfaceSamples(
        owner_ids=("palm",),
        points_owner_local_m=points,
        face_indices=np.asarray([[0, 1]], dtype=np.int64),
        barycentric=np.asarray([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=np.float64),
        sampling_seed=17,
        oversample_factor=2,
    )
    core = GeometrySourceCore(container, spec, cache, home, GeometryIdentity("physical-sha", "domain-sha"))
    return container, core, GeometrySourceCfg(home_points_per_owner=2, home_surface_oversample_factor=2)


def _anchor() -> tuple[AnchorRealization, AnchorClassificationStats]:
    samples = AnchorSamples(
        anchors_hand_m=np.asarray([[0.01, 0.02, 0.03], [-0.01, 0.0, 0.02]], dtype=np.float64),
        finger_names=("index", "index"),
        seed_ids=("seed/index", "seed/index"),
        surface_mask=np.asarray([True, False]),
        radial_support_radius_m=0.05,
        radial_decay_scale_m=0.025,
        surface_fraction=0.5,
        sampling_seed=1_000_003,
        algorithm_version="synthetic-fast-winding-v2:cuda:0",
    )
    realization = AnchorRealization(
        bank_index=1,
        bank_size=8,
        root_seed=0,
        derived_seed=1_000_003,
        samples=samples,
        realization_hash=_anchor_realization_hash(samples),
        sampling_version=samples.algorithm_version,
    )
    return realization, AnchorClassificationStats(512, 2, 3, 0, 1.25)


def test_source_base_and_anchor_shard_round_trip_without_pickle(tmp_path: Path) -> None:
    container, core, config = _core()
    realization, stats = _anchor()
    store = GeometrySourceArtifactStore(tmp_path, mode="read-write", dataset_manifest_sha256="dataset-sha")

    base_reference = store.write_base(core, config)
    anchor_reference = store.write_anchor(container, config, realization, stats)
    loaded, loaded_base_reference = store.load_base(container, config)
    loaded_realization, loaded_stats, loaded_anchor_reference = store.load_anchor(container, config, 1)

    assert loaded_base_reference == base_reference
    assert loaded_anchor_reference == anchor_reference
    torch.testing.assert_close(loaded.spec_cpu.space_screws, core.spec_cpu.space_screws, atol=0.0, rtol=0.0)
    assert np.array_equal(loaded.home_surface.points_owner_local_m, core.home_surface.points_owner_local_m)
    assert np.array_equal(loaded.geometry_cache.records[0].surface_mesh.vertices, core.geometry_cache.records[0].surface_mesh.vertices)
    assert loaded.surface_sampling_arrays is not None
    assert loaded.warp_surface_views is not None
    expected_triangles = np.asarray(core.geometry_cache.records[0].surface_mesh.triangles)
    assert np.array_equal(loaded.surface_sampling_arrays.triangles_owner_local_m[0], expected_triangles)
    assert loaded.surface_sampling_arrays.face_area_cdf[0][-1] == 1.0
    assert loaded.warp_surface_views[0].vertices.dtype == np.float32
    assert loaded.warp_surface_views[0].faces.dtype == np.int32
    assert loaded.warp_surface_views[0].face_altitudes_m.shape == (len(loaded.warp_surface_views[0].faces), 3)
    assert loaded.identity == core.identity
    assert loaded_realization.realization_hash == realization.realization_hash
    assert loaded_realization.bank_index == realization.bank_index
    assert loaded_realization.derived_seed == realization.derived_seed
    assert np.array_equal(loaded_realization.samples.anchors_hand_m, realization.samples.anchors_hand_m)
    assert np.array_equal(loaded_realization.samples.surface_mask, realization.samples.surface_mask)
    assert loaded_stats.query_point_count == stats.query_point_count
    assert loaded_stats.elapsed_seconds == 0.0  # build wall time不属于 deterministic shard identity
    manifest = json.loads((tmp_path / base_reference.relative_path / "manifest.json").read_text(encoding="utf-8"))
    assert all(record["dtype"] != "object" for record in manifest["arrays"].values())


def test_source_artifact_readonly_corruption_fails_closed(tmp_path: Path) -> None:
    container, core, config = _core()
    writer = GeometrySourceArtifactStore(tmp_path, mode="read-write")
    reference = writer.write_base(core, config)
    array_path = tmp_path / reference.relative_path / "arrays" / "home_points.npy"
    payload = bytearray(array_path.read_bytes())
    payload[-1] ^= 0x01
    array_path.write_bytes(payload)

    reader = GeometrySourceArtifactStore(tmp_path, mode="readonly")
    with pytest.raises(ValueError, match="digest mismatch"):
        reader.load_base(container, config)


def test_source_artifact_rejects_nondeterministic_rewrite(tmp_path: Path) -> None:
    container, core, config = _core()
    store = GeometrySourceArtifactStore(tmp_path, mode="read-write")
    store.write_base(core, config)
    changed = GeometrySourceCore(
        core.container,
        core.spec_cpu,
        core.geometry_cache,
        HomeSurfaceSamples(
            **{**core.home_surface.__dict__, "points_owner_local_m": core.home_surface.points_owner_local_m + 0.001}
        ),
        core.identity,
    )

    with pytest.raises(RuntimeError, match="differs from deterministic rebuild"):
        store.write_base(changed, config)


def test_source_artifact_concurrent_writers_publish_one_complete_identity(tmp_path: Path) -> None:
    """两个 writer 的临时目录不可见，最终只能观察到一个相同 digest 的完整目录。"""

    _container, core, config = _core()

    def write_once():
        return GeometrySourceArtifactStore(tmp_path, mode="read-write").write_base(core, config)

    with ThreadPoolExecutor(max_workers=2) as executor:
        references = tuple(executor.map(lambda _index: write_once(), range(2)))
    assert references[0] == references[1]
    final = tmp_path / references[0].relative_path
    assert (final / "manifest.json").is_file()
    assert (final / "COMPLETE").is_file()
    assert not tuple(final.parent.glob(".base.tmp-*"))


def test_source_artifact_incomplete_directory_and_io_do_not_change_torch_rng(tmp_path: Path) -> None:
    """Readonly 不得接纳半成品；key/write/load 不得消费全局 Torch RNG。"""

    container, core, config = _core()
    store = GeometrySourceArtifactStore(tmp_path, mode="read-write")
    key = store.key(container, config)
    incomplete = store.base_path(key)
    incomplete.mkdir(parents=True)
    (incomplete / "manifest.json").write_text("{}", encoding="utf-8")
    before = torch.random.get_rng_state().clone()
    with pytest.raises(FileNotFoundError, match="missing or incomplete"):
        GeometrySourceArtifactStore(tmp_path, mode="readonly").load_base(container, config)
    assert torch.equal(torch.random.get_rng_state(), before)
    incomplete.rename(tmp_path / "incomplete-hidden")
    reference = store.write_base(core, config)
    store.load_base(container, config)
    assert len(reference.manifest_digest) == 64
    assert torch.equal(torch.random.get_rng_state(), before)


def test_source_arena_size_includes_query_and_warp_static_arrays() -> None:
    """512 MiB arena 记账必须覆盖新增 source arrays，而非只覆盖原 trimesh/home payload。"""

    _container, core, _config = _core()
    query_arrays = OwnerSurfaceSamplingArrays(
        (np.zeros((2, 3, 3), dtype=np.float64),),
        (np.zeros((2, 3), dtype=np.float64),),
        (np.ones(2, dtype=np.float64),),
    )
    warp_view = WarpSurfaceView(
        vertices=np.zeros((4, 3), dtype=np.float32),
        faces=np.zeros((2, 3), dtype=np.int32),
        source_face_indices=np.arange(2, dtype=np.int32),
        face_altitudes_m=np.ones((2, 3), dtype=np.float32),
        audit=WarpSurfaceAudit(2, 2, 0, 1.0, 0.0, 0.0),
    )
    enriched = GeometrySourceCore(
        core.container,
        core.spec_cpu,
        core.geometry_cache,
        core.home_surface,
        core.identity,
        query_arrays,
        (warp_view,),
    )
    expected_increment = sum(
        array.nbytes
        for array in (
            *query_arrays.triangles_owner_local_m,
            *query_arrays.face_normals_owner_local,
            *query_arrays.face_area_cdf,
            warp_view.vertices,
            warp_view.faces,
            warp_view.source_face_indices,
            warp_view.face_altitudes_m,
        )
    )
    assert geometry_source_array_nbytes(enriched) - geometry_source_array_nbytes(core) == expected_increment
