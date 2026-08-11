"""robots owner-local collision union 与表面采样合同。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import trimesh
from anymani.assets.bank import HandContainer, HandContainerCfg
from anymani.robots.geometry_kinematics import lower_hand_geometry_semantics
from anymani.robots.owner_geometry import (
    materialize_owner_geometry_cache,
    sample_owner_home_surfaces,
    sample_palm_anchor_supports,
    strict_owner_union,
)

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
    assert all(record.mesh.is_volume for record in cache.records)
    assert sum(record.boolean_applied for record in cache.records) >= 1

    samples = sample_owner_home_surfaces(cache, points_per_owner=16, sampling_seed=17)
    repeated = sample_owner_home_surfaces(cache, points_per_owner=16, sampling_seed=17)
    assert samples.points_owner_local_m.shape == (21, 16, 3)
    assert np.array_equal(samples.points_owner_local_m, repeated.points_owner_local_m)
    assert np.array_equal(samples.face_indices, repeated.face_indices)
    for record, points in zip(cache.records, samples.points_owner_local_m):
        distances = trimesh.proximity.signed_distance(record.mesh, points)
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
    assert np.array_equal(anchors.anchors_hand_m, repeated_anchors.anchors_hand_m)
    assert len(set(anchors.finger_names)) == 4  # 仅 provenance；网络仍读取统一 40-anchor 集合


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
