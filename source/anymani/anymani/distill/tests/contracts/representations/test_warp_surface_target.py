"""Warp GPU owner-local BVH 最近面查询合同。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from anymani.assets.bank import HandContainer, HandContainerCfg
from anymani.distill.representations.targets.warp_surface import query_owner_surfaces_warp
from anymani.robots.geometry_kinematics import lower_hand_geometry_semantics
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
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Warp CUDA contract requires an NVIDIA GPU")
def test_warp_owner_query_returns_surface_point_face_and_barycentric_provenance() -> None:
    """真实 owner boundary 点查询自身时距离为零，最近点和来源必须闭合。"""

    container = HandContainer.from_cfg(
        HandContainerCfg(path=_MOTHER_ROOT),
        require_geometry_semantics=True,
    )
    assert container.geometry_semantics is not None
    spec_cpu = lower_hand_geometry_semantics(container.geometry_semantics)
    geometry_cache = materialize_owner_geometry_cache(container, spec_cpu)
    surface = sample_owner_home_surfaces(geometry_cache, points_per_owner=8, sampling_seed=29)
    warp_cache = materialize_warp_owner_geometry_cache(geometry_cache, device="cuda:0")

    spec = spec_cpu.to(device="cuda:0", dtype=torch.float32)
    owner_transforms = spec.owner_home_transforms.unsqueeze(0).expand(2, -1, -1, -1).clone()
    local_points = torch.as_tensor(surface.points_owner_local_m, device="cuda:0", dtype=torch.float32)
    query_h = (
        torch.einsum("bgij,gnj->bgni", owner_transforms[..., :3, :3], local_points)
        + owner_transforms[..., :3, 3].unsqueeze(-2)
    )
    result = query_owner_surfaces_warp(query_h, owner_transforms, warp_cache)
    torch.cuda.synchronize()

    assert result.distance_m.shape == (2, 21, 8)
    assert torch.min(result.distance_m).item() >= 0.0
    assert torch.max(result.distance_m).item() < 2.0e-5
    torch.testing.assert_close(result.closest_point_h_m, query_h, atol=2.0e-5, rtol=0.0)
    assert torch.all(result.face_index >= 0)
    assert torch.all(result.feature_margin_m >= -1.0e-7)
    torch.testing.assert_close(
        result.barycentric.sum(dim=-1),
        torch.ones_like(result.distance_m),
        atol=2.0e-5,
        rtol=0.0,
    )
