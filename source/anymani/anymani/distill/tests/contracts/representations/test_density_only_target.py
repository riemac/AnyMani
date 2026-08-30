r"""Density-only Warp teacher 与 v0.7.5 联合 teacher 的 zero-order parity。"""

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
from anymani.distill.representations.sources.kinematics import (
    forward_owner_transforms_and_spatial_screws,
    lower_hand_geometry_semantics,
)
from anymani.distill.representations.targets.density_field import generate_density_field_targets
from anymani.distill.representations.targets.geometry_field import generate_geometry_field_targets

pytestmark = pytest.mark.contract

_MOTHER_ROOT = (
    Path(__file__).resolve().parents[4]
    / "assets"
    / "generated"
    / "2026-06-10_11-30-08"
    / "single_palm_leap"
    / "right_t4_i4_m4_r4"
)
_requires_runtime = pytest.mark.skipif(
    not _MOTHER_ROOT.is_dir() or not torch.cuda.is_available(),
    reason="density-only parity requires the local mother asset and CUDA Warp",
)


@_requires_runtime
def test_density_only_teacher_matches_joint_teacher_without_materializing_kappa_output() -> None:
    r"""相同 q/query/sigma/owner poses 下，distance、density 与 mask 必须逐元素相同。"""

    container = HandContainer.from_cfg(HandContainerCfg(path=_MOTHER_ROOT), require_geometry_semantics=True)
    assert container.geometry_semantics is not None
    spec_cpu = lower_hand_geometry_semantics(container.geometry_semantics)
    geometry_cache = materialize_owner_geometry_cache(container, spec_cpu)
    warp_cache = materialize_warp_owner_geometry_cache(geometry_cache, device="cuda:0")
    spec = spec_cpu.to(device="cuda:0", dtype=torch.float32)
    surface_sampling = materialize_owner_surface_sampling_cache(
        geometry_cache,
        device="cuda:0",
        dtype=torch.float32,
    )
    anchors = sample_palm_anchor_supports(
        geometry_cache,
        container.geometry_semantics,
        spec_cpu,
        anchors_per_finger=10,
        sampling_seed=101,
    )
    q = spec.q_home.unsqueeze(0)
    owner_transforms, current_screws = forward_owner_transforms_and_spatial_screws(spec, q)
    queries = sample_spatial_queries(
        q,
        spec,
        surface_sampling,
        torch.as_tensor(anchors.anchors_hand_m, device=q.device, dtype=q.dtype),
        config=SpatialQuerySamplerCfg(query_count=64),
        sampling_seed=103,
        owner_transforms=owner_transforms,
    )
    density_only = generate_density_field_targets(
        q,
        spec,
        geometry_cache,
        warp_cache,
        queries,
        sampling_seed=107,
        owner_transforms=owner_transforms,
    )
    reference, sensitivity = generate_geometry_field_targets(
        q,
        spec,
        geometry_cache,
        warp_cache,
        queries,
        edge_sampling_seed=107,
        owner_transforms=owner_transforms,
        current_spatial_screws=current_screws,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(density_only.distance, reference.distance, atol=0.0, rtol=0.0)
    torch.testing.assert_close(density_only.density, reference.density, atol=0.0, rtol=0.0)
    torch.testing.assert_close(density_only.bandwidths, reference.bandwidths, atol=0.0, rtol=0.0)
    assert torch.equal(density_only.valid_mask, reference.valid_mask)
    assert density_only.provenance["first_order_teacher"] == "absent"
    assert sensitivity.kappa.numel() > 0  # reference 路径确实物化了新路径已删除的一阶输出
