r"""Schema-3 good-pregrasp Top-8 catalog的纯文件系统合同。"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from anymani.pregrasp.good_catalog import (
    GOOD_PREGRASP_TOP_K,
    GoodPregraspCandidate,
    GoodPregraspCatalog,
    GoodPregraspConflictError,
    GoodPregraspEntry,
    GoodPregraspKey,
    GoodPregraspMember,
    GoodPregraspMetrics,
    GoodPregraspMissError,
)
from anymani.pregrasp.mvp80_strict_search import deepest_contact_normal_from_buffers
from anymani.pregrasp.strict_gate import MVP80_STRICT_GOOD_PREGRASP_GATE

_HASH = "1" * 64


def _key(*, scale: float = 1.1) -> GoodPregraspKey:
    r"""构造只改变exact object scale的合法key。"""

    return GoodPregraspKey(
        asset_id="asset-a",
        source_content_hash=_HASH,
        physical_geometry_hash="2" * 64,
        canonical_schema_digest="3" * 64,
        routing_digest="4" * 64,
        object_asset_id="DexCube",
        object_asset_sha256="5" * 64,
        object_scale=scale,
        physics_identity_digest="6" * 64,
        generation_identity_digest="7" * 64,
    )


def _candidate(rank: int) -> GoodPregraspCandidate:
    r"""用不同active q构造八个严格$q_0=u_0$候选。"""

    q = (0.01 * rank, *([0.0] * 15))
    return GoodPregraspCandidate(
        q_state_rad=q,
        q_target_rad=q,
        active_joint_mask=(True, *([False] * 15)),
        object_position_h_m=(0.0, 0.08 + rank * 1.0e-4, 0.054),
    )


def _metrics(rank: int) -> GoodPregraspMetrics:
    r"""构造通过cold-reset gate的紧凑统计fixture。"""

    return GoodPregraspMetrics(
        joint_limit_margin_fraction=0.2,
        envelope_fingers=("thumb", "index", "ring"),
        envelope_sector_min_deg=45.0,
        envelope_tip_center_distance_m=(0.08, 0.081, 0.082),
        penetration_depth_max_m=0.0,
        object_displacement_max_m=0.001 + rank * 1.0e-5,
        object_tilt_max_deg=1.0,
        peak_linear_velocity_m_s=0.02,
        peak_off_axis_angular_velocity_rad_s=0.1,
        palm_contact_fraction=1.0,
        owner_contact_fraction=(1.0, *([0.0] * 20)),
        peak_angular_velocity_rad_s=0.2,
    )


def _entry(*, scale: float = 1.1) -> GoodPregraspEntry:
    r"""构造rank连续、候选互异的完整Top-8 entry。"""

    return GoodPregraspEntry(
        key=_key(scale=scale),
        members=tuple(
            GoodPregraspMember(
                rank=rank,
                candidate=_candidate(rank),
                metrics=_metrics(rank),
                selection_score=(1.0, -float(rank)),
            )
            for rank in range(GOOD_PREGRASP_TOP_K)
        ),
    )


def test_catalog_publishes_idempotently_and_resolves_exact_scale(tmp_path) -> None:
    r"""同key同payload幂等；其他scale必须miss而不能近邻复用。"""

    catalog = GoodPregraspCatalog(tmp_path / "catalog")
    entry = _entry()
    first = catalog.publish(entry)
    second = catalog.publish(entry)
    assert first == second
    assert catalog.resolve(entry.key) == entry
    with pytest.raises(GoodPregraspMissError):
        catalog.resolve(_key(scale=1.2))


def test_catalog_resolve_many_reads_shared_index_once(tmp_path, monkeypatch) -> None:
    r"""80-key preload应只读一次共同index，并保持请求顺序与重复key。"""

    catalog = GoodPregraspCatalog(tmp_path / "catalog")
    first = _entry(scale=1.1)
    second = _entry(scale=1.2)
    catalog.publish(first)
    catalog.publish(second)
    original = catalog._load_index
    calls = 0

    def counted_load():
        r"""记录batch resolve的index读取次数。"""

        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(catalog, "_load_index", counted_load)
    resolved = catalog.resolve_many((second.key, first.key, second.key))
    assert resolved == (second, first, second)
    assert calls == 1


def test_catalog_rejects_same_key_with_changed_top8(tmp_path) -> None:
    r"""同一物理key不能静默覆盖另一组ranked candidates。"""

    catalog = GoodPregraspCatalog(tmp_path / "catalog")
    entry = _entry()
    catalog.publish(entry)
    changed = replace(
        entry,
        members=(
            replace(entry.members[0], selection_score=(2.0, 0.0)),
            *entry.members[1:],
        ),
    )
    with pytest.raises(GoodPregraspConflictError):
        catalog.publish(changed)


def test_candidate_requires_equal_target_upright_and_ghost_zero() -> None:
    r"""MVP candidate拒绝PD preload、倾斜object与ghost非零状态。"""

    candidate = _candidate(1)
    with pytest.raises(ValueError, match="q_target_rad"):
        replace(candidate, q_target_rad=(0.2, *([0.0] * 15)))
    with pytest.raises(ValueError, match="upright"):
        replace(candidate, object_orientation_h_wxyz=(0.0, 0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="inactive"):
        replace(
            candidate,
            q_state_rad=(0.01, 0.02, *([0.0] * 14)),
            q_target_rad=(0.01, 0.02, *([0.0] * 14)),
        )


def test_strict_gate_requires_total_angular_and_validates_every_top8_member() -> None:
    r"""Strict v5拒绝缺失/超限总角速度，并对Top-8逐项执行同一谓词。"""

    entry = _entry()
    MVP80_STRICT_GOOD_PREGRASP_GATE.validate_entry(entry)
    missing_total = replace(entry.members[0].metrics, peak_angular_velocity_rad_s=None)
    assert MVP80_STRICT_GOOD_PREGRASP_GATE.violations(missing_total) == ("missing_total_angular_velocity",)
    unstable = replace(entry.members[0].metrics, peak_angular_velocity_rad_s=2.01)
    assert MVP80_STRICT_GOOD_PREGRASP_GATE.violations(unstable) == ("peak_angular_velocity",)
    rejected_entry = replace(entry, members=(replace(entry.members[0], metrics=unstable), *entry.members[1:]))
    with pytest.raises(ValueError, match="rejected Top-8"):
        MVP80_STRICT_GOOD_PREGRASP_GATE.validate_entry(rejected_entry)


def test_deepest_contact_normal_unpacking_preserves_physx_direction_per_env() -> None:
    r"""Packed contact buffers按env分组选择最负separation及其world normal。"""

    normals = torch.tensor(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    separations = torch.tensor(((-0.001,), (-0.003,), (-0.002,)))
    counts = torch.tensor(((2,), (1,)), dtype=torch.int32)
    starts = torch.tensor(((0,), (2,)), dtype=torch.int32)
    depth, normal = deepest_contact_normal_from_buffers(
        normals,
        separations,
        counts,
        starts,
        environment_count=2,
        body_count=1,
    )
    torch.testing.assert_close(depth, torch.tensor((0.003, 0.002)))
    torch.testing.assert_close(normal, torch.tensor(((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))))
