r"""Geometry SSL runtime 的 q cursor、Q block 与 resident window 合同。"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import torch
from anymani.distill.models.input_adapters.geometry import GeometryPaddingCfg, StaticGeometryEvidence
from anymani.distill.representations.geometry import (
    OnlineGeometrySample,
    SobolJointSampler,
    pad_online_geometry_samples,
    split_online_geometry_sample,
)
from anymani.distill.representations.queries.spatial_sampling import SpatialQueryBatch
from anymani.distill.representations.targets.field_samples import (
    FieldTargetBatch,
    QueryStratum,
    SensitivityTargetBatch,
)
from anymani.distill.ssl.runtime import (
    GeometrySSLRuntimeCfg,
    ResidentGeometryAssetWindow,
    WindowedOnlineGeometryBatcher,
)
from anymani.distill.ssl.runtime.validation import _update_training_q_bank_digest

pytestmark = pytest.mark.contract


def _spec() -> SimpleNamespace:
    return SimpleNamespace(joint_limits=torch.tensor([[-1.0, 1.0], [-0.5, 0.5]], dtype=torch.float64))


def _sample(q: torch.Tensor) -> OnlineGeometrySample:
    """构造 Q 个构型共享 edge selectors 的最小 teacher block。"""

    batch_size, owner_count, query_count, bandwidth_count, edge_count = q.shape[0], 3, 3, 2, 2
    points = torch.arange(batch_size * owner_count * query_count * 3, dtype=torch.float64).reshape(
        batch_size, owner_count, query_count, 3
    )
    strata = torch.full((batch_size, owner_count, query_count), int(QueryStratum.OWNER_SHELL), dtype=torch.long)
    distance = torch.full((batch_size, owner_count, query_count), 0.02, dtype=torch.float64)
    field = FieldTargetBatch(
        query_points=points,
        query_stratum=strata,
        distance=distance,
        density=torch.full((batch_size, owner_count, query_count, bandwidth_count), 0.5, dtype=torch.float64),
        valid_mask=torch.ones(batch_size, owner_count, query_count, dtype=torch.bool),
        owner_role=torch.tensor([0, 1, 1]),
        bandwidths=torch.tensor([0.01, 0.03], dtype=torch.float64),
        provenance={"frame": "h", "length_unit": "m"},
    )
    sensitivity = SensitivityTargetBatch(
        owner_index=torch.tensor([0, 1]),
        query_index=torch.tensor([1, 2]),
        joint_index=torch.tensor([0, 1]),
        ancestor_mask=torch.tensor([True, False]),
        closest_point=torch.zeros(batch_size, edge_count, 3, dtype=torch.float64),
        closest_source=torch.zeros(batch_size, edge_count, dtype=torch.long),
        uniqueness_margin=torch.ones(batch_size, edge_count, dtype=torch.float64),
        kappa=torch.zeros(batch_size, edge_count, dtype=torch.float64),
        field_sensitivity=torch.zeros(batch_size, edge_count, bandwidth_count, dtype=torch.float64),
        valid_mask=torch.ones(batch_size, edge_count, dtype=torch.bool),
        provenance={"frame": "h", "distance_unit": "m", "joint_unit": "rad"},
    )
    evidence = StaticGeometryEvidence(
        anchors=torch.tensor([[0.0, 0.0, 0.0], [0.02, 0.01, 0.0]], dtype=torch.float64),
        home_surface_points=torch.tensor(
            [
                [[-0.02, -0.01, 0.0], [0.02, 0.01, 0.0]],
                [[0.02, -0.01, 0.01], [0.04, 0.01, 0.01]],
                [[0.05, -0.01, 0.01], [0.07, 0.01, 0.01]],
            ],
            dtype=torch.float64,
        ),
        home_surface_mask=torch.ones(3, 2, dtype=torch.bool),
        palm_normal=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        space_screws=torch.tensor(
            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, -0.02, 0.0]],
            dtype=torch.float64,
        ),
        q_home=torch.zeros(2, dtype=torch.float64),
        entity_role=torch.tensor([0, 1, 1]),
        entity_joint_index=torch.tensor([-1, 0, 1]),
        joint_entity_index=torch.tensor([1, 2]),
        shortest_path=torch.tensor([[0, 1, 2], [1, 0, 1], [2, 1, 0]]),
        parent_direction=torch.tensor([[0, 1, 2], [0, 0, 1], [0, 0, 0]]),
        child_direction=torch.tensor([[0, 0, 0], [1, 0, 0], [2, 1, 0]]),
    )
    return OnlineGeometrySample(
        asset_id="synthetic",
        q=q,
        evidence=evidence,
        queries=SpatialQueryBatch(points, strata, torch.full_like(strata, -1), torch.full_like(strata, -1)),
        field_targets=field,
        sensitivity_targets=sensitivity,
        q_index=torch.arange(batch_size),
    )


def test_sobol_cursor_resume_reproduces_the_next_q_block() -> None:
    """保存每资产 cursor 后，resume 的下一个 Q 构型块必须逐元素一致。"""

    first = SobolJointSampler(_spec(), seed=20260813)
    first.draw(5, device="cpu", dtype=torch.float64)
    state = first.state_dict()
    expected_next = first.draw(3, device="cpu", dtype=torch.float64)

    resumed = SobolJointSampler(_spec(), seed=20260813)
    resumed.load_state_dict(state)
    actual_next = resumed.draw(3, device="cpu", dtype=torch.float64)

    torch.testing.assert_close(actual_next, expected_next, atol=0.0, rtol=0.0)
    assert resumed.cursor == 8


def test_q_block_split_matches_padding_of_individual_q_samples() -> None:
    """Q=2 block 只切 batch 轴时，结果必须等于两个 Q=1 样本的 padding。"""

    block = _sample(torch.tensor([[0.1, -0.2], [0.3, 0.4]], dtype=torch.float64))
    split = split_online_geometry_sample(block)
    block_batch = pad_online_geometry_samples(list(split), padding=GeometryPaddingCfg(max_joint_count=2, max_tip_count=1))
    individual_batch = pad_online_geometry_samples(
        [_sample(block.q[index : index + 1]) for index in range(2)],
        padding=GeometryPaddingCfg(max_joint_count=2, max_tip_count=1),
    )

    torch.testing.assert_close(block_batch.q, individual_batch.q)
    torch.testing.assert_close(block_batch.q_index, torch.tensor([0, 1]))
    torch.testing.assert_close(block_batch.field_targets.density, individual_batch.field_targets.density)
    torch.testing.assert_close(
        block_batch.sensitivity_targets.field_sensitivity,
        individual_batch.sensitivity_targets.field_sensitivity,
    )


def test_validation_digest_covers_sigma_query_routing_and_valid_support() -> None:
    """固定 bank identity 必须对实际 sigma、query routing 与监督有效支持集敏感。"""

    batch = pad_online_geometry_samples(
        list(split_online_geometry_sample(_sample(torch.tensor([[0.1, -0.2]], dtype=torch.float64)))),
        padding=GeometryPaddingCfg(max_joint_count=2, max_tip_count=1),
    )

    def digest_for(value) -> str:
        digest = hashlib.sha256()
        _update_training_q_bank_digest(digest, value)
        return digest.hexdigest()

    baseline = digest_for(batch)
    batch.field_targets.bandwidths[0, 0] += 1.0e-4
    changed_sigma = digest_for(batch)
    batch.queries.workspace_anchor_index[0, 0, 0] = 1
    changed_routing = digest_for(batch)
    batch.sensitivity_targets.valid_mask[0, 0] = False
    changed_support = digest_for(batch)

    assert changed_sigma != baseline
    assert changed_routing != changed_sigma
    assert changed_support != changed_routing


def test_resident_window_evicts_old_asset_and_enforces_cap() -> None:
    """窗口切换必须释放旧 BVH lease，重复 ensure 不增加 resident 数。"""

    runtimes = tuple(SimpleNamespace(asset_id=f"asset-{index}") for index in range(3))
    released: list[str] = []

    def loader(runtime, *, device, dtype):
        return SimpleNamespace(source=runtime, warp_cache=SimpleNamespace(asset_id=runtime.asset_id))

    def releaser(state):
        released.append(state.warp_cache.asset_id)
        return True

    window = ResidentGeometryAssetWindow(
        runtimes,
        device="cpu",
        dtype=torch.float32,
        max_resident_assets=2,
        loader=loader,
        releaser=releaser,
    )
    window.ensure(("asset-0", "asset-1"))
    window.ensure(("asset-0", "asset-1"))
    assert window.resident_asset_ids == ("asset-0", "asset-1")
    first_events = window.drain_telemetry_events()
    assert len(first_events) == 1  # 稳态 ensure 不得制造同步/telemetry 开销
    assert first_events[0]["resident_asset_count"] == 2
    assert first_events[0]["resident_owner_bvh_count"] == 0  # synthetic cache 未构造真实 Warp handles
    window.ensure(("asset-2",))
    assert window.resident_asset_ids == ("asset-2",)
    assert set(released) == {"asset-0", "asset-1"}
    with pytest.raises(ValueError, match="exceeds max_resident_assets"):
        window.ensure(("asset-0", "asset-1", "asset-2"))
    window.release_all()
    assert released == ["asset-0", "asset-1", "asset-2"]
    final_events = window.drain_telemetry_events()
    assert [event["event"] for event in final_events] == ["resident_window", "resident_window_release_all"]


def test_runtime_config_freezes_declared_microbatch_axes() -> None:
    """配置中的逻辑 batch 必须与 A_mb*Q_mb 一致，避免 silent reshape。"""

    config = GeometrySSLRuntimeCfg(assets_per_microbatch=2, q_per_asset_per_microbatch=2)
    assert config.max_resident_assets == 20
    with pytest.raises(ValueError, match="resident"):
        GeometrySSLRuntimeCfg(max_resident_assets=1, assets_per_microbatch=2)


def test_epoch_scheduler_finishes_each_resident_window_before_switching() -> None:
    """资产数超过 cap 时，同一 window 的全部 q coverage 应连续完成，避免反复重建 BVH。"""

    runtimes = tuple(
        SimpleNamespace(asset_id=f"asset-{index}", spec_cpu=_spec()) for index in range(5)
    )
    runtime = WindowedOnlineGeometryBatcher(
        runtimes,
        SimpleNamespace(),
        seed=7,
        runtime_config=GeometrySSLRuntimeCfg(
            max_resident_assets=4,
            assets_per_microbatch=2,
            q_per_asset_per_microbatch=1,
            q_per_asset_per_epoch=2,
            epochs=1,
        ),
        field_config=SimpleNamespace(),
        query_config=SimpleNamespace(),
        target_config=SimpleNamespace(),
        padding=GeometryPaddingCfg(max_joint_count=2, max_tip_count=1),
    )

    assert runtime._groups == ((0, 1), (2, 3), (0, 1), (2, 3), (4,), (4,))
    assert runtime.blocks_per_epoch == 6
