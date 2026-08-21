r"""Geometry SSL runtime 的 q cursor、Q block 与 resident window 合同。"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from anymani.distill.methods.multi_anchor_gaussian_implicit_field.augmentation import maybe_rewrite_batch
from anymani.distill.methods.multi_anchor_gaussian_implicit_field.batch import (
    OnlineGeometrySample,
    pad_online_geometry_samples,
    split_online_geometry_sample,
)
from anymani.distill.methods.multi_anchor_gaussian_implicit_field.config import JointSignRewriteCfg
from anymani.distill.methods.multi_anchor_gaussian_implicit_field.state_measure import SobolJointSampler
from anymani.distill.models.input_adapters.geometry import GeometryPaddingCfg, StaticGeometryEvidence
from anymani.distill.representations.queries.spatial_sampling import SpatialQueryBatch
from anymani.distill.representations.targets.field_samples import (
    FieldTargetBatch,
    QueryStratum,
    SensitivityTargetBatch,
)
from anymani.distill.ssl.runtime import GeometrySSLRuntimeCfg, ResidentGeometryAssetWindow
from anymani.distill.ssl.runtime import validation as validation_runtime
from anymani.distill.ssl.runtime.sampling import OnlineMinibatchSchedule, OnlineSamplingCfg
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
        active_mask=torch.tensor([True, False]),
        closest_point=torch.zeros(batch_size, edge_count, 3, dtype=torch.float64),
        closest_source=torch.zeros(batch_size, edge_count, dtype=torch.long),
        uniqueness_margin=torch.ones(batch_size, edge_count, dtype=torch.float64),
        kappa=torch.tensor([[0.3, 0.0]] * batch_size, dtype=torch.float64),
        field_sensitivity=torch.tensor([[[0.12, -0.04], [0.0, 0.0]]] * batch_size, dtype=torch.float64),
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


def test_joint_sign_rewrite_keeps_density_and_flips_matching_kappa(monkeypatch: pytest.MonkeyPatch) -> None:
    """选中改写后 density/distance 不变，对应 JOINT 的 κ/g 翻号。"""

    sample = _sample(torch.tensor([[0.25, -0.4]], dtype=torch.float64))
    batch = pad_online_geometry_samples(
        list(split_online_geometry_sample(sample)),
        padding=GeometryPaddingCfg(max_joint_count=2, max_tip_count=1),
    )
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.zeros(*args))
    rewritten = maybe_rewrite_batch(
        batch,
        config=JointSignRewriteCfg(probability=0.20, seed_offset=0),
        step=0,
        seed=0,
    )

    torch.testing.assert_close(rewritten.field_targets.density, batch.field_targets.density)
    torch.testing.assert_close(rewritten.field_targets.distance, batch.field_targets.distance)
    assert int((rewritten.q[0] / batch.q[0]).tolist().count(-1.0)) == 1
    flipped = (rewritten.q[0] / batch.q[0]) < 0
    joint_index = int(torch.where(flipped)[0][0])
    edge_sign = torch.where(batch.sensitivity_targets.joint_index[0] == joint_index, -1.0, 1.0)
    torch.testing.assert_close(rewritten.sensitivity_targets.kappa, batch.sensitivity_targets.kappa * edge_sign)
    torch.testing.assert_close(
        rewritten.sensitivity_targets.field_sensitivity,
        batch.sensitivity_targets.field_sensitivity * edge_sign.unsqueeze(-1),
    )


def test_trainer_modules_do_not_construct_sobol_or_read_method_representation() -> None:
    """lifecycle 与 independent q-bank 必须走 method 封闭 sampler，不得直接读 representation。"""

    from anymani.distill.ssl.runtime import lifecycle, validation

    forbidden = ("SobolJointSampler(", "method.representation.")
    for module in (lifecycle, validation):
        source = Path(inspect.getsourcefile(module) or "").read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source, f"{module.__name__} still contains {token}"
    assert "make_independent_samplers" in Path(inspect.getsourcefile(validation) or "").read_text(encoding="utf-8")


def test_q_block_split_matches_padding_of_individual_q_samples() -> None:
    """Q=2 block 只切 batch 轴时，结果必须等于两个 Q=1 样本的 padding。"""

    block = _sample(torch.tensor([[0.1, -0.2], [0.3, 0.4]], dtype=torch.float64))
    split = split_online_geometry_sample(block)
    block_batch = pad_online_geometry_samples(
        list(split), padding=GeometryPaddingCfg(max_joint_count=2, max_tip_count=1)
    )
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


def test_runtime_config_freezes_declared_minibatch_axes() -> None:
    """配置中的逻辑 batch 必须与 A_mb*Q_mb 一致，避免 silent reshape。"""

    config = GeometrySSLRuntimeCfg(assets_per_minibatch=2, q_per_asset_per_minibatch=2)
    assert config.max_resident_assets == 20
    with pytest.raises(ValueError, match="resident"):
        GeometrySSLRuntimeCfg(max_resident_assets=1, assets_per_minibatch=2)


def test_epoch_scheduler_finishes_each_resident_window_before_switching() -> None:
    """资产数超过 cap 时，同一 window 的全部 q coverage 应连续完成，避免反复重建 BVH。"""

    schedule = OnlineMinibatchSchedule(
        5,
        OnlineSamplingCfg(
            epochs=1,
            q_per_asset_per_epoch=2,
            assets_per_minibatch=2,
            q_per_asset_per_minibatch=1,
            shuffle_assets=False,
            seed=7,
        ),
        max_resident_assets=4,
    )
    items = tuple(schedule.next() for _ in range(schedule.minibatches_per_epoch))
    assert tuple(item.resident_asset_indices for item in items) == (
        (0, 1, 2, 3),
        (0, 1, 2, 3),
        (0, 1, 2, 3),
        (0, 1, 2, 3),
        (4,),
        (4,),
    )


def test_epoch_scheduler_preserves_a_smaller_tail_q_block() -> None:
    """不可整除的 q coverage 必须保留真实尾块，不能拒绝配置或重复补齐样本。"""

    schedule = OnlineMinibatchSchedule(
        2,
        OnlineSamplingCfg(
            epochs=1,
            q_per_asset_per_epoch=5,
            assets_per_minibatch=2,
            q_per_asset_per_minibatch=2,
            shuffle_assets=False,
            seed=11,
        ),
        max_resident_assets=2,
    )
    items = tuple(schedule.next() for _ in range(schedule.minibatches_per_epoch))
    assert tuple(item.q_per_asset for item in items) == (2, 2, 1)


def test_validation_ablation_marks_single_q_same_asset_shuffle_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """单 q 尾块没有合法同资产置换，应记录缺测而不是补样本或终止生命周期。"""

    monkeypatch.setattr(validation_runtime, "geometry_ssl_ablation_forward", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        validation_runtime,
        "geometry_ssl_reconstruction_metrics_per_sample",
        lambda _prediction, _batch: {"density": [1.0], "kappa": [2.0], "derived_field": [3.0]},
    )
    batch = SimpleNamespace(
        q=torch.zeros(1, 2),
        q_index=torch.tensor([7]),
        asset_ids=("asset-a",),
        evidence=object(),
        queries=SimpleNamespace(query_points_h=torch.zeros(1, 1, 1, 3)),
        field_targets=SimpleNamespace(bandwidths=torch.ones(1)),
        sensitivity_targets=SimpleNamespace(
            owner_index=torch.tensor([0]),
            query_index=torch.tensor([0]),
            joint_index=torch.tensor([0]),
        ),
    )

    def model(*_args: object, **_kwargs: object) -> object:
        return object()

    evidence = validation_runtime.fixed_validation_ablation_evidence(model, (batch,))

    assert evidence["records"][0]["metrics"]["same_asset_q_shuffle"] is None
