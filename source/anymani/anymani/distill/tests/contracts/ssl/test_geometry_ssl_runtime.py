r"""Geometry SSL runtime 的 q cursor、minibatch 与 resident window 合同。"""

from __future__ import annotations

import hashlib
import inspect
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
import torch
from anymani.distill.methods.contracts import (
    AdditiveStatistic,
    MethodParameterGroup,
    MethodStep,
    ObjectiveTermResult,
)
from anymani.distill.methods.multi_anchor_gaussian_implicit_field import evaluation as method_evaluation
from anymani.distill.methods.multi_anchor_gaussian_implicit_field import method as method_module
from anymani.distill.methods.multi_anchor_gaussian_implicit_field.augmentation import (
    maybe_rewrite_batch,
    permute_online_geometry_sample,
    sample_entity_permutation,
    validate_entity_permutation_transform,
)
from anymani.distill.methods.multi_anchor_gaussian_implicit_field.batch import (
    OnlineGeometrySample,
    pad_online_geometry_samples,
    split_online_geometry_sample,
    split_padded_online_geometry_batch,
)
from anymani.distill.methods.multi_anchor_gaussian_implicit_field.config import (
    EntityPermutationCfg,
    JointSignRewriteCfg,
)
from anymani.distill.methods.multi_anchor_gaussian_implicit_field.evaluation import update_evaluation_digest
from anymani.distill.methods.multi_anchor_gaussian_implicit_field.method import (
    MultiAnchorGaussianMethod,
    PhysicalAuditHandle,
    _derive_padding,
    _forward_microbatch_samples,
)
from anymani.distill.methods.multi_anchor_gaussian_implicit_field.state_measure import SobolJointSampler
from anymani.distill.models.backbones.geometry_transformer import GraphBiasedTransformerCfg
from anymani.distill.models.decoders.representations.implicit_field import (
    DistanceSensitivityDecoderCfg,
    GeometrySSLDecoderCfg,
    ScalarSigmaFiLMDensityDecoderCfg,
)
from anymani.distill.models.geometry_ssl import GeometrySSLModel, GeometrySSLModelCfg
from anymani.distill.models.input_adapters.geometry import (
    GeometryEncoderCfg,
    GeometryPaddingCfg,
    SO2AnchorFrontendCfg,
    SO2AnchorRelationEncoder,
    StaticGeometryEvidence,
    pad_static_geometry_evidence,
)
from anymani.distill.representations.queries.spatial_sampling import SpatialQueryBatch
from anymani.distill.representations.sources import cache as source_cache_module
from anymani.distill.representations.sources.cache import GeometrySourceArena
from anymani.distill.representations.targets.field_samples import (
    FieldTargetBatch,
    QueryStratum,
    SensitivityTargetBatch,
)
from anymani.distill.ssl.runtime import ResidentGeometryAssetWindow

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


def test_padding_derivation_reads_only_typed_semantic_axis_lengths() -> None:
    r"""全 catalog padding 不得为离散 shape 上限构造 POE/FK/graph 浮点张量。"""

    def asset(asset_id: str, *, joint_count: int, tip_count: int) -> SimpleNamespace:
        r"""构造只暴露 padding 所需 typed 轴的最小资产。"""

        owners = (SimpleNamespace(role="palm"),) + tuple(
            SimpleNamespace(role="joint") for _ in range(joint_count)
        ) + tuple(SimpleNamespace(role="tip") for _ in range(tip_count))
        semantics = SimpleNamespace(
            active_joint_names=tuple(f"joint_{index}" for index in range(joint_count)),
            owners=owners,
        )
        return SimpleNamespace(asset_id=asset_id, geometry_semantics=semantics)

    padding = _derive_padding(
        (asset("small", joint_count=7, tip_count=3), asset("large", joint_count=16, tip_count=4)),
        max_graph_distance=8,
    )

    assert padding.max_joint_count == 16  # 当前正式 catalog 的 $N_J^{max}$ 数值锚点
    assert padding.max_tip_count == 4  # 当前正式 catalog 的 $N_{tip}^{max}$ 数值锚点
    assert padding.max_graph_distance == 8  # backbone graph-distance 截断不由资产重写


def test_train_and_evaluation_use_64_sample_microbatches() -> None:
    r"""train/eval 的普通批处理路径统一使用 64-sample 形状合同。"""

    assert _forward_microbatch_samples("train") == 64
    assert _forward_microbatch_samples("eval") == 64


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


def test_lazy_sobol_checkpoint_only_materializes_visited_assets() -> None:
    """epoch-0/早期 checkpoint 的 sampler state必须稀疏，不遍历未访问 catalog。"""

    sources = SimpleNamespace(
        assets=tuple(
            SimpleNamespace(
                geometry_semantics=SimpleNamespace(
                    active_joint_names=("j0", "j1"),
                    joints=(),
                )
            )
            for _ in range(10)
        ),
        asset_ids=tuple(f"asset-{index}" for index in range(10)),
        __len__=lambda self: 10,
    )
    # 直接构造轻量测试替身，避免 typed semantics lowering；只核对 state_dict 不访问缺失 sampler。
    lazy = method_module.LazySobolSamplers.__new__(method_module.LazySobolSamplers)
    lazy.sources = sources
    lazy.seed = 17
    lazy._samplers = {3: SobolJointSampler(_spec(), seed=20)}
    lazy._samplers[3].draw(5, device="cpu", dtype=torch.float64)

    state = lazy.state_dict()

    assert state.keys() == {"3"}
    assert state["3"]["cursor"] == 5


def test_geometry_source_arena_enforces_entry_and_byte_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""CPU source 复用必须由 entry/array bytes 双上限约束，且 clear 后不留进程状态。"""

    monkeypatch.setattr(
        source_cache_module,
        "_cache_key",
        lambda container, _config: container.asset_id,
    )
    arena = GeometrySourceArena(
        max_entries=2,
        max_bytes=12,
        size_of=lambda source: source.size_bytes,
    )
    config = SimpleNamespace()
    containers = tuple(SimpleNamespace(asset_id=f"asset-{index}") for index in range(3))
    sources = tuple(SimpleNamespace(asset_id=item.asset_id, size_bytes=6) for item in containers)

    first = arena.load_or_create(containers[0], config=config, materialize=lambda: sources[0])
    repeated = arena.load_or_create(containers[0], config=config, materialize=lambda: sources[0])
    arena.load_or_create(containers[1], config=config, materialize=lambda: sources[1])
    arena.load_or_create(containers[2], config=config, materialize=lambda: sources[2])

    assert repeated is first  # 命中返回同一 immutable source，不重复 realization
    assert arena.resident_count == 2  # 三项访问后只保留两个 MRU source
    assert arena.resident_bytes == 12  # 测试数组预算与 entry cap 同时闭合
    assert arena.stats()["hits"] == 1
    assert arena.stats()["misses"] == 3
    assert arena.stats()["evictions"] == 1
    arena.clear()
    assert arena.resident_count == arena.resident_bytes == 0
    oversize = SimpleNamespace(asset_id="oversize", size_bytes=13)
    arena.load_or_create(
        SimpleNamespace(asset_id="oversize"),
        config=config,
        materialize=lambda: oversize,
    )
    assert arena.resident_count == arena.resident_bytes == 0  # 单项也不得突破 12 B 测试硬界


def test_geometry_source_arena_materializes_one_copy_per_key_under_prefetch_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""并行 prefetch 同一资产只能执行一次 mesh/source realization。"""

    monkeypatch.setattr(source_cache_module, "_cache_key", lambda container, _config: container.asset_id)
    arena = GeometrySourceArena(max_entries=2, max_bytes=16, size_of=lambda source: source.size_bytes)
    container = SimpleNamespace(asset_id="shared")
    source = SimpleNamespace(asset_id="shared", size_bytes=4)
    materialize_count = 0

    def materialize():
        nonlocal materialize_count
        materialize_count += 1
        return source

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(arena.load_or_create, container, config=SimpleNamespace(), materialize=materialize)
            for _ in range(16)
        ]
    assert all(future.result() is source for future in futures)
    assert materialize_count == 1  # per-key lock 消除并发重复 Boolean/source 构造
    assert arena.stats()["misses"] == 1
    assert arena.stats()["hits"] == 15


def test_physical_audit_handle_cancel_stops_cooperative_worker() -> None:
    r"""训练异常 teardown 必须停止后台 audit，而不是继续遍历完整资产 catalog。"""

    cancel_event = Event()
    started = Event()

    def audit_worker() -> dict[str, object]:
        started.set()
        cancel_event.wait(timeout=5.0)  # 正式 worker 在两项 source 之间检查同一 Event
        if cancel_event.is_set():
            raise RuntimeError("physical asset audit cancelled before completion")
        return {"status": "unexpected-completion"}

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(audit_worker)
    handle = PhysicalAuditHandle(future, executor, cancel_event)
    assert started.wait(timeout=1.0)
    handle.cancel()
    handle.cancel()  # teardown 可由嵌套 finally 重复调用，必须保持幂等
    assert future.done()


def test_joint_sign_rewrite_keeps_density_and_flips_matching_kappa(monkeypatch: pytest.MonkeyPatch) -> None:
    """选中改写后 density/distance 不变，对应 JOINT 的 κ/g 翻号。"""

    sample = _sample(torch.tensor([[0.25, -0.4]], dtype=torch.float64))
    sample = replace(
        sample,
        sensitivity_targets=replace(
            sample.sensitivity_targets,
            central_difference=torch.tensor([[0.31, 0.02]], dtype=torch.float64),
            central_difference_valid_mask=torch.ones(1, 2, dtype=torch.bool),
            central_difference_plus_face=torch.tensor([[10, 20]]),
            central_difference_minus_face=torch.tensor([[11, 21]]),
        ),
    )
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
    torch.testing.assert_close(
        rewritten.sensitivity_targets.central_difference,
        batch.sensitivity_targets.central_difference * edge_sign,
    )
    flipped_edge = edge_sign < 0
    assert rewritten.sensitivity_targets.central_difference_plus_face is not None
    assert rewritten.sensitivity_targets.central_difference_minus_face is not None
    torch.testing.assert_close(
        rewritten.sensitivity_targets.central_difference_plus_face,
        torch.where(
            flipped_edge,
            batch.sensitivity_targets.central_difference_minus_face,
            batch.sensitivity_targets.central_difference_plus_face,
        ),
    )
    torch.testing.assert_close(
        rewritten.sensitivity_targets.central_difference_minus_face,
        torch.where(
            flipped_edge,
            batch.sensitivity_targets.central_difference_plus_face,
            batch.sensitivity_targets.central_difference_minus_face,
        ),
    )


def test_entity_permutation_synchronously_relabels_every_owner_axis() -> None:
    r"""合法 $P$ 同步变换 evidence/query/field/selector/provenance，JOINT coordinate axis 保持不变。"""

    sample = _sample(torch.tensor([[0.25, -0.4], [0.1, 0.2]], dtype=torch.float64))
    permutation = torch.tensor([2, 0, 1], dtype=torch.long)  # new slot -> old slot
    inverse = torch.tensor([1, 2, 0], dtype=torch.long)
    permuted = permute_online_geometry_sample(sample, permutation)

    torch.testing.assert_close(permuted.q, sample.q, atol=0.0, rtol=0.0)
    torch.testing.assert_close(permuted.evidence.space_screws, sample.evidence.space_screws, atol=0.0, rtol=0.0)
    torch.testing.assert_close(permuted.evidence.q_home, sample.evidence.q_home, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        permuted.evidence.home_surface_points,
        sample.evidence.home_surface_points[permutation],
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        permuted.evidence.shortest_path,
        sample.evidence.shortest_path[permutation][:, permutation],
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        permuted.queries.query_points_h,
        sample.queries.query_points_h[:, permutation],
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        permuted.field_targets.density,
        sample.field_targets.density[:, permutation],
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        permuted.sensitivity_targets.owner_index,
        inverse[sample.sensitivity_targets.owner_index],
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        permuted.sensitivity_targets.joint_index,
        sample.sensitivity_targets.joint_index,
        atol=0.0,
        rtol=0.0,
    )
    assert torch.equal(permuted.evidence.joint_entity_index, inverse[sample.evidence.joint_entity_index])
    closest_owner = permuted.sensitivity_targets.closest_source.bitwise_right_shift(32)
    assert torch.equal(closest_owner, permuted.sensitivity_targets.owner_index.expand_as(closest_owner))


def test_unsynchronized_entity_permutation_is_rejected_with_the_first_bad_axis() -> None:
    r"""只重排 token 而保留原 graph 时，完整 entity permutation contract 必须 fail closed。"""

    sample = _sample(torch.tensor([[0.25, -0.4], [0.1, 0.2]], dtype=torch.float64))
    permutation = torch.tensor([2, 0, 1], dtype=torch.long)
    synchronized = permute_online_geometry_sample(sample, permutation)
    validate_entity_permutation_transform(sample, synchronized, permutation)

    malformed_evidence = replace(
        synchronized.evidence,
        shortest_path=sample.evidence.shortest_path,
    )
    malformed = replace(synchronized, evidence=malformed_evidence)
    with pytest.raises(ValueError, match=r"sample\.evidence\.shortest_path"):
        validate_entity_permutation_transform(sample, malformed, permutation)


def test_entity_permutation_seed_is_stable_and_q_block_specific() -> None:
    """稳定 digest 不消费全局 RNG，同一资产/q-block 可复现而相邻 block 使用独立双射。"""

    config = EntityPermutationCfg(enabled=True, seed_offset=31_337)
    rng_before = torch.get_rng_state().clone()
    first = sample_entity_permutation(12, asset_id="asset-a", q_block_start=64, root_seed=17, config=config)
    repeated = sample_entity_permutation(12, asset_id="asset-a", q_block_start=64, root_seed=17, config=config)
    next_block = sample_entity_permutation(12, asset_id="asset-a", q_block_start=72, root_seed=17, config=config)

    assert torch.equal(first, repeated)
    assert not torch.equal(first, next_block)
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_full_model_is_equivariant_to_legal_entity_permutation() -> None:
    r"""同步 $P$ 后 unified $Z$/density 随 owner 轴等变，sampled-edge $\kappa$ 数值保持不变。"""

    torch.manual_seed(91)
    sample = _sample(torch.tensor([[0.25, -0.4], [0.1, 0.2]], dtype=torch.float64))
    permutation = torch.tensor([2, 0, 1], dtype=torch.long)
    permuted = permute_online_geometry_sample(sample, permutation)
    model = GeometrySSLModel(
        GeometrySSLModelCfg(
            encoder=GeometryEncoderCfg(
                frontend=SO2AnchorFrontendCfg(relation_width=8, home_width=8, screw_width=8),
                backbone=GraphBiasedTransformerCfg(
                    hidden_width=16,
                    layers=1,
                    attention_heads=4,
                    feedforward_width=24,
                    dropout=0.0,
                ),
            ),
            ssl_decoders=GeometrySSLDecoderCfg(
                density=ScalarSigmaFiLMDensityDecoderCfg(hidden_width=16, residual_blocks=1),
                sensitivity=DistanceSensitivityDecoderCfg(
                    hidden_width=16,
                    residual_blocks=2,
                    readout_rank=8,
                ),
            ),
        )
    ).to(dtype=torch.float64).eval()

    def predict(value: OnlineGeometrySample):
        return model(
            value.q,
            value.evidence,
            value.queries.query_points_h,
            value.field_targets.bandwidths,
            owner_index=value.sensitivity_targets.owner_index,
            query_index=value.sensitivity_targets.query_index,
            joint_index=value.sensitivity_targets.joint_index,
        )

    canonical_prediction = predict(sample)
    permuted_prediction = predict(permuted)
    torch.testing.assert_close(
        permuted_prediction.latents.entities,
        canonical_prediction.latents.entities[:, permutation],
        atol=2.0e-6,
        rtol=2.0e-6,
    )
    torch.testing.assert_close(
        permuted_prediction.density,
        canonical_prediction.density[:, permutation],
        atol=2.0e-6,
        rtol=2.0e-6,
    )
    torch.testing.assert_close(
        permuted_prediction.kappa,
        canonical_prediction.kappa,
        atol=2.0e-6,
        rtol=2.0e-6,
    )


def test_reused_batch_draws_a_new_joint_sign_rewrite_for_each_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""同一 teacher batch 的相邻 mini-epoch 使用不同 forward index，并重新选择等价 JOINT 坐标。"""

    sample = _sample(torch.tensor([[0.25, -0.4]], dtype=torch.float64))
    batch = pad_online_geometry_samples(
        list(split_online_geometry_sample(sample)),
        padding=GeometryPaddingCfg(max_joint_count=2, max_tip_count=1),
    )
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.zeros(*args))
    first = maybe_rewrite_batch(batch, config=JointSignRewriteCfg(probability=0.20, seed_offset=0), step=0, seed=0)
    second = maybe_rewrite_batch(batch, config=JointSignRewriteCfg(probability=0.20, seed_offset=0), step=1, seed=0)

    first_flipped = torch.where((first.q[0] / batch.q[0]) < 0)[0]
    second_flipped = torch.where((second.q[0] / batch.q[0]) < 0)[0]
    assert first_flipped.tolist() == [0]
    assert second_flipped.tolist() == [1]


def test_trainer_and_checkpoint_only_use_method_contract() -> None:
    """Trainer/checkpoint 不得读取 concrete geometry model、batch、source、sampler 或 config。"""

    from anymani.distill.ssl import checkpoint
    from anymani.distill.ssl.runtime import lifecycle, post_training

    forbidden = (
        "GeometrySSLModel",
        "PaddedOnlineGeometryBatch",
        "SobolJointSampler",
        "method.config",
        "method.train_sources",
        "method.require_model",
    )
    for module in (lifecycle, post_training, checkpoint):
        source = Path(inspect.getsourcefile(module) or "").read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source, f"{module.__name__} still contains {token}"
    lifecycle_source = Path(inspect.getsourcefile(lifecycle) or "").read_text(encoding="utf-8")
    post_training_source = Path(inspect.getsourcefile(post_training) or "").read_text(encoding="utf-8")
    assert "method.open_session" in lifecycle_source
    assert "method.evaluate_session" not in lifecycle_source
    assert "method.evaluate_session" in post_training_source


def test_q_block_split_matches_padding_of_individual_q_samples() -> None:
    """Q=2 block 只切 batch 轴时，结果必须等于两个 Q=1 样本的 padding。"""

    block = _sample(torch.tensor([[0.1, -0.2], [0.3, 0.4]], dtype=torch.float64))
    split = split_online_geometry_sample(block)
    block_batch = pad_online_geometry_samples([block], padding=GeometryPaddingCfg(max_joint_count=2, max_tip_count=1))
    individual_batch = pad_online_geometry_samples(
        list(split),
        padding=GeometryPaddingCfg(max_joint_count=2, max_tip_count=1),
    )

    torch.testing.assert_close(block_batch.q, individual_batch.q)
    torch.testing.assert_close(block_batch.q_index, torch.tensor([0, 1]))
    torch.testing.assert_close(block_batch.field_targets.density, individual_batch.field_targets.density)
    torch.testing.assert_close(
        block_batch.sensitivity_targets.field_sensitivity,
        individual_batch.sensitivity_targets.field_sensitivity,
    )


def test_forward_microbatch_split_preserves_sample_axis_and_nested_targets() -> None:
    r"""GPU microbatch 只切样本轴，拼回后必须逐元素等于原 logical minibatch。"""

    block = _sample(
        torch.tensor(
            [[0.1, -0.2], [0.3, 0.4], [-0.5, 0.2], [0.7, -0.1], [-0.3, -0.4]],
            dtype=torch.float64,
        )
    )
    batch = pad_online_geometry_samples(
        list(split_online_geometry_sample(block)),
        padding=GeometryPaddingCfg(max_joint_count=2, max_tip_count=1),
    )
    pieces = split_padded_online_geometry_batch(batch, microbatch_size=2)

    assert tuple(piece.q.shape[0] for piece in pieces) == (2, 2, 1)
    assert tuple(asset_id for piece in pieces for asset_id in piece.asset_ids) == batch.asset_ids
    torch.testing.assert_close(torch.cat([piece.q for piece in pieces]), batch.q)
    for piece in pieces:
        torch.testing.assert_close(piece.evidence.home_surface_points, batch.evidence.home_surface_points)
    torch.testing.assert_close(
        torch.cat([piece.evidence_row_index for piece in pieces if piece.evidence_row_index is not None]),
        batch.evidence_row_index,
    )
    torch.testing.assert_close(
        torch.cat([piece.field_targets.density for piece in pieces]),
        batch.field_targets.density,
    )
    torch.testing.assert_close(
        torch.cat([piece.sensitivity_targets.field_sensitivity for piece in pieces]),
        batch.sensitivity_targets.field_sensitivity,
    )


def test_streaming_backward_matches_full_group_additive_gradient(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""逐块 task gradients 必须形成 FairGrad shared 方向和各自 private reader 梯度。"""

    block = _sample(torch.tensor([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.2], [0.7, -0.1]]))
    batch = pad_online_geometry_samples(
        list(split_online_geometry_sample(block)),
        padding=GeometryPaddingCfg(max_joint_count=2, max_tip_count=1),
    )
    shared = torch.nn.Parameter(torch.tensor(2.0))
    density_private = torch.nn.Parameter(torch.tensor(1.0))
    kappa_private = torch.nn.Parameter(torch.tensor(3.0))
    names = ("density", "kappa")
    enabled = {name: SimpleNamespace(weight=1.0) for name in names}

    def synthetic_forward(microbatch, **_kwargs):
        r"""为每个样本贡献 $p^2$，使 microbatch numerator 可加且 denominator 为样本数。"""

        count = shared.new_tensor(float(microbatch.q.shape[0]))
        objectives = {}
        for name in names:
            private = density_private if name == "density" else kappa_private
            numerator = (shared.square() + private.square()) * count
            statistic = AdditiveStatistic(name, numerator, count)
            objectives[name] = ObjectiveTermResult(name, (statistic,), {"loss": statistic.mean})
        prediction = SimpleNamespace(
            density=torch.zeros_like(microbatch.field_targets.density),
            kappa=torch.zeros_like(microbatch.sensitivity_targets.kappa),
        )
        return MethodStep(objectives=objectives, sample_count=microbatch.q.shape[0]), prediction

    fake_method = SimpleNamespace(
        config=SimpleNamespace(
            objectives=SimpleNamespace(enabled=lambda: enabled),
            joint_sign_rewrite=SimpleNamespace(),
            fairgrad=SimpleNamespace(near_opposition_tolerance=1.0e-6),
        ),
        _forward_with_prediction=synthetic_forward,
        optimizer_parameter_groups=lambda: (
            MethodParameterGroup("shared_encoder", (shared,)),
            MethodParameterGroup("density_reader", (density_private,)),
            MethodParameterGroup("kappa_reader", (kappa_private,)),
        ),
    )
    fake_method._q_per_asset_block = MultiAnchorGaussianMethod._q_per_asset_block  # type: ignore[attr-defined]
    fake_method._training_minibatch_denominators = lambda batch: (  # type: ignore[attr-defined]
        MultiAnchorGaussianMethod._training_minibatch_denominators(fake_method, batch)
    )
    monkeypatch.setattr(method_module, "maybe_rewrite_batch", lambda value, **_kwargs: value)

    update = MultiAnchorGaussianMethod.backward_update(fake_method, batch, forward_step=0, microbatch_size=4)

    assert update.sample_count == 4
    assert update.terms == {"density": pytest.approx(5.0), "kappa": pytest.approx(13.0)}
    assert float(shared.grad) == pytest.approx(2.0**0.5)  # 两个同向单位 task gradients 的解析和
    assert float(density_private.grad) == pytest.approx(2.0)
    assert float(kappa_private.grad) == pytest.approx(6.0)


def test_streaming_backward_collects_unified_z_gradient_sufficient_statistics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""microbatch-denominator 下的 rho/kappa $Z$ 梯度应产生可重算 norm、dot、cosine 与 Gram 证据。"""

    block = _sample(torch.tensor([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.2], [0.7, -0.1]]))
    batch = pad_online_geometry_samples(
        list(split_online_geometry_sample(block)),
        padding=GeometryPaddingCfg(max_joint_count=2, max_tip_count=1),
    )
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    density_private = torch.nn.Parameter(torch.tensor(0.0))
    kappa_private = torch.nn.Parameter(torch.tensor(0.0))
    enabled = {name: SimpleNamespace(weight=1.0) for name in ("density", "kappa")}

    def synthetic_forward(microbatch, **_kwargs):
        count = parameter.new_tensor(float(microbatch.q.shape[0]))
        entities = parameter * torch.ones(microbatch.q.shape[0], 1, 1)
        rho_numerator = entities.square().sum()  # $\nabla_ZL_\rho$ 在 $p=2$ 时为正
        kappa_numerator = (entities - 3.0).square().sum()  # $\nabla_ZL_\kappa$ 在 $p=2$ 时为负
        objectives = {
            "density": ObjectiveTermResult(
                "density",
                (AdditiveStatistic("density", rho_numerator, count),),
                {"loss": rho_numerator / count},
            ),
            "kappa": ObjectiveTermResult(
                "kappa",
                (AdditiveStatistic("kappa", kappa_numerator, count),),
                {"loss": kappa_numerator / count},
            ),
        }
        prediction = SimpleNamespace(
            latents=SimpleNamespace(entities=entities),
            density=torch.zeros_like(microbatch.field_targets.density),
            kappa=torch.zeros_like(microbatch.sensitivity_targets.kappa),
        )
        return MethodStep(objectives=objectives, sample_count=microbatch.q.shape[0]), prediction

    fake_method = SimpleNamespace(
        config=SimpleNamespace(
            objectives=SimpleNamespace(enabled=lambda: enabled),
            joint_sign_rewrite=SimpleNamespace(),
            fairgrad=SimpleNamespace(near_opposition_tolerance=1.0e-6),
        ),
        _forward_with_prediction=synthetic_forward,
        optimizer_parameter_groups=lambda: (
            MethodParameterGroup("shared_encoder", (parameter,)),
            MethodParameterGroup("density_reader", (density_private,)),
            MethodParameterGroup("kappa_reader", (kappa_private,)),
        ),
    )
    fake_method._q_per_asset_block = MultiAnchorGaussianMethod._q_per_asset_block  # type: ignore[attr-defined]
    fake_method._training_minibatch_denominators = lambda realized: (  # type: ignore[attr-defined]
        MultiAnchorGaussianMethod._training_minibatch_denominators(fake_method, realized)
    )
    monkeypatch.setattr(method_module, "maybe_rewrite_batch", lambda value, **_kwargs: value)

    update = MultiAnchorGaussianMethod.backward_update(
        fake_method,
        batch,
        forward_step=0,
        microbatch_size=4,
        collect_z_gradients=True,
    )

    evidence = update.gradient_evidence
    assert evidence["raw/rho_norm"] > 0.0
    assert evidence["raw/kappa_norm"] > 0.0
    assert evidence["raw/dot"] < 0.0
    assert evidence["raw/cosine"] == pytest.approx(-1.0)
    assert evidence["raw/gram_determinant"] == pytest.approx(0.0, abs=1.0e-12)
    assert evidence["raw/joint_norm"] > 0.0
    assert evidence["fairgrad/shared_conflict_blocked"] == 1.0


def test_ragged_anchor_padding_mask_matches_independent_relation_encoding() -> None:
    r"""30/40-anchor 资产可共享 batch，零 padding 不得进入 $SO(2)$ anchor attention。"""

    full = _sample(torch.tensor([[0.1, -0.2]], dtype=torch.float64)).evidence
    short = replace(full, anchors=full.anchors[:1])
    padded = pad_static_geometry_evidence(
        (full, short),
        config=GeometryPaddingCfg(max_joint_count=2, max_tip_count=1),
    )
    assert padded.anchor_valid_mask is not None
    assert padded.anchor_valid_mask.sum(dim=-1).tolist() == [2, 1]

    torch.manual_seed(47)
    encoder = SO2AnchorRelationEncoder(relation_width=8, length_scale_m=0.1).to(dtype=torch.float64).eval()
    points = torch.tensor(
        [[[0.01, 0.02, 0.03]], [[0.04, -0.01, 0.02]]],
        dtype=torch.float64,
    )
    together = encoder(points, padded.anchors, padded.palm_normal, padded.anchor_valid_mask)
    full_alone = encoder(points[0], full.anchors, full.palm_normal)
    short_alone = encoder(points[1], short.anchors, short.palm_normal)

    torch.testing.assert_close(together[0], full_alone, atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(together[1], short_alone, atol=1.0e-12, rtol=1.0e-12)


def test_validation_digest_covers_static_evidence_sigma_routing_and_support() -> None:
    """固定 bank identity 必须对 model input、实际 sigma、query routing 与监督支持集敏感。"""

    batch = pad_online_geometry_samples(
        list(split_online_geometry_sample(_sample(torch.tensor([[0.1, -0.2]], dtype=torch.float64)))),
        padding=GeometryPaddingCfg(max_joint_count=2, max_tip_count=1),
    )

    def digest_for(value) -> str:
        digest = hashlib.sha256()
        update_evaluation_digest(digest, value)
        return digest.hexdigest()

    baseline = digest_for(batch)
    batch.field_targets.bandwidths[0, 0] += 1.0e-4
    changed_sigma = digest_for(batch)
    batch.queries.workspace_anchor_index[0, 0, 0] = 1
    changed_routing = digest_for(batch)
    batch.sensitivity_targets.valid_mask[0, 0] = False
    changed_support = digest_for(batch)
    batch.evidence.anchors[0, 0, 0] += 1.0e-4
    changed_evidence = digest_for(batch)
    assert batch.anchor_index is not None
    batch.anchor_index[0] += 1
    changed_anchor_bank = digest_for(batch)

    assert changed_sigma != baseline
    assert changed_routing != changed_sigma
    assert changed_support != changed_routing
    assert changed_evidence != changed_support
    assert changed_anchor_bank != changed_evidence


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


def test_resident_window_consumes_pinned_prefetch_buffer_without_second_provider_lookup() -> None:
    r"""next-buffer LRU 插入后，current core 必须由 handle 强引用直接交给 loader，不能 cache miss 重建。"""

    prepared = tuple(SimpleNamespace(asset_id=f"asset-{index}") for index in range(2))

    def unexpected_get(_asset_id: str):
        raise AssertionError("prepared current buffer must bypass source_provider.get")

    provider = SimpleNamespace(get=unexpected_get)
    window = ResidentGeometryAssetWindow(
        prepared,
        device="cpu",
        dtype=torch.float32,
        max_resident_assets=2,
        loader=lambda source, **_kwargs: SimpleNamespace(
            source=source,
            device_source=SimpleNamespace(release=lambda: True),
            warp_cache=SimpleNamespace(handles=()),
        ),
        catalog_ids=tuple(source.asset_id for source in prepared),
        source_provider=provider,
    )

    states = window.ensure(
        tuple(source.asset_id for source in prepared),
        prefetch_sources=False,
        prepared_sources={source.asset_id: source for source in prepared},
    )

    assert tuple(state.source.asset_id for state in states) == ("asset-0", "asset-1")
    window.release_all()


def test_resident_window_default_path_avoids_memory_snapshot_and_reloads_changed_bank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """resource profile 关闭时不触发 CUDA 同步；同资产切换 $A^{(k)}$ 仍必须更新 resident state。"""

    runtime = SimpleNamespace(asset_id="asset-a")
    loaded: list[int] = []
    released: list[int] = []

    def loader(source, *, bank_index: int, **_kwargs):
        loaded.append(bank_index)
        return SimpleNamespace(
            source=source,
            bank_index=bank_index,
            warp_cache=SimpleNamespace(handles=()),
        )

    window = ResidentGeometryAssetWindow(
        (runtime,),
        device="cuda:0",
        dtype=torch.float32,
        max_resident_assets=1,
        loader=loader,
        releaser=lambda state: released.append(state.bank_index) or True,
        resource_profile=False,
    )
    monkeypatch.setattr(
        window,
        "_memory_snapshot",
        lambda: (_ for _ in ()).throw(AssertionError("default resident path must not sample device memory")),
    )

    first = window.ensure(("asset-a",), bank_index=0)[0]
    repeated = window.ensure(("asset-a",), bank_index=0)[0]
    changed = window.ensure(("asset-a",), bank_index=1)[0]

    assert first is repeated
    assert changed.bank_index == 1
    assert loaded == [0, 1]
    assert released == [0]
    window.release_all()


def test_validation_ablation_marks_single_q_same_asset_shuffle_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """单 q 尾块没有合法同资产置换，应记录缺测而不是补样本或终止生命周期。"""

    monkeypatch.setattr(method_evaluation, "geometry_ssl_ablation_forward", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        method_evaluation,
        "geometry_ssl_reconstruction_metrics_per_sample",
        lambda _prediction, _batch: {"density": [1.0], "kappa": [2.0], "derived_field": [3.0]},
    )
    batch = SimpleNamespace(
        q=torch.zeros(1, 2),
        q_index=torch.tensor([7]),
        asset_ids=("asset-a",),
        evidence=object(),
        evidence_row_index=None,
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

    evidence = method_evaluation.fixed_evaluation_ablation_evidence(model, (batch,))

    assert evidence["records"][0]["metrics"]["same_asset_q_shuffle"] is None
