r"""Pregrasp partial-reset sidecar与policy-step action数学合同。"""

from __future__ import annotations

import pytest
import torch

from anymani.pregrasp import (
    GoodPregraspCandidate,
    GoodPregraspEntry,
    GoodPregraspKey,
    GoodPregraspMember,
    GoodPregraspMetrics,
)

from anymani.tasks.hetero.mdp.runtime_state import (
    HeterogeneousPregraspState,
    ResolvedPregraspBatch,
    compute_policy_step_masked_relative_target,
    normalize_env_ids,
    synchronize_action_reset,
)


def _batch(batch_size: int, *, state_value: float, target_value: float, digest_digit: str) -> ResolvedPregraspBatch:
    r"""构造前10个joint有效、actual/target分离的CPU fixture。"""

    active = torch.zeros(batch_size, 16, dtype=torch.bool)  # canonical$[K,16]$ mask
    active[:, :10] = True
    q_state = torch.zeros(batch_size, 16)  # actual$\mathbf q_s$
    q_target = torch.zeros(batch_size, 16)  # preload$\mathbf q_t$
    q_state[:, :10] = state_value
    q_target[:, :10] = target_value
    position = torch.tensor((0.01, 0.06, 0.08)).repeat(batch_size, 1)  # $p_{ho}$，单位m
    quaternion = torch.tensor((1.0, 0.0, 0.0, 0.0)).repeat(batch_size, 1)  # identity$R_{ho}$
    digest = digest_digit * 64
    return ResolvedPregraspBatch(
        q_state_rad=q_state,
        q_target_rad=q_target,
        active_joint_mask=active,
        object_position_h_m=position,
        object_quat_h_wxyz=quaternion,
        record_digests=(digest,) * batch_size,
        lookup_digests=(digest,) * batch_size,
    )


def test_normalize_env_ids_rejects_duplicate_and_out_of_range_rows() -> None:
    r"""Partial reset索引必须唯一、非空、合法并保持caller顺序。"""

    assert normalize_env_ids([3, 1], num_envs=4, device="cpu").tolist() == [3, 1]
    assert normalize_env_ids(None, num_envs=4, device="cpu").tolist() == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="duplicate"):
        normalize_env_ids([1, 1], num_envs=4, device="cpu")
    with pytest.raises(ValueError, match="out-of-range"):
        normalize_env_ids([4], num_envs=4, device="cpu")


def test_batch_rejects_nonzero_ghost_and_non_unit_object_quaternion() -> None:
    r"""Provider batch边界不能修补ghost或坏$SO(3)$输入。"""

    valid = _batch(2, state_value=0.1, target_value=0.12, digest_digit="1")
    bad_state = valid.q_state_rad.clone()
    bad_state[:, 15] = 0.1  # inactive canonical slot
    with pytest.raises(ValueError, match="ghost"):
        ResolvedPregraspBatch(
            q_state_rad=bad_state,
            q_target_rad=valid.q_target_rad,
            active_joint_mask=valid.active_joint_mask,
            object_position_h_m=valid.object_position_h_m,
            object_quat_h_wxyz=valid.object_quat_h_wxyz,
            record_digests=valid.record_digests,
            lookup_digests=valid.lookup_digests,
        )
    bad_quaternion = valid.object_quat_h_wxyz * 2.0
    with pytest.raises(ValueError, match="unit"):
        ResolvedPregraspBatch(
            q_state_rad=valid.q_state_rad,
            q_target_rad=valid.q_target_rad,
            active_joint_mask=valid.active_joint_mask,
            object_position_h_m=valid.object_position_h_m,
            object_quat_h_wxyz=bad_quaternion,
            record_digests=valid.record_digests,
            lookup_digests=valid.lookup_digests,
        )


def test_good_pregrasp_entry_rank_builds_equal_state_target_batch() -> None:
    r"""Schema-3 Top-8的指定rank必须保留$q_0=u_0$与upright object pose。"""

    active = (True, *([False] * 15))
    key = GoodPregraspKey(
        asset_id="asset-a",
        source_content_hash="1" * 64,
        physical_geometry_hash="2" * 64,
        canonical_schema_digest="3" * 64,
        routing_digest="4" * 64,
        object_asset_id="DexCube",
        object_asset_sha256="5" * 64,
        object_scale=1.1,
        physics_identity_digest="6" * 64,
        generation_identity_digest="7" * 64,
    )
    metrics = GoodPregraspMetrics(
        joint_limit_margin_fraction=0.2,
        envelope_fingers=("thumb", "index", "ring"),
        envelope_sector_min_deg=45.0,
        envelope_tip_center_distance_m=(0.08, 0.09, 0.09),
        penetration_depth_max_m=0.0,
        object_displacement_max_m=0.001,
        object_tilt_max_deg=1.0,
        peak_linear_velocity_m_s=0.02,
        peak_off_axis_angular_velocity_rad_s=0.1,
        palm_contact_fraction=1.0,
        owner_contact_fraction=(1.0, *([0.0] * 20)),
    )
    members = []
    for rank in range(8):
        q = (0.01 * rank, *([0.0] * 15))
        members.append(
            GoodPregraspMember(
                rank=rank,
                candidate=GoodPregraspCandidate(
                    q_state_rad=q,
                    q_target_rad=q,
                    active_joint_mask=active,
                    object_position_h_m=(0.0, 0.08 + rank * 1.0e-4, 0.054),
                ),
                metrics=metrics,
                selection_score=(1.0, -float(rank)),
            )
        )
    entry = GoodPregraspEntry(key=key, members=tuple(members))
    batch = ResolvedPregraspBatch.from_good_entries((entry,), rank=3, device="cpu")
    assert torch.equal(batch.q_state_rad, batch.q_target_rad)
    assert batch.q_state_rad[0, 0].item() == pytest.approx(0.03)
    assert torch.equal(batch.object_quat_h_wxyz, torch.tensor(((1.0, 0.0, 0.0, 0.0),)))
    assert batch.record_digests == (entry.digest,)
    assert batch.lookup_digests == (entry.key.digest,)


def test_partial_reset_installs_preload_without_touching_other_rows() -> None:
    r"""乱序partial reset只更新$\{3,1\}$，并把action target恢复为$q_t$而不是$q_s$。"""

    sidecar = HeterogeneousPregraspState(num_envs=4, device="cpu")
    all_ids = torch.arange(4)
    sidecar.install(all_ids, _batch(4, state_value=0.1, target_value=0.12, digest_digit="1"))
    untouched_target_before = sidecar.q_target_rad[[0, 2]].clone()  # 非reset episodes基线
    partial_ids = torch.tensor((3, 1))  # 非连续且乱序，直接证伪pairwise advanced indexing
    partial = _batch(2, state_value=0.2, target_value=0.23, digest_digit="2")
    sidecar.install(partial_ids, partial)
    assert torch.equal(sidecar.q_target_rad[[0, 2]], untouched_target_before)
    assert torch.allclose(sidecar.q_state_rad[partial_ids, :10], torch.full((2, 10), 0.2))
    assert torch.allclose(sidecar.q_target_rad[partial_ids, :10], torch.full((2, 10), 0.23))
    assert bool(((sidecar.q_target_rad - sidecar.q_state_rad)[partial_ids, :10].abs() > 0.0).all())

    # 六个action buffers先填不同sentinel，确保同步不会全量清空非reset rows。
    buffers = [torch.arange(64, dtype=torch.float32).reshape(4, 16) + offset for offset in range(6)]
    snapshots = [buffer[[0, 2]].clone() for buffer in buffers]
    mask = synchronize_action_reset(
        env_ids=partial_ids,
        sidecar=sidecar,
        joint_ids=slice(None),
        raw_actions=buffers[0],
        processed_actions=buffers[1],
        executed_actions=buffers[2],
        current_targets=buffers[3],
        previous_targets=buffers[4],
        pregrasp_targets=buffers[5],
    )
    for buffer, snapshot in zip(buffers, snapshots):
        assert torch.equal(buffer[[0, 2]], snapshot)  # non-reset rows逐位保持
    assert torch.equal(buffers[0][partial_ids], torch.zeros(2, 16))
    assert torch.equal(buffers[1][partial_ids], torch.zeros(2, 16))
    assert torch.equal(buffers[2][partial_ids], torch.zeros(2, 16))
    for target_buffer in buffers[3:]:
        assert torch.equal(target_buffer[partial_ids], sidecar.q_target_rad[partial_ids])
    assert torch.equal(mask, sidecar.active_joint_mask[partial_ids])


def test_unresolved_sidecar_row_fails_before_action_buffer_mutation() -> None:
    r"""未命中cache的row不能获得q-home或actual-q fallback。"""

    sidecar = HeterogeneousPregraspState(num_envs=2, device="cpu")
    buffers = [torch.ones(2, 16) for _ in range(6)]
    before = [buffer.clone() for buffer in buffers]
    with pytest.raises(RuntimeError, match="unresolved"):
        synchronize_action_reset(
            env_ids=torch.tensor((1,)),
            sidecar=sidecar,
            joint_ids=slice(None),
            raw_actions=buffers[0],
            processed_actions=buffers[1],
            executed_actions=buffers[2],
            current_targets=buffers[3],
            previous_targets=buffers[4],
            pregrasp_targets=buffers[5],
        )
    assert all(torch.equal(buffer, snapshot) for buffer, snapshot in zip(buffers, before))


def test_policy_delta_is_accumulated_once_and_ghost_is_zero() -> None:
    r"""一次$1/24$ rad update后，重复physics apply不需要也不能再次推进target。"""

    previous = torch.full((2, 16), 0.2)
    delta = torch.full((2, 16), 1.0 / 24.0)
    lower = torch.full((2, 16), -1.0)
    upper = torch.full((2, 16), 1.0)
    active = torch.zeros(2, 16, dtype=torch.bool)
    active[:, :10] = True
    updated = compute_policy_step_masked_relative_target(previous, delta, lower, upper, active)
    assert torch.allclose(updated[:, :10], torch.full((2, 10), 0.2 + 1.0 / 24.0))
    assert torch.equal(updated[:, 10:], torch.zeros(2, 6))
    held_targets = [updated.clone() for _ in range(6)]  # apply_actions六次只读取，不调用transition
    assert all(torch.equal(target, updated) for target in held_targets)
