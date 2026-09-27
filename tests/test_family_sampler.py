from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import anymani.distill.il.train_family as training


def _compact_batch(
    family: str, asset_ids: tuple[str, ...], *, value_offset: float = 0.0
) -> training.CompactFamilyBatch:
    pair_count = len(asset_ids)
    values = value_offset + np.arange(pair_count, dtype=np.float32)
    target = np.zeros((pair_count, 16), dtype=np.float32)
    target[:, 0] = values
    geometry_tokens = np.zeros((pair_count, 21, 128), dtype=np.float32)
    geometry_tokens[:, 0, 0] = values
    static_jnt_valid = np.ones((pair_count, 16), dtype=bool)
    static_tip_valid = np.ones((pair_count, 4), dtype=bool)
    return training.CompactFamilyBatch(
        initial_history=np.zeros((pair_count, 30, 16, 5), dtype=np.float32),
        frames_current=np.zeros((1, pair_count, 16, 5), dtype=np.float32),
        frames_contact=np.zeros((1, pair_count, 21, 1), dtype=np.float32),
        sample_steps=np.asarray([0], dtype=np.int64),
        sample_row=np.zeros(pair_count, dtype=np.int64),
        env_index=np.arange(pair_count, dtype=np.int64),
        target=target,
        behavior_action=np.zeros((pair_count, 16), dtype=np.float32),
        geometry_tokens=geometry_tokens,
        fk_target=np.zeros((pair_count, 16, 3), dtype=np.float32),
        family_ids=(family,) * pair_count,
        asset_ids=asset_ids,
        static_limits=np.zeros((pair_count, 16, 2), dtype=np.float32),
        static_jnt_valid=static_jnt_valid,
        static_tip_valid=static_tip_valid,
        static_owner_valid=np.ones((pair_count, 21), dtype=bool),
        static_joint_kinematics=np.zeros((pair_count, 16, 15), dtype=np.float32),
        static_shortest_path=np.zeros((pair_count, 21, 21), dtype=np.int16),
        static_parent_direction=np.zeros((pair_count, 21, 21), dtype=np.int16),
        static_child_direction=np.zeros((pair_count, 21, 21), dtype=np.int16),
    )


def _training_and_validation_views() -> tuple[
    tuple[training.CompactFamilyBatch, training.CompactFamilyBatch],
    tuple[training.CompactFamilyBatch, training.CompactFamilyBatch],
]:
    leap = _compact_batch("leap", ("l1", "l1", "l2", "l2", "leap_validation_only", "leap_validation_only"))
    allegro = _compact_batch("allegro", ("a1", "a1", "a2", "a2", "a2", "allegro_validation_only"), value_offset=100.0)
    train = (leap.subset(np.asarray([3, 0, 1])), allegro.subset(np.asarray([4, 0, 2, 1, 3])))
    validation = (leap.subset(np.asarray([4, 5])), allegro.subset(np.asarray([5])))
    return train, validation


def test_compact_view_projects_reordered_repeated_labels_and_targets() -> None:
    source = _compact_batch("leap", ("l0", "l1", "l2", "l3", "l4", "l5"))
    view = source.subset(np.asarray([5, 2, 5, 0]))

    family_ids, asset_ids = training._view_labels(view)
    assert family_ids == ("leap", "leap", "leap", "leap")
    assert asset_ids == ("l5", "l2", "l5", "l0")

    batch = view.materialize(torch.tensor([3, 1, 0, 2]))
    assert batch.asset_ids == ("l0", "l2", "l5", "l5")
    assert batch.target[:, 0].tolist() == [0.0, 2.0, 5.0, 5.0]
    assert batch.geometry.tokens[:, 0, 0].tolist() == [0.0, 2.0, 5.0, 5.0]
    with pytest.raises(ValueError, match=r"must lie in \[0,4\)"):
        view.materialize(torch.tensor([-1]))
    with pytest.raises(TypeError, match="integer dtype"):
        view.subset(np.asarray([0.5]))


def test_train_weights_use_only_projected_train_axis_and_balance_hands() -> None:
    train, validation = _training_and_validation_views()
    family_ids = tuple(label for view in train for label in training._view_labels(view)[0])
    asset_ids = tuple(label for view in train for label in training._view_labels(view)[1])
    weights = training.family_asset_weights(family_ids, asset_ids, dtype=torch.float64)

    assert len(weights) == sum(view.sample_count for view in train) == 8
    assert not set(asset_ids).intersection({"leap_validation_only", "allegro_validation_only"})
    assert training._view_asset_counts(train) == Counter({"a2": 3, "l1": 2, "a1": 2, "l2": 1})
    assert training._view_asset_counts(validation) == Counter({"leap_validation_only": 2, "allegro_validation_only": 1})

    by_family = Counter()
    by_asset = Counter()
    for weight, family, asset in zip(weights.tolist(), family_ids, asset_ids, strict=True):
        by_family[family] += weight
        by_asset[family, asset] += weight
    assert by_family == pytest.approx({"leap": 0.5, "allegro": 0.5})
    assert by_asset == pytest.approx(
        {("leap", "l1"): 0.25, ("leap", "l2"): 0.25, ("allegro", "a1"): 0.25, ("allegro", "a2"): 0.25}
    )


def test_global_materialization_keeps_source_grouping_and_rejects_out_of_domain_indices() -> None:
    train, _ = _training_and_validation_views()
    indices = torch.tensor([3, 0, 7, 1, 3, 6], dtype=torch.long)
    batch = training._materialize_global_indices(train, indices)

    assert batch.sample_count == indices.numel()
    assert batch.family_ids == ("leap", "leap", "allegro", "allegro", "allegro", "allegro")
    assert batch.asset_ids == ("l2", "l1", "a2", "a2", "a2", "a1")
    assert batch.target[:, 0].tolist() == [3.0, 0.0, 104.0, 103.0, 104.0, 101.0]
    assert batch.geometry.tokens[:, 0, 0].tolist() == batch.target[:, 0].tolist()

    with pytest.raises(ValueError, match=r"must lie in \[0,8\)"):
        training._materialize_global_indices(train, torch.tensor([0, 8]))


def test_training_updates_materialize_full_batches_and_count_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    train, _ = _training_and_validation_views()
    batch_sizes: list[int] = []

    class TinyStudent(torch.nn.Module):
        variant = "n040"
        family_actor_config = {"variant": "n040"}

        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.1))
            self.global_log_std = torch.nn.Parameter(torch.zeros(16), requires_grad=False)

    def fake_losses(actor: TinyStudent, batch: training.FamilySampleBatch, *, lambda_fk: float):
        del lambda_fk
        batch_sizes.append(batch.sample_count)
        loss = (actor.scale * batch.target[:, 0] / 100.0 - 1.0).square().mean()
        return loss, loss, loss.new_zeros(())

    monkeypatch.setattr(training, "_forward_losses", fake_losses)
    result = training.fit_family_student(
        TinyStudent(),
        train,
        max_updates=2,
        batch_size=16,
        seed=42,
        start_processed_samples=7,
    )

    assert batch_sizes == [16, 16]
    assert result["processed_samples"] == 39


def test_corrected_sampler_identity_rejects_legacy_checkpoint_protocol() -> None:
    bundle = SimpleNamespace(dataset_sha256="dataset", n040_sha256="encoder")
    actor = SimpleNamespace(family_actor_config={"variant": "n040"})
    identity = training._run_identity(
        bundle,
        actor,
        representation="n040",
        seed=42,
        batch_size=2048,
        learning_rate=3e-4,
        max_epochs=50,
        max_seconds=7200.0,
        tf32=False,
    )
    assert identity["sampling_version"] == training.TRAINING_SAMPLING_VERSION
    training._require_sampling_version(
        {"protocol": {"sampling_version": training.TRAINING_SAMPLING_VERSION}},
        checkpoint=Path("new.pt"),
    )
    with pytest.raises(ValueError, match="sampling_version"):
        training._require_sampling_version({"protocol": {"batch_size": 2048}}, checkpoint=Path("legacy.pt"))
