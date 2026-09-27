"""Train BC from frozen teacher means with balanced sampling over training views."""

from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import random
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast
import h5py
import numpy as np
import torch
from anymani.distill.il.family_student import (
    FAMILY_STUDENT_ACTOR_ABI,
    FAMILY_STUDENT_ARTIFACT_TYPE,
    FAMILY_STUDENT_SCHEMA_VERSION,
    build_family_student,
    family_student_source_code_files,
    family_student_source_code_hash,
    load_family_student,
    save_family_student_checkpoint,
)
from anymani.distill.models.family_rotation_policy import (
    JOINT_KINEMATICS_WIDTH,
    JOINT_ORIGIN_WIDTH,
    LINK_LENGTH_M,
    FamilyRotationActorOutput,
    FamilyRotationStudentActor,
)
from anymani.distill.models.palm_rotation_policy import PalmRotationActorObservation, PalmRotationGeometry

Representation = Literal["n040", "no_z", "fk"]
QUALITY_DURATION_S = 30.0
QUALITY_NET_TURNS = 0.5
QUALITY_NET_PATH_RATIO = 0.7
VALIDATION_REPLICA_MODULUS = 4
VALIDATION_REPLICA_REMAINDER = 3
DEFAULT_BATCH_SIZE = 2048
DEFAULT_LEARNING_RATE = 0.0003
DEFAULT_MAX_EPOCHS = 50
DEFAULT_MAX_SECONDS = 7200.0
DEFAULT_MAX_RAM_GIB = 16.0
SAMPLE_READ_CHUNK_ROWS = 4
TRAINING_SAMPLING_VERSION = "compact_view_projected_v1"
METRICS_FILENAME = "metrics.jsonl"
TRAINING_REPORT_FILENAME = "training-report.json"
TRAINING_REPORT_ARTIFACT_TYPE = "anymani.family_distilled_actor_training_report"
TRAINING_REPORT_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class FamilySampleBatch:
    """A minibatch of joint histories, geometry tokens and teacher action targets."""

    observation: PalmRotationActorObservation
    geometry: PalmRotationGeometry
    joint_kinematics: torch.Tensor
    target: torch.Tensor
    family_ids: tuple[str, ...]
    asset_ids: tuple[str, ...]
    joint_valid: torch.Tensor
    fk_target: torch.Tensor | None = None
    behavior_action: torch.Tensor | None = None
    step_index: torch.Tensor | None = None

    def __post_init__(self) -> None:
        count = self.target.shape[0]
        expected = {
            "target": (count, 16),
            "joint_kinematics": (count, 16, JOINT_KINEMATICS_WIDTH),
            "joint_valid": (count, 16),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(
                    f"family sample {name} must have shape {shape}, got {tuple(getattr(self, name).shape)}"
                )
        if self.fk_target is not None and tuple(self.fk_target.shape) != (count, 16, JOINT_ORIGIN_WIDTH):
            raise ValueError("family sample fk_target must have shape [B,16,3]")
        if self.behavior_action is not None and tuple(self.behavior_action.shape) != (count, 16):
            raise ValueError("family sample behavior_action must have shape [B,16]")
        if len(self.family_ids) != count or len(self.asset_ids) != count:
            raise ValueError("family/asset labels must align with sample axis")
        if self.joint_valid.dtype != torch.bool:
            raise ValueError("family sample joint_valid must be boolean")
        if not torch.equal(self.joint_valid, self.observation.jnt_valid):
            raise ValueError("family sample joint_valid disagrees with actor observation jnt_valid")
        if not bool(self.joint_valid.any(dim=-1).all().item()):
            raise ValueError("family sample contains a sample with zero valid joints")
        tensors = [
            self.observation.jnt_current,
            self.geometry.tokens,
            self.joint_kinematics,
            self.target,
            self.joint_valid,
        ]
        if self.fk_target is not None:
            tensors.append(self.fk_target)
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("family sample tensors must share one device")
        for name, tensor in (
            ("joint_kinematics", self.joint_kinematics),
            ("target", self.target),
            ("geometry.tokens", self.geometry.tokens),
        ):
            if not bool(torch.isfinite(tensor).all().item()):
                raise ValueError(f"family sample {name} contains non-finite values")
        if self.fk_target is not None and (not bool(torch.isfinite(self.fk_target).all().item())):
            raise ValueError("family sample fk_target contains non-finite values")

    @property
    def sample_count(self) -> int:
        return int(self.target.shape[0])


def _compact_view_indices(index: np.ndarray | torch.Tensor | slice, sample_count: int) -> np.ndarray:
    if isinstance(index, slice):
        selected = np.arange(sample_count, dtype=np.int64)[index]
    elif isinstance(index, torch.Tensor):
        if index.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("compact view indices must use an integer dtype")
        selected = index.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
    else:
        raw = np.asarray(index)
        if raw.dtype.kind not in "iu":
            raise TypeError("compact view indices must use an integer dtype")
        selected = raw.astype(np.int64, copy=False).reshape(-1)
    if selected.size and (np.any(selected < 0) or np.any(selected >= sample_count)):
        raise ValueError(f"compact view index must lie in [0,{sample_count})")
    return np.asarray(selected, dtype=np.int64)


@dataclass(frozen=True)
class CompactFamilyBatch:
    """Store trajectories once and materialize only the requested history minibatch."""

    initial_history: np.ndarray
    frames_current: np.ndarray
    frames_contact: np.ndarray
    sample_steps: np.ndarray
    sample_row: np.ndarray
    env_index: np.ndarray
    target: np.ndarray
    behavior_action: np.ndarray
    geometry_tokens: np.ndarray
    fk_target: np.ndarray
    family_ids: tuple[str, ...]
    asset_ids: tuple[str, ...]
    static_limits: np.ndarray
    static_jnt_valid: np.ndarray
    static_tip_valid: np.ndarray
    static_owner_valid: np.ndarray
    static_joint_kinematics: np.ndarray
    static_shortest_path: np.ndarray
    static_parent_direction: np.ndarray
    static_child_direction: np.ndarray
    pair_index: np.ndarray | None = None

    def __post_init__(self) -> None:
        quality_env = self.initial_history.shape[0]
        underlying_count = self.target.shape[0]
        if self.initial_history.shape != (quality_env, 30, 16, 5):
            raise ValueError("compact initial_history must have shape [Q,30,16,5]")
        if self.frames_current.ndim != 4 or self.frames_current.shape[1:] != (quality_env, 16, 5):
            raise ValueError("compact frames_current must have shape [T,Q,16,5]")
        if self.frames_contact.shape != (self.frames_current.shape[0], quality_env, 21, 1):
            raise ValueError("compact frames_contact must align with frames_current")
        expected = {
            "sample_row": (underlying_count,),
            "env_index": (underlying_count,),
            "target": (underlying_count, 16),
            "behavior_action": (underlying_count, 16),
            "geometry_tokens": (underlying_count, 21, 128),
            "fk_target": (underlying_count, 16, 3),
            "static_limits": (quality_env, 16, 2),
            "static_jnt_valid": (quality_env, 16),
            "static_tip_valid": (quality_env, 4),
            "static_owner_valid": (quality_env, 21),
            "static_joint_kinematics": (quality_env, 16, 15),
            "static_shortest_path": (quality_env, 21, 21),
            "static_parent_direction": (quality_env, 21, 21),
            "static_child_direction": (quality_env, 21, 21),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"compact {name} shape {getattr(self, name).shape} != {shape}")
        if (
            self.sample_steps.ndim != 1
            or len(self.family_ids) != underlying_count
            or len(self.asset_ids) != underlying_count
        ):
            raise ValueError("compact sample axis and labels disagree")
        if underlying_count and (np.any(self.sample_row < 0) or np.any(self.sample_row >= len(self.sample_steps))):
            raise ValueError("compact sample_row contains an invalid sample axis index")
        if underlying_count and (np.any(self.env_index < 0) or np.any(self.env_index >= quality_env)):
            raise ValueError("compact env_index contains an invalid quality-env index")
        if self.pair_index is not None:
            if self.pair_index.ndim != 1 or np.any(self.pair_index < 0) or np.any(self.pair_index >= underlying_count):
                raise ValueError("compact pair_index is not a valid underlying pair view")
        for name, value in (
            ("initial_history", self.initial_history),
            ("frames_current", self.frames_current),
            ("frames_contact", self.frames_contact),
            ("target", self.target),
            ("geometry_tokens", self.geometry_tokens),
            ("fk_target", self.fk_target),
            ("static_limits", self.static_limits),
            ("static_joint_kinematics", self.static_joint_kinematics),
        ):
            if not bool(np.isfinite(value).all()):
                raise ValueError(f"compact {name} contains non-finite values")

    @property
    def sample_count(self) -> int:
        return int(self.target.shape[0] if self.pair_index is None else self.pair_index.shape[0])

    def subset(self, index: np.ndarray | torch.Tensor | slice) -> CompactFamilyBatch:
        selected = _compact_view_indices(index, self.sample_count)
        base = selected if self.pair_index is None else self.pair_index[selected]
        return replace(self, pair_index=np.asarray(base, dtype=np.int64))

    def materialize(self, index: np.ndarray | torch.Tensor | slice) -> FamilySampleBatch:
        selected = _compact_view_indices(index, self.sample_count)
        base = selected if self.pair_index is None else self.pair_index[selected]
        rows = self.sample_row[base]
        environments = self.env_index[base]
        steps = self.sample_steps[rows]
        history = reconstruct_history(self.initial_history, self.frames_current, steps, environments)
        current = np.asarray(self.frames_current[steps, environments], dtype=np.float32)
        contact = np.asarray(self.frames_contact[steps, environments], dtype=np.float32)
        jnt_valid = np.asarray(self.static_jnt_valid[environments], dtype=bool)
        tip_valid = np.asarray(self.static_tip_valid[environments], dtype=bool)
        owner_valid = np.asarray(self.static_owner_valid[environments], dtype=bool)
        observation = PalmRotationActorObservation(
            jnt_current=torch.from_numpy(current),
            jnt_history=torch.from_numpy(np.asarray(history, dtype=np.float32)),
            jnt_limits=torch.from_numpy(np.asarray(self.static_limits[environments], dtype=np.float32)),
            owner_contact=torch.from_numpy(contact),
            jnt_valid=torch.from_numpy(jnt_valid),
            tip_valid=torch.from_numpy(tip_valid),
            owner_valid=torch.from_numpy(owner_valid),
        )
        geometry = PalmRotationGeometry(
            tokens=torch.from_numpy(np.asarray(self.geometry_tokens[base], dtype=np.float32)),
            owner_valid=torch.from_numpy(owner_valid),
            shortest_path=torch.from_numpy(np.asarray(self.static_shortest_path[environments], dtype=np.int64)),
            parent_direction=torch.from_numpy(np.asarray(self.static_parent_direction[environments], dtype=np.int64)),
            child_direction=torch.from_numpy(np.asarray(self.static_child_direction[environments], dtype=np.int64)),
        )
        return FamilySampleBatch(
            observation=observation,
            geometry=geometry,
            joint_kinematics=torch.from_numpy(np.asarray(self.static_joint_kinematics[environments], dtype=np.float32)),
            target=torch.from_numpy(np.asarray(self.target[base], dtype=np.float32)),
            family_ids=tuple((self.family_ids[item] for item in base.tolist())),
            asset_ids=tuple((self.asset_ids[item] for item in base.tolist())),
            joint_valid=torch.from_numpy(jnt_valid),
            fk_target=torch.from_numpy(np.asarray(self.fk_target[base], dtype=np.float32)),
            behavior_action=torch.from_numpy(np.asarray(self.behavior_action[base], dtype=np.float32)),
            step_index=torch.from_numpy(np.asarray(steps, dtype=np.int64)),
        )


FamilyView = FamilySampleBatch | CompactFamilyBatch


@dataclass(frozen=True)
class FamilySource:
    """One demonstration source with its immutable identity and trajectory splits."""

    path: Path
    metadata: dict[str, object]
    dataset_sha256: str
    source_sha256: str
    collection_identity: str
    family: str
    n040_sha256: str
    training: tuple[CompactFamilyBatch, ...]
    validation: tuple[CompactFamilyBatch, ...]
    report: dict[str, object]


@dataclass(frozen=True)
class FamilyDatasetBundle:
    """Combine family sources while preserving their identities and split boundaries."""

    sources: tuple[FamilySource, ...]
    training: tuple[CompactFamilyBatch, ...]
    validation: tuple[CompactFamilyBatch, ...]
    dataset_sha256: str
    n040_sha256: str
    reports: tuple[dict[str, object], ...]


def _labels(value: object, count: int, *, name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) * count
    if isinstance(value, torch.Tensor):
        raw = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, np.ndarray):
        raw = value.reshape(-1).tolist()
    elif isinstance(value, Sequence):
        raw = list(value)
    else:
        return (str(value),) * count
    if len(raw) != count:
        raise ValueError(f"{name} labels must contain {count} entries, got {len(raw)}")
    return tuple((str(item) for item in raw))


def _per_sample_action_mse(prediction: torch.Tensor, target: torch.Tensor, joint_valid: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[-1] != 16:
        raise ValueError(f"action prediction/target must both have shape [B,16], got {prediction.shape}/{target.shape}")
    if joint_valid.shape != target.shape:
        raise ValueError(f"joint_valid must have shape {tuple(target.shape)}, got {tuple(joint_valid.shape)}")
    if joint_valid.dtype != torch.bool:
        joint_valid = joint_valid != 0
    if not bool(joint_valid.any(dim=-1).all().item()):
        raise ValueError("action MSE requires at least one valid joint per sample")
    squared = (prediction - target).square()
    return (squared * joint_valid.to(dtype=squared.dtype)).sum(dim=-1) / joint_valid.sum(dim=-1).clamp_min(1)


def family_asset_weights(
    family_ids: Sequence[object],
    asset_ids: Sequence[object],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Give families equal weight, then hands within each family, then samples within each hand."""
    if len(family_ids) != len(asset_ids) or not family_ids:
        raise ValueError("family and asset labels must be non-empty and have equal length")
    families = tuple((str(item) for item in family_ids))
    assets = tuple((str(item) for item in asset_ids))
    family_set = tuple(dict.fromkeys(families))
    asset_sets: dict[str, set[str]] = {family: set() for family in family_set}
    counts: Counter[tuple[str, str]] = Counter(zip(families, assets))
    for family, asset in zip(families, assets):
        asset_sets[family].add(asset)
    weights = torch.empty(len(families), dtype=dtype, device=device)
    for index, (family, asset) in enumerate(zip(families, assets)):
        weights[index] = 1.0 / (len(family_set) * len(asset_sets[family]) * counts[family, asset])
    return weights / weights.sum().clamp_min(torch.finfo(dtype).eps)


def balanced_family_action_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    joint_valid: torch.Tensor,
    family_ids: Sequence[object] | None = None,
    asset_ids: Sequence[object] | None = None,
) -> torch.Tensor:
    """Average valid-joint action error with equal hand and family weights."""
    per_sample = _per_sample_action_mse(prediction, target, joint_valid)
    if family_ids is None and asset_ids is None:
        return per_sample.mean()
    if family_ids is None or asset_ids is None:
        raise ValueError("family_ids and asset_ids must be provided together")
    if len(family_ids) != per_sample.shape[0] or len(asset_ids) != per_sample.shape[0]:
        raise ValueError("family/asset labels must align with prediction batch")
    weights = family_asset_weights(family_ids, asset_ids, device=per_sample.device, dtype=per_sample.dtype)
    return (per_sample * weights).sum()


def _per_sample_fk_mse(
    prediction: torch.Tensor, target_m: torch.Tensor, joint_valid: torch.Tensor, *, link_length_m: float = LINK_LENGTH_M
) -> torch.Tensor:
    expected = (*target_m.shape[:-1], JOINT_ORIGIN_WIDTH)
    if prediction.shape != expected or target_m.ndim != 3 or target_m.shape[1] != 16:
        raise ValueError(f"FK prediction/target must have shape [B,16,3], got {prediction.shape}/{target_m.shape}")
    if joint_valid.shape != target_m.shape[:2]:
        raise ValueError("FK joint_valid shape disagrees with target")
    normalized_target = target_m / link_length_m
    per_joint = (prediction - normalized_target).square().mean(dim=-1)
    return (per_joint * joint_valid.to(dtype=per_joint.dtype)).sum(dim=-1) / joint_valid.sum(dim=-1).clamp_min(1)


def _coerce_sample_batch(raw: FamilySampleBatch | tuple[Any, ...]) -> FamilySampleBatch:
    if isinstance(raw, FamilySampleBatch):
        return raw
    if not isinstance(raw, tuple) or len(raw) != 8:
        raise TypeError("family student batches must be FamilySampleBatch or an 8-field tuple")
    observation, geometry, target, family, asset, joint_valid = raw[:6]
    joint_kinematics = raw[6]
    fk_target = raw[7]
    if not isinstance(observation, PalmRotationActorObservation) or not isinstance(geometry, PalmRotationGeometry):
        raise TypeError("family student tuple must begin with actor observation and geometry")
    target_tensor = cast(torch.Tensor, target)
    valid_tensor = cast(torch.Tensor, joint_valid).bool()
    count = target_tensor.shape[0]
    return FamilySampleBatch(
        observation=observation,
        geometry=geometry,
        joint_kinematics=cast(torch.Tensor, joint_kinematics).float(),
        target=target_tensor.float(),
        family_ids=_labels(family, count, name="family"),
        asset_ids=_labels(asset, count, name="asset"),
        joint_valid=valid_tensor,
        fk_target=None if fk_target is None else cast(torch.Tensor, fk_target).float(),
    )


def _coerce_view(raw: FamilyView | tuple[Any, ...]) -> FamilyView:
    if isinstance(raw, CompactFamilyBatch):
        return raw
    return _coerce_sample_batch(raw)


def _view_labels(view: FamilyView) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not isinstance(view, CompactFamilyBatch) or view.pair_index is None:
        return (view.family_ids, view.asset_ids)
    selected = view.pair_index.tolist()
    return (
        tuple(view.family_ids[index] for index in selected),
        tuple(view.asset_ids[index] for index in selected),
    )


def _materialize_global_indices(views: Sequence[FamilyView], index: torch.Tensor) -> FamilySampleBatch:
    if not isinstance(index, torch.Tensor) or index.ndim != 1:
        raise TypeError("balanced sample indices must be a one-dimensional torch tensor")
    if index.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("balanced sample indices must use an integer dtype")
    index = index.detach().to(device="cpu", dtype=torch.long)
    if index.numel() == 0:
        raise ValueError("balanced sample index must not be empty")
    pieces: list[FamilySampleBatch] = []
    offsets: list[tuple[int, int, FamilyView]] = []
    cursor = 0
    for view in views:
        stop = cursor + view.sample_count
        offsets.append((cursor, stop, view))
        cursor = stop
    if cursor < 1:
        raise ValueError("balanced sample views contain no training samples")
    if bool((index < 0).any().item()) or bool((index >= cursor).any().item()):
        raise ValueError(f"balanced sample index must lie in [0,{cursor})")
    for start, stop, view in offsets:
        local_mask = (index >= start) & (index < stop)
        if not bool(local_mask.any().item()):
            continue
        local = index[local_mask] - start
        if isinstance(view, CompactFamilyBatch):
            pieces.append(view.materialize(local))
        else:
            pieces.append(_slice_batch(view, local.to(device=view.target.device)))
    if not pieces:
        raise ValueError("balanced sample index selected no family view")
    batch = _concat_batches(pieces)
    if batch.sample_count != index.numel():
        raise RuntimeError(f"balanced sampler requested {index.numel()} rows but materialized {batch.sample_count}")
    # Keep the source-grouped batch order used by the original training loop.
    return batch


def _view_asset_counts(views: Sequence[FamilyView]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for view in views:
        counts.update(_view_labels(view)[1])
    return counts


def _slice_batch(batch: FamilySampleBatch, index: torch.Tensor | slice) -> FamilySampleBatch:
    selected = (
        torch.arange(batch.sample_count, device=batch.target.device)[index] if isinstance(index, slice) else index
    )
    labels = selected.detach().cpu().tolist()
    return FamilySampleBatch(
        observation=PalmRotationActorObservation(
            jnt_current=batch.observation.jnt_current[index],
            jnt_history=batch.observation.jnt_history[index],
            jnt_limits=batch.observation.jnt_limits[index],
            owner_contact=batch.observation.owner_contact[index],
            jnt_valid=batch.observation.jnt_valid[index],
            tip_valid=batch.observation.tip_valid[index],
            owner_valid=batch.observation.owner_valid[index],
        ),
        geometry=PalmRotationGeometry(
            tokens=batch.geometry.tokens[index],
            owner_valid=batch.geometry.owner_valid[index],
            shortest_path=batch.geometry.shortest_path[index],
            parent_direction=batch.geometry.parent_direction[index],
            child_direction=batch.geometry.child_direction[index],
        ),
        joint_kinematics=batch.joint_kinematics[index],
        target=batch.target[index],
        family_ids=tuple((batch.family_ids[item] for item in labels)),
        asset_ids=tuple((batch.asset_ids[item] for item in labels)),
        joint_valid=batch.joint_valid[index],
        fk_target=None if batch.fk_target is None else batch.fk_target[index],
        behavior_action=None if batch.behavior_action is None else batch.behavior_action[index],
        step_index=None if batch.step_index is None else batch.step_index[index],
    )


def _concat_batches(batches: Sequence[FamilySampleBatch]) -> FamilySampleBatch:
    if not batches:
        raise ValueError("cannot concatenate empty family batches")
    fk = (
        None
        if all((batch.fk_target is None for batch in batches))
        else torch.cat([cast(torch.Tensor, batch.fk_target) for batch in batches], dim=0)
    )
    behavior = (
        None
        if all((batch.behavior_action is None for batch in batches))
        else torch.cat([cast(torch.Tensor, batch.behavior_action) for batch in batches], dim=0)
    )
    step = (
        None
        if all((batch.step_index is None for batch in batches))
        else torch.cat([cast(torch.Tensor, batch.step_index) for batch in batches], dim=0)
    )
    observations = [batch.observation for batch in batches]
    geometries = [batch.geometry for batch in batches]
    return FamilySampleBatch(
        observation=PalmRotationActorObservation(
            jnt_current=torch.cat([item.jnt_current for item in observations]),
            jnt_history=torch.cat([item.jnt_history for item in observations]),
            jnt_limits=torch.cat([item.jnt_limits for item in observations]),
            owner_contact=torch.cat([item.owner_contact for item in observations]),
            jnt_valid=torch.cat([item.jnt_valid for item in observations]),
            tip_valid=torch.cat([item.tip_valid for item in observations]),
            owner_valid=torch.cat([item.owner_valid for item in observations]),
        ),
        geometry=PalmRotationGeometry(
            tokens=torch.cat([item.tokens for item in geometries]),
            owner_valid=torch.cat([item.owner_valid for item in geometries]),
            shortest_path=torch.cat([item.shortest_path for item in geometries]),
            parent_direction=torch.cat([item.parent_direction for item in geometries]),
            child_direction=torch.cat([item.child_direction for item in geometries]),
        ),
        joint_kinematics=torch.cat([batch.joint_kinematics for batch in batches]),
        target=torch.cat([batch.target for batch in batches]),
        family_ids=tuple((label for batch in batches for label in batch.family_ids)),
        asset_ids=tuple((label for batch in batches for label in batch.asset_ids)),
        joint_valid=torch.cat([batch.joint_valid for batch in batches]),
        fk_target=fk,
        behavior_action=behavior,
        step_index=step,
    )


def _move_batch(batch: FamilySampleBatch, device: torch.device) -> FamilySampleBatch:
    if batch.target.device == device:
        return batch
    return FamilySampleBatch(
        observation=PalmRotationActorObservation(
            jnt_current=batch.observation.jnt_current.to(device),
            jnt_history=batch.observation.jnt_history.to(device),
            jnt_limits=batch.observation.jnt_limits.to(device),
            owner_contact=batch.observation.owner_contact.to(device),
            jnt_valid=batch.observation.jnt_valid.to(device),
            tip_valid=batch.observation.tip_valid.to(device),
            owner_valid=batch.observation.owner_valid.to(device),
        ),
        geometry=PalmRotationGeometry(
            tokens=batch.geometry.tokens.to(device),
            owner_valid=batch.geometry.owner_valid.to(device),
            shortest_path=batch.geometry.shortest_path.to(device),
            parent_direction=batch.geometry.parent_direction.to(device),
            child_direction=batch.geometry.child_direction.to(device),
        ),
        joint_kinematics=batch.joint_kinematics.to(device),
        target=batch.target.to(device),
        family_ids=batch.family_ids,
        asset_ids=batch.asset_ids,
        joint_valid=batch.joint_valid.to(device),
        fk_target=None if batch.fk_target is None else batch.fk_target.to(device),
        behavior_action=None if batch.behavior_action is None else batch.behavior_action.to(device),
        step_index=None if batch.step_index is None else batch.step_index.to(device),
    )


def _forward_losses(
    actor: FamilyRotationStudentActor, batch: FamilySampleBatch, *, lambda_fk: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output = actor(batch.observation, batch.geometry, joint_kinematics=batch.joint_kinematics)
    if not isinstance(output, FamilyRotationActorOutput):
        raise TypeError("family student actor returned an unexpected output type")
    bc = _per_sample_action_mse(output.mean, batch.target, batch.joint_valid).mean()
    fk = output.mean.new_zeros(())
    if lambda_fk > 0.0:
        if output.fk_prediction is None or batch.fk_target is None:
            raise ValueError("FK loss requires FK actor output and joint_origin_fk targets")
        fk = _per_sample_fk_mse(output.fk_prediction, batch.fk_target, batch.joint_valid).mean()
    return (bc + lambda_fk * fk, bc, fk)


def fit_family_student(
    actor: FamilyRotationStudentActor,
    batches: Iterable[FamilyView | tuple[Any, ...]],
    *,
    max_updates: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    lambda_fk: float | None = None,
    seed: int = 42,
    max_seconds: float | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    sampling_generator: torch.Generator | None = None,
    sampling_generator_state: torch.Tensor | None = None,
    start_update: int = 0,
    start_processed_samples: int = 0,
) -> dict[str, object]:
    """Optimize a student against frozen teacher means and optional FK supervision."""
    if max_updates < 1 or batch_size < 1 or learning_rate <= 0.0:
        raise ValueError("family student max_updates, batch_size and learning_rate must be positive")
    if max_seconds is not None and max_seconds <= 0.0:
        raise ValueError("family student max_seconds must be positive when provided")
    if lambda_fk is None:
        lambda_fk = 0.1 if actor.variant == "fk" else 0.0
    if lambda_fk < 0.0 or not math.isfinite(lambda_fk):
        raise ValueError("family student lambda_fk must be finite and non-negative")
    views = tuple((_coerce_view(batch) for batch in batches))
    if not views or sum((view.sample_count for view in views)) < 1:
        raise ValueError("family student training requires at least one sample")
    if actor.variant == "fk" and any(
        (isinstance(view, FamilySampleBatch) and view.fk_target is None for view in views)
    ):
        raise ValueError("FK student training requires samples/joint_origin_fk")
    if actor.variant != "fk" and lambda_fk != 0.0:
        raise ValueError("n040/no_z student must not receive FK loss")
    device = next(actor.parameters()).device
    families = tuple((label for view in views for label in _view_labels(view)[0]))
    assets = tuple((label for view in views for label in _view_labels(view)[1]))
    weights = family_asset_weights(families, assets, device=torch.device("cpu"))
    total_count = sum((view.sample_count for view in views))
    if weights.numel() != total_count:
        raise RuntimeError(f"sampler has {weights.numel()} weights for {total_count} training samples")
    generator = sampling_generator or torch.Generator(device="cpu")
    if sampling_generator_state is not None:
        generator.set_state(sampling_generator_state.cpu())
    elif sampling_generator is None:
        generator.manual_seed(seed)
    trainable = [parameter for parameter in actor.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("family student has no trainable actor parameters")
    if actor.global_log_std.requires_grad:
        raise ValueError("global_log_std must be frozen for mean supervision")
    optimizer_instance = optimizer or torch.optim.Adam(trainable, lr=learning_rate)
    if optimizer is not None and (not optimizer_instance.param_groups):
        raise ValueError("family student optimizer has no parameter groups")
    actor.train()
    sigma_before = actor.global_log_std.detach().clone()
    started = time.perf_counter()
    initial_loss: float | None = None
    last_loss = float("nan")
    last_bc = float("nan")
    last_fk = 0.0
    update_count = 0
    processed_samples = int(start_processed_samples)
    while update_count < max_updates:
        if max_seconds is not None and time.perf_counter() - started >= max_seconds:
            break
        indices = torch.multinomial(weights, batch_size, replacement=True, generator=generator)
        selected = _move_batch(_materialize_global_indices(views, indices), device)
        if selected.sample_count != batch_size:
            raise RuntimeError(f"training update requires {batch_size} rows, got {selected.sample_count}")
        optimizer_instance.zero_grad(set_to_none=True)
        total, bc, fk = _forward_losses(actor, selected, lambda_fk=lambda_fk)
        if not bool(torch.isfinite(total).item()):
            raise ValueError("family student loss became non-finite")
        if initial_loss is None:
            initial_loss = float(total.detach().item())
        total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        if not bool(torch.isfinite(torch.as_tensor(gradient_norm)).item()):
            raise ValueError("family student gradient norm became non-finite")
        optimizer_instance.step()
        update_count += 1
        processed_samples += selected.sample_count
        last_loss = float(total.detach().item())
        last_bc = float(bc.detach().item())
        last_fk = float(fk.detach().item())
    if not torch.equal(sigma_before, actor.global_log_std.detach()):
        raise RuntimeError("family student global_log_std changed during mean supervision")
    return {
        "updates": update_count,
        "total_updates": start_update + update_count,
        "processed_samples": processed_samples,
        "initial_loss": initial_loss,
        "last_loss": last_loss,
        "last_bc_loss": last_bc,
        "last_fk_loss": last_fk,
        "elapsed_seconds": time.perf_counter() - started,
        "sample_count": total_count,
        "optimizer": optimizer_instance,
        "optimizer_state_dict": optimizer_instance.state_dict(),
        "sampling_generator": generator,
        "sampling_generator_state": generator.get_state().clone(),
    }


def _metric_accumulator() -> dict[str, float]:
    return {"samples": 0.0, "action_mse_sum": 0.0, "zero_mse_sum": 0.0, "previous_mse_sum": 0.0, "fk_mse_sum": 0.0}


def _finish_metric(value: dict[str, float]) -> dict[str, float | int]:
    count = int(value["samples"])
    if count == 0:
        return {
            "samples": 0,
            "mean_action_mse": float("nan"),
            "zero_action_mse": float("nan"),
            "previous_action_mse": float("nan"),
        }
    result: dict[str, float | int] = {
        "samples": count,
        "mean_action_mse": value["action_mse_sum"] / count,
        "zero_action_mse": value["zero_mse_sum"] / count,
        "previous_action_mse": value["previous_mse_sum"] / count,
    }
    if value.get("has_fk", 0.0) > 0.0:
        result["fk_mse"] = value["fk_mse_sum"] / count
    return result


def balanced_mean_action_mse(
    per_asset: Mapping[str, Mapping[str, object]] | torch.Tensor | None = None,
    target: torch.Tensor | None = None,
    joint_valid: torch.Tensor | None = None,
    family_ids: Sequence[object] | None = None,
    asset_ids: Sequence[object] | None = None,
    *,
    prediction: torch.Tensor | None = None,
) -> float:
    """Compute the hand- and family-balanced validation action error."""
    if per_asset is not None and (not isinstance(per_asset, Mapping)):
        if prediction is not None:
            raise ValueError("balanced_mean_action_mse received prediction twice")
        prediction = cast(torch.Tensor, per_asset)
        per_asset = None
    if per_asset is None:
        if prediction is None or target is None or joint_valid is None or (family_ids is None) or (asset_ids is None):
            raise ValueError("balanced_mean_action_mse requires per_asset or complete tensor inputs")
        if len(family_ids) != len(asset_ids) or len(family_ids) != prediction.shape[0]:
            raise ValueError("balanced_mean_action_mse labels must align with prediction batch")
        values = _per_sample_action_mse(prediction, target, joint_valid).detach().cpu()
        groups: dict[tuple[str, str], list[float]] = defaultdict(list)
        for family, asset, value in zip(family_ids, asset_ids, values.tolist(), strict=True):
            groups[str(family), str(asset)].append(float(value))
        per_asset = {
            f"{family}/{asset}": {"mean_action_mse": float(np.mean(values_for_asset))}
            for (family, asset), values_for_asset in groups.items()
        }
    family_means: dict[str, list[float]] = defaultdict(list)
    for key, metrics in per_asset.items():
        family, _, _ = str(key).partition("/")
        if "mean_action_mse" in metrics:
            value = float(cast(Any, metrics["mean_action_mse"]))
        elif "action_mse_sum" in metrics and "samples" in metrics:
            value = float(cast(Any, metrics["action_mse_sum"])) / max(float(cast(Any, metrics["samples"])), 1.0)
        else:
            raise ValueError(f"per_asset metric {key!r} lacks mean_action_mse or sum/count")
        if math.isfinite(value):
            family_means[family].append(value)
    if not family_means:
        return float("nan")
    return float(np.mean([np.mean(values) for values in family_means.values()]))


def evaluate_family_student(
    actor: FamilyRotationStudentActor, batches: Iterable[FamilyView], *, batch_size: int = DEFAULT_BATCH_SIZE
) -> dict[str, object]:
    """Evaluate action prediction on held-out trajectories without changing weights."""
    if batch_size < 1:
        raise ValueError("family student validation batch_size must be positive")
    was_training = actor.training
    actor.eval()
    started = time.perf_counter()
    total = _metric_accumulator()
    by_family: dict[str, dict[str, float]] = defaultdict(_metric_accumulator)
    by_asset: dict[str, dict[str, float]] = defaultdict(_metric_accumulator)
    device = next(actor.parameters()).device
    with torch.no_grad():
        for source_view in batches:
            if source_view.sample_count == 0:
                continue
            for start in range(0, source_view.sample_count, batch_size):
                stop = min(start + batch_size, source_view.sample_count)
                source_index = torch.arange(start, stop, dtype=torch.long)
                if isinstance(source_view, CompactFamilyBatch):
                    chunk = source_view.materialize(source_index)
                else:
                    chunk = _slice_batch(source_view, source_index)
                chunk = _move_batch(chunk, device)
                output = actor(chunk.observation, chunk.geometry, joint_kinematics=chunk.joint_kinematics)
                prediction = output.mean
                target = chunk.target
                valid = chunk.joint_valid
                previous = chunk.observation.jnt_current[..., 2]
                action_mse = _per_sample_action_mse(prediction, target, valid)
                zero_mse = _per_sample_action_mse(torch.zeros_like(target), target, valid)
                previous_mse = _per_sample_action_mse(previous, target, valid)
                if not bool(torch.isfinite(action_mse).all().item()):
                    raise ValueError("family student validation prediction is non-finite")
                fk_mse = None
                if output.fk_prediction is not None and chunk.fk_target is not None:
                    fk_mse = _per_sample_fk_mse(output.fk_prediction, chunk.fk_target, valid)
                error_columns = [action_mse, zero_mse, previous_mse]
                if fk_mse is not None:
                    error_columns.append(fk_mse)
                error_cpu = torch.stack(error_columns, dim=1).detach().cpu().numpy()
                for row in range(chunk.sample_count):
                    family = chunk.family_ids[row]
                    asset = chunk.asset_ids[row]
                    asset_key = f"{family}/{asset}"
                    values = tuple((float(value) for value in error_cpu[row, :3]))
                    for accumulator in (total, by_family[family], by_asset[asset_key]):
                        accumulator["samples"] += 1.0
                        accumulator["action_mse_sum"] += values[0]
                        accumulator["zero_mse_sum"] += values[1]
                        accumulator["previous_mse_sum"] += values[2]
                        if fk_mse is not None:
                            accumulator["has_fk"] = 1.0
                            accumulator["fk_mse_sum"] += float(error_cpu[row, 3])
    actor.train(was_training)
    elapsed = time.perf_counter() - started
    finished = _finish_metric(total)
    per_asset = {key: _finish_metric(value) for key, value in sorted(by_asset.items())}
    result: dict[str, object] = {
        "validation_samples": finished["samples"],
        "mean_action_mse": finished["mean_action_mse"],
        "zero_action_mse": finished["zero_action_mse"],
        "previous_action_mse": finished["previous_action_mse"],
        "balanced_mean_action_mse": balanced_mean_action_mse(per_asset),
        "skill_vs_zero": 1.0 - float(finished["mean_action_mse"]) / float(finished["zero_action_mse"])
        if math.isfinite(float(finished["zero_action_mse"])) and float(finished["zero_action_mse"]) > 0.0
        else float("nan"),
        "validation_seconds": elapsed,
        "per_family": {key: _finish_metric(value) for key, value in sorted(by_family.items())},
        "per_asset": per_asset,
    }
    if "fk_mse" in finished:
        result["fk_mse"] = finished["fk_mse"]
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_family_metadata(path: str | os.PathLike[str]) -> dict[str, object]:
    """Read and validate the trajectory identity and actor input schema."""
    from anymani.distill.il.family_dataset import read_family_metadata as read_metadata_impl

    value = read_metadata_impl(Path(path))
    if not isinstance(value, Mapping):
        raise ValueError("family_dataset.read_family_metadata must return a mapping")
    return dict(value)


def reconstruct_history(
    initial_history: np.ndarray, frames: np.ndarray, step_indices: np.ndarray, env_indices: np.ndarray
) -> np.ndarray:
    """Reconstruct oldest-to-current History30 without crossing episode boundaries."""
    from anymani.distill.il.family_dataset import reconstruct_history as reconstruct_history_impl

    value = reconstruct_history_impl(initial_history, frames, step_indices, env_indices)
    return np.asarray(value)


def quality_episode_mask(**final: np.ndarray) -> np.ndarray:
    """Select complete, safe demonstrations using the fixed rotation thresholds."""
    from anymani.distill.il.family_dataset import quality_episode_mask as quality_episode_mask_impl

    value = quality_episode_mask_impl(**final)
    return np.asarray(value, dtype=bool)


def _h5_dataset(handle: h5py.File, name: str) -> h5py.Dataset:
    value = handle.get(name)
    if not isinstance(value, h5py.Dataset):
        raise ValueError(f"family dataset misses required dataset {name!r}")
    return cast(h5py.Dataset, value)


def _metadata_contract(metadata: Mapping[str, object], path: Path) -> tuple[str, str, str]:
    if metadata.get("artifact_type") != "anymani.family_teacher_trajectory":
        raise ValueError(f"{path}: family trajectory artifact_type is invalid")
    if metadata.get("schema_version") != "1.0.0":
        raise ValueError(f"{path}: family trajectory schema must be 1.0.0")
    if metadata.get("completed") is not True:
        raise ValueError(f"{path}: family trajectory is not complete")
    abi = metadata.get("actor_abi")
    if not isinstance(abi, Mapping) or dict(abi) != FAMILY_STUDENT_ACTOR_ABI:
        raise ValueError(f"{path}: family trajectory actor_abi disagrees with student ABI")
    family = metadata.get("family")
    teacher_sha = metadata.get("teacher_checkpoint_sha256")
    n040_sha = metadata.get("n040_sha256")
    if not isinstance(family, str) or not family:
        raise ValueError(f"{path}: family metadata is missing")
    if not isinstance(teacher_sha, str) or not teacher_sha:
        raise ValueError(f"{path}: teacher_checkpoint_sha256 is missing")
    if not isinstance(n040_sha, str) or not n040_sha:
        raise ValueError(f"{path}: n040_sha256 is missing")
    ordered_assets = metadata.get("ordered_assets")
    if isinstance(ordered_assets, (str, bytes)) or not isinstance(ordered_assets, Sequence):
        raise ValueError(f"{path}: ordered_assets must be an ordered sequence")
    protocol = metadata.get("protocol")
    if not isinstance(protocol, Mapping) or "action_mode" not in protocol or "action_seed" not in protocol:
        raise ValueError(f"{path}: protocol.action_mode/action_seed are required")
    return (family, teacher_sha, n040_sha)


def _collection_identity(path: Path, metadata: Mapping[str, object]) -> str:
    protocol = cast(Mapping[str, object], metadata["protocol"])
    with h5py.File(path, "r") as handle:
        env_asset = np.asarray(_h5_dataset(handle, "env_asset_index")).astype(np.int64, copy=False)
        env_replica = np.asarray(_h5_dataset(handle, "env_replica_index")).astype(np.int64, copy=False)
        axes = {
            "expected_steps": int(handle.attrs.get("expected_steps", -1)),
            "recorded_steps": int(handle.attrs.get("recorded_steps", -1)),
            "sample_stride": int(handle.attrs.get("sample_stride", -1)),
            "sample_count": int(handle.attrs.get("sample_count", -1)),
        }
    payload = {
        "teacher_checkpoint_sha256": metadata["teacher_checkpoint_sha256"],
        "cohort_sha256": metadata["cohort_sha256"],
        "n040_sha256": metadata["n040_sha256"],
        "family": metadata["family"],
        "ordered_assets": metadata["ordered_assets"],
        "protocol": dict(protocol),
        "env_asset_index": env_asset.tolist(),
        "env_replica_index": env_replica.tolist(),
        "axes": axes,
    }
    return _stable_digest(payload)


def _asset_name(metadata: Mapping[str, object], asset_index: int) -> str:
    ordered = metadata.get("ordered_assets", [])
    if isinstance(ordered, Sequence) and (not isinstance(ordered, (str, bytes))) and (asset_index < len(ordered)):
        item = ordered[asset_index]
        if isinstance(item, Mapping):
            for key in ("asset_id", "id", "name"):
                if key in item:
                    return str(item[key])
        return str(item)
    return str(asset_index)


def _quality_for_handle(handle: h5py.File) -> np.ndarray:
    stored_raw = np.asarray(_h5_dataset(handle, "final/quality_episode_mask"))
    if stored_raw.ndim != 1 or stored_raw.size < 1:
        raise ValueError("stored family quality mask must be a nonempty [N] vector")
    if stored_raw.dtype.kind not in "biuf" or not bool(np.isfinite(stored_raw).all()):
        raise ValueError("stored family quality mask must be finite numeric/bool")
    if not bool(np.isin(stored_raw, (0, 1)).all()):
        raise ValueError("stored family quality mask must contain exact 0/1 values")
    stored_quality = stored_raw.astype(bool, copy=False)
    final = {
        name: np.asarray(_h5_dataset(handle, f"final/{name}")).astype(np.float64, copy=False)
        for name in ("net_turns", "path_turns", "duration_s")
    }
    final["termination_drop"] = np.asarray(_h5_dataset(handle, "final/termination_drop"))
    final["termination_axis"] = np.asarray(_h5_dataset(handle, "final/termination_axis"))
    for name, value in final.items():
        if value.ndim != 1 or value.shape != stored_quality.shape:
            raise ValueError(
                f"final/{name} shape {value.shape} disagrees with stored quality shape {stored_quality.shape}"
            )
    quality_gate = quality_episode_mask(**final)
    active_raw = np.asarray(_h5_dataset(handle, "frames/active"))
    if active_raw.ndim != 2 or active_raw.shape[1] != stored_quality.shape[0]:
        raise ValueError("frames/active shape disagrees with stored quality env axis")
    if active_raw.dtype.kind not in "biuf" or not bool(np.isfinite(active_raw).all()):
        raise ValueError("frames/active must be finite numeric/bool")
    if not bool(np.isin(active_raw, (0, 1)).all()):
        raise ValueError("frames/active must contain exact 0/1 values")
    active_count = active_raw.astype(bool, copy=False).sum(axis=0)
    recorded_steps = int(handle.attrs.get("recorded_steps", active_raw.shape[0]))
    protocol_ok = (recorded_steps == int(QUALITY_DURATION_S * 20.0)) & (active_count == int(QUALITY_DURATION_S * 20.0))
    policy_dataset = handle.get("final/policy_step_count")
    if isinstance(policy_dataset, h5py.Dataset):
        policy_steps = np.asarray(policy_dataset)
        if policy_steps.shape != stored_quality.shape or policy_steps.dtype.kind not in "iu":
            raise ValueError("final/policy_step_count must be integer [N] aligned with quality mask")
        protocol_ok &= policy_steps == int(QUALITY_DURATION_S * 20.0)
    recomputed = quality_gate & protocol_ok
    if not np.array_equal(recomputed, stored_quality):
        mismatched = np.flatnonzero(recomputed != stored_quality).tolist()
        raise ValueError(
            f"stored family quality mask disagrees with recomputed gate/protocol at env rows {mismatched[:8]}"
        )
    return recomputed


def _estimate_source_bytes(
    handle: h5py.File,
    quality_env: np.ndarray,
    active: np.ndarray,
    sample_steps: np.ndarray,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    selected = np.flatnonzero(quality_env)
    count_env = len(selected)
    if active.ndim != 2 or active.shape[1] != len(quality_env):
        raise ValueError("family active must have shape [T,N] aligned with final/static env axis")
    if sample_steps.ndim != 1 or np.any(sample_steps < 0) or np.any(sample_steps >= active.shape[0]):
        raise ValueError("family sample step_index must lie in [0,T)")
    pair_count = int(np.asarray(active[np.asarray(sample_steps, dtype=np.int64)][:, selected], dtype=bool).sum())
    if batch_size < 1:
        raise ValueError("resident estimate batch_size must be positive")
    estimated = 0

    def nbytes(shape: Sequence[int], itemsize: int) -> int:
        return math.prod((int(dimension) for dimension in shape)) * int(itemsize)

    for name in ("frames/jnt_current", "frames/owner_contact"):
        dataset = _h5_dataset(handle, name)
        shape = (dataset.shape[0], count_env, *dataset.shape[2:])
        estimated += nbytes(shape, dataset.dtype.itemsize)
    active_dataset = _h5_dataset(handle, "frames/active")
    estimated += nbytes(active_dataset.shape, np.dtype(np.bool_).itemsize)
    for name in ("initial_history",):
        dataset = _h5_dataset(handle, name)
        shape = (count_env, *dataset.shape[1:])
        estimated += nbytes(shape, dataset.dtype.itemsize)
    for name in (
        "static/actor_jnt_limits",
        "static/jnt_valid",
        "static/tip_valid",
        "static/owner_valid",
        "static/joint_kinematics",
        "static/shortest_path",
        "static/parent_direction",
        "static/child_direction",
    ):
        dataset = _h5_dataset(handle, name)
        raw_shape = dataset.shape
        quality_shape = (count_env, *dataset.shape[1:])
        output_itemsize = (
            np.dtype(np.int64).itemsize
            if "direction" in name or "shortest" in name
            else np.dtype(np.bool_).itemsize
            if name.endswith("valid")
            else np.dtype(np.float32).itemsize
        )
        estimated += nbytes(raw_shape, dataset.dtype.itemsize) + nbytes(quality_shape, output_itemsize)
    for name in (
        "samples/teacher_mean",
        "samples/behavior_action",
        "samples/geometry_tokens",
        "samples/joint_origin_fk",
    ):
        dataset = _h5_dataset(handle, name)
        filtered_shape = (min(SAMPLE_READ_CHUNK_ROWS, dataset.shape[0]), count_env, *dataset.shape[2:])
        gathered_shape = (pair_count, *dataset.shape[2:])
        estimated += nbytes(filtered_shape, dataset.dtype.itemsize)
        estimated += nbytes(gathered_shape, dataset.dtype.itemsize)
    estimated += pair_count * (np.dtype(np.int64).itemsize * 3 + 64)
    per_sample_bytes = (
        30 * 16 * 5 * 4
        + 16 * 5 * 4
        + 16 * 2 * 4
        + 21 * 1 * 4
        + 16
        + 4
        + 21
        + 21 * 128 * 4
        + 3 * 21 * 21 * 8
        + 16 * 15 * 4
        + 16 * 4 * 2
        + 16 * 3 * 4
        + 8
    )
    estimated += 2 * batch_size * per_sample_bytes
    return estimated


def estimate_compact_resident_bytes(
    *,
    recorded_steps: int,
    env_count: int,
    quality_env_count: int,
    sample_count: int,
    active_pair_count: int,
    asset_count: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    sample_chunk_rows: int = SAMPLE_READ_CHUNK_ROWS,
) -> int:
    """Estimate memory before loading compact trajectory arrays."""
    dimensions = (
        recorded_steps,
        env_count,
        quality_env_count,
        sample_count,
        active_pair_count,
        asset_count,
        batch_size,
        sample_chunk_rows,
    )
    if any((int(value) < 0 for value in dimensions)) or batch_size < 1 or sample_chunk_rows < 1:
        raise ValueError("compact resident shape counts must be non-negative and batch/chunk sizes positive")

    def bytes_for(shape: Sequence[int], itemsize: int) -> int:
        return math.prod((int(value) for value in shape)) * itemsize

    resident = bytes_for((recorded_steps, quality_env_count, 16, 5), 4)
    resident += bytes_for((recorded_steps, quality_env_count, 21, 1), 4)
    resident += bytes_for((recorded_steps, env_count), 1)
    resident += bytes_for((quality_env_count, 30, 16, 5), 4)
    static_suffixes = (
        (16, 2, 4),
        (16, 1, 1),
        (4, 1, 1),
        (21, 1, 1),
        (16, 15, 4),
        (21, 21, 8),
        (21, 21, 8),
        (21, 21, 8),
    )
    for dim0, dim1, itemsize in static_suffixes:
        resident += bytes_for((asset_count, dim0, dim1), itemsize)
        resident += bytes_for((quality_env_count, dim0, dim1), itemsize)
    sample_suffixes = ((16, 4), (16, 4), (21 * 128, 4), (16 * 3, 4))
    for dim1, itemsize in sample_suffixes:
        resident += bytes_for((min(sample_chunk_rows, sample_count), quality_env_count, dim1), itemsize)
        resident += bytes_for((active_pair_count, dim1), itemsize)
    resident += bytes_for((active_pair_count,), 8 * 3 + 64)
    per_sample = (
        30 * 16 * 5 * 4
        + 16 * 5 * 4
        + 16 * 2 * 4
        + 21 * 4
        + 16
        + 4
        + 21
        + 21 * 128 * 4
        + 3 * 21 * 21 * 8
        + 16 * 15 * 4
        + 16 * 4 * 2
        + 16 * 3 * 4
        + 8
    )
    return resident + 2 * batch_size * per_sample


def _read_sample_pairs(
    handle: h5py.File,
    quality_env: np.ndarray,
    sample_steps: np.ndarray,
    active: np.ndarray,
    *,
    chunk_rows: int = SAMPLE_READ_CHUNK_ROWS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if chunk_rows < 1:
        raise ValueError("sample read chunk_rows must be positive")
    quality = np.asarray(quality_env, dtype=bool)
    selected = np.flatnonzero(quality).astype(np.int64, copy=False)
    sample_count = len(sample_steps)
    teacher_dataset = _h5_dataset(handle, "samples/teacher_mean")
    behavior_dataset = _h5_dataset(handle, "samples/behavior_action")
    geometry_dataset = _h5_dataset(handle, "samples/geometry_tokens")
    fk_dataset = _h5_dataset(handle, "samples/joint_origin_fk")
    pair_count = int(np.asarray(active[sample_steps][:, selected], dtype=bool).sum())
    sample_rows = np.empty((pair_count,), dtype=np.int64)
    env_indices = np.empty((pair_count,), dtype=np.int64)
    targets = np.empty((pair_count, 16), dtype=np.float32)
    behaviors = np.empty((pair_count, 16), dtype=np.float32)
    tokens = np.empty((pair_count, 21, 128), dtype=np.float32)
    fk_targets = np.empty((pair_count, 16, 3), dtype=np.float32)
    cursor = 0
    for start in range(0, sample_count, chunk_rows):
        stop = min(start + chunk_rows, sample_count)
        active_chunk = np.asarray(active[sample_steps[start:stop]][:, selected], dtype=bool)
        local_rows, local_env = np.nonzero(active_chunk)
        if local_rows.size == 0:
            continue
        local_rows = np.asarray(local_rows, dtype=np.int64)
        local_env = np.asarray(local_env, dtype=np.int64)
        quality_env_index = local_env
        teacher_chunk = np.asarray(teacher_dataset[start:stop, selected], dtype=np.float32)
        behavior_chunk = np.asarray(behavior_dataset[start:stop, selected], dtype=np.float32)
        geometry_chunk = np.asarray(geometry_dataset[start:stop, selected], dtype=np.float32)
        fk_chunk = np.asarray(fk_dataset[start:stop, selected], dtype=np.float32)
        count = len(local_rows)
        stop_cursor = cursor + count
        sample_rows[cursor:stop_cursor] = start + local_rows
        env_indices[cursor:stop_cursor] = quality_env_index
        targets[cursor:stop_cursor] = teacher_chunk[local_rows, local_env]
        behaviors[cursor:stop_cursor] = behavior_chunk[local_rows, local_env]
        tokens[cursor:stop_cursor] = geometry_chunk[local_rows, local_env]
        fk_targets[cursor:stop_cursor] = fk_chunk[local_rows, local_env]
        cursor = stop_cursor
    if pair_count == 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0, 16), dtype=np.float32),
            np.empty((0, 16), dtype=np.float32),
            np.empty((0, 21, 128), dtype=np.float32),
            np.empty((0, 16, 3), dtype=np.float32),
        )
    if cursor != pair_count:
        raise RuntimeError(f"sample pair reader filled {cursor} rows but expected {pair_count}")
    return (sample_rows, env_indices, targets, behaviors, tokens, fk_targets)


def _array_finite(value: np.ndarray, *, name: str) -> None:
    if value.dtype.kind in "fc" and (not bool(np.isfinite(value).all())):
        raise ValueError(f"family dataset {name} contains non-finite values")


def _make_split_batch(
    *,
    metadata: Mapping[str, object],
    family: str,
    selected_env: np.ndarray,
    initial_history: np.ndarray,
    frames_current: np.ndarray,
    frames_contact: np.ndarray,
    sample_steps: np.ndarray,
    pair_sample_row: np.ndarray,
    pair_env_index: np.ndarray,
    pair_target: np.ndarray,
    pair_behavior_action: np.ndarray,
    pair_geometry_tokens: np.ndarray,
    pair_fk_target: np.ndarray,
    env_asset_index: np.ndarray,
    static_limits: np.ndarray,
    static_jnt_valid: np.ndarray,
    static_tip_valid: np.ndarray,
    static_owner_valid: np.ndarray,
    static_joint_kinematics: np.ndarray,
    static_shortest_path: np.ndarray,
    static_parent_direction: np.ndarray,
    static_child_direction: np.ndarray,
) -> CompactFamilyBatch | None:
    if selected_env.size == 0:
        return None
    pair_mask = np.isin(pair_env_index, selected_env)
    if not bool(pair_mask.any()):
        return None
    all_pairs = bool(pair_mask.all())
    sample_row = np.asarray(pair_sample_row if all_pairs else pair_sample_row[pair_mask], dtype=np.int64)
    global_env = np.asarray(pair_env_index if all_pairs else pair_env_index[pair_mask], dtype=np.int64)
    target = np.asarray(pair_target if all_pairs else pair_target[pair_mask], dtype=np.float32)
    behavior = np.asarray(pair_behavior_action if all_pairs else pair_behavior_action[pair_mask], dtype=np.float32)
    tokens = np.asarray(pair_geometry_tokens if all_pairs else pair_geometry_tokens[pair_mask], dtype=np.float32)
    fk = np.asarray(pair_fk_target if all_pairs else pair_fk_target[pair_mask], dtype=np.float32)
    _array_finite(target, name="samples/teacher_mean")
    _array_finite(behavior, name="samples/behavior_action")
    _array_finite(tokens, name="samples/geometry_tokens")
    _array_finite(fk, name="samples/joint_origin_fk")
    if (
        target.shape != (len(sample_row), 16)
        or behavior.shape != target.shape
        or tokens.shape != (len(sample_row), 21, 128)
    ):
        raise ValueError("family samples teacher/behavior/geometry shapes disagree with canonical ABI")
    if fk.shape != (len(sample_row), 16, 3):
        raise ValueError("family samples FK shape disagrees with canonical ABI")
    if np.any(np.abs(target) > 1.0 + 1e-05):
        raise ValueError("family teacher_mean must stay within canonical action interval [-1,1]")
    asset_index = np.asarray(env_asset_index, dtype=np.int64)
    env_assets = asset_index[global_env]
    compact_jnt_valid = np.asarray(static_jnt_valid[asset_index], dtype=bool)
    compact_tip_valid = np.asarray(static_tip_valid[asset_index], dtype=bool)
    compact_owner_valid = np.asarray(static_owner_valid[asset_index], dtype=bool)
    _array_finite(np.asarray(static_limits[asset_index]), name="static/actor_jnt_limits")
    _array_finite(np.asarray(static_joint_kinematics[asset_index]), name="static/joint_kinematics")
    expected_owner = np.concatenate(
        (np.ones((len(asset_index), 1), dtype=bool), compact_jnt_valid, compact_tip_valid), axis=-1
    )
    if not np.array_equal(compact_owner_valid, expected_owner):
        raise ValueError("family static owner_valid disagrees with PALM/JOINT/TIP masks")
    ordered_assets = cast(Sequence[object], metadata["ordered_assets"])
    asset_name_table = tuple((_asset_name(metadata, asset_index) for asset_index in range(len(ordered_assets))))
    asset_labels = tuple((asset_name_table[int(asset_index)] for asset_index in env_assets))
    return CompactFamilyBatch(
        initial_history=np.asarray(initial_history, dtype=np.float32),
        frames_current=np.asarray(frames_current, dtype=np.float32),
        frames_contact=np.asarray(frames_contact, dtype=np.float32),
        sample_steps=np.asarray(sample_steps, dtype=np.int64),
        sample_row=sample_row,
        env_index=global_env,
        target=target,
        behavior_action=behavior,
        geometry_tokens=tokens,
        fk_target=fk,
        family_ids=(family,) * len(sample_row),
        asset_ids=asset_labels,
        static_limits=np.asarray(static_limits[asset_index], dtype=np.float32),
        static_jnt_valid=compact_jnt_valid,
        static_tip_valid=compact_tip_valid,
        static_owner_valid=compact_owner_valid,
        static_joint_kinematics=np.asarray(static_joint_kinematics[asset_index], dtype=np.float32),
        static_shortest_path=np.asarray(static_shortest_path[asset_index], dtype=np.int64),
        static_parent_direction=np.asarray(static_parent_direction[asset_index], dtype=np.int64),
        static_child_direction=np.asarray(static_child_direction[asset_index], dtype=np.int64),
    )


def _load_source(path: Path, *, max_ram_gib: float, expected_n040: str | None, batch_size: int) -> FamilySource:
    metadata = read_family_metadata(path)
    family, teacher_sha, n040_sha = _metadata_contract(metadata, path)
    if expected_n040 is not None and n040_sha != expected_n040:
        raise ValueError(f"{path}: n040_sha256 drift: expected {expected_n040}, got {n040_sha}")
    dataset_sha = _sha256(path)
    source_sha = teacher_sha
    collection_identity = _collection_identity(path, metadata)
    with h5py.File(path, "r") as handle:
        quality = _quality_for_handle(handle)
        env_asset = np.asarray(_h5_dataset(handle, "env_asset_index")).astype(np.int64, copy=False)
        env_replica = np.asarray(_h5_dataset(handle, "env_replica_index")).astype(np.int64, copy=False)
        sample_steps = np.asarray(_h5_dataset(handle, "samples/step_index")).astype(np.int64, copy=False)
        active_probe = np.asarray(_h5_dataset(handle, "frames/active")).astype(bool, copy=False)
        estimate = _estimate_source_bytes(handle, quality, active_probe, sample_steps, batch_size=batch_size)
        if estimate > max_ram_gib * 1024**3:
            raise MemoryError(
                f"{path}: estimated resident {estimate / 1024**3:.2f} GiB exceeds fail-closed budget {max_ram_gib:.2f} GiB"
            )
        static_limits = np.asarray(_h5_dataset(handle, "static/actor_jnt_limits"))
        static_jnt_valid = np.asarray(_h5_dataset(handle, "static/jnt_valid"))
        static_tip_valid = np.asarray(_h5_dataset(handle, "static/tip_valid"))
        static_owner_valid = np.asarray(_h5_dataset(handle, "static/owner_valid"))
        static_joint_kinematics = np.asarray(_h5_dataset(handle, "static/joint_kinematics"))
        static_shortest = np.asarray(_h5_dataset(handle, "static/shortest_path"))
        static_parent = np.asarray(_h5_dataset(handle, "static/parent_direction"))
        static_child = np.asarray(_h5_dataset(handle, "static/child_direction"))
        selected_quality = np.flatnonzero(quality)
        frames_current = np.asarray(_h5_dataset(handle, "frames/jnt_current")[:, selected_quality], dtype=np.float32)
        frames_contact = np.asarray(_h5_dataset(handle, "frames/owner_contact")[:, selected_quality], dtype=np.float32)
        initial_quality = np.asarray(_h5_dataset(handle, "initial_history")[selected_quality], dtype=np.float32)
        _array_finite(frames_current, name="frames/jnt_current")
        _array_finite(frames_contact, name="frames/owner_contact")
        _array_finite(initial_quality, name="initial_history")
        pair_sample_row, pair_env_index, pair_target, pair_behavior, pair_tokens, pair_fk = _read_sample_pairs(
            handle, quality, sample_steps, active_probe, chunk_rows=SAMPLE_READ_CHUNK_ROWS
        )
        env_asset_quality = env_asset[selected_quality]
        env_replica_quality = env_replica[selected_quality]
        train_env = np.flatnonzero(env_replica_quality % VALIDATION_REPLICA_MODULUS != VALIDATION_REPLICA_REMAINDER)
        validation_env = np.flatnonzero(
            env_replica_quality % VALIDATION_REPLICA_MODULUS == VALIDATION_REPLICA_REMAINDER
        )
        all_batch = _make_split_batch(
            metadata=metadata,
            family=family,
            selected_env=np.arange(len(selected_quality), dtype=np.int64),
            initial_history=initial_quality,
            frames_current=frames_current,
            frames_contact=frames_contact,
            sample_steps=sample_steps,
            pair_sample_row=pair_sample_row,
            pair_env_index=pair_env_index,
            pair_target=pair_target,
            pair_behavior_action=pair_behavior,
            pair_geometry_tokens=pair_tokens,
            pair_fk_target=pair_fk,
            env_asset_index=env_asset_quality,
            static_limits=static_limits,
            static_jnt_valid=static_jnt_valid,
            static_tip_valid=static_tip_valid,
            static_owner_valid=static_owner_valid,
            static_joint_kinematics=static_joint_kinematics,
            static_shortest_path=static_shortest,
            static_parent_direction=static_parent,
            static_child_direction=static_child,
        )
        if all_batch is None:
            train_batch = None
            validation_batch = None
        else:
            train_selector = np.flatnonzero(np.isin(all_batch.env_index, train_env))
            validation_selector = np.flatnonzero(np.isin(all_batch.env_index, validation_env))
            train_batch = all_batch.subset(train_selector)
            validation_batch = all_batch.subset(validation_selector)
        training = () if train_batch is None or train_batch.sample_count == 0 else (train_batch,)
        validation = () if validation_batch is None or validation_batch.sample_count == 0 else (validation_batch,)
        ordered_assets = metadata.get("ordered_assets", [])
        if isinstance(ordered_assets, Sequence) and (not isinstance(ordered_assets, (str, bytes))):
            asset_indices = range(len(ordered_assets))
        else:
            asset_indices = sorted(set((int(item) for item in env_asset)))
        all_asset_names = tuple((_asset_name(metadata, int(index)) for index in asset_indices))
        quality_asset_names = tuple(sorted({_asset_name(metadata, int(item)) for item in env_asset_quality}))
        training_asset_counts = _view_asset_counts(training)
        validation_asset_counts = _view_asset_counts(validation)
        data_asset_names = tuple(sorted(set(training_asset_counts) | set(validation_asset_counts)))
        zero_data_assets = tuple((name for name in all_asset_names if name not in data_asset_names))
        asset_report = {
            name: {
                "quality": name in quality_asset_names,
                "samples": training_asset_counts[name] + validation_asset_counts[name],
                "training_samples": training_asset_counts[name],
                "validation_samples": validation_asset_counts[name],
            }
            for name in all_asset_names
        }
        report = {
            "path": str(path),
            "family": family,
            "quality_asset_count": len(quality_asset_names),
            "quality_env_count": int(quality.sum()),
            "training_samples": int(sum((batch.sample_count for batch in training))),
            "validation_samples": int(sum((batch.sample_count for batch in validation))),
            "zero_data_assets": zero_data_assets,
            "assets": asset_report,
            "estimated_resident_bytes": estimate,
            "source_sha256": source_sha,
            "collection_identity": collection_identity,
            "dataset_sha256": dataset_sha,
            "teacher_checkpoint_sha256": teacher_sha,
            "cohort_sha256": metadata["cohort_sha256"],
            "n040_sha256": n040_sha,
            "ordered_assets": metadata["ordered_assets"],
        }
    return FamilySource(
        path=path,
        metadata=metadata,
        dataset_sha256=dataset_sha,
        source_sha256=source_sha,
        collection_identity=collection_identity,
        family=family,
        n040_sha256=n040_sha,
        training=training,
        validation=validation,
        report=report,
    )


def load_family_sources(
    dataset_paths: Sequence[str | os.PathLike[str]],
    *,
    max_ram_gib: float = DEFAULT_MAX_RAM_GIB,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> FamilyDatasetBundle:
    """Validate HDF5 sources and assemble compact training and validation views."""
    if not dataset_paths:
        raise ValueError("family student requires at least one --dataset")
    if max_ram_gib <= 0.0 or not math.isfinite(max_ram_gib):
        raise ValueError("max_ram_gib must be positive and finite")
    if batch_size < 1:
        raise ValueError("family source batch_size must be positive")
    paths = tuple((Path(path) for path in dataset_paths))
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate family dataset path is rejected")
    preflight_estimated = 0
    preflight_collection_identities: set[str] = set()
    preflight_n040: str | None = None
    for path in paths:
        preflight_metadata = read_family_metadata(path)
        _, _, preflight_source_n040 = _metadata_contract(preflight_metadata, path)
        preflight_collection = _collection_identity(path, preflight_metadata)
        if preflight_collection in preflight_collection_identities:
            raise ValueError(f"duplicate family collection identity is rejected: {preflight_collection}")
        if preflight_n040 is not None and preflight_source_n040 != preflight_n040:
            raise ValueError(f"{path}: n040_sha256 drift: expected {preflight_n040}, got {preflight_source_n040}")
        preflight_collection_identities.add(preflight_collection)
        preflight_n040 = preflight_source_n040 if preflight_n040 is None else preflight_n040
        preflight_estimated += estimate_family_resident_bytes(path, batch_size=batch_size)
    if preflight_estimated > max_ram_gib * 1024**3:
        raise MemoryError(
            f"family sources estimated resident {preflight_estimated / 1024**3:.2f} GiB exceeds fail-closed budget {max_ram_gib:.2f} GiB"
        )
    sources: list[FamilySource] = []
    collection_identities: set[str] = set()
    dataset_hashes: set[str] = set()
    n040: str | None = None
    for path in paths:
        source = _load_source(path, max_ram_gib=max_ram_gib, expected_n040=n040, batch_size=batch_size)
        if source.collection_identity in collection_identities:
            raise ValueError(f"duplicate family collection identity is rejected: {source.collection_identity}")
        if source.dataset_sha256 in dataset_hashes:
            raise ValueError(f"duplicate family dataset SHA is rejected: {source.dataset_sha256}")
        collection_identities.add(source.collection_identity)
        dataset_hashes.add(source.dataset_sha256)
        n040 = source.n040_sha256 if n040 is None else n040
        sources.append(source)
    assert n040 is not None
    identity = [(source.dataset_sha256, source.collection_identity) for source in sources]
    return FamilyDatasetBundle(
        sources=tuple(sources),
        training=tuple((batch for source in sources for batch in source.training)),
        validation=tuple((batch for source in sources for batch in source.validation)),
        dataset_sha256=_stable_digest(identity),
        n040_sha256=n040,
        reports=tuple((source.report for source in sources)),
    )


def estimate_family_resident_bytes(path: str | os.PathLike[str], *, batch_size: int = DEFAULT_BATCH_SIZE) -> int:
    """Estimate the complete resident memory required by all sources."""
    source_path = Path(path)
    metadata = read_family_metadata(source_path)
    _metadata_contract(metadata, source_path)
    with h5py.File(source_path, "r") as handle:
        quality = _quality_for_handle(handle)
        active = np.asarray(_h5_dataset(handle, "frames/active")).astype(bool, copy=False)
        steps = np.asarray(_h5_dataset(handle, "samples/step_index")).astype(np.int64, copy=False)
        return _estimate_source_bytes(handle, quality, active, steps, batch_size=batch_size)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_reproducible_training() -> None:
    """Fix CPU reductions and CUDA kernels for repeatable BC updates."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _reproducibility_settings() -> dict[str, object]:
    return {
        "cpu_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _rng_state(generator: torch.Generator) -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "sampling_generator": generator.get_state().clone(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, object]) -> None:
    if isinstance(state.get("python"), tuple):
        random.setstate(cast(Any, state["python"]))
    if isinstance(state.get("numpy"), tuple):
        np.random.set_state(cast(Any, state["numpy"]))
    if isinstance(state.get("torch"), torch.Tensor):
        torch.set_rng_state(cast(torch.Tensor, state["torch"]).cpu())
    cuda_state = state.get("cuda")
    if torch.cuda.is_available() and isinstance(cuda_state, list):
        torch.cuda.set_rng_state_all(
            [cast(torch.Tensor, value).cpu() for value in cuda_state if isinstance(value, torch.Tensor)]
        )


def _source_code_hash() -> str:
    return family_student_source_code_hash(family_student_source_code_files())


def _source_code_files() -> dict[str, str]:
    return family_student_source_code_files()


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            json.dump(_json_ready(payload), stream, ensure_ascii=False, indent=2, allow_nan=False, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def _atomic_append_jsonl(path: Path, record: Mapping[str, object]) -> None:
    previous = path.read_text(encoding="utf-8") if path.exists() else ""
    if previous and (not previous.endswith("\n")):
        raise ValueError(f"metrics file {path} ends with an incomplete JSONL record")
    encoded = json.dumps(_json_ready(record), ensure_ascii=False, separators=(",", ":"), allow_nan=False, default=str)
    _atomic_write_text(path, previous + encoded + "\n")


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def _load_json_report(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read recoverable training report {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("training-report.json root must be an object")
    if value.get("artifact_type") != TRAINING_REPORT_ARTIFACT_TYPE:
        raise ValueError("training-report.json artifact_type mismatch")
    if value.get("schema_version") != TRAINING_REPORT_SCHEMA_VERSION:
        raise ValueError("training-report.json schema_version mismatch")
    if not isinstance(value.get("run_identity"), Mapping):
        raise ValueError("training-report.json misses run_identity")
    if not isinstance(value.get("epochs"), list):
        raise ValueError("training-report.json epochs must be a list")
    return value


def _load_metrics_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    records: list[dict[str, object]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read metrics file {path}: {exc}") from exc
    seen: set[int] = set()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise ValueError(f"metrics file {path} has an empty line at {line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"metrics file {path} has invalid JSON at line {line_number}") from exc
        if not isinstance(value, dict) or not isinstance(value.get("epoch"), int):
            raise ValueError(f"metrics file {path} line {line_number} lacks integer epoch")
        epoch = int(value["epoch"])
        if epoch in seen:
            raise ValueError(f"metrics file {path} repeats epoch {epoch}")
        seen.add(epoch)
        records.append(value)
    return records


def _json_finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _checkpoint_metric(value: float | None) -> float | None:
    return None if value is None or not math.isfinite(float(value)) else float(value)


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return _json_finite(float(value))
    if isinstance(value, (int, np.integer)) and (not isinstance(value, bool)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _run_identity(
    bundle: FamilyDatasetBundle,
    actor: FamilyRotationStudentActor,
    *,
    representation: Representation,
    seed: int,
    batch_size: int,
    learning_rate: float,
    max_epochs: int,
    max_seconds: float,
    tf32: bool,
) -> dict[str, object]:
    return {
        "dataset_sha256": bundle.dataset_sha256,
        "n040_sha256": bundle.n040_sha256,
        "representation": representation,
        "actor_config": dict(actor.family_actor_config),
        "seed": seed,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_epochs": max_epochs,
        "max_seconds": max_seconds,
        "lambda_fk": 0.1 if representation == "fk" else 0.0,
        "gradient_clip": 1.0,
        "amp": False,
        "tf32": bool(tf32),
        "sampling_version": TRAINING_SAMPLING_VERSION,
        "reproducibility": _reproducibility_settings(),
    }


def _require_sampling_version(metadata: Mapping[str, object], *, checkpoint: Path) -> None:
    protocol = metadata.get("protocol")
    version = protocol.get("sampling_version") if isinstance(protocol, Mapping) else None
    if version != TRAINING_SAMPLING_VERSION:
        raise ValueError(
            f"resume checkpoint {checkpoint} uses sampling_version {version!r}; expected {TRAINING_SAMPLING_VERSION!r}"
        )


def _validate_output_state(
    output: Path, *, resume: Path | None, run_identity: Mapping[str, object]
) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    report_path = output / TRAINING_REPORT_FILENAME
    metrics_path = output / METRICS_FILENAME
    if resume is None:
        if output.exists():
            existing = list(output.iterdir()) if output.is_dir() else [output]
            if existing:
                raise FileExistsError(
                    f"new family student run refuses non-empty output_dir {output}; use --resume for an existing run"
                )
        return (None, [])
    if not output.is_dir():
        raise FileNotFoundError(f"resume requires existing output_dir {output}")
    if resume.resolve().parent != output.resolve():
        raise ValueError("resume checkpoint must be inside the same output_dir")
    if not report_path.is_file() or not metrics_path.is_file():
        raise ValueError("resume requires both training-report.json and metrics.jsonl")
    report = _load_json_report(report_path)
    if dict(cast(Mapping[str, object], report["run_identity"])) != dict(run_identity):
        raise ValueError("resume output run_identity disagrees with dataset/variant/training config")
    metrics = _load_metrics_records(metrics_path)
    epochs = cast(list[object], report["epochs"])
    if len(metrics) != len(epochs):
        raise ValueError("metrics.jsonl and training-report.json epoch counts disagree")
    for index, (metric, epoch_report) in enumerate(zip(metrics, epochs, strict=True)):
        if not isinstance(epoch_report, Mapping):
            raise ValueError(f"training-report.json epoch {index} must be an object")
        for key in ("epoch", "update", "processed_samples"):
            if metric.get(key) != epoch_report.get(key):
                raise ValueError(f"metrics.jsonl and training-report.json disagree at epoch {index} field {key}")
    return (report, metrics)


def run_family_training(
    dataset_paths: Sequence[str | os.PathLike[str]],
    *,
    output_dir: str | os.PathLike[str],
    representation: Representation = "n040",
    seed: int = 42,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    device: torch.device | str | None = None,
    resume: str | os.PathLike[str] | None = None,
    max_updates: int | None = None,
    max_ram_gib: float = DEFAULT_MAX_RAM_GIB,
    tf32: bool = False,
) -> dict[str, object]:
    """Run the configured BC budget and select the best balanced validation checkpoint."""
    run_started = time.perf_counter()
    if representation not in {"n040", "no_z", "fk"}:
        raise ValueError(f"unknown family student representation {representation!r}")
    if max_epochs < 1 or max_seconds <= 0.0:
        raise ValueError("max_epochs and max_seconds must be positive")
    if max_updates is not None and max_updates < 1:
        raise ValueError("max_updates must be positive when provided")
    _configure_reproducible_training()
    _seed_everything(seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
    print(f"Loading {len(dataset_paths)} demonstration datasets...", flush=True)
    bundle = load_family_sources(dataset_paths, max_ram_gib=max_ram_gib, batch_size=batch_size)
    if not bundle.training:
        raise ValueError("family student dataset has no quality training samples")
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output = Path(output_dir)
    resume_path = None if resume is None else Path(resume)
    start_epoch = 0
    start_update = 0
    processed_samples = 0
    optimizer_state: Mapping[str, object] | None = None
    sampling_state: torch.Tensor | None = None
    resume_metadata: Mapping[str, object] = {}
    if resume_path is None:
        actor = build_family_student(representation, device=target_device)
    else:
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError("resume checkpoint root must be a mapping")
        actor, resume_metadata_value = load_family_student(
            resume_path,
            device=target_device,
            expected_variant=representation,
            expected_dataset_sha256=bundle.dataset_sha256,
            expected_n040_sha256=bundle.n040_sha256,
        )
        resume_metadata = resume_metadata_value
        _require_sampling_version(resume_metadata, checkpoint=resume_path)
        training_state = payload.get("training_state", {})
        if not isinstance(training_state, Mapping):
            raise ValueError("resume checkpoint training_state must be a mapping")
        start_epoch = int(cast(Any, training_state.get("epoch", 0)))
        start_update = int(cast(Any, training_state.get("update", 0)))
        processed_samples = int(cast(Any, training_state.get("processed_samples", 0)))
        optimizer_state_value = payload.get("optimizer_state_dict", {})
        optimizer_state = optimizer_state_value if isinstance(optimizer_state_value, Mapping) else None
        rng = payload.get("rng_state", {})
        if isinstance(rng, Mapping):
            _restore_rng_state(rng)
            if isinstance(rng.get("sampling_generator"), torch.Tensor):
                sampling_state = rng["sampling_generator"].clone()
    run_identity = _run_identity(
        bundle,
        actor,
        representation=representation,
        seed=seed,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        max_seconds=max_seconds,
        tf32=tf32,
    )
    existing_report, metrics_records = _validate_output_state(output, resume=resume_path, run_identity=run_identity)
    report_path = output / TRAINING_REPORT_FILENAME
    metrics_path = output / METRICS_FILENAME
    if existing_report is None:
        output.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(metrics_path, "")
        epoch_reports: list[dict[str, object]] = []
        cumulative_wall = 0.0
        best_mse = float("inf")
        best_epoch: int | None = None
        report = {
            "artifact_type": TRAINING_REPORT_ARTIFACT_TYPE,
            "schema_version": TRAINING_REPORT_SCHEMA_VERSION,
            "run_identity": run_identity,
            "status": "running",
            "started_unix_s": time.time(),
            "last_resume_unix_s": None,
            "cumulative_wall_seconds": cumulative_wall,
            "max_seconds": max_seconds,
            "best_validation_balanced_mean_action_mse": _checkpoint_metric(best_mse),
            "best_epoch": best_epoch,
            "epochs": epoch_reports,
            "metrics_path": str(metrics_path),
            "report_path": str(report_path),
        }
        _atomic_write_json(report_path, report)
    else:
        report = existing_report
        epoch_reports = [cast(dict[str, object], item) for item in cast(list[object], report["epochs"])]
        cumulative_wall = float(cast(Any, report.get("cumulative_wall_seconds", 0.0)))
        if not math.isfinite(cumulative_wall) or cumulative_wall < 0.0 or cumulative_wall >= max_seconds:
            raise RuntimeError("resume training wall budget is exhausted or invalid")
        stored_best = report.get("best_validation_balanced_mean_action_mse")
        if stored_best is None:
            best_mse = float("inf")
        else:
            best_mse = float(cast(Any, stored_best))
            if not math.isfinite(best_mse) or best_mse < 0.0:
                raise ValueError("training-report.json best validation metric is invalid")
        best_epoch_value = report.get("best_epoch")
        best_epoch = None if best_epoch_value is None else int(cast(Any, best_epoch_value))
        if best_epoch is not None and (not (output / "best.pt").is_file()):
            raise FileNotFoundError("resume report has a best epoch but output_dir/best.pt is missing")
        if epoch_reports:
            latest = epoch_reports[-1]
            if any(
                (
                    int(cast(Any, latest.get(key, -1))) != expected
                    for key, expected in (
                        ("epoch", start_epoch),
                        ("update", start_update),
                        ("processed_samples", processed_samples),
                    )
                )
            ):
                raise ValueError("resume checkpoint progress disagrees with training-report.json")
        report["status"] = "running"
        report["last_resume_unix_s"] = time.time()
        _atomic_write_json(report_path, report)
    train_samples = sum((batch.sample_count for batch in bundle.training))
    updates_per_epoch = max(1, math.ceil(train_samples / batch_size))
    update_budget = max_updates if max_updates is not None else max_epochs * updates_per_epoch
    if resume_path is not None:
        prior_protocol = resume_metadata.get("protocol", {})
        if isinstance(prior_protocol, Mapping):
            prior_batch = prior_protocol.get("batch_size")
            prior_lr = prior_protocol.get("learning_rate")
            if prior_batch is not None and int(cast(Any, prior_batch)) != batch_size:
                raise ValueError("resume batch_size disagrees with checkpoint protocol")
            if prior_lr is not None and float(cast(Any, prior_lr)) != learning_rate:
                raise ValueError("resume learning_rate disagrees with checkpoint protocol")
    trainable = [parameter for parameter in actor.parameters() if parameter.requires_grad]
    optimizer: torch.optim.Optimizer | None = torch.optim.Adam(trainable, lr=learning_rate)
    if optimizer_state is not None:
        optimizer.load_state_dict(dict(optimizer_state))
    generator: torch.Generator | None = None
    invocation_started = time.perf_counter()
    prior_cumulative_wall = cumulative_wall
    epoch = start_epoch
    while start_update < update_budget and epoch < max_epochs:
        elapsed_this_invocation = time.perf_counter() - invocation_started
        remaining = max_seconds - prior_cumulative_wall - elapsed_this_invocation
        if remaining <= 0.0:
            break
        epoch_started = time.perf_counter()
        result = fit_family_student(
            actor,
            bundle.training,
            max_updates=min(updates_per_epoch, update_budget - start_update),
            batch_size=batch_size,
            learning_rate=learning_rate,
            lambda_fk=0.1 if representation == "fk" else 0.0,
            seed=seed,
            max_seconds=remaining,
            optimizer=optimizer,
            sampling_generator=generator,
            sampling_generator_state=sampling_state,
            start_update=start_update,
            start_processed_samples=processed_samples,
        )
        optimizer = cast(torch.optim.Optimizer, result["optimizer"])
        generator = cast(torch.Generator, result["sampling_generator"])
        sampling_state = None
        completed_updates = int(cast(Any, result["updates"]))
        if completed_updates == 0:
            break
        start_update = int(cast(Any, result["total_updates"]))
        processed_samples = int(cast(Any, result["processed_samples"]))
        epoch += 1
        validation = evaluate_family_student(actor, bundle.validation, batch_size=batch_size)
        validation_mse = float(cast(Any, validation.get("balanced_mean_action_mse", float("nan"))))
        improved = math.isfinite(validation_mse) and validation_mse < best_mse
        next_best_mse = validation_mse if improved else best_mse
        next_best_epoch = epoch if improved else best_epoch
        epoch_wall = time.perf_counter() - epoch_started
        current_cumulative_wall = prior_cumulative_wall + (time.perf_counter() - invocation_started)
        metadata = {
            "family": "shared",
            "representation": representation,
            "variant": representation,
            "families": sorted({source.family for source in bundle.sources}),
            "dataset_sha256": bundle.dataset_sha256,
            "n040_sha256": bundle.n040_sha256,
            "teacher_checkpoint_sha256": [source.metadata["teacher_checkpoint_sha256"] for source in bundle.sources],
            "source_sha256": [source.source_sha256 for source in bundle.sources],
            "source_sha256_kind": "teacher_checkpoint",
            "collection_identity": [source.collection_identity for source in bundle.sources],
            "cohort_sha256": [source.metadata["cohort_sha256"] for source in bundle.sources],
            "ordered_assets": [source.metadata["ordered_assets"] for source in bundle.sources],
            "source_protocol": [source.metadata["protocol"] for source in bundle.sources],
            "asset_provenance": [
                {
                    "family": source.family,
                    "ordered_assets": source.metadata["ordered_assets"],
                    "teacher_checkpoint_sha256": source.metadata["teacher_checkpoint_sha256"],
                    "cohort_sha256": source.metadata["cohort_sha256"],
                    "n040_sha256": source.metadata["n040_sha256"],
                }
                for source in bundle.sources
            ],
            "actor_abi": dict(FAMILY_STUDENT_ACTOR_ABI),
            "quality": {
                "duration_s": QUALITY_DURATION_S,
                "net_turns_min": QUALITY_NET_TURNS,
                "net_path_ratio_min": QUALITY_NET_PATH_RATIO,
                "reports": bundle.reports,
            },
            "split": {
                "validation": "env_replica_index % 4 == 3",
                "train_remainder": [0, 1, 2],
                "complete_replica_isolation": True,
            },
            "protocol": {
                "objective": "teacher_mean_behavior_action_baseline",
                "best_selection": "balanced_mean_action_mse_asset_then_family",
                "history": "H0 includes current0; tail30(H0 + frames/current[1:t+1])",
                "fk_target_units": "meters divided by 0.1 m",
                "lambda_fk": 0.1 if representation == "fk" else 0.0,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "gradient_clip": 1.0,
                "amp": False,
                "tf32": bool(tf32),
                "sampling_version": TRAINING_SAMPLING_VERSION,
                "reproducibility": run_identity["reproducibility"],
                "sigma": "global_log_std_frozen_mean_supervision",
            },
            "source_code_files": _source_code_files(),
            "source_code_hash": _source_code_hash(),
        }
        training_state = {
            "epoch": epoch,
            "update": start_update,
            "processed_samples": processed_samples,
            "max_epochs": max_epochs,
            "max_updates": update_budget,
            "cumulative_wall_seconds": current_cumulative_wall,
            "best_validation_balanced_mean_action_mse": _checkpoint_metric(next_best_mse),
            "best_epoch": next_best_epoch,
        }
        rng_state = _rng_state(generator)
        save_kwargs = {
            "metadata": metadata,
            "optimizer_state_dict": cast(Mapping[str, object], result["optimizer_state_dict"]),
            "training_state": training_state,
            "rng_state": rng_state,
        }
        save_family_student_checkpoint(output / "last.pt", actor, **save_kwargs)
        periodic_path = output / f"epoch-{epoch:04d}.pt"
        save_family_student_checkpoint(periodic_path, actor, **save_kwargs)
        best_path: Path | None = None
        if improved:
            best_mse = next_best_mse
            best_epoch = next_best_epoch
            best_path = output / "best.pt"
            save_family_student_checkpoint(best_path, actor, **save_kwargs)
        elif (output / "best.pt").is_file():
            best_path = output / "best.pt"
        train_record = {
            "updates": int(cast(Any, result["updates"])),
            "total_updates": int(cast(Any, result["total_updates"])),
            "processed_samples": int(cast(Any, result["processed_samples"])),
            "initial_loss": result.get("initial_loss"),
            "last_loss": result.get("last_loss"),
            "last_bc_loss": result.get("last_bc_loss"),
            "last_fk_loss": result.get("last_fk_loss"),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "sample_count": int(cast(Any, result["sample_count"])),
        }
        epoch_record: dict[str, object] = {
            "epoch": epoch,
            "update": start_update,
            "processed_samples": processed_samples,
            "updates": completed_updates,
            "epoch_wall_seconds": epoch_wall,
            "cumulative_wall_seconds": current_cumulative_wall,
            "train": train_record,
            "validation": validation,
            "best_validation_balanced_mean_action_mse": _checkpoint_metric(best_mse),
            "best_epoch": best_epoch,
            "checkpoint_paths": {
                "last": str(output / "last.pt"),
                "periodic": str(periodic_path),
                "best": None if best_path is None else str(best_path),
            },
        }
        epoch_record = cast(dict[str, object], _json_ready(epoch_record))
        _atomic_append_jsonl(metrics_path, epoch_record)
        epoch_reports.append(epoch_record)
        report["epochs"] = epoch_reports
        report["status"] = "running"
        report["cumulative_wall_seconds"] = current_cumulative_wall
        report["best_validation_balanced_mean_action_mse"] = _checkpoint_metric(best_mse)
        report["best_epoch"] = best_epoch
        report["last_update_unix_s"] = time.time()
        _atomic_write_json(report_path, report)
        print(
            f"Round {epoch}/{max_epochs} | updates {start_update}/{update_budget} | "
            f"balanced validation MSE {validation_mse:.6f}",
            flush=True,
        )
    invocation_elapsed = time.perf_counter() - invocation_started
    cumulative_wall = prior_cumulative_wall + invocation_elapsed
    if start_update >= update_budget or epoch >= max_epochs:
        status = "completed"
    elif cumulative_wall >= max_seconds:
        status = "time_limit"
    else:
        status = "stopped"
    report["status"] = status
    report["cumulative_wall_seconds"] = cumulative_wall
    report["best_validation_balanced_mean_action_mse"] = _checkpoint_metric(best_mse)
    report["best_epoch"] = best_epoch
    report["finished_unix_s"] = time.time()
    _atomic_write_json(report_path, report)
    best_metric = _checkpoint_metric(best_mse)
    return {
        "artifact_type": FAMILY_STUDENT_ARTIFACT_TYPE,
        "schema": FAMILY_STUDENT_SCHEMA_VERSION,
        "representation": representation,
        "output_dir": str(output),
        "training_report_path": str(report_path),
        "metrics_path": str(metrics_path),
        "status": status,
        "epochs": epoch_reports,
        "last_epoch": epoch,
        "updates": start_update,
        "processed_samples": processed_samples,
        "best_validation_mse": best_metric,
        "best_balanced_mean_action_mse": best_metric,
        "best_epoch": best_epoch,
        "cumulative_wall_seconds": cumulative_wall,
        "elapsed_seconds": time.perf_counter() - run_started,
        "validation": epoch_reports[-1]["validation"] if epoch_reports else None,
        "dataset_sha256": bundle.dataset_sha256,
        "n040_sha256": bundle.n040_sha256,
        "reports": bundle.reports,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for offline student training."""
    parser = argparse.ArgumentParser(description="Offline three-variant family student mean imitation")
    parser.add_argument(
        "--dataset", action="append", required=True, help="family_teacher_trajectory HDF5; repeat per source"
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--representation", choices=("n040", "no_z", "fk"), default="n040")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max_epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--max_seconds", type=float, default=DEFAULT_MAX_SECONDS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max_updates", type=int, default=None)
    parser.add_argument("--max_ram_gib", type=float, default=DEFAULT_MAX_RAM_GIB)
    parser.add_argument("--tf32", action="store_true", help="explicitly enable CUDA TF32; default is disabled")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = run_family_training(
        args.dataset,
        output_dir=args.output_dir,
        representation=args.representation,
        seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        max_seconds=args.max_seconds,
        device=args.device,
        resume=args.resume,
        max_updates=args.max_updates,
        max_ram_gib=args.max_ram_gib,
        tf32=args.tf32,
    )
    summary = {
        "status": report.get("status"),
        "training_report_path": report.get("training_report_path"),
        "metrics_path": report.get("metrics_path"),
        "updates": report.get("updates"),
        "processed_samples": report.get("processed_samples"),
        "best_balanced_mean_action_mse": report.get("best_balanced_mean_action_mse"),
        "cumulative_wall_seconds": report.get("cumulative_wall_seconds"),
    }
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":"), allow_nan=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
__all__ = [
    "CompactFamilyBatch",
    "FamilyDatasetBundle",
    "FamilySampleBatch",
    "FamilyView",
    "FamilySource",
    "balanced_family_action_mse",
    "build_arg_parser",
    "evaluate_family_student",
    "estimate_family_resident_bytes",
    "estimate_compact_resident_bytes",
    "family_asset_weights",
    "fit_family_student",
    "load_family_sources",
    "main",
    "METRICS_FILENAME",
    "TRAINING_REPORT_FILENAME",
    "quality_episode_mask",
    "read_family_metadata",
    "reconstruct_history",
    "run_family_training",
]
