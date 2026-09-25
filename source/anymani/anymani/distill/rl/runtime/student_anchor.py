'Load and sample offline family-student anchors from frozen demonstration datasets.'

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import torch

from anymani.distill.il.family_student import FAMILY_STUDENT_ACTOR_ABI
from anymani.distill.models.palm_rotation_policy import PalmRotationActorObservation, PalmRotationGeometry


def _counter(value: object, label: str) -> int:
    'Handle counter.'

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer counter")
    result = int(value)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


class _SourceRowLruCache:
    'Contract for source row lru cache.'

    def __init__(self, capacity_bytes: int) -> None:
        if capacity_bytes < 0:
            raise ValueError("anchor source row cache capacity must be non-negative")
        self.capacity_bytes = int(capacity_bytes)
        self._entries: OrderedDict[tuple[int, str, str, int], tuple[np.ndarray, int]] = OrderedDict()
        self.bytes_used = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.bytes_loaded = 0
        self.load_seconds = 0.0

    @property
    def entry_count(self) -> int:
        'Handle entry count.'

        return len(self._entries)

    def get(
        self,
        key: tuple[int, str, str, int],
        loader: Callable[[], np.ndarray],
    ) -> np.ndarray:
        'Return get.'

        cached = self._entries.get(key)
        if cached is not None:
            self.hits += 1
            self._entries.move_to_end(key)
            return cached[0]

        self.misses += 1
        started = perf_counter()
        value = loader()
        self.load_seconds += perf_counter() - started
        if not isinstance(value, np.ndarray):
            raise TypeError("source row cache loader must return a numpy ndarray")
        size = int(value.nbytes)
        self.bytes_loaded += size
        if size == 0 or size > self.capacity_bytes:
            # Oversized rows are still returned for correctness but never retained beyond this call.
            return value
        while self.bytes_used + size > self.capacity_bytes and self._entries:
            _old_key, (_old_value, old_size) = self._entries.popitem(last=False)
            self.bytes_used -= old_size
            self.evictions += 1
        self._entries[key] = (value, size)
        self.bytes_used += size
        return value

    def clear(self) -> None:
        'Handle clear.'

        self._entries.clear()
        self.bytes_used = 0

    def state_dict(self) -> dict[str, int | float]:
        'Handle state dict.'

        return {
            "capacity_bytes": self.capacity_bytes,
            "bytes_used": self.bytes_used,
            "entry_count": self.entry_count,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "bytes_loaded": self.bytes_loaded,
            "load_seconds": self.load_seconds,
        }

    def load_telemetry(self, state: Mapping[str, object]) -> None:
        'Load telemetry.'

        if _counter(state.get("capacity_bytes", -1), "cache.capacity_bytes") != self.capacity_bytes:
            raise ValueError("anchor source row cache capacity disagrees")
        saved_bytes = _counter(state.get("bytes_used", 0), "cache.bytes_used")
        if saved_bytes > self.capacity_bytes:
            raise ValueError("anchor source row cache telemetry exceeds its capacity")
        _counter(state.get("entry_count", 0), "cache.entry_count")
        for name in ("hits", "misses", "evictions", "bytes_loaded"):
            value = _counter(state.get(name, -1), f"cache.{name}")
            setattr(self, name, value)
        load_seconds = state.get("load_seconds", 0.0)
        if isinstance(load_seconds, bool) or not isinstance(load_seconds, (int, float)):
            raise ValueError("cache.load_seconds must be numeric")
        if not math.isfinite(float(load_seconds)) or float(load_seconds) < 0.0:
            raise ValueError("cache.load_seconds must be finite and non-negative")
        self.load_seconds = float(load_seconds)
        # bytes_used/entry_count describe the intentionally empty post-resume cache and are not trusted.
        self.clear()


@dataclass(frozen=True)
class FamilyStudentAnchorBatch:
    'Contract for family student anchor batch.'

    observation: PalmRotationActorObservation
    geometry: PalmRotationGeometry
    joint_kinematics: torch.Tensor  # shapes [B,16,15]
    target: torch.Tensor  # shapes [B,16]
    joint_valid: torch.Tensor  # shapes [B,16]
    fk_target: torch.Tensor | None  # shapes [B,16,3]
    source_index: torch.Tensor  # shapes [B]
    sample_index: torch.Tensor  # shapes [B]
    env_index: torch.Tensor  # shapes [B]

    def __post_init__(self) -> None:
        'Validate the declared contract.'

        count = self.target.shape[0]
        expected = {
            "target": (count, 16),
            "joint_kinematics": (count, 16, 15),
            "joint_valid": (count, 16),
            "source_index": (count,),
            "sample_index": (count,),
            "env_index": (count,),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"anchor {name} shape {tuple(getattr(self, name).shape)} != expected {shape}")
        if self.observation.jnt_current.shape != (count, 16, 5):
            raise ValueError("anchor actor current must have shape [B,16,5]")
        if self.observation.jnt_history.shape != (count, 30, 16, 5):
            raise ValueError("anchor actor history must have shape [B,30,16,5]")
        if self.observation.jnt_limits.shape != (count, 16, 2):
            raise ValueError("anchor actor limits must have shape [B,16,2]")
        if self.geometry.tokens.shape != (count, 21, 128):
            raise ValueError("anchor geometry tokens must have shape [B,21,128]")
        if self.joint_valid.dtype != torch.bool or not torch.equal(self.joint_valid, self.observation.jnt_valid):
            raise ValueError("anchor joint_valid must be bool and equal Actor observation mask")
        if any(getattr(self, name).dtype != torch.int64 for name in ("source_index", "sample_index", "env_index")):
            raise ValueError("anchor provenance indices must be int64")
        if self.fk_target is not None and self.fk_target.shape != (count, 16, 3):
            raise ValueError("anchor FK target must have shape [B,16,3]")
        tensors = [self.joint_kinematics, self.target, self.geometry.tokens]
        if self.fk_target is not None:
            tensors.append(self.fk_target)
        if any(not bool(torch.isfinite(tensor).all().item()) for tensor in tensors):
            raise ValueError("anchor tensors must be finite")

    def to(self, device: torch.device | str) -> FamilyStudentAnchorBatch:
        'Handle to.'

        return FamilyStudentAnchorBatch(
            observation=PalmRotationActorObservation(
                self.observation.jnt_current.to(device),
                self.observation.jnt_history.to(device),
                self.observation.jnt_limits.to(device),
                self.observation.owner_contact.to(device),
                self.observation.jnt_valid.to(device),
                self.observation.tip_valid.to(device),
                self.observation.owner_valid.to(device),
            ),
            geometry=PalmRotationGeometry(
                self.geometry.tokens.to(device),
                self.geometry.owner_valid.to(device),
                self.geometry.shortest_path.to(device),
                self.geometry.parent_direction.to(device),
                self.geometry.child_direction.to(device),
            ),
            joint_kinematics=self.joint_kinematics.to(device),
            target=self.target.to(device),
            joint_valid=self.joint_valid.to(device),
            fk_target=None if self.fk_target is None else self.fk_target.to(device),
            source_index=self.source_index.to(device),
            sample_index=self.sample_index.to(device),
            env_index=self.env_index.to(device),
        )

    def select(self, selection: slice | torch.Tensor) -> FamilyStudentAnchorBatch:
        'Select the declared contract.'

        index = selection
        return FamilyStudentAnchorBatch(
            observation=PalmRotationActorObservation(
                self.observation.jnt_current[index],
                self.observation.jnt_history[index],
                self.observation.jnt_limits[index],
                self.observation.owner_contact[index],
                self.observation.jnt_valid[index],
                self.observation.tip_valid[index],
                self.observation.owner_valid[index],
            ),
            geometry=PalmRotationGeometry(
                self.geometry.tokens[index],
                self.geometry.owner_valid[index],
                self.geometry.shortest_path[index],
                self.geometry.parent_direction[index],
                self.geometry.child_direction[index],
            ),
            joint_kinematics=self.joint_kinematics[index],
            target=self.target[index],
            joint_valid=self.joint_valid[index],
            fk_target=None if self.fk_target is None else self.fk_target[index],
            source_index=self.source_index[index],
            sample_index=self.sample_index[index],
            env_index=self.env_index[index],
        )


@dataclass
class _AnchorSource:
    'Contract for anchor source.'

    path: Path
    family: str
    metadata: dict[str, object]
    handle: h5py.File
    env_asset_index: np.ndarray
    env_replica_index: np.ndarray
    train_env_mask: np.ndarray
    train_env_index: np.ndarray
    sample_steps: np.ndarray
    static: dict[str, np.ndarray]


def _sha256_file(path: Path) -> str:
    'Handle sha256 file.'

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_family_student_anchor_paths(lock_path: str | Path | None = None) -> tuple[Path, ...]:
    'Resolve family student anchor paths.'

    root = Path(__file__).resolve().parents[6]
    resolved_lock = (
        Path(lock_path).expanduser().resolve()
        if lock_path is not None
        else root
        / "logs/benchmarks/family_teacher_distillation/shared-student-20260913/artifacts/final-shared-student/configuration/dataset-lock.json"
    )
    if not resolved_lock.is_file():
        raise FileNotFoundError(f"family student anchor dataset lock is missing: {resolved_lock}")
    try:
        document = json.loads(resolved_lock.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read family student anchor dataset lock {resolved_lock}: {error}") from error
    sources = document.get("sources") if isinstance(document, Mapping) else None
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)) or not sources:
        raise ValueError("family student anchor dataset lock must contain a nonempty sources list")
    paths: list[Path] = []
    for source in sources:
        if not isinstance(source, Mapping) or not isinstance(source.get("path"), str):
            raise ValueError("family student anchor dataset lock source lacks a path")
        path = Path(str(source["path"])).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"family student anchor source is missing: {path}")
        paths.append(path)
    if len(set(paths)) != len(paths):
        raise ValueError("family student anchor dataset lock repeats a source path")
    return tuple(paths)


def resolve_family_student_anchor_source_hashes(lock_path: str | Path | None = None) -> dict[str, str]:
    'Resolve family student anchor source hashes.'

    root = Path(__file__).resolve().parents[6]
    resolved_lock = (
        Path(lock_path).expanduser().resolve()
        if lock_path is not None
        else root
        / "logs/benchmarks/family_teacher_distillation/shared-student-20260913/artifacts/final-shared-student/configuration/dataset-lock.json"
    )
    if not resolved_lock.is_file():
        raise FileNotFoundError(f"family student anchor dataset lock is missing: {resolved_lock}")
    try:
        document = json.loads(resolved_lock.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read family student anchor dataset lock {resolved_lock}: {error}") from error
    sources = document.get("sources") if isinstance(document, Mapping) else None
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)) or not sources:
        raise ValueError("family student anchor dataset lock must contain a nonempty sources list")
    result: dict[str, str] = {}
    for source in sources:
        if not isinstance(source, Mapping) or not isinstance(source.get("path"), str):
            raise ValueError("family student anchor dataset lock source lacks a path")
        digest = source.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("family student anchor dataset lock source lacks a 64-character SHA-256")
        path = Path(str(source["path"])).expanduser().resolve()
        result[str(path)] = digest
    return result


class FamilyStudentAnchorSampler:
    'Contract for family student anchor sampler.'

    _STATIC_NAMES = (
        "actor_jnt_limits",
        "jnt_valid",
        "tip_valid",
        "owner_valid",
        "joint_kinematics",
        "shortest_path",
        "parent_direction",
        "child_direction",
    )
    _STATIC_SHAPES = {
        "actor_jnt_limits": (16, 2),
        "jnt_valid": (16,),
        "tip_valid": (4,),
        "owner_valid": (21,),
        "joint_kinematics": (16, 15),
        "shortest_path": (21, 21),
        "parent_direction": (21, 21),
        "child_direction": (21, 21),
    }

    def __init__(
        self,
        dataset_paths: Sequence[str | Path] | None = None,
        *,
        batch_size: int = 2048,
        seed: int = 42,
        sample_row_block_size: int = 8,
        cache_capacity_bytes: int = 8 * 1024**3,
        train_replica_modulus: int = 4,
        validation_replica_remainder: int = 3,
        source_hashes: Mapping[str | Path, str] | None = None,
        compute_source_hash: bool = False,
    ) -> None:
        'Initialize the instance.'

        if batch_size < 1:
            raise ValueError("family student anchor batch_size must be positive")
        if sample_row_block_size < 1:
            raise ValueError("family student anchor sample_row_block_size must be positive")
        if train_replica_modulus < 2 or not 0 <= validation_replica_remainder < train_replica_modulus:
            raise ValueError("anchor train/validation replica split is invalid")
        paths = tuple(Path(path).expanduser().resolve() for path in (dataset_paths or resolve_family_student_anchor_paths()))
        if not paths:
            raise ValueError("family student anchor sampler requires at least one source")
        if len(set(paths)) != len(paths):
            raise ValueError("family student anchor sampler source paths must be unique")
        self.dataset_paths = paths
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.sample_row_block_size = int(sample_row_block_size)
        self.cache_capacity_bytes = _counter(cache_capacity_bytes, "cache_capacity_bytes")
        self.train_replica_modulus = int(train_replica_modulus)
        self.validation_replica_remainder = int(validation_replica_remainder)
        self.generator = torch.Generator(device="cpu").manual_seed(self.seed)
        self.sampled_batches = 0
        self.sampled_samples = 0
        self.last_sample_seconds = 0.0
        self.total_sample_seconds = 0.0
        self._sources: list[_AnchorSource] = []
        self._sample_row_counts: tuple[int, ...] = ()
        self._group_candidates: dict[tuple[str, str], np.ndarray] = {}
        self._group_order: tuple[tuple[str, str], ...] = ()
        self._family_group_indices: dict[str, tuple[int, ...]] = {}
        self._initial_history_cache: dict[int, np.ndarray] = {}
        self._source_sha_by_index: tuple[str, ...] = ()
        self._row_cache = _SourceRowLruCache(self.cache_capacity_bytes)
        self.source_hashes = {
            str(Path(key).expanduser().resolve()): str(value)
            for key, value in (source_hashes.items() if source_hashes is not None else ())
        }
        if any(len(value) != 64 for value in self.source_hashes.values()):
            raise ValueError("family student anchor source hashes must be 64-character SHA-256 strings")
        declared_source_hashes = dict(self.source_hashes)
        self._open_sources()
        # Any retained cache entry must be keyed by the actual source bytes, never a path/metadata surrogate.
        # `compute_source_hash=False` remains available for uncached shape/index canaries (`capacity=0`).
        verify_source_bytes = compute_source_hash or self.cache_capacity_bytes > 0
        if verify_source_bytes:
            try:
                expected_hashes = declared_source_hashes
                actual_hashes = {str(path): _sha256_file(path) for path in self.dataset_paths}
                if expected_hashes and expected_hashes != actual_hashes:
                    raise ValueError("family student anchor source bytes disagree with the declared SHA-256 lock")
                self.source_hashes = actual_hashes
            except BaseException:
                self.close()
                raise
        elif self.source_hashes and set(self.source_hashes) != {str(path) for path in self.dataset_paths}:
            raise ValueError("family student anchor source hashes must cover every dataset path")
        self._source_sha_by_index = tuple(self.source_hashes[str(path)] for path in self.dataset_paths)

    @staticmethod
    def _metadata(handle: h5py.File, path: Path) -> dict[str, object]:
        'Handle metadata.'

        raw = handle.attrs.get("metadata_json")
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            metadata = json.loads(str(raw))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}: family trajectory metadata_json is invalid: {error}") from error
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{path}: family trajectory metadata must be an object")
        artifact_type = handle.attrs.get("artifact_type")
        schema = handle.attrs.get("schema_version")
        if isinstance(artifact_type, bytes):
            artifact_type = artifact_type.decode("utf-8")
        if isinstance(schema, bytes):
            schema = schema.decode("utf-8")
        if artifact_type != "anymani.family_teacher_trajectory" or schema != "1.0.0":
            raise ValueError(f"{path}: unsupported family trajectory artifact/schema")
        completed = handle.attrs.get("completed")
        if isinstance(completed, np.bool_):
            completed = bool(completed)
        if completed is not True:
            raise ValueError(f"{path}: family trajectory is incomplete")
        abi = metadata.get("actor_abi")
        if not isinstance(abi, Mapping) or dict(abi) != FAMILY_STUDENT_ACTOR_ABI:
            raise ValueError(f"{path}: family trajectory actor ABI disagrees with student ABI")
        family = metadata.get("family")
        if not isinstance(family, str) or not family:
            raise ValueError(f"{path}: family trajectory family identity is missing")
        return dict(metadata)

    @staticmethod
    def _dataset(handle: h5py.File, path: str, source_path: Path) -> h5py.Dataset:
        'Handle dataset.'

        value = handle.get(path)
        if not isinstance(value, h5py.Dataset):
            raise ValueError(f"{source_path}: required dataset {path!r} is missing")
        return value

    def _open_sources(self) -> None:
        'Handle open sources.'

        candidate_chunks: dict[tuple[str, str], list[np.ndarray]] = {}
        try:
            for source_index, path in enumerate(self.dataset_paths):
                handle = h5py.File(path, mode="r")
                metadata = self._metadata(handle, path)
                env_asset = np.asarray(self._dataset(handle, "env_asset_index", path), dtype=np.int64)
                env_replica = np.asarray(self._dataset(handle, "env_replica_index", path), dtype=np.int64)
                quality_raw = np.asarray(self._dataset(handle, "final/quality_episode_mask", path))
                if (
                    quality_raw.ndim != 1
                    or quality_raw.shape != env_asset.shape
                    or quality_raw.dtype.kind not in "biuf"
                    or not np.isfinite(quality_raw).all()
                    or not np.isin(quality_raw, (0, 1)).all()
                ):
                    raise ValueError(f"{path}: final/quality_episode_mask must be finite exact 0/1 on env axis")
                quality = quality_raw.astype(bool, copy=False)
                if env_asset.ndim != 1 or env_replica.shape != env_asset.shape or quality.shape != env_asset.shape:
                    raise ValueError(f"{path}: env routing/quality axes disagree")
                train_env = quality & (env_replica % self.train_replica_modulus != self.validation_replica_remainder)
                sample_steps = np.asarray(self._dataset(handle, "samples/step_index", path), dtype=np.int64)
                active_dataset = self._dataset(handle, "frames/active", path)
                if active_dataset.ndim != 2 or active_dataset.shape[1] != env_asset.size:
                    raise ValueError(f"{path}: sample/frame active axes disagree")
                if (
                    sample_steps.ndim != 1
                    or sample_steps.size < 1
                    or np.any(sample_steps < 0)
                    or np.any(sample_steps >= active_dataset.shape[0])
                    or np.any(np.diff(sample_steps) <= 0)
                ):
                    raise ValueError(f"{path}: sample/frame active axes disagree")
                active_raw = np.asarray(active_dataset[sample_steps, :])
                if (
                    active_raw.dtype.kind not in "biuf"
                    or not np.isfinite(active_raw).all()
                    or not np.isin(active_raw, (0, 1)).all()
                ):
                    raise ValueError(f"{path}: frames/active must contain finite exact 0/1 values")
                active = active_raw.astype(bool, copy=False)
                static = {
                    name: np.asarray(self._dataset(handle, f"static/{name}", path)) for name in self._STATIC_NAMES
                }
                asset_count = static["jnt_valid"].shape[0]
                if any(value.shape[0] != asset_count for value in static.values()) or any(
                    tuple(value.shape[1:]) != self._STATIC_SHAPES[name] for name, value in static.items()
                ):
                    raise ValueError(f"{path}: static asset axes disagree")
                if np.any(env_asset < 0) or np.any(env_asset >= asset_count):
                    raise ValueError(f"{path}: env asset routing lies outside static asset axis")
                source = _AnchorSource(
                    path=path,
                    family=str(metadata["family"]),
                    metadata=metadata,
                    handle=handle,
                    env_asset_index=env_asset,
                    env_replica_index=env_replica,
                    train_env_mask=train_env,
                    train_env_index=np.flatnonzero(train_env).astype(np.int64, copy=False),
                    sample_steps=sample_steps,
                    static=static,
                )
                self._sources.append(source)
                for sample_index in range(sample_steps.size):
                    candidate_env = np.flatnonzero(active[sample_index] & train_env)
                    if candidate_env.size == 0:
                        continue
                    for asset_index in np.unique(env_asset[candidate_env]).tolist():
                        selected_env = candidate_env[env_asset[candidate_env] == asset_index]
                        records = np.column_stack(
                            (
                                np.full(selected_env.size, source_index, dtype=np.int64),
                                np.full(selected_env.size, sample_index, dtype=np.int64),
                                selected_env.astype(np.int64, copy=False),
                            )
                        )
                        asset_identity = self._asset_identity(metadata, int(asset_index))
                        key = (source.family, asset_identity)
                        candidate_chunks.setdefault(key, []).append(records)
        except BaseException:
            self.close()
            raise


        self._group_candidates = {
            key: np.concatenate(chunks, axis=0) for key, chunks in candidate_chunks.items()
        }
        self._group_order = tuple(sorted(self._group_candidates))
        family_indices: dict[str, list[int]] = {}
        for index, (family, _asset) in enumerate(self._group_order):
            family_indices.setdefault(family, []).append(index)
        self._family_group_indices = {family: tuple(indices) for family, indices in sorted(family_indices.items())}
        self._sample_row_counts = tuple(source.sample_steps.size for source in self._sources)
        if len(self._group_order) < 1:
            raise ValueError("family student anchor sampler has no active train split samples")
        if self.batch_size < len(self._group_order):
            raise ValueError(
                f"anchor batch_size {self.batch_size} is smaller than balanced family/asset groups "
                f"{len(self._group_order)}"
            )
        if not self.source_hashes:
            self.source_hashes = {
                str(source.path): self._metadata_identity_hash(source.metadata, source.path) for source in self._sources
            }

    @staticmethod
    def _metadata_identity_hash(metadata: Mapping[str, object], path: Path) -> str:
        'Handle metadata identity hash.'

        payload = json.dumps(
            {"path": str(path), "metadata": dict(metadata)}, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _asset_identity(metadata: Mapping[str, object], asset_index: int) -> str:
        'Handle asset identity.'

        assets = metadata.get("ordered_assets")
        if isinstance(assets, Sequence) and not isinstance(assets, (str, bytes)) and asset_index < len(assets):
            value = assets[asset_index]
            if isinstance(value, Mapping):
                for key in ("asset_id", "source_member_key", "asset_index"):
                    if key in value:
                        return str(value[key])
            return str(value)
        return f"asset_index_{asset_index}"

    @property
    def group_count(self) -> int:
        'Handle group count.'

        return len(self._group_order)

    @property
    def train_candidate_count(self) -> int:
        'Train candidate count.'

        return sum(int(records.shape[0]) for records in self._group_candidates.values())

    @property
    def source_data_hash(self) -> str:
        'Handle source data hash.'

        payload = json.dumps(
            {"paths": [str(path) for path in self.dataset_paths], "hashes": self.source_hashes},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def cache_stats(self) -> dict[str, int | float]:
        'Handle cache stats.'

        return self._row_cache.state_dict()

    def _sample_indices(self) -> np.ndarray:
        'Handle sample indices; shapes [B,source/sample/env].'

        group_count = len(self._group_order)


        families = tuple(self._family_group_indices)
        family_base, family_remainder = divmod(self.batch_size, len(families))
        family_order = torch.randperm(len(families), generator=self.generator).tolist()
        family_counts = np.full(len(families), family_base, dtype=np.int64)
        family_counts[np.asarray(family_order[:family_remainder], dtype=np.int64)] += 1
        counts = np.zeros(group_count, dtype=np.int64)
        for family_index, family in enumerate(families):
            groups = self._family_group_indices[family]
            group_base, group_remainder = divmod(int(family_counts[family_index]), len(groups))
            counts[np.asarray(groups, dtype=np.int64)] = group_base
            group_order = torch.randperm(len(groups), generator=self.generator).tolist()
            counts[np.asarray([groups[index] for index in group_order[:group_remainder]], dtype=np.int64)] += 1
        # shapes [1,2048,21,128]




        row_blocks = {
            source_index: np.asarray(
                torch.randperm(row_count, generator=self.generator)[: min(self.sample_row_block_size, row_count)]
                .numpy(),
                dtype=np.int64,
            )
            for source_index, row_count in enumerate(self._sample_row_counts)
        }
        selected: list[np.ndarray] = []
        for group_index, key in enumerate(self._group_order):
            candidates = self._group_candidates[key]
            allowed = np.zeros(candidates.shape[0], dtype=bool)
            for source_index, rows in row_blocks.items():
                source_mask = candidates[:, 0] == source_index
                if source_mask.any():
                    allowed[source_mask] = np.isin(candidates[source_mask, 1], rows)
            pool = candidates[allowed]
            if pool.shape[0] == 0:
                pool = candidates
            choices = torch.randint(pool.shape[0], (int(counts[group_index]),), generator=self.generator).numpy()
            selected.append(pool[choices])
        result = np.concatenate(selected, axis=0)
        shuffle = torch.randperm(result.shape[0], generator=self.generator).numpy()
        return result[shuffle]

    def _read_row_subset(
        self,
        source_index: int,
        source: _AnchorSource,
        dataset_path: str,
        dataset: h5py.Dataset,
        row: int,
        envs: np.ndarray,
    ) -> np.ndarray:
        'Read row subset.'

        if envs.ndim != 1 or envs.size == 0:
            raise ValueError("HDF5 row subset requires a non-empty one-dimensional environment index")
        unique_env, inverse = np.unique(envs.astype(np.int64, copy=False), return_inverse=True)
        train_env_index = source.train_env_index
        local_env = np.searchsorted(train_env_index, unique_env)
        if (
            np.any(local_env >= train_env_index.size)
            or not np.array_equal(train_env_index[local_env], unique_env)
        ):
            raise ValueError("anchor row requested an environment outside the train split")
        cache_key = (int(source_index), self._source_sha_by_index[source_index], dataset_path, int(row))
        values = self._row_cache.get(
            cache_key,
            # h5py's fancy env indexing would decompress the same chunk through thousands of scalar
            # selections; read one contiguous compressed row, then retain only train envs in the LRU.
            lambda: np.asarray(dataset[int(row), :])[train_env_index],
        )
        return values[local_env[inverse]]

    def _paired_values(
        self,
        source_index: int,
        dataset_path: str,
        dataset: h5py.Dataset,
        rows: np.ndarray,
        envs: np.ndarray,
    ) -> np.ndarray:
        'Handle paired values; shapes [row_i,env_i].'

        result = np.empty((rows.size, *dataset.shape[2:]), dtype=dataset.dtype)
        for row in np.unique(rows).tolist():
            positions = np.flatnonzero(rows == row)
            source = self._sources[source_index]
            result[positions] = self._read_row_subset(
                source_index,
                source,
                dataset_path,
                dataset,
                int(row),
                envs[positions],
            )
        return result

    def _initial_history(self, source_index: int, source: _AnchorSource) -> np.ndarray:
        'Handle initial history; shapes [Q,30,16,5].'

        cached = self._initial_history_cache.get(source_index)
        if cached is None:
            dataset = self._dataset(source.handle, "initial_history", source.path)
            cached = np.asarray(dataset[:], dtype=np.float32)
            self._initial_history_cache[source_index] = cached
        return cached

    def _read_history(
        self,
        source_index: int,
        source: _AnchorSource,
        steps: np.ndarray,
        envs: np.ndarray,
    ) -> np.ndarray:
        'Read history; shapes [B,30,16,5].'

        frame_dataset = self._dataset(source.handle, "frames/jnt_current", source.path)
        initial = self._initial_history(source_index, source)
        history = np.empty((steps.size, 30, 16, 5), dtype=np.float32)
        for step in np.unique(steps).tolist():
            positions = np.flatnonzero(steps == step)
            unique_env, inverse = np.unique(envs[positions], return_inverse=True)
            if step == 0:
                values = initial[unique_env]
            elif step < 30:
                h0 = initial[unique_env, int(step) :]
                future = np.stack(
                    [
                        self._read_row_subset(
                            source_index,
                            source,
                            "frames/jnt_current",
                            frame_dataset,
                            frame,
                            unique_env,
                        )
                        for frame in range(1, int(step) + 1)
                    ],
                    axis=1,
                )
                values = np.concatenate((h0, future), axis=1)
            else:
                values = np.stack(
                    [
                        self._read_row_subset(
                            source_index,
                            source,
                            "frames/jnt_current",
                            frame_dataset,
                            frame,
                            unique_env,
                        )
                        for frame in range(int(step) - 29, int(step) + 1)
                    ],
                    axis=1,
                )
            history[positions] = values[inverse]
        return history

    def sample(self) -> FamilyStudentAnchorBatch:
        'Handle sample.'

        started = perf_counter()
        records = self._sample_indices()
        source_ids, sample_ids, env_ids = records.T
        batch = records.shape[0]
        current = np.empty((batch, 16, 5), dtype=np.float32)
        contact = np.empty((batch, 21, 1), dtype=np.float32)
        history = np.empty((batch, 30, 16, 5), dtype=np.float32)
        target = np.empty((batch, 16), dtype=np.float32)
        fk_target = np.empty((batch, 16, 3), dtype=np.float32)
        limits = np.empty((batch, 16, 2), dtype=np.float32)
        joint_valid = np.empty((batch, 16), dtype=bool)
        tip_valid = np.empty((batch, 4), dtype=bool)
        owner_valid = np.empty((batch, 21), dtype=bool)
        kinematics = np.empty((batch, 16, 15), dtype=np.float32)
        tokens = np.empty((batch, 21, 128), dtype=np.float32)
        shortest = np.empty((batch, 21, 21), dtype=np.int64)
        parent = np.empty_like(shortest)
        child = np.empty_like(shortest)
        for source_index in np.unique(source_ids).tolist():
            positions = np.flatnonzero(source_ids == source_index)
            source = self._sources[int(source_index)]
            rows = sample_ids[positions]
            envs = env_ids[positions]
            steps = source.sample_steps[rows]
            current_dataset = self._dataset(source.handle, "frames/jnt_current", source.path)
            contact_dataset = self._dataset(source.handle, "frames/owner_contact", source.path)
            teacher_dataset = self._dataset(source.handle, "samples/teacher_mean", source.path)
            geometry_dataset = self._dataset(source.handle, "samples/geometry_tokens", source.path)
            fk_dataset = self._dataset(source.handle, "samples/joint_origin_fk", source.path)
            current[positions] = self._paired_values(source_index, "frames/jnt_current", current_dataset, steps, envs)
            contact[positions] = self._paired_values(source_index, "frames/owner_contact", contact_dataset, steps, envs)
            target[positions] = self._paired_values(source_index, "samples/teacher_mean", teacher_dataset, rows, envs)
            tokens[positions] = self._paired_values(source_index, "samples/geometry_tokens", geometry_dataset, rows, envs)
            fk_target[positions] = self._paired_values(source_index, "samples/joint_origin_fk", fk_dataset, rows, envs)
            history[positions] = self._read_history(int(source_index), source, steps, envs)
            asset_ids = source.env_asset_index[envs]
            limits[positions] = source.static["actor_jnt_limits"][asset_ids]
            joint_valid[positions] = source.static["jnt_valid"][asset_ids]
            tip_valid[positions] = source.static["tip_valid"][asset_ids]
            owner_valid[positions] = source.static["owner_valid"][asset_ids]
            kinematics[positions] = source.static["joint_kinematics"][asset_ids]
            shortest[positions] = source.static["shortest_path"][asset_ids]
            parent[positions] = source.static["parent_direction"][asset_ids]
            child[positions] = source.static["child_direction"][asset_ids]
        owner_expected = np.concatenate((np.ones((batch, 1), dtype=bool), joint_valid, tip_valid), axis=1)
        if not np.array_equal(owner_valid, owner_expected):
            raise ValueError("anchor static owner_valid disagrees with PALM/JOINT/TIP masks")
        if (
            not np.isfinite(current).all()
            or not np.isfinite(history).all()
            or not np.isfinite(contact).all()
            or not np.isfinite(target).all()
            or not np.isfinite(tokens).all()
            or not np.isfinite(fk_target).all()
        ):
            raise ValueError("anchor source returned non-finite actor state or teacher target")
        observation = PalmRotationActorObservation(
            torch.from_numpy(current),
            torch.from_numpy(history),
            torch.from_numpy(limits),
            torch.from_numpy(contact),
            torch.from_numpy(joint_valid),
            torch.from_numpy(tip_valid),
            torch.from_numpy(owner_valid),
        )
        geometry = PalmRotationGeometry(
            torch.from_numpy(tokens),
            torch.from_numpy(owner_valid),
            torch.from_numpy(shortest),
            torch.from_numpy(parent),
            torch.from_numpy(child),
        )
        self.sampled_batches += 1
        self.sampled_samples += batch
        self.last_sample_seconds = perf_counter() - started
        self.total_sample_seconds += self.last_sample_seconds
        return FamilyStudentAnchorBatch(
            observation=observation,
            geometry=geometry,
            joint_kinematics=torch.from_numpy(kinematics),
            target=torch.from_numpy(target),
            joint_valid=torch.from_numpy(joint_valid),
            fk_target=torch.from_numpy(fk_target),
            source_index=torch.from_numpy(source_ids.copy()),
            sample_index=torch.from_numpy(sample_ids.copy()),
            env_index=torch.from_numpy(env_ids.copy()),
        )

    def state_dict(self) -> dict[str, object]:
        'Handle state dict.'

        return {
            "schema_version": "1.0.0",
            "seed": self.seed,
            "batch_size": self.batch_size,
            "sample_row_block_size": self.sample_row_block_size,
            "train_replica_modulus": self.train_replica_modulus,
            "validation_replica_remainder": self.validation_replica_remainder,
            "dataset_paths": [str(path) for path in self.dataset_paths],
            "source_hashes": dict(self.source_hashes),
            "source_data_hash": self.source_data_hash,
            "group_keys": [list(key) for key in self._group_order],
            "train_candidate_count": self.train_candidate_count,
            "sampled_batches": self.sampled_batches,
            "sampled_samples": self.sampled_samples,
            "sampling": {
                "last_seconds": self.last_sample_seconds,
                "total_seconds": self.total_sample_seconds,
            },
            "cache": self._row_cache.state_dict(),
            "generator_state": self.generator.get_state().clone(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        'Load state dict.'

        if state.get("schema_version") != "1.0.0":
            raise ValueError("family student anchor sampler schema mismatch")
        if _counter(state.get("batch_size", -1), "batch_size") != self.batch_size:
            raise ValueError("family student anchor sampler batch size disagrees")
        if _counter(state.get("seed", -1), "seed") != self.seed:
            raise ValueError("family student anchor sampler seed disagrees")
        if _counter(state.get("sample_row_block_size", -1), "sample_row_block_size") != self.sample_row_block_size:
            raise ValueError("family student anchor sampler sample row block size disagrees")
        if _counter(state.get("train_replica_modulus", -1), "train_replica_modulus") != self.train_replica_modulus:
            raise ValueError("family student anchor sampler train split disagrees")
        if _counter(state.get("validation_replica_remainder", -1), "validation_replica_remainder") != self.validation_replica_remainder:
            raise ValueError("family student anchor sampler validation split disagrees")
        dataset_paths = state.get("dataset_paths", ())
        if not isinstance(dataset_paths, Sequence) or isinstance(dataset_paths, (str, bytes)):
            raise ValueError("family student anchor sampler dataset paths are malformed")
        if tuple(str(path) for path in dataset_paths) != tuple(str(path) for path in self.dataset_paths):
            raise ValueError("family student anchor sampler dataset paths disagree")
        saved_hashes = state.get("source_hashes", {})
        if not isinstance(saved_hashes, Mapping):
            raise ValueError("family student anchor sampler source hashes are malformed")
        normalized_hashes = {
            str(Path(key).expanduser().resolve()): str(value) for key, value in saved_hashes.items()
        }
        if normalized_hashes != self.source_hashes:
            raise ValueError("family student anchor sampler source hashes disagree")
        if state.get("source_data_hash") != self.source_data_hash:
            raise ValueError("family student anchor sampler source data identity disagrees")
        group_keys_value = state.get("group_keys", ())
        if not isinstance(group_keys_value, Sequence) or isinstance(group_keys_value, (str, bytes)):
            raise ValueError("family student anchor sampler group keys are malformed")
        group_keys = tuple(tuple(str(item) for item in key) for key in group_keys_value)
        if group_keys != self._group_order:
            raise ValueError("family student anchor sampler family/asset groups disagree")
        if _counter(state.get("train_candidate_count", -1), "train_candidate_count") != self.train_candidate_count:
            raise ValueError("family student anchor sampler train candidate count disagrees")
        generator_state = state.get("generator_state")
        if not isinstance(generator_state, torch.Tensor):
            raise ValueError("family student anchor sampler generator state is missing")
        self.generator.set_state(generator_state.detach().cpu())
        sampled_batches = _counter(state.get("sampled_batches", 0), "sampled_batches")
        sampled_samples = _counter(state.get("sampled_samples", 0), "sampled_samples")
        self.sampled_batches = sampled_batches
        self.sampled_samples = sampled_samples
        sampling_state = state.get("sampling", {})
        if not isinstance(sampling_state, Mapping):
            raise ValueError("family student anchor sampler sampling telemetry is malformed")
        last_seconds = sampling_state.get("last_seconds", 0.0)
        total_seconds = sampling_state.get("total_seconds", 0.0)
        for name, value in (("last_seconds", last_seconds), ("total_seconds", total_seconds)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"sampling.{name} must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"sampling.{name} must be finite and non-negative")
        self.last_sample_seconds = float(last_seconds)
        self.total_sample_seconds = float(total_seconds)
        cache_state = state.get("cache")
        if cache_state is not None:
            if not isinstance(cache_state, Mapping):
                raise ValueError("family student anchor sampler cache telemetry is malformed")
            self._row_cache.load_telemetry(cache_state)

    def close(self) -> None:
        'Close the declared contract.'

        for source in self._sources:
            with suppress(OSError, RuntimeError):
                source.handle.close()
        self._sources.clear()
        self._row_cache.clear()
        self._initial_history_cache.clear()

    def __enter__(self) -> FamilyStudentAnchorSampler:
        'Handle enter.'

        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        'Handle exit.'

        del exc_type, exc_value, traceback
        self.close()


__all__ = [
    "FamilyStudentAnchorBatch",
    "FamilyStudentAnchorSampler",
    "resolve_family_student_anchor_paths",
    "resolve_family_student_anchor_source_hashes",
]
