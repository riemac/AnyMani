"""Teacher trajectory schema and CPU readers. Record 20 Hz joint frames and initial History30, cache supervision every four steps, and keep whole trajectories in one split. Joint observations use 16 padded slots; geometry uses 21 owner tokens of width 128."""

from __future__ import annotations
import json
import operator
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, Final, cast
import h5py
import numpy as np

ARTIFACT_TYPE: Final[str] = "anymani.family_teacher_trajectory"
SCHEMA_VERSION: Final[str] = "1.0.0"
HISTORY_LENGTH: Final[int] = 30
JOINT_COUNT: Final[int] = 16
JOINT_FEATURES: Final[int] = 5
OWNER_COUNT: Final[int] = 21
TIP_COUNT: Final[int] = 4
GEOMETRY_TOKEN_WIDTH: Final[int] = 128
JOINT_KINEMATICS_WIDTH: Final[int] = 15
CONTROL_FREQUENCY_HZ: Final[float] = 20.0
QUALITY_HORIZON_STEPS: Final[int] = 600
DURATION_ATOL_S: Final[float] = 0.001
_REQUIRED_STATIC_SHAPES: Final[dict[str, tuple[int, ...]]] = {
    "actor_jnt_limits": (JOINT_COUNT, 2),
    "jnt_valid": (JOINT_COUNT,),
    "tip_valid": (TIP_COUNT,),
    "owner_valid": (OWNER_COUNT,),
    "shortest_path": (OWNER_COUNT, OWNER_COUNT),
    "parent_direction": (OWNER_COUNT, OWNER_COUNT),
    "child_direction": (OWNER_COUNT, OWNER_COUNT),
    "joint_kinematics": (JOINT_COUNT, JOINT_KINEMATICS_WIDTH),
}
_ACTION_LOW: Final[float] = -1.0
_ACTION_HIGH: Final[float] = 1.0
_MASK_LOW: Final[float] = 0.0
_MASK_HIGH: Final[float] = 1.0
_RANGE_EPS: Final[float] = 1e-06
_CANONICAL_ACTOR_ABI: Final[dict[str, object]] = {
    "arm": "direct_token",
    "history_encoder": "tcn",
    "history_length": HISTORY_LENGTH,
    "joint_count": JOINT_COUNT,
    "owner_count": OWNER_COUNT,
    "geometry_width": GEOMETRY_TOKEN_WIDTH,
    "actor_contact": "tip-only-binary",
    "phase_clock_enabled": False,
    "joint_kinematics_width": JOINT_KINEMATICS_WIDTH,
}


class FamilyDatasetError(ValueError):
    """Invalid or inconsistent family-demonstration data."""


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    raise TypeError(f"metadata value {type(value).__name__} is not JSON serializable")


def _json_roundtrip(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, default=_json_default)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise FamilyDatasetError(f"metadata must be strict JSON: {error}") from error
    if not isinstance(decoded, dict):
        raise FamilyDatasetError("metadata JSON root must be an object")
    return decoded


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FamilyDatasetError(f"metadata.{name} must be a non-empty string")
    return value


def _validate_metadata(metadata: Mapping[str, Any], asset_count: int) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    normalized = _json_roundtrip(metadata)
    required = {
        "family",
        "teacher_checkpoint_sha256",
        "cohort_sha256",
        "n040_sha256",
        "ordered_assets",
        "protocol",
        "actor_abi",
    }
    missing = sorted(required.difference(normalized))
    if missing:
        raise FamilyDatasetError(f"metadata is missing canonical fields: {', '.join(missing)}")
    family = _nonempty_string(normalized["family"], "family")
    checkpoint_sha = _nonempty_string(normalized["teacher_checkpoint_sha256"], "teacher_checkpoint_sha256")
    cohort_sha = _nonempty_string(normalized["cohort_sha256"], "cohort_sha256")
    n040_sha = _nonempty_string(normalized["n040_sha256"], "n040_sha256")
    identities = normalized["ordered_assets"]
    if isinstance(identities, (str, bytes)) or not isinstance(identities, Sequence):
        raise FamilyDatasetError("metadata.ordered_assets must be an ordered sequence")
    if len(identities) != asset_count:
        raise FamilyDatasetError(
            f"metadata.ordered_assets length {len(identities)} disagrees with static asset count {asset_count}"
        )
    protocol = normalized["protocol"]
    if not isinstance(protocol, Mapping):
        raise FamilyDatasetError("metadata.protocol must be a mapping")
    if "action_mode" not in protocol or "action_seed" not in protocol:
        raise FamilyDatasetError("metadata.protocol must contain action_mode and action_seed")
    action_mode = _nonempty_string(protocol["action_mode"], "protocol.action_mode")
    if action_mode not in {"mean", "sample"}:
        raise FamilyDatasetError("metadata.protocol.action_mode must be 'mean' or 'sample'")
    action_seed = protocol["action_seed"]
    if action_mode == "mean":
        if action_seed is not None:
            raise FamilyDatasetError("metadata.protocol.action_seed must be null for mean mode")
    else:
        if isinstance(action_seed, (bool, np.bool_)):
            raise FamilyDatasetError("metadata.protocol.action_seed must not be bool")
        try:
            operator.index(action_seed)
        except (TypeError, ValueError) as error:
            raise FamilyDatasetError("metadata.protocol.action_seed must be an integer for sample mode") from error
    actor_abi = normalized["actor_abi"]
    if not isinstance(actor_abi, Mapping):
        raise FamilyDatasetError("metadata.actor_abi must be a mapping")
    if dict(actor_abi) != _CANONICAL_ACTOR_ABI:
        raise FamilyDatasetError(
            f"metadata.actor_abi disagrees with canonical ABI {_CANONICAL_ACTOR_ABI!r}; got {dict(actor_abi)!r}"
        )
    normalized["family"] = family
    normalized["teacher_checkpoint_sha256"] = checkpoint_sha
    normalized["cohort_sha256"] = cohort_sha
    normalized["n040_sha256"] = n040_sha
    normalized["ordered_assets"] = list(identities)
    normalized["protocol"] = dict(protocol)
    normalized["actor_abi"] = dict(actor_abi)
    return normalized


def _array(value: Any, name: str) -> np.ndarray:
    if np.ma.isMaskedArray(value):
        raise FamilyDatasetError(f"{name} must be an unmasked numpy array")
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise FamilyDatasetError(f"{name} must be array-like") from error
    return array


def _float_array(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    array = _array(value, name)
    if array.shape != shape:
        raise FamilyDatasetError(f"{name} shape {array.shape} != expected {shape}")
    if array.dtype.kind != "f":
        raise FamilyDatasetError(f"{name} must be floating with shape {shape}, got dtype {array.dtype}")
    if not bool(np.isfinite(array).all()):
        raise FamilyDatasetError(f"{name} must contain finite values")
    return array.astype(np.float32, copy=False)


def _mask_array(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    array = _array(value, name)
    if array.shape != shape:
        raise FamilyDatasetError(f"{name} shape {array.shape} != expected {shape}")
    if array.dtype.kind not in "biuf":
        raise FamilyDatasetError(f"{name} must be bool/integer/float mask, got dtype {array.dtype}")
    if array.dtype.kind == "f" and (not bool(np.isfinite(array).all())):
        raise FamilyDatasetError(f"{name} must contain finite values")
    if np.any(array < _MASK_LOW) or np.any(array > _MASK_HIGH):
        raise FamilyDatasetError(f"{name} values must lie in [0,1]")
    if not bool(np.isin(array, (0, 1)).all()):
        raise FamilyDatasetError(f"{name} values must be exactly 0 or 1")
    return array.astype(np.float32, copy=False)


def _bool_mask(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    return _mask_array(value, name, shape).astype(bool, copy=False)


def _index_vector(value: Any, name: str, length: int) -> np.ndarray:
    array = _array(value, name)
    if array.shape != (length,):
        raise FamilyDatasetError(f"{name} shape {array.shape} != expected {(length,)}")
    if array.dtype.kind not in "iu" or array.dtype.kind == "b":
        raise FamilyDatasetError(f"{name} must be an integer vector")
    if np.any(array < 0):
        raise FamilyDatasetError(f"{name} must contain nonnegative ids")
    return array.astype(np.int64, copy=False)


def _scalar_step(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise FamilyDatasetError("step must be an integer, not bool")
    try:
        step = operator.index(value)
    except (TypeError, ValueError) as error:
        raise FamilyDatasetError("step must be an integer") from error
    if step < 0:
        raise FamilyDatasetError("step must be nonnegative")
    return int(step)


def _validate_range(array: np.ndarray, name: str, low: float, high: float) -> None:
    if np.any(array < low - _RANGE_EPS) or np.any(array > high + _RANGE_EPS):
        raise FamilyDatasetError(f"{name} values must lie in [{low},{high}]")


def _exact_zero(values: np.ndarray, name: str) -> None:
    if values.size and np.any(values != 0.0):
        raise FamilyDatasetError(f"{name} ghost/padding values must be exactly zero")


def _validate_static(static: Mapping[str, Any], asset_count: int) -> dict[str, np.ndarray]:
    if not isinstance(static, Mapping):
        raise TypeError("static must be a mapping of named arrays")
    missing = sorted(set(_REQUIRED_STATIC_SHAPES).difference(static))
    if missing:
        raise FamilyDatasetError(f"static is missing required fields: {', '.join(missing)}")
    normalized: dict[str, np.ndarray] = {}
    for name, suffix_shape in _REQUIRED_STATIC_SHAPES.items():
        expected = (asset_count, *suffix_shape)
        if name in {"jnt_valid", "tip_valid", "owner_valid"}:
            normalized[name] = _bool_mask(static[name], f"static.{name}", expected)
        elif name in {"actor_jnt_limits", "joint_kinematics"}:
            normalized[name] = _float_array(static[name], f"static.{name}", expected)
        else:
            array = _array(static[name], f"static.{name}")
            if array.shape != expected or array.dtype.kind not in "biuf":
                raise FamilyDatasetError(
                    f"static.{name} must be numeric with shape {expected}, got {array.shape}/{array.dtype}"
                )
            if array.dtype.kind == "f":
                if not bool(np.isfinite(array).all()):
                    raise FamilyDatasetError(f"static.{name} must contain finite values")
                normalized[name] = array.astype(np.float32, copy=False)
            else:
                normalized[name] = array.copy()
    limits = normalized["actor_jnt_limits"]
    if np.any(limits[..., 0] > limits[..., 1]):
        raise FamilyDatasetError("static.actor_jnt_limits lower bound must not exceed upper bound")
    expected_owner_valid = np.concatenate(
        (np.ones((asset_count, 1), dtype=bool), normalized["jnt_valid"], normalized["tip_valid"]), axis=1
    )
    if not np.array_equal(normalized["owner_valid"], expected_owner_valid):
        raise FamilyDatasetError("static.owner_valid must equal PALM/JOINT/TIP validity concatenation")
    for name, value in static.items():
        if name in normalized:
            continue
        if not isinstance(name, str) or not name or "/" in name:
            raise FamilyDatasetError(f"static field name {name!r} is not a safe HDF5 dataset name")
        array = _array(value, f"static.{name}")
        if array.ndim < 1 or array.shape[0] != asset_count:
            raise FamilyDatasetError(f"static.{name} first dimension {array.shape[:1]} != asset count {asset_count}")
        if array.dtype.kind == "f":
            if not bool(np.isfinite(array).all()):
                raise FamilyDatasetError(f"static.{name} must contain finite values")
            normalized[name] = array.astype(np.float32, copy=False)
        elif array.dtype.kind in "biu":
            normalized[name] = array.copy()
        else:
            raise FamilyDatasetError(f"static.{name} must be numeric")
    return normalized


def _write_compressed(group: h5py.Group, name: str, data: np.ndarray) -> h5py.Dataset:
    try:
        return group.create_dataset(name, data=data, dtype=data.dtype, compression="lzf", shuffle=True)
    except (RuntimeError, ValueError):
        return group.create_dataset(
            name, data=data, dtype=data.dtype, compression="gzip", compression_opts=4, shuffle=True
        )


def _create_stream_dataset(
    group: h5py.Group, name: str, suffix_shape: tuple[int, ...], dtype: np.dtype[Any]
) -> h5py.Dataset:
    shape = (0, *suffix_shape)
    maxshape = (None, *suffix_shape)
    chunks = (1, *suffix_shape)
    try:
        return group.create_dataset(
            name, shape=shape, maxshape=maxshape, dtype=dtype, chunks=chunks, compression="lzf", shuffle=True
        )
    except (RuntimeError, ValueError):
        return group.create_dataset(
            name,
            shape=shape,
            maxshape=maxshape,
            dtype=dtype,
            chunks=chunks,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )


def _resize_and_write(dataset: h5py.Dataset, index: int, value: Any) -> None:
    dataset.resize((index + 1, *dataset.shape[1:]))
    dataset[index] = value


def _required_group(parent: h5py.Group | h5py.File, name: str, where: str) -> h5py.Group:
    value = parent.get(name)
    if not isinstance(value, h5py.Group):
        raise RuntimeError(f"{where} lacks required group {name!r}")
    return cast(h5py.Group, value)


def _required_dataset(parent: h5py.Group | h5py.File, name: str, where: str) -> h5py.Dataset:
    value = parent.get(name)
    if not isinstance(value, h5py.Dataset):
        raise RuntimeError(f"{where} lacks required dataset {name!r}")
    return cast(h5py.Dataset, value)


def _store_metadata(file: h5py.File, metadata: Mapping[str, Any]) -> None:
    encoded = json.dumps(metadata, ensure_ascii=False, allow_nan=False, default=_json_default)
    file.attrs["metadata_json"] = encoded
    group = file.require_group("metadata")
    group.attrs["json"] = encoded
    for key, value in metadata.items():
        if not isinstance(key, str) or not key or "/" in key:
            continue
        try:
            if isinstance(value, (str, int, float, bool)):
                group.attrs[key] = value
                if key not in {"artifact_type", "schema_version"}:
                    file.attrs[key] = value
            else:
                group.attrs[key] = json.dumps(value, ensure_ascii=False, allow_nan=False, default=_json_default)
        except (TypeError, ValueError):
            continue


def _validate_env_mapping(
    env_asset_index: Any, env_replica_index: Any, static_asset_count: int
) -> tuple[np.ndarray, np.ndarray]:
    asset_array = _array(env_asset_index, "env_asset_index")
    if asset_array.ndim != 1 or asset_array.size < 1:
        raise FamilyDatasetError("env_asset_index must be a nonempty one-dimensional vector")
    env_count = int(asset_array.shape[0])
    assets = _index_vector(asset_array, "env_asset_index", env_count)
    replicas = _index_vector(env_replica_index, "env_replica_index", env_count)
    if np.any(assets >= static_asset_count):
        raise FamilyDatasetError(f"env_asset_index contains id >= static asset count {static_asset_count}")
    return (assets, replicas)


def _validate_initial_history(value: Any, env_count: int, joint_valid_by_env: np.ndarray) -> np.ndarray:
    history = _float_array(value, "initial_history", (env_count, HISTORY_LENGTH, JOINT_COUNT, JOINT_FEATURES))
    invalid_joint = np.broadcast_to((~joint_valid_by_env)[:, None, :, None], history.shape)
    _exact_zero(history[invalid_joint], "initial_history")
    return history


def _validate_summary_array(value: Any, name: str, env_count: int) -> np.ndarray:
    array = _array(value, f"trajectory_summary.{name}")
    if array.shape != (env_count,):
        raise FamilyDatasetError(f"trajectory_summary.{name} shape {array.shape} != expected {(env_count,)}")
    if array.dtype.kind == "f":
        if not bool(np.isfinite(array).all()):
            raise FamilyDatasetError(f"trajectory_summary.{name} must contain finite values")
        return array.astype(np.float32, copy=False)
    if array.dtype.kind in "biu":
        return array.copy()
    raise FamilyDatasetError(f"trajectory_summary.{name} must be numeric")


def _validated_terminal_flags(value: Any, name: str, env_count: int) -> np.ndarray:
    return _bool_mask(value, f"trajectory_summary.{name}", (env_count,))


class FamilyTrajectoryWriter:
    """Append bounded trajectory chunks and verify their static and temporal contracts."""

    def __init__(
        self,
        path: str | Path,
        metadata: dict[str, Any],
        *,
        steps: int,
        env_asset_index: np.ndarray,
        env_replica_index: np.ndarray,
        static: dict[str, np.ndarray],
        initial_history: np.ndarray,
        sample_stride: int = 4,
    ) -> None:
        try:
            expected_steps = operator.index(steps)
        except (TypeError, ValueError) as error:
            raise FamilyDatasetError("steps must be an integer") from error
        if isinstance(steps, (bool, np.bool_)) or expected_steps < 1:
            raise FamilyDatasetError("steps must be a positive integer")
        try:
            stride = operator.index(sample_stride)
        except (TypeError, ValueError) as error:
            raise FamilyDatasetError("sample_stride must be an integer") from error
        if isinstance(sample_stride, (bool, np.bool_)) or stride < 1:
            raise FamilyDatasetError("sample_stride must be a positive integer")
        raw_static = (
            _array(static.get("actor_jnt_limits"), "static.actor_jnt_limits")
            if isinstance(static, Mapping) and "actor_jnt_limits" in static
            else None
        )
        if raw_static is None or raw_static.ndim != 3 or raw_static.shape[1:] != (JOINT_COUNT, 2):
            raise FamilyDatasetError("static.actor_jnt_limits must have shape [A,16,2] to define asset count")
        asset_count = int(raw_static.shape[0])
        if asset_count < 1:
            raise FamilyDatasetError("static asset axis A must be positive")
        assets, replicas = _validate_env_mapping(env_asset_index, env_replica_index, asset_count)
        env_count = int(assets.size)
        normalized_static = _validate_static(static, asset_count)
        joint_valid_by_env = normalized_static["jnt_valid"][assets]
        owner_valid_by_env = normalized_static["owner_valid"][assets]
        history = _validate_initial_history(initial_history, env_count, joint_valid_by_env)
        normalized_metadata = _validate_metadata(metadata, asset_count)
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(f"family trajectory output already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            handle = h5py.File(destination, mode="x")
        except FileExistsError:
            raise
        except OSError as error:
            raise FamilyDatasetError(f"cannot create family trajectory output {destination}: {error}") from error
        self.path = destination
        self._file = handle
        self._closed = False
        self._finalized = False
        self._expected_steps = int(expected_steps)
        self._sample_stride = int(stride)
        self._env_count = env_count
        self._asset_count = asset_count
        self._env_asset_index = assets.copy()
        self._env_replica_index = replicas.copy()
        self._static = {name: np.array(value, copy=True) for name, value in normalized_static.items()}
        self._initial_history = np.array(history, copy=True)
        self._joint_valid_by_env = np.array(joint_valid_by_env, copy=True)
        self._owner_valid_by_env = np.array(owner_valid_by_env, copy=True)
        self._history_state = np.array(history, copy=True)
        self._active_previous = np.ones(env_count, dtype=bool)
        self._have_previous_active = False
        self._written_steps = 0
        self._sample_count = 0
        self._metadata = dict(normalized_metadata)
        self._metadata.update(
            {
                "completed": False,
                "incomplete": True,
                "expected_steps": self._expected_steps,
                "nominal_steps": self._expected_steps,
                "steps": 0,
                "recorded_steps": 0,
                "sample_count": 0,
                "sample_stride": self._sample_stride,
            }
        )
        try:
            self._file.attrs["artifact_type"] = ARTIFACT_TYPE
            self._file.attrs["schema_version"] = SCHEMA_VERSION
            self._file.attrs["completed"] = False
            self._file.attrs["incomplete"] = True
            self._file.attrs["expected_steps"] = self._expected_steps
            self._file.attrs["steps"] = 0
            self._file.attrs["sample_stride"] = self._sample_stride
            self._file.attrs["env_count"] = self._env_count
            self._file.attrs["asset_count"] = self._asset_count
            self._file.attrs["history_semantics"] = "H0_includes_current0_oldest_to_latest"
            _store_metadata(self._file, self._metadata)
            _write_compressed(self._file, "env_asset_index", self._env_asset_index)
            _write_compressed(self._file, "env_replica_index", self._env_replica_index)
            _write_compressed(self._file, "initial_history", self._initial_history)
            static_group = self._file.create_group("static")
            for name, value in self._static.items():
                _write_compressed(static_group, name, value)
            frames = self._file.create_group("frames")
            self._frames_jnt_current = _create_stream_dataset(
                frames, "jnt_current", (self._env_count, JOINT_COUNT, JOINT_FEATURES), np.dtype("float32")
            )
            self._frames_owner_contact = _create_stream_dataset(
                frames, "owner_contact", (self._env_count, OWNER_COUNT, 1), np.dtype("float32")
            )
            self._frames_active = _create_stream_dataset(frames, "active", (self._env_count,), np.dtype("float32"))
            samples = self._file.create_group("samples")
            self._samples_step_index = _create_stream_dataset(samples, "step_index", (), np.dtype("int64"))
            self._samples_teacher_mean = _create_stream_dataset(
                samples, "teacher_mean", (self._env_count, JOINT_COUNT), np.dtype("float32")
            )
            self._samples_behavior_action = _create_stream_dataset(
                samples, "behavior_action", (self._env_count, JOINT_COUNT), np.dtype("float32")
            )
            self._samples_geometry_tokens = _create_stream_dataset(
                samples, "geometry_tokens", (self._env_count, OWNER_COUNT, GEOMETRY_TOKEN_WIDTH), np.dtype("float32")
            )
            self._samples_joint_origin_fk = _create_stream_dataset(
                samples, "joint_origin_fk", (self._env_count, JOINT_COUNT, 3), np.dtype("float32")
            )
            self._file.flush()
        except BaseException:
            try:
                self._file.attrs["completed"] = False
                self._file.attrs["incomplete"] = True
                self._file.flush()
                self._file.close()
            finally:
                self._closed = True
            raise

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("family trajectory writer is closed")
        if self._finalized:
            raise RuntimeError("family trajectory writer is already finalized")

    def _validate_step_arrays(
        self,
        *,
        jnt_current: Any,
        owner_contact: Any,
        teacher_mean: Any,
        behavior_action: Any,
        geometry_tokens: Any,
        active: Any,
        history: Any,
        joint_origin_fk: Any,
        sample: bool,
    ) -> tuple[np.ndarray, ...]:
        current = _float_array(jnt_current, "jnt_current", (self._env_count, JOINT_COUNT, JOINT_FEATURES))
        owner = _float_array(owner_contact, "owner_contact", (self._env_count, OWNER_COUNT, 1))
        teacher = _float_array(teacher_mean, "teacher_mean", (self._env_count, JOINT_COUNT))
        behavior = _float_array(behavior_action, "behavior_action", (self._env_count, JOINT_COUNT))
        tokens = _float_array(geometry_tokens, "geometry_tokens", (self._env_count, OWNER_COUNT, GEOMETRY_TOKEN_WIDTH))
        active_float = _mask_array(active, "active", (self._env_count,))
        history_array = _float_array(history, "history", (self._env_count, HISTORY_LENGTH, JOINT_COUNT, JOINT_FEATURES))
        _validate_range(teacher, "teacher_mean", _ACTION_LOW, _ACTION_HIGH)
        _validate_range(behavior, "behavior_action", _ACTION_LOW, _ACTION_HIGH)
        _validate_range(owner, "owner_contact", _MASK_LOW, _MASK_HIGH)
        _validate_range(tokens, "geometry_tokens", -np.inf, np.inf)
        active_bool = active_float.astype(bool, copy=False)
        if self._have_previous_active and np.any(~self._active_previous & active_bool):
            raise FamilyDatasetError("active mask may only transition true to false; reactivation is forbidden")
        _exact_zero(current[~self._joint_valid_by_env], "jnt_current")
        invalid_history = np.broadcast_to((~self._joint_valid_by_env)[:, None, :, None], history_array.shape)
        _exact_zero(history_array[invalid_history], "history")
        _exact_zero(teacher[~self._joint_valid_by_env], "teacher_mean")
        _exact_zero(behavior[~self._joint_valid_by_env], "behavior_action")
        _exact_zero(owner[~self._owner_valid_by_env], "owner_contact")
        _exact_zero(tokens[~self._owner_valid_by_env], "geometry_tokens")
        if sample:
            if joint_origin_fk is None:
                raise FamilyDatasetError("joint_origin_fk is required on every sampled step")
            fk = _float_array(joint_origin_fk, "joint_origin_fk", (self._env_count, JOINT_COUNT, 3))
            _exact_zero(fk[~self._joint_valid_by_env], "joint_origin_fk")
        elif joint_origin_fk is None:
            fk = np.empty((0,), dtype=np.float32)
        else:
            fk = _float_array(joint_origin_fk, "joint_origin_fk", (self._env_count, JOINT_COUNT, 3))
            _exact_zero(fk[~self._joint_valid_by_env], "joint_origin_fk")
        return (current, owner, teacher, behavior, tokens, active_float, history_array, fk)

    def append(
        self,
        step: int,
        *,
        jnt_current: np.ndarray,
        owner_contact: np.ndarray,
        teacher_mean: np.ndarray,
        behavior_action: np.ndarray,
        geometry_tokens: np.ndarray,
        active: np.ndarray,
        history: np.ndarray,
        joint_origin_fk: np.ndarray | None = None,
    ) -> None:
        self._ensure_open()
        index = _scalar_step(step)
        if index != self._written_steps:
            raise FamilyDatasetError(f"step must be contiguous: expected {self._written_steps}, got {index}")
        sample = index % self._sample_stride == 0
        current, owner, teacher, behavior, tokens, active_float, history_array, fk = self._validate_step_arrays(
            jnt_current=jnt_current,
            owner_contact=owner_contact,
            teacher_mean=teacher_mean,
            behavior_action=behavior_action,
            geometry_tokens=geometry_tokens,
            active=active,
            history=history,
            joint_origin_fk=joint_origin_fk,
            sample=sample,
        )
        active_bool = active_float.astype(bool, copy=False)
        active_indices = np.flatnonzero(active_bool)
        if active_indices.size:
            expected = np.empty((active_indices.size, HISTORY_LENGTH, JOINT_COUNT, JOINT_FEATURES), dtype=np.float32)
            for local_index, env_index in enumerate(active_indices.tolist()):
                if not self._have_previous_active or not self._active_previous[env_index]:
                    expected[local_index] = self._initial_history[env_index]
                    if not np.allclose(
                        current[env_index], self._initial_history[env_index, -1], rtol=1e-05, atol=2e-06
                    ):
                        raise FamilyDatasetError(
                            f"jnt_current[{env_index}] does not match initial_history H0 latest frame"
                        )
                else:
                    expected[local_index, :-1] = self._history_state[env_index, 1:]
                    expected[local_index, -1] = current[env_index]
            actual = history_array[active_indices]
            difference = np.abs(actual - expected)
            if difference.size and (not np.allclose(actual, expected, rtol=1e-05, atol=2e-06)):
                raise FamilyDatasetError(
                    f"history continuity mismatch on active rows: active_count={active_indices.size}, max_abs_error={float(difference.max()):.6g}"
                )
            self._history_state[active_indices] = expected
        inactive_indices = np.flatnonzero(~active_bool)
        if inactive_indices.size:
            self._history_state[inactive_indices] = self._initial_history[inactive_indices]
        self._active_previous = active_bool.copy()
        self._have_previous_active = True
        _resize_and_write(self._frames_jnt_current, self._written_steps, current)
        _resize_and_write(self._frames_owner_contact, self._written_steps, owner)
        _resize_and_write(self._frames_active, self._written_steps, active_float)
        if sample:
            sample_index = self._sample_count
            _resize_and_write(self._samples_step_index, sample_index, np.int64(index))
            _resize_and_write(self._samples_teacher_mean, sample_index, teacher)
            _resize_and_write(self._samples_behavior_action, sample_index, behavior)
            _resize_and_write(self._samples_geometry_tokens, sample_index, tokens)
            _resize_and_write(self._samples_joint_origin_fk, sample_index, fk)
            self._sample_count += 1
        self._written_steps += 1
        self._file.attrs["recorded_steps"] = self._written_steps
        self._file.attrs["steps"] = self._written_steps
        self._file.attrs["sample_count"] = self._sample_count
        self._metadata["recorded_steps"] = self._written_steps
        self._metadata["steps"] = self._written_steps
        self._metadata["sample_count"] = self._sample_count
        _store_metadata(self._file, self._metadata)
        self._file.flush()

    def finalize(self, trajectory_summary: dict[str, np.ndarray]) -> None:
        self._ensure_open()
        if not isinstance(trajectory_summary, Mapping):
            raise TypeError("trajectory_summary must be a mapping")
        required = ("net_turns", "path_turns", "duration_s", "termination_drop", "termination_axis")
        missing = [name for name in required if name not in trajectory_summary]
        if missing:
            raise FamilyDatasetError(f"trajectory_summary is missing fields: {', '.join(missing)}")
        summary: dict[str, np.ndarray] = {
            name: _validate_summary_array(value, name, self._env_count) for name, value in trajectory_summary.items()
        }
        drop = _validated_terminal_flags(trajectory_summary["termination_drop"], "termination_drop", self._env_count)
        axis = _validated_terminal_flags(trajectory_summary["termination_axis"], "termination_axis", self._env_count)
        if self._written_steps < self._expected_steps:
            terminated_value = trajectory_summary.get("terminated")
            if terminated_value is None:
                raise FamilyDatasetError(
                    "early finalization requires trajectory_summary.terminated proof for every env"
                )
            terminated = _validated_terminal_flags(terminated_value, "terminated", self._env_count)
            if not bool(terminated.all()):
                raise FamilyDatasetError("early finalization requires terminated=True for every env")
        quality = quality_episode_mask(summary["net_turns"], summary["path_turns"], summary["duration_s"], drop, axis)
        if self._written_steps < self._expected_steps:
            quality[:] = False
        if "policy_step_count" in summary:
            policy_steps = summary["policy_step_count"]
            if policy_steps.dtype.kind not in "iu" or np.any(policy_steps < 0):
                raise FamilyDatasetError("trajectory_summary.policy_step_count must be nonnegative integer")
            quality &= policy_steps == QUALITY_HORIZON_STEPS
            quality &= self._written_steps >= QUALITY_HORIZON_STEPS
        final_group = self._file.create_group("final")
        for name, value in summary.items():
            _write_compressed(final_group, name, value)
        _write_compressed(final_group, "quality_episode_mask", quality.astype(bool))
        qualified_assets: list[int] = []
        zero_assets: list[int] = []
        for asset_index in range(self._asset_count):
            has_qualified = bool(quality[self._env_asset_index == asset_index].any())
            (qualified_assets if has_qualified else zero_assets).append(asset_index)
        self._metadata.update(
            {
                "completed": True,
                "incomplete": False,
                "steps": self._written_steps,
                "recorded_steps": self._written_steps,
                "sample_count": self._sample_count,
                "qualified_episode_count": int(quality.sum()),
                "qualified_asset_indices": qualified_assets,
                "zero_qualified_asset_indices": zero_assets,
                "has_qualified_episode": bool(quality.any()),
            }
        )
        self._file.attrs["completed"] = True
        self._file.attrs["incomplete"] = False
        self._file.attrs["recorded_steps"] = self._written_steps
        self._file.attrs["steps"] = self._written_steps
        self._file.attrs["sample_count"] = self._sample_count
        self._file.attrs["qualified_episode_count"] = int(quality.sum())
        self._file.attrs["has_qualified_episode"] = bool(quality.any())
        _store_metadata(self._file, self._metadata)
        self._file.flush()
        self._finalized = True

    def close(self) -> None:
        if self._closed:
            return
        if not self._finalized:
            self._metadata.update(
                {
                    "completed": False,
                    "incomplete": True,
                    "recorded_steps": self._written_steps,
                    "steps": self._written_steps,
                    "sample_count": self._sample_count,
                }
            )
            self._file.attrs["completed"] = False
            self._file.attrs["incomplete"] = True
            self._file.attrs["recorded_steps"] = self._written_steps
            self._file.attrs["steps"] = self._written_steps
            self._file.attrs["sample_count"] = self._sample_count
            _store_metadata(self._file, self._metadata)
        self._file.flush()
        self._file.close()
        self._closed = True

    def __enter__(self) -> FamilyTrajectoryWriter:
        self._ensure_open()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: Any) -> bool:
        self.close()
        return False

    def __del__(self) -> None:
        if getattr(self, "_closed", True):
            return
        with suppress(Exception):
            self.close()


def read_family_metadata(path: str | Path) -> dict[str, Any]:
    """Read and validate the trajectory identity and actor input schema."""
    destination = Path(path)
    if not destination.is_file():
        raise FileNotFoundError(destination)
    with h5py.File(destination, mode="r") as handle:
        artifact_type = _decode_scalar(handle.attrs.get("artifact_type"))
        schema = _decode_scalar(handle.attrs.get("schema_version"))
        if artifact_type != ARTIFACT_TYPE:
            raise FamilyDatasetError(f"unsupported artifact_type {artifact_type!r}")
        if schema != SCHEMA_VERSION:
            raise FamilyDatasetError(f"unsupported family trajectory schema_version {schema!r}")
        completed = bool(_decode_scalar(handle.attrs.get("completed", False)))
        incomplete = bool(_decode_scalar(handle.attrs.get("incomplete", True)))
        if not completed or incomplete:
            raise RuntimeError(f"family trajectory {destination} is incomplete and cannot be read")
        metadata_group = _required_group(handle, "metadata", "completed family trajectory")
        static_group = _required_group(handle, "static", "completed family trajectory")
        frames = _required_group(handle, "frames", "completed family trajectory")
        samples = _required_group(handle, "samples", "completed family trajectory")
        final_group = _required_group(handle, "final", "completed family trajectory")
        initial_history_dataset = _required_dataset(handle, "initial_history", "completed family trajectory")
        quality_dataset = _required_dataset(final_group, "quality_episode_mask", "completed family trajectory final")
        asset_count = int(_decode_scalar(handle.attrs.get("asset_count", -1)))
        if asset_count < 1:
            raise RuntimeError("completed family trajectory has invalid asset_count")
        for static_name in _REQUIRED_STATIC_SHAPES:
            static_dataset = _required_dataset(static_group, static_name, "completed family trajectory static")
            expected_static_shape = (asset_count, *_REQUIRED_STATIC_SHAPES[static_name])
            if static_dataset.shape != expected_static_shape:
                raise RuntimeError(f"static/{static_name} shape {static_dataset.shape} != {expected_static_shape}")
        raw_json = handle.attrs.get("metadata_json")
        if raw_json is None:
            raw_json = metadata_group.attrs.get("json")
        if raw_json is None:
            raise RuntimeError("completed family trajectory lacks metadata JSON")
        try:
            metadata = json.loads(_decode_scalar(raw_json))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError("family trajectory metadata JSON is invalid") from error
        if not isinstance(metadata, dict) or metadata.get("completed") is not True:
            raise RuntimeError("family trajectory metadata does not prove completed=True")
        recorded_steps = int(_decode_scalar(handle.attrs.get("recorded_steps", -1)))
        sample_count = int(_decode_scalar(handle.attrs.get("sample_count", -1)))
        expected_steps = int(_decode_scalar(handle.attrs.get("expected_steps", -1)))
        sample_stride = int(_decode_scalar(handle.attrs.get("sample_stride", -1)))
        frame_jnt_current = _required_dataset(frames, "jnt_current", "completed family trajectory frames")
        _required_dataset(frames, "owner_contact", "completed family trajectory frames")
        frame_active = _required_dataset(frames, "active", "completed family trajectory frames")
        sample_step_index = _required_dataset(samples, "step_index", "completed family trajectory samples")
        sample_teacher = _required_dataset(samples, "teacher_mean", "completed family trajectory samples")
        sample_behavior = _required_dataset(samples, "behavior_action", "completed family trajectory samples")
        sample_tokens = _required_dataset(samples, "geometry_tokens", "completed family trajectory samples")
        sample_fk = _required_dataset(samples, "joint_origin_fk", "completed family trajectory samples")
        env_assets = _required_dataset(handle, "env_asset_index", "completed family trajectory")
        env_replicas = _required_dataset(handle, "env_replica_index", "completed family trajectory")
        if recorded_steps != int(frame_jnt_current.shape[0]):
            raise RuntimeError("metadata recorded_steps disagrees with frames/jnt_current length")
        if sample_count != int(sample_step_index.shape[0]):
            raise RuntimeError("metadata sample_count disagrees with samples/step_index length")
        if expected_steps < 1 or sample_stride < 1 or recorded_steps < 1 or (recorded_steps > expected_steps):
            raise RuntimeError("completed family trajectory has invalid lifecycle counters")
        if env_assets.shape != (frame_active.shape[1],) or env_replicas.shape != env_assets.shape:
            raise RuntimeError("completed family trajectory environment index shape is inconsistent")
        env_count = int(frame_active.shape[1])
        if frame_jnt_current.shape != (recorded_steps, env_count, JOINT_COUNT, JOINT_FEATURES):
            raise RuntimeError("completed family trajectory jnt_current shape is inconsistent")
        if _required_dataset(frames, "owner_contact", "completed family trajectory frames").shape != (
            recorded_steps,
            env_count,
            OWNER_COUNT,
            1,
        ) or frame_active.shape != (recorded_steps, env_count):
            raise RuntimeError("completed family trajectory contact/active shape is inconsistent")
        expected_sample_count = (recorded_steps - 1) // sample_stride + 1
        if sample_count != expected_sample_count:
            raise RuntimeError("metadata sample_count does not match sample_stride and recorded_steps")
        if (
            sample_teacher.shape != (sample_count, env_count, JOINT_COUNT)
            or sample_behavior.shape != sample_teacher.shape
        ):
            raise RuntimeError("completed family trajectory action sample shape is inconsistent")
        if sample_tokens.shape != (sample_count, env_count, OWNER_COUNT, GEOMETRY_TOKEN_WIDTH):
            raise RuntimeError("completed family trajectory geometry sample shape is inconsistent")
        if sample_fk.shape != (sample_count, env_count, JOINT_COUNT, 3):
            raise RuntimeError("completed family trajectory FK sample shape is inconsistent")
        if initial_history_dataset.shape != (env_count, HISTORY_LENGTH, JOINT_COUNT, JOINT_FEATURES):
            raise RuntimeError("completed family trajectory initial_history shape is inconsistent")
        if initial_history_dataset.dtype != np.dtype("float32"):
            raise RuntimeError("completed family trajectory initial_history must be float32")
        if sample_step_index.dtype.kind not in "iu" or not np.array_equal(
            np.asarray(sample_step_index, dtype=np.int64), np.arange(0, recorded_steps, sample_stride, dtype=np.int64)
        ):
            raise RuntimeError("samples/step_index is not the exact stride subsequence of dense steps")
        if quality_dataset.shape != (env_count,):
            raise RuntimeError("final/quality_episode_mask shape is inconsistent")
        result = dict(metadata)
        result.update(
            {
                "artifact_type": artifact_type,
                "schema_version": schema,
                "completed": completed,
                "incomplete": incomplete,
                "expected_steps": expected_steps,
                "nominal_steps": expected_steps,
                "steps": recorded_steps,
                "recorded_steps": recorded_steps,
                "sample_stride": sample_stride,
                "sample_count": sample_count,
                "env_count": int(_decode_scalar(handle.attrs.get("env_count", frame_active.shape[1]))),
                "asset_count": asset_count,
                "path": str(destination),
                "qualified_episode_count": int(
                    _decode_scalar(handle.attrs.get("qualified_episode_count", quality_dataset.shape[0]))
                ),
                "has_qualified_episode": bool(_decode_scalar(handle.attrs.get("has_qualified_episode", False))),
            }
        )
        return result


def reconstruct_history(
    initial_history: np.ndarray, frames: np.ndarray, step_indices: np.ndarray, env_indices: np.ndarray
) -> np.ndarray:
    """Reconstruct oldest-to-current History30 without crossing episode boundaries."""
    raw_initial = _array(initial_history, "initial_history")
    if raw_initial.ndim < 1:
        raise FamilyDatasetError("initial_history must have an environment axis")
    init = _float_array(
        raw_initial, "initial_history", (int(raw_initial.shape[0]), HISTORY_LENGTH, JOINT_COUNT, JOINT_FEATURES)
    )
    frame_array = _array(frames, "frames")
    if frame_array.ndim != 4 or frame_array.shape[1:] != (init.shape[0], JOINT_COUNT, JOINT_FEATURES):
        raise FamilyDatasetError(
            f"frames shape {frame_array.shape} != expected [T,{init.shape[0]},{JOINT_COUNT},{JOINT_FEATURES}]"
        )
    if frame_array.shape[0] < 1:
        raise FamilyDatasetError("frames must contain at least current0")
    if frame_array.dtype.kind != "f" or not bool(np.isfinite(frame_array).all()):
        raise FamilyDatasetError("frames must be finite floating values")
    frame_array = frame_array.astype(np.float32, copy=False)
    step_array = _array(step_indices, "step_indices")
    env_array = _array(env_indices, "env_indices")
    if step_array.ndim == 0:
        step_array = step_array.reshape(1)
    if env_array.ndim == 0:
        env_array = env_array.reshape(1)
    if step_array.ndim != 1 or env_array.ndim != 1 or step_array.shape != env_array.shape:
        raise FamilyDatasetError("step_indices and env_indices must be one-dimensional vectors of equal length")
    if step_array.size < 1:
        raise FamilyDatasetError("history reconstruction batch must be nonempty")
    if step_array.dtype.kind not in "iu" or env_array.dtype.kind not in "iu":
        raise FamilyDatasetError("step_indices and env_indices must be integer vectors")
    if np.any(step_array < 0) or np.any(step_array >= frame_array.shape[0]):
        raise FamilyDatasetError("step_indices contain a value outside dense frame range")
    if np.any(env_array < 0) or np.any(env_array >= init.shape[0]):
        raise FamilyDatasetError("env_indices contain a value outside initial_history environment range")
    output = np.empty((step_array.size, HISTORY_LENGTH, JOINT_COUNT, JOINT_FEATURES), dtype=np.float32)
    for batch_index, (step_value, env_value) in enumerate(zip(step_array.tolist(), env_array.tolist(), strict=True)):
        step = int(step_value)
        env = int(env_value)
        if step == 0:
            output[batch_index] = init[env]
            continue
        first_new = max(1, step - HISTORY_LENGTH + 1)
        new_frames = frame_array[first_new : step + 1, env]
        prefix_count = HISTORY_LENGTH - int(new_frames.shape[0])
        if prefix_count > 0:
            prefix = init[env, -prefix_count:]
            output[batch_index] = np.concatenate((prefix, new_frames), axis=0)
        else:
            output[batch_index] = new_frames[-HISTORY_LENGTH:]
    return output


def _quality_float(value: Any, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = _array(value, name)
    if shape is not None and array.shape != shape:
        raise FamilyDatasetError(f"{name} shape {array.shape} != expected {shape}")
    if array.ndim != 1 or array.size < 1:
        raise FamilyDatasetError(f"{name} must be a nonempty one-dimensional vector")
    if array.dtype.kind not in "fiu" or array.dtype.kind == "b":
        raise FamilyDatasetError(f"{name} must be finite numeric values")
    if array.dtype.kind == "f" and (not bool(np.isfinite(array).all())):
        raise FamilyDatasetError(f"{name} must be finite floating values")
    return array.astype(np.float64, copy=False)


def quality_episode_mask(
    net_turns: np.ndarray,
    path_turns: np.ndarray,
    duration_s: np.ndarray,
    termination_drop: np.ndarray,
    termination_axis: np.ndarray,
    *,
    min_turns: float = 0.5,
    min_direction: float = 0.7,
    horizon_s: float = 30.0,
) -> np.ndarray:
    """Select complete, safe demonstrations using the fixed rotation thresholds."""
    net = _quality_float(net_turns, "net_turns")
    path = _quality_float(path_turns, "path_turns", net.shape)
    duration = _quality_float(duration_s, "duration_s", net.shape)
    drop = _bool_mask(termination_drop, "termination_drop", net.shape)
    axis = _bool_mask(termination_axis, "termination_axis", net.shape)
    try:
        turns_threshold = float(min_turns)
        direction_threshold = float(min_direction)
        horizon = float(horizon_s)
    except (TypeError, ValueError) as error:
        raise FamilyDatasetError("quality thresholds must be finite numbers") from error
    if not np.isfinite(turns_threshold) or turns_threshold < 0.0:
        raise FamilyDatasetError("min_turns must be finite and nonnegative")
    if not np.isfinite(direction_threshold) or not 0.0 <= direction_threshold <= 1.0:
        raise FamilyDatasetError("min_direction must lie in [0,1]")
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise FamilyDatasetError("horizon_s must be finite and positive")
    if np.any(path < 0.0) or np.any(duration < 0.0):
        raise FamilyDatasetError("path_turns and duration_s must be nonnegative")
    with np.errstate(divide="ignore", invalid="ignore"):
        direction = np.divide(net, path, out=np.full_like(net, -np.inf), where=path > 0.0)
    return (
        (duration >= horizon - DURATION_ATOL_S)
        & (net >= turns_threshold)
        & (direction >= direction_threshold)
        & ~drop
        & ~axis
    )


def episode_replica_split(
    replica_index: np.ndarray, *, validation_modulus: int = 4, validation_remainder: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Split complete replicas so overlapping histories cannot cross data partitions."""
    array = _array(replica_index, "replica_index")
    if array.ndim != 1 or array.size < 1 or array.dtype.kind not in "iu" or (array.dtype.kind == "b"):
        raise FamilyDatasetError("replica_index must be a nonempty integer vector")
    if np.any(array < 0):
        raise FamilyDatasetError("replica_index must contain nonnegative ids")
    try:
        modulus = operator.index(validation_modulus)
        remainder = operator.index(validation_remainder)
    except (TypeError, ValueError) as error:
        raise FamilyDatasetError("replica modulo rule must be integer") from error
    if isinstance(validation_modulus, (bool, np.bool_)) or modulus < 2:
        raise FamilyDatasetError("validation_modulus must be at least 2")
    if isinstance(validation_remainder, (bool, np.bool_)) or not 0 <= remainder < modulus:
        raise FamilyDatasetError("validation_remainder must lie within validation_modulus")
    validation = np.flatnonzero(array % modulus == remainder).astype(np.int64, copy=False)
    training = np.flatnonzero(array % modulus != remainder).astype(np.int64, copy=False)
    return (training, validation)


split_episode_replicas = episode_replica_split
replica_modulo_split = episode_replica_split
__all__ = [
    "ARTIFACT_TYPE",
    "SCHEMA_VERSION",
    "HISTORY_LENGTH",
    "JOINT_COUNT",
    "JOINT_FEATURES",
    "OWNER_COUNT",
    "TIP_COUNT",
    "GEOMETRY_TOKEN_WIDTH",
    "JOINT_KINEMATICS_WIDTH",
    "FamilyDatasetError",
    "FamilyTrajectoryWriter",
    "episode_replica_split",
    "quality_episode_mask",
    "read_family_metadata",
    "reconstruct_history",
    "replica_modulo_split",
    "split_episode_replicas",
]
