'Persist immutable per-episode evidence in typed Parquet shards.'

from __future__ import annotations

import math
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import polars as pl


_INTEGER_COLUMNS = (
    "env_id",
    "episode_id",
    "asset_index",
    "policy_version_start",
    "policy_version_end",
    "policy_steps",
    "goal_count",
    "frontier_count",
)

_FLOAT_COLUMNS = ("duration_s", "net_turns", "absolute_path_turns", "max_positive_net_turns")  # Duration is seconds; turn fields are revolutions.

_BOOL_COLUMNS = ("termination_drop", "termination_axis", "termination_timeout", "censored")
_EXTRA_INTEGER_COLUMNS = ("orientation_goal_count", "adr_position_level", "goal_count_first30")
_EXTRA_FLOAT_COLUMNS = (
    "adr_position_offset_x_h_m", "adr_position_offset_y_h_m", "net_turns_first30", "absolute_path_turns_first30",  # Offsets are metres; turn fields are revolutions.
)
_EXTRA_BOOL_COLUMNS = ("first30_complete", "first30_safe")
_FIRST30_COLUMNS = {
    "net_turns_first30", "absolute_path_turns_first30", "goal_count_first30", "first30_complete", "first30_safe",
}


def write_episode_evidence(
    destination: Path,
    columns: Mapping[str, np.ndarray],
    *,
    reward_sums: Mapping[str, np.ndarray],
    identity_digest: str,
    segment_id: str,
    policy_dt_s: float = 0.05,
) -> Path:
    'Write E episode rows with reward sums, duration, revolutions, and safety labels.'

    destination = Path(destination)
    if destination.suffix != ".parquet" or destination.exists():
        raise ValueError("episode evidence requires a new .parquet destination")
    if not re.fullmatch(r"[0-9a-f]{64}", identity_digest) or not segment_id:
        raise ValueError("episode evidence requires method identity and segment ID")
    if not math.isfinite(policy_dt_s) or policy_dt_s <= 0:
        raise ValueError("policy_dt_s must be finite and positive")


    expected = set(_INTEGER_COLUMNS + _FLOAT_COLUMNS + _BOOL_COLUMNS)
    optional = set(_EXTRA_INTEGER_COLUMNS + _EXTRA_FLOAT_COLUMNS + _EXTRA_BOOL_COLUMNS)
    if not expected <= set(columns) or set(columns) - expected - optional:
        raise ValueError(f"episode columns differ from schema: {set(columns) ^ expected}")
    integers = (*_INTEGER_COLUMNS, *(key for key in _EXTRA_INTEGER_COLUMNS if key in columns))
    floats = (*_FLOAT_COLUMNS, *(key for key in _EXTRA_FLOAT_COLUMNS if key in columns))
    bools = (*_BOOL_COLUMNS, *(key for key in _EXTRA_BOOL_COLUMNS if key in columns))
    arrays = {name: np.asarray(value) for name, value in columns.items()}
    count = arrays["env_id"].size
    if count == 0 or any(value.shape != (count,) for value in arrays.values()):
        raise ValueError("episode evidence requires nonempty aligned [E] arrays")
    for name in integers:
        if arrays[name].dtype.kind not in "iu" or np.any(arrays[name] < 0):
            raise ValueError(f"{name} must contain nonnegative integer values")
    for name in bools:
        if arrays[name].dtype.kind != "b":
            raise ValueError(f"{name} must be boolean")
    for name in floats:
        if arrays[name].dtype.kind not in "fiu" or not np.isfinite(arrays[name]).all():
            raise ValueError(f"{name} must contain finite physical values")


    terminal = arrays["termination_drop"] | arrays["termination_axis"] | arrays["termination_timeout"]  # shapes [E]
    if np.any(terminal == arrays["censored"]):
        raise ValueError("a row must be either terminal or censored, not both")
    keys = np.stack((arrays["env_id"], arrays["episode_id"]), axis=1)  # shapes [E,2]
    if np.unique(keys, axis=0).shape[0] != count:
        raise ValueError("duplicate episode identity within shard")
    if np.any(arrays["policy_version_end"] < arrays["policy_version_start"]):
        raise ValueError("episode policy versions must be monotone")
    if np.any(arrays["policy_steps"] < 1) or not np.allclose(
        arrays["duration_s"], arrays["policy_steps"] * policy_dt_s, rtol=1e-6, atol=1e-5
    ):
        raise ValueError("duration must agree with observed policy steps, not planned horizon")
    if np.any(arrays["absolute_path_turns"] + 1e-5 < np.abs(arrays["net_turns"])):
        raise ValueError("absolute path cannot be smaller than signed net rotation")
    if np.any(arrays["max_positive_net_turns"] + 1e-5 < np.maximum(arrays["net_turns"], 0)):
        raise ValueError("positive frontier cannot precede current positive net rotation")


    joint_window = bool(set(columns) & {"absolute_path_turns_first30", "goal_count_first30", "first30_safe"})
    if joint_window:
        if not _FIRST30_COLUMNS.issubset(columns):
            raise ValueError("joint first30 episode evidence requires the complete window fields")
        if np.any(arrays["first30_complete"] != (arrays["duration_s"] >= 30.0)):
            raise ValueError("first30 completion must correspond to thirty actual seconds")
        if np.any(arrays["absolute_path_turns_first30"] + 1e-5 < np.abs(arrays["net_turns_first30"])):
            raise ValueError("first30 absolute path cannot be smaller than signed net rotation")
        if np.any(arrays["goal_count_first30"] > arrays["goal_count"]):
            raise ValueError("first30 goals cannot exceed the episode goal count")
        unsafe_prefix = (~arrays["first30_complete"]) | (
            (arrays["duration_s"] <= 30.0) & (arrays["termination_drop"] | arrays["termination_axis"])
        )
        if np.any(arrays["first30_safe"] & unsafe_prefix):
            raise ValueError("first30 safety contradicts observed completion or boundary failure")


    for name, value in reward_sums.items():
        value = np.asarray(value)  # shapes [E]
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            raise ValueError(f"invalid reward term name: {name}")
        if value.shape != (count,) or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError(f"reward sum {name} must be a finite [E] array")
        arrays[f"reward_sum/{name}"] = value.astype(np.float64)
    table = pl.DataFrame(arrays).with_columns(
        [pl.col(name).cast(pl.Int64) for name in integers]
        + [pl.col(name).cast(pl.Float64) for name in floats]
        + [
            pl.lit("1.2.0" if joint_window else ("1.1.0" if set(columns) & optional else "1.0.0")).alias("schema_version"),
            pl.lit(identity_digest).alias("identity_digest"),
            pl.lit(segment_id).alias("segment_id"),
            pl.lit(policy_dt_s).alias("policy_dt_s"),
        ]
    )

    return _publish_table(destination, table)


def write_first_window_evidence(
    destination: Path,
    columns: Mapping[str, np.ndarray],
    *,
    identity_digest: str,
    segment_id: str,
    policy_dt_s: float = 0.05,
) -> Path:
    'Write E settled-window rows with turns, goals, safety, and policy versions.'
    destination = Path(destination)
    integers = ("env_id", "episode_id", "asset_index", "policy_version_start", "policy_version_end", "policy_steps", "goal_count")
    floats = ("duration_s", "net_turns", "absolute_path_turns")
    bools = ("complete", "safe", "termination_drop", "termination_axis", "termination_timeout")
    if destination.suffix != ".parquet" or destination.exists():
        raise ValueError("first-window evidence requires a new .parquet destination")
    if not re.fullmatch(r"[0-9a-f]{64}", identity_digest) or not segment_id:
        raise ValueError("first-window evidence requires method identity and segment ID")
    if not math.isfinite(policy_dt_s) or policy_dt_s <= 0 or set(columns) != set(integers + floats + bools):
        raise ValueError("first-window evidence has invalid time scale or columns")
    arrays = {name: np.asarray(value) for name, value in columns.items()}
    count = arrays["env_id"].size
    if count < 1 or any(value.shape != (count,) for value in arrays.values()):
        raise ValueError("first-window columns must be nonempty aligned vectors")
    for name in integers:
        if arrays[name].dtype.kind not in "iu" or np.any(arrays[name] < 0):
            raise ValueError(f"{name} must contain nonnegative integer values")
    for name in floats:
        if arrays[name].dtype.kind not in "fiu" or not np.isfinite(arrays[name]).all():
            raise ValueError(f"{name} must contain finite physical values")
    if any(arrays[name].dtype.kind != "b" for name in bools):
        raise ValueError("first-window event columns must be boolean")


    duration = arrays["duration_s"]  # values 20Hz; units Hz
    failure = arrays["termination_drop"] | arrays["termination_axis"]
    if np.any(duration <= 0) or np.any(duration > 30.0 + 1e-5):
        raise ValueError("first-window duration must be in (0,30] seconds")
    if np.any(arrays["policy_steps"] < 1) or not np.allclose(duration, arrays["policy_steps"] * policy_dt_s, rtol=1e-6, atol=1e-5):
        raise ValueError("first-window duration and observed step count disagree")
    if np.any(arrays["complete"] != (duration >= 30.0)) or np.any(~arrays["complete"] & ~failure):
        raise ValueError("an incomplete first-window row must end in a physical failure")
    if np.any(arrays["safe"] != (arrays["complete"] & ~failure)):
        raise ValueError("first-window safety must match completion and physical termination")
    if np.any(arrays["absolute_path_turns"] + 1e-5 < np.abs(arrays["net_turns"])):
        raise ValueError("first-window path must bound signed net rotation")
    if np.any(arrays["policy_version_end"] < arrays["policy_version_start"]):
        raise ValueError("first-window policy versions must be monotone")
    if np.unique(np.stack((arrays["env_id"], arrays["episode_id"]), axis=1), axis=0).shape[0] != count:  # shapes [E,2]
        raise ValueError("duplicate first-window episode key")
    table = pl.DataFrame(arrays).with_columns(
        [pl.col(name).cast(pl.Int64) for name in integers]
        + [pl.col(name).cast(pl.Float64) for name in floats]
        + [pl.lit("1.0.0").alias("schema_version"), pl.lit(identity_digest).alias("identity_digest"),
           pl.lit(segment_id).alias("segment_id"), pl.lit(policy_dt_s).alias("policy_dt_s"),
           pl.lit(30.0).alias("window_seconds")]  # units s
    )
    return _publish_table(destination, table)


def _publish_table(destination: Path, table: pl.DataFrame) -> Path:
    'Handle publish table.'

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".episode-", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        table.write_parquet(temporary, compression="zstd")
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
