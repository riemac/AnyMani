'Aggregate the settled first 30-second training window per asset; inputs are revolutions and CPU facts.'

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np


_COLUMN_KINDS = {
    "asset_index": "iu",  # shapes [E]
    "net_turns": "f",  # shapes [E]
    "absolute_path_turns": "f",  # shapes [E]
    "goal_count": "iu",  # shapes [E]
    "safe": "b",  # shapes [E]
    "policy_version_start": "iu",  # shapes [E]
    "policy_version_end": "iu",  # shapes [E]
}



_DIRECTION_TOLERANCE = 1e-6
_PROXY_MINIMUM_WINDOWS = 16
_Window = tuple[float, float, int, bool, int, int]
_Metrics = dict[str, int | float | None]


def _integer(value: object, name: str, minimum: int) -> int:
    'Handle integer.'

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, not boolean or floating point")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _validated_window(values: object) -> _Window:
    'Handle validated window.'

    expected = (float, float, int, bool, int, int)
    if not isinstance(values, (list, tuple)) or len(values) != len(expected):
        raise ValueError("a window must contain [net, path, goal, safe, policy_start, policy_end]")
    if any(type(value) is not kind for value, kind in zip(values, expected, strict=True)):
        raise ValueError("window scalar types must be float, float, int, bool, int, int")
    net, path, goals, _, start, end = cast(_Window, tuple(values))


    if not math.isfinite(net) or not math.isfinite(path):
        raise ValueError("net_turns and absolute_path_turns must be finite Python floats")
    if path < 0 or goals < 0 or start < 0 or end < start:
        raise ValueError("path/goal_count/policy versions must be nonnegative and policy end >= start")
    try:
        float(goals)
    except OverflowError as error:
        raise ValueError("goal_count must admit a finite Python float summary") from error
    if path == 0.0:
        if net != 0.0:
            raise ValueError("zero absolute path requires exactly zero net_turns")
    elif abs(net / path) > 1.0 + _DIRECTION_TOLERANCE:
        raise ValueError("absolute path is smaller than abs(net_turns) beyond floating-point tolerance")
    return cast(_Window, tuple(values))


def _direction(window: _Window) -> float:
    'Handle direction.'
    net, path = window[:2]
    return 0.0 if path == 0.0 else max(-1.0, min(1.0, net / path))


def _median(values: Sequence[int | float]) -> float:
    'Handle median.'
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    low, high = float(ordered[middle - 1]), float(ordered[middle])
    if low <= 0.0 <= high:
        return (low + high) / 2.0
    return low + (high - low) / 2.0


class FirstWindowStatistics:
    'Aggregate up to 32 first-window records per asset; report safety fractions in [0,1].'

    def __init__(self, asset_count: int, max_windows_per_asset: int = 32, minimum_windows: int = 16) -> None:
        'Initialize the instance.'
        self.asset_count = _integer(asset_count, "asset_count", 0)
        self.max_windows_per_asset = _integer(max_windows_per_asset, "max_windows_per_asset", 1)
        self.minimum_windows = _integer(minimum_windows, "minimum_windows", 1)
        if self.max_windows_per_asset > 32:
            raise ValueError("max_windows_per_asset must be <= 32")
        self._windows: list[deque[_Window]] = [
            deque(maxlen=self.max_windows_per_asset) for _ in range(self.asset_count)
        ]

    def add_batch(self, columns: Mapping[str, np.ndarray]) -> None:
        'Handle add batch; shapes [E].'

        if not isinstance(columns, Mapping) or any(name not in columns for name in _COLUMN_KINDS):
            raise ValueError(f"columns must supply all required arrays: {tuple(_COLUMN_KINDS)}")
        arrays = [columns[name] for name in _COLUMN_KINDS]
        for (name, kinds), array in zip(_COLUMN_KINDS.items(), arrays, strict=True):
            if not isinstance(array, np.ndarray) or np.ma.isMaskedArray(array) or array.ndim != 1:
                raise ValueError(f"{name} must be an unmasked CPU numpy [E] array")
            if array.dtype.kind not in kinds:
                raise ValueError(f"{name} has invalid dtype {array.dtype}; expected numpy kind {kinds}")
            if array.dtype.kind == "f" and not np.isfinite(array).all():
                raise ValueError(f"{name} must contain only finite values")
        count = arrays[0].size
        if any(array.shape != (count,) for array in arrays):
            raise ValueError("all required columns must be aligned [E] arrays")


        pending: list[tuple[int, _Window]] = []
        for asset, net, path, goals, safe, start, end in zip(*arrays, strict=True):
            index = int(asset)
            if not 0 <= index < self.asset_count:
                raise ValueError(f"asset_index {index} outside [0, {self.asset_count})")
            net_value, path_value = float(net), float(path)
            if (net != 0.0 and net_value == 0.0) or (path != 0.0 and path_value == 0.0):
                raise ValueError("physical values must not underflow to zero in Python float state")
            window = _validated_window((net_value, path_value, int(goals), bool(safe), int(start), int(end)))
            pending.append((index, window))
        for index, window in pending:
            self._windows[index].append(window)

    def per_asset(self) -> dict[int, dict[str, int | float | None]]:
        'Handle per asset.'
        return {index: self.summary([index]) for index in range(self.asset_count)}

    def summary(self, asset_indices: Sequence[int] | None = None) -> _Metrics:
        'Handle summary.'

        if asset_indices is None:
            selected = list(range(self.asset_count))
        else:
            if (
                not isinstance(asset_indices, (Sequence, np.ndarray))
                or isinstance(asset_indices, (str, bytes))
                or (isinstance(asset_indices, np.ndarray) and asset_indices.ndim != 1)
            ):
                raise ValueError("asset_indices must be a one-dimensional integer sequence")
            selected = [_integer(index, "asset_index", 0) for index in asset_indices]
            if any(index >= self.asset_count for index in selected) or len(set(selected)) != len(selected):
                raise ValueError("asset_indices must be unique and inside the configured asset range")


        counts = [len(self._windows[index]) for index in selected]
        observed = [self._windows[index] for index in selected if self._windows[index]]
        values = [
            (
                len(windows),
                _median([window[0] for window in windows]),
                _median([window[2] for window in windows]),
                _median([_direction(window) for window in windows]),
                sum(window[3] for window in windows) / len(windows),  # Safety fraction in [0,1].
            )
            for windows in observed
        ]


        medians = [_median([row[column] for row in values]) if values else None for column in (1, 2, 3)]
        safe_fraction = sum(row[4] for row in values) / len(values) if values else None
        eligible_net = [
            net
            for n, net, _, direction, safe in values
            if n >= _PROXY_MINIMUM_WINDOWS and direction >= 0.7 and safe >= 0.75
        ]
        return {
            "first30_asset_count": len(selected),
            "first30_observed_assets": len(observed),
            "first30_qualified_assets": sum(count >= self.minimum_windows for count in counts),
            "first30_window_count": sum(counts),
            "first30_windows_min": min(counts, default=0),
            "first30_windows_max": max(counts, default=0),
            "first30_net_median": medians[0],
            "first30_goal_median": medians[1],
            "first30_direction_median": medians[2],
            "first30_safe_fraction": safe_fraction,  # Safety fraction in [0,1].
            "first30_one_turn_proxy_assets": sum(net >= 1.0 for net in eligible_net),
            "first30_two_turn_proxy_assets": sum(net >= 2.0 for net in eligible_net),
            "first30_policy_start_min": min((window[4] for windows in observed for window in windows), default=None),
            "first30_policy_end_max": max((window[5] for windows in observed for window in windows), default=None),
        }

    def state_dict(self) -> dict[str, object]:
        'Handle state dict; shapes [a], [net,path,goal,safe,start,end].'
        return {
            "asset_count": self.asset_count,
            "max_windows_per_asset": self.max_windows_per_asset,
            "minimum_windows": self.minimum_windows,
            "windows": [[list(window) for window in windows] for windows in self._windows],
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        'Load state dict.'

        expected = {"asset_count", "max_windows_per_asset", "minimum_windows", "windows"}
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError(f"first-window state must contain exactly {sorted(expected)}")
        for name in ("asset_count", "max_windows_per_asset", "minimum_windows"):
            if type(state[name]) is not int or state[name] != getattr(self, name):
                raise ValueError(f"state configuration {name} does not match this statistics object")
        histories = state["windows"]
        if type(histories) is not list or len(histories) != self.asset_count:
            raise ValueError("state windows must be a list with one history per configured asset")


        restored: list[deque[_Window]] = []
        for history in histories:
            if type(history) is not list or len(history) > self.max_windows_per_asset:
                raise ValueError("state asset histories must be lists within the configured window capacity")
            if any(type(window) is not list for window in history):
                raise ValueError("state windows must contain Python lists of scalar facts")
            restored.append(deque((_validated_window(window) for window in history), maxlen=self.max_windows_per_asset))
        self._windows = restored
