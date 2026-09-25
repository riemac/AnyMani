'Runtime contracts for optimization evidence.'

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch


def write_optimization_evidence(destination: Path, payload: Mapping[str, Any], *, max_bytes: int = 1 << 30) -> int:
    'Write optimization evidence.'
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    total = 0

    def copy_data(value: Any) -> Any:
        'Copy data.'
        nonlocal total
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value.copy())
        if isinstance(value, torch.Tensor):
            total += value.numel() * value.element_size()
            if total > max_bytes:
                raise ValueError("optimization audit exceeds tensor byte budget")
            return value.detach().cpu().clone()
        if isinstance(value, Mapping):
            return {key: copy_data(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(copy_data(item) for item in value)
        if isinstance(value, np.generic):
            return value.item()
        if value is None:
            return None
        for scalar_type in (bool, str, int, float):
            if isinstance(value, scalar_type):
                return scalar_type(value)
        raise TypeError(f"optimization audit only accepts data, got {type(value).__name__}")

    snapshot = copy_data(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".optimization-", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(snapshot, temporary)
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return total


def read_optimization_evidence(source: Path) -> dict[str, Any]:
    'Read optimization evidence.'
    result = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(result, dict) or result.get("artifact_type") != "palm-rotation-optimization-audit":
        raise ValueError("not a palm-rotation optimization audit")
    return result
