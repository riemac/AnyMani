r"""稀疏优化审计包：保存调用方已有的dataset、模型、优化器与RNG，不重跑物理。

只在预定rollout边界写入。Tensor复制到CPU后与训练可变buffer断开；
原子发布完整.pt文件，不覆盖既有审计包。默认单包上限1 GiB。
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch


def write_optimization_evidence(destination: Path, payload: Mapping[str, Any], *, max_bytes: int = 1 << 30) -> int:
    r"""保存纯数据审计包；返回未压缩Tensor bytes，拒绝模型实例和不透明Python对象。"""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    total = 0

    def copy_data(value: Any) -> Any:
        r"""递归转换state_dict/NumPy RNG数据，保证weights_only读取不执行任意对象构造。"""
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
                return scalar_type(value)  # TorchVersion等标量子类不以可执行Python对象形式保存。
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
    r"""以CPU及weights-only方式读取审计数据，不构造训练环境或自动恢复模型。"""
    result = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(result, dict) or result.get("artifact_type") != "palm-rotation-optimization-audit":
        raise ValueError("not a palm-rotation optimization audit")
    return result
