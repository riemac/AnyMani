r"""Geometry SSL 实验快照 registry。

每个 registry entry 指向一个语义自包含的 Python 快照。快照同时导出训练配置，以及在论文
作图或泛化分析时可选使用的 validation/evaluation 配置。这里使用显式注册而不是自动扫描目录，
因为实验名称、默认版本和历史 preset 都是需要审计的科研身份。
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


@dataclass(frozen=True)
class ExperimentPreset:
    r"""一个可被 CLI 和 ConfigStore 加载的完整实验快照。

    ``EXPERIMENT`` 是训练主配置；validation/evaluation 导出保持在同一快照中，但不属于训练
    生命周期。``config_sha256`` 用于 checkpoint lineage，使同名文件被修改后不会伪装成同一实验。
    """

    name: str
    module_name: str
    module: ModuleType
    pretrain: Any
    validation: Any | None
    evaluation: Any | None
    path: Path
    config_sha256: str


# 只有这里声明公开实验名称；版本化快照作为默认值，历史文件只通过显式名称访问。
_MODULES: dict[str, str] = {
    "geometry_ssl_multitask_representation_v0_7_3": (
        "anymani.distill.ssl.experiments.geometry_ssl_multitask_representation_v0_7_3"
    ),
    "multi_anchor_gaussian_implicit_field": (
        "anymani.distill.ssl.experiments.multi_anchor_gaussion_implicit_field"
    ),
}


def available_experiments() -> tuple[str, ...]:
    r"""返回可被 ``--config`` 直接引用的稳定实验名称。"""

    return tuple(_MODULES)


def _module_file(module: ModuleType) -> Path:
    r"""解析 Python 快照文件路径，失败时拒绝生成无 provenance 的配置。"""

    module_file = getattr(module, "__file__", None)
    if module_file is None:
        raise ValueError(f"experiment module {module.__name__!r} has no source file")
    return Path(module_file).resolve()


def _build_preset(name: str, module: ModuleType) -> ExperimentPreset:
    r"""从模块导出训练及可选事后配置，并计算快照内容 identity。"""

    path = _module_file(module)
    if not hasattr(module, "EXPERIMENT"):
        raise TypeError(f"experiment snapshot {path} must export EXPERIMENT")
    return ExperimentPreset(
        name=name,
        module_name=module.__name__,
        module=module,
        pretrain=module.EXPERIMENT,
        validation=getattr(module, "VALIDATION_EXPERIMENT", None),
        evaluation=getattr(module, "EVALUATION_EXPERIMENT", None),
        path=path,
        config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _load_path(path: Path) -> ExperimentPreset:
    r"""按 Python 文件路径加载实验快照，不依赖当前 package import 名称。"""

    path = path.expanduser().resolve(strict=True)
    if path.suffix != ".py":
        raise ValueError(f"experiment config path must point to a .py file: {path}")
    file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    module_name = f"anymani_external_experiment_{file_digest[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for experiment config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    preset = _build_preset(f"{path.stem}_{file_digest[:12]}", module)
    return preset


def load_experiment(config_ref: str | Path) -> ExperimentPreset:
    r"""按 registry 名称或 Python 文件路径加载完整实验快照。

    Args:
        config_ref: registry 中的实验名，或包含 ``EXPERIMENT`` 导出的 Python 文件路径。

    Returns:
        ExperimentPreset: 训练配置、可选后处理配置和快照 provenance。

    Raises:
        KeyError: ``config_ref`` 不是已注册实验名。
        FileNotFoundError: 路径配置不存在。
        TypeError: 快照没有导出训练根配置。
    """

    if isinstance(config_ref, Path) or str(config_ref).endswith(".py"):
        return _load_path(Path(config_ref))
    try:
        module_name = _MODULES[str(config_ref)]
    except KeyError as exc:
        names = ", ".join(available_experiments())
        raise KeyError(f"unknown experiment {config_ref!r}; available: {names}") from exc
    return _build_preset(str(config_ref), importlib.import_module(module_name))


DEFAULT_EXPERIMENT_NAME = "geometry_ssl_multitask_representation_v0_7_3"


__all__ = [
    "DEFAULT_EXPERIMENT_NAME",
    "ExperimentPreset",
    "available_experiments",
    "load_experiment",
]
