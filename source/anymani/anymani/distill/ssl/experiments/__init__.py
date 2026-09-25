"""Explicit registry for the published N040 pretraining snapshot. Custom Python snapshot paths remain supported by the loader."""


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


    name: str
    module_name: str
    module: ModuleType
    pretrain: Any
    evaluation: Any | None
    path: Path
    config_sha256: str


_MODULES: dict[str, str] = {
    "geometry_ssl_density_material_jacobian_se3_v0_8_1": (
        "anymani.distill.ssl.experiments.geometry_ssl_density_material_jacobian_se3_v0_8_1"
    ),
}


def available_experiments() -> tuple[str, ...]:


    return tuple(_MODULES)


def _module_file(module: ModuleType) -> Path:


    module_file = getattr(module, "__file__", None)
    if module_file is None:
        raise ValueError(f"experiment module {module.__name__!r} has no source file")
    return Path(module_file).resolve()


def _build_preset(name: str, module: ModuleType) -> ExperimentPreset:


    path = _module_file(module)
    if not hasattr(module, "EXPERIMENT"):
        raise TypeError(f"experiment snapshot {path} must export EXPERIMENT")
    return ExperimentPreset(
        name=name,
        module_name=module.__name__,
        module=module,
        pretrain=module.EXPERIMENT,
        evaluation=getattr(module, "EVALUATION_EXPERIMENT", None),
        path=path,
        config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _load_path(path: Path) -> ExperimentPreset:


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


    if isinstance(config_ref, Path) or str(config_ref).endswith(".py"):
        return _load_path(Path(config_ref))
    try:
        module_name = _MODULES[str(config_ref)]
    except KeyError as exc:
        names = ", ".join(available_experiments())
        raise KeyError(f"unknown experiment {config_ref!r}; available: {names}") from exc
    return _build_preset(str(config_ref), importlib.import_module(module_name))


DEFAULT_EXPERIMENT_NAME = "geometry_ssl_density_material_jacobian_se3_v0_8_1"


__all__ = [
    "DEFAULT_EXPERIMENT_NAME",
    "ExperimentPreset",
    "available_experiments",
    "load_experiment",
]
