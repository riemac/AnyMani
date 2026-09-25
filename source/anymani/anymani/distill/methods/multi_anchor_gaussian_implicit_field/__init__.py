"""Shared geometry-pretraining contracts used by the N040 method."""


from importlib import import_module
from typing import Any

from . import objectives as _objectives  # noqa: F401
from .config import (
    DensityObjectiveCfg,
    EntityPermutationCfg,
    FairGradCfg,
    JointConfigurationMeasureCfg,
    JointSignRewriteCfg,
    KappaObjectiveCfg,
    MultiAnchorGaussianObjectivesCfg,
)

__all__ = [
    "DensityObjectiveCfg",
    "EntityPermutationCfg",
    "FairGradCfg",
    "JointConfigurationMeasureCfg",
    "JointSignRewriteCfg",
    "KappaObjectiveCfg",
    "MultiAnchorGaussianMethod",
    "MultiAnchorGaussianMethodCfg",
    "MultiAnchorGaussianObjectivesCfg",
    "RetainedLoadReport",
    "load_retained_geometry_artifact",
]

_LAZY_EXPORTS = {
    "MultiAnchorGaussianMethod": (".method", "MultiAnchorGaussianMethod"),
    "MultiAnchorGaussianMethodCfg": (".config", "MultiAnchorGaussianMethodCfg"),
    "RetainedLoadReport": (".artifact", "RetainedLoadReport"),
    "load_retained_geometry_artifact": (".artifact", "load_retained_geometry_artifact"),
}


def __getattr__(name: str) -> Any:

    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
