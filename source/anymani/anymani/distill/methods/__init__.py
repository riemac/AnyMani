"""Task-free scientific method contracts used by the published training path."""


from importlib import import_module
from typing import Any

from .contracts import EmbodimentMethod, FeatureSpec, MethodStep, MethodUpdate

__all__ = [
    "EmbodimentMethod",
    "FeatureSpec",
    "MethodStep",
    "MethodUpdate",
    "MultiAnchorGaussianMethod",
    "MultiAnchorGaussianMethodCfg",
]

_LAZY_EXPORTS = {
    "MultiAnchorGaussianMethod": (".multi_anchor_gaussian_implicit_field.method", "MultiAnchorGaussianMethod"),
    "MultiAnchorGaussianMethodCfg": (".multi_anchor_gaussian_implicit_field.config", "MultiAnchorGaussianMethodCfg"),
}


def __getattr__(name: str) -> Any:
    """Load the shared base method API only when requested."""
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
