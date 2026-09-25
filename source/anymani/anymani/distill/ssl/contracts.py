"""Build the configured pretraining and evaluation runtime."""


from __future__ import annotations

from typing import Any, ClassVar, Protocol, runtime_checkable


@runtime_checkable
class RuntimeBoundCfg(Protocol):


    runtime_type: ClassVar[type[Any]]


def build_runtime(config: RuntimeBoundCfg) -> Any:


    runtime_type = getattr(type(config), "runtime_type", None)
    if runtime_type is None or not callable(runtime_type):
        raise TypeError(f"pretraining config {type(config).__name__} does not declare a callable runtime_type")
    return runtime_type(config)


__all__ = [
    "RuntimeBoundCfg",
    "build_runtime",
]
