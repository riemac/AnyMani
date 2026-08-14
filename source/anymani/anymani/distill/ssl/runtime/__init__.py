r"""Geometry SSL runtime 的稳定公开接口。

实现按职责分散在 ``assets``、``scheduler``、``objective``、``validation``、``checkpointing`` 与
``trainer``；包入口只 re-export window/q scheduler 的公共类型，避免 import ``runtime`` 时加载
HandBank、模型、TensorBoard 或训练器生命周期。
"""

from .scheduler import (
    GeometrySSLRuntimeCfg,
    GeometrySSLRuntimeState,
    ResidentGeometryAssetWindow,
    WindowedOnlineGeometryBatcher,
    runtime_state_from_dict,
)

__all__ = [
    "GeometrySSLRuntimeCfg",
    "GeometrySSLRuntimeState",
    "ResidentGeometryAssetWindow",
    "WindowedOnlineGeometryBatcher",
    "runtime_state_from_dict",
]
