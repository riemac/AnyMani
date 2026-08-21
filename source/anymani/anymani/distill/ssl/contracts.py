r"""Embodiment pretraining 组件装配与跨阶段数据语义合同。

本模块只定义稳定边界，不定义 hand、Gaussian field、网络结构或损失公式。具体配置通过
``ClassVar runtime_type`` 绑定本地运行时；Hydra 只组合 concrete dataclass，最高 façade 不维护
``kind -> constructor`` 注册表，也不解析任何组件内部字段。
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol, runtime_checkable


@runtime_checkable
class RuntimeBoundCfg(Protocol):
    r"""所有可由最高 façade 构造的 concrete 配置必须满足的最小协议。"""

    runtime_type: ClassVar[type[Any]]  # 不进入 Hydra/YAML，只绑定同一 owner 的 runtime


def build_runtime(config: RuntimeBoundCfg) -> Any:
    r"""只通过 concrete cfg 自己声明的 ``runtime_type`` 构造运行时。

    Args:
        config (CfgT): Hydra compose 后恢复的 concrete dataclass。

    Returns:
        Any: 对应 role runtime；具体类型由调用方的 role contract 收窄。

    Raises:
        TypeError: 配置未声明可调用的 ``runtime_type`` 时抛出。
    """

    runtime_type = getattr(type(config), "runtime_type", None)  # ClassVar 必须来自 concrete cfg 类型
    if runtime_type is None or not callable(runtime_type):
        raise TypeError(f"pretraining config {type(config).__name__} does not declare a callable runtime_type")
    return runtime_type(config)  # 所有 runtime 构造阶段只保存 cfg，不执行 IO/CUDA


__all__ = [
    "RuntimeBoundCfg",
    "build_runtime",
]
