r"""后序变异工具的公共基础协议。

本模块只定义 `MutatorBase`——一个最小协议类，供所有后序变异工具壳继承。
它不承载任何配置逻辑，只确保所有工具都能被 `HandMutator` 流水线以统一接口
调度。

设计说明
--------

### 为什么用继承而不是 Protocol

当前规模下变异工具数量有限且实现语义相近，显式继承比纯 Protocol 更直接。
若后续工具数量激增或需要类型系统外部消费，可改为 `typing.Protocol`。

### 职责边界

`MutatorBase` 现在规定两层接口：

- `describe_sampling()`：给出本工具需要的独立采样维度及其分布描述
- `mutate()`：接收上游已经采样好的参数，执行一次确定性变换

返回 `None` 仍表示该工具认为本次变异无效或必须拒绝（由流水线层决定如何处理拒绝）。
"""

from __future__ import annotations

from typing import Any

from ...asset_base import HandCfg
from ._distribution import ScalarDistributionCfg


class MutatorBase:
    r"""所有后序变异工具的最小基类。

    子类只需实现 `mutate`，其余调度逻辑由 `HandMutator` 流水线负责。
    """

    def describe_sampling(self, target: HandCfg) -> dict[str, ScalarDistributionCfg]:
        r"""描述当前工具在给定 `HandCfg` 上需要的独立采样维度。

        Args:
            target (HandCfg): 当前 pre-made 基座 hand；分布维度通常依赖它实际有哪些
                fingers / joints / tip geometries。

        Returns:
            dict[str, ScalarDistributionCfg]: `局部参数名 -> 分布描述` 映射。
            空字典表示该工具当前不需要额外随机变量。
        """

        return {}

    def mutate(self, target: HandCfg, *, sampled_params: dict[str, Any] | None = None) -> HandCfg | None:
        r"""对已构建的 `HandCfg` 执行一次局部变异。

        Args:
            target (HandCfg): 待变异的整手配置，通常来自 pre-made 阶段产物。
            sampled_params (dict[str, Any] | None): 上游已经采样好的局部参数。
                key 的语义由各具体 mutator 自己定义。

        Returns:
            HandCfg | None: 变异后的整手配置；返回 ``None`` 表示本次变异被拒绝。
        """

        raise NotImplementedError


__all__ = ["MutatorBase"]
