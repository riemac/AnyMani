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

`MutatorBase` 只规定：

- 所有工具都有一个以 `HandCfg` 为入参、以 `HandCfg | None` 为出参的 `mutate` 方法
- 返回 `None` 表示该工具认为本次变异无效或必须拒绝（由流水线层决定如何处理拒绝）
"""

from __future__ import annotations

from ...asset_base import HandCfg


class MutatorBase:
    r"""所有后序变异工具的最小基类。

    子类只需实现 `mutate`，其余调度逻辑由 `HandMutator` 流水线负责。
    """

    def mutate(self, target: HandCfg) -> HandCfg | None:
        r"""对已构建的 `HandCfg` 执行一次局部变异。

        Args:
            target (HandCfg): 待变异的整手配置，通常来自 pre-made 阶段产物。

        Returns:
            HandCfg | None: 变异后的整手配置；返回 ``None`` 表示本次变异被拒绝。
        """

        raise NotImplementedError


__all__ = ["MutatorBase"]
