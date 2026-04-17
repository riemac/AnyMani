# FIXME：post-mutate 是基于 Monte Carlo Sampling 的联合采样思想来构造的
#
r"""后序变异流水线：用开放 term container 编排多个局部连续参数工具。

本模块对应 `pre-made -> validator -> HandCfg -> post-mutate -> HandCfg` 里的
post-mutate 阶段，但这里的职责现在被显式收窄为：

1. 持有一组声明式 `MutatorTerm`
2. 给出这些 term 的联合分布描述
3. 按 `order` 把上游已采样参数依次 lower 成确定性 hand 变换

# NOTE:
`joint_delete` 已明确回归 pre-made connectivity 主线，因此不再属于这里的 term；
`finger_replace` 当前仓内没有真实调用链，也不再纳入 post-mutate container。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Literal

from ...asset_base import AssetCfgBase, HandCfg
from ...validator import HandValidator, HandValidatorCfg
from ._base import MutatorBase
from ._distribution import ScalarDistributionCfg


@dataclass
class MutatorTerm(AssetCfgBase):
    r"""单个 post-mutate term 的声明式包装壳。

    用户真正声明的是 term 名字与其 `cfg`：

    ```python
    class MyMutatorCfg(HandMutatorCfg):
        scale = MutatorTerm(cfg=LinkScaleCfg(...))
        tip = MutatorTerm(cfg=TipReplaceCfg(...))
    ```

    这样新增工具时只需要：

    1. 写好新的 `*Cfg`
    2. 写好对应 `*Mutator`
    3. 在用户自己的 `HandMutatorCfg` 子类里把它挂成一个类属性

    而不需要再回头改 `HandMutatorCfg` 本身。
    """

    cfg: AssetCfgBase = ...
    """term 真正持有的具体 mutator cfg。"""

    def to_dict(self) -> dict[str, Any]:
        r"""把 term 序列化成 YAML / summary 友好的结构。"""

        return {
            "cfg_type": type(self.cfg).__name__,
            "cfg": self.cfg.to_dict(),
        }


@dataclass
class HandMutatorCfg(AssetCfgBase):
    r"""post-mutate 开放式 term container。

    与旧版“固定字段容器”不同，这个 cfg 不再预声明具体工具字段，而是只保留：

    - `order`
    - `on_reject`
    - `step_validate`
    - `terms`

    其中 `terms` 可以来自两条路径：

    1. Python 子类上的 `MutatorTerm` 类属性
    2. loader / 调试脚本直接传入的 `terms={...}`
    """

    class_type: type["HandMutator"] | None = None
    """关联的流水线运行时类。"""

    order: tuple[str, ...] = ()
    """term 执行顺序。空元组表示按声明顺序自动展开。"""

    on_reject: Literal["abort", "skip"] = "abort"
    """某个 term 返回 `None` 时的处理策略。"""

    step_validate: bool = False
    """是否在每个 term 执行后立刻跑一次 post-mutate 轻量校验。"""

    terms: dict[str, MutatorTerm] = field(default_factory=dict)
    """显式传入的 term 映射；主要服务 loader / 测试 / 动态构造场景。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = HandMutator

        merged_terms: "OrderedDict[str, MutatorTerm]" = OrderedDict()

        # 先按 MRO 自左向右收集类属性上的 term，保证“基类先、子类后”的声明顺序稳定。
        for cfg_class in reversed(type(self).__mro__):
            for name, value in cfg_class.__dict__.items():
                if isinstance(value, MutatorTerm):
                    merged_terms[name] = value.copy()

        # 再用显式传入的 `terms={...}` 覆盖同名项；这条路径更像 loader/runtime override。
        for name, term in self.terms.items():
            if not isinstance(term, MutatorTerm):
                raise TypeError(f"Mutator term {name!r} must be a MutatorTerm, got {type(term).__name__}")
            merged_terms[name] = term.copy()

        self.terms = dict(merged_terms)
        for name, term in self.terms.items():
            setattr(self, name, term)

        if not self.order:
            self.order = tuple(self.terms.keys())
        else:
            unknown_names = [name for name in self.order if name not in self.terms]
            if unknown_names:
                raise ValueError(f"Unknown mutator terms in order: {unknown_names}")
            remaining_names = [name for name in self.terms if name not in self.order]
            self.order = tuple((*self.order, *remaining_names))

    def has_terms(self) -> bool:
        r"""当前 container 是否至少启用了一个 term。"""

        return bool(self.terms)

    def ordered_terms(self) -> list[tuple[str, MutatorTerm]]:
        r"""按最终执行顺序返回 term 列表。"""

        return [(name, self.terms[name]) for name in self.order if name in self.terms]

    def to_dict(self) -> dict[str, Any]:
        r"""把开放式 term container 压平成 YAML / summary 友好的 mapping。"""

        dumped: dict[str, Any] = {
            "order": list(self.order),
            "on_reject": self.on_reject,
            "step_validate": self.step_validate,
        }
        for name, term in self.ordered_terms():
            dumped[name] = term.to_dict()
        return dumped


class HandMutator(MutatorBase):
    r"""post-mutate 流水线运行时壳。

    它现在只负责编排，不再拥有任何“固定工具表”知识：

    - term 的名字来自 `HandMutatorCfg`
    - term 的具体类型来自 `term.cfg.class_type`
    - term 的随机性来自上游注入的 `sampled_terms`
    """

    cfg: HandMutatorCfg

    def __init__(self, cfg: HandMutatorCfg):
        self.cfg = cfg

    def describe_sampling(self, target: HandCfg) -> dict[str, dict[str, ScalarDistributionCfg]]:
        r"""描述整个 post-mutate container 的独立联合分布。"""

        distribution_plan: dict[str, dict[str, ScalarDistributionCfg]] = {}
        for term_name, tool in self._build_tools():
            distribution_plan[term_name] = tool.describe_sampling(target)
        return distribution_plan

    def mutate(
        self,
        target: HandCfg,
        *,
        sampled_params: dict[str, dict[str, Any]] | None = None,
    ) -> HandCfg | None:
        r"""按 `order` 顺序执行 post-mutate term。

        Args:
            target (HandCfg): 当前 pre-made 基座 hand。
            sampled_params (dict[str, dict[str, Any]] | None): 上游已经采样好的
                `term_name -> {local_param_name -> sampled_value}` 映射。

        Returns:
            HandCfg | None: 成功时返回后序派生 hand；拒绝时返回 `None`。
        """

        current = target
        validator = HandValidator(HandValidatorCfg()) if self.cfg.step_validate else None
        sampled_terms = sampled_params or {}

        for term_name, tool in self._build_tools():
            result = tool.mutate(current, sampled_params=sampled_terms.get(term_name, {}))
            if result is None:
                if self.cfg.on_reject == "abort":
                    return None
                continue

            if validator is not None:
                validation = validator.validate_post_mutate(result)
                if not validation:
                    if self.cfg.on_reject == "abort":
                        return None
                    continue

            current = result

        return current

    def _build_tools(self) -> list[tuple[str, MutatorBase]]:
        r"""根据开放式 term container 动态实例化 mutator。"""

        tools: list[tuple[str, MutatorBase]] = []
        for term_name, term in self.cfg.ordered_terms():
            mutator_class = getattr(term.cfg, "class_type", None)
            if mutator_class is None:
                raise ValueError(f"Mutator term {term_name!r} has no class_type on cfg {type(term.cfg).__name__}")
            tools.append((term_name, mutator_class(term.cfg)))
        return tools


__all__ = ["MutatorTerm", "HandMutatorCfg", "HandMutator"]
