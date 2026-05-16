r"""post-mutate 流水线：IsaacLab-style cfg 容器 + 联合 Monte Carlo ApplyOnce。

本模块对应 `pre-made -> validator -> HandCfg -> post-mutate -> validator -> HandCfg`
里的 post-mutate 阶段。抽象关系严格对齐用户笔记中的 IsaacLab 类比：

- ``HandMutatorCfg`` 相当于 ``RewardsCfg``，只是开放式声明容器；
- ``MutatorBaseCfg`` 相当于 ``RewTerm``，每个具体 mutator cfg 自己就是 term；
- pipeline 不拥有 ``terms`` 字段，不拥有 ``MutatorTerm``，也不理解统一分布 schema。

从工程优化角度，后变异仍分为 3 个阶段：

1. Declare：扫描 ``HandMutatorCfg`` 上声明的 ``MutatorBaseCfg``；
2. Sample：各 term 把自己的高层字段 lowering 成 callable / 常量，pipeline 联合采样；
3. Apply：所有 patch 基于同一个原始 ``HandCfg`` 生成，最后一次性 apply。

# NOTE:
`joint_delete` 已明确回归 pre-made connectivity 主线，因此不属于这里的 term。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from ...asset_base import AssetCfgBase, HandCfg
from .base import HandPatch, MutatorBase, MutatorBaseCfg, _sample_value


@dataclass
class HandMutatorCfg(AssetCfgBase):
    r"""post-mutate 开放式 term container。

    类似 IsaacLab 中 ``RewardsCfg`` 和 ``RewTerm`` 的关系：
    ``HandMutatorCfg`` 相当于 ``RewardsCfg``，``MutatorBaseCfg`` 相当于
    ``RewTerm``。因此启用哪些算子应写成类属性：

    ```python
    class MyMutateCfg(HandMutatorCfg):
        link_scale = LinkScaleCfg(...)
        mount_perturb = MountPerturbCfg(...)
    ```

    这里不再提供 ``terms``、``order``、``on_reject`` 等字段；这些属于
    运行策略或旧包装层，不应该污染科研配置接口。
    """

    class_type: type["HandMutator"] | None = field(init=False, default=None, repr=False)
    """内部运行时绑定字段，不参与用户配置初始化。"""

    def __post_init__(self) -> None:
        r"""补齐运行时类，保持和项目内其他 cfg 的 `class_type` 约定一致。"""

        if self.class_type is None:
            self.class_type = HandMutator

    def ordered_terms(self) -> list[tuple[str, MutatorBaseCfg]]:
        r"""按 class attribute 声明顺序返回启用的 post-mutate term。

        Python 3.7+ 保证类 ``__dict__`` 保留定义顺序，因此这里天然得到和
        IsaacLab cfg 相同的“从上到下声明即执行顺序”。实例属性允许覆盖同名
        类属性，但不引入额外 order 字段。
        """

        ordered: "OrderedDict[str, MutatorBaseCfg]" = OrderedDict()

        # 先扫描基类再扫描子类，保证子类同名声明覆盖父类默认项，同时保留子类声明顺序。
        for cls in reversed(type(self).mro()):
            for name, value in cls.__dict__.items():
                if name.startswith("_"):
                    continue
                if isinstance(value, MutatorBaseCfg):
                    ordered[name] = value.copy()

        # 再扫描实例字典，允许 recipe loader 或临时代码动态塞入 cfg 实例。
        for name, value in self.__dict__.items():
            if name.startswith("_") or name == "class_type":
                continue
            if isinstance(value, MutatorBaseCfg):
                ordered[name] = value

        return list(ordered.items())

    def has_terms(self) -> bool:
        r"""返回当前 cfg 是否声明了至少一个后变异算子。"""

        return bool(self.ordered_terms())


class HandMutator:
    r"""post-mutate 流水线运行时。

    runtime 只负责联合采样、patch 合成和一次性 apply；每个 mutator 的复杂
    分布语义、耦合关系、目标实体解析都留在对应 ``MutatorBase`` 子类内部。
    """

    cfg: HandMutatorCfg

    def __init__(self, cfg: HandMutatorCfg):
        r"""绑定一份已经声明好 term 的 ``HandMutatorCfg``。"""

        self.cfg = cfg

    def _make_runtime(self, cfg: MutatorBaseCfg) -> MutatorBase:
        r"""由 term cfg 创建对应 mutator runtime。"""

        runtime_cls = getattr(cfg, "class_type", None)
        if runtime_cls is None:
            raise TypeError(f"mutator cfg {cfg!r} does not define class_type")
        return runtime_cls(cfg)

    def describe_sampling(self, target: HandCfg) -> dict[str, dict[str, Any]]:
        r"""返回联合采样计划，结构为 ``term -> local variable -> callable/value``。"""

        plan: dict[str, dict[str, Any]] = {}
        for name, cfg in self.cfg.ordered_terms():
            runtime = self._make_runtime(cfg)
            plan[name] = dict(runtime.describe_sampling(target))
        return plan

    def sample_batch(self, target: HandCfg, *, batch_size: int) -> list[dict[str, dict[str, Any]]]:
        r"""采样 ``batch_size`` 组互相独立的联合后变异参数。"""

        sample_plan = self.describe_sampling(target)
        batch: list[dict[str, dict[str, Any]]] = [
            {term_name: {} for term_name in sample_plan}
            for _ in range(max(int(batch_size), 0))
        ]
        for term_name, distribution_map in sample_plan.items():
            for local_name, distribution in distribution_map.items():
                for sample in batch:
                    sample[term_name][local_name] = _sample_value(distribution)
        return batch

    def plan_patch(
        self,
        target: HandCfg,
        *,
        sampled_params: dict[str, dict[str, Any]] | None = None,
    ) -> HandPatch:
        r"""基于同一个原始 ``HandCfg`` 合成所有 term 的 deferred patch。"""

        sampled_params = sampled_params or {}
        composed = HandPatch()
        touched_paths: set[tuple[Any, ...]] = set()
        for name, cfg in self.cfg.ordered_terms():
            runtime = self._make_runtime(cfg)
            patch = runtime.plan_patch(target, sampled_params=sampled_params.get(name, {}))
            for op in patch.ops:
                if op.path in touched_paths:
                    raise ValueError(f"post-mutate patch conflict at path {op.path!r}")
                touched_paths.add(op.path)
            composed.extend(patch)
        return composed

    def mutate(
        self,
        target: HandCfg,
        *,
        sampled_params: dict[str, dict[str, Any]] | None = None,
    ) -> HandCfg | None:
        r"""一次性应用合成 patch；失败时返回 ``None``，交给 validator/refill 层补采。"""

        try:
            return self.plan_patch(target, sampled_params=sampled_params).apply(target)
        except Exception:
            return None

    def mutate_batch(
        self,
        target: HandCfg,
        *,
        sampled_batch: list[dict[str, dict[str, Any]]] | None = None,
        batch_size: int | None = None,
    ) -> list[tuple[HandCfg | None, dict[str, dict[str, Any]]]]:
        r"""批量 helper：采样可并行，Python object patch apply 保持确定性串行。"""

        if sampled_batch is None:
            sampled_batch = self.sample_batch(target, batch_size=int(batch_size or 1))
        results: list[tuple[HandCfg | None, dict[str, dict[str, Any]]]] = []
        for sampled_params in sampled_batch:
            results.append((self.mutate(target, sampled_params=sampled_params), sampled_params))
        return results


__all__ = ["HandMutatorCfg", "HandMutator"]
