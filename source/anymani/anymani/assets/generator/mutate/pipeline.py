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
from ...handedness import lower_hand_to_handedness
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

    class_type: type[HandMutator] | None = field(init=False, default=None, repr=False)
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

        ordered: OrderedDict[str, MutatorBaseCfg] = OrderedDict()

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

    所有 morphology 公式都定义在 canonical right-hand 空间。若输入是物理 left，
    pipeline 先通过严格 involution 恢复 canonical right，完成采样与联合 patch，
    再把结果反射回 left：

    $$
    H_L' = \mathcal M\!\left(
        F\!\left(\mathcal M(H_L);\,\xi\right)
    \right),
    $$

    其中 $\mathcal M^2=I$，$F$ 是 post-mutate 联合算子，$\xi$ 是已采样参数。
    因此 ``link_scale`` 的 CMC1 边界公式、``mount_perturb`` 的局部增量和
    ``tip_replace`` 的功能相位都只维护一份，不在每个 term 中复制 handedness 分支。
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
        r"""在 canonical right-hand morphology 上生成联合采样计划。

        左右镜像手必须共享同一随机变量定义。若 sampler 直接读取物理 left 的
        local frame，局部 $x$ 方向和 CMC1 边界会带入 handedness，导致相同 seed
        不再表示镜像 morphology。这里先 canonicalize，确保 paired sample 可审计。
        """

        canonical_target = _canonicalize_for_mutation(target)  # sampler 永远读取 right-hand frame 下的几何真源
        plan: dict[str, dict[str, Any]] = {}
        for name, cfg in self.cfg.ordered_terms():
            runtime = self._make_runtime(cfg)
            plan[name] = dict(runtime.describe_sampling(canonical_target))  # 同一 morphology 的左右手得到同构随机变量集合
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
        r"""基于同一个 canonical ``HandCfg`` 合成所有 term 的 deferred patch。

        该低层入口保留 ``HandPatch`` 的路径级冲突诊断，因此要求输入已处于
        canonical right-hand 空间。一般调用方应使用 ``mutate()``；它负责 left/right
        往返 lowering。generator 主链和批量入口均走 ``mutate()``。

        Raises:
            ValueError: 输入为物理 left 时抛出，防止把 right-hand patch 错施加到 left。
        """

        if target.handedness != "right":
            raise ValueError(
                "HandMutator.plan_patch expects canonical right-hand input; "
                "use HandMutator.mutate for handedness-aware post-mutate."
            )
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
        r"""在 canonical right 空间应用联合 patch，再恢复输入物理 handedness。

        Args:
            target (HandCfg): 物理 right 或严格镜像后的物理 left。
            sampled_params: 已确定的联合随机参数 $\xi$；同一 payload 对左右手表达
                同一个 canonical morphology 变体。

        Returns:
            HandCfg | None: 与输入 handedness 相同的变异结果；失败时返回 ``None``。
        """

        try:
            target_handedness = target.handedness  # 记录物理输出侧，canonicalization 后不能从临时对象猜测
            canonical_target = _canonicalize_for_mutation(target)  # left -> right，right -> 深拷贝
            canonical_mutated = self.plan_patch(
                canonical_target,
                sampled_params=sampled_params,
            ).apply(canonical_target)  # 所有 term 基于同一 canonical 真源联合规划并一次 apply
            return lower_hand_to_handedness(canonical_mutated, target_handedness)  # 恢复物理 handedness 与 same-$q$ 合同
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


def _canonicalize_for_mutation(target: HandCfg) -> HandCfg:
    r"""把物理手恢复到 post-mutate 唯一使用的 canonical right-hand 空间。

    Args:
        target (HandCfg): handedness 必须明确为 ``left`` 或 ``right`` 的物理手。

    Returns:
        HandCfg: canonical right-hand 深拷贝；对 left 执行一次严格整手反射。
    """

    return lower_hand_to_handedness(target, "right")  # $\mathcal M(H_L)=H_R$，同侧调用保持函数式副本


__all__ = ["HandMutatorCfg", "HandMutator"]
