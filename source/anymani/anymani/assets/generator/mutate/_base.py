r"""后序变异工具的公共基础协议。

本模块只定义 `MutatorBase`——一个最小协议类，供所有后序变异工具壳继承。
它不承载任何配置逻辑，只确保所有工具都能被 `HandMutator` 流水线以统一接口
调度。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ...asset_base import AssetCfgBase, HandCfg


@dataclass
class MutatorBaseCfg(AssetCfgBase):
    """所有后序变异算子配置的最小公共基类。

    为 MutatorTerm.cfg 提供类型收窄——
    所有合法的 mutator cfg 都必须有 class_type，
    而 class_type 的内部复杂行为由各子类自行掌控。

    各子类负责自己声明：
    - 变异作用于 HandCfg 的哪些属性路径
    - 采用什么分布及裁剪约束
    - per-entity 精调机制（如 per_joint_delta_distribution）
    - 统一暴露什么接口给用户配置使用
        - 一个约定，内部属性用 "_" 下划线前缀，用户配置接口后，可由 `__post_init__()` 解析
        - 用户好友的对外接口属性则不用 "_"
    """
    class_type: type["MutatorBase"] | None = None


@dataclass(frozen=True)
class SampleSpec:
    r"""pipeline 可批量采样的最小随机变量描述。

    各 mutator 仍保留自己的高层配置语义；只有在进入联合 Monte Carlo
    采样前，才把复杂配置 lowering 成若干 `SampleSpec`。这避免把所有
    post-mutate 算子硬压成同一种 public distribution cfg。
    """

    name: str
    distribution: Any


@dataclass
class PatchOp:
    r"""一次延迟写入操作。

    `path` 是冲突检测和 metadata 的稳定语义键；`apply` 只在最终
    `HandPatch.apply()` 阶段作用于深拷贝后的 `HandCfg`。
    """

    path: tuple[Any, ...]
    apply: Callable[[HandCfg], None] = field(repr=False)


@dataclass
class HandPatch:
    r"""post-mutate 的函数式 patch 容器。

    每个 term 只基于同一个原始 `HandCfg` 生成 patch；pipeline 负责把 patch
    组合后一次性应用，避免 A term 原地改完再把中间对象交给 B term。
    """

    ops: list[PatchOp] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(self, path: tuple[Any, ...], apply: Callable[[HandCfg], None]) -> None:
        self.ops.append(PatchOp(path=path, apply=apply))

    def extend(self, other: "HandPatch") -> None:
        self.ops.extend(other.ops)
        self.metadata.update(other.metadata)

    def apply(self, target: HandCfg) -> HandCfg:
        mutated = target.copy()
        for op in self.ops:
            op.apply(mutated)
        return mutated.replace(fingers=mutated.fingers, palm=mutated.palm, metadata=dict(mutated.metadata))


class MutatorBase:
    r"""所有后序变异算子的最小基类。

    位于 Sample 层级，负责：读原始 cfg 和 HandCfg 相关属性，返回要修改的属性，以及采样到的参数，后续交由 HandMutator 并行 apply
    """

    cfg: MutatorBaseCfg

    def __init__(self, cfg: MutatorBaseCfg) -> None:
        self.cfg = cfg

    def describe_sampling(self, target: HandCfg) -> dict[str, Any]:
        r"""把当前算子的高层随机语义 lowering 成 pipeline 可采样变量。"""

        return {}

    def plan_patch(self, target: HandCfg, sampled_params: dict[str, Any] | None = None) -> HandPatch:
        r"""基于同一个原始 `HandCfg` 生成延迟 patch。"""

        return HandPatch()

    def mutate(self, target: HandCfg, *, sampled_params: dict[str, Any] | None = None) -> HandCfg | None:
        r"""兼容单算子直接调用：plan patch 后一次性 apply。"""

        try:
            return self.plan_patch(target, sampled_params=sampled_params).apply(target)
        except Exception:
            return None

__all__ = ["HandPatch", "MutatorBase", "MutatorBaseCfg", "PatchOp", "SampleSpec"]
