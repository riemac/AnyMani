r"""后序变异工具的公共基础协议。

本模块只定义 `MutatorBase`——一个最小协议类，供所有后序变异工具壳继承。
它不承载任何配置逻辑，只确保所有工具都能被 `HandMutator` 流水线以统一接口
调度。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import math
import random
from typing import Any

from ...asset_base import AssetCfgBase, HandCfg


@dataclass
class MutatorBaseCfg(AssetCfgBase):
    r"""所有后序变异算子配置的最小公共基类。

    这里的抽象关系对齐 IsaacLab 的 ``RewardsCfg`` / ``RewTerm``：

    - ``HandMutatorCfg`` 是开放容器，像 ``RewardsCfg`` 一样只负责声明有哪些项；
    - ``MutatorBaseCfg`` 自身就是 term，像 ``RewTerm`` 一样直接描述一个算子；
    - 不再额外包一层 ``MutatorTerm``，也不把所有算子压成统一分布配置类。

    各子类负责自己声明：
    - 变异作用于 HandCfg 的哪些属性路径
    - 采用什么分布及裁剪约束
    - per-entity 精调机制（例如按 joint child 语义名配置不同范围）
    - 统一暴露什么接口给用户配置使用
        - 一个约定，内部属性用 "_" 下划线前缀，用户配置接口后，可由 `__post_init__()` 解析
        - 用户好友的对外接口属性则不用 "_"
    """
    class_type: type["MutatorBase"] | None = field(init=False, default=None, repr=False)
    r"""内部运行时绑定字段，不作为研究配置接口的一部分。"""


@dataclass(frozen=True)
class SampleSpec:
    r"""pipeline 可批量采样的最小随机变量描述。

    各 mutator 仍保留自己的高层配置语义；只有在进入联合 Monte Carlo
    采样前，才把复杂配置 lowering 成若干 `SampleSpec`。这避免把所有
    post-mutate 算子硬压成同一种 public distribution cfg。

    科研上这里刻意不把“采样变量”提升成统一 public config，
    原因是每个算子本身的随机语义并不相同：
    - link_scale 的随机变量是长度比例 / 绝对增量；
    - mount_perturb 的随机变量是局部位姿增量；
    - limit_tweak 的随机变量是关节合法角域的微调量；
    - tip_replace 的随机变量则是 tip family / scale / geometry lowering。

    换句话说，`SampleSpec` 只是 pipeline 内部的运输盒子，不是研究接口。
    """

    name: str
    distribution: Any


@dataclass
class PatchOp:
    r"""一次延迟写入操作。

    `path` 是冲突检测和 metadata 的稳定语义键；`apply` 只在最终
    `HandPatch.apply()` 阶段作用于深拷贝后的 `HandCfg`。

    这里采用“先收集 patch、后一次性 apply”的原因不是为了抽象洁癖，
    而是为了避免 A term 先原地改掉对象后，B term 再基于被污染的中间态
    继续采样，最终把联合 Monte Carlo 语义退化成串行链式变换。
    """

    path: tuple[Any, ...]
    apply: Callable[[HandCfg], None] = field(repr=False)


@dataclass
class HandPatch:
    r"""post-mutate 的函数式 patch 容器。

    每个 term 只基于同一个原始 `HandCfg` 生成 patch；pipeline 负责把 patch
    组合后一次性应用，避免 A term 原地改完再把中间对象交给 B term。

    这相当于把所有算子的作用域固定在同一份原始几何图上：
    - 采样阶段只决定“要改什么”；
    - patch 阶段只描述“如何改”；
    - apply 阶段才真正把改动写回 HandCfg。

    这种分层对 post-mutate 尤其重要，因为不同算子可能同时触碰
    joint origin、limit、tip inertial、mount pose 等字段；若没有 patch
    合成层，冲突只能靠运行时偶然顺序发现。
    """

    ops: list[PatchOp] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(self, path: tuple[Any, ...], apply: Callable[[HandCfg], None]) -> None:
        self.ops.append(PatchOp(path=path, apply=apply))

    def extend(self, other: "HandPatch") -> None:
        self.ops.extend(other.ops)
        for key, value in other.metadata.items():
            if isinstance(self.metadata.get(key), dict) and isinstance(value, dict):
                merged = dict(self.metadata[key])  # type: ignore[index]
                merged.update(value)  # 当前只需要一层 term-name -> payload 的浅合并
                self.metadata[key] = merged
            else:
                self.metadata[key] = value

    def apply(self, target: HandCfg) -> HandCfg:
        mutated = target.copy()
        for op in self.ops:
            op.apply(mutated)
        if self.metadata:
            mutated.metadata = {**mutated.metadata, **self.metadata}  # patch metadata 需要真正写回 HandCfg，供 sidecar / summary 导出消费
        return mutated.replace(fingers=mutated.fingers, palm=mutated.palm, metadata=dict(mutated.metadata))


class MutatorBase:
    r"""所有后序变异算子的最小基类。

    位于 Sample 层级，负责：

    1. 读取原始 cfg 和当前 HandCfg 的几何 / 运动学上下文；
    2. 说明当前算子会采样哪些局部随机变量；
    3. 把这些变量 lower 成 deferred patch；
    4. 交给 HandMutator 做联合采样与一次性 apply。

    这个基类故意很薄，因为真正复杂的建模语义应写在具体 mutator 里。
    """

    cfg: MutatorBaseCfg

    def __init__(self, cfg: MutatorBaseCfg) -> None:
        self.cfg = cfg

    def describe_sampling(self, target: HandCfg) -> dict[str, Any]:
        r"""把当前算子的高层随机语义 lowering 成 pipeline 可采样变量。

        返回值的 key 只是一组局部随机变量名，value 可以是 callable、
        常量，或者 mutator 自己能理解的内部采样描述。pipeline 不解释
        这些 value 的物理意义，只负责统一调用。
        """

        return {}

    def plan_patch(self, target: HandCfg, sampled_params: dict[str, Any] | None = None) -> HandPatch:
        r"""基于同一个原始 `HandCfg` 生成延迟 patch。

        这里返回的是“修改计划”，不是立即写回对象。这样做可以让多个
        mutator 的 patch 先发生语义级合成，再在最后一步落盘式 apply。
        """

        return HandPatch()

    def mutate(self, target: HandCfg, *, sampled_params: dict[str, Any] | None = None) -> HandCfg | None:
        r"""兼容单算子直接调用：plan patch 后一次性 apply。

        单算子 direct-call 主要用于测试和快速验证；真实生成路径里通常
        还是由 HandMutator 先联合采样再统一 apply。
        """

        try:
            return self.plan_patch(target, sampled_params=sampled_params).apply(target)
        except Exception:
            return None


def _make_range_sampler(
    value_range: tuple[float, float],
    *,
    distrib: str | dict[str, Any] = "uniform",
    boundary_policy: str | None = None,
) -> Callable[[], float]:
    r"""把算子自己的高层范围字段 lowering 成私有采样 callable。

    这个 helper 不是 public distribution schema，只是运行时内部的数值采样器。
    public API 仍然是各 mutator 自己的 ``link_scale`` / ``pos_range`` /
    ``joint_range`` / ``scale`` 等字段；pipeline 只看到 callable 并在联合
    Monte Carlo 阶段调用它。

    这里之所以直接返回 callable，而不是返回某种统一分布对象，
    是因为不同算子的数值语义不同，既有区间采样，也有在
    几何约束下的截断采样。把它们统一成 public config 只会掩盖差异。
    """

    low, high = float(value_range[0]), float(value_range[1])  # 当前随机变量的合法区间
    if low > high:
        low, high = high, low  # 允许研究者临时写反范围，运行时按闭区间解释

    # 解析字符串或 dict 形式的分布描述；dict 只作为本算子内部语义，不提升为公共 Cfg。
    # 这里保留最小的接口面：只关心分布类型、sigma 规则和边界策略。
    distrib_type = distrib.get("type", "uniform") if isinstance(distrib, dict) else distrib
    distrib_type = str(distrib_type).lower()
    policy = boundary_policy or ("none" if distrib_type == "uniform" else "clip")

    def _project(sample: float) -> float:
        r"""按边界策略把样本投影回当前随机变量的合法区间。"""

        # 首版把 truncate / resample 都压成“有界投影”，原因不是数学上完全等价，
        # 而是为了让 quick path 保持确定、可解释，后续若要研究严格的 rejection
        # sampling，可再在内部 helper 里细分，而不暴露给 public config。
        if policy in {"clip", "truncate", "resample"}:
            return max(low, min(high, float(sample)))  # 首版 truncate/resample 统一为有界投影
        return float(sample)

    def _sample_uniform() -> float:
        r"""均匀采样 $x\sim U[a,b]$。"""

        return random.uniform(low, high)

    def _sample_normal() -> float:
        r"""正态采样，默认用区间半宽作为 $3\sigma$ 的数值锚点。"""

        # 这里使用区间中心作为均值、半宽反推 sigma。
        # 这对应用户常用的“给一个语义范围，再让分布在该范围内自然衰减”的研究习惯。
        center = 0.5 * (low + high)  # 区间中心，作为零偏扰动或 scale 中心
        half_width = max(0.5 * abs(high - low), 1e-12)  # 防止退化范围导致 sigma 为 0
        if isinstance(distrib, dict) and "sigma" in distrib:
            sigma = float(distrib["sigma"]) * half_width  # dict sigma 解释为半宽比例
        else:
            sigma_rule = float(distrib.get("sigma_rule", 3.0)) if isinstance(distrib, dict) else 3.0
            sigma = half_width / max(abs(sigma_rule), 1e-12)  # 默认 $3\sigma$ 覆盖配置范围
        return _project(random.gauss(center, sigma))

    if distrib_type == "normal":
        return _sample_normal
    if distrib_type == "uniform":
        return _sample_uniform
    raise ValueError(f"unsupported mutate distribution type: {distrib_type!r}")


def _sample_value(distribution: Any) -> Any:
    r"""pipeline 内部采样入口：callable 调用，常量原样返回。"""

    return distribution() if callable(distribution) else distribution


__all__ = ["HandPatch", "MutatorBase", "MutatorBaseCfg", "PatchOp", "SampleSpec"]
