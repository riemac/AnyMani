r"""关节限位微调算子：在已有 HandCfg 上对 joint limit 做小范围参数修改。


"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
from ...asset_schema_core import Vector2

from ...asset_base import AssetCfgBase, HandCfg
from ._base import MutatorBase


# ============================================================================
#  配置类
# ============================================================================

# FIXME:
@dataclass
class LimitTweakCfg(AssetCfgBase):
    r"""关节限位微调工具配置。"""

    class_type: type["LimitTweakMutator"] | None = None
    """关联的运行时类。"""

    disturb_unit: Literal["deg", "rad"] = "deg"
    """微调范围的单位，默认为度。"""

    disturb_object: Literal["independent", "shared"] = "independent"
    """扰动对象。适用于所有活动关节。

    所有关节限位 $[q_{min},\ q_{max}]$ 都有 $q_{min}$ 和$q_{min}$ 两个边界：
    - independent: 微调可以独立 (independent) 地对 $q_{min}$ 和 $q_{min}$ 进行。
    - shared: 微调可以共享 (shared) 同一扰动值。
    """

    disturb_type: Literal["add", "scale"] = "add"
    """扰动类型。默认为添加。

    - `add`: 在原有值基础上添加
    - `scale`: 在原有值基础上按比例缩放
    """

    joint_range: Vector2 | None = None
    """关节限位微调范围配置，同时适用于所有活动关节。表示在原有 HandCfg 的 joint limit 基础上进行微调的范围。None 表示不进行操作

    - 当扰动类型为 `add` 时，可表示为 (-5, 5)。单位为 `deg` 表示原有基础上 ±5度 的范围内扰动
    - 当扰动类型为 `scale` 时，可表示为 (0.9, 1.1)，表示在原有基础上 ±10% 的范围内扰动。
    """

    clip: dict[str, float] | None = None
    """裁剪范围。默认不裁剪。

    - {"abs": 10}：表示微调后关节限位的绝对值不超过 10 度（disturb_unit 为 `deg` 时）。即 $\abs{q_{min}^\prime-q_{min}}\le10^\circ,\ \abs{q_{max}^\prime-q_{max}}\le10^\circ $
    - {"rel": 0.2}：表示微调后关节限位的相对值不超过 20%。即 $\abs{q_{min}^\prime-q_{min}}\le0.2\abs{q_{min}},\ \abs{q_{max}^\prime-q_{max}}\le0.2\abs{q_{max}} $
    """

    distrib: Literal["uniform", "normal"] | dict[str, Any] = "uniform"
    """分布类型。可选正态分布/均匀分布。适用于所选中的全部关节对象，但这不表示不同关节每次采样相同，他们有不同的种子数。

    支持以下两种输入格式：
    1. 字符串简写（使用默认参数）：
       - "uniform"：在 `link_size` 定义的范围内做均匀采样（默认）。
       - "normal"：以原尺寸为中心，`link_size` 定义的范围作为 ±3σ 的区间，做正态分布采样。

    2. 字典详细配置（用于自定义分布参数）：
       - {"type": "normal", "sigma_rule": 1}：使用 1σ 法则（即范围的半宽作为 1σ，分布更平缓，贴近均匀分布）。这个 sigma_rule 必须大于0
       - {"type": "normal", "sigma": 1/3}：直接指定 σ 为各关节限位范围半宽的 1/3, 相当于 3σ 法则。 $W=(b-a)/2, W=n\sigma, \sigma=W/n=(b-a)/2n$，
       - {"type": "uniform"}：等同于 "uniform"。
    """

    boundary_policy: Literal["none", "clip", "truncate", "resample"] | None = None
    r"""关节限位扰动的边界处理策略。

    该字段只规定采样结果超出 `joint_range` 或 `clip` 所定义边界时如何处理，
    不改变扰动类型 `disturb_type`，也不改变基础分布 `distrib`。

    - ``"none"``：不做额外边界处理，适合均匀分布已经严格落在合法区间内的情形。
    - ``"clip"``：把越界样本裁剪到边界上，实现简单，但会增加边界点的概率质量。
    - ``"truncate"``：直接使用截断分布采样，概率语义更干净。
    - ``"resample"``：拒绝越界样本并重新采样，即 rejection sampling。

    默认值为 ``None`` 时，可由运行时根据 `distrib` 自动选择：
    均匀分布通常等价于 ``"none"``；正态分布通常使用 ``"truncate"`` 或 ``"resample"``。
    """

    _distribution: Any = field(init=False, repr=False)
    """内部解析 disturb / distrib / boundary_policy 后生成的 scipy.stats 冻结分布对象，供运行时直接调用 .rvs()。

    封装了分布形态和采样策略的“采样器工厂（Sampler Callable），主要是因为不同分布类型和采样方法的差异，需要统一接口。
    """


class LimitTweakMutator(MutatorBase):
    r"""关节限位微调运行时壳。

    在已构建好的 `HandCfg` 上对目标关节的 `limit.lower` / `limit.upper`
    做小范围参数修改，不改变拓扑和几何。
    """

    cfg: LimitTweakCfg

    def __init__(self, cfg: LimitTweakCfg):
        self.cfg = cfg
        # 这里还需要读取具体拓扑的手再处理，因为不同关节拓扑的手可能有不同的自由度等，因此不能只读取 LinkScaleCfg
        #


__all__ = ["LimitTweakCfg", "LimitTweakMutator"]
