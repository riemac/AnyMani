# FIXME: 配置类继承 MutatorBaseCfg
r"""连杆长度缩放变异算子：在已有 HandCfg 上对 link 两岸距离做 ± 扰动。

需结合 `AnyMani/source/anymani/anymani/assets/doc/长度变异示意.jpg` 理解
"""

from __future__ import annotations

from dataclasses import dataclass, field, MISSING
import math
from typing import Any, Literal
from ...asset_schema_core import Vector2, Vector6

from ...asset_base import AssetCfgBase, HandCfg
from ...asset_schema_core import PoseCfg
from ._base import MutatorBaseCfg, MutatorBase


# ============================================================================
#  配置类
# ============================================================================

# FIXME: 暴露统一对外配置接口，便于用户使用，由 __post_init__() 解析后赋给内部统一规范属性
# 需求: 这个是服务于修改连杆尺寸的变异算子配置类
# 输入: 有很多不同种的模式，应对不同情况和精细度的需求
# mode1: 统一修改连杆长度，设置连杆长度和分布类型、分布参数 $l=(l_{min},\ l_{max}),\ D_{type},\ D_{params}$。默认用均匀分布
# mode2: 按 joint 类型，每个类型单独设置分布类型和参数，如 $l=(l_{min}^{MCP1},\ l_{max}^{MCP1}),\ D_{type}^{MCP1},\ D_{params}^{MCP1}$。默认用均匀分布
# 此外，也允许对宽度、高度设置变异，提供更细粒度的控制，注意这里是 link mesh 为 box 类型的才可以。但默认为长度，因为这个属性用的最多，也适用于 cylinder
#
class LinkScaleCfg(MutatorBaseCfg):
    r"""连杆尺寸缩放算子配置类。

    主要是对 pre-made 产生后的某具体拓扑灵巧手 (HandCfg) 其手指 joint/child link 的尺寸进行缩放变异，不包括手掌和指尖（leap non-thumb finger 的固定部分也被视为手掌的一部分）。
    缩放变异是在原有 HandCfg 基础上，对其尺寸进行扰动变异，是增量型的，而非重新在指定范围赋予新的尺寸值
    """

    class_type: type["LinkScaleMutator"] = LinkScaleMutator
    """关联的运行时类。"""

    link_type: str = "box"
    """Joint/child link mesh 的种类。默认为 urdf 中最常见的 "box"。这个需要和实际 HandCfg/urdf 对应"""

    scale_type: Literal["abs", "rel"] = "rel"
    """缩放语义：是绝对长度扰动(cm)，还是相对比例扰动(%)。默认为相对比例扰动。"""

    link_scale: Vector2 | Vector6 | dict[str, Vector2 | Vector6]  = field(default=MISSING)
    """连杆尺寸变异范围配置。可以是单一的 `(min, max)`，也可以是针对不同 joint type 的细粒度配置字典。也可以控制宽度和高度。

    - Vector2: $l=(l_{min},\ l_{max})$. 适用于所有 joint/child link 的尺寸修改范围。例如
        - (-1, 1)，且缩放语义为 `abs`时，为表示长度在原基础上 ±1cm 的范围内扰动。
        - (0.9, 1.1)，且缩放语义为 `rel`时，为表示长度在原基础上 ±10% 的范围内扰动。
    - Vector6: $l=(l_{min},\ l_{max},\ w_{min},\ w_{max},\ h_{min},\ h_{max})$，适用于同时控制长度、宽度和高度的范围。注意这里是仅适用于 "box" 的连杆类型
    - dict[str, Vector2 | Vector6]: 针对不同 joint type 的细粒度配置字典。
        > 这里的 joint type 匹配是一个复杂点，可能涉及到需要重构 HandCfg 的相关属性。
        > 目前的产物 `AnyMani/source/anymani/anymani/assets/generated/2026-04-17_13-39-16/single_palm_leap/left_t3_i1_m4_r4/02f3d7f4/hand.urdf`
        > 观察它的 link 名称，一个想法是 child link 名称的精准匹配。首先匹配时要分 thumb finger 和 non-thumb finger
        > 例如典型真实人手的 thumb finger，有 CMC, MCP, DIP 3种 joint 类型。而 non-thumb finger 包括 MCP, PIP, DIP 3种 joint 类型。
        > 而像 leap/allegro, 我这里对 thumb 的 CMC 还进一步细化为 CMC1/CMC2, non-thumb 的 MCP 细化为 MCP1/MCP2
        > 其中两种类型重名，但其中 MCP 的语义和实际运动能力在 thumb 和 non-thumb 并不一致。
        > 但目前 HandCfg 相关属性不包含这些信息。仅 urdf 产物里的命名对 link 命名显示指定，建议在 HandCfg 里设置相关属性。
        > 综合来说，这里的目的是跨手指同类型关节（通过名称匹配机制）的统一变异范围处理。
    """

    clip: Vector2 | Vector6 | dict[str, Vector2 | Vector6] | None = None
    """变动幅度裁剪范围，是对尺寸增量的绝对值 (cm)限制。不论缩放是长度扰动还是相对比例扰动皆一致。

    - Vector2: $(\Delta s_{min}, \Delta s_{max})$，适用于所有 joint/child link 的所有尺寸类型（长宽高）变动幅度裁剪范围。例如 (-0.5, 1) 表示变化在 [原尺寸-0.5, 原尺寸+1.5]cm 范围内
    - Vector6: $(\Delta l_{min}, \Delta l_{max}, \Delta w_{min}, \Delta w_{max}, \Delta h_{min}, \Delta h_{max})$，适用于分别控制长度、宽度和高度的变动幅度裁剪范围。注意这里是仅适用于 "box" 的连杆类型
    - dict[str, Vector2 | Vector6]: 针对不同 joint type 的细粒度配置字典，语义同上 link_scale 的 dict 配置
    - None: 不进行额外裁剪
    """

    distrib: Literal["uniform", "normal"] | dict[str, Any] = "uniform"
    """分布类型。可选正态分布/均匀分布。适用于所选中的全部关节对象，但这不表示不同关节每次采样相同，他们有不同的种子数。

    支持以下两种输入格式：
    1. 字符串简写（使用默认参数）：
       - "uniform"：在 `link_size` 定义的范围内做均匀采样（默认）。
       - "normal"：以原尺寸为中心，`link_size` 定义的范围作为 ±3σ 的区间，做正态分布采样。

    2. 字典详细配置（用于自定义分布参数）：
       - {"type": "normal", "sigma_rule": 1}：使用 1σ 法则（即范围的半宽作为 1σ，分布更平缓，贴近均匀分布）。这个 sigma_rule 必须大于0
       - {"type": "normal", "sigma": 1/3}：直接指定 σ 为各关节子连杆的尺寸范围半宽的 1/3, 相当于 3σ 法则。 $W=(b-a)/2, W=n\sigma, \sigma=W/n=(b-a)/2n$，
       - {"type": "uniform"}：等同于 "uniform"。
    """

    boundary_policy: Literal["none", "clip", "truncate", "resample"] | None = None
    r"""连杆尺寸扰动的边界处理策略。

    该字段只规定采样得到的尺寸扰动超出 `link_scale` 或 `clip` 所定义边界时如何处理，
    不改变扰动语义 `scale_type`，也不改变基础分布 `distrib`。

    - ``"none"``：不做额外边界处理，适合均匀分布已经严格落在合法区间内的情形。
    - ``"clip"``：把越界样本裁剪到边界上，实现简单，但会增加边界点的概率质量。
    - ``"truncate"``：直接使用截断分布采样，概率语义更干净。
    - ``"resample"``：拒绝越界样本并重新采样，即 rejection sampling。

    默认值为 ``None`` 时，可由运行时根据 `distrib` 自动选择：
    均匀分布通常等价于 ``"none"``；正态分布通常使用 ``"truncate"`` 或 ``"resample"``。
    """

    _link_meshes: list[Any] = field(default_factory=list)
    """内部使用的 link mesh 列表。无论是配置长度、宽度还是某类关节类型，最后都经__post_init__()解析成为内部的某种规范统一表示，并交由运行时类处理"""

    _distribution: Any = field(init=False, repr=False)
    """内部解析 disturb / distrib / boundary_policy 后生成的 scipy.stats 冻结分布对象，供运行时直接调用 .rvs()。

    封装了分布形态和采样策略的“采样器工厂（Sampler Callable），主要是因为不同分布类型和采样方法的差异，需要统一接口。
    """

    def __post_init__(self):
        pass


# ============================================================================
#  运行时壳
# ============================================================================


class LinkScaleMutator(MutatorBase):
    r"""连杆长度缩放运行时壳。

    在已构建好的 `HandCfg` 上对指定（或全部）关节的 `origin.pos` 做
    方向保持的等比缩放，不改变拓扑与旋转。
    """

    cfg: LinkScaleCfg

    def __init__(self, cfg: LinkScaleCfg):
        self.cfg = cfg
        # 这里还需要读取具体拓扑的手再处理，因为不同关节拓扑的手可能有不同的自由度等，因此不能只读取 LinkScaleCfg


__all__ = ["LinkScaleCfg", "LinkScaleMutator"]
