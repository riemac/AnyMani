r"""整手级生成器的声明式配置类和运行时类。

这是 generator 体系中与用户需求最直接对齐的一层：

- **Pre**：承接整手级 preset 池、thumb / non-thumb regroup、组合过滤；
- **Post**：在 canonical `HandCfg` 上做整手级派生与 lineage 汇总。

设计说明
--------

### 为什么 hand-level pre 是真正的“组合器”

finger-level pre 只负责产生单根 finger 候选；真正把这些候选拼成一只手，并施加
“至少几根手指”“thumb/non-thumb 如何分工”“是否允许 mixed family”等约束的，
应当是 hand-level pre。

### 与 builder/hand_builders.py 的关系

这里的 hand-level pre 不取代 `HumanLikeHandBuilder` / `GripperLikeHandBuilder` 的 build 职责。
它只是决定“要把哪些 recipe 送去 build”。真正构造 `HandCfg` 的动作仍由 builder 完成。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..asset_base import AssetCfgBase, HandCfg
from ..asset_builders import HandBuilderCfg
from ..asset_generator import Generator, GeneratorCfg


# ============================================================================
#  整手级前序 / 后序配置
# ============================================================================


@dataclass
class HandPreGeneratorCfg(GeneratorCfg):
    r"""整手级前序生成器配置基类。"""

    class_type: type["HandPreGenerator"] | None = None
    """关联的整手级前序生成器类。"""

    preset_names: list[str] = field(default_factory=list)
    """整手 preset 名称池。为空时表示以后实现时由上层实验配置给出。"""

    min_finger_count: int = 3
    """整手至少应保留的 finger 数量。默认 3。"""

    regroup_mode: Literal["cartesian", "subset"] = "cartesian"
    """finger 候选重新组装成整手 blueprint 的方式。默认做笛卡尔积。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = HandPreGenerator


@dataclass
class HumanLikeHandPreGeneratorCfg(HandPreGeneratorCfg):
    r"""类人手前序生成器配置。"""

    class_type: type["HumanLikeHandPreGenerator"] | None = None
    """关联的类人手前序生成器类。"""

    handedness_pool: tuple[Literal["left", "right"], ...] = ("right",)
    """左右手候选池。默认只生成右手。"""

    min_non_thumb_fingers: int = 2
    """非拇指手指最少保留数量。默认 2。"""

    allow_mixed_non_thumb_family: bool = False
    """是否允许不同 non-thumb family 混搭。默认关闭。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = HumanLikeHandPreGenerator


@dataclass
class GripperLikeHandPreGeneratorCfg(HandPreGeneratorCfg):
    r"""夹爪手前序生成器配置。"""

    class_type: type["GripperLikeHandPreGenerator"] | None = None
    """关联的夹爪手前序生成器类。"""

    num_fingers_pool: tuple[int, ...] = (3,)
    """夹爪手 finger 数量池。默认只生成 3 指夹爪。"""

    allow_rotational_symmetry: bool = True
    """是否优先保持环形/圆周对称布局。默认开启。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = GripperLikeHandPreGenerator


@dataclass
class HandPostGeneratorCfg(GeneratorCfg):
    r"""整手级后序生成器配置。"""

    class_type: type["HandPostGenerator"] | None = None
    """关联的整手级后序生成器类。"""

    allow_topology_mutation: bool = False
    """是否允许在 hand-level 进一步改整手拓扑。默认关闭。"""

    max_mutations_per_hand: int = 2
    """单只 hand 最多允许的整手级派生数。默认 2。"""

    emit_lineage_metadata: bool = True
    """是否在派生结果中记录 lineage / provenance。默认记录。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = HandPostGenerator


# ============================================================================
#  整手级前序 / 后序运行时类
# ============================================================================


class HandPreGenerator(Generator):
    r"""整手级前序生成器基类。"""

    cfg: HandPreGeneratorCfg

    def __init__(self, cfg: HandPreGeneratorCfg):
        self.cfg = cfg

    def generate(self, target: AssetCfgBase | None = None) -> list[HandBuilderCfg] | None:
        r"""把 finger / palm 候选重新组装成 hand blueprint。"""
        pass

        # TODO:算法之一（hand-level regroup 基类合同）
        # ────────────────────────────────────────
        # 输入
        #   `target` — 预期为 hand-level builder recipe，或更上层实验配置给出的 blueprint 种子。
        #   `preset_names` — 整手 preset 名称池。
        #   `regroup_mode` — finger 候选如何组合成整手 blueprint。
        #
        # 输出：一组待 build 的 `HandBuilderCfg` blueprint。
        #
        # ── 候选收集 ──
        #   1. 从 palm/finger 子层拿到已经预展开的候选池。
        #   2. 在 hand-level 统一做 thumb/non-thumb 或 gripper 对称布局的 regroup。
        #
        # ── 整手过滤 ──
        #   1. 至少满足 `min_finger_count`。
        #   2. 以后实现时可在这里加入 finger 间距、mount 冲突与 symmetry 约束。
        #
        # IDEA：hand-level pre 是“组合器”，不是 `HandBuilder.build()` 的替代品。


class HumanLikeHandPreGenerator(HandPreGenerator):
    r"""类人手前序生成器。"""

    cfg: HumanLikeHandPreGeneratorCfg

    def __init__(self, cfg: HumanLikeHandPreGeneratorCfg):
        self.cfg = cfg

    def generate(self, target: AssetCfgBase | None = None) -> list[HandBuilderCfg] | None:
        r"""生成类人手 blueprint。"""
        pass

        # TODO:算法之一（thumb / non-thumb regroup）
        # ────────────────────────────────────────
        # 输入
        #   `target` — 预期为类人手 `HandBuilderCfg` 种子。
        #   `preset_names` — 整手 preset 名称池，例如 `leap` / `allegro`。
        #   `handedness_pool` — 左右手候选。
        #   `min_non_thumb_fingers` — 非拇指最少保留数量。
        #
        # 输出：一组 `HumanLikeHandBuilderCfg` 风格的 blueprint。
        #
        # ── preset 池扩展 ──
        #   1. 从 preset 名称解析出 palm_cfg、finger_cfg、thumb_cfg 的种子组合。
        #   2. 非拇指 finger 候选来自 finger-level pre 的 delete/regroup 结果。
        #
        # ── 类人手 regroup ──
        #   1. 拇指单独占位；non-thumb 候选按 index/middle/ring/little 位置展开。
        #   2. 当 `regroup_mode="cartesian"` 时，对位置槽位做笛卡尔积；
        #      当 `regroup_mode="subset"` 时，仅做保序子集选择。
        #   3. 若 `allow_mixed_non_thumb_family=False`，则以后实现时限制 non-thumb 来自同一 family。
        #
        # ── 与 Get-Zero 的交叉验证 ──
        #   作为启发，LEAP family 可写成近似的：
        #   $$
        #   N_{hand} = N_{thumb} \times N_{index} \times N_{middle} \times N_{ring}
        #   $$
        #   但 AnyMani 的实现不直接绑定 `_2/_3/_4` 命名，也不直接做 XML 重写。
        #
        # IDEA：这一层正是用户明确要求前置的 `preset + joint delete + regroup` 的最终汇合点。


class GripperLikeHandPreGenerator(HandPreGenerator):
    r"""夹爪手前序生成器。"""

    cfg: GripperLikeHandPreGeneratorCfg

    def __init__(self, cfg: GripperLikeHandPreGeneratorCfg):
        self.cfg = cfg

    def generate(self, target: AssetCfgBase | None = None) -> list[HandBuilderCfg] | None:
        r"""生成夹爪手 blueprint。"""
        pass

        # TODO:算法之一（gripper-like 对称 regroup）
        # ────────────────────────────────────────
        # 输入
        #   `num_fingers_pool` — 夹爪手指数量候选。
        #   `allow_rotational_symmetry` — 是否保持圆周对称布局。
        #
        # 输出：一组 `GripperLikeHandBuilderCfg` 风格的 blueprint。
        #
        # ── 数量扩展 ──
        #   1. 对每个候选 `num_fingers` 生成一套对称 finger 布局 blueprint。
        #
        # ── 对称过滤 ──
        #   1. 若要求 rotational symmetry，则以后实现时在 mount 角度上施加均匀分布约束。
        #
        # IDEA：夹爪手的 pre 阶段重点不在 thumb/non-thumb，而在环形布局与数量枚举。


class HandPostGenerator(Generator):
    r"""整手级后序生成器。"""

    cfg: HandPostGeneratorCfg

    def __init__(self, cfg: HandPostGeneratorCfg):
        self.cfg = cfg

    def generate(self, target: AssetCfgBase | None = None) -> list[HandCfg] | None:
        r"""在 canonical `HandCfg` 上生成整手级派生候选。"""
        pass

        # TODO:算法之一（hand-level 后序派生）
        # ────────────────────────────────────────
        # 输入
        #   `target` — 预期为 canonical `HandCfg`。
        #   `allow_topology_mutation` — 是否开放更高层整手拓扑派生。
        #   `max_mutations_per_hand` — 整手 mutation budget。
        #
        # 输出：一组派生后的 `HandCfg`。
        #
        # ── 预算控制 ──
        #   1. 对 hand-level 派生数施加全局 budget，避免组合爆炸。
        #
        # ── 整手级派生 ──
        #   1. 可在这一层承接整根 finger 替换、整手 metadata 重标记、可选 wrist 链派生等高层操作。
        #   2. 这里默认不重复做 preset-derived joint delete；那个主路径已在 pre 完成。
        #
        # ── lineage 记录 ──
        #   1. 若 `emit_lineage_metadata=True`，则以后实现时把 pre/build/post 的来源一起写入 `HandCfg.metadata`。
        #
        # IDEA：hand-level post 更像“派生管理器”，而不是第二个 hand builder。


__all__ = [
    "HandPreGeneratorCfg",
    "HumanLikeHandPreGeneratorCfg",
    "GripperLikeHandPreGeneratorCfg",
    "HandPostGeneratorCfg",
    "HandPreGenerator",
    "HumanLikeHandPreGenerator",
    "GripperLikeHandPreGenerator",
    "HandPostGenerator",
]