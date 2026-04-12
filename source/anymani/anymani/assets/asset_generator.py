r"""手部资产生成层的顶层抽象合同。

本文件在生成层扮演的角色，尽量与 `asset_builders.py` 保持同构：

- 这里定义**共享抽象合同**与**阶段级容器配置**；
- 真正的 joint / finger / palm / hand 级正文，下沉到 `generator/` 子目录；
- 顶层 `AssetGenerator` 只负责组织五阶段流水线，不在这里塞入具体枚举算法。

设计说明
--------

### 五阶段流水线

当前收敛后的生成流程为：

1. **Pre**：操作 builder recipe/cfg 树，负责 preset 扩展、joint delete、regroup。
2. **Build**：复用现有 `HandBuilderCfg -> HandCfg` 造骨架流程。
3. **Post**：操作 canonical `HandCfg` 树，负责后序拓扑/几何/limit/mount 派生。
4. **Validate**：在 `HandCfg` 上做 fatal invariant + soft report 检查。
5. **Export**：把通过的 `HandCfg` 导出为 URDF + metadata + debug 子资产。

### 文件组织

为避免 `asset_generator.py` 再次膨胀，真正的伪代码 skeleton 按层级拆分到：

- `generator/joint_generators.py`
- `generator/finger_generators.py`
- `generator/palm_generators.py`
- `generator/hand_generators.py`

因此，这里的类更多是在回答“流水线阶段如何拼装”，而不是“单层算法如何实现”。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .asset_base import AssetCfgBase, HandCfg
from .asset_builders import HandBuilderCfg
from .asset_exporters import HandExporterCfg
from .asset_validators import HandValidatorCfg


@dataclass
class GeneratorCfg(AssetCfgBase):
    r"""生成器配置基类。

    这里的“生成器”指的是生成层的一个局部操作算子：它可以是 pre 阶段的
    recipe 扩展器，也可以是 post 阶段的 `HandCfg` 派生器。
    """

    class_type: type["Generator"] | None = None
    """关联的生成器运行时类。"""

    name: str = "generator"
    """当前生成器节点的逻辑名称。用于 lineage / provenance 记录。"""


class Generator:
    r"""生成器基类。

    `Generator` 只表达“从一个输入对象派生出若干候选对象”的职责，不预设
    这个输入到底是 builder recipe 还是 canonical `HandCfg`。
    """

    cfg: GeneratorCfg

    def __init__(self, cfg: GeneratorCfg):
        self.cfg = cfg

    def generate(self, target: AssetCfgBase | None = None) -> list[AssetCfgBase] | None:
        r"""生成一批候选资产对象。

        Args:
            target (AssetCfgBase | None): 输入目标。pre 阶段通常是 builder cfg，
                post 阶段通常是 `HandCfg`。

        Returns:
            list[AssetCfgBase] | None: 生成结果列表。当前阶段只保留接口骨架。
        """
        pass

        # TODO:算法之一（局部生成器通用合同）
        # ────────────────────────────────────────
        # 输入
        #   `target`：上一阶段传入的 recipe 或 canonical asset。
        #   `self.cfg`：当前局部生成器的声明式参数。
        #
        # 输出：候选对象列表；对象类型由具体层级决定。
        #
        # ── 规范化输入 ──
        #   1. 若 `target is None`，则退回到 cfg 中携带的默认 recipe / default plan。
        #   2. 若 `target` 类型不符合当前生成器职责边界，则在实现阶段显式报错。
        #
        # ── 产生候选 ──
        #   1. 应优先产出“结构上合法、语义上可解释”的候选，而非无约束随机扰动。
        #   2. 所有候选都应保留 lineage 线索，便于后续 validator / exporter 记录 provenance。
        #
        # IDEA：局部生成器不直接决定流水线顺序；顺序由顶层 `AssetGenerator.generate()` 统一调度。


from .generator.finger_generators import (
    FingerPostGenerator,
    FingerPostGeneratorCfg,
    FingerPreGenerator,
    FingerPreGeneratorCfg,
)
from .generator.hand_generators import (
    GripperLikeHandPreGenerator,
    GripperLikeHandPreGeneratorCfg,
    HandPostGenerator,
    HandPostGeneratorCfg,
    HandPreGenerator,
    HandPreGeneratorCfg,
    HumanLikeHandPreGenerator,
    HumanLikeHandPreGeneratorCfg,
)
from .generator.joint_generators import (
    JointPostGenerator,
    JointPostGeneratorCfg,
    JointPreGenerator,
    JointPreGeneratorCfg,
)
from .generator.palm_generators import (
    PalmPostGenerator,
    PalmPostGeneratorCfg,
    PalmPreGenerator,
    PalmPreGeneratorCfg,
)


@dataclass
class PreAssetGeneratorCfg(AssetCfgBase):
    r"""前序生成阶段配置容器。

    `Pre` 只操作 builder recipe/cfg 树，不直接触碰 canonical `HandCfg`。这一步
    的目标是得到一批“准备交给 Build 的 blueprint”。
    """

    Joint: JointPreGeneratorCfg | None = None
    """joint-level 前序生成器。主要负责 canonical joint role 规范化等叶级准备。"""

    Finger: FingerPreGeneratorCfg | None = None
    """finger-level 前序生成器。主要负责 preset finger 的 joint delete / regroup。"""

    Palm: PalmPreGeneratorCfg | None = None
    """palm-level 前序生成器。主要负责 palm preset 池与 design-frame 约束。"""

    Hand: HandPreGeneratorCfg | None = None
    """hand-level 前序生成器。主要负责整手级 regroup、组合与过滤。"""


@dataclass
class PostAssetGeneratorCfg(AssetCfgBase):
    r"""后序生成阶段配置容器。

    `Post` 只操作 canonical `HandCfg` 树，不再重写 preset recipe。本阶段负责把
    已建好的 hand 视作一个可验证、可导出的对象，再做派生变异。
    """

    Joint: JointPostGeneratorCfg | None = None
    """joint-level 后序生成器。主要负责 limit / 局部几何等叶级派生。"""

    Finger: FingerPostGeneratorCfg | None = None
    """finger-level 后序生成器。主要负责 tip 替换、finger 几何派生等。"""

    Palm: PalmPostGeneratorCfg | None = None
    """palm-level 后序生成器。主要负责 palm 尺寸派生与挂载基准扰动接口。"""

    Hand: HandPostGeneratorCfg | None = None
    """hand-level 后序生成器。主要负责整手级派生、mutation budget 与 lineage 汇总。"""


@dataclass
class AssetGeneratorCfg(AssetCfgBase):
    r"""资产生成器配置类。

    与 `AssetGenerator` 一一对应，用于声明五阶段流水线的阶段入口，而不是把
    所有枚举算法混写到单一大对象里。
    """

    class_type: type["AssetGenerator"] | None = None
    """关联的资产生成器类。"""

    Pre: PreAssetGeneratorCfg = field(default_factory=PreAssetGeneratorCfg)
    """前序生成阶段入口。只消费 recipe/cfg 树。"""

    Build: HandBuilderCfg = field(default_factory=HandBuilderCfg)
    """手级构建器配置入口。负责把 blueprint 组装成 canonical `HandCfg`。"""

    Post: PostAssetGeneratorCfg = field(default_factory=PostAssetGeneratorCfg)
    """后序生成阶段入口。只消费 canonical `HandCfg`。"""

    Validate: HandValidatorCfg = field(default_factory=HandValidatorCfg)
    """手级验证器配置入口。"""

    Export: HandExporterCfg = field(default_factory=HandExporterCfg)
    """手级导出器配置入口。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = AssetGenerator


class AssetGenerator:
    r"""资产生成器。

    `AssetGenerator` 的职责是组织五阶段流水线，而不是取代各层 generator /
    builder / validator / exporter 的局部职责。
    """

    cfg: AssetGeneratorCfg

    def __init__(self, cfg: AssetGeneratorCfg):
        self.cfg = cfg

    def generate(self) -> list[HandCfg] | None:
        r"""生成一批整手资产。

        Returns:
            list[HandCfg] | None: 通过 build/post/validate 的候选 hand 列表。
                当前阶段只保留框架与伪算法，不写正式实现。
        """
        pass

        # TODO:算法之一（五阶段资产流水线）
        # ────────────────────────────────────────
        # 输入
        #   `self.cfg.Pre`      — recipe/cfg 树上的前序生成计划。
        #   `self.cfg.Build`    — canonical `HandCfg` 的装配入口。
        #   `self.cfg.Post`     — `HandCfg` 上的后序派生计划。
        #   `self.cfg.Validate` — hard error + soft report 的验证入口。
        #   `self.cfg.Export`   — `URDF + metadata + debug` 的导出入口。
        #
        # 输出：通过验证的 `HandCfg` 列表，以及由 exporter 落盘的资产集合。
        #
        # ── Phase 1：Pre（recipe 扩展） ──
        #   1. Joint/Finger/Palm/Hand 四层前序生成器依次作用在 builder recipe 树上。
        #   2. 这里的核心动作包括：preset 池扩展、joint delete、finger regroup、整手组合过滤。
        #   3. 输出是若干待 build 的 hand blueprint，而不是 `HandCfg`。
        #
        # ── Phase 2：Build（装配 canonical hand） ──
        #   1. 对每个 blueprint 调用 `HandBuilder(cfg).build()`。
        #   2. build 完成后得到 canonical `HandCfg`，后续阶段禁止再回写 recipe。
        #
        # ── Phase 3：Post（`HandCfg` 派生） ──
        #   1. 在 `HandCfg.copy()` / `replace()` 风格下做 topology/geometry/limit/mount 派生。
        #   2. post 的默认职责不是“再做一遍 preset delete”，而是对已建对象派生。
        #
        # ── Phase 4：Validate（硬错误 + 软报告） ──
        #   1. fatal invariant 直接 reject。
        #   2. soft report 留给研究分析与分布过滤，不强行中断流水线。
        #
        # ── Phase 5：Export（正式产物） ──
        #   1. 导出正式 URDF。
        #   2. 同步写 metadata sidecar 与 joint/finger/palm 级 debug 子资产。
        #   3. 以后实现时可选地做 `HandCfg -> URDF -> reparse` 的 round-trip 自检。
        #
        # IDEA：顶层流水线不应重新发明 joint/finger/palm/hand 的局部算法，
        #   只负责阶段顺序、候选汇总、过滤点与 provenance 记录。


__all__ = [
    "GeneratorCfg",
    "Generator",
    "PreAssetGeneratorCfg",
    "PostAssetGeneratorCfg",
    "AssetGeneratorCfg",
    "AssetGenerator",
    "JointPreGeneratorCfg",
    "JointPostGeneratorCfg",
    "FingerPreGeneratorCfg",
    "FingerPostGeneratorCfg",
    "PalmPreGeneratorCfg",
    "PalmPostGeneratorCfg",
    "HandPreGeneratorCfg",
    "HandPostGeneratorCfg",
    "JointPreGenerator",
    "JointPostGenerator",
    "FingerPreGenerator",
    "FingerPostGenerator",
    "PalmPreGenerator",
    "PalmPostGenerator",
    "HandPreGenerator",
    "HumanLikeHandPreGeneratorCfg",
    "HumanLikeHandPreGenerator",
    "GripperLikeHandPreGeneratorCfg",
    "GripperLikeHandPreGenerator",
    "HandPostGenerator",
]
