r"""掌级生成器的声明式配置类和运行时类。

掌级生成器在当前阶段的职责相对克制：

- **Pre**：选择与规范化 palm preset 池；
- **Post**：对已建好的 `PalmCfg` 做局部尺寸或挂载基准派生接口。

这里故意不把真实的 palm 几何公式挪出 builder；掌级 generator 更像 preset / mutation
的调度节点，而不是掌部构造器本身。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..asset_base import AssetCfgBase, PalmCfg
from ..asset_builders import PalmBuilderCfg
from ..asset_generator import Generator, GeneratorCfg


@dataclass
class PalmPreGeneratorCfg(GeneratorCfg):
    r"""掌级前序生成器配置。"""

    class_type: type["PalmPreGenerator"] | None = None
    """关联的掌级前序生成器类。"""

    preset_names: list[str] = field(default_factory=list)
    """掌部 preset 名称池。

    例如 `leap` / `allegro`；为空时表示以后实现时由上层 hand preset 决定。"""

    preserve_original_frame: bool = True
    """是否保留 palm preset 的原始 design frame。默认保留。"""

    prefer_collision_first: bool = True
    """是否优先选择 collision-first 的 palm preset。默认启用。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = PalmPreGenerator


@dataclass
class PalmPostGeneratorCfg(GeneratorCfg):
    r"""掌级后序生成器配置。"""

    class_type: type["PalmPostGenerator"] | None = None
    """关联的掌级后序生成器类。"""

    allow_size_mutation: bool = False
    """是否允许在 palm-level 改基础尺寸。默认关闭。"""

    mount_translation_jitter_cm: float = 0.0
    """掌部挂载基准的平移微扰幅度（厘米）。默认不扰动。"""

    mount_yaw_jitter_rad: float = 0.0
    """掌部挂载基准的 yaw 微扰幅度（弧度）。默认不扰动。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = PalmPostGenerator


class PalmPreGenerator(Generator):
    r"""掌级前序生成器。"""

    cfg: PalmPreGeneratorCfg

    def __init__(self, cfg: PalmPreGeneratorCfg):
        self.cfg = cfg

    def generate(self, target: AssetCfgBase | None = None) -> list[PalmBuilderCfg] | None:
        r"""生成掌级前序候选。"""
        pass

        # TODO:算法之一（palm preset 池）
        # ────────────────────────────────────────
        # 输入
        #   `target` — 预期为 palm-level builder recipe，或由 hand preset 拆出的 palm recipe。
        #   `preset_names` — 可选的 palm preset 名称池。
        #
        # 输出：规范化后的 `PalmBuilderCfg` 候选列表。
        #
        # ── preset 选择 ──
        #   1. 若显式给出 `preset_names`，则以后实现时按名单展开 palm recipe 池。
        #   2. 若为空，则退回 hand-level preset 所附带的 palm 选择。
        #
        # ── frame 约束 ──
        #   1. 若 `preserve_original_frame=True`，则以后实现时不强制重写 preset 的原始 palm frame。
        #   2. 这与当前 `ComPalmBuilderCfg(preset=...)` 的思路一致。
        #
        # IDEA：掌级前序生成的主要价值是维持 palm preset 池与 mount baseline 的可枚举性。


class PalmPostGenerator(Generator):
    r"""掌级后序生成器。"""

    cfg: PalmPostGeneratorCfg

    def __init__(self, cfg: PalmPostGeneratorCfg):
        self.cfg = cfg

    def generate(self, target: AssetCfgBase | None = None) -> list[PalmCfg] | None:
        r"""在 canonical `PalmCfg` 上生成后序派生候选。"""
        pass

        # TODO:算法之一（palm-level 后序派生）
        # ────────────────────────────────────────
        # 输入
        #   `target` — 预期为 canonical `PalmCfg`。
        #   可调参数：
        #     `allow_size_mutation`
        #     `mount_translation_jitter_cm`
        #     `mount_yaw_jitter_rad`
        #
        # 输出：一组派生后的 `PalmCfg`。
        #
        # ── 尺寸派生 ──
        #   1. 若允许，则以后实现时围绕 palm 基础尺寸或 compound collision 布局做有限派生。
        #   2. 派生后仍需保持 finger mount baseline 的可解释性。
        #
        # ── 挂载基准派生 ──
        #   1. 对 palm metadata 中记录的 finger mount baseline 做小范围平移/yaw 扰动。
        #   2. 这一步只改基准，不直接重排整只手的 finger 组合。
        #
        # IDEA：掌级后序派生应保持设计帧稳定，避免把 palm 变成一个隐藏的手级 mutator。


__all__ = [
    "PalmPreGeneratorCfg",
    "PalmPostGeneratorCfg",
    "PalmPreGenerator",
    "PalmPostGenerator",
]