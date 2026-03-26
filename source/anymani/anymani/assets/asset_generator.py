"""手部资产生成的顶层编排入口。

本模块刻意保持很薄：

- `asset_schema_*` 定义 canonical 声明层；
- `asset_builders.py` 负责组装 `HandCfg`；
- `asset_validators.py` 负责检查生成结果；
- `asset_exporters.py` 负责序列化生成结果。

`asset_generator.py` 只负责把这些子系统串起来。

这种拆分是架构性的，不是为了形式好看。把问题拆开之后，我们就能在
不同文件里分别回答下面四个问题：

- 什么是合法的 hand 描述？ -> schema
- 怎么组装出一个 hand？    -> builder
- 它需要满足哪些策略？    -> validator
- 怎么检查或落盘？        -> exporter
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .asset_builders import HandBuilder, HandBuilderCfg
from .asset_exporters import Exporter, ExporterCfg
from .asset_schema_core import AssetCfgBase
from .asset_schema_embodiment import HandCfg
from .asset_validators import Validator, ValidatorCfg


@dataclass
class AssetGeneratorCfg(AssetCfgBase):
    r"""生成器编排层的顶层配置。

    这个配置是 pipeline 的入口。它不会重新定义底层 schema 概念，
    只是把三个运行时阶段串起来，共同作用于 canonical :class:`HandCfg`。
    """

    class_type: type["AssetGenerator"] | None = None
    """关联的 asset generator 运行时类。"""

    builder: HandBuilderCfg = field(default_factory=HandBuilderCfg)
    """顶层 hand builder 配置。"""

    validator: ValidatorCfg = field(default_factory=ValidatorCfg)
    """顶层 validator 配置。"""

    exporter: ExporterCfg = field(default_factory=ExporterCfg)
    """顶层 exporter 配置。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = AssetGenerator


class AssetGenerator:
    r"""负责构建、验证并导出 hand 资产的顶层运行时对象。

    整个 pipeline 的返回值仍然是 :class:`HandCfg`。导出在这里被视为
    一种带副作用的检查 / 调试步骤，而不是新的权威表示。这样可以让
    canonical object 始终留在 schema 层，避免“runtime hand”和
    “declaration hand”之间悄悄漂移。
    """

    cfg: AssetGeneratorCfg

    def __init__(self, cfg: AssetGeneratorCfg):
        self.cfg = cfg
        # 运行时类保持可配置，这样流水线可以在不改编排外壳的情况下
        # 被替换或扩展。
        builder_type = cfg.builder.class_type or HandBuilder
        validator_type = cfg.validator.class_type or Validator
        exporter_type = cfg.exporter.class_type or Exporter
        self.builder = builder_type(cfg.builder)
        self.validator = validator_type(cfg.validator)
        self.exporter = exporter_type(cfg.exporter)

    def build(self) -> HandCfg:
        r"""从配置好的 builder 构建一个 :class:`HandCfg`。

        Returns:
            HandCfg: 已构建但尚未通过 validator 的 hand 资产。
        """

        return self.builder.build()

    def validate(self, hand: HandCfg) -> HandCfg:
        r"""验证一个已构建的 :class:`HandCfg`。

        Args:
            hand (HandCfg): 已构建的 hand 资产。

        Returns:
            HandCfg: 通过验证的 hand 资产。
        """

        return self.validator.validate(hand)

    def export(self, hand: HandCfg) -> dict[str, Any]:
        r"""导出一个已经通过验证的 :class:`HandCfg`。

        Args:
            hand (HandCfg): 已验证的 hand 资产。

        Returns:
            dict[str, Any]: 由配置的 exporter 生成的导出载荷。
        """

        return self.exporter.export(hand)

    def generate(self) -> HandCfg:
        r"""运行完整的 build -> validate -> export 流水线。

        Returns:
            HandCfg: 最终通过验证的 hand 对象。
        """

        hand = self.build()
        hand = self.validate(hand)
        # 导出目前只是一个带副作用的调试 / 序列化落点。
        # 流水线仍然返回经过验证的 canonical hand object，这样研究代码
        # 的其余部分可以继续直接围绕声明层对象工作。
        _ = self.export(hand)
        return hand


__all__ = ["AssetGeneratorCfg", "AssetGenerator"]
