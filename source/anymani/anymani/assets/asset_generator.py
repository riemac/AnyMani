"""手部资产生成器的空骨架。

本文件回到你主导的顶层生成器定义，不再默认实现 build / validate /
export 的完整运行流程。当前只保留：

- 生成器配置对象 `AssetGeneratorCfg`
- 生成器运行时对象 `AssetGenerator`

以及它们与 Builder / Validator / Exporter 三类子系统的关系占位。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .asset_base import AssetCfgBase
from .asset_builders import HandBuilderCfg
from .asset_exporters import HandExporter
from .asset_validators import HandValidatorCfg


@dataclass
class AssetGeneratorCfg(AssetCfgBase):
    r"""资产生成器配置类。

    当前只保留生成器的骨架接口，不在这里默认决定批量生成算法、导出策略
    或验证规则之间的编排细节。
    """

    class_type: type["AssetGenerator"] | None = None
    """关联的资产生成器类。"""

    Build: HandBuilderCfg = field(default_factory=HandBuilderCfg)
    """手级构建器配置入口。"""

    Validate: HandValidatorCfg = field(default_factory=HandValidatorCfg)
    """手级验证器配置入口。"""

    Export: type[HandExporter] = HandExporter
    """手级导出器类入口。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = AssetGenerator


class AssetGenerator:
    r"""资产生成器。

    这里预期承担的是规模化资产生成职责，也就是把 Builder、Validator
    和 Exporter 组织起来，最终批量产出 `HandCfg`、URDF、yaml 以及
    相关附带资产。

    但当前阶段，这些编排细节仍由你主导，因此这里只保留运行时壳子。
    """

    cfg: AssetGeneratorCfg

    def __init__(self, cfg: AssetGeneratorCfg):
        self.cfg = cfg

    def generate(self) -> None:
        r"""生成一组手部资产。

        Raises:
            NotImplementedError: 当前只是生成器入口骨架，尚未填入真实实现。
        """

        raise NotImplementedError("AssetGenerator 目前只保留骨架，等待你的生成算法实现。")


__all__ = ["AssetGeneratorCfg", "AssetGenerator"]
