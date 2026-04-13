r"""手部资产生成工具层：当前只保留 `RecipeLoader`。

这里的定位已经从“用户运行层”收敛为“内部配置装载层”。
也就是说，本子包现在只负责一件事：

1. 把 YAML / dict 解析成 `HandGeneratorCfg`

而不会再负责：

- 批量运行 orchestration
- 额外补 sidecar / tree 的落盘逻辑
- 作为顶层 CLI 入口直接驱动生成

这样调整的原因，不是为了削减能力，而是为了让资产生成系统的两条路径
职责更清晰：

1. 正式整手主线：`HandGenerator`
2. 声明式配置装载：`RecipeLoader`

模块分工
--------

- ``recipe_loader`` → YAML ↔ HandGeneratorCfg 双向转换，提供 schema 校验与历史字段兼容桥接

使用示例
--------

纯 Python 驱动::

    from anymani.assets.tool import RecipeLoader
    from anymani.assets.generator import HandGeneratorCfg, HandGenerator

    cfg = RecipeLoader.load("leap_variant.yaml")
    gen = HandGenerator(cfg)
"""

from .recipe_loader import RecipeLoader

__all__ = ["RecipeLoader"]
