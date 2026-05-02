r"""手部资产脚本入口层：quick 运行脚本 + `RecipeLoader`。

这里统一承载研究者会直接打开、直接运行或直接调用的脚本级入口：

1. `quick_pre_made.py`：完整 pre-made 拓扑枚举入口；
2. `quick_post_mutate.py`：对单个 pre-made topology 做独立后变异入口；
3. `RecipeLoader`：把 YAML / dict 解析成 `HandGeneratorCfg`。

# NOTE:
这里替代历史上的 `assets/tool/`。旧目录不再保留兼容壳，是为了避免同一件事
同时出现 `tool` 和 `scripts` 两套入口，降低科研脚本定位成本。

这样调整的原因，不是为了削减能力，而是为了让资产生成系统的三条路径
职责更清晰：

1. 正式整手主线：`HandGenerator`
2. 声明式配置装载：`RecipeLoader`
3. 研究 quick 脚本：`assets/scripts/quick_*.py`

模块分工
--------

- ``recipe_loader`` → YAML ↔ HandGeneratorCfg 双向转换，提供 schema 校验与历史字段兼容桥接
- ``quick_pre_made`` → 直接运行 pre-made 全量/局部枚举
- ``quick_post_mutate`` → 直接运行某一 topology 的后变异

使用示例
--------

纯 Python 驱动::

    from anymani.assets.scripts import RecipeLoader
    from anymani.assets.generator import HandGeneratorCfg, HandGenerator

    cfg = RecipeLoader.load("leap_variant.yaml")
    gen = HandGenerator(cfg)
"""

from .recipe_loader import RecipeLoader

__all__ = ["RecipeLoader"]
