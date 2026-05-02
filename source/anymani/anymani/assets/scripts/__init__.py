r"""手部资产脚本入口层：quick 运行脚本。

这里统一承载研究者会直接打开、直接运行或直接调用的脚本级入口：

1. `quick_pre_made.py`：完整 pre-made 拓扑枚举入口；
2. `quick_post_mutate.py`：对单个 pre-made topology 做独立后变异入口。

# NOTE:
这里替代历史上的 `assets/tool/`。旧目录不再保留兼容壳，是为了避免同一件事
同时出现 `tool` 和 `scripts` 两套入口，降低科研脚本定位成本。

这样调整的原因，不是为了削减能力，而是为了让资产生成系统的三条路径
职责更清晰：

1. 正式整手主线：`HandGenerator`
2. 声明式配置装载：`generator/_recipe_loader.py`
3. 研究 quick 脚本：`assets/scripts/quick_*.py`

模块分工
--------

- ``quick_pre_made`` → 直接运行 pre-made 全量/局部枚举
- ``quick_post_mutate`` → 直接运行某一 topology 的后变异

使用示例
--------

纯 Python 驱动::

    from anymani.assets.generator import HandGeneratorCfg, HandGenerator

    cfg = HandGeneratorCfg(mode="made")
    gen = HandGenerator(cfg)
"""
