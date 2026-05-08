# generator/ 开发约定

本目录是多元化大批量资产生产的核心模块，会频繁迭代。以下约定是在这里工作时必须遵守的。

## 最高 façade

`HandGeneratorCfg` 是本模块唯一的最高配置入口。不允许再在它上面包装新的 cfg 类（如 `QuickRunCfg`）。所有生成逻辑的可配置项都应该收口到 `HandGeneratorCfg` 的字段里。

## hand_generator.py 的瘦身原则

`hand_generator.py` 是核心文件，只保留 `HandGeneratorCfg` 和 `HandGenerator` 这两个核心配置类/运行时类。helper 函数、工具函数、辅助数据结构不要写进 `hand_generator.py`，而是放到本目录下的其他文件（如 `_premade.py`、`_recolor.py`）或新建文件里。