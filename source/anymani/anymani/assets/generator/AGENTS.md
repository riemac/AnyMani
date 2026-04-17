# generator/ 开发约定

本目录是多元化大批量资产生产的核心模块，会频繁迭代。以下约定是在这里工作时必须遵守的。

## 最高 façade

`HandGeneratorCfg` 是本模块唯一的最高配置入口。不允许再在它上面包装新的 cfg 类（如 `QuickRunCfg`）。所有生成逻辑的可配置项都应该收口到 `HandGeneratorCfg` 的字段里。

## hand_generator.py 的瘦身原则

`hand_generator.py` 是核心文件，只保留 `HandGeneratorCfg` 和 `HandGenerator` 这两个核心配置类/运行时类。helper 函数、工具函数、辅助数据结构不要写进 `hand_generator.py`，而是放到本目录下的其他文件（如 `_premade.py`、`_recolor.py`）或新建文件里。

## quick.py 的定位

`quick.py` 是用户友好的直接使用入口——把 `HandGeneratorCfg` 的常用字段提取成顶部大写常量，用户打开文件改几个值就能跑 `python quick.py`。它不引入新的配置层，只是一个方便的脚本壳。如果不同使用场景的配置差异较大（比如 pre-made 全量枚举 vs 单拓扑 post-mutate 采样），可以新建 `quick_xxx.py` 独立脚本，避免全挤在一个文件里变得臃肿。