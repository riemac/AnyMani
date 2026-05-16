# generator/ 开发约定

本目录是多元化大批量资产生产的核心模块，会频繁迭代。以下约定是在这里工作时必须遵守的。

## 最高 façade

`HandGeneratorCfg` 是本模块唯一的最高配置入口。不允许再在它上面包装新的 cfg 类（如 `QuickRunCfg`）。所有生成逻辑的可配置项都应该收口到 `HandGeneratorCfg` 的字段里。

## hand_generator.py 的瘦身原则

`hand_generator.py` 是核心文件，只保留 `HandGeneratorCfg`、`HandGenerator`、对外主入口方法，以及少量必须和 façade 状态绑定的运行时方法。helper 函数、工具函数、辅助数据结构不要写进 `hand_generator.py`。

当前目录路由固定为：

- `premade/`：pre-made topology / connectivity / identity / batch orchestration
- `runtime/`：run 生命周期、recipe I/O、mutate-only restore、quota batch helper
- `quota/`：accepted/output mode quota 的分配与 forced-mode lowering
- `presentation/`：recolor lowering、ASCII tree 等展示层工具
- `mutate/`：post-mutate term 与 pipeline

新增模块优先进入上述语义子包；不要再在 `generator/` 根目录新增 `_xxx.py`。

## 工程优化

#TODO:包括GPU并行（pre-made也好，post-mutate也好，采样也好，apply也好，都尽可能的并行），Patch / Delta Merge / Deferred Execution 等规则和约定
