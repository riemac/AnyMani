# generator/ 开发约定

本目录是多元化大批量资产生产的核心模块，会频繁迭代。以下约定是在这里工作时必须遵守的。

## 最高 façade

`HandGeneratorCfg` 是本模块唯一的最高配置入口。不允许再在它上面包装新的 cfg 类（如 `QuickRunCfg`）。所有生成逻辑的可配置项都应该收口到 `HandGeneratorCfg` 的字段里。

当前 façade 已明确包含：

- `Made`
- `Mutate`
- `Validate`
- `Export`
- `Physics`

新增阶段配置前，先判断它是不是其实应该属于这几个入口之一，而不是再包一层 runner cfg。

## hand_generator.py 的瘦身原则

`hand_generator.py` 是核心文件，只保留 `HandGeneratorCfg`、`HandGenerator`、对外主入口方法，以及少量必须和 façade 状态绑定的运行时方法。helper 函数、工具函数、辅助数据结构不要写进 `hand_generator.py`。

当前目录路由固定为：

- `premade/`：pre-made topology / connectivity / identity / batch orchestration
- `runtime/`：run 生命周期、recipe I/O、mutate-only restore、逐槽独立联合 proposal
- `presentation/`：recolor lowering、ASCII tree 等展示层工具
- `mutate/`：post-mutate term 与 pipeline

新增模块优先进入上述语义子包；不要再在 `generator/` 根目录新增 `_xxx.py`。

## physics closure 边界

`generator/` 只负责决定**何时**执行 physics closure，不负责实现 physics closure 本身。

当前约定：

- `asset_physics.py`：负责“如何由最终 collision 几何计算 `mass / inertial`”；
- `hand_generator.py`：只在 pre-made / post-mutate 后、validator 前调用；
- `mutator`：只专注几何 / 运动学语义，不内嵌最终动力学闭包；
- `exporter`：只消费闭包后的 `HandCfg`，不再偷偷补 inertia。

如果你发现自己想在 `hand_generator.py` 里写一大段质量 / 惯量数学，基本说明放错地方了。

## 工程优化

优化必须由 profile 证明瓶颈后再改变执行模型：

- pre-made topology、Python object 组装、文件 I/O 通常是 CPU/branch-heavy，不因“GPU 更快”而强制搬运；
- post-mutate 中的大批同形张量采样/数值计算可以评估 GPU/vectorized backend，但必须保留 deterministic seed
  与 CPU reference contract；
- Patch/Delta Merge/Deferred Execution 只有在 benchmark 同时证明吞吐收益、内存可控且错误定位不退化时
  才进入主路径；
- 任何并行化都要比较 wall time、峰值内存、固定 seed 下的联合 proposal 与导出结果等价性，不以 kernel 数量代替收益。
