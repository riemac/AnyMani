# AGENTS.md

本目录承载 `tasks/gm` 的 **grasp cache 契约层**。它不是训练算法目录，也不是资产生成器目录；它描述 generalized manipulation MDP 如何把“稳定手-物初始状态分布”作为 reset plugin 消费。

## 职责边界

- `grasp_cache` 定义 cache key、tensor schema、metadata、store/sampler 接口，以及未来 `gm/mdp/events.py` 调用 reset plugin 时必须遵守的数学语义。
- 离线 cache 生成入口应放在 `scripts/gm/` 一类脚本层，脚本可以调用本目录契约，但不要把 Isaac Sim AppLauncher、批量 asset-bank split、训练启动逻辑塞进本目录。
- `distill` 只负责选择训练 asset bank、记录 manifest、把 cache root / cache manifest 注入环境配置；不要把 cache tensor schema 复制到 `distill/rl`。
- `assets` 只生产 hand asset bundle 与几何/动力学闭包；grasp cache 是任务相关的 hand-object 初始状态分布，不属于纯资产生成 contract。

## 当前决策

- 主线 cache 粒度为 `asset_id / object_id / scale_bucket / pose_distribution`，即最终训练消费的是 per-asset validated cache。
- topology 或 nominal cache 只允许作为离线生成 warm start；不能直接作为 post-mutate asset 的正式 reset cache。
- 插上 cache reset plugin 后，object pose reset 不再使用独立的普通 pose DR；object pose 由 cache entry 中的稳定相对位姿控制。
- 无 cache 消融与 cache reset 是两套互斥 reset 模式：前者可继续用随机 joint/object pose DR，后者只采样 validated cache entry。
- cube 的 `<=1 cm` 平移扰动与 yaw-uniform 应进入 cache generation / validation 分布；online reset 第一版不再额外施加强扰动。
- 多资产并行生成只是待验证目标：需要确认 IsaacLab mixed-articulation batching、joint schema、per-env asset metadata 之后才能写成正式实现。

## 测试策略

本目录优先 TDD：`GraspCacheKey` 路径、metadata round-trip、tensor shape 校验、sampler 索引、reset request 与 cache shard 匹配都应先有纯 Python / tensor contract test。离线 Isaac Sim cache generation 和稳定性验证属于重型 pipeline，用小规模 headless smoke 与人工/统计检查补充，不替代 schema/store/sampler 单元测试。

## 暂时产物位置

短期内，cache 大文件允许自包含地放在本目录下的非源码产物子目录中，例如：

```text
grasp_cache/
  artifacts/          # 本地生成的大 cache；应保持 gitignored 或通过外部 artifact 系统管理
  manifests/          # 小型 manifest / index，可视情况进入版本管理
```

长期更合理的重构方向是在 `AnyMani/source/anymani/` 下建立统一产物根目录，避免 `assets/generated/`、`outputs/`、任务目录之间分散存放实验产物。
