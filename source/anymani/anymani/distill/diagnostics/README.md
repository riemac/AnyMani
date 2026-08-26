# Geometry SSL Diagnostics

较低的平均重建误差并不能证明模型学到了跨手型几何。误差可能集中在少数 owner 或尺度上，decoder 可能只依赖 query 而忽略 latent，validation 也可能因为重复物理几何或可变 q bank 而泄漏。`diagnostics` 因此不是训练日志的附属工具，而是 Geometry SSL 方法中负责区分这些解释的证据层。

> **Active development.** executable 已采用单一 PALM/JOINT/TIP typed `Z`、teacher-only naive baseline、rho/kappa 双主任务和每 4 epoch 的训练期 Z-gradient proxy。derived-field、density JVP 与 selected-parameter full-gradient Gram 已作为手动 evaluation API 保留；正式 256-epoch 训练与 checkpoint 后机制评估尚未运行。

## 三层证据职责

`recording` 忠实写出调用方已经计算的 TensorBoard、JSONL、NPZ 与 YAML，不重新运行模型或决定 checkpoint。`evaluation` 在固定 bank 上运行模型和 method-specific probes，形成可配对、可分层的充分统计。`analysis` 只读冻结 artifact，完成 morphology/q 聚合、bootstrap、曲线和跨 variant 比较，不 import model、method 或 teacher。

## 一次观测应保留什么

最小分析单位不是“某个 step 的 total loss”，而是带有 asset、q、owner、query stratum、sigma、distance shell、ancestor relation 与 validity mask 的预测—真值配对。训练记录同时保存 rho/kappa 的 raw loss、teacher-baseline normalized loss、skill、active/zero 分支和 denominator，使 tail asset group、跨结构 padding 和非光滑 mask 不会因先做全局标量混合而静默改变统计权重。

论文数据效率曲线以训练优化器首次消费的新 asset–configuration pairs 累计数 `new_pairs_seen` 为主横轴。JSONL 同时保存 `pair_uses`、`optimizer_update`、`teacher_pairs_realized` 与墙钟，避免多 mini-epoch 方法在相同新数据横轴上隐藏额外计算；逐 update TensorBoard 服务优化诊断，epoch/validation TensorBoard 才使用新 pair 横轴。validation/evaluation pairs 单独计数，不进入训练预算。

dense evidence 保留统一 $Z$、density/kappa prediction 与 target、closest-source、全部 mask/selectors 和采样 provenance，而不是只保存最终 error。这样 success threshold、sign accuracy、tolerance curve 与新分层可以在不重跑模型的情况下改变。被 mask 排除的一阶样本仍携带原始数值和排除原因；runtime evidence 另外记录 resident assets、BVH/triangle 数、load/release 时间、显存变化与吞吐。

## Checkpoint 选择不是单一均值

Generated validation assets 先按 `physical_geometry_hash` 与 train 整组隔离。固定 bank 为每项 held-out morphology 保存相同的 Sobol q、query realization、sigma 与 teacher digest；每次评估只改变模型参数。Density 与 kappa 相对冻结的 teacher-only naive baseline 归一化，再在 morphology 和物理 strata 上聚合。Epoch-0 network 只表示初始化，learned query-only decoder 只检验 shortcut；二者都不替代 naive baseline。

Pure pretrain 只更新和保存 immutable checkpoints，不在训练进程内执行 fixed-bank selection。Checkpoint 后 validation 保存 baseline identity、完整 selection history 与 promotion evidence；恢复或比较若丢失这些 lineage，即使权重可加载，也不能声称延续同一个选择过程。

## 反事实干预

重建任务最危险的捷径是 decoder 绕过 morphology latent。固定 ablations 因此保持 query、sigma、selectors 和 decoder 不变，只干预表示：

- query-only 把统一 $Z$ 清零，测量纯坐标路径能解释多少误差；
- same-asset q shuffle 错配同一手型的构型 latent，分离静态形态记忆与当前 q；
- cross-asset shuffle 同时破坏形态与构型匹配；
- JOINT-token shuffle 只错配有效 JOINT token，检查 kappa reader 是否读取 joint-specific response；
- 完整 joint-coordinate rewrite 检查 density 不变与 kappa 变号，不能用手工 latent sign flip 替代。

这些比较以相同 asset/q 样本配对，最终采用 asset/q 两级 bootstrap，而不是把同一手的数千 query 当成独立样本。canonical protocol 使用 2000 次重采样；区间反映当前固定 bank 上的配对不确定性，不外推为 official hand 或新 topology 的总体置信区间。

训练期以稀疏 cadence 记录实际 minibatch 上的 representation-gradient proxy；selected checkpoints 在独立固定 bank 上重算 representation、last-block 与 full retained-encoder gradients。两条 matched pilots 使用相同 cadence，诊断耗时单独记录。只有 proxy 与 full gradient 的 cosine sign、norm ordering 和趋势稳定一致后，前者才可作为 future balancing 输入。

## 证据边界

上述诊断可以证伪输入泄漏、latent bypass、错误 routing、mask coverage 退化和 checkpoint lineage 漂移，但不能单独证明 official zero-shot、cross-topology 泛化或 PPO transfer。任何结论至少应共同引用 code revision、resolved config、asset manifest、teacher-baseline artifact、selection history 与 checkpoint；TensorBoard 曲线或最终平均值不能替代这些事实源。

一次 run 的原始证据写入 `logs/ssl/<experiment>/<UTC timestamp>/`：dataset/resolved YAML 保存实验与资产选择事实，TensorBoard 尽量提供在线可视标量与稀疏 histogram，JSONL 保存 append-only 指标和 runtime 事件，NPZ 保存可重新阈值化的 dense arrays，YAML 保存 teacher baseline、q-bank、selection、ablation、gradient summary 与 lineage。TensorBoard 服务观察，JSONL/NPZ/YAML 保证可重算；三者都不反向决定模型输入或 optimizer。
