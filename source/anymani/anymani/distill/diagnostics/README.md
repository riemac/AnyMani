# Geometry SSL Diagnostics

较低的平均重建误差并不能证明模型学到了跨手型几何。误差可能集中在少数 owner 或尺度上，decoder 可能只依赖 query 而忽略 latent，validation 也可能因为重复物理几何或可变 q bank 而泄漏。`diagnostics` 因此不是训练日志的附属工具，而是 Geometry SSL 方法中负责区分这些解释的证据层。

## 一次观测应保留什么

最小分析单位不是“某个 step 的 total loss”，而是带有 asset、q、owner、query stratum、sigma、distance shell、ancestor relation 与 validity mask 的预测—真值配对。`GeometrySSLRunLogger` 同时保存六项 loss 的 numerator 与 denominator，使 tail asset group、跨结构 padding 和非光滑 mask 不会因先做 microbatch mean 而静默改变统计权重。

dense evidence 保留 $Z^{(0)}$、$Z^{(1)}$、density/κ errors、closest-source 与采样 provenance。被 mask 排除的一阶样本仍携带原始数值和排除原因；否则低误差可能只是有效区域不断缩小。runtime evidence 另外记录 resident assets、BVH/triangle 数、load/release 时间、显存变化与吞吐，从而把表示误差与资源退化分开。

## Checkpoint 选择不是单一均值

generated validation assets 先按 `physical_geometry_hash` 与 train 整组隔离。固定 bank 为每项 held-out morphology 保存相同的 Sobol q、query realization、sigma 与 teacher digest；每次评估只改变模型参数。density、κ 与 derived-field errors 先相对 initialization baseline 归一化，再在 morphology 和物理 strata 上聚合，避免 owner 数较多或 query 较容易的手型主导 checkpoint score。

initial baseline、历史 best 与完整 selection history 都属于 resume state。`best.pt` 的 promotion 已由 runtime 执行，而不是分析脚本事后猜测；恢复训练若丢失这些 lineage，即使权重可加载，也不能声称延续同一个模型选择过程。

## 反事实干预

重建任务最危险的捷径是 decoder 绕过 morphology latent。固定 ablations 因此保持 query、sigma、selectors 和 decoder 不变，只干预表示：

- query-only 把 $Z^{(0)}$ 与 $Z^{(1)}$ 清零，测量纯坐标路径能解释多少误差；
- same-asset q shuffle 错配同一手型的构型 latent，分离静态形态记忆与当前 q；
- cross-asset shuffle 同时破坏形态与构型匹配；
- first-order zero、JOINT shuffle 与 sign flip 检查 κ reader 是否真正使用有类型的一阶 carrier。

这些比较以相同 asset/q 样本配对，最终采用 asset/q 两级 bootstrap，而不是把同一手的数千 query 当成独立样本。canonical protocol 使用 2000 次重采样；区间反映当前固定 bank 上的配对不确定性，不外推为 official hand 或新 topology 的总体置信区间。

训练形态还有一套独立 q bank，在 initialization 与 final checkpoint 上从 cursor 0 重新流式生成。两次运行必须得到相同 q/query/teacher SHA-256；这项证据用于区分“模型改变”与“评估 realization 改变”。

## 证据边界

上述诊断可以证伪输入泄漏、latent bypass、错误 routing、mask coverage 退化和 checkpoint lineage 漂移，但不能单独证明 official zero-shot、cross-topology 泛化或 PPO transfer。任何结论至少应共同引用 code revision、resolved config、asset manifest、calibrated objective、selection history 与 checkpoint；TensorBoard 曲线或最终平均值不能替代这些事实源。

一次 run 的原始证据写入 `logs/ssl/<experiment>/<UTC timestamp>/`：dataset/resolved YAML 保存实验与资产选择事实，TensorBoard 用于在线观察，JSONL 保存 append-only 标量与 runtime 事件，NPZ 保存 dense arrays，YAML 保存 expanded manifest、calibration、q-bank、selection 与 ablation。`recording` 负责落盘，`evaluation` 负责固定前向与分层统计，`analysis` 只读这些产物并生成 bootstrap 等派生结果；三者都不反向参与模型输入或 optimizer。
