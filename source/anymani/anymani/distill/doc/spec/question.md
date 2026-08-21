# Geometry SSL 正式实验前的工程问题

> 记录日期：2026-08-19。本文只记录 canonical 多锚点 Gaussian 隐式场预训练的工程债与验收条件，不重新决定科研路线。修复不得暗自改变全资产随机顺序、每资产 Sobol 构型覆盖、query/sigma 测度、objective reduction、validation split 或 retained/disposable 信息边界。

## 当前判断

schema 4 方法聚合根、五项 objective、window-major resident window 与 calibration 身份核对已经落地；contract tests 与 synthetic integration 可以通过。正式 8192-asset 的 epoch / `q_per_asset_per_epoch` 尚未拍板，当前 `20×256` 只是脚手架。S0 的 window/lease 项已在代码中闭合，剩余阻断主要是真实端到端 smoke、dirty-worktree lineage 与正式评估证据。

## S0：正式 pilot 前必须解决

### 1. 主训练路径没有真正使用 20-asset resident window

DONE：`OnlineMinibatchSchedule` 现为 window-major；`realize_minibatch()` 把完整 `resident_asset_indices` 交给 `ResidentGeometryAssetWindow.ensure()`。同一窗完成全部 q coverage 后再切窗。旧 `WindowedOnlineGeometryBatcher` 已删除。45-asset 的 2944 minibatch 只是历史探针，不是 8192-asset 正式预算。

修复验收条件：

- resident window 切换不能改变每个 epoch 的全资产随机排列、每资产 Sobol cursor、尾组和 resume 后的下一样本；
- 同一 20-asset window 内应复用已物化的 device source 与 Warp BVH，而不是按两个资产的 minibatch 重建；
- runtime evidence 应记录每次切窗的 asset IDs、BVH 数、三角形数、加载/释放时间以及 CUDA/PyTorch 显存变化；
- 用短 schedule 验证切窗次数随 window 数量而不是 minibatch 数量增长。

### 2. 已驱逐的 device state 仍被外部字典持有

DONE：lifecycle 不再维护跨 minibatch 的 `states_by_id`。method 只在一次 `realize_minibatch()` 内用局部 `states_by_id` 索引当前 window，函数返回后不保留 GPU state 强引用。

`representations/sources/collision_geometry.py::release_warp_owner_geometry_cache()` 已明确说明：registry lease 归零后，调用方还必须丢弃自身 cache/device-state 引用，Python 引用计数归零时底层 BVH 才会析构。尤其需要注意：calibration cache 和 validation cache 在函数局部作用域中会长期存活，可能使 resident cap 失去实际显存约束。

修复验收条件：

- 切窗后不再存在指向已驱逐 device state 的强引用；
- calibration、validation 和异常退出均有确定的 teardown；
- 连续跨多个 window 后，Warp cache entry/lease 数和 driver 视角显存回到稳定平台，而不是随访问过的资产总数单调增长；
- 回归测试同时检查 registry lease、state 可回收性和 `finally` 清理路径。

### 3. active method 缺少真实端到端 smoke

integration 已改接五项 method objective 与 padding batch，但仍缺少一份走正式 `fit_embodiment_pretrain()` 的最小真实资产 smoke：一个 optimizer update、五项有限 loss/gradient、calibration artifact 身份、checkpoint reload 与 retained artifact 不含 disposable reader。该 smoke 只验证执行闭环，不宣称表征有效或跨手型泛化成立。

### 4. 实验代码 lineage 不能区分 dirty worktree

`ssl/runtime/run.py::PretrainRun.code_revision()` 只记录 `git rev-parse HEAD`。当前工作树存在未提交和未追踪改动时，checkpoint 会把实际运行代码误记为 HEAD 对应版本，无法复现实验。

修复验收条件：正式实验优先在干净、已提交的工作树运行；运行元数据至少记录 commit、dirty 状态和可审计的 diff/untracked manifest 指纹。不得把大型 diff 正文重复写入每个 checkpoint，可在 run 根目录保存一次并由 checkpoint 引用其哈希。

## S1：形成正式科研结论前必须补齐

### 5. unseen suites 尚无模型评估

当前 unseen-variant-set、unseen-mother 和 official-zero-shot 只进入资产与 physical identity 审计；真正的 model forward、指标聚合和 artifact 只覆盖 validation split。正式结果不能据此声称已经评估 cross-mother、unseen morphology 或 official zero-shot。

需要补充冻结 checkpoint 上的独立 evaluation lifecycle，并保证 unseen suite 不参与训练、loss calibration 或 checkpoint selection。当前 official-zero-shot 为空集，也必须在结果中显式报告为空，而不是产生隐式成功。

### 6. diagnostics logger 未接入正式 lifecycle

`diagnostics/recording/geometry_ssl.py::GeometrySSLRunLogger` 已定义 TensorBoard、JSONL、runtime JSONL 和 dense NPZ，但正式 lifecycle 目前只直接追加简化的 `metrics.jsonl`。`log_every_updates` 也未控制实际记录频率。

需要接入至少以下证据：五项 $(asset,q)$ 等权均值、按 asset/owner/query stratum/sigma/distance shell/ancestor 分层的误差与有效率、gradient norm、resident-window telemetry、q/query/teacher digest，以及固定 validation 和 independent-q replay 的 provenance。

### 7. checkpoint selection 仍是全局聚合

`ssl/runtime/lifecycle.py::_evaluate_validation()` 当前把所有 validation batch 的 component numerator/denominator 全局相加。代码库已有 stratified validation helper，但没有接入 selection。全局聚合可能使 owner 更多、有效 query 更多或结构规模更大的 morphology 主导 checkpoint 选择。

需要先明确并记录 selection 的科研语义，再接入 morphology-balanced、stratum-aware 的聚合。修复不能悄悄创造新的 loss reduction；selection metric 与训练 objective reduction 应被视为两个显式合同。

### 8. 最近面非光滑区域的证据不完整

当前一阶 mask 包含 Warp face validity、owner-shell、distance epsilon 和 triangle feature margin，但没有全局 second-nearest/medial-axis margin。在指间 gap、凹面、owner overlap 或近等距表面附近，closest source 切换会使 kappa/JVP target 分段不光滑。

首轮实验至少应报告按 owner、query stratum 和 distance shell 分层的 edge valid rate、mask 原因与最近 source 稳定性。是否物化全局 second-nearest margin属于科研与计算成本共同决定的问题，不应在普通重构中默认加入或删除。

## S2：非阻断清理

- term ablation 只能通过把 `OBJECTIVES_CFG` 对应字段设为 `None` 显式关闭，不得再恢复联合六项 runtime。
- `ssl/calibration.py`、`ssl/runtime/objective.py`、`WindowedOnlineGeometryBatcher` 与 `GeometryFieldObjective` 运行时已删除。
- `{h}` 的 `palm_normal=(0,0,1)` 目前是 input adapter 中的硬假设；应在资产语义或 lowering validator 中显式声明并验证，而不是依靠读者推断。
- parametric Gaussian components、旧 attention-bias 和 temporal candidate scaffold 不属于当前 canonical 隐式场调用图，不应被误报为已验证能力。

## 与科研实验的边界

上述工程修复只负责让同一个实验合同被高效、可恢复、可审计地执行。以下问题仍必须由实验回答，不能通过工程重构替代：

- implicit field 是否优于 analytic direct geometry compression；
- density、kappa、derived-field、Sobolev 和 chain 分别提供了什么可迁移信息；
- anchor/query/sigma 当前 preset 是否覆盖了真正与接触相关的空间尺度；
- zero-order 与 first-order latent 是否能在未见构型、未见形态和 PPO 中被有效读取；
- 预训练相对同一前端 scratch 是否带来可重复的样本效率、最终性能或最差手型收益。
