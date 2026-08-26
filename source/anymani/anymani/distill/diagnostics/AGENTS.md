# AGENTS.md

`diagnostics/` 是 AnyMani 学习管线的证据层，负责忠实记录运行事实、执行固定诊断和只读分析。当前处于 active development：executable 已迁移到 unified-Z、teacher-only baseline、rho/kappa 双主任务和训练期稀疏 Z-gradient evidence；正式 256-epoch run 尚未执行，不得把合同通过写成学习结论。

## Project Structure

```text
diagnostics/
├── recording/
│   └── geometry_ssl.py      TensorBoard、JSONL、runtime JSONL、dense NPZ
├── evaluation/
│   ├── geometry_ssl.py      固定前向、分层 metric 与反事实干预
│   └── surface_reference.py 外部表面参考测度
├── analysis/
│   └── geometry_ssl.py      artifact-only 聚合、paired bootstrap 与分析 CLI
└── README.md                人类阅读的证据语义与当前迁移边界
```

`recording` 只落盘调用方已经形成的事实，不重新运行模型、改变权重或选择 checkpoint。`evaluation` 可以消费 concrete method 的公开 fixed-evaluation/probe surface、模型输出与物理 target，但不运行 optimizer。`analysis` 只读 `logs/` 下的 YAML、JSONL、NPZ、TensorBoard 和 checkpoint metadata，不 import model、method 或 teacher。

依赖方向保持为 `method public evidence surface -> diagnostics.evaluation -> recording/analysis artifacts`。SSL post-training runtime 负责加载 checkpoint 和编排 probe；model forward 与 optimizer loop 不得依赖 analysis。避免 method 与 diagnostics 互相 import 内部 helper。

## Development Style And Conventions

### Evidence media

- TensorBoard 服务在线趋势与人工观察，不是唯一事实源。
- `metrics.jsonl` 与 `runtime.jsonl` 保存 append-only 标量、预算坐标和生命周期事件。
- NPZ 保存 fixed-bank prediction、target、mask、selectors、strata 与 selected latent，使任意 tolerance threshold 可事后重算。
- YAML 保存 schema、dataset/code/checkpoint lineage、teacher baseline、q-bank digest、ablation/gradient summary 与分析结果。
- 不在 event file 中塞入完整 latent、逐样本 dense arrays 或 full gradient vectors；这些对象必须有可审计 artifact。

每项证据必须锚定 code revision/worktree fingerprint、resolved config、dataset/physical split identity、checkpoint hash、fixed-bank digest、formula identity 和 artifact schema。训练 pairs、validation/evaluation pairs 与 probe wall time 分开计数。

### Current and target schemas

当前 executable 使用单一 PALM/JOINT/TIP typed `Z`，只训练 teacher-baseline-normalized rho/kappa。derived-field、真实 density JVP 和调用方选择参数层级的 full-gradient Gram 只由显式 evaluation API 执行；训练结束不得自动调用。

Teacher-only naive baseline 与 epoch-0 network、learned query-only decoder 是三种不同参照。前者定义 normalized loss/skill，epoch-0 只表示初始化，query-only 检验 decoder bypass。禁止互相替代或复用同一个 artifact 字段。

### Probe semantics

反事实 probe 固定 query、sigma、selectors、targets 和 decoder，只改变待检验的 representation/input 对应关系。Same-asset cross-q 与 cross-morphology latent shuffle 检查 matched latent dependency；JOINT shuffle 只错配有效 JOINT token，不重排 PALM/TIP、graph、selector 或 target。

合法 entity permutation 必须同步重排 token、role/mask、graph matrices、routing、selectors 和输出轴，验证 permutation equivariance。它与故意破坏 joint binding 的 JOINT shuffle 是不同合同。

Joint-sign probe 必须执行完整 coordinate rewrite 并检查 observable density 不变、kappa 变号；手工把 latent 取负只能验证 reader 代数，不代表 encoder gauge contract。

### Gradient evidence

Matched pilots 使用相同 cadence：训练期稀疏记录 unified-Z task-gradient proxy；selected checkpoints 在 fixed bank 上重算 representation、last-block 与 full retained-encoder gradients。保存 norm、dot/cosine、Gram condition、candidate-direction projection 和小步长真实 loss 变化，不默认保存完整梯度向量。

Representation gradient 是低成本 surrogate，只有与 full-parameter gradient 的 cosine sign、norm ordering 和趋势实测一致后，才能作为 future balancing 输入。Diagnostics 只提供证据，不选择 FairGrad、CAGrad、PCGrad、GradNorm 或 FAMO。

## Important Semantics

Raw loss 服务优化；teacher-baseline normalized loss 与 skill 判断是否超过朴素解；tolerance curve、structural-zero false positive、active sign/vector response 和 latent intervention解释机制；最终跨手型效用由相同预算的 PPO transfer 验证。

最小统计单位保留 `(asset_id,q_index)` 配对。先在 morphology 内聚合 q，再对 morphology 等权；bootstrap 不能把同一手的 query 当作独立 morphology。被 mask 排除的样本仍保存原值和排除原因，避免低误差由有效区域收缩伪造。

Probe、logger 和 analysis 不生成物理真值；distance、closest point、Jacobian、rho/kappa target 与有效性来自 `representations`/method。Research 可以消费 evidence 和绘图，但 `source/anymani` 不依赖 Research vault。

## Common Operations And Tools

```bash
source /home/hac/isaac/env_isaaclab/bin/activate
python -m anymani.distill.ssl.evaluate --checkpoint <checkpoint.pt>
python -m anymani.distill.diagnostics.analysis.geometry_ssl <input.yaml> <output.yaml>
pytest source/anymani/anymani/distill/tests/contracts/ssl/test_geometry_ssl_logging.py -q
pytest source/anymani/anymani/distill/tests/contracts/ssl/test_geometry_ssl_ablation_analysis.py -q
```

项目专有 probe 和分析入口记录在本目录 README，不抽成通用 skill。修改 schema 时同步 method evaluation、post-training runtime、tests、README 与 checkpoint/artifact identity。
