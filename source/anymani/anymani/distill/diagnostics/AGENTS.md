# AGENTS.md

`diagnostics/` 是 AnyMani SSL、RL及后续IL学习管线的证据层，负责忠实记录运行事实、执行固定诊断和只读分析。合同、smoke和性能probe只验证相应边界；不得把它们写成表示学习、策略能力或泛化结论。

## Project Structure

```text
diagnostics/
├── recording/
│   ├── geometry_ssl.py      SSL TensorBoard、JSONL、runtime JSONL、dense NPZ
│   └── rl/                  PPO Parquet/HDF5、进程阶段、资源曲线与fatal watchdog
├── evaluation/
│   ├── geometry_ssl.py      固定前向、分层 metric 与反事实干预
│   ├── surface_reference.py 外部表面参考测度
│   └── rl/                  fixed first-trajectory能力归约与pair诊断
├── analysis/
│   ├── geometry_ssl.py      SSL artifact-only聚合、paired bootstrap
│   └── rl/                  PPO run、Parquet和runtime evidence只读汇总
└── README.md                人类阅读的证据语义与当前迁移边界
```

`recording` 只落盘调用方已经形成的事实，不重新运行模型、改变权重或选择checkpoint。`evaluation`可以消费concrete method的公开fixed-evaluation/probe surface、模型输出与物理target，但不运行optimizer。`analysis`只读`logs/`下的YAML、JSONL、NPZ、Parquet、HDF5、TensorBoard和checkpoint metadata，不import model、method、environment或teacher。

依赖方向保持为 `method public evidence surface -> diagnostics.evaluation -> recording/analysis artifacts`。SSL post-training runtime 负责加载 checkpoint 和编排 probe；model forward 与 optimizer loop 不得依赖 analysis。避免 method 与 diagnostics 互相 import 内部 helper。

## Development Style And Conventions

### Evidence media

- TensorBoard 服务在线趋势与人工观察，不是唯一事实源。
- `metrics.jsonl` 与 `runtime.jsonl` 保存 append-only 标量、预算坐标和生命周期事件。
- NPZ 保存 fixed-bank prediction、target、mask、selectors、strata 与 selected latent，使任意 tolerance threshold 可事后重算。
- YAML 保存 schema、dataset/code/checkpoint lineage、teacher baseline、q-bank digest、ablation/gradient summary 与分析结果。
- Parquet保存RL逐update的global/cell/asset标量；HDF5保存selected-checkpoint的dense first trajectories。
- RL runtime JSONL由child写phase、parent写RSS/swap/NVML采样；PhysX/CUDA fatal log只终止进程并引用最近的故障前checkpoint。
- 不在 event file 中塞入完整 latent、逐样本 dense arrays 或 full gradient vectors；这些对象必须有可审计 artifact。

每项证据必须锚定 code revision/worktree fingerprint、resolved config、dataset/physical split identity、checkpoint hash、fixed-bank digest、formula identity 和 artifact schema。训练 pairs、validation/evaluation pairs 与 probe wall time 分开计数。

### SSL schemas

当前 executable 使用单一 PALM/JOINT/TIP typed `Z`，只训练 teacher-baseline-normalized rho/kappa。derived-field、真实 density JVP 和调用方选择参数层级的 full-gradient Gram 只由显式 evaluation API 执行；训练结束不得自动调用。

Teacher-only naive baseline 与 epoch-0 network、learned query-only decoder 是三种不同参照。前者定义 normalized loss/skill，epoch-0 只表示初始化，query-only 检验 decoder bypass。禁止互相替代或复用同一个 artifact 字段。

### RL schemas

掌旋PPO每个update写1条global、每个实际active morphology cell一条cell、每个支持资产一条asset。完整MVP80因此固定89行；single/few-support closure不得伪造空cell。Global row同时保存environment-step、rollout+policy、PPO-update、epoch-total时间，以及PyTorch allocated/reserved和CUDA driver free/total显存。

固定能力评估只消费deterministic actor mean和每replica第一条trajectory，所有terminal量必须来自automatic reset前冻结的snapshot。Strict goal hit同时包含完整SO(3) orientation-keypoint与固定position-anchor门，只表示moving-goal tracking；物理持续旋转由signed net turns、absolute path和directional consistency描述。Schema 1.2保留逐trajectory paired goal/turn ratio、pulse重计、双门占比和最后goal时刻，禁止用goal count或ratio-of-medians替代轴向净转。任意支持集可以形成逐资产中位数；只有完整80-row cohort才应用54/80、8-cell各5/10与pair诊断。R1筛查、训练proxy、return或瞬时速度均不替代正式R16能力结论。

Runtime watchdog识别`Scene state is corrupted`、contact compression CUDA failure等不可恢复错误后应非零终止，不在损坏后写新checkpoint。显存安全门必须使用driver余量；`torch.cuda.max_memory_allocated()`不覆盖PhysX、context或其他直接CUDA allocation。

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

RL recorder同样不生成reward、terminal或contact真值；这些量来自task/command冻结状态。Recorder不能为填满schema重新计算MDP，也不能根据在线指标选择checkpoint、课程或gradient balancing方法。

## Common Operations And Tools

```bash
source /home/hac/isaac/env_isaaclab/bin/activate
python -m anymani.distill.ssl.evaluate --checkpoint <checkpoint.pt>
python -m anymani.distill.diagnostics.analysis.geometry_ssl <input.yaml> <output.yaml>
python -m anymani.distill.diagnostics.analysis.rl.palm_rotation <run-dir>
python scripts/benchmarks/benchmark_heterogeneous_rl.py --output_dir <evidence-dir> -- <training-command>
pytest source/anymani/anymani/distill/tests/contracts/ssl/test_geometry_ssl_logging.py -q
pytest source/anymani/anymani/distill/tests/contracts/rl/test_palm_rotation_recording.py -q
pytest source/anymani/anymani/distill/tests/contracts/rl/test_rl_runtime_diagnostics.py -q
```

项目专有 probe 和分析入口记录在本目录 README，不抽成通用 skill。修改 schema 时同步 method evaluation、post-training runtime、tests、README 与 checkpoint/artifact identity。
