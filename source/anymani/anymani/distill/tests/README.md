# Distill Tests

测试按“要证伪的科研命题”分层，而不是按实现文件机械镜像。默认 pytest 只运行不启动 Isaac Sim、Kit 或 `AppLauncher` 的 contract tests；GPU 性能和短训练 sanity 必须显式选择。

```text
tests/
├── contracts/
│   ├── models/                  # 同结构组 tensor、owner、奇偶性、routing 与 checkpoint-retention
│   ├── objectives/              # loss 数学、weighting 与 gradient 合同
│   ├── representations/         # frame、gauge、field、query、cache 与 provenance
│   ├── rl/                      # registry、YAML、adapter 与 temporal reset
│   └── ssl/                     # experiment resolution、split 与 leakage boundary
├── integration/                 # synthetic batch -> encoder -> decoder -> loss -> backward
├── performance/                 # 指定 GPU 的显式 latency / memory suite
└── training_sanity/             # tiny overfit 与短训练闭环，不进入默认 pytest
```

目录只在出现真实测试时创建，不用空测试或 import-only 测试伪造完成度。实现顺序遵循 contract TDD：先固定物理公式和 schema，再固定模型张量合同与 objective gradient，之后才运行端到端 synthetic integration、tiny overfit 和正式实验。

## Retained geometry encoder 性能合同

当前硬门槛针对 RTX 5070 Ti、batch size $B=4096$、单结构组下隐式主线的完整在线 retained geometry encoder。计时从 GPU-resident $q$、screw/anchors、raw home geometry 与 topology 开始；joint limits 只服务 q 采样，不进入 encoder。计时覆盖 adapter、集合聚合、backbone 以及 $z^{(0)}/z_i^{(1)}$ heads，排除磁盘/CPU cache materialization、host-to-device copy、decoder、policy head、Isaac Sim 与 `env.step`。未来若激活解析直接候选，同一门槛从 GPU-resident $q$、cached local support points、topology/poses 开始，并必须计入批量 FK/刚体点变换。

快速 suite 使用 20 次预热和 50 次 CUDA Event 计时，要求 p95 不超过 40 ms，并报告 median 与 max。200 次深度 profile 只用于候选比较、模块拆时和峰值显存分析，不进入日常测试。
