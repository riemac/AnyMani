# AGENTS.md

本文件约束 `distill/tests/`。仓库级 TDD 与 IsaacSim smoke 服从根 `AGENTS.md`；物理/模型边界服从 `distill/AGENTS.md`。测试证伪公式、shape、gauge、routing、生命周期或性能命题，不追求脱离科研含义的行覆盖率。

## Project Structure

```text
tests/
├── contracts/
│   ├── representations/     frame/unit、PoE、gauge、field/query、cache/provenance
│   ├── models/              entity/joint/anchor shape、奇偶性、routing、retained keys
│   ├── objectives/          gauge、selector 与候选重建公式
│   ├── ssl/                 façade、split、预实验、minibatch/复用、checkpoint
│   └── rl/                  alias、YAML、adapter、observer、masked PPO
├── integration/             synthetic / padded / real mother：encoder → loss → backward
├── performance/             RTX 5070 Ti retained encoder latency
└── training_sanity/         真实 mother 固定 batch tiny-overfit
```

| 路径 | 验证对象 | 结论边界 |
| --- | --- | --- |
| `contracts/representations/` | 物理真值与 provenance | 网络学习能力 |
| `contracts/models/` | 张量合同与 retained keys | teacher 生成 |
| `contracts/ssl/` | 配置、生命周期、身份核对 | 正式训练效果 |
| `integration/` | 跨层闭环 | 泛化结论 |
| `performance/` | 指定硬件 latency | policy/env 总周期 |
| `training_sanity/` | 明显可优化性 | 论文级 learning claim |

目录只在出现真实测试时创建。helper 遵守 declaration gate：至少两个已写测试共享同一语义才抽取。

## Development Style And Conventions

### 默认与显式 suite

默认 pytest 只收集不启动 Isaac Sim/Kit/`AppLauncher` 的 contract。`performance` 与 `training_sanity` 必须 marker + 显式路径。distill-owned Isaac 证据在 `smokes/distill/`。依赖训练曲线或人工视觉时，只报告观测并请求研究者确认。

### TDD 顺序

1. 最小失败 contract：符号、单位、shape、mask、gauge、错误条件。
2. 最小实现使该 contract 通过；不顺手声明邻近 registry。
3. boundary：结构变化、joint-sign、`SO(2)`、reflection、scale、结构零、非光滑 mask。
4. 底层稳定后再写 integration、tiny overfit、性能 suite。
5. 修 bug 先加能复现原错误的回归测试。

## Important Semantics

### 当前 SSL 合同

配置测试锁定四角色根配置、`max_epochs / num_minibatches / mini_epochs / microbatch_size` 预算接口、baseline-normalized rho/kappa、validation sigma `4/16/64 mm`、`anchor bank=8` 和 unified representation schema。teacher-baseline artifact 核对 dataset/formula/method identity，并保存 code/worktree lineage；正式 preset 与 baseline pass 预算可以不同。Trainer/checkpoint 通过 Method 窄接口工作；integration 直接运行 method objective。

### Retained encoder 性能门槛

RTX 5070 Ti、`B=4096`、单结构组、20 预热 + 50 CUDA Event，p95 ≤ 40 ms。从 GPU-resident `q` 与静态证据开始，覆盖 adapter、聚合和 backbone final-norm unified $Z$。计时保持已声明语义和 shape，并对 learned activation 每次重算；decoder 成本单独报告。

### 失败信息

assertion 必须含实际/期望 shape、frame、单位、owner/joint id、有效样本数、误差与容差来源。随机测试固定 seed。

## Common Operations And Tools

```bash
source /home/hac/isaac/env_isaaclab/bin/activate
pytest source/anymani/anymani/distill/tests/contracts -q
pytest source/anymani/anymani/distill/tests/integration -q
pytest source/anymani/anymani/distill/tests/performance -m performance -q -s
pytest source/anymani/anymani/distill/tests/training_sanity -m training_sanity -q
```
