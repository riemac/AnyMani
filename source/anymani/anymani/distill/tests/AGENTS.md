# AGENTS.md

本文件约束 `distill/tests/` 的测试分层与科研验收语义；仓库级 contract TDD、IsaacSim smoke 和运行命令仍服从 AnyMani 根 `AGENTS.md`，distill 的物理表示与模型边界服从上一级 `distill/AGENTS.md`。

## 核心原则

测试首先证伪数学、物理和训练合同，不追求脱离科研含义的行覆盖率。一个测试必须能回答“哪条公式、shape、gauge、routing、生命周期或性能命题被验证”；纯 import、永真 shape 或只检查对象存在的测试不能宣称算法完成。

## 目录所有权

| 路径 | 拥有 | 不拥有 |
| --- | --- | --- |
| `contracts/representations/` | frame/unit、PoE、gauge pair、scale law、field/query、semantic coverage、cache/provenance | 神经网络学习能力 |
| `contracts/models/` | 同结构组 entity/joint/anchor shape、owner 同索引、奇偶性、routing、retained/disposable keys | 物理 target 生成 |
| `contracts/objectives/` | loss 公式、weighting、reduction、stop-gradient 与 autograd 路径 | optimizer 收敛结论 |
| `contracts/ssl/` | experiment resolution、split、leakage、checkpoint 与 logging schema | 正式训练效果 |
| `contracts/rl/` | alias、YAML、adapter、observer、temporal reset 与 checkpoint preflight | IsaacSim runtime state |
| `integration/` | synthetic batch → encoder → decoder → loss → backward 的跨层闭环 | 大规模泛化结论 |
| `performance/` | 指定硬件、shape、warmup、计时边界、latency 与 peak-memory 回归 | policy/env 总周期 |
| `training_sanity/` | tiny overfit、短训练日志闭环和明显退化检测 | 论文级 learning claim |

目录在出现真实测试时再创建，不添加 `.gitkeep`、空测试或 placeholder body。测试 helper 也遵守 declaration gate：只有至少两个已写测试共享同一语义且重复会掩盖科研意图时，才提议 fixture/helper。

## 默认与显式 suite

- 默认 `pytest` 只能收集不启动 Isaac Sim、Kit、`AppLauncher` 或完整训练的 contract tests。
- `performance` 与 `training_sanity` 必须通过 marker 和显式路径运行，不能依赖机器是否“碰巧没有 GPU”来控制默认收集。
- distill-owned Isaac Sim、USD、PhysX handle、sensor buffer、reset/step 生命周期统一放在 `source/anymani/anymani/smokes/distill/`，task-owned 历史 smoke 保持原路径；两者都不进入本测试树。
- 若测试结论依赖训练曲线、simulation 或人工视觉判断，agent 只能报告观测证据并请求研究者确认，不能自行宣称通过。

## TDD 顺序

1. 先写最小失败 contract，固定符号、单位、shape、mask、gauge 与错误条件。
2. 实现最小数学或模型路径，使该 contract 通过；不得顺手声明邻近 registry、preset、adapter 或默认候选。
3. 添加 boundary case：结构模式变化、axis sign/zero rewrite、$SO(2)$ gauge、reflection 反例、uniform scale、非祖先结构零与非光滑区域掩码。
4. 只有底层 contract 稳定后，才写 synthetic integration、tiny overfit 与性能 suite。
5. 修复 bug 时先添加能稳定复现原错误的回归测试，再修改实现。

## Retained geometry encoder 性能门槛

快速性能 suite 固定为 RTX 5070 Ti、batch size $B=4096$、单结构组、20 次预热和 50 次 CUDA Event 计时，要求 p95 不超过 40 ms，并报告 median/max。当前只验收隐式主线：从 GPU-resident $q$ 与静态证据开始，覆盖 adapter、集合聚合、backbone 与 $z^{(0)}/z_i^{(1)}$ heads。未来若激活解析直接候选，同一门槛从 GPU-resident $q$、cached local support points 与 topology/poses 开始并必须计入批量 FK/刚体点变换。磁盘/CPU cache materialization、host-to-device copy、decoder、policy、Isaac Sim 与 `env.step` 均在边界外。

性能测试必须先验证输出与 contract test 等价，再测 latency；禁止通过 `torch.no_grad()` 之外的语义删减、缓存 PPO full fine-tune 下会 stale 的 learned activation，或缩小已声明 shape 来通过门槛。200 次深度 profile 是显式诊断，不进入日常 pytest。

## 失败信息

assertion message 应包含能定位科研合同的量：实际/期望 shape、frame、单位、owner/joint id、有效样本数、误差绝对值与容差来源。随机测试固定 seed；数值容差根据 dtype、公式尺度和累积路径定义，不使用没有解释的宽松阈值。
