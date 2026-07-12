# AGENTS.md

`distill` 服务于 AnyMani 的**训练管线与网络架构**阶段：消费 `tasks` 定义的 Isaac Lab 环境，训练、蒸馏和评估面向手型泛化的手内操作策略。手资产如何 lower 成 robot cfg 属于 `robots`，不在 `distill` 内复制 env cfg wrapper。

## 目录职责

```text
distill.py
train.py
play.py
il/       # 模仿学习 / 蒸馏入口
rl/       # 强化学习入口
models/   # policy、encoder、tokenizer、adapter 等共享网络定义
```

- `rl/` 与 `il/` 是算法入口层，不拥有网络结构本体。
- `models/` 是 teacher、student、unified policy、morphology encoder、token encoder 的共享落点。
- 不要让 RL policy、IL student、distillation model 各自复制一套网络实现；科研语义相同的结构必须共享同一个模型定义。

## 依赖边界

```text
assets -> robots -> tasks -> distill
```

- `assets` 生产 hand asset bank；不要把训练算法、rollout dataset、checkpoint 逻辑塞进去。
- `robots` 把 generated hand / 真实手 lower 成 Isaac Lab robot cfg；不要把 reward、reset 或训练入口塞进去。
- `tasks` 定义 Isaac Lab 环境语义：scene、obs、action、reward、reset、termination；不要把网络结构塞进去。
- `distill` 可以消费 `tasks` 暴露的 Gym task 和 agent YAML，但不得通过 env cfg wrapper 反向接管 MDP 语义。

## 训练入口边界

训练入口按 task family 区分，不宣称尚不存在的统一：

- `tasks/inhand` 的 rl_games 路线使用 `scripts/rl_games/train.py` 与 `scripts/rl_games/play.py`；
- GM MLP alias 使用 `python -m anymani.distill.train` 与 `python -m anymani.distill.play`。

新增训练路线优先新增 Gym task alias + agent YAML；只有训练/回放编排本身变化时才扩展入口参数。不要为
临时 debug 创建 `train_xxx.py` 或 distill-owned env cfg wrapper；reset/step/PhysX 验证放到显式 smoke。

## 模型规模与频率硬约束

手内操作是接触密集的高频闭环任务，推理频率反向约束网络规模与结构选型。**硬指标（下限，非目标值）：推理频率 ≥ 20 Hz，即单步前向控制周期 Δt ≤ 50 ms。** 该预算覆盖单步前向叠加并行 rollout 后的总开销，并给参数量、层数、宽度和在线几何/边特征计算定上界；静态量应离线预计算缓存。

RL teacher 的网络规模以 PPO 稳定性为第一约束，不以容量最大化为目标。大容量模型优先放到 student / distill 阶段吸收；若扩大 teacher，必须同时复核推理频率、rollout 吞吐和 PPO 收敛稳定性。

## 测试策略

`distill/models` 是最适合 TDD 的层：tokenizer、mask、type ids、relation batch、attention bias、backbone、heads 都应先用小 tensor 写 shape / mask / routing / zero-init 测试，再实现。尤其要测试 `valid_mask=True` 的内部约定、`JOINT`-only action routing、`hybrid_se3` bias 的 `[B,H,T,T]` shape 与初始近似 no-bias。

`distill/rl` 的 rl_games adapter 可用 contract tests 验证 flat obs/action 与 grouped token batch 的互转；真正启动 IsaacLab rollout 只做 smoke / integration。不要把 Isaac Sim 依赖引入纯模型单元测试。

## 位姿与旋转的网络表示

网络张量不继承“矩阵 > 轴角 > 四元数”的统一排序。每个 feature group 必须先声明它表达的是群元素、
局部误差、绝对姿态还是移动 reference 下的相对量：

- **几何计算**：$T_i^{-1}T_j$ 的 composition/inverse 先在 $SE(3)$ 语义中定义；送入网络时再按消融选择
  matrix、rot6d、quaternion 或 local $se(3)$ coordinates。
- **Relative log feature**：只有 reference/goal 对 policy 可观测或其更新律确定，且误差分布远离 $\pi$
  branch 时，才直接使用 $\log(R_{ref}R^{-1})^{\vee}$。隐藏且移动的 reference 会引入部分可观测；
  这不是“相对表示天然非平稳”，而是 observation contract 缺少 reference state。
- **Absolute orientation feature**：不存在全局连续、无奇异的三维最小参数化。rot6d/full matrix 通常更适合
  全姿态输入；quaternion 也允许使用，但必须固定 `(w,x,y,z)`、规范化和符号策略，并测试 $q\sim-q$。
- **Quaternion runtime state**：可以直接消费 Isaac Lab canonical quaternion，不要求为遵守风格而立即转换。
  `quat_unique` 只选择一个符号分支，并不会消除所有边界不连续；需要时间连续性时应显式做相邻帧符号对齐。
- **Euler/RPY**：不作为匿名网络 feature；只有外部 schema 明确要求时使用，并记录顺序与 frame。
- **动作输出**：当前 teacher 是关节空间标量，不因 orientation feature 选型而强制 SE(3) 化。

Representation 是实验 contract。变更时至少测试 shape、frame/reference、符号/branch 边界以及 goal resample
前后的 Markov 信息是否完整。

## 技术栈

| 层级 | 默认选择 | 说明 |
|------|----------|------|
| RL | `rl_games` | Isaac Lab 环境交互、teacher policy、family-level specialist 训练优先沿用现有生态。 |
| IL / 蒸馏 | 原生 PyTorch training loop | BC、DAgger、RMA、teacher-student distillation、morphology mask、privileged feature distillation 等研究逻辑显式写出。 |
| 模型网络 | PyTorch `torch.nn.Module` | `models/` 统一承载 policy、encoder、tokenizer、adapter，不随 `rl/` 或 `il/` 分裂。 |
| 数据 | HDF5 + PyTorch `Dataset` / `DataLoader` | 早期保持直接、透明；若轨迹规模或吞吐成为瓶颈，再讨论 Zarr、Parquet 或数据服务化。 |
| Hugging Face | ML artifact 管理生态 | 不作为开发主框架；长期用于 release 级 checkpoint、dataset、model card、dataset card、demo。 |

暂不优先引入 Hugging Face `Trainer`。AnyMani 的核心变量是手型拓扑、joint-centric representation、mask / padding、teacher-student 对齐与 Isaac Lab rollout，不是标准 NLP / CV supervised pipeline。`accelerate` 仅在 IL / 蒸馏明确需要多 GPU、AMP 或分布式时再引入。

## RL 训练日志分析

`rl_games` / Isaac Lab 训练默认会产出 TensorBoard event 文件。用户给出 run 目录或 `events.out.tfevents.*` 时，agent 应优先直接解析原始 scalar 数据，而不是要求用户截图 TensorBoard 曲线。TensorBoard UI 仍可作为用户人工查看曲线的工具，但 agent 的诊断应基于 `tag / step / wall_time / value` 这类结构化数据。

常用工具（位于 uv 环境）如下：

| 工具 | 用途 | 何时需要 |
|------|------|----------|
| `tensorboard`| 读取 event 文件中的 scalar、tag、step、wall time | 单次 run 诊断的默认底座；当前环境已有，优先使用。 |
| `pandas`| 将 scalar 转为表格后做筛选、统计、异常检测和跨 run 对比 | 日志较多、需要按 tag / run / epoch 查询或聚合时。 |

## 可复现元数据

正式模型或数据集至少应记录：代码 commit、asset version、task / obs / action schema、teacher policy、训练配置、评估协议、关键指标。

## 相关研究文档

| 文档 | 用途 |
|------|------|
| `AnyMani/Research/总体/科研背景说明.md` | 项目目标、资产生成与训练管线总背景。 |
| `AnyMani/Research/总体/网络架构.md` | token / projection / attention / edge feature 的研究设计。 |
| `AnyMani/Research/总体/层次通才策略训练.md` | teacher / student 训练阶段的任务语义。 |
| `AnyMani/Research/总体/sim2sim.md` | 真实 URDF 与 generated asset 的特征语义对齐问题。 |
