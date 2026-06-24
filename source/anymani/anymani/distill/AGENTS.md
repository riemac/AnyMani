# AGENTS.md

`distill` 服务于 AnyMani 的**训练管线与网络架构**阶段：消费 `tasks` 定义的 Isaac Lab 环境，训练、蒸馏和评估面向手型泛化的手内操作策略。手资产如何 lower 成 robot cfg 属于 `robots`，不在 `distill` 内复制 env cfg wrapper。

## 目录职责

```text
distill.py
train.py
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

当前 RL 训练入口统一为 `python -m anymani.distill.train`。新增训练路线时优先新增
Gym task alias + agent YAML；只有当训练编排本身发生变化时才扩展入口参数。不要为
临时 debug/smoke 训练重新创建 `train_mvp.py`、`train_xxx.py` 或 distill-owned env cfg
wrapper；运行时 reset/step/PhysX 验证放到 `source/anymani/anymani/smokes/` 显式 smoke。

## 模型规模与频率硬约束

手内操作是接触密集的高频闭环任务，推理频率反向约束网络规模与结构选型。**硬指标（下限，非目标值）：推理频率 ≥ 20 Hz，即单步前向控制周期 Δt ≤ 50 ms。** 该预算覆盖单步前向叠加并行 rollout 后的总开销，并给参数量、层数、宽度和在线几何/边特征计算定上界；静态量应离线预计算缓存。

RL teacher 的网络规模以 PPO 稳定性为第一约束，不以容量最大化为目标。大容量模型优先放到 student / distill 阶段吸收；若扩大 teacher，必须同时复核推理频率、rollout 吞吐和 PPO 收敛稳定性。

## 测试策略

`distill/models` 是最适合 TDD 的层：tokenizer、mask、type ids、relation batch、attention bias、backbone、heads 都应先用小 tensor 写 shape / mask / routing / zero-init 测试，再实现。尤其要测试 `valid_mask=True` 的内部约定、`JOINT`-only action routing、`hybrid_se3` bias 的 `[B,H,T,T]` shape 与初始近似 no-bias。

`distill/rl` 的 rl_games adapter 可用 contract tests 验证 flat obs/action 与 grouped token batch 的互转；真正启动 IsaacLab rollout 只做 smoke / integration。不要把 Isaac Sim 依赖引入纯模型单元测试。

## 位姿与旋转的网络表示

总纲沿用 `tasks/gm/AGENTS.md`〖数学偏好〗：矩阵为默认载体，需要线性 / 向量形式时用轴角 / 旋量，回避欧拉角与裸四元数。下面是这条主线落到**网络张量**上的补充，按角色区分，不必教条：

- **位姿 / 几何特征**（如相对位姿边特征 $E_{ij}=T_i^{-1}T_j$、frame 朝向）：作为网络张量优先用旋转矩阵，或其前两列构成的 6D 连续表示（Zhou et al. 2019）——它们连续、无双重覆盖，对学习友好。需要做线性差分（残差、增量、插值）时再取 $so(3)/se(3)$ 旋量。
- **obs 输入**：表征偏好同上，但这里是“偏好”而非硬约束——网络非线性且容量足够时，连续表征之间（矩阵 / 6D / 单位四元数）对输入端的影响有限。真正的底线是**不要喂裸四元数（双重覆盖 $q\sim-q$）或欧拉角（gimbal 跳变）这类不连续 / 多值量**。command 当前以 $so(3)$ 轴（$k^{h}$）表达旋转轴，保留。
- **动作输出**：当前 teacher 动作是**关节空间**逐关节 rad 增量标量，旋转表示不适用于动作本身，无需 SE(3) 化。

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
