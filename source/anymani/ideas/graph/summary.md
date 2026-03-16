---
title: 第四轮 Idea 讨论总结——聚焦手型泛化
date: 2026-03-10
project: AnyMani
tags: [graph-attention, embodiment-generalization, architecture, GET-Zero, TRO-Grasp]
type: research
---

## 背景

用户决定暂时放弃物体泛化，只专注手型（embodiment）泛化方向。核心判断："仅能实现手型的手内操作任务真正泛化，便已能超越 GET-Zero 并产出一篇成果了。"

本轮在完整阅读 GET-Zero（正文 1-6 页）和 T(R,O) Grasp（正文 1-8 页 + 附录 11-12 页）论文、以及双方核心代码后，围绕架构设计做了深入讨论。

## 关键决策

### 1. GET-Zero 的精确短板（补读论文/代码后确认）

- **embodiment 编码极弱**：只有 joint 3D pos/rot from URDF（fixed local）+ SPD/parent-child 标量 bias。论文自述未编码 joint limits、link geometry、motor strength、friction
- **泛化范围窄**：所有实验仅在 LEAP Hand family 内（236 个变体，同一 finger layout、同一 link 形状）。论文自认 "unlikely to transfer to a new robot hand model"
- **训练范式笨重**：44 个 per-embodiment RL expert → 7h/each demonstration → BC 蒸馏

### 2. 架构主体方向确定

**两阶段数据流（基于用户草图）**：

1. **Cross-Attention（结构-状态对齐）**
   - URDF 信息（joint limits, rest pose）+ BPS link 几何 → MLP → **K, V**（static stream）
   - Joint state（angle, velocity, prev action）→ MLP → **Q**（dynamic stream）
   - 多对多 cross-attention：每个 joint 的 query attend 到所有 URDF tokens

2. **Self-Attention + Graph Bias（关节间协调）**
   - cross-attention 输出作为 joint tokens
   - 图结构（SPD, parent-child, E_RR relative SE(3)）以 bias 注入 attention score

3. **Policy Head** → per-joint action

### 3. 方案 A vs 方案 B（均保留）

| 方案 | Self-Attention 层 | 边特征角色 | 复杂度 |
|------|-----------------|----------|-------|
| A（Graph Transformer） | 标准 self-attn + bias | 只影响 attention score | 低 |
| B（Relational Attention） | edge-conditioned V: $V_{ij}=g(h_i,h_j,e_{ij})$ | 同时影响 score 和 Value | 中 |

- 方案 A 适合快速验证
- 方案 B 表达力更强，且后期扩展到物体节点（OR/RR attention）更自然
- 两者前端和后端完全相同，差异仅在 self-attention 层

### 4. BPS 编码位置

**决定**：BPS 放入 cross-attention 的 static stream（和 URDF 信息一起作为 K/V），而非 self-attention 的 bias。理由：BPS 是 link 的静态几何属性，不是关节间的关系。

### 5. 变长输入处理

- 如果采用端到端 RL（每种手型一个 env group），batch 内 DoF 一致，**不需要 padding**
- 如果采用 GET-Zero 式 BC 蒸馏，蒸馏数据集混合不同 DoF → 需要 padding + mask
- 线性层 $Y=XW+b$ 天然处理任意 token 数

## 术语分类

| 术语 | 含义 | 代表工作 |
|------|------|---------|
| MPNN | 邻域消息传递，边参与消息函数 | GCN, GAT, GraphSAGE |
| Graph Transformer | 全局 attention + 图 bias | Graphormer (Ying 2021) |
| Relational Attention | 全局 attention + edge features 进入 V/Q/K | Shaw 2018, Edge-Augmented Transformer |
| GET-Zero | Graphormer 式 Graph Transformer | Patel & Song 2024 |
| TRO-Grasp GraphLayer | 异构 Relational Attention (OR + RR) | Fei 等 2025 |
| 我们的方案 A | Embodiment-Conditioned Graph Transformer | — |
| 我们的方案 B | Embodiment-Conditioned Relational Graph Transformer | — |

## 前三轮中搁置的设计

以下来自前三轮，因放弃物体泛化而暂时搁置：
- object memory slots（1/2/4 个 slots 的讨论）
- hand↔object cross-attention
- per-finger local latent 对物体属性的估计
- local loss 预测接触/滑移、global loss 预测物体 extrinsics 的监督拆分

## 遗留问题

1. **训练范式**：端到端 RL vs GET-Zero 式 per-expert BC 蒸馏——尚未讨论
2. **实验设计**：具体用哪些手型变体、baseline 选择——尚未讨论
3. **Self-modeling loss**：是否保留 FK 预测、是否扩展——尚未讨论
4. **方案 A vs B 的最终选择**：需要实验验证

## 下一步方向

1. 讨论训练范式和实验设计
2. 开始实现方案 A 的最小原型（或 A/B 并行 ablation）
3. 确定实验用的 hand family 列表和对应 URDF

## 讨论记录

详细讨论纪要见 `discuss3.ipynb`（Cell 1-7，共 6 轮 + 开头总结）
