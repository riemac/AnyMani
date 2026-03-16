---
title: 第五轮 Idea 讨论总结——从精读到可实现架构
date: 2026-03-10
project: AnyMani
tags: [graph-attention, embodiment-generalization, architecture, GET-Zero, TRO-Grasp, method-draft]
type: research
---

## 背景与动机

本轮讨论的起点，是在前四轮已经基本收敛的前提下，先**精读 GET-Zero 与 T(R,O) Grasp 的论文和代码**，再反过来审视我们自己的手型泛化架构是否真的合理、可实现、且能支撑论文叙事。

本轮开头的既有共识包括：

- 只做 **hand embodiment generalization**，暂不做 object generalization；
- 动作空间坚持 **joint space**；
- token 粒度坚持 **joint-level**；
- 主体方向偏向 **Graph Transformer / Relational Attention**；
- 用户对“直接端到端多手型 RL 很难”保持警惕。

在这个基础上，本轮重点回答了四类问题：

1. GET-Zero 和 T(R,O) 各自真正的长板和短板是什么；
2. Cross-Attention 里的“cross”到底跨什么；
3. static stream / relation message / policy head 该如何设计，才不至于信息流断裂；
4. 这套架构在 `>20Hz` 约束下是否现实，以及时间维该怎么建模。

## 关键决策

### 1. 精读后的方法定位：GET 做骨架，T(R,O) 提供关系编码启发

本轮精读后，方法定位被压缩得更清楚：

- **GET-Zero 的强项**：joint-level tokenization + graph-biased attention，适合做 online joint policy backbone；
- **GET-Zero 的短板**：embodiment 编码过弱，只在 LEAP family 内验证，且图信息主要以离散 bias 注入 attention score；
- **T(R,O) 的强项**：连续几何关系进入 message/value，OR/RR attention 的关系建模很强；
- **T(R,O) 的短板**：graph diffusion 对在线控制仍偏重，不适合直接当 policy 主干。

因此，本轮形成的明确判断是：

> **方法主线应当是“GET 风格的 online joint policy syntax + TRO 风格的 relation encoding”。**

### 2. Cross-Attention 的语义拍板：sample-wise conditioning，而不是 URDF 库检索

本轮最重要的术语澄清之一是：

> **Cross-Attention 里的 cross，跨的是 two streams / two modalities，不是跨整个训练集的 URDF 库。**

也就是说，对第 $b$ 个样本，做的是：

$$
H^{(b)} = \mathrm{CrossAttn}(D^{(b)}, S^{(b)}, S^{(b)}),
$$

其中：

- $D^{(b)}$：当前这只手在当前时刻的 dynamic joint tokens；
- $S^{(b)}$：同一只手的 static embodiment tokens；
- 注意力仅在**当前手内部**发生。

因此，以下做法被明确排除：

- 把所有训练过的 URDF 同时送进 memory；
- 让当前状态去“检索最像的手型”；
- 将 cross-attention 理解为 hand retrieval。

### 3. static stream 的结构拍板：不是单个 global URDF 向量，而是 per-joint static token set

本轮进一步明确：

- 如果 static stream 只有一个 global hand embedding，**不值得做 cross-attention**；
- 若 static stream 是 **per-joint / per-link structured tokens**，cross-attention 才有意义。

因此，当前推荐的 static stream 为：

$$
S = [s_1, s_2, \dots, s_J, s_{hand}],
$$

其中 $s_j$ 是第 $j$ 个 joint / attached-link 的 static token，$s_{hand}$ 是可选的全手 summary token。

候选字段包括：

- joint limits；
- joint axis / joint type；
- rest-pose transform；
- attached-link geometry（BPS 或轻量几何编码）；
- depth / parent-child count / finger id 等拓扑量；
- 可选 actuator / inertia / fingertip 类型。

### 4. 主体架构当前版本拍板：Residual Cross-Attn + Residual Relational Self-Attn + Local Bypass

本轮不是简单停留在“Cross-Attn -> RelSelfAttn -> Head”的草图，而是进一步把**信息保真路径**补齐了。

当前最成熟的版本可写为：

1. **Local conditioning**
$$
\hat d_j = \gamma(s_j) \odot d_j + \beta(s_j)
\quad \text{or} \quad
\hat d_j = \mathrm{MLP}_{cond}([d_j, s_j]).
$$

2. **Residual Cross-Attention block**
$$
H^{ca} = \hat D + \mathrm{CrossAttn}(\mathrm{LN}(\hat D), \mathrm{LN}(S), \mathrm{LN}(S)),
$$
$$
H^{ca} = H^{ca} + \mathrm{FFN}(\mathrm{LN}(H^{ca})).
$$

3. **Residual Relational Self-Attention block**
$$
H^{rel} = H^{ca} + \mathrm{RelSelfAttn}(\mathrm{LN}(H^{ca}), e),
$$
$$
H^{rel} = H^{rel} + \mathrm{FFN}(\mathrm{LN}(H^{rel})).
$$

4. **Local main path + relational residual head**
$$
a_j = \pi_{local}(\hat d_j) + \Delta\pi_{rel}([\hat d_j, h_j^{rel}]).
$$

可选地，可对 relational residual 再加一个 gate：

$$
a_j = a_j^{local} + \lambda_j \odot \Delta a_j^{rel},
\qquad
\lambda_j = \sigma(W h_j^{rel}).
$$

这一步的核心结论是：

> **当前架构最需要补的不是更多 token，而是 residual、FFN 和 local bypass。**

### 5. 对“head 直接接最终隐状态”的判断：问题不在 head，而在是否缺少保底通路

本轮专门讨论了一个深度学习层面的质疑：

> 仅靠 `Relational Self-Attention -> per-joint policy head` 会不会丢掉原始 joint state 信息？

结论是：

- `final hidden state -> head` 本身不是错误，GET-Zero 就这么做；
- 但如果高层图里**没有** residual / FFN / local bypass，那么确实会形成不必要的信息瓶颈；
- 因此，我们当前版本明确采用：
  - block 内 residual；
  - block 后 FFN；
  - head 前 local main path + relational residual。

### 6. 频率与时间步建模：GET 是 short-history feature stack，不是 temporal transformer

本轮还回答了两个工程问题：

#### 推理频率

- GET 的真实硬件部署代码 `get_zero/get_zero/deploy/leap/deploy.py` 明确以 `hz = 20` 运行；
- 控制循环里每步只调用一次网络前向，并监控 `actual_hz_running_average`；
- 复杂度上，16-DoF 手的 $J^2$ 规模很小，attention 前向本身不太像会成为 20Hz 的主瓶颈；
- 实际瓶颈更可能来自：
  - 电机通信；
  - 观测构建；
  - 相机 / AR tag；
  - Python 控制循环。

#### 时间步建模

- GET **不是纯单时间步策略**；
- 也**不是 temporal transformer**；
- 它是在 tokenization 之前，把 history 沿 feature 维堆叠到每个 token 里。

换句话说，GET 更像：

$$
	ext{short-history feature stack} + \text{joint-token graph transformer},
$$

而不是：

$$
	ext{temporal transformer} + \text{graph transformer}.
$$

对我们的方法，这直接导向一个建议：

> **如果目标是 `>20Hz`，就不要在时间轴上再做一个 attention；history 要么 feature-stack，要么放图外的小 temporal encoder。**

## 术语澄清

| 术语 | 本轮中的精确定义 |
|---|---|
| Graph Transformer | 全局 attention + 图结构 bias，图信息主要进入 attention score |
| Relational Attention | 全局 attention，但 edge features 不只进 score，也进 Value / message |
| Sample-wise conditioning | 当前样本内，dynamic tokens 只查询当前这只手的 static tokens |
| Static token set | per-joint / per-link 的结构化 token 集，而不是单个 global URDF embedding |
| Local main path + relational residual | 低层 proprio 控制走 local path，结构/协调信息走 relational residual |
| Short-history feature stack | 历史沿 feature 维堆叠，而不是时间维 token 化 |

## 遗留问题

本轮虽然把主体架构和语义澄清了很多，但以下问题仍未最终定稿：

1. **训练范式**：
	- GET 式 per-expert distillation
	- 端到端 RL（多 env group）
	- 或 hybrid

2. **历史建模的最终选型**：
	- feature-stack
	- single-step + previous action / target
	- 图外小 GRU / TCN

3. **policy head 的具体实现**：
	- concat
	- residual sum
	- gated residual

4. **per-joint static token 字段表**：仍需细化

5. **relation edge $e_{ij}$ 字段表**：仍需细化

6. **辅助损失**：
	- 是否保留 FK self-modeling
	- 是否扩展到更多 embodiment-aware auxiliary losses

7. **真实 20Hz 预算**：
	- network / tokenizer / env / hardware I/O 的频率预算仍需实测验证

## 下一步方向

基于本轮讨论，明确的下一步应当是：

1. **把 Method 章节初稿写出来**，把当前拍板版架构固化为可实现文档；
2. **补齐 `per-joint static token` 与 `relation edge e_{ij}` 的字段表**；
3. **决定历史建模方案**（优先在 `feature-stack` 与 `图外小 temporal encoder` 之间收敛）；
4. **决定训练范式**（至少给第一版原型一个默认 recipe）；
5. **做 20Hz 预算与 forward benchmark**，避免方法从一开始就超出控制频率预算。

## 讨论记录

本轮详细纪要见：

- `AnyMani/source/anymani/ideas/graph/discuss4.ipynb` 的 `5.1`–`5.8`

它们覆盖了：

- 论文 / 代码精读结论；
- cross-attention 语义澄清；
- static stream 与 relation stream 的角色划分；
- 修正版总图；
- 信息流与残差设计；
- 频率与时间步建模判断。
