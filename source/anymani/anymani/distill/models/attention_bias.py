r"""attention graph bias $b_{ij}$ 的接口契约。

本模块记录 teacher 第一版当前已裁定的 edge-bias 默认路线：
`HybridGraphSE3Bias`。它只定义契约，不实现 PyTorch module，也不接入
Transformer backbone。

== 基本公式 ==

对第 $h$ 个 attention head，标准注意力 logits 可写为：

$$
a_{ij}^{(h)} = \frac{q_i^{(h)\top} k_j^{(h)}}{\sqrt{d_h}} + b_{ij}^{(h)}.
$$

本模块只负责把已经构造好的 edge feature 转成 $b_{ij}^{(h)}$。teacher 当前阶段
不承担 sim2sim，因此默认可以使用当前 FK 下的 all-pairs dynamic SE(3) edge feature；
student / 真实 URDF 部署阶段再单独处理 frame 语义对齐问题。

== 消融矩阵（已裁定默认）==

1. **`none` / 无 bias**：$b_{ij}=0$。最稳，作为必要 baseline。
2. **`structural` / 结构 bias**：Graphormer 风格的有向 edge type、kinematic
   distance bucket、same-finger 二值关系，每个 head 一个或一组标量：
   $$
   b_{ij}^{(h)} =
   \beta_{\phi(i,j)}^{(h)}
   +\gamma_{d(i,j)}^{(h)}
   +\delta_{\mathrm{same\_finger}(i,j)}^{(h)}.
   $$
   它检验“拓扑先验本身”是否有帮助，但无法表达 post-mutate 的连续几何差异。
3. **`se3` / 连续 SE(3) bias**：只用 all-pairs dynamic SE(3) edge feature 生成 bias：
   $$
   b_{ij}^{(h)} = f_{\theta}^{(h)}\!\left(\tilde E_{ij}^{t}\right).
   $$
   它检验连续几何是否足够，且不依赖离散图先验。
4. **`hybrid_se3` / 默认主线**：结构 bias + 连续 SE(3) bias：
   $$
   b_{ij}^{(h)} =
   \beta_{\phi(i,j)}^{(h)}
   +\gamma_{d(i,j)}^{(h)}
   +\delta_{\mathrm{same\_finger}(i,j)}^{(h)}
   + f_{\theta}^{(h)}\!\left(\tilde E_{ij}^{t}, m_{ij}\right).
   $$
   这是 teacher 第一版推荐默认：离散结构帮助 PPO 早期快速识别近邻/同指关系，
   连续 $SE(3)$ 边保留 mount perturb、link scale、tip offset 与当前 FK 姿态。

HGT 三元组 bias 与 TRO-style value injection 暂不进入第一版默认实现：
前者参数表更大，后者需要改 value/message 计算路径；二者可在 `hybrid_se3`
稳定后作为增强路线，而不是最初 PPO 的复杂度来源。

== 当前可能有用的边信息候选 ==

- token type pair：palm-joint、joint-tip、joint-joint、tip-joint 等；
- 同一手指链 / 不同手指链；
- parent-child 方向；
- kinematic graph 最短路径距离；
- joint frame / tip frame / palm frame 间相对位姿；
- post-mutate 后根关节挂载点变化带来的静态几何差异，优先作为
  `PALM -> root JOINT` edge feature，而不是 flatten 到 palm token 或塞进 root joint token。

== static edge 与 dynamic edge 的区分 ==

这里不能笼统说“edge feature 在 episode 内不变”。更准确地分两层：

1. **Static embodiment edge**：
   由资产本身决定，在 episode 内不随 $q_t$ 变化。例如 palm→root mount、相邻
   joint 在 home pose / URDF parent-child frame 下的相对位姿、joint→tip offset。
   它描述的是 morphology。teacher 阶段即使拓扑相同，post-mutate 的 link length、
   mount pose、tip offset 仍会让这些连续几何边发生变化，因此比 get-zero 的 hop
   distance 更有信息量。
2. **Dynamic kinematic edge**：
   由当前关节角 $q_t$ 经 FK 计算得到，例如当前姿态下 joint/link frame 间的
   相对 SE(3)。它随时间变，类似 `tro-grasp` 中根据 noisy/current link pose
   重算的 `E_RR` / `E_OR`。它可能很有用，但也更接近显式 FK 特征，sim2sim 时
   会暴露真实 URDF frame 语义不对齐问题。

teacher 当前合意先使用 `current_all_pairs_se3`：

$$
E_{ij}^{t}=\log\left(T_i(q_t)^{-1}T_j(q_t)\right)\in\mathbb{R}^{6}.
$$

这里的 token frame 包括 `PALM`、所有 `JOINT`、所有 `TIP`。对 `palm→root`，
该动态表达退化为挂载点；对 `palm→descendant joint/tip`，它等于挂载点与后续
FK 链的复合；对 `joint→joint`、`joint→tip`，它提供当前姿态下的相对几何。
这比 get-zero 的 hop distance 更适合 teacher 阶段的同拓扑异几何训练。

== bias 与 mask 的边界 ==

`b_{ij}^{(h)}` 是**软 inductive bias**，不是 attention mask。全局 self-attention
仍允许所有有效 token 两两通信，保证跨指协同、手掌-指尖交互和非相邻补偿动作
可以被策略直接利用。padding / invalid token 的屏蔽应由 backbone 的 valid-mask
路径处理，不应混入 edge bias 的物理语义。

输出形状约定：

```text
attention_bias: [B, H, T, T]  # H 为 attention heads，T 为 token 数
```

其中第 $h$ 个 head 在 logits 上执行：

$$
\mathrm{logits}^{(h)}_{ij}
= \frac{Q_i^{(h)}K_j^{(h)\top}}{\sqrt{d_h}} + b_{ij}^{(h)}.
$$

== 归一化与初始化 ==

- 方向必须保留：$E_{ij}\neq E_{ji}$，不能把 parent→child 与 child→parent 合并。
- 平移项应按 hand scale / palm extent / hand radius 归一化，避免米制平移量纲与
  $so(3)$ 旋量量纲混在一起。
- 旋转项可用 $\log(R_i^{-1}R_j)/\pi$ 归一化到近似 $[-1,1]$。
- `HybridGraphSE3Bias` 的连续 MLP 最后一层应零初始化，或加 learnable gate
  $\alpha \approx 0$，使训练初期近似 `NoBias`，避免随机 bias 让 softmax 过早饱和。

== 关于 mount / chirality 的重要裁定 ==

mount pose 是关系量，不是节点内禀量。更准确地说，根手指挂载点是
`palm frame -> root joint frame` 的相对变换：

$$
E_{p\to r_i} = T_{\text{palm}}^{-1} T_{r_i}.
$$

把所有 $E_{p\to r_i}$ flatten 到单个 palm token 会遇到两个问题：

1. 手指数 $N_t$ 可变，palm token 需要可变长度 set encoder 或固定 padding；
2. flatten 后 mount 与具体 root joint 的 binding 容易变弱，attention 还要重新学
   “第几个 mount 对应哪个 root”。

把它放进 root joint token 也不理想，因为 root joint 会比内部 joint 多一个
“palm mount”私有字段，破坏 `JOINT` token 的同构性。当前倾向：mount pose 属于
edge feature / graph relation，chirality（所谓 left/right）由这些连续关系诱导，
不作为独立离散输入 embedding。

TOAGENT:
    这里是契约接口，teacher 阶段当前合意为：默认使用 `hybrid_se3` logits bias，
    并保留 `none` / `structural` / `se3` 消融。不要在本文件实现 attention layer；
    真实实现应等 backbone bias 注入路径确定后再落。
"""

# TODO: 定义 `AttentionBiasProvider` 协议或基类，输入 type_ids、valid_mask、
#       `RelationFeatureBatch`（含 all-pairs dynamic SE(3)、edge_type、distance_bucket、
#       same_finger），输出 `[B, num_heads, T, T]` 或 `None`。

# TODO: 实现 `NoAttentionBias`：返回 `None` 或全零 bias，用作 PPO 稳定 baseline。

# TODO: 实现 `StructuralGraphBias`：学习 `edge_type`、`distance_bucket`、`same_finger`
#       的 per-head 标量表，输出 `[B,H,T,T]`。

# TODO: 实现 `SE3EdgeBias`：用小 MLP 将 normalized `edge_feat: [B,T,T,F_e]` 映射到
#       per-head logits bias `[B,H,T,T]`，要求最后一层零初始化或 gate 近零初始化。

# TODO: 实现 `HybridGraphSE3Bias`：把 `StructuralGraphBias` 与 `SE3EdgeBias` 相加，
#       作为 teacher 第一版默认 bias provider。

# TODO: 若采用 PyTorch 标准 `TransformerEncoderLayer`，需要确认是否容易注入
#       per-head bias；若不方便，可能需要自定义 attention layer。此为实现期问题，
#       当前只预留设计接口。
