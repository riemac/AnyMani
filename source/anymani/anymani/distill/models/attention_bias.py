r"""attention graph bias $b_{ij}$ 的接口契约。

对应 `Research/总体/网络架构.md` §8（边关系 / graph bias：暂不裁定，但接口预留）。

== 基本公式 ==

对第 $h$ 个 attention head，标准注意力 logits 可写为：

$$
a_{ij}^{(h)} = \frac{q_i^{(h)\top} k_j^{(h)}}{\sqrt{d_h}} + b_{ij}^{(h)}.
$$

本模块只负责把已经构造好的 edge feature 转成 $b_{ij}^{(h)}$。teacher 当前阶段
不承担 sim2sim，因此默认可以使用当前 FK 下的 all-pairs dynamic SE(3) edge feature；
student / 真实 URDF 部署阶段再单独处理 frame 语义对齐问题。

== 候选路线（不裁定）==

1. **无 bias**：$b_{ij}=0$。最稳，作为第一版 baseline。
2. **边类型标量 bias**：Graphormer 风格，每种边类型每个 head 一个标量：
   $$
   b_{ij}^{(h)} = \beta_{\phi(i,j)}^{(h)}.
   $$
3. **HGT 风格类型三元组 bias**：同时考虑 source type、edge type、target type：
   $$
   b_{ij}^{(h)} = \beta_{\phi(i,j)}^{(h)} +
                  \mu_{(\tau_i,\phi(i,j),\tau_j)}^{(h)}.
   $$
4. **连续边特征 bias**：用相对位姿、距离、同指关系等 edge feature 生成 bias：
   $$
   b_{ij}^{(h)} = {w_{\phi(i,j)}^{(h)}}^\top \mathrm{edge\_feat}(i,j).
   $$
5. **TRO-style edge feature**：不仅给 logits 加标量，也可能让边特征参与 value
   构造。这个表达力更强，teacher 后续可考虑；第一版先只做 logits bias。

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
    这里是待定接口，但 teacher 阶段当前合意为：使用 all-pairs dynamic SE(3)
    edge feature 生成 logits bias。仍建议保留 `NoAttentionBias` 作为消融基线，
    但不再把它作为 teacher 默认路线。
"""

# TODO: 定义 `AttentionBiasProvider` 协议或基类，输入 type_ids、valid_mask、
#       edge features（teacher 默认含 all-pairs dynamic SE(3)，可叠加 edge type），
#       输出 `[B, num_heads, T, T]` 或 `None`。

# TODO: 实现 `SE3EdgeBias`：用小 MLP 将 `edge_feat: [B,T,T,F_e]` 映射到
#       per-head logits bias `[B,H,T,T]`。同时保留 `NoAttentionBias` 作消融。

# TODO: 若采用 PyTorch 标准 `TransformerEncoderLayer`，需要确认是否容易注入
#       per-head bias；若不方便，可能需要自定义 attention layer。此为实现期问题，
#       当前只预留设计接口。
