r"""Encoder-only self-attention 主干设计契约。

对应 `Research/总体/网络架构.md` §5（Encoder-only 主干）。

== 输入输出 ==

主干只消费 tokenizer 投影后的统一 token 序列：

$$
H^{(0)} \in \mathbb{R}^{B\times T\times D},
$$

其中 $T=N_p+N_j+N_t$。到达本模块时，palm/joint/tip 的原始维度差异已经由
`tokenizer.py` 消除；主干不再做分组隔离，而是让所有 token 在同一个序列里
做双向 self-attention。

== 为什么中间不再分组 ==

分组 projection 的目的，是把异构物理语义投影到同一隐空间；投影后继续分组隔离，
会阻断 joint 与 tip、palm 与 joint 的信息通信，违背 attention 主干的意义。
因此当前原则是：

```text
输入端分组 → 中间统一 self-attention → 输出端按 token 类型路由
```

== 注意力形式 ==

每一层可写为：

$$
H^{(l+1)} = \mathrm{EncoderLayer}(H^{(l)},\; M,\; b_{ij}),
$$

其中 $M$ 是 padding mask，$b_{ij}$ 是 attention bias。teacher 当前合意使用
由 all-pairs dynamic SE(3) edge feature 产生的 $b_{ij}$；同时保留
$b_{ij}=0$ 的 no-bias 路径作为消融基线。

== RL 稳定性取舍 ==

- 起步建议 Pre-LN，PPO 中通常比 Post-LN 更稳；
- dropout 初期可为 0，避免在线 RL 的额外随机性；
- 层数与宽度先保守（如 $D=128$、4 层、4 heads），待 teacher 稳定后消融。

TOAGENT:
    本文件当前为设计契约，不写 `nn.TransformerEncoderLayer` 真实代码。实现时
    必须显式处理 mask 语义转换：项目内部 `valid_mask=True` 表示有效，而 PyTorch
    key_padding_mask 多为 `True=屏蔽`。
"""

# TODO: 定义 `EmbodimentEncoderBackbone`，输入 `tokens: [B,T,D]`、
#       `valid_mask: [B,T]`、可选 `attention_bias: [B,H,T,T]`，输出同形状 `[B,T,D]`。

# TODO: attention bias 的生成与主干分离：backbone 只接收已经计算好的 bias，
#       不直接解析 TokenType / edge feature。这样可保持 Graphormer/HGT/TRO-style
#       relative pose bias 的可插拔性。

# TODO: 若后续加入 factorized time attention，应新增独立 time backbone，
#       不在本 joint/palm/tip token 主干里混入时间轴 causal mask。
