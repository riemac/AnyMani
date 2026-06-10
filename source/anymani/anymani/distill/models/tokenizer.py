r"""分组 tokenizer 设计契约 —— palm / joint / tip 各自投影，再聚合为统一 token 序列。

对应 `Research/总体/网络架构.md` §3（当前收敛的主干）与 §4（Projection 层形式）。

== 本模块的职责 ==

tokenizer 消费**已经整理好的 token-ready 张量**，而不是直接解析 URDF / YAML / IsaacLab
环境对象。它负责：

1. 分别接收 `PALM` / `JOINT` / `TIP` 三组原始特征；
2. 使用各自 projection $P_{\tau}$ 投影到统一 $D$ 维；
3. 可选叠加 type embedding $e_{\tau}$；
4. 按固定顺序聚合为 $H^{(0)}\in\mathbb{R}^{B\times T\times D}$；
5. 合并各组 mask，形成模型内部统一的 valid-token mask。

== 为什么不是共享一个 Linear ==

三类输入的物理语义不同，甚至原始维度也不应相同：

- `JOINT`: $[q, \dot q, a_{t-1}, \text{axis}, \text{limit}, \text{child geometry}, \ldots]$；
- `TIP`  : $[\text{tip mesh descriptor}, \text{contact point/force}, \ldots]$；
- `PALM` : $[\text{palm/global context}, \ldots]$。

特别注意：finger root 的 mount pose 不在此处 flatten 到 `PALM` token，也不放进
root `JOINT` token 的私有字段。它应作为 `PALM -> root JOINT` 的 edge feature，
以保留“哪个 mount 绑定到哪个 finger/root”的关系结构，并避免可变手指数导致的
palm token 固定维度问题。

强行 padding 到同一 raw feature 维度再共享 Linear，会让同一权重同时解释关节角、
接触力和 palm 几何，属于语义混叠。分组 projection 是必要复杂度。

== 计划中的输入/输出 shape ==

输入（具体 feature 维度待定）：

```text
palm_x  : [B, N_p, d_palm]
joint_x : [B, N_j, d_joint]
tip_x   : [B, N_t, d_tip]

palm_mask  : [B, N_p]  # True=有效，False=padding
joint_mask : [B, N_j]
tip_mask   : [B, N_t]
```

输出：

```text
tokens     : [B, T, D], T=N_p+N_j+N_t
valid_mask : [B, T],    True=有效 token
type_ids   : [B, T],    TokenType 的整数编码
slices     : palm/joint/tip 在 token 轴上的切片，用于输出路由
```

NOTE: PyTorch `nn.TransformerEncoder` 的 `src_key_padding_mask` 语义通常是
      `True=屏蔽`，与本项目内部 `True=有效` 相反。转换应只发生在 backbone
      边界，避免 mask 语义在各文件中散落。

TOAGENT:
    本文件当前只写设计契约，不写 `nn.Module`。实现时应保留本 docstring 中的
    shape 与语义说明，并把 TODO 转化为具体类（如 `TypedTokenBatch`、
    `GroupedTokenizer`、`TypeProjection`）。
"""

# TODO: 定义 `TypedTokenBatch` 数据结构，显式承载 palm/joint/tip 三组输入张量、
#       mask、asset/embodiment id 等元数据。不要用裸 tuple 传递，避免 shape 误读。

# TODO: 定义 `GroupedTokenizer`，内部包含 `P_palm`、`P_joint`、`P_tip` 三个 projection。
#       当前设计倾向每个 projection 为轻量 MLP：Linear → LayerNorm → GELU → Linear。

# TODO: projection 后追加 type embedding。是否需要独立 `[HAND]` / `[CMD]` token，
#       仍属 `网络架构.md` §10 待定项，不在第一版强行加入。

# NOTE: 不提供 `left/right` type embedding。左右手 / chirality 由 palm-frame mount
#       layout 与 edge geometry 诱导，而不是作为独立离散输入直接喂给 policy。

# TODO: 返回值应包含 token 轴切片信息，如 `joint_slice`，使 actor head 只读取
#       `JOINT` token，不依赖硬编码的 token 数。
