r"""输出头设计契约：joint action / global value / tip auxiliary。

对应 `Research/总体/网络架构.md` §6（输出头）。

== 输出路由原则 ==

Encoder 输出为：

$$
H^{(L)} \in \mathbb{R}^{B\times T\times D}.
$$

但不是所有 token 都输出动作。当前裁定：

- `JOINT` token → actor action head，输出每个可控关节的增量动作均值；
- `TIP` token   → 可选 auxiliary head，如 fingertip FK / contact 自监督；
- `PALM` token  → 当前无明确 aux 价值，通常仅参与 value pooling 或丢弃。

== Actor ==

动作空间为 joint space，因此 actor 只读取 revolute joint token：

$$
\mu_j = f_{\text{act}}(h_j^{(L)}),\quad j\in\mathcal{J}_{\text{revolute}}.
$$

其中 $\mu_j$ 对应第 $j$ 个关节的 raw relative delta action 均值。log_std 初期
建议为全局可学习参数（scalar 或 action_dim 向量），不要让每个 token 自己预测
log_std，以免 PPO 方差估计过早复杂化。

== Critic ==

critic 输出标量 $V(s)$。它可以从全 token pool，也可以只从 joint token pool，或
从 palm/hand-level token 读取。当前不裁定具体 pooling 方式。

== Auxiliary ==

`TIP` token 可选接 FK / contact 相关自监督。由于 tip 是接触主场，且我们已有独立
tip token，把 FK aux 放在 tip 上比 get-zero 式“所有 joint 都预测 self-modeling”
更聚焦。但这仍是增强项，不应阻碍第一版主干跑通。

TOAGENT:
    本文件当前只定义输出语义契约。实现时要避免把 action head 误接到 palm/tip。
    建议所有 head 都通过 tokenizer 返回的 `joint_slice` / `tip_slice` 路由，而不是
    通过硬编码 token 数切片。
"""

# TODO: 定义 `ActionHead`：输入 `joint_tokens: [B,N_j,D]`，输出
#       `mean: [B,N_j,action_dim_per_joint]`，再由 policy flatten 为 rl_games 需要的
#       `[B, action_dim]`。

# TODO: 定义 `ValueHead`：输入全体 `tokens: [B,T,D]` 与 `valid_mask: [B,T]`，输出
#       `value: [B,1]` 或 `[B]`。pooling 方式由 `HeadsCfg.value_pool` 决定。

# TODO: 定义 `TipAuxHead`：输入 `tip_tokens: [B,N_t,D]`，输出具体 aux target。
#       目标候选包括 fingertip root-frame position、pose、contact state 等，
#       当前不写死。
