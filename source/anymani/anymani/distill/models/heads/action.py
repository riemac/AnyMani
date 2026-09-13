r"""JOINT-only action head contract。

若最终 backbone latent 为 $H^{(L)}\in\mathbb R^{B\times T\times D}$，action head 只读取
由 grouped-token metadata 标出的有效 JOINT tokens：

$$
\mu_j
=
f_{act}(h_j^{(L)}),
\qquad
j\in\mathcal J_{revolute}.
$$

$\mu_j$ 对应第 $j$ 个可控 revolute joint 的 raw relative-delta action mean。当前每关节
动作维度为 1；log-standard-deviation 初期仍倾向 global scalar 或 action-dimension 参数，
不因 token 化而默认让每个 token 独立预测 exploration scale。

输出必须按 canonical ordered joint schema flatten 为 rl_games action tensor，并显式拒绝
permuted/equal-DOF mismatch。PALM、TIP 与 pooled hand latent 不直接进入 action slot。
"""
