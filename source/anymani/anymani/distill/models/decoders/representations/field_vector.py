r"""Fixed ordered queries 的 field-vector decoder contract。

对每个有效表面归属体，decoder 从同索引 entity latent 预测 $N_Q$ 个固定 hand-frame
query values；$N_Q$ 是 fixed-query baseline 的 query 轴，$K$ 仍专指 physical anchors：

$$
\widehat{\mathbf f}_g(q)
=
[\widehat f_g(b_1^h;q),\ldots,\widehat f_g(b_{N_Q}^h;q)]^{\top}
\in\mathbb R^{N_Q}.
$$

$b_k^h$ 来自共享 ordered BPS layout；$f$ 可以是 UDF、SDF、truncated distance、density
或 occupancy。output order 由 basis order 决定，不是无序点集。decoder 不重新生成 basis，
也不读取 target labels；valid groups/queries 由 target routing mask 控制。

具体 MLP/linear depth、共享或 type-specific 参数、$N_Q$ 与 loss 尚未裁定。该 decoder 是
fixed-query baseline，默认 pretraining-only；PPO 继承产生 $Z^{(0)}/Z^{(1)}$ 的 input
adapter + backbone + retained latent heads。
"""
