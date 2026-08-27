r"""Parametric Gaussian decoder 的 query-space field objective contract。

decoder 输出 components 后，先在 target queries 上计算 induced density
$\widehat\rho_g(x_k^h;q)$，再与 physical density $\rho_g(x_k^h;q)$ 比较：

$$
\mathcal L_{GaussianField}
=
\mathcal L_{field}
+
\lambda_{scale}\mathcal R_{scale}
+
\lambda_{amp}\mathcal R_{amplitude}
+
\lambda_{collapse}\mathcal R_{collapse}.
$$

$\mathcal L_{field}$ 复用 sampled-field reconstruction；其余项分别约束 covariance axis
过小/过大、amplitude 爆炸与 components 全部坍缩到同一点。各 $\lambda\ge0$ 是尚未
裁定的无量纲权重，不能在 scaffold 中给出伪默认值。

主 loss 不直接逐 component 对齐参数，因此对 component permutation 不敏感。若未来加入
Hungarian/optimal-transport parameter-set loss，它只能作为额外消融，并必须解释 target
Gaussian set 如何确定、为何不是任意拟合结果。
"""
