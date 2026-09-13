r"""Per-semantic-group 的非归一化 parametric Gaussian field 候选。

本模块记录**显式 Gaussian components/splats-like** 分支，不是当前 active multiscale
Gaussian distance shell。当前首选 field 位于 ``density.py``：它直接把 posed surface 的 UDF
经多个 $\sigma$ 映射为连续 density，不要求 decoder 回归 component correspondence。保留本模块
用于未来压缩/参数化 field 对照，不代表 explicit mixture 与当前主线同优先级。

decoder 可以输出 $R$ 个 anisotropic Gaussian components，并由它们诱导连续场：

$$
\widehat\rho_g(x^h;q)
=
\sum_{r=1}^{R}
a_{g,r}
\exp\!\left[
-\frac12
(x^h-\mu_{g,r}^h)^{\top}
(\Sigma_{g,r}^h)^{-1}
(x^h-\mu_{g,r}^h)
\right].
$$

$a_{g,r}\ge0$ 是非归一化 amplitude；$\mu_{g,r}^h\in\mathbb R^3$ 是 hand-frame中心，单位 m；$\Sigma_{g,r}^h\in\mathbb R^{3\times3}$ 是 symmetric positive-definite covariance，单位 $\mathrm{m}^2$。

不要求 $\sum_r a_{g,r}=1$，以免把大/小body 强行压成相同总质量的概率分布。

该对象更准确地称为 parametric Gaussian field 或 splat-like field，而不是默认等同于
图形学中含投影与 alpha compositing 的 Gaussian splatting。component 没有天然顺序，
同一物理场也可能由多组不同参数近似；因此主监督在 query space 比较 induced field，
而不是要求数据集提供唯一 Gaussian parameter target。$R$、SPD parameterization 与
component-collapse regularization 留待实现/消融阶段裁定。
"""
