r"""由静态 local geometry 与当前 $q$ 构造 posed physical oracle。

设 $T_{hg}(q)\in SE(3)$ 把 semantic group $g$ 的 local coordinates 变到 hand frame
``{h}``，$x^h\in\mathbb R^3$ 是 ``{h}`` 中的 query point，单位 m。在线 target
generation 不必每个 batch 真正移动 mesh 或重建 BVH；可将 query 反变换到缓存 local
geometry：

$$
x^g
=
T_{hg}(q)^{-1}x^h.
$$

若 local collision surface 为 $\mathcal M_g$，当前 hand-frame physical surface 是
$\mathcal S_g^h(q)=T_{hg}(q)\mathcal M_g$。对任意 local-frame gauge
$G_g\in SE(3)$，同步执行 $T'_{hg}=T_{hg}G_g$ 与
$\mathcal M'_g=G_g^{-1}\mathcal M_g$ 后，$\mathcal S_g^h(q)$ 必须保持不变。

稠密 current surface、distance、最近点、surface Jacobian 与 field samples 只服务
target/evaluation。条件隐式路线的 retained input 只能消费 $q$、screw、topology 与 home
geometry。解析直接压缩只保留为未来候选占位；若激活，可消费由缓存 $p_{g,r}^{local}$ 与当前
$T_{hg}(q)$ 普通前向得到的有限 support points $p_{g,r}^h(q)$。任何路线都禁止把 target-side
最近点或 dynamic all-pairs pose answer 注入 retained encoder。
"""
