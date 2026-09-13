r"""UDF、SDF 与 truncated distance field 的共同数学契约。

给定 hand-frame query $x^h\in\mathbb R^3$ 与 semantic group 的当前 surface
$\mathcal S_g^h(q)$，unsigned distance 为：

$$
u_g(x^h;q)
=
\min_{y\in\mathcal S_g^h(q)}\|x^h-y\|_2
\in\mathbb R_{\ge0},
$$

单位为 m。UDF 只需要可信 surface evidence，适合 non-watertight mesh 与 posed BPS
safe baseline，但不区分实体内部、外部或 penetration side。

Signed distance $s_g(x^h;q)\in\mathbb R$ 当前候选约定为体内负、体外正；它需要闭合、
定向或其他可靠 inside/outside evidence。Truncated SDF/UDF 使用物理半径 $\tau>0$ m
截断后再按 $\tau$ 归一化，得到有界无量纲 target。$\tau$ 尚未裁定，必须与 hand/tip
尺度和 contact-relevant band 联合消融，不能作为匿名常数写死。

同一 physical surface 在 remeshing、点数、点排列或 pure local-frame gauge 改写后，
distance target 应保持一致；真实改变 $q$、尺度、wedge 方向或装配位置时则应改变。
"""
