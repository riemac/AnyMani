r"""Shared hand-frame Basis Point Set query 对照契约。

BPS 当前不是 active representation route。主线采用 mount-conditioned physical anchors、
all-anchor query relations 与 sampled conditional-implicit field；本模块保留 fixed ordered query
对照，用于必要时区分“field family 的收益”和“query layout 的收益”。

hand frame ``{h}`` 只固定有向 palm normal $z_h=n_p$；绕 $n_p$ 的 $x^h/y^h$ basis
属于 $SO(2)$ gauge，不承载 palm 宽度、手指伸展或 thumb-side 语义。第 $r$ 个 basis
point 只是一个位置向量：

$$
b_r^h=(x_r^h,y_r^h,z_r^h)\in\mathbb R^3,
$$

三个坐标单位均为 m。basis point 不是 frame，没有自己的 orientation，不从多个 palm
frame 取均值；每个点独立产生一个 field scalar，固定顺序组成 $N_Q$ 维 vector。$K$
始终表示 physical anchor 数，不能复用为 BPS/query 数。

若执行该对照，跨资产共享 uniform hand-frame basis 会让所有 embodiment 与 surface owners
在相同 ordered positions 上读取 field。成对 $SO(2)$ 验收必须让 basis 与完整坐标化输入共同
旋转；把 basis 固定到任意 $x/y$ 方向只能作为明确的 gauge-sensitive control。basis 不能跟随
当前 owner 运动，否则 $q$ 引起的真实 placement 变化会被 query layout 抵消。

workspace bounds、点数 $N_Q$ 与 layout 尚未裁定。它们必须覆盖 generated family 的真实
reachable envelope，并以 sphere/capsule/wedge/custom-tip 的 shape discrimination 与
held-out morphology coverage 选择，而不是凭视觉密度决定。
"""
