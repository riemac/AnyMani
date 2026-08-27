r"""PALM / JOINT / TIP 分组 projection 与统一 token batch 契约。

本模块消费已经整理好的 token-ready tensors，不直接解析 URDF、sidecar 或 IsaacLab env。
三类输入具有不同物理语义和原始维度，因此分别投影到统一隐空间：

$$
h_i^{(0)}
=
P_{\tau_i}(x_i)
\in\mathbb R^D,
$$

其中 $x_i$ 是第 $i$ 个原始 token feature，$\tau_i\in\{\mathrm{PALM},
\mathrm{JOINT},\mathrm{TIP}\}$ 是语义角色，$P_{\tau_i}$ 是 type-specific projection，
$D$ 是统一 token width。不能先把关节角、contact force 与 palm geometry 填零到相同
raw width 后强迫共享一个 Linear。

逻辑输入输出：

```text
palm_x / joint_x / tip_x       : [B, N_type, d_type]

tokens     : [B, T, D]，T=N_p+N_j+N_t
type_ids   : [B, T]
owner_ids  : [B, T]，与 surface-owner/decoder axis 直接同索引
slices     : PALM/JOINT/TIP 在同结构组 token 轴上的稳定路由 metadata
```

同一次前向固定结构模式与 $N_E/N_J$，不使用 entity padding；不同结构模式分别前向。输出
必须携带 role/owner metadata，使 action、value 与 SSL decoder 不依赖跨手型固定槽位。
projection hidden width 与具体 feature dimension 尚未裁定，不能在本 scaffold 中写死。

Finger mount 是 PALM 与 root JOINT 的关系量，不应把可变长度 mount set flatten 到 PALM
token，也不应只给 root JOINT 增加破坏同构性的私有槽。type embedding 是 projection 后的
可选角色标记；``left/right`` 不作为默认离散 type，chirality 优先由 hand-frame mount/
physical geometry 连续关系表达。当前 geometry encoder 不增加 HAND/CMD token；未来策略侧
若比较全局 token，必须作为独立下游消融。
"""
