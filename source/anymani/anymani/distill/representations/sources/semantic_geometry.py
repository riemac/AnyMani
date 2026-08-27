r"""Collision pieces 到 semantic geometry group 的可审计映射契约。

field target 的基本单位是经过审核的 PALM/JOINT/TIP physical entity；第 $g$ 个实体同时是
第 $g$ 个表面归属体与 SSL decoder 归属轴，因此 $G=N_E$，不再维护 group-to-token 多对一
映射。该实体仍不要求与原始 URDF link 或 PhysX rigid body 一一对应：一个 TIP entity 可以
合并最后活动关节后的多个 fixed descendants，一个 JOINT entity 可以拥有同一 child body
上的多个 collision shapes，PALM entity 也可包含经人工确认的 fixed root shell。

可信度顺序固定为：

```text
generated asset manifest / generator truth
    > 人工查看 URDF 与 collision 可视化后确认的 official-asset sidecar
        > URDF graph traversal 产生的候选映射
```

graph heuristic 只负责提出候选、做 coverage validator 或缺省回退，不能把“最后活动
关节后的全部 descendants”静默宣布为跨资产统一的 fingertip 语义。若 DIP、nail 与 tip
已经合并在同一 rigid body 内，只能将整个经确认的 distal union 视作 TIP、显式 mesh
segmentation，或拒绝该资产进入需要更细语义的实验。

正式 cross-source 数据必须支持 owner-colored collision render，并验证每个 collision
piece 恰好归属一个 group。映射不明确时 fail closed；不能为追求数据量制造静默标签错误。

当前阶段还要求 source 显式区分 palm collision solid 与 finger bodies。设
$\Omega_p^h\subset\mathbb R^3$ 是 palm 在 hand semantic axes ``{h}`` 中的 physical
solid，第 $j$ 个 finger branch 的 mount-conditioned anchor support
$\mathcal A_j^h\subset\Omega_p^h$ 只能覆盖 mount 邻域内经过 contact-facing 过滤的 palm
surface/interior。finger skin、free space、wrist/base 杂散结构与远离 manipulation basin 的
候选点不得静默进入 anchor source。

anchors 是从这些 physical supports 取得的 landmarks，不是 raw URDF joint-frame origins。
generated source 使用 generator/mount truth；official LEAP/Allegro 使用版本化 sidecar、collision
可视化与人工审核。anchor support、sampling rule、asset version 与 semantic owner mapping 都属于
provenance，确保 `{h}` origin 改写、remeshing 与 cross-source 对照可以被独立复核。
"""
