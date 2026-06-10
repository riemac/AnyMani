r"""token 间关系 / edge feature 构造契约。

本模块负责定义“哪些 token pair 之间有什么关系特征”。它位于 `tokenizer.py`
与 `attention_bias.py` 之间：

```text
tokenizer 输出 token frames / type_ids / valid_mask
    └── relations.py 构造 edge_feat(i,j)
          └── attention_bias.py 将 edge_feat(i,j) 映射为 attention logits bias b_ij
```

== 当前 teacher 合意 ==

teacher 不承担 sim2sim，因此可以直接使用通用建模资产当前 palm/joint/tip frame 下的
all-pairs dynamic SE(3) 关系：

$$
E_{ij}^{t}=\log\left(T_i(q_t)^{-1}T_j(q_t)\right)\in\mathbb{R}^{6}.
$$

其中 token frame 集合包括：

- `PALM`：当前采用 generated asset 的 canonical palm frame；
- `JOINT`：每个 revolute joint frame；
- `TIP`：每个 fingertip/contact sensor frame。

这个定义自动覆盖 mount 语义：

- `palm -> root joint`：就是 finger mount / 指根挂载位姿；
- `palm -> descendant joint`：mount 与链式 FK 的复合；
- `palm -> tip`：mount、整根手指 FK 与 tip offset 的复合；
- `joint -> joint`：当前姿态下的关节间相对位姿；
- `joint -> tip`：当前姿态下关节到指尖 frame 的相对位姿。

因此不需要新增 `MOUNT` token。若 `MOUNT` token 只存 $T_{palm}^{-1}T_{root}$，
它并没有比 `palm -> root joint` edge 多任何物理信息，反而增加 token 类型与
路由复杂度。

== 与 static edge 的关系 ==

static embodiment edge（如 home pose / URDF rest pose 下的相邻相对位姿）仍可作为
消融或额外输入，但 teacher 当前主路线是 dynamic all-pairs SE(3)。
get-zero 式 hop distance 在 same-topology teacher 阶段通常是常量，对 post-mutate
几何变化没有信息量，因此不作为 teacher 主特征。

== sim2sim 边界 ==

这里默认 generated asset 的 palm/joint/tip frame 语义可信。真实 Leap / Allegro URDF
的 frame 语义对齐问题留给 student / sim2sim 阶段。该问题可通过虚拟 palm frame、
mesh-based alignment、BPS/OBB/PCA 等方式探索，但不应阻塞 teacher 的训练脚手架。

TOAGENT:
    本文件当前只写设计契约，不实现 FK。实现时应避免把 edge construction 写进
    `attention_bias.py`；后者只负责 edge_feat → logits bias。
"""

# TODO: 定义 `RelationFeatureBatch` 数据结构，至少包含：
#       `edge_feat: [B,T,T,F_e]`、`edge_valid_mask: [B,T,T]`、`edge_type_ids: [B,T,T]`。

# TODO: 定义 `TokenFrameProvider` / `TokenFrameBatch`，从环境或资产缓存中提供
#       `T_i(q_t): [B,T,4,4]`。teacher 第一版可直接用 Isaac/asset 已知 frame；
#       student/sim2sim 再替换成对齐后的虚拟 frame。

# TODO: 定义 `AllPairsDynamicSE3RelationBuilder`，计算
#       $E_{ij}^{t}=\log(T_i(q_t)^{-1}T_j(q_t))$，输出 6D se(3) 向量。

# TODO: 同时生成离散 edge_type，如 palm-joint、joint-tip、same-finger、self-edge、
#       padding-edge，用于给 SE(3) edge feature 附加语义标签。
