r"""mesh / link 几何特征 adapter 设计契约。

对应 `Research/总体/网络架构.md` §3（分组 projection）、§8（边关系候选）以及
`Research/总体/关于 mesh 特征和 joint-centric unified representation.md`。

== 问题背景 ==

generated assets 中，手指连杆大多由 box / cylinder / sphere 等 primitive 描述，
joint frame、mesh frame、child link frame 的关系由 generator 明确约定。因此低维
几何特征（如 $l,w,h$、mesh offset $d$、tip radius/height）有清楚语义。

真实 Leap / Allegro URDF 则不同：visual mesh 与 collision mesh 不规则，joint frame
可能嵌在复杂机械壳体内部，坐标系位置和朝向也不一定与 generated assets 的理想
约定一致。`Research/总体/fig/joint frame in visual mesh.png` 与
`Research/总体/fig/joint frame in collision mesh.png` 展示了这个对齐困难。

因此 mesh 特征不能写死成一个永久固定的字段列表。它需要 adapter 化：

```text
asset geometry / mesh descriptor  ──>  adapter  ──>  token feature / embedding
```

== 候选 adapter（不裁定）==

1. **primitive_low_dim**：
   直接使用 generated asset 的 primitive 真值，如 box 的 $(l,w,h)$、cylinder 的
   $(r,l)$、sphere 的 $r$、mesh offset pose 等。优点是低维、可解释、与 generator
   语义精确对齐；缺点是迁移到真实 URDF 时需要语义对齐工程。

2. **BPS / point-cloud descriptor**：
   借鉴 `tro-grasp`，对 link / tip 点云做 Basis Point Set 编码，再拼接 centroid、
   scale 等。优点是能表达真实 mesh，不要求 primitive 参数；缺点是维度高、
   坐标系和采样策略会引入新的 sim2sim 对齐问题。

3. **offline embedding**：
   离线用自编码器 / contrastive objective / hand-designed alignment 生成固定 embedding，
   policy 训练时只消费 embedding。优点是把复杂对齐问题隔离在离线流程；缺点是
   需要额外数据和验证。

4. **none / ablation**：
   不使用 mesh 特征，只保留 joint limit / axis 等基础形态量，作为对照。
   mount / relative pose 这类关系量不应被混写为节点内禀几何，而应走 edge feature。

== 输出归属 ==

- 普通 child link 几何：更自然地并入 `JOINT` token 的静态特征；
- fingertip mesh：更自然地并入 `TIP` token，与接触动态量共址；
- palm 几何：当前变化小，可并入 `PALM` / hand-level context，后续视消融决定。

TOAGENT:
    本文件当前只写 adapter 设计契约。实现时不要把 URDF 解析逻辑塞进这里；
    那属于 `rl/geo_obs.py` 或 assets 侧。这里只处理“已提取几何张量如何编码”。
"""

# TODO: 定义 `MeshAdapter` 协议或基类，输入原始几何 tensor / descriptor 字典，
#       输出可拼进 palm/joint/tip 原始 token 特征的张量或已对齐到 $D$ 的 embedding。

# TODO: 第一版建议实现 `PrimitiveLowDimAdapter`，只消费 generated asset 已有的
#       box/cylinder/sphere 参数与 mesh offset。真实 URDF 的 BPS / offline embedding
#       暂不进入 teacher RL 第一版。

# TODO: 为 sim2sim 预留 `GeometryAlignmentAdapter`，用于把真实 Leap/Allegro 的
#       mesh descriptor 映射到 generated asset 的几何语义空间。此项是研究难点，
#       不能在脚手架阶段假装已经解决。
