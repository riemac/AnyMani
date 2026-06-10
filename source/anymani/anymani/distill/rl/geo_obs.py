r"""PALM / JOINT / TIP 几何观测契约 —— teacher RL 的静态形态特征入口。

本模块记录 AnyMani teacher 阶段当前合意的 geometry obs schema。它不是网络
`nn.Module`，也不直接实现 mesh encoder；它负责定义从 hand asset / hand.yaml /
URDF / IsaacLab runtime 中应提取哪些**形态学字段**，再交给
`distill/models` 的 tokenizer / geometry adapter 编码为 token。

== 当前上下文 ==

我们已经把 policy 输入结构收敛为三类 token：

```text
PALM token  +  JOINT token × N_j  +  TIP token × N_t
```

并且 teacher 阶段的 token 间关系暂定使用 all-pairs dynamic SE(3) edge：

$$
E_{ij}^{t} = \mathrm{Log}\left(T_i(q_t)^{-1}T_j(q_t)\right) \in \mathbb{R}^{6}.
$$

因此本文件只处理**节点自身的几何 / 形态特征**，不再处理 mount pose、
joint-to-joint relative pose、palm-to-tip pose 等关系量。关系量属于
`distill/models/relations.py`，attention bias 属于 `distill/models/attention_bias.py`。

== 与 observations.py 的边界 ==

`tasks/gm/mdp/observations.py` 负责运行时 obs MDP 的浅契约，尤其是：

- state obs：$q,\dot q,\Delta a_{t-1}$ 等动态关节本体感受；
- contact obs：per-tip 接触点 / 接触力等动态触觉；
- joint limits：语义上属于 geometry / morphology，但 teacher RL 中应由
  IsaacLab runtime 的 `asset.data.soft_joint_pos_limits` 提取，作为不进历史堆叠的
  静态 obs term 暴露给 policy。

limits 不在本模块重复提取。原因有两个：

1. soft limits 是 actor clamp 真正使用的行为边界，比 hand.yaml 的 hard / authored
   limits 更接近策略实际会遇到的边界；
2. 未来若训练中做轻微关节限位 domain randomization，runtime soft limits 才是
   当前样本的真实值，重复从 asset sidecar 解析会不同步。

== teacher vs student / sim2sim ==

当前 teacher 不承担 sim2sim，因此本文件默认 generated asset 的 `{h}` / palm frame /
joint frame / tip frame 语义可信。真实 Leap / Allegro URDF 的 frame 对齐、虚拟 palm
frame、BPS / mesh alignment 等问题留给 student / sim2sim 阶段。本文件会保留
adapter 接口意识，但不把对齐问题提前做成 teacher 阶段的阻塞项。

== 当前第一批 teacher 目标 ==

第一批 teacher 目标是：

`AnyMani/source/anymani/anymani/assets/generated/2026-06-10_11-30-08/single_palm_leap/right_t4_i4_m4_r4/` 下的 post-mutate 产物（待指定）

该 asset bank 的实际 post-mutate 配置来自
`AnyMani/source/anymani/anymani/assets/config/asset_gen_cfg.py`：

- `link_scale` 会变手指 link 的 $l,w,h$；
- 当前没有单独的 `mesh_offset d` mutator，$d$ 仍应作为常量/字段保留；
- `mount_perturb` 会变 palm→finger root 的关系，但它属于 edge / relations；
- `limit_tweak` 会变 soft joint limits，应由 runtime obs term 暴露；
- `tip_replace` 已经会替换 custom tip（round / wedge / thinner / leap_cube / cs），
  因此 tip 不能只用 procedural `cs_radius/cs_height` 这类低维参数表达。

TOAGENT:
    本文件是科研设计契约，当前仍以重注释为主。实现时可在 docstring 下方逐步添加
    dataclass / extractor，但不要删除这里的数学语义和边界说明。若实现完成，
    应把对应 TODO 更新为 DONE 或普通科研注释。
"""

from __future__ import annotations

__all__: list[str] = []


# =============================================================================
# 1. JOINT geometry schema
# =============================================================================

r"""TODO(JOINT geometry): revolute joint token 的静态形态字段。

JOINT token 只对应 surviving revolute joints。fixed root segment、fixed tip joint、
palm 不在 JOINT token 序列中。这样 actor head 可以保持最干净的映射：

$$
h^{\mathrm{joint}}_i \longmapsto \Delta q_i.
$$

当前 JOINT geometry 第一版字段：

```text
q_min, q_max       : runtime soft joint limits，单位 rad；语义归 geometry，来源归 observations.py
axis_h0            : 关节旋转轴在 home pose 的 hand semantic frame {h} 中表达，shape [3]
l, w, h            : 当前 joint child link collision/mesh 的 canonical extents，单位 m
d                  : child mesh 沿 finger 生长方向的 offset，单位 m
```

== q_min / q_max ==

关节上下限属于 morphology，因为它们不随时间步变化，不应被 history stack 重复复制。
但它们的**提取职责**不在本模块：teacher RL 应通过 `observations.py` 从
`asset.data.soft_joint_pos_limits` 读取当前 runtime soft limits。若未来做 limit domain
randomization，policy 看到的正是当下 env 实例使用的真实 clamp 边界。

本模块只记录：JOINT token 的 geometry schema 中存在 `q_min/q_max` 两个槽位，
且它们由 obs MDP 的静态 ObsTerm 提供，不从 hand.yaml 重复解析。

== axis_h0 ==

关节轴不是长度/位置，而是“该关节绕哪个方向旋转”的 signed unit vector。当前选择
将它表达在 hand semantic frame `{h}` 的 home pose 下：

$$
\mathbf{a}^{h}_{i,0} = R^{h}_{j_i}(q=0)\,\mathbf{a}^{j_i}_{i},
\qquad \|\mathbf{a}^{h}_{i,0}\|_2=1.
$$

其中 $\mathbf{a}^{j_i}_{i}$ 是 JointCfg.axis 在本 joint local frame 中的定义，
$R^{h}_{j_i}(q=0)$ 是 home pose 下 joint frame 到 `{h}` 的旋转。这样网络看到的是
“在手的语义坐标系中，这个 joint 是屈伸轴、外展轴，还是拇指特殊相位轴”。

NOTE: generated assets 的 `{h}` / palm frame 由资产建模约定给出；真实 URDF 的
`{a}->{h}` 对齐属于 sim2sim 阶段，不在 teacher 第一版解决。

== l / w / h / d ==

当前第一批资产中 finger links 实际主要用 box collision/visual。虽然 schema 支持
cylinder / sphere / ellipse 等类型，但第一版 teacher 不把 `geom_type` 喂给网络，
而统一使用 canonical extents：

- $l$：沿 finger 生长方向 / 当前 builder 约定主长度方向的长度；
- $w$：局部横向宽度；
- $h$：局部厚度；
- $d$：mesh 相对 joint frame 沿长度方向的 offset。

`asset_gen_cfg.py` 中当前 `link_scale` 会 mutate $l,w,h$，但没有独立 mutate offset $d$。
因此 $d$ 目前更多是 family / preset 常量或局部几何锚点；仍保留是为了让网络知道
collision 皮肤相对 joint frame 是否存在系统性平移。

Question / 后续风险:
    fixed root segment 的 collision 几何当前不属于 JOINT token，也不属于 TIP token。
    对第一批 in-hand manipulation teacher，先让它通过 dynamic SE(3) edge 的 frame
    关系间接进入结构；若后续发现 root-fixed skin 与物体接触显著影响策略，可能需要
    增加 LINK / ROOT_SEGMENT token 或把 root-fixed extents 挂到 palm→root edge feature。
"""


# =============================================================================
# 2. PALM geometry schema
# =============================================================================

r"""TODO(PALM geometry): palm token 的静态几何字段与动态预留字段。

PALM token 是 hand-level anchor，不出动作，不携带 $q/\dot q$。它的职责不是
flatten 可变长度的 mount set；mount / palm→joint / palm→tip 关系已由
all-pairs dynamic SE(3) edge 表达。

当前 PALM 静态 geometry 第一版字段：

```text
palm_extent_x, palm_extent_y, palm_extent_z : palm collision 在 palm/{h} frame 下的 extents，单位 m
```

当前不喂 `palm_geom_type`，因为第一批 generated assets 的 palm 实际只走 box 路线。
若以后启用 cylinder / ellipse / composite palm 且同一 teacher bank 内混合类型，再考虑
加入 geom_type 或 BPS/OBB 等更强描述。

== palm 是否 post-mutate ==

当前 `asset_gen_cfg.py` 未启用 palm shape / palm extents 的 post-mutate。第一版 teacher
可把 palm extents 当作 schema 中的常量字段保留；它在当前 bank 中未必提供训练信息，
但对后续同一模型扩展到 palm perturb 或 Allegro/Leap 混合 teacher 是必要接口。

== gravity axis：动态预留，不属于静态 geometry ==

用户提到的 AnyRotate 风格 gravity axis 不应放进本文件的静态 geometry 提取。它是
运行时手姿态相对重力的动态上下文，更适合未来作为 PALM / global dynamic obs：

$$
\mathbf{g}^{h}_{t} = R^{h}_{w}(t)\,\frac{\mathbf{g}^{w}}{\|\mathbf{g}^{w}\|}
                 = R_{wh}(t)^{\top}\frac{\mathbf{g}^{w}}{\|\mathbf{g}^{w}\|}.
$$

这个量只描述“重力在手语义 frame 中指向哪里”，对绕重力方向的 yaw 旋转天然不敏感：
当手心朝上/朝下不变，只是绕竖直轴转动时，$\mathbf{g}^{h}_{t}$ 可能保持不变或变化很小。
这与用户直觉中的“某些手姿态旋转对手内操作策略等价”一致。

当前固定手姿态 teacher 可先不实现 `gravity_axis_h`；但应在 PALM token 的动态字段中
预留该接口，而不是把它混进本模块的静态 geometry cache。
"""


# =============================================================================
# 3. TIP geometry schema
# =============================================================================

r"""TODO(TIP geometry): fingertip token 的静态 mesh-aware 描述符。

TIP token 是当前几何 obs 中最重要的部分之一，因为第一批 teacher asset bank 已经包含
`tip_replace`：procedural `cs` 与 custom tips（round / wedge / thinner / leap_cube 等）
会在同一 post-mutate bank 中共存。不能只用 `cs_radius/cs_height` 这类 procedural 参数，
否则 custom mesh tip 的真实接触形状会被错误压成旧基线。

当前 TIP geometry 第一版采用：**低维统计 + 小 BPS**。

== 低维统计（mesh low-dimensional descriptor）==

对每个 tip 的实际 collision mesh / collision geometry，在 tip frame 下离线计算并缓存：

```text
tip_extent_x, tip_extent_y, tip_extent_z   # AABB extents，单位 m
tip_centroid_x, tip_centroid_y, tip_centroid_z  # mesh/point-cloud centroid，单位 m
tip_scale                                  # 例如 max(extents) 或等效半径，单位 m
tip_volume / surface_area / covariance_eigs # 可选，视实现成本与 mesh 质量决定
```

`tip_type` / `custom_tip_type` / `mesh_path` 保留为 metadata 和调试字段，不默认作为
policy 的 one-hot 输入。原因是：teacher 第一批 tip 类型有限，直接喂离散类型会让策略
记住当前类型表，而不是从几何形状中学习接触差异。若后续实验发现 BPS 不稳，可把
tip_type one-hot 作为消融项，而不是默认主线。

== 小 BPS（Basis Point Set）==

为覆盖 custom tip 的细粒度形状差异，第一版同时加入一个轻量 BPS descriptor：

1. 在 canonical tip frame 中固定一组 basis points $\{\mathbf{b}_m\}_{m=1}^{M}$；
2. 对实际 tip collision mesh 表面 $\mathcal{S}_{tip}$ 计算最近距离：
   $$
   \mathrm{bps}_m = \min_{\mathbf{x}\in\mathcal{S}_{tip}}\|\mathbf{b}_m-\mathbf{x}\|_2;
   $$
3. 将 $\mathrm{bps}\in\mathbb{R}^{M}$ 与低维统计一起作为 TIP token 的静态几何字段。

建议 $M$ 从 32 或 64 起步。BPS 距离可按 `tip_scale` 归一化，同时把 `tip_scale` 本身
作为低维字段保留，避免丢掉绝对尺寸。所有 tip descriptor 都应离线预计算或在 asset
加载时缓存一次，不能每个控制步重复采样/求最近点。

== 与 TRO-Grasp 的关系 ==

TRO-Grasp 对 robot link 使用 BPS 几何嵌入，对 object patch 使用 VQ-VAE token。
AnyMani 第一版 tip mesh 不建议上 VQ-VAE：tip 类型数量少、mesh 尺度小、teacher PPO
需要高频推理，VQ-VAE 训练和部署成本都偏高。低维统计 + 小 BPS 是更轻的折中。

== 坐标系 ==

所有 tip mesh 特征都应在 tip local frame / contact sensor frame 下计算。这样它与
contact obs 的局部坐标系一致：静态 tip shape 与动态 contact point/force 在同一个
token 内对齐。
"""


# =============================================================================
# 4. 输出组织建议
# =============================================================================

r"""TODO(output organization): geo_obs.py 的输出应按 token 类型组织，而非旧 joint-centric 扁平表。

建议 extractor 最终输出一个结构化 batch，而不是单个大 tensor：

```text
PalmGeo:
    extents_h              [B, N_p=1, 3]

JointGeo:
    limits                 [B, N_j, 2]   # 由 observations.py / runtime obs 提供，此处只引用
    axis_h0                [B, N_j, 3]
    link_lwhd              [B, N_j, 4]

TipGeo:
    lowdim                 [B, N_t, D_low]
    bps                    [B, N_t, M]
```

其中 `B` 可以是 asset batch / env batch，取决于接入 RL 的位置。若同一 asset 在多个
env clone 中复用，应优先在 asset 维缓存，再按 env 的 asset id gather，避免重复存储。

`models/tokenizer.py` 再消费这些结构化字段，分别投影成 PALM / JOINT / TIP token。

TOAGENT:
    实现时优先保持 schema 清楚，不要为了 rl_games 的扁平 obs 接口把所有字段过早
    拼成不可读的一维向量。rl_games adapter 可以在最后一步 flatten / concatenate，
    但本模块的语义输出应保留分组结构。
"""
