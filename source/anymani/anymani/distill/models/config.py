r"""AnyMani policy model 的集中式配置契约。

本文件是 `distill/models` 的配置单一事实源，服务于科研核对而非软件封装。
所有会影响网络结构、token 语义、mask 路径、attention bias、输出头与几何
adapter 的关键开关，都应优先在这里声明，再由 `tokenizer.py`、`backbone.py`、
`heads.py`、`geometry/mesh_adapter.py` 等模块消费。

== 为什么集中在一个文件 ==

用户会逐行核对网络结构和实验假设。若 tokenizer / backbone / heads 各自散落
dataclass，容易出现：

1. 结构开关分散，难以判断一次实验到底启用了哪些机制；
2. teacher 与 student 复用同一模型时，配置项被复制出两个版本；
3. 后续写 Hydra / YAML / rl_games adapter 时，不清楚哪个字段是唯一入口。

因此本文件只做一件事：把当前网络设计中的**可裁定配置面**集中写清楚。
具体 `nn.Module` 实现暂不在本轮落地。

== 形状与符号约定 ==

- $B$：batch size。
- $D$：投影后的统一 token 隐空间维度。
- $N_p$：palm token 数，当前通常为 1。
- $N_j$：revolute joint token 数。
- $N_t$：tip token 数，即手指数。
- $T=N_p+N_j+N_t$：聚合后的总 token 数。

TOAGENT:
    本文件是脚手架阶段的配置契约。字段可以增补，但不要随意删除已有字段和
    其注释语义。若某字段实验后被废弃，应先标记为 deprecated / rejected，
    待用户确认后再清理。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TypeProjectionCfg:
    r"""单一 token 类型的 projection 配置。

    每类 token 先用自己的 projection $P_\tau$ 从原始物理特征空间映射到统一
    隐空间：

    $$
    h_i^{(0)} = P_{\tau_i}(x_i) \in \mathbb{R}^{D}.
    $$

    这里不默认三类共享 projection，因为 palm / joint / tip 的原始特征语义
    不同：joint 有 $q,\dot q$ 和 action history，tip 有接触与指尖几何，palm
    更像全局 anchor。共享一个原始 Linear 会把不同物理量强行塞进同一解释器。
    """

    input_dim: int | None = None
    """该类型原始特征维度；`None` 表示该类型的具体字段尚未冻结。"""

    hidden_dim: int | None = None
    """轻量 MLP 中间维度；`None` 时默认等于全局 `embed_dim`。"""

    activation: str = "gelu"
    """projection 内部非线性，候选 `gelu` / `elu` / `relu`；具体实现待定。"""

    use_layer_norm: bool = True
    """是否在 projection 内对异构物理量做 LayerNorm，以缓解量纲尺度差异。"""


@dataclass(slots=True)
class TokenizerCfg:
    r"""palm / joint / tip 分组 tokenizer 配置。

    tokenizer 的职责不是从 URDF 解析特征，而是消费已经整理好的 token-ready
    张量，并把三类异构输入投影到统一 $D$ 维 token 空间。
    """

    embed_dim: int = 128
    """统一 token 隐空间维度 $D$；teacher 起步建议 128，后续可做 192/256 消融。"""

    palm: TypeProjectionCfg = field(default_factory=TypeProjectionCfg)
    """`PALM` token projection 配置；当前通常 $N_p=1$。"""

    joint: TypeProjectionCfg = field(default_factory=TypeProjectionCfg)
    """`JOINT` token projection 配置；唯一出动作的 token 类型。"""

    tip: TypeProjectionCfg = field(default_factory=TypeProjectionCfg)
    """`TIP` token projection 配置；承载指尖几何与接触，不出动作。"""

    use_type_embedding: bool = True
    """是否加入 token type embedding $e_{\tau_i}$，标记 palm / joint / tip 角色。"""

    concat_order: tuple[str, str, str] = ("palm", "joint", "tip")
    """投影后 token 轴拼接顺序；输出路由按该顺序切片，默认与 `TokenType` 顺序一致。"""


@dataclass(slots=True)
class RelationFeatureCfg:
    r"""节点间关系 / edge feature 的配置契约。

    本配置把 mount pose 从 palm / root joint 的节点特征中剥离出来，明确归入
    edge feature。核心原因是：mount 是 `palm -> finger/root` 的关系量，既不是
    palm 的固定长度单体属性，也不是 root joint 的内禀状态。

    需要特别区分两类 edge feature，但 teacher 当前阶段更偏向直接使用动态 SE(3)：

    1. **static embodiment edge**：只由 URDF / HandCfg / generator 资产决定，
       在 episode 内不随 $q_t$ 变化。例如 palm→root mount、相邻 joint 在 home pose
       或 URDF parent-child frame 下的相对位姿、joint→tip offset。它描述形态。
    2. **dynamic kinematic edge**：由当前关节状态 $q_t$ 经 FK 得到，随时间变化。
       例如当前姿态下任意两 link/joint frame 的相对 SE(3)。它描述姿态与几何的
       当前组合，表达力强。teacher 不承担 sim2sim，因此可以先使用它；student /
       真实 URDF 部署阶段再处理 frame 语义对齐问题。

    例如第 $i$ 根手指的根挂载可写为：

    $$
    E_{p\to r_i} = T_{\text{palm}}^{-1}T_{r_i},
    $$

    它天然绑定一条 `PALM -> root JOINT` 边。把它放在边上可以同时保留：
    ① 可变手指数；② mount 与 root joint 的 binding；③ `JOINT` token 的同构性。
    """

    mount_feature_location: str = "edge"
    """mount pose 的归属，当前裁定为 `edge`；不推荐 `palm_flatten` 或 `root_joint_field`。"""

    static_edge_mode: str = "adjacent_rest_se3"
    """静态边特征模式；作为 dynamic edge 的形态补充，不用纯 hop count 做 teacher 主信号。"""

    dynamic_edge_mode: str = "current_all_pairs_se3"
    """动态边特征模式；teacher 当前合意为 `current_all_pairs_se3`，即所有 token pair 的当前 FK 相对位姿。"""

    include_palm_to_root_pose: bool = True
    """是否把 palm frame 到每个 root joint frame 的相对位姿作为 edge feature。"""

    include_joint_to_joint_pose: bool = True
    """是否把运动学链上 parent joint 到 child joint 的相对位姿作为 edge feature。"""

    include_joint_to_tip_pose: bool = True
    """是否把 distal joint 到 fingertip frame 的相对位姿作为 edge feature。"""

    include_hop_distance: bool = False
    """是否加入 get-zero 式最短路径跳数；teacher 同拓扑阶段默认 False，避免无信息常量。"""

    include_edge_type: bool = True
    """是否加入离散 edge type（palm-joint、joint-tip、same-finger 等），作为 SE(3) 的语义标签。"""

    use_discrete_handedness: bool = False
    """是否把 left/right 作为离散输入；默认 False，chirality 由 mount layout 连续关系诱导。"""


@dataclass(slots=True)
class AttentionBiasCfg:
    r"""attention logits 中 $b_{ij}$ 的可插拔 bias 配置。

    标准 self-attention 的 logits 为：

    $$
    a_{ij}^{(h)} = \frac{q_i^{(h)\top} k_j^{(h)}}{\sqrt{d_h}} + b_{ij}^{(h)}.
    $$

    teacher 第一版默认使用 `mode="hybrid_se3"`，即把结构先验与连续几何边特征
    相加注入 logits：

    $$
    b_{ij}^{(h)} =
      \beta_{\phi(i,j)}^{(h)}
      + \gamma_{d(i,j)}^{(h)}
      + \delta_{\mathrm{same\_finger}(i,j)}^{(h)}
      + f_{\theta}^{(h)}\!\left(\tilde E_{ij}^{t}, m_{ij}\right),
    $$

    其中 $\phi(i,j)$ 是有向 edge type，$d(i,j)$ 是运动学图最短路径 bucket，
    `same_finger` 是同指链二值关系，$\tilde E_{ij}^{t}$ 是归一化后的 dynamic
    all-pairs SE(3) 相对位姿，$m_{ij}$ 是可选结构 metadata。这样既保留
    Graphormer 风格的离散拓扑 inductive bias，又不丢 post-mutate mount / link
    / tip 几何变化带来的连续差异。

    消融矩阵固定为：

    - `none`：$b_{ij}=0$，检验 Transformer 本体是否已足够；
    - `structural`：只用 edge type / distance / same-finger；
    - `se3`：只用 dynamic SE(3) edge MLP；
    - `hybrid_se3`：结构 bias + SE(3) edge MLP，teacher 默认。

    PPO 稳定性约束：SE(3) MLP 最后一层建议零初始化，或用 learnable gate
    $\alpha\approx0$ 起步，使初期网络近似 no-bias，避免随机 $b_{ij}$ 直接打爆
    softmax logits。
    """

    mode: str = "hybrid_se3"
    """bias 模式：`none` / `structural` / `se3` / `hybrid_se3`；teacher 默认 `hybrid_se3`。"""

    num_edge_types: int | None = None
    """离散有向 edge type 数；`structural` / `hybrid_se3` 模式需要。"""

    num_distance_buckets: int = 6
    """运动学图距离 bucket 数，建议对应 `0,1,2,3,4,>=5`；只作结构 bias，不作硬 mask。"""

    edge_feature_dim: int | None = None
    """连续边特征维度；`se3` / `hybrid_se3` 模式需要，第一版至少含 normalized $se(3)$ 6D。"""

    se3_bias_hidden_dim: int | None = None
    """SE(3) edge MLP hidden dim；`None` 时建议实现默认取 token `embed_dim // 2`。"""

    se3_translation_scale: str = "palm_extent"
    """平移归一化锚点：`palm_extent` / `hand_radius` / `fixed_meter`；避免米制平移尺度主导旋量。"""

    se3_rotation_scale: float = 3.141592653589793
    r"""旋转 log 向量归一化尺度，默认除以 $\pi$，使 $so(3)$ 量大致落入 $[-1,1]$。"""

    use_same_finger_bias: bool = True
    """是否加入同指链二值结构 bias，帮助模型区分 intra-finger coordination 与 inter-finger coordination。"""

    use_zero_init_bias_mlp: bool = True
    """是否要求 SE(3) bias MLP 最后一层零初始化；PPO 初期稳定性优先。"""

    use_bias_gate: bool = True
    """是否使用 learnable gate $\alpha$ 缩放连续 bias；建议 $\alpha$ 初始化接近 0。"""

    relation_features: RelationFeatureCfg = field(default_factory=RelationFeatureCfg)
    """边特征的语义归属配置，尤其是 palm→root mount pose 的位置。"""


@dataclass(slots=True)
class BackboneCfg:
    r"""Encoder-only self-attention 主干配置。

    主干只处理投影后的同维 token 序列 $H\in\mathbb{R}^{B\times T\times D}$，
    不再知道原始输入属于 palm/joint/tip 的不同维度；类型信息已通过
    type embedding 或 token 特征进入 $H$。
    """

    num_layers: int = 4
    """Transformer Encoder 层数；teacher 起步建议 4，复杂消融可到 6。"""

    num_heads: int = 4
    """多头注意力 head 数；需整除 `TokenizerCfg.embed_dim`。"""

    mlp_ratio: float = 2.0
    """FFN hidden dim 与 embed dim 比值；RL 起步用 2.0–3.0 避免过重。"""

    dropout: float = 0.0
    """dropout 概率；PPO 在线 RL 初期常设 0，蒸馏阶段可再启用。"""

    pre_norm: bool = True
    """是否使用 Pre-LN；RL 训练中通常比 Post-LN 稳定。"""

    attention_bias: AttentionBiasCfg = field(default_factory=AttentionBiasCfg)
    """attention bias 配置；teacher 默认启用 SE(3) edge feature bias，no-bias 仅作消融。"""


@dataclass(slots=True)
class HeadsCfg:
    r"""输出头配置：actor / critic / auxiliary。

    输出端按 token 类型路由：只有 `JOINT` token 进入 actor head；`TIP` 可选接
    FK / contact auxiliary head；`PALM` 当前没有明确 aux 价值。
    """

    action_dim_per_joint: int = 1
    """每个 revolute joint 输出动作维度；当前 raw relative delta action 为 1。"""

    action_log_std_mode: str = "global_action_dim"
    """log_std 参数化：`global_scalar` / `global_action_dim` / `none`；初期不做 token-wise。"""

    value_pool: str = "mean"
    """critic pooling：`mean` / `attention` / `palm` / `hand_cls`，当前不裁定最终方案。"""

    enable_tip_aux: bool = False
    """是否启用 tip auxiliary head；可用于 fingertip FK / contact 自监督，初期可关闭。"""

    tip_aux_target: str | None = None
    """tip aux 目标类型，如 `fk_pos_root` / `fk_pose_root` / `contact_state`，待实验裁定。"""


@dataclass(slots=True)
class GeometryAdapterCfg:
    r"""mesh / link 几何编码器配置。

    几何 adapter 处理的是“原始几何张量 → token 可消费特征/embedding”，不是
    从 URDF 提取几何。URDF / HandCfg / asset metadata 的解析属于 `rl/geo_obs.py`。
    """

    mode: str = "primitive_low_dim"
    """几何编码模式：`primitive_low_dim` / `bps` / `offline_embedding` / `none`。"""

    primitive_feature_dim: int | None = None
    """box/cylinder/sphere 等低维 primitive 几何特征维度，字段尚未冻结。"""

    bps_points: int | None = None
    """BPS 基点数量；仅 `bps` 模式需要，借鉴 tro-grasp 但不默认启用。"""

    output_dim: int | None = None
    """几何 adapter 输出维度；可等于 token 原始特征的一部分或直接对齐 `embed_dim`。"""


@dataclass(slots=True)
class EmbodimentPolicyCfg:
    r"""完整 embodiment policy 配置。

    该配置装配 tokenizer → geometry adapter → backbone → heads 的所有结构面。
    teacher 与 student 尽量共用此配置 schema：teacher 是固定拓扑退化情形，
    student 才真正启用可变 token 数与 mask。
    """

    tokenizer: TokenizerCfg = field(default_factory=TokenizerCfg)
    """分组 tokenizer 配置。"""

    geometry: GeometryAdapterCfg = field(default_factory=GeometryAdapterCfg)
    """几何 adapter 配置；只负责可学习/可替换编码，不负责资产解析。"""

    backbone: BackboneCfg = field(default_factory=BackboneCfg)
    """Encoder-only 主干配置。"""

    heads: HeadsCfg = field(default_factory=HeadsCfg)
    """actor / critic / aux 输出头配置。"""

    mask_convention: str = "valid_true"
    """模型内部 mask 约定：`True=有效 token`；进 PyTorch attention 前再显式转换。"""
