r"""多锚点条件隐式路线的可部署运动学—几何编码器。

该模块编码的是 ``current q + ordered screws + topology + home physical geometry``，不是
从 URDF 提取几何，也不是直接计算 target field。static primitive/mesh evidence、semantic
groups 与 cache provenance 来自 ``distill.representations.sources``；当前 soft joint
limits 等 runtime observation 仍由 tasks/adapter 边界提供。

允许的模型侧处理包括：

- 将 meter/radian/screw quantities 按明确物理尺度归一化；
- 把 static per-group geometry descriptor 投影为可训练 embedding；
- 按 asset id gather 已缓存的 raw geometry evidence；
- 将 geometry fields 绑定到 PALM/JOINT/TIP owner，而不 flatten 可变长度 mount set。

可比较的静态输入编码包括 primitive low-dimensional descriptors、surface/BPS descriptors、
offline geometry embeddings 与 no-geometry ablation；它们是 input-side representation
候选，不得与当前 posed BPS/field reconstruction target 混为一谈。generated primitives
与 official irregular mesh 必须通过同一 physical-source/semantic-group contract 对齐，
不能让 mesh path、custom-tip type id 或 asset id 成为默认 shortcut。

本路线禁止读取 BPS/UDF/density target values、current posed surface、最近点、surface
Jacobian 或 current all-pairs dynamic $SE(3)$ answer-like features。解析直接压缩只保留为未来
候选占位：若后续激活，它可由缓存 local support points 与当前 FK/刚体位姿生成 current physical
points，但必须使用独立显式 adapter，不能静默塞进条件隐式 experiment。

编码器实现的映射为：

$$
E_\theta
\left(
q,\ q_{home},\ topology,\ home\_screws,\ home\_surface,\ anchors
\right)
\longrightarrow
\left(Z^{(0)},\{z_i^{(1)}\}_{i=1}^{N_J}\right).
$$

$Z^{(0)}\in\mathbb R^{B\times G\times D_0}$ 描述当前整手逐 owner 零阶碰撞几何，
$z_i^{(1)}\in\mathbb R^{D_1}$ 描述沿物理关节坐标 $q_i$ 的局部运动学—几何响应。
二者都对 hand-frame 面内 $SO(2)$ 坐标重写不变；成对 joint-sign 重写下零阶为偶、一阶为奇。

共享点—锚点前端先沿全部等地位锚点聚合，再沿每个归属体的真实基准表面点聚合；约 $N_E$ 个
PALM/JOINT/TIP 表征单元随后进入图偏置整手 Transformer。不同手指只负责生成锚点采样分布，
手指标识不进入网络。

PPO 完整微调时不能永久缓存依赖可学习参数的静态激活；可以安全缓存的是不可变原始描述与几何。
任何学习式输出能否缓存都必须服从优化器更新周期和延迟实测。

张量轴约定：

```text
q                    : [B, N_J]       # 当前物理关节角，rad
anchors              : [K, 3]         # `{h}` 中的固定物理锚点，m
home_surface_points  : [G, M, 3]      # 归属体真实并集边界样本，m
space_screws         : [N_J, 6]       # 基准 `{h}` 空间旋量
entity tokens        : [B, G, D]       # G=N_E，不使用实体填充
Z^(0)                : [B, G, D_0]     # 零阶表征
Z^(1)                : [B, N_J, D_1]   # 逐 JOINT 一阶表征
```

这里 $B$ 是同一结构模式的微批次大小，$G$ 是 PALM/JOINT/TIP 归属体数，$K$ 是当前资产
实际锚点数，$M$ 是统一存储预算而非所有归属体的真实点数。`home_surface_mask[g,r]` 指示
第 $r$ 个存储位置是否属于归属体 $g$ 的真实表面样本；注意力权重只在有效点上归一化。

点—锚点关系不直接把三维向量交给普通多层感知机，而先形成绕有向手心法向 $n_p$ 的
$SO(2)$ 不变标量。对物理点 $p$、锚点 $c_k$ 与锚点中心 $\bar c$，令：

$$
r_k=p-c_k,
\qquad
b_k=c_k-\bar c.
$$

编码器使用法向高度、面内长度、面内内积与有向叉积。共同平移点与锚点时 $r_k/b_k$
不变；共同绕 $n_p$ 旋转时这些标量不变；镜像时有向叉积翻号，因此左右手性仍可区分。
锚点编号不携带语义，任何锚点排列都只重排集合轴，不改变聚合结果。

关节符号规范通过真实物理输入的成对改写训练与验收。成对改写：

$$
(\mathcal S_i,q_i,q_{home,i})
\longmapsto
(-\mathcal S_i,-q_i,-q_{home,i})
$$

表示同一物理运动。轴线上一点 $p_i=\omega_i\times v_i$、轴向投影和有向叉积共同定义
screw–anchor 关系；它们经同一共享投影和 anchor-set attention 得到一个完整的
$f_i^{screw}$。网络不接收裸 $s_i\in\{-1,+1\}$，也不把 $f_i^{screw}$ 拆成可执行的
even/odd 双支。

当前构型使用带符号的物理坐标：

$$
\theta_i=\frac{q_i-q_{home,i}}{\pi}.
$$

所有 JOINT 共享一个普通 residual head：

$$
z_i^{(1)}=H_1\!\left([z_i^{(0)}\Vert f_i^{screw}]\right).
$$

普通 $H_1$ 不人为硬编码 parity；训练目标对同步改写后的成对 latent 约束
$Z^{(0)}$ 为偶、对应 $z_i^{(1)}$ 为奇。测试仍需检查广播、索引和资产 lowering
是否同步翻转了全部物理量。

关节限位不属于本模块输入。两个仅限位不同、运动学与碰撞几何相同的手，在相同物理 $q$
下必须得到相同几何表征。限位只服务构型采样、边界验证和后续策略局部状态。

数值锚点：canonical 配置使用 $D_0=128$、$D_1=64$、2 层、4 头、随机失活为 0，
固定长度数值尺度为 0.1 m。它们是工程起点，不是实验结论。正式选择必须同时报告留出误差、
参数量、激活显存，以及 RTX 5070 Ti、$B=4096$ 下完整保留路径的 p95 延迟。

NOTE: 该实现不追求镜像不变性。左右镜像、拇指挂载方向和锚点星座手性属于真实形态差异，
必须允许零阶表征发生变化。

NOTE: 基准表面点只表示真实碰撞皮肤，不含实体内部点；实体内部点只允许出现在物理锚点采样中，
两种点集的监督语义不得复用或混写。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from anymani.assets.asset_schema_geometry import HandGeometrySemanticsCfg
from anymani.distill.representations.sources.collision_geometry import AnchorSamples, HomeSurfaceSamples
from anymani.distill.representations.sources.kinematics import EmbodimentGeometrySpec

from ..backbones.geometry_transformer import GraphBiasedTransformer, GraphBiasedTransformerCfg


@dataclass(frozen=True)
class SO2AnchorFrontendCfg:
    r"""点/旋量—锚点前端、owner 内集合聚合与角色 embedding 容量。

    ``length_scale_m=0.1`` 只把米制一次量和二次量变为网络易处理的无量纲数值；它对所有手型
    共用，不按资产尺寸归一化，因此不会删除真实尺度。anchor、home point 与 screw relation 的
    投影权重跨 owner、JOINT 和资产共享。
    """

    relation_width: int = 64  # 单个 point/screw—anchor relation 的隐藏宽度 $D_r$
    home_width: int = 64  # home surface 与当前 joint-motion property 的聚合宽度
    screw_width: int = 64  # 完整学习式 $f_i^{screw}$ 的固定宽度 $D_s$
    role_width: int = 8  # PALM/JOINT/TIP 三类共享角色 embedding 宽度
    length_scale_m: float = 0.1  # 全手共享 SI 长度数值尺度，单位 m

    def __post_init__(self) -> None:
        r"""拒绝空前端容量或退化的物理数值尺度。"""

        if min(self.relation_width, self.home_width, self.screw_width, self.role_width) < 1:
            raise ValueError("all geometry frontend widths must be positive")
        if self.length_scale_m <= 0.0:  # 一次/二次关系分别除以 $L$/$L^2$
            raise ValueError("length_scale_m must be strictly positive")


@dataclass(frozen=True)
class GeometryLatentHeadsCfg:
    r"""零阶 owner 表征与逐 JOINT 一阶 residual 表征的固定类型宽度。"""

    zero_order_width: int = 128  # 每 owner 零阶 latent 宽度 $D_0$
    first_order_width: int = 64  # 每活动 JOINT 一阶 latent 宽度 $D_1$
    first_order_source: str = "residual_screw"  # 当前 $H_1([z_i^{(0)}\Vert f_i^{screw}])$ 路线

    def __post_init__(self) -> None:
        r"""拒绝空 latent，并防止未实现的一阶候选被静默当作 canonical。"""

        if self.zero_order_width < 1 or self.first_order_width < 1:
            raise ValueError("zero/first-order latent widths must be positive")
        if self.first_order_source != "residual_screw":
            raise ValueError("only first_order_source='residual_screw' is currently implemented")


@dataclass(frozen=True)
class GeometryEncoderCfg:
    r"""部署保留几何编码器的前端、整手主干与类型化 heads 组合。

    canonical experiment 明确使用 2 层 graph-biased encoder-only Transformer，而不是依赖底层
    默认值。全部容量仍是首个可运行锚点；正式选择需同时比较留出误差、参数量、激活显存和
    RTX 5070 Ti、$B=4096$ 下 retained path 的 p95 延迟。
    """

    frontend: SO2AnchorFrontendCfg = SO2AnchorFrontendCfg()  # 点/旋量—anchor 与 owner 内聚合
    backbone: GraphBiasedTransformerCfg = GraphBiasedTransformerCfg()  # 全连接图偏置整手上下文
    heads: GeometryLatentHeadsCfg = GeometryLatentHeadsCfg()  # $Z^{(0)}$ 与 $z_i^{(1)}$


@dataclass(frozen=True)
class GeometryPaddingCfg:
    r"""跨结构同批的保守 JOINT/TIP/entity 上限。

    默认最多 5 根手指、每指 4 个活动自由度，因此 $N_J^{max}=20$；每指最多一个 TIP owner，
    所以 $N_E^{max}=1+20+5=26$。上限只服务稠密 batch 容器，不改变模型共享权重或实际输出长度。
    """

    max_joint_count: int = 20  # $5\times4$ 活动 JOINT 上限
    max_tip_count: int = 5  # 每根手指一个 TIP owner
    max_graph_distance: int = 8  # padding 图关系的末桶值

    def __post_init__(self) -> None:
        """拒绝无法容纳 PALM 或没有有效图桶的上限。"""

        if self.max_joint_count < 1 or self.max_tip_count < 1 or self.max_graph_distance < 1:
            raise ValueError("padding joint/tip counts and graph distance must be positive")

    @property
    def max_owner_count(self) -> int:
        """返回 PALM + JOINT + TIP 的最大 entity/owner 数。"""

        return 1 + self.max_joint_count + self.max_tip_count


@dataclass(frozen=True)
class StaticGeometryEvidence:
    r"""同一 asset/结构模式中允许进入 retained encoder 的静态物理证据。

    该类型没有关节限位、当前构型表面、距离、最近点、Jacobian、接触、动作、历史或物体状态。
    `q_home` 是基准表面与空间旋量的参考构型，不是关节限位。
    """

    anchors: torch.Tensor  # `[K,3]` 或 `[B,K,3]`，`{h}`，m；锚点等地位
    home_surface_points: torch.Tensor  # `[G,M,3]` 或 `[B,G,M,3]`，真实 owner boundary
    home_surface_mask: torch.Tensor  # `[G,M]` 或 `[B,G,M]`
    palm_normal: torch.Tensor  # `[3]`，有向 $n_p=z_h$
    space_screws: torch.Tensor  # `[N_J,6]` 或 `[B,N_J,6]`，基准 `{h}` 空间旋量
    q_home: torch.Tensor  # `[N_J]` 或 `[B,N_J]`，rad
    entity_role: torch.Tensor  # `[G]`，0=PALM、1=JOINT、2=TIP
    entity_joint_index: torch.Tensor  # `[G]`，JOINT entity 对应坐标；其他为 -1
    joint_entity_index: torch.Tensor  # `[N_J]`，每个 JOINT 坐标对应实体索引
    shortest_path: torch.Tensor  # `[G,G]`，无向图距离
    parent_direction: torch.Tensor  # `[G,G]`，有向 parent 距离桶
    child_direction: torch.Tensor  # `[G,G]`，有向 child 距离桶
    entity_valid_mask: torch.Tensor | None = None  # `[G]` 或 `[B,G]`，padding owner 为 False
    joint_valid_mask: torch.Tensor | None = None  # `[N_J]` 或 `[B,N_J]`，padding JOINT 为 False
    anchor_valid_mask: torch.Tensor | None = None  # `[K]` 或 `[B,K]`，跨资产 anchor padding 为 False

    def __post_init__(self) -> None:
        r"""验证实体、关节、锚点与图轴严格闭合。

        每个归属体必须至少保留一个真实表面点；JOINT 实体索引与关节坐标索引必须互为逆映射。
        PALM/TIP 可以在关节属性位置使用合法零值，但不能通过错误索引伪装成 JOINT。
        """

        if self.anchors.ndim not in {2, 3} or self.anchors.shape[-1] != 3 or self.anchors.shape[-2] == 0:
            raise ValueError("anchors must have non-empty shape [K,3] or [B,K,3]")
        batched = self.anchors.ndim == 3  # True 表示同结构微批次内每个样本有独立形态证据
        anchor_mask_shape = self.anchors.shape[:-1]
        anchor_valid = (
            self.anchor_valid_mask
            if self.anchor_valid_mask is not None
            else torch.ones(anchor_mask_shape, device=self.anchors.device, dtype=torch.bool)
        )
        if anchor_valid.shape != anchor_mask_shape or anchor_valid.dtype != torch.bool:
            raise ValueError("anchor_valid_mask must have bool shape [K] or [B,K]")
        if torch.any(anchor_valid.sum(dim=-1) == 0):
            raise ValueError("every asset must retain at least one valid physical anchor")
        if self.home_surface_points.ndim != (4 if batched else 3) or self.home_surface_points.shape[-1] != 3:
            raise ValueError("home_surface_points must have shape [G,M,3] or [B,G,M,3]")
        if batched:
            batch_size, owner_count, home_count = self.home_surface_points.shape[:3]
            if self.anchors.shape[0] != batch_size:
                raise ValueError("batched anchors/home_surface_points must share B")
            expected_mask_shape = (batch_size, owner_count, home_count)
        else:
            owner_count, home_count = self.home_surface_points.shape[:2]  # 单资产 $G,M$
            expected_mask_shape = (owner_count, home_count)
        if self.home_surface_mask.shape != expected_mask_shape:
            raise ValueError("home_surface_mask must align with home_surface_points")
        if self.home_surface_mask.dtype != torch.bool:
            raise TypeError("home_surface_mask must use torch.bool")
        if self.palm_normal.shape not in ({(3,), (batch_size, 3)} if batched else {(3,)}):
            raise ValueError("palm_normal must have shape [3] or [B,3]")
        if not torch.allclose(
            torch.linalg.vector_norm(self.palm_normal, dim=-1),
            torch.ones(self.palm_normal.shape[:-1], dtype=self.palm_normal.dtype, device=self.palm_normal.device),
            atol=1.0e-6,
            rtol=1.0e-6,
        ):
            raise ValueError("palm_normal must be a unit vector")
        if self.space_screws.ndim != (3 if batched else 2) or self.space_screws.shape[-1] != 6:
            raise ValueError("space_screws must have shape [N_J,6] or [B,N_J,6]")
        joint_count = self.space_screws.shape[-2]  # 活动 JOINT 数 $N_J$
        expected_q_home_shape = (batch_size, joint_count) if batched else (joint_count,)
        if self.q_home.shape != expected_q_home_shape:
            raise ValueError("q_home must align with space_screws")
        if batched and self.space_screws.shape[0] != batch_size:
            raise ValueError("batched screws/home geometry must share B")
        entity_mask_shape = (batch_size, owner_count) if batched else (owner_count,)
        joint_mask_shape = (batch_size, joint_count) if batched else (joint_count,)
        entity_valid = (
            self.entity_valid_mask
            if self.entity_valid_mask is not None
            else torch.ones(entity_mask_shape, device=self.home_surface_mask.device, dtype=torch.bool)
        )
        joint_valid = (
            self.joint_valid_mask
            if self.joint_valid_mask is not None
            else torch.ones(joint_mask_shape, device=self.q_home.device, dtype=torch.bool)
        )
        if entity_valid.shape != entity_mask_shape or entity_valid.dtype != torch.bool:
            raise ValueError("entity_valid_mask must have bool shape [G] or [B,G]")
        if joint_valid.shape != joint_mask_shape or joint_valid.dtype != torch.bool:
            raise ValueError("joint_valid_mask must have bool shape [N_J] or [B,N_J]")
        if torch.any(self.home_surface_mask.sum(dim=-1)[entity_valid] == 0):
            raise ValueError("every valid owner must have at least one home surface point")

        allowed_entity_shapes = {(owner_count,), (batch_size, owner_count)} if batched else {(owner_count,)}
        if self.entity_role.shape not in allowed_entity_shapes or self.entity_joint_index.shape not in allowed_entity_shapes:
            raise ValueError("entity_role/entity_joint_index must have shape [G] or [B,G]")
        allowed_joint_shapes = {(joint_count,), (batch_size, joint_count)} if batched else {(joint_count,)}
        if self.joint_entity_index.shape not in allowed_joint_shapes:
            raise ValueError("joint_entity_index must have shape [N_J] or [B,N_J]")
        graph_shape = (owner_count, owner_count)  # 未 padding 的共享结构关系
        batched_graph_shape = (batch_size, owner_count, owner_count) if batched else graph_shape
        if any(
            matrix.shape not in {graph_shape, batched_graph_shape}
            for matrix in (self.shortest_path, self.parent_direction, self.child_direction)
        ):
            raise ValueError("graph relation matrices must have shape [G,G] or [B,G,G]")

        check_batch = batch_size if batched else 1
        role_batch = self.entity_role if self.entity_role.ndim == 2 else self.entity_role.unsqueeze(0).expand(check_batch, -1)
        entity_joint_batch = (
            self.entity_joint_index
            if self.entity_joint_index.ndim == 2
            else self.entity_joint_index.unsqueeze(0).expand(check_batch, -1)
        )
        joint_entity_batch = (
            self.joint_entity_index
            if self.joint_entity_index.ndim == 2
            else self.joint_entity_index.unsqueeze(0).expand(check_batch, -1)
        )
        entity_valid_batch = entity_valid if entity_valid.ndim == 2 else entity_valid.unsqueeze(0)
        joint_valid_batch = joint_valid if joint_valid.ndim == 2 else joint_valid.unsqueeze(0)
        for batch_index in range(check_batch):
            expected_joint_entities = torch.where(
                (role_batch[batch_index] == 1) & entity_valid_batch[batch_index]
            )[0]
            valid_joint_slots = torch.where(joint_valid_batch[batch_index])[0]
            mapped_entities = joint_entity_batch[batch_index, valid_joint_slots]
            if not torch.equal(expected_joint_entities, mapped_entities):
                raise ValueError("valid joint_entity_index must equal ordered valid JOINT entities")
            if not torch.equal(entity_joint_batch[batch_index, mapped_entities], valid_joint_slots):
                raise ValueError("entity/joint routing must be exact inverses on valid slots")


@dataclass(frozen=True)
class GeometryLatents:
    r"""部署保留的类型化零阶与逐 JOINT 一阶表征。

    `zero_order[g]` 与物理实体/表面归属体同索引；`first_order[i]` 与活动关节坐标同索引。
    二者不是把一个匿名向量任意切成两半，而是具有不同的规范变换：零阶对成对关节符号为偶，
    一阶对自身关节符号为奇。
    """

    zero_order: torch.Tensor  # `[B,G,D_0]`，$SO(2)$ 不变且 joint-sign 偶
    first_order: torch.Tensor  # `[B,N_J,D_1]`，$SO(2)$ 不变且对自身 joint sign 为奇


class SO2AnchorRelationEncoder(nn.Module):
    r"""把物理点相对完整 anchor 星座编码为 origin/$SO(2)$ 不变特征。

    所有锚点使用同一个关系投影与同一个注意力打分。第 $k$ 个锚点只通过其物理坐标出现，
    没有手指标识或自由嵌入。对点 $p$，令
    $r_k=p-c_k$、$b_k=c_k-\bar c$；特征使用高度、面内范数、内积和有向叉积：

    $$
    \left[n_p^Tr_k,\|r_{k,\perp}\|,n_p^Tb_k,\|b_{k,\perp}\|,
    r_{k,\perp}^Tb_{k,\perp},n_p^T(r_{k,\perp}\times b_{k,\perp})\right].
    $$

    $b_k$ 为第二关系向量提供对锚点排列等变的物理定义；最后一项在镜像下翻号，因此不会
    把真实手性误删为坐标规范。
    """

    def __init__(self, relation_width: int, length_scale_m: float) -> None:
        r"""构造共享关系投影与锚点集合打分器。

        Args:
            relation_width (int): 每个点—锚点对投影后的隐藏宽度 $D_r$。
            length_scale_m (float): 所有手型共享的米制数值尺度，必须大于零。

        关系投影输入固定为六个无量纲标量。注意力打分器对全部锚点共享参数，不持有锚点编号
        嵌入，因此实际锚点数 $K$ 可以随资产变化，输出宽度仍保持 $D_r$。
        """

        super().__init__()
        self.length_scale_m = float(length_scale_m)  # 全手共享固定米制数值尺度
        self.relation_mlp = nn.Sequential(
            nn.Linear(6, relation_width),
            nn.GELU(),
            nn.Linear(relation_width, relation_width),
            nn.GELU(),
        )
        self.attention_score = nn.Linear(relation_width, 1)  # 每个锚点共用的集合权重函数

    def relation_scalars(
        self,
        points: torch.Tensor,
        anchors: torch.Tensor,
        palm_normal: torch.Tensor,
        anchor_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""返回每个点—锚点对的六个物理标量。

        Args:
            points (torch.Tensor): 任意前导轴的物理点 ``[...,3]``，表达于 `{h}`，单位 m。
            anchors (torch.Tensor): 完整锚点集合 ``[K,3]``，表达于 `{h}`，单位 m。
            palm_normal (torch.Tensor): 有向手心法向 ``[3]``，无量纲。

        Returns:
            torch.Tensor: ``[...,K,6]`` 的无量纲 $SO(2)$ 不变标量。
        """

        valid = (
            anchor_valid_mask
            if anchor_valid_mask is not None
            else torch.ones(anchors.shape[:-1], dtype=torch.bool, device=anchors.device)
        )
        if valid.shape != anchors.shape[:-1] or valid.dtype != torch.bool:
            raise ValueError("anchor_valid_mask must align with anchors")
        valid_float = valid.to(dtype=anchors.dtype)
        if anchors.ndim == 2:
            center = (anchors * valid_float[:, None]).sum(dim=0, keepdim=True) / valid_float.sum().clamp_min(1.0)
            anchor_centered = anchors - center  # `[K,3]`
            anchors_for_points = anchors  # 依赖 PyTorch 前导轴广播
            centered_for_points = anchor_centered
            normal_for_points = palm_normal
        elif anchors.ndim == 3:
            if points.shape[0] != anchors.shape[0]:
                raise ValueError("batched points and anchors must share B")
            singleton_axes = (1,) * (points.ndim - 2)  # owner/query 等点轴均由 1 广播
            center = (anchors * valid_float.unsqueeze(-1)).sum(dim=1, keepdim=True) / valid_float.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0).unsqueeze(-1)
            anchor_centered = anchors - center  # `[B,K,3]`
            anchors_for_points = anchors.view(anchors.shape[0], *singleton_axes, anchors.shape[1], 3)
            centered_for_points = anchor_centered.view(
                anchors.shape[0], *singleton_axes, anchors.shape[1], 3
            )
            if palm_normal.ndim == 1:
                normal_for_points = palm_normal  # 所有资产共享 `{h}` 有向 palm normal
            else:
                normal_for_points = palm_normal.view(palm_normal.shape[0], *singleton_axes, 1, 3)
        else:
            raise ValueError("anchors must have shape [K,3] or [B,K,3]")

        relation = points.unsqueeze(-2) - anchors_for_points  # $r_k=p-c_k$，`[...,K,3]`，m
        relation_height = torch.sum(relation * normal_for_points, dim=-1, keepdim=True)  # $n_p^Tr_k$
        anchor_height = torch.sum(centered_for_points * normal_for_points, dim=-1, keepdim=True)  # $n_p^Tb_k$
        relation_plane = relation - relation_height * normal_for_points  # $r_{k,\perp}$，m
        anchor_plane = centered_for_points - anchor_height * normal_for_points  # $b_{k,\perp}$，m
        relation_radius = torch.linalg.vector_norm(relation_plane, dim=-1, keepdim=True)  # $\|r_\perp\|$，m
        anchor_radius = torch.linalg.vector_norm(anchor_plane, dim=-1, keepdim=True)  # $\|b_\perp\|$，m
        dot = torch.sum(relation_plane * anchor_plane, dim=-1, keepdim=True)  # $r_\perp^Tb_\perp$，m²
        anchor_plane_broadcast = anchor_plane.expand_as(relation_plane)  # 锚点集合广播到全部点轴
        chirality = torch.sum(
            torch.cross(relation_plane, anchor_plane_broadcast, dim=-1) * normal_for_points,
            dim=-1,
            keepdim=True,
        )  # $n_p^T(r_\perp\times b_\perp)$，m²；reflection 下翻号

        linear_scale = self.length_scale_m  # 固定全局尺度，只改善数值条件，不按手型消除物理尺度
        quadratic_scale = linear_scale * linear_scale  # 二次标量的对应尺度，单位 m²
        return torch.cat(
            (
                relation_height / linear_scale,
                relation_radius / linear_scale,
                anchor_height.expand_as(relation_height) / linear_scale,
                anchor_radius.expand_as(relation_radius) / linear_scale,
                dot / quadratic_scale,
                chirality / quadratic_scale,
            ),
            dim=-1,
        )  # `[...,K,6]`，全部无量纲且 $SO(2)$ 不变

    def encode_per_anchor(
        self,
        points: torch.Tensor,
        anchors: torch.Tensor,
        palm_normal: torch.Tensor,
        anchor_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""对每个点—锚点对使用共享多层感知机，保留实际 $K$ 轴。"""

        return self.relation_mlp(
            self.relation_scalars(points, anchors, palm_normal, anchor_valid_mask)
        )  # `[...,K,D_r]`

    def forward(
        self,
        points: torch.Tensor,
        anchors: torch.Tensor,
        palm_normal: torch.Tensor,
        anchor_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""沿完整等地位锚点集合执行可变 $K$ 注意力池化。

        权重函数对每个锚点共享，因此锚点重排只会同步重排权重与特征，求和结果保持不变。
        """

        per_anchor = self.encode_per_anchor(points, anchors, palm_normal, anchor_valid_mask)  # `[...,K,D_r]`
        logits = self.attention_score(per_anchor)
        if anchor_valid_mask is not None:
            mask = anchor_valid_mask
            while mask.ndim < logits.ndim - 1:
                mask = mask.unsqueeze(-2)
            logits = logits.masked_fill(~mask.unsqueeze(-1), torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-2)  # `[...,K,1]`，对排列等变
        return torch.sum(weights * per_anchor, dim=-2)  # `[...,D_r]`，对排列不变


class ImplicitGeometryEncoder(nn.Module):
    r"""从静态手型证据与当前物理 $q$ 输出类型化整手几何表征。

    前向只接收部署可见信息。当前表面、距离、最近点与解析 Jacobian 只在训练监督侧出现，
    不允许通过便利特征进入本类。静态原始几何可以缓存，但本类产生的学习式激活在 PPO 微调时
    必须随参数更新重新计算。
    """

    def __init__(self, config: GeometryEncoderCfg) -> None:
        r"""组装部署保留的点—锚点前端、整手主干与零/一阶输出头。

        Args:
            config (GeometryEncoderCfg): 前端、graph-biased backbone 与类型化 heads 配置。

        模块生命周期：本类全部参数随 SSL 检查点迁入 PPO；密度和灵敏度解码器不在本类中，
        因而导出保留检查点时无需根据参数名字猜测哪些层应删除。

        一阶路径复用唯一的 $f_i^{screw}$，不建立第二套逐锚点网络。完整 screw feature
        同时进入 JOINT entity 输入和整手主干后的共享 residual head。
        """

        super().__init__()
        self.config = config
        frontend = config.frontend  # 点/旋量—anchor 与 owner 内聚合容量
        backbone = config.backbone  # 全手上下文容量与图偏置桶
        heads = config.heads  # 零阶/一阶输出类型宽度
        self.point_anchor_encoder = SO2AnchorRelationEncoder(frontend.relation_width, frontend.length_scale_m)
        self.home_point_projection = nn.Sequential(
            nn.Linear(frontend.relation_width, frontend.home_width),
            nn.GELU(),
        )
        self.home_attention_score = nn.Linear(frontend.home_width, 1)  # 每个归属体内共享的表面点打分
        self.screw_relation_projection = nn.Sequential(
            nn.Linear(9, frontend.relation_width),
            nn.GELU(),
            nn.Linear(frontend.relation_width, frontend.relation_width),
            nn.GELU(),
        )  # 每个 screw–anchor pair 共用的九标量学习式投影
        self.screw_attention_score = nn.Linear(frontend.relation_width, 1)  # 沿完整实际 K 轴共享打分
        self.screw_projection = nn.Linear(frontend.relation_width, frontend.screw_width)  # 唯一 $f_i^{screw}$
        self.joint_motion_projection = nn.Sequential(
            nn.Linear(1 + frontend.screw_width, frontend.home_width),
            nn.GELU(),
            nn.Linear(frontend.home_width, frontend.home_width),
            nn.GELU(),
        )
        self.role_embedding = nn.Embedding(3, frontend.role_width)  # 角色不编码 finger identity
        entity_input_width = frontend.home_width * 2 + frontend.screw_width + frontend.role_width
        self.entity_projection = nn.Linear(
            entity_input_width, backbone.hidden_width
        )  # 属性拼接到共享表征宽度
        self.backbone = GraphBiasedTransformer(backbone)
        self.zero_order_head = nn.Linear(backbone.hidden_width, heads.zero_order_width)  # $Z^{(0)}$ 投影
        self.first_order_head = nn.Sequential(
            nn.Linear(heads.zero_order_width + frontend.screw_width, heads.first_order_width),
            nn.GELU(),
            nn.Linear(heads.first_order_width, heads.first_order_width),
        )  # 所有 JOINT 共享的 canonical residual $H_1([z_i^{(0)}\Vert f_i^{screw}])$

    def encode_points(self, points: torch.Tensor, evidence: StaticGeometryEvidence) -> torch.Tensor:
        r"""公开共享点—锚点前端，供基准表面点与训练期查询点使用同一组参数。

        Args:
            points (torch.Tensor): ``[...,3]`` 的 `{h}` 中物理点，单位 m。
            evidence (StaticGeometryEvidence): 提供完整锚点集合与有向手心法向。

        Returns:
            torch.Tensor: ``[...,D_r]`` 的共享关系特征。
        """

        return self.point_anchor_encoder(
            points,
            evidence.anchors,
            evidence.palm_normal,
            evidence.anchor_valid_mask,
        )

    def _home_features(self, evidence: StaticGeometryEvidence) -> torch.Tensor:
        r"""先沿锚点、再沿每个归属体的真实表面点聚合。

        无效存储位置在 softmax 前填入浮点最小值，因此不会获得权重；每个归属体至少有一个有效点
        已由 ``StaticGeometryEvidence`` 验证。
        """

        point_features = self.encode_points(evidence.home_surface_points, evidence)  # `[G,M,D_r]`
        point_features = self.home_point_projection(point_features)  # `[G,M,D_home]`
        logits = self.home_attention_score(point_features).squeeze(-1)  # `[G,M]` 或 `[B,G,M]`
        logits = logits.masked_fill(~evidence.home_surface_mask, torch.finfo(logits.dtype).min)  # 无效存储点零权重
        weights = torch.softmax(logits, dim=-1)  # 每个样本/归属体沿自身有效表面点归一化
        return torch.sum(weights.unsqueeze(-1) * point_features, dim=-2)  # `[G,D_home]` 或 `[B,G,D_home]`

    def _screw_features(self, evidence: StaticGeometryEvidence) -> torch.Tensor:
        r"""构造单一、固定宽度的学习式 $f_i^{screw}$。

        空间旋量满足 $v=-\omega\times p$；单位轴的最小范数轴上一点为
        $p=\omega\times v$。每个 anchor 的九标量关系由轴点的六个 $SO(2)$ 不变量与
        三个保留关节正方向语义的有向标量组成：

        $$
        [\phi(p_i,c_k),\ n_p^T\omega_i,\
        \omega_{i,\perp}^Tr_{ik,\perp}/L,\
        n_p^T(\omega_{i,\perp}\times r_{ik,\perp})/L].
        $$

        九维关系只经过一套共享投影和一套 anchor attention，输出一个 $f_i^{screw}$；
        不把有向量拆成独立网络分支，也不输入裸 sign bit。
        """

        omega = evidence.space_screws[..., :3]  # `[N_J,3]` 或 `[B,N_J,3]`，关节符号奇
        linear = evidence.space_screws[..., 3:]  # 同轴线分量，单位 m
        axis_point = torch.cross(omega, linear, dim=-1)  # $p_i=\omega_i\times v_i$，关节符号偶，单位 m
        axis_point_relations = self.point_anchor_encoder.relation_scalars(
            axis_point, evidence.anchors, evidence.palm_normal, evidence.anchor_valid_mask
        )  # `[...,N_J,K,6]`，轴线位置、绝对尺度与手性关系

        if evidence.anchors.ndim == 2:
            relation = axis_point[:, None, :] - evidence.anchors[None, :, :]  # `[N_J,K,3]`
            normal = evidence.palm_normal
        else:
            relation = axis_point.unsqueeze(-2) - evidence.anchors.unsqueeze(1)  # `[B,N_J,K,3]`
            normal = evidence.palm_normal
            if normal.ndim == 2:
                normal = normal[:, None, None, :]  # `[B,1,1,3]` 广播到 JOINT/anchor
        relation_height = torch.sum(relation * normal, dim=-1, keepdim=True)  # `[...,N_J,K,1]`
        relation_plane = relation - relation_height * normal  # 轴点关系的面内分量，m
        omega_normal = evidence.palm_normal
        if omega.ndim == 3 and omega_normal.ndim == 2:
            omega_normal = omega_normal[:, None, :]  # `[B,1,3]`
        omega_height = torch.sum(omega * omega_normal, dim=-1, keepdim=True)  # $n_p^T\omega_i$，奇
        omega_plane = omega - omega_height * omega_normal  # $\omega_{i,\perp}$，奇
        dot = torch.sum(omega_plane.unsqueeze(-2) * relation_plane, dim=-1, keepdim=True)  # sign 奇，m
        cross = torch.sum(
            torch.cross(omega_plane.unsqueeze(-2).expand_as(relation_plane), relation_plane, dim=-1) * normal,
            dim=-1,
            keepdim=True,
        )  # $n_p^T(\omega_\perp\times r_\perp)$，sign 奇，单位 m
        anchor_count = evidence.anchors.shape[-2]
        directed_relations = torch.cat(
            (
                omega_height.unsqueeze(-2).expand(*omega_height.shape[:-2], omega_height.shape[-2], anchor_count, 1),
                dot / self.config.frontend.length_scale_m,
                cross / self.config.frontend.length_scale_m,
            ),
            dim=-1,
        )  # `[...,N_J,K,3]`，无量纲有向旋量—锚点关系
        relation_tokens = self.screw_relation_projection(
            torch.cat((axis_point_relations, directed_relations), dim=-1)
        )  # `[...,N_J,K,D_r]`，唯一 screw relation token
        screw_logits = self.screw_attention_score(relation_tokens)
        if evidence.anchor_valid_mask is not None:
            anchor_mask = evidence.anchor_valid_mask
            while anchor_mask.ndim < screw_logits.ndim - 1:
                anchor_mask = anchor_mask.unsqueeze(-2)
            screw_logits = screw_logits.masked_fill(~anchor_mask.unsqueeze(-1), torch.finfo(screw_logits.dtype).min)
        weights = torch.softmax(screw_logits, dim=-2)  # 沿有效 K 轴
        summary = torch.sum(weights * relation_tokens, dim=-2)  # `[...,N_J,D_r]`，anchor permutation 不变
        return self.screw_projection(summary)  # `[...,N_J,D_s]`，单一 $f_i^{screw}$

    def forward(self, q: torch.Tensor, evidence: StaticGeometryEvidence) -> GeometryLatents:
        r"""计算部署保留的零阶/一阶几何表征。

        Args:
            q (torch.Tensor): 当前物理关节角，形状 `[B,N_J]`，单位 rad。
            evidence (StaticGeometryEvidence): 当前结构模式的静态物理证据。

        Returns:
            GeometryLatents: `[B,G,D_0]` 零阶与 `[B,N_J,D_1]` 一阶表征。
        """

        joint_count = evidence.space_screws.shape[-2]  # 活动 JOINT 数 $N_J$
        owner_count = evidence.entity_role.shape[-1]  # 实体/归属体数 $G=N_E$
        if q.ndim != 2 or q.shape[1] != joint_count:
            raise ValueError(f"q must have shape [B,{joint_count}], got {tuple(q.shape)}")
        if q.device != evidence.anchors.device:
            raise ValueError("q and StaticGeometryEvidence tensors must share a device")
        evidence_is_batched = evidence.anchors.ndim == 3
        if evidence_is_batched and evidence.anchors.shape[0] != q.shape[0]:
            raise ValueError("batched StaticGeometryEvidence must share B with q")

        batch_size = q.shape[0]  # 同一次 padding/同结构微批次大小 $B$
        entity_valid = (
            evidence.entity_valid_mask
            if evidence.entity_valid_mask is not None
            else torch.ones(batch_size, owner_count, device=q.device, dtype=torch.bool)
        )
        joint_valid = (
            evidence.joint_valid_mask
            if evidence.joint_valid_mask is not None
            else torch.ones(batch_size, joint_count, device=q.device, dtype=torch.bool)
        )
        if entity_valid.ndim == 1:
            entity_valid = entity_valid.unsqueeze(0).expand(batch_size, -1)
        if joint_valid.ndim == 1:
            joint_valid = joint_valid.unsqueeze(0).expand(batch_size, -1)

        home_feature = self._home_features(evidence)  # `[G,D_home]` 或 `[B,G,D_home]`
        screw_feature = self._screw_features(evidence)  # `[...,N_J,D_s]` 完整 $f_i^{screw}$
        theta = ((q - evidence.q_home) / math.pi) * joint_valid  # padding JOINT 的运动严格为零
        screw_batch = (
            screw_feature if evidence_is_batched else screw_feature.unsqueeze(0).expand(q.shape[0], -1, -1)
        )
        screw_batch = screw_batch * joint_valid.unsqueeze(-1)  # padding screw 不携带形态证据
        motion_input = torch.cat(
            (
                theta.unsqueeze(-1),
                screw_batch,
            ),
            dim=-1,
        )  # `[B,N_J,1+D_s]`，带符号当前构型与完整旋量语义
        joint_motion_feature = self.joint_motion_projection(motion_input)  # `[B,N_J,D_home]`

        entity_motion = torch.zeros(
            batch_size, owner_count, self.config.frontend.home_width, device=q.device, dtype=q.dtype
        )  # PALM/TIP 合法零值
        entity_screw = torch.zeros(
            batch_size, owner_count, self.config.frontend.screw_width, device=q.device, dtype=q.dtype
        )  # PALM/TIP 无 JOINT 旋量
        if evidence.joint_entity_index.ndim == 1:
            entity_motion[:, evidence.joint_entity_index] = joint_motion_feature  # 同结构共享 routing
            entity_screw[:, evidence.joint_entity_index] = screw_batch
        else:
            routing = evidence.joint_entity_index.clamp_min(0).unsqueeze(-1).expand(
                -1, -1, self.config.frontend.home_width
            )
            entity_motion.scatter_(1, routing, joint_motion_feature * joint_valid.unsqueeze(-1))
            screw_routing = evidence.joint_entity_index.clamp_min(0).unsqueeze(-1).expand(
                -1, -1, self.config.frontend.screw_width
            )
            entity_screw.scatter_(1, screw_routing, screw_batch)

        role = self.role_embedding(evidence.entity_role)
        if role.ndim == 2:
            role = role.unsqueeze(0).expand(batch_size, -1, -1)  # 同结构共享角色轴
        home = home_feature if evidence_is_batched else home_feature.unsqueeze(0).expand(batch_size, -1, -1)
        entity_input = torch.cat((entity_motion, home, entity_screw, role), dim=-1)  # 按属性直接拼接
        tokens = self.entity_projection(entity_input) * entity_valid.unsqueeze(-1)  # padding token 输入严格为零
        contextual = self.backbone(
            tokens,
            evidence.shortest_path,
            evidence.parent_direction,
            evidence.child_direction,
            entity_valid,
        )  # `[B,G,D]`，全连接整手上下文
        zero_order = self.zero_order_head(contextual) * entity_valid.unsqueeze(-1)  # padding owner 输出严格为零

        if evidence.joint_entity_index.ndim == 1:
            joint_zero_order = zero_order.index_select(1, evidence.joint_entity_index)  # 同结构共享 routing
        else:
            gather_routing = evidence.joint_entity_index.clamp_min(0).unsqueeze(-1).expand(
                -1, -1, self.config.heads.zero_order_width
            )
            joint_zero_order = torch.gather(zero_order, 1, gather_routing)
        first_order_input = torch.cat(
            (joint_zero_order, screw_batch), dim=-1
        )  # `[B,N_J,D_0+D_s]`，所有 JOINT 共享同一输入类型
        first_order = self.first_order_head(first_order_input) * joint_valid.unsqueeze(-1)  # canonical residual
        return GeometryLatents(zero_order=zero_order, first_order=first_order)


def build_static_geometry_evidence(
    semantics: HandGeometrySemanticsCfg,
    spec: EmbodimentGeometrySpec,
    home_surface: HomeSurfaceSamples,
    anchors: AnchorSamples,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> StaticGeometryEvidence:
    r"""把 assets/robots 静态真值转换为 retained encoder 的类型化输入。

    Args:
        semantics (HandGeometrySemanticsCfg): owner、joint、role 与静态 frame 事实。
        spec (EmbodimentGeometrySpec): representations source lower 的 screw、home pose 与图关系。
        home_surface (HomeSurfaceSamples): owner-local boundary points，shape `[G,M,3]`，m。
        anchors (AnchorSamples): 已变换到 `{h}` 的完整 anchor realization，shape `[K,3]`，m。
        device (torch.device | str): encoder 输入设备。
        dtype (torch.dtype): encoder 输入浮点 dtype。

    Returns:
        StaticGeometryEvidence: 不含 limits、current distance、最近点、Jacobian 或 target label 的
        retained input package。

    home surface points 从 owner-local 经过 home $T_{hg}(q_{home})$ 变换到 `{h}`；这一步只在
    静态 materialization 中执行。`joint_entity_index` 与 `entity_joint_index` 由 owner/joint
    sidecar 生成互逆映射，确保 decoder owner 轴和一阶 JOINT 轴不会由训练脚本重新猜测。
    """

    if len(semantics.owners) != spec.owner_home_transforms.shape[0]:
        raise ValueError("semantics/spec owner axes must match")
    if home_surface.points_owner_local_m.shape[0] != len(semantics.owners):
        raise ValueError("home surface owner axis must match semantics")
    if spec.owner_graph_shortest is None or spec.owner_graph_parent is None or spec.owner_graph_child is None:
        raise ValueError("robots spec must provide all owner graph relations")
    target_device = torch.device(device)
    owner_home = spec.owner_home_transforms.to(device=target_device, dtype=dtype)
    local_home = torch.as_tensor(home_surface.points_owner_local_m, device=target_device, dtype=dtype)
    home_points = torch.einsum(
        "gij,gmj->gmi",
        owner_home[:, :3, :3],
        local_home,
    ) + owner_home[:, None, :3, 3]  # `[G,M,3]`，严格回到 `{h}`
    anchor_points = torch.as_tensor(anchors.anchors_hand_m, device=target_device, dtype=dtype)
    role_index = {"palm": 0, "joint": 1, "tip": 2}
    entity_role = torch.tensor(
        [role_index[owner.role] for owner in semantics.owners], device=target_device, dtype=torch.long
    )
    joint_index_by_name = {name: index for index, name in enumerate(spec.joint_names)}
    entity_joint_index = torch.full(
        (len(semantics.owners),), -1, device=target_device, dtype=torch.long
    )  # PALM/TIP 使用合法零值
    for owner in semantics.owners:
        if owner.role == "joint":
            if owner.joint_name not in joint_index_by_name:
                raise ValueError(f"owner '{owner.owner_id}' joint is missing from spec joint axis")
            entity_joint_index[owner.owner_index] = joint_index_by_name[owner.joint_name]
    joint_entity_index = torch.tensor(
        [owner.owner_index for owner in semantics.owners if owner.role == "joint"],
        device=target_device,
        dtype=torch.long,
    )
    if tuple(entity_joint_index[joint_entity_index].tolist()) != tuple(range(len(spec.joint_names))):
        raise ValueError("owner JOINT axis is not the exact inverse of joint axis")
    return StaticGeometryEvidence(
        anchors=anchor_points,
        home_surface_points=home_points,
        home_surface_mask=torch.ones(home_points.shape[:2], device=target_device, dtype=torch.bool),
        palm_normal=torch.tensor((0.0, 0.0, 1.0), device=target_device, dtype=dtype),
        space_screws=spec.space_screws.to(device=target_device, dtype=dtype),
        q_home=spec.q_home.to(device=target_device, dtype=dtype),
        entity_role=entity_role,
        entity_joint_index=entity_joint_index,
        joint_entity_index=joint_entity_index,
        shortest_path=spec.owner_graph_shortest.to(device=target_device),
        parent_direction=spec.owner_graph_parent.to(device=target_device),
        child_direction=spec.owner_graph_child.to(device=target_device),
        anchor_valid_mask=torch.ones(anchor_points.shape[:-1], device=target_device, dtype=torch.bool),
    )


def stack_static_geometry_evidence(
    evidences: Sequence[StaticGeometryEvidence],
) -> StaticGeometryEvidence:
    r"""把同一结构的多只资产堆成一个无 entity padding 的形态批次。

    Args:
        evidences (Sequence[StaticGeometryEvidence]): 每项是单资产未批处理的静态证据；要求相同
            owner/JOINT 数、role/routing 与图关系，但 anchors、home surface、screws 和 q_home 可不同。

    Returns:
        StaticGeometryEvidence: variable morphology 字段新增 B 轴，结构字段继续共享。

    Raises:
        ValueError: 输入为空、已有 B 轴、点预算不同或结构关系不一致时抛出。不同结构应分开前向，
        不能用 padding 掩盖。
    """

    if not evidences:
        raise ValueError("at least one StaticGeometryEvidence is required")
    if any(evidence.anchors.ndim != 2 for evidence in evidences):
        raise ValueError("stack_static_geometry_evidence expects unbatched asset evidence")
    reference = evidences[0]  # 同结构共享的 owner/JOINT routing 和图关系真值
    shared_fields = (
        "entity_role",
        "entity_joint_index",
        "joint_entity_index",
        "shortest_path",
        "parent_direction",
        "child_direction",
    )
    for evidence_index, evidence in enumerate(evidences[1:], start=1):
        for field_name in shared_fields:
            if not torch.equal(getattr(reference, field_name), getattr(evidence, field_name)):
                raise ValueError(
                    f"evidence[{evidence_index}] field '{field_name}' differs; split into another structure minibatch"
                )
        if evidence.anchors.shape != reference.anchors.shape:
            raise ValueError("same minibatch requires one configured anchor count K")
        if evidence.home_surface_points.shape != reference.home_surface_points.shape:
            raise ValueError("same minibatch requires one configured home point budget M")
        if evidence.space_screws.shape != reference.space_screws.shape:
            raise ValueError("same minibatch requires one active JOINT axis length")

    return StaticGeometryEvidence(
        anchors=torch.stack([evidence.anchors for evidence in evidences], dim=0),
        home_surface_points=torch.stack([evidence.home_surface_points for evidence in evidences], dim=0),
        home_surface_mask=torch.stack([evidence.home_surface_mask for evidence in evidences], dim=0),
        palm_normal=torch.stack([evidence.palm_normal for evidence in evidences], dim=0),
        space_screws=torch.stack([evidence.space_screws for evidence in evidences], dim=0),
        q_home=torch.stack([evidence.q_home for evidence in evidences], dim=0),
        entity_role=reference.entity_role,
        entity_joint_index=reference.entity_joint_index,
        joint_entity_index=reference.joint_entity_index,
        shortest_path=reference.shortest_path,
        parent_direction=reference.parent_direction,
        child_direction=reference.child_direction,
        anchor_valid_mask=torch.stack(
            [
                evidence.anchor_valid_mask
                if evidence.anchor_valid_mask is not None
                else torch.ones(evidence.anchors.shape[0], device=evidence.anchors.device, dtype=torch.bool)
                for evidence in evidences
            ],
            dim=0,
        ),
    )


def pad_static_geometry_evidence(
    evidences: Sequence[StaticGeometryEvidence],
    *,
    config: GeometryPaddingCfg = GeometryPaddingCfg(),
) -> StaticGeometryEvidence:
    r"""把不同 owner/JOINT 长度的资产填充为统一 20-JOINT/26-entity batch。

    只有张量容器被填充；有效 owner/JOINT 的原始顺序、图距离和物理证据保持不变。padding 区域
    anchor/home/screw/q 均为零，routing 为 -1，attention/loss 由显式 mask 屏蔽。
    """

    if not evidences:
        raise ValueError("at least one StaticGeometryEvidence is required")
    if any(evidence.anchors.ndim != 2 for evidence in evidences):
        raise ValueError("padding expects one unbatched StaticGeometryEvidence per asset")
    reference = evidences[0]
    max_anchor_count = max(evidence.anchors.shape[0] for evidence in evidences)
    home_budget = reference.home_surface_points.shape[1]
    for evidence in evidences:
        if evidence.home_surface_points.shape[1] != home_budget:
            raise ValueError("padded batch requires one configured home point budget M")

    batch_size = len(evidences)
    max_owner_count = config.max_owner_count
    max_joint_count = config.max_joint_count
    device = reference.anchors.device
    dtype = reference.anchors.dtype
    anchors = torch.zeros(batch_size, max_anchor_count, 3, device=device, dtype=dtype)
    anchor_valid = torch.zeros(batch_size, max_anchor_count, device=device, dtype=torch.bool)
    home_points = torch.zeros(batch_size, max_owner_count, home_budget, 3, device=device, dtype=dtype)
    home_mask = torch.zeros(batch_size, max_owner_count, home_budget, device=device, dtype=torch.bool)
    screws = torch.zeros(batch_size, max_joint_count, 6, device=device, dtype=dtype)
    q_home = torch.zeros(batch_size, max_joint_count, device=device, dtype=dtype)
    entity_role = torch.zeros(batch_size, max_owner_count, device=device, dtype=torch.long)
    entity_joint_index = torch.full(
        (batch_size, max_owner_count), -1, device=device, dtype=torch.long
    )
    joint_entity_index = torch.full(
        (batch_size, max_joint_count), -1, device=device, dtype=torch.long
    )
    entity_valid = torch.zeros(batch_size, max_owner_count, device=device, dtype=torch.bool)
    joint_valid = torch.zeros(batch_size, max_joint_count, device=device, dtype=torch.bool)
    graph_shape = (batch_size, max_owner_count, max_owner_count)
    shortest_path = torch.full(graph_shape, config.max_graph_distance, device=device, dtype=torch.long)
    parent_direction = torch.full_like(shortest_path, config.max_graph_distance)
    child_direction = torch.full_like(shortest_path, config.max_graph_distance)

    for batch_index, evidence in enumerate(evidences):
        owner_count = evidence.home_surface_points.shape[0]
        joint_count = evidence.space_screws.shape[0]
        anchor_count = evidence.anchors.shape[0]
        tip_count = int(torch.count_nonzero(evidence.entity_role == 2))
        if joint_count > max_joint_count:
            raise ValueError(f"asset[{batch_index}] has {joint_count} joints, exceeds {max_joint_count}")
        if tip_count > config.max_tip_count:
            raise ValueError(f"asset[{batch_index}] has {tip_count} TIP owners, exceeds {config.max_tip_count}")
        if owner_count > max_owner_count:
            raise ValueError(f"asset[{batch_index}] has {owner_count} owners, exceeds {max_owner_count}")

        anchors[batch_index, :anchor_count] = evidence.anchors
        source_anchor_mask = evidence.anchor_valid_mask
        anchor_valid[batch_index, :anchor_count] = (
            source_anchor_mask
            if source_anchor_mask is not None
            else torch.ones(anchor_count, device=device, dtype=torch.bool)
        )
        home_points[batch_index, :owner_count] = evidence.home_surface_points
        home_mask[batch_index, :owner_count] = evidence.home_surface_mask
        screws[batch_index, :joint_count] = evidence.space_screws
        q_home[batch_index, :joint_count] = evidence.q_home
        entity_role[batch_index, :owner_count] = evidence.entity_role
        entity_joint_index[batch_index, :owner_count] = evidence.entity_joint_index
        joint_entity_index[batch_index, :joint_count] = evidence.joint_entity_index
        entity_valid[batch_index, :owner_count] = True
        joint_valid[batch_index, :joint_count] = True
        shortest_path[batch_index, :owner_count, :owner_count] = evidence.shortest_path
        parent_direction[batch_index, :owner_count, :owner_count] = evidence.parent_direction
        child_direction[batch_index, :owner_count, :owner_count] = evidence.child_direction

    return StaticGeometryEvidence(
        anchors=anchors,
        home_surface_points=home_points,
        home_surface_mask=home_mask,
        palm_normal=torch.stack([evidence.palm_normal for evidence in evidences], dim=0),
        space_screws=screws,
        q_home=q_home,
        entity_role=entity_role,
        entity_joint_index=entity_joint_index,
        joint_entity_index=joint_entity_index,
        shortest_path=shortest_path,
        parent_direction=parent_direction,
        child_direction=child_direction,
        entity_valid_mask=entity_valid,
        joint_valid_mask=joint_valid,
        anchor_valid_mask=anchor_valid,
    )


__all__ = [
    "GeometryEncoderCfg",
    "GeometryLatentHeadsCfg",
    "GeometryPaddingCfg",
    "GeometryLatents",
    "ImplicitGeometryEncoder",
    "SO2AnchorRelationEncoder",
    "SO2AnchorFrontendCfg",
    "StaticGeometryEvidence",
    "build_static_geometry_evidence",
    "pad_static_geometry_evidence",
    "stack_static_geometry_evidence",
]
