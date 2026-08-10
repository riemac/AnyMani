r"""多锚点条件隐式路线的可部署几何编码器。

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

static learned embedding 在 PPO full fine-tune 时不能永久缓存为 stale activation；安全缓存
的是 immutable raw descriptor/geometry，learned output 是否缓存必须服从 optimizer update
lifecycle 与 latency profile。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from ..backbones.geometry_transformer import GraphBiasedTransformer


@dataclass(frozen=True)
class GeometryEncoderConfig:
    r"""Retained geometry encoder 的显式容量与物理归一化配置。"""

    relation_width: int = 64  # 单个 point–anchor relation 的隐藏宽度
    home_width: int = 64  # 每个 owner home surface 聚合后的宽度
    screw_width: int = 64  # 固定宽度 $f_i^{screw}$ 的偶/奇分支宽度
    hidden_width: int = 128  # 整手 Transformer 隐藏宽度
    zero_order_width: int = 128  # $D_0$
    first_order_width: int = 64  # $D_1$
    transformer_layers: int = 3  # Pre-LN entity blocks 数
    attention_heads: int = 4  # 图偏置注意力头数
    feedforward_width: int = 256  # Transformer FFN 宽度
    dropout: float = 0.0  # geometry contract 默认不使用随机 dropout
    length_scale_m: float = 0.1  # 全手共享固定米制数值尺度，不按手型归一化
    max_graph_distance: int = 8  # 图偏置最远桶

    def __post_init__(self) -> None:
        r"""拒绝无法形成合法注意力或物理归一化的配置。"""

        widths = (
            self.relation_width,
            self.home_width,
            self.screw_width,
            self.hidden_width,
            self.zero_order_width,
            self.first_order_width,
            self.feedforward_width,
        )
        if any(width <= 0 for width in widths):
            raise ValueError("all geometry encoder widths must be positive")
        if self.hidden_width % self.attention_heads != 0:
            raise ValueError("hidden_width must be divisible by attention_heads")
        if self.length_scale_m <= 0.0:
            raise ValueError("length_scale_m must be strictly positive")


@dataclass(frozen=True)
class StaticGeometryEvidence:
    r"""同一 asset/结构模式中允许进入 retained encoder 的静态物理证据。

    该类型刻意没有 joint limits、current posed surface、distance、最近点、Jacobian、contact、
    action、history 或 object state。`q_home` 是基准表面与空间旋量的参考构型，不是 limits。
    """

    anchors: torch.Tensor  # `[K,3]`，`{h}`，m；所有 anchors 在网络中等地位
    home_surface_points: torch.Tensor  # `[G,M,3]`，真实 owner union boundary，`{h}`，m
    home_surface_mask: torch.Tensor  # `[G,M]`，允许不同 owner 有不同有效 $M_g$
    palm_normal: torch.Tensor  # `[3]`，有向 $n_p=z_h$
    space_screws: torch.Tensor  # `[N_J,6]`，基准 `{h}` 空间旋量
    q_home: torch.Tensor  # `[N_J]`，rad
    entity_role: torch.Tensor  # `[G]`，0=PALM、1=JOINT、2=TIP
    entity_joint_index: torch.Tensor  # `[G]`，JOINT entity 对应坐标；其他为 -1
    joint_entity_index: torch.Tensor  # `[N_J]`，每个 JOINT 坐标对应 entity index
    shortest_path: torch.Tensor  # `[G,G]`，无向图距离
    parent_direction: torch.Tensor  # `[G,G]`，有向 parent 距离桶
    child_direction: torch.Tensor  # `[G,G]`，有向 child 距离桶

    def __post_init__(self) -> None:
        r"""验证实体、关节、锚点与图轴严格闭合。"""

        if self.anchors.ndim != 2 or self.anchors.shape[1] != 3 or self.anchors.shape[0] == 0:
            raise ValueError("anchors must have non-empty shape [K,3]")
        if self.home_surface_points.ndim != 3 or self.home_surface_points.shape[-1] != 3:
            raise ValueError("home_surface_points must have shape [G,M,3]")
        owner_count, home_count = self.home_surface_points.shape[:2]  # $G$ 与统一存储预算 $M$
        if self.home_surface_mask.shape != (owner_count, home_count):
            raise ValueError("home_surface_mask must have shape [G,M]")
        if self.home_surface_mask.dtype != torch.bool or torch.any(self.home_surface_mask.sum(dim=1) == 0):
            raise ValueError("every owner must have at least one valid home surface point")
        if self.palm_normal.shape != (3,):
            raise ValueError("palm_normal must have shape [3]")
        if not torch.allclose(
            torch.linalg.vector_norm(self.palm_normal),
            torch.ones((), dtype=self.palm_normal.dtype, device=self.palm_normal.device),
            atol=1.0e-6,
            rtol=1.0e-6,
        ):
            raise ValueError("palm_normal must be a unit vector")
        if self.space_screws.ndim != 2 or self.space_screws.shape[1] != 6:
            raise ValueError("space_screws must have shape [N_J,6]")
        joint_count = self.space_screws.shape[0]  # 活动 JOINT 数 $N_J$
        if self.q_home.shape != (joint_count,) or self.joint_entity_index.shape != (joint_count,):
            raise ValueError("q_home and joint_entity_index must have shape [N_J]")
        if self.entity_role.shape != (owner_count,) or self.entity_joint_index.shape != (owner_count,):
            raise ValueError("entity_role and entity_joint_index must have shape [G]")
        graph_shape = (owner_count, owner_count)  # 整手结构关系矩阵共同形状
        if any(
            matrix.shape != graph_shape
            for matrix in (self.shortest_path, self.parent_direction, self.child_direction)
        ):
            raise ValueError("graph relation matrices must have shape [G,G]")
        expected_joint_entities = torch.nonzero(self.entity_role == 1, as_tuple=False).flatten()  # JOINT entity indices
        if not torch.equal(expected_joint_entities, self.joint_entity_index):
            raise ValueError("joint_entity_index must equal the ordered entity_role==JOINT indices")
        if not torch.equal(self.entity_joint_index[self.joint_entity_index], torch.arange(joint_count, device=self.q_home.device)):
            raise ValueError("entity_joint_index and joint_entity_index must be exact inverses on JOINT entities")


@dataclass(frozen=True)
class GeometryLatents:
    r"""部署保留的类型化零阶与逐 JOINT 一阶表征。"""

    zero_order: torch.Tensor  # `[B,G,D_0]`，$SO(2)$ 不变且 joint-sign 偶
    first_order: torch.Tensor  # `[B,N_J,D_1]`，$SO(2)$ 不变且对自身 joint sign 为奇


class SO2AnchorRelationEncoder(nn.Module):
    r"""把物理点相对完整 anchor 星座编码为 origin/$SO(2)$ 不变特征。

    所有 anchors 使用同一个关系投影与同一个 attention score。第 $k$ 个 anchor 只通过其
    物理坐标出现，没有 finger ID 或自由 embedding。对点 $p$，令
    $r_k=p-c_k$、$b_k=c_k-\bar c$；特征使用高度、面内范数、内积和有向叉积：

    $$
    \left[n_p^Tr_k,\|r_{k,\perp}\|,n_p^Tb_k,\|b_{k,\perp}\|,
    r_{k,\perp}^Tb_{k,\perp},n_p^T(r_{k,\perp}\times b_{k,\perp})\right].
    $$

    $b_k$ 为 canonical 文档中未明确的第二关系向量提供 permutation-equivariant 物理定义；
    最后一项在 reflection 下翻号，因此不删除 chirality。
    """

    def __init__(self, relation_width: int, length_scale_m: float) -> None:
        super().__init__()
        self.length_scale_m = float(length_scale_m)  # 全手共享固定米制数值尺度
        self.relation_mlp = nn.Sequential(
            nn.Linear(6, relation_width),
            nn.GELU(),
            nn.Linear(relation_width, relation_width),
            nn.GELU(),
        )
        self.attention_score = nn.Linear(relation_width, 1)  # 每个 anchor 共用的集合权重函数

    def relation_scalars(
        self,
        points: torch.Tensor,
        anchors: torch.Tensor,
        palm_normal: torch.Tensor,
    ) -> torch.Tensor:
        r"""返回每个 point–anchor pair 的六个物理标量，形状 `[...,K,6]`。"""

        relation = points.unsqueeze(-2) - anchors  # $r_k=p-c_k$，形状 `[...,K,3]`，单位 m
        anchor_centered = anchors - anchors.mean(dim=0, keepdim=True)  # $b_k=c_k-\bar c$，origin-independent
        relation_height = torch.sum(relation * palm_normal, dim=-1, keepdim=True)  # $n_p^Tr_k$，m
        anchor_height = torch.sum(anchor_centered * palm_normal, dim=-1, keepdim=True)  # $n_p^Tb_k$，m
        relation_plane = relation - relation_height * palm_normal  # $r_{k,\perp}$，m
        anchor_plane = anchor_centered - anchor_height * palm_normal  # $b_{k,\perp}$，m
        relation_radius = torch.linalg.vector_norm(relation_plane, dim=-1, keepdim=True)  # $\|r_\perp\|$，m
        anchor_radius = torch.linalg.vector_norm(anchor_plane, dim=-1, keepdim=True)  # $\|b_\perp\|$，m
        dot = torch.sum(relation_plane * anchor_plane, dim=-1, keepdim=True)  # $r_\perp^Tb_\perp$，m²
        anchor_plane_broadcast = anchor_plane.expand_as(relation_plane)  # 把 `[K,3]` 广播到全部 point 前导轴
        chirality = torch.sum(
            torch.cross(relation_plane, anchor_plane_broadcast, dim=-1) * palm_normal,
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
    ) -> torch.Tensor:
        r"""对每个 point–anchor pair 使用共享 MLP，保留实际 $K$ 轴。"""

        return self.relation_mlp(self.relation_scalars(points, anchors, palm_normal))  # `[...,K,D_r]`

    def forward(
        self,
        points: torch.Tensor,
        anchors: torch.Tensor,
        palm_normal: torch.Tensor,
    ) -> torch.Tensor:
        r"""沿完整等地位 anchor set 执行可变 $K$ attention pooling。"""

        per_anchor = self.encode_per_anchor(points, anchors, palm_normal)  # `[...,K,D_r]`
        weights = torch.softmax(self.attention_score(per_anchor), dim=-2)  # `[...,K,1]`，permutation-equivariant
        return torch.sum(weights * per_anchor, dim=-2)  # `[...,D_r]`，permutation-invariant


class ImplicitGeometryEncoder(nn.Module):
    r"""从静态手型证据与当前物理 q 输出类型化整手几何表征。"""

    def __init__(self, config: GeometryEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.point_anchor_encoder = SO2AnchorRelationEncoder(config.relation_width, config.length_scale_m)
        self.home_point_projection = nn.Sequential(
            nn.Linear(config.relation_width, config.home_width),
            nn.GELU(),
        )
        self.home_attention_score = nn.Linear(config.home_width, 1)  # 每个 owner 内共享的 surface-point score
        self.screw_even_projection = nn.Linear(config.relation_width, config.screw_width)  # axis-line 偶特征
        self.screw_attention_score = nn.Linear(config.relation_width, 1)  # 沿 anchors 的共享 screw 权重
        self.screw_odd_projection = nn.Linear(3, config.screw_width, bias=False)  # 结构性 joint-sign 奇 carrier
        self.joint_motion_projection = nn.Sequential(
            nn.Linear(1 + 2 * config.screw_width, config.home_width),
            nn.GELU(),
            nn.Linear(config.home_width, config.home_width),
            nn.GELU(),
        )
        role_width = 8  # 三种 owner role 的轻量学习式标识，不编码 finger identity
        self.role_embedding = nn.Embedding(3, role_width)
        entity_input_width = config.home_width + config.home_width + config.screw_width + role_width
        self.entity_projection = nn.Linear(entity_input_width, config.hidden_width)  # 属性拼接到共享 token width
        self.backbone = GraphBiasedTransformer(
            hidden_width=config.hidden_width,
            layers=config.transformer_layers,
            attention_heads=config.attention_heads,
            feedforward_width=config.feedforward_width,
            dropout=config.dropout,
            max_graph_distance=config.max_graph_distance,
        )
        self.zero_order_head = nn.Linear(config.hidden_width, config.zero_order_width)  # $Z^{(0)}$ 投影
        self.first_order_coefficient = nn.Sequential(
            nn.Linear(config.zero_order_width + config.screw_width, config.first_order_width),
            nn.GELU(),
            nn.Linear(config.first_order_width, config.first_order_width),
            nn.Tanh(),
        )  # 只读符号偶上下文的系数
        self.first_order_carrier = nn.Linear(config.screw_width, config.first_order_width, bias=False)  # 奇 carrier 投影

    def encode_points(self, points: torch.Tensor, evidence: StaticGeometryEvidence) -> torch.Tensor:
        r"""公开共享点—锚点前端，供 home points 与训练期 queries 使用同一组参数。"""

        return self.point_anchor_encoder(points, evidence.anchors, evidence.palm_normal)

    def _home_features(self, evidence: StaticGeometryEvidence) -> torch.Tensor:
        r"""先沿 anchors、再沿每个 owner 的真实 surface points 聚合。"""

        point_features = self.encode_points(evidence.home_surface_points, evidence)  # `[G,M,D_r]`
        point_features = self.home_point_projection(point_features)  # `[G,M,D_home]`
        logits = self.home_attention_score(point_features).squeeze(-1)  # `[G,M]` owner 内共享打分
        logits = logits.masked_fill(~evidence.home_surface_mask, torch.finfo(logits.dtype).min)  # 无效 padding 点零权重
        weights = torch.softmax(logits, dim=1)  # 每个 owner 沿自身有效 surface points 归一化
        return torch.sum(weights.unsqueeze(-1) * point_features, dim=1)  # `[G,D_home]`

    def _screw_features(self, evidence: StaticGeometryEvidence) -> tuple[torch.Tensor, torch.Tensor]:
        r"""构造固定宽度的 joint-sign 偶上下文与奇 carrier。

        空间旋量满足 $v=-\omega\times p$；单位轴的最小范数轴上一点为
        $p=\omega\times v$，在 $(\omega,v)\mapsto(-\omega,-v)$ 下保持不变。
        """

        omega = evidence.space_screws[:, :3]  # `[N_J,3]`，joint-sign 奇
        linear = evidence.space_screws[:, 3:]  # `[N_J,3]`，joint-sign 奇，单位 m
        axis_point = torch.cross(omega, linear, dim=-1)  # $p_i=\omega_i\times v_i$，joint-sign 偶，单位 m
        per_anchor_even = self.point_anchor_encoder.encode_per_anchor(
            axis_point, evidence.anchors, evidence.palm_normal
        )  # `[N_J,K,D_r]`，$SO(2)$ 不变且 sign 偶
        weights = torch.softmax(self.screw_attention_score(per_anchor_even), dim=1)  # `[N_J,K,1]`
        even_summary = torch.sum(weights * per_anchor_even, dim=1)  # `[N_J,D_r]`，sign 偶
        even_feature = self.screw_even_projection(even_summary)  # 固定宽度 $f_i^{screw,even}$

        relation = axis_point[:, None, :] - evidence.anchors[None, :, :]  # axis point 到全部 anchors，`[N_J,K,3]`
        relation_height = torch.sum(relation * evidence.palm_normal, dim=-1, keepdim=True)  # `[N_J,K,1]`
        relation_plane = relation - relation_height * evidence.palm_normal  # 轴点关系的面内分量，m
        omega_height = torch.sum(omega * evidence.palm_normal, dim=-1, keepdim=True)  # $n_p^T\omega_i$，sign 奇
        omega_plane = omega - omega_height * evidence.palm_normal  # $\omega_{i,\perp}$，sign 奇
        dot = torch.sum(omega_plane[:, None, :] * relation_plane, dim=-1, keepdim=True)  # sign 奇，单位 m
        cross = torch.sum(
            torch.cross(omega_plane[:, None, :].expand_as(relation_plane), relation_plane, dim=-1)
            * evidence.palm_normal,
            dim=-1,
            keepdim=True,
        )  # $n_p^T(\omega_\perp\times r_\perp)$，sign 奇，单位 m
        odd_basis = torch.cat(
            (
                omega_height[:, None, :].expand(-1, evidence.anchors.shape[0], -1),
                dot / self.config.length_scale_m,
                cross / self.config.length_scale_m,
            ),
            dim=-1,
        )  # `[N_J,K,3]`，无量纲且每一维严格 sign 奇
        odd_summary = torch.sum(weights * odd_basis, dim=1)  # `[N_J,3]`，使用偶权重保持 sign 奇
        odd_feature = self.screw_odd_projection(odd_summary)  # 固定宽度 $f_i^{screw,odd}$，无 bias
        return even_feature, odd_feature

    def forward(self, q: torch.Tensor, evidence: StaticGeometryEvidence) -> GeometryLatents:
        r"""计算部署保留的零阶/一阶几何表征。

        Args:
            q (torch.Tensor): 当前物理关节角，形状 `[B,N_J]`，单位 rad。
            evidence (StaticGeometryEvidence): 当前结构模式的静态物理证据。

        Returns:
            GeometryLatents: `[B,G,D_0]` 零阶与 `[B,N_J,D_1]` 一阶表征。
        """

        joint_count = evidence.space_screws.shape[0]  # 活动 JOINT 数 $N_J$
        owner_count = evidence.entity_role.shape[0]  # 实体/owner 数 $G=N_E$
        if q.ndim != 2 or q.shape[1] != joint_count:
            raise ValueError(f"q must have shape [B,{joint_count}], got {tuple(q.shape)}")
        if q.device != evidence.anchors.device:
            raise ValueError("q and StaticGeometryEvidence tensors must share a device")

        home_feature = self._home_features(evidence)  # `[G,D_home]`，静态真实 collision skin
        screw_even, screw_odd = self._screw_features(evidence)  # `[N_J,D_s]` 偶/奇固定宽度旋量证据
        theta = (q - evidence.q_home) / math.pi  # 相对基准物理角除以 $\pi$，形状 `[B,N_J]`
        signed_motion = theta.unsqueeze(-1) * screw_odd.unsqueeze(0)  # sign×sign=偶，保留运动方向与轴语义
        motion_input = torch.cat(
            (
                theta.square().unsqueeze(-1),
                screw_even.unsqueeze(0).expand(q.shape[0], -1, -1),
                signed_motion,
            ),
            dim=-1,
        )  # `[B,N_J,1+2D_s]`，成对 joint-sign 下严格为偶
        joint_motion_feature = self.joint_motion_projection(motion_input)  # `[B,N_J,D_home]` 偶特征

        batch_size = q.shape[0]  # 同结构模式 microbatch 大小 $B$
        entity_motion = torch.zeros(
            batch_size, owner_count, self.config.home_width, device=q.device, dtype=q.dtype
        )  # PALM/TIP 合法零值
        entity_screw_even = torch.zeros(
            batch_size, owner_count, self.config.screw_width, device=q.device, dtype=q.dtype
        )  # PALM/TIP 无 JOINT screw
        entity_motion[:, evidence.joint_entity_index] = joint_motion_feature  # JOINT entity 对齐当前 q
        entity_screw_even[:, evidence.joint_entity_index] = screw_even.unsqueeze(0)  # JOINT 静态轴线偶证据

        role = self.role_embedding(evidence.entity_role).unsqueeze(0).expand(batch_size, -1, -1)  # `[B,G,8]`
        home = home_feature.unsqueeze(0).expand(batch_size, -1, -1)  # `[B,G,D_home]`，不缓存 learned activation
        entity_input = torch.cat((entity_motion, home, entity_screw_even, role), dim=-1)  # 按属性直接拼接
        tokens = self.entity_projection(entity_input)  # `[B,G,D]`，整手主干输入
        contextual = self.backbone(
            tokens,
            evidence.shortest_path,
            evidence.parent_direction,
            evidence.child_direction,
        )  # `[B,G,D]`，全连接整手上下文
        zero_order = self.zero_order_head(contextual)  # `[B,G,D_0]`，输入结构保证 joint-sign 偶

        joint_zero_order = zero_order.index_select(1, evidence.joint_entity_index)  # `[B,N_J,D_0]`
        even_context = torch.cat(
            (joint_zero_order, screw_even.unsqueeze(0).expand(batch_size, -1, -1)), dim=-1
        )  # 一阶 head 的 sign 偶系数输入
        coefficient = self.first_order_coefficient(even_context)  # `[B,N_J,D_1]`，sign 偶
        carrier = self.first_order_carrier(screw_odd).unsqueeze(0)  # `[1,N_J,D_1]`，sign 奇且无 bias
        first_order = coefficient * carrier  # 偶×奇=奇，结构性满足 $z_i'=s_i z_i$
        return GeometryLatents(zero_order=zero_order, first_order=first_order)


__all__ = [
    "GeometryEncoderConfig",
    "GeometryLatents",
    "ImplicitGeometryEncoder",
    "SO2AnchorRelationEncoder",
    "StaticGeometryEvidence",
]
