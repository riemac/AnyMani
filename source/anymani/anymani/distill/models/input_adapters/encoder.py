r"""Geometry SSL retained encoder：SO(2)-aware point/anchor frontend 与 graph-biased trunk。

本模块只拥有可学习的输入编码器，不生成物理 teacher、target、loss 或跨结构 padding。输入是
``StaticGeometryEvidence`` 与当前物理 ``q``；输出是唯一的 ``[B,G,D]`` PALM/JOINT/TIP entity
表征，JOINT consumer 通过静态 routing 从同一 ``Z`` gather。PPO full fine-tune 时，所有依赖参数的
activation 都必须随参数更新重算；只有原始静态 evidence 可以在上游缓存。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from ..backbones.geometry_transformer import GraphBiasedTransformer, GraphBiasedTransformerCfg
from .evidence import StaticGeometryEvidence


@dataclass(frozen=True)
class SO2AnchorFrontendCfg:
    r"""点/旋量—锚点前端、owner 内集合聚合与角色 embedding 容量。"""

    relation_width: int = 64  # 单个 point/screw-anchor relation 的隐藏宽度 $D_r$
    home_width: int = 64  # home surface 与当前 joint-motion property 的聚合宽度
    screw_width: int = 64  # 完整学习式 $f_i^{screw}$ 的固定宽度 $D_s$
    role_width: int = 8  # PALM/JOINT/TIP 三类共享角色 embedding 宽度
    length_scale_m: float = 0.1  # 全手共享 SI 长度数值尺度，单位 m

    def __post_init__(self) -> None:
        r"""拒绝空前端容量或退化的物理数值尺度。"""

        if min(self.relation_width, self.home_width, self.screw_width, self.role_width) < 1:
            raise ValueError("all geometry frontend widths must be positive")
        if self.length_scale_m <= 0.0:
            raise ValueError("length_scale_m must be strictly positive")


@dataclass(frozen=True)
class GeometryEncoderCfg:
    r"""部署保留几何编码器的前端与统一整手主干组合。

    canonical experiment 使用 4 层 graph-biased encoder-only Transformer；全部容量是首个可运行
    锚点，正式选择仍需同时报告留出误差、参数量、激活显存和 retained p95 延迟。
    """

    frontend: SO2AnchorFrontendCfg = SO2AnchorFrontendCfg()  # 点/旋量-anchor 与 owner 内聚合
    backbone: GraphBiasedTransformerCfg = GraphBiasedTransformerCfg()  # 全手上下文容量与图偏置桶


@dataclass(frozen=True)
class GeometryLatents:
    r"""部署保留的单一 PALM/JOINT/TIP typed entity 表征。"""

    entities: torch.Tensor  # `[B,G,D]`，与 PALM/JOINT/TIP owner 同索引的统一 $Z$


class SO2AnchorRelationEncoder(nn.Module):
    r"""把物理点相对完整 anchor 星座编码为 origin/SO(2) 不变特征。

    对点 $p$、anchor $c_k$ 与 anchor center $\bar c$，令 $r_k=p-c_k$、$b_k=c_k-\bar c$；
    输入使用法向高度、面内长度、面内内积与有向叉积。anchor 编号没有语义 embedding，集合
    permutation 只重排内部轴，不改变输出。
    """

    def __init__(self, relation_width: int, length_scale_m: float) -> None:
        r"""构造共享关系投影与锚点集合打分器。"""

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
        anchor_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""返回每个点-anchor 对的六个无量纲物理标量。"""

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
            anchor_centered = anchors - center
            anchors_for_points = anchors
            centered_for_points = anchor_centered
            normal_for_points = palm_normal
        elif anchors.ndim == 3:
            if points.shape[0] != anchors.shape[0]:
                raise ValueError("batched points and anchors must share B")
            singleton_axes = (1,) * (points.ndim - 2)
            center = (anchors * valid_float.unsqueeze(-1)).sum(dim=1, keepdim=True) / valid_float.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0).unsqueeze(-1)
            anchor_centered = anchors - center
            anchors_for_points = anchors.view(anchors.shape[0], *singleton_axes, anchors.shape[1], 3)
            centered_for_points = anchor_centered.view(anchors.shape[0], *singleton_axes, anchors.shape[1], 3)
            if palm_normal.ndim == 1:
                normal_for_points = palm_normal
            else:
                normal_for_points = palm_normal.view(palm_normal.shape[0], *singleton_axes, 1, 3)
        else:
            raise ValueError("anchors must have shape [K,3] or [B,K,3]")

        relation = points.unsqueeze(-2) - anchors_for_points  # $r_k=p-c_k$，`[...,K,3]`，m
        relation_height = torch.sum(relation * normal_for_points, dim=-1, keepdim=True)  # $n_p^Tr_k$
        anchor_height = torch.sum(centered_for_points * normal_for_points, dim=-1, keepdim=True)  # $n_p^Tb_k$
        relation_plane = relation - relation_height * normal_for_points
        anchor_plane = centered_for_points - anchor_height * normal_for_points
        relation_radius = torch.linalg.vector_norm(relation_plane, dim=-1, keepdim=True)
        anchor_radius = torch.linalg.vector_norm(anchor_plane, dim=-1, keepdim=True)
        dot = torch.sum(relation_plane * anchor_plane, dim=-1, keepdim=True)
        anchor_plane_broadcast = anchor_plane.expand_as(relation_plane)
        chirality = torch.sum(
            torch.cross(relation_plane, anchor_plane_broadcast, dim=-1) * normal_for_points,
            dim=-1,
            keepdim=True,
        )

        linear_scale = self.length_scale_m
        quadratic_scale = linear_scale * linear_scale
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
        )

    def encode_per_anchor(
        self,
        points: torch.Tensor,
        anchors: torch.Tensor,
        palm_normal: torch.Tensor,
        anchor_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""对每个 point-anchor 对使用共享 MLP，保留实际 K 轴。"""

        return self.relation_mlp(self.relation_scalars(points, anchors, palm_normal, anchor_valid_mask))

    def forward(
        self,
        points: torch.Tensor,
        anchors: torch.Tensor,
        palm_normal: torch.Tensor,
        anchor_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""沿完整等地位 anchor 集合执行可变 K 注意力池化。"""

        per_anchor = self.encode_per_anchor(points, anchors, palm_normal, anchor_valid_mask)
        logits = self.attention_score(per_anchor)
        if anchor_valid_mask is not None:
            mask = anchor_valid_mask
            while mask.ndim < logits.ndim - 1:
                mask = mask.unsqueeze(-2)
            logits = logits.masked_fill(~mask.unsqueeze(-1), torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-2)
        return torch.sum(weights * per_anchor, dim=-2)


class ImplicitGeometryEncoder(nn.Module):
    r"""从静态手型证据与当前物理 q 输出统一整手 geometry representation。

    当前 posed surface、距离、最近点与解析 Jacobian 只在训练监督侧出现，不允许通过便利特征
    进入本类。完整 screw relation 与 current q 只写入对应 JOINT entity；backbone final-norm
    token 直接是 retained $Z$。
    """

    def __init__(self, config: GeometryEncoderCfg) -> None:
        r"""组装部署保留的 point-anchor frontend 与 graph-biased backbone。"""

        super().__init__()
        self.config = config
        frontend = config.frontend
        backbone = config.backbone
        self.point_anchor_encoder = SO2AnchorRelationEncoder(frontend.relation_width, frontend.length_scale_m)
        self.home_point_projection = nn.Sequential(
            nn.Linear(frontend.relation_width, frontend.home_width),
            nn.GELU(),
        )
        self.home_attention_score = nn.Linear(frontend.home_width, 1)
        self.screw_relation_projection = nn.Sequential(
            nn.Linear(9, frontend.relation_width),
            nn.GELU(),
            nn.Linear(frontend.relation_width, frontend.relation_width),
            nn.GELU(),
        )
        self.screw_attention_score = nn.Linear(frontend.relation_width, 1)
        self.screw_projection = nn.Linear(frontend.relation_width, frontend.screw_width)
        self.joint_motion_projection = nn.Sequential(
            nn.Linear(1 + frontend.screw_width, frontend.home_width),
            nn.GELU(),
            nn.Linear(frontend.home_width, frontend.home_width),
            nn.GELU(),
        )
        self.role_embedding = nn.Embedding(3, frontend.role_width)
        entity_input_width = frontend.home_width * 2 + frontend.screw_width + frontend.role_width
        self.entity_projection = nn.Linear(entity_input_width, backbone.hidden_width)
        self.backbone = GraphBiasedTransformer(backbone)

    def encode_points(
        self,
        points: torch.Tensor,
        evidence: StaticGeometryEvidence,
        evidence_row_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""公开共享 point-anchor frontend，供 home surface 与 query point 共用参数。"""

        if evidence_row_index is not None:
            if evidence.anchors.ndim != 3 or evidence_row_index.shape != (points.shape[0],):
                raise ValueError("query evidence_row_index requires [A,K,3] evidence and shape [B]")
            anchors = evidence.anchors[evidence_row_index]
            palm_normal = evidence.palm_normal[evidence_row_index]
            anchor_valid = (
                evidence.anchor_valid_mask[evidence_row_index]
                if evidence.anchor_valid_mask is not None
                else None
            )
        else:
            anchors = evidence.anchors
            palm_normal = evidence.palm_normal
            anchor_valid = evidence.anchor_valid_mask
        return self.point_anchor_encoder(points, anchors, palm_normal, anchor_valid)

    def _home_features(self, evidence: StaticGeometryEvidence) -> torch.Tensor:
        r"""先沿 anchor、再沿每个 owner 的真实表面点聚合。"""

        point_features = self.encode_points(evidence.home_surface_points, evidence)
        point_features = self.home_point_projection(point_features)
        logits = self.home_attention_score(point_features).squeeze(-1)
        logits = logits.masked_fill(~evidence.home_surface_mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        return torch.sum(weights.unsqueeze(-1) * point_features, dim=-2)

    def _screw_features(self, evidence: StaticGeometryEvidence) -> torch.Tensor:
        r"""构造单一、固定宽度的学习式 $f_i^{screw}$。"""

        omega = evidence.space_screws[..., :3]
        linear = evidence.space_screws[..., 3:]
        axis_point = torch.cross(omega, linear, dim=-1)  # $p_i=\omega_i\times v_i$，m
        axis_point_relations = self.point_anchor_encoder.relation_scalars(
            axis_point, evidence.anchors, evidence.palm_normal, evidence.anchor_valid_mask
        )

        if evidence.anchors.ndim == 2:
            relation = axis_point[:, None, :] - evidence.anchors[None, :, :]
            normal = evidence.palm_normal
        else:
            relation = axis_point.unsqueeze(-2) - evidence.anchors.unsqueeze(1)
            normal = evidence.palm_normal
            if normal.ndim == 2:
                normal = normal[:, None, None, :]
        relation_height = torch.sum(relation * normal, dim=-1, keepdim=True)
        relation_plane = relation - relation_height * normal
        omega_normal = evidence.palm_normal
        if omega.ndim == 3 and omega_normal.ndim == 2:
            omega_normal = omega_normal[:, None, :]
        omega_height = torch.sum(omega * omega_normal, dim=-1, keepdim=True)
        omega_plane = omega - omega_height * omega_normal
        dot = torch.sum(omega_plane.unsqueeze(-2) * relation_plane, dim=-1, keepdim=True)
        cross = torch.sum(
            torch.cross(omega_plane.unsqueeze(-2).expand_as(relation_plane), relation_plane, dim=-1) * normal,
            dim=-1,
            keepdim=True,
        )
        anchor_count = evidence.anchors.shape[-2]
        directed_relations = torch.cat(
            (
                omega_height.unsqueeze(-2).expand(*omega_height.shape[:-2], omega_height.shape[-2], anchor_count, 1),
                dot / self.config.frontend.length_scale_m,
                cross / self.config.frontend.length_scale_m,
            ),
            dim=-1,
        )
        relation_tokens = self.screw_relation_projection(torch.cat((axis_point_relations, directed_relations), dim=-1))
        screw_logits = self.screw_attention_score(relation_tokens)
        if evidence.anchor_valid_mask is not None:
            anchor_mask = evidence.anchor_valid_mask
            while anchor_mask.ndim < screw_logits.ndim - 1:
                anchor_mask = anchor_mask.unsqueeze(-2)
            screw_logits = screw_logits.masked_fill(~anchor_mask.unsqueeze(-1), torch.finfo(screw_logits.dtype).min)
        weights = torch.softmax(screw_logits, dim=-2)
        summary = torch.sum(weights * relation_tokens, dim=-2)
        return self.screw_projection(summary)

    def forward(
        self,
        q: torch.Tensor,
        evidence: StaticGeometryEvidence,
        evidence_row_index: torch.Tensor | None = None,
    ) -> GeometryLatents:
        r"""计算部署保留的统一 PALM/JOINT/TIP entity 表征。"""

        joint_count = evidence.space_screws.shape[-2]
        owner_count = evidence.entity_role.shape[-1]
        if q.ndim != 2 or q.shape[1] != joint_count:
            raise ValueError(f"q must have shape [B,{joint_count}], got {tuple(q.shape)}")
        if q.device != evidence.anchors.device:
            raise ValueError("q and StaticGeometryEvidence tensors must share a device")
        evidence_is_batched = evidence.anchors.ndim == 3
        if evidence_row_index is not None:
            if not evidence_is_batched:
                raise ValueError("evidence_row_index requires batched StaticGeometryEvidence")
            if evidence_row_index.shape != (q.shape[0],) or evidence_row_index.dtype != torch.long:
                raise ValueError("evidence_row_index must have shape [B] and dtype torch.long")
            if torch.any(evidence_row_index < 0) or torch.any(evidence_row_index >= evidence.anchors.shape[0]):
                raise IndexError("evidence_row_index contains a row outside StaticGeometryEvidence")
        elif evidence_is_batched and evidence.anchors.shape[0] != q.shape[0]:
            raise ValueError("batched StaticGeometryEvidence must share B with q unless row routing is provided")

        def route_rows(value: torch.Tensor) -> torch.Tensor:
            return value[evidence_row_index] if evidence_row_index is not None else value

        batch_size = q.shape[0]
        evidence_batch_size = evidence.anchors.shape[0] if evidence_is_batched else batch_size
        entity_valid_source = (
            evidence.entity_valid_mask
            if evidence.entity_valid_mask is not None
            else torch.ones(evidence_batch_size, owner_count, device=q.device, dtype=torch.bool)
        )
        joint_valid_source = (
            evidence.joint_valid_mask
            if evidence.joint_valid_mask is not None
            else torch.ones(evidence_batch_size, joint_count, device=q.device, dtype=torch.bool)
        )
        entity_valid = route_rows(entity_valid_source) if entity_valid_source.ndim == 2 else entity_valid_source
        joint_valid = route_rows(joint_valid_source) if joint_valid_source.ndim == 2 else joint_valid_source
        if entity_valid.ndim == 1:
            entity_valid = entity_valid.unsqueeze(0).expand(batch_size, -1)
        if joint_valid.ndim == 1:
            joint_valid = joint_valid.unsqueeze(0).expand(batch_size, -1)

        home_feature = self._home_features(evidence)
        screw_feature = self._screw_features(evidence)
        q_home = route_rows(evidence.q_home) if evidence.q_home.ndim == 2 else evidence.q_home
        theta = ((q - q_home) / math.pi) * joint_valid
        screw_batch = route_rows(screw_feature) if evidence_is_batched else screw_feature.unsqueeze(0).expand(q.shape[0], -1, -1)
        screw_batch = screw_batch * joint_valid.unsqueeze(-1)
        motion_input = torch.cat((theta.unsqueeze(-1), screw_batch), dim=-1)
        joint_motion_feature = self.joint_motion_projection(motion_input)

        entity_motion = torch.zeros(
            batch_size, owner_count, self.config.frontend.home_width, device=q.device, dtype=q.dtype
        )
        entity_screw = torch.zeros(
            batch_size, owner_count, self.config.frontend.screw_width, device=q.device, dtype=q.dtype
        )
        if evidence.joint_entity_index.ndim == 1:
            entity_motion[:, evidence.joint_entity_index] = joint_motion_feature
            entity_screw[:, evidence.joint_entity_index] = screw_batch
        else:
            routed_joint_entities = route_rows(evidence.joint_entity_index)
            routing = routed_joint_entities.clamp_min(0).unsqueeze(-1).expand(-1, -1, self.config.frontend.home_width)
            entity_motion.scatter_(1, routing, joint_motion_feature * joint_valid.unsqueeze(-1))
            screw_routing = routed_joint_entities.clamp_min(0).unsqueeze(-1).expand(
                -1, -1, self.config.frontend.screw_width
            )
            entity_screw.scatter_(1, screw_routing, screw_batch)

        role_source = self.role_embedding(evidence.entity_role)
        role = route_rows(role_source) if role_source.ndim == 3 else role_source
        if role.ndim == 2:
            role = role.unsqueeze(0).expand(batch_size, -1, -1)
        home = route_rows(home_feature) if evidence_is_batched else home_feature.unsqueeze(0).expand(batch_size, -1, -1)
        entity_input = torch.cat((entity_motion, home, entity_screw, role), dim=-1)
        tokens = self.entity_projection(entity_input) * entity_valid.unsqueeze(-1)
        entities = self.backbone(
            tokens,
            route_rows(evidence.shortest_path) if evidence.shortest_path.ndim == 3 else evidence.shortest_path,
            route_rows(evidence.parent_direction) if evidence.parent_direction.ndim == 3 else evidence.parent_direction,
            route_rows(evidence.child_direction) if evidence.child_direction.ndim == 3 else evidence.child_direction,
            entity_valid,
        )
        return GeometryLatents(entities=entities)


__all__ = [
    "GeometryEncoderCfg",
    "GeometryLatents",
    "ImplicitGeometryEncoder",
    "SO2AnchorFrontendCfg",
    "SO2AnchorRelationEncoder",
]
