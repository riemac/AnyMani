r"""Proper-SE(3)-invariant point/anchor 与 screw-line/anchor geometry encoder。

Legacy point relation 已基于 $p-a_k$、anchor-centered layout 与共同 palm normal 构造 invariant scalars。
本模块替换 origin-dependent screw axis point relation。对 screw line $(\omega,v)$，先取一个轴上代表点
$p_0=\omega\times v$，再对每个 anchor $a_k$ 投影掉轴向 gauge：

$$
r_{ik}=p_0-a_k,\qquad
r_{ik}^{\perp}=r_{ik}-(r_{ik}^{T}\omega_i)\omega_i.
$$

$r_{ik}^{\perp}$ 与物理 axis line 唯一对应；改变 `{h}` origin 或沿轴改选 $p_0$ 都不改变它。后续只使用
$r^{\perp}$、$\omega$、palm normal 和 anchor layout 的长度、点积与有向叉积，因此输入在 proper SE(3)
共同重写下保持不变，同时保留 reflection chirality 与 joint-sign directed features。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..backbones.geometry_transformer import GraphBiasedTransformerCfg
from .encoder import GeometryEncoderCfg, ImplicitGeometryEncoder, SO2AnchorFrontendCfg
from .evidence import StaticGeometryEvidence


@dataclass(frozen=True)
class SE3InvariantAnchorFrontendCfg:
    r"""SE(3)-invariant point/line-anchor frontend 宽度与共同米制尺度。"""

    relation_width: int = 64
    home_width: int = 64
    screw_width: int = 64
    role_width: int = 8
    length_scale_m: float = 0.1

    def __post_init__(self) -> None:
        r"""拒绝空容量或非正米制尺度。"""

        if min(self.relation_width, self.home_width, self.screw_width, self.role_width) < 1:
            raise ValueError("SE3-invariant frontend widths must be positive")
        if self.length_scale_m <= 0.0:
            raise ValueError("SE3-invariant frontend length_scale_m must be positive")


@dataclass(frozen=True)
class SE3InvariantGeometryEncoderCfg:
    r"""独立 N040 frontend 与共享 graph-biased backbone 配置。"""

    frontend: SE3InvariantAnchorFrontendCfg = field(default_factory=SE3InvariantAnchorFrontendCfg)
    backbone: GraphBiasedTransformerCfg = field(default_factory=GraphBiasedTransformerCfg)


class SE3InvariantGeometryEncoder(ImplicitGeometryEncoder):
    r"""以 line-anchor invariant scalars 替换 legacy axis-point screw path 的统一 encoder。"""

    def __init__(self, config: SE3InvariantGeometryEncoderCfg = SE3InvariantGeometryEncoderCfg()) -> None:
        r"""复用稳定 point frontend/backbone 参数布局，并冻结独立 SE3 config identity。"""

        legacy_frontend = SO2AnchorFrontendCfg(
            relation_width=config.frontend.relation_width,
            home_width=config.frontend.home_width,
            screw_width=config.frontend.screw_width,
            role_width=config.frontend.role_width,
            length_scale_m=config.frontend.length_scale_m,
        )
        super().__init__(GeometryEncoderCfg(frontend=legacy_frontend, backbone=config.backbone))
        self.se3_config = config

    def screw_line_relation_scalars(
        self,
        space_screws: torch.Tensor,
        anchors: torch.Tensor,
        palm_normal: torch.Tensor,
        anchor_valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""返回每个 screw-line/anchor pair 的六个 invariant scalars 与垂直关系向量。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: `[...,N_J,K,6]` scalars 与 `[...,N_J,K,3]` $r^\perp$。
        """

        omega = space_screws[..., :3]  # `[...,N_J,3]`，单位轴
        linear = space_screws[..., 3:]  # `[...,N_J,3]`，m
        axis_point = torch.cross(omega, linear, dim=-1)  # origin-dependent $p_0$，只作中间代表
        if anchors.ndim == 2:
            valid = (
                anchor_valid_mask
                if anchor_valid_mask is not None
                else torch.ones(anchors.shape[0], device=anchors.device, dtype=torch.bool)
            )
            center = (anchors * valid[:, None].to(anchors.dtype)).sum(dim=0) / valid.sum().clamp_min(1)
            anchor_centered = anchors - center
            relation = axis_point[:, None, :] - anchors[None, :, :]
            omega_pair = omega[:, None, :]
            normal = palm_normal
        elif anchors.ndim == 3:
            valid = (
                anchor_valid_mask
                if anchor_valid_mask is not None
                else torch.ones(anchors.shape[:2], device=anchors.device, dtype=torch.bool)
            )
            weight = valid.to(anchors.dtype)
            center = (anchors * weight.unsqueeze(-1)).sum(dim=1) / weight.sum(dim=1, keepdim=True).clamp_min(1.0)
            anchor_centered = anchors - center.unsqueeze(1)
            relation = axis_point.unsqueeze(-2) - anchors.unsqueeze(1)
            omega_pair = omega.unsqueeze(-2)
            normal = palm_normal[:, None, None, :]
        else:
            raise ValueError("anchors must have shape [K,3] or [B,K,3]")

        # 投影掉 axis direction，删除 origin translation 和 point-on-line gauge。
        axial = torch.sum(relation * omega_pair, dim=-1, keepdim=True)
        perpendicular = relation - axial * omega_pair  # $r^\perp$，m
        relation_height = torch.sum(perpendicular * normal, dim=-1, keepdim=True)
        relation_plane = perpendicular - relation_height * normal
        anchor_normal = palm_normal if anchors.ndim == 2 else palm_normal[:, None, :]
        anchor_height = torch.sum(anchor_centered * anchor_normal, dim=-1, keepdim=True)
        anchor_plane = anchor_centered - anchor_height * anchor_normal
        if anchors.ndim == 3:
            anchor_height = anchor_height.unsqueeze(1)
            anchor_plane_pair = anchor_plane.unsqueeze(1)
        else:
            anchor_height = anchor_height.unsqueeze(0)
            anchor_plane_pair = anchor_plane.unsqueeze(0)
        relation_radius = torch.linalg.vector_norm(relation_plane, dim=-1, keepdim=True)
        anchor_radius = torch.linalg.vector_norm(anchor_plane_pair, dim=-1, keepdim=True)
        dot = torch.sum(relation_plane * anchor_plane_pair, dim=-1, keepdim=True)
        chirality = torch.sum(
            torch.cross(relation_plane, anchor_plane_pair.expand_as(relation_plane), dim=-1) * normal,
            dim=-1,
            keepdim=True,
        )
        scale = self.se3_config.frontend.length_scale_m
        scalars = torch.cat(
            (
                relation_height / scale,
                relation_radius / scale,
                anchor_height.expand_as(relation_height) / scale,
                anchor_radius.expand_as(relation_radius) / scale,
                dot / (scale * scale),
                chirality / (scale * scale),
            ),
            dim=-1,
        )
        return scalars, perpendicular

    def _screw_features(
        self,
        evidence: StaticGeometryEvidence,
        evidence_row_index: torch.Tensor | None = None,
        joint_coordinate_sign: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""构造 proper-SE(3)-invariant、joint-sign-aware 的学习式 screw-line feature。"""

        anchors = evidence.anchors
        normal = evidence.palm_normal
        anchor_valid = evidence.anchor_valid_mask
        screws = evidence.space_screws
        if joint_coordinate_sign is not None:
            if evidence_row_index is not None:
                anchors = anchors[evidence_row_index]
                normal = normal[evidence_row_index]
                anchor_valid = anchor_valid[evidence_row_index] if anchor_valid is not None else None
                screws = screws[evidence_row_index]
            screws = screws * joint_coordinate_sign.unsqueeze(-1)
        scalars, perpendicular = self.screw_line_relation_scalars(screws, anchors, normal, anchor_valid)
        omega = screws[..., :3]
        if anchors.ndim == 2:
            omega_normal = normal
            relation_normal = normal
        else:
            omega_normal = normal[:, None, :]
            relation_normal = normal[:, None, None, :]
        omega_height = torch.sum(omega * omega_normal, dim=-1, keepdim=True)
        omega_plane = omega - omega_height * omega_normal
        relation_height = torch.sum(perpendicular * relation_normal, dim=-1, keepdim=True)
        relation_plane = perpendicular - relation_height * relation_normal
        dot = torch.sum(omega_plane.unsqueeze(-2) * relation_plane, dim=-1, keepdim=True)
        cross = torch.sum(
            torch.cross(omega_plane.unsqueeze(-2).expand_as(relation_plane), relation_plane, dim=-1)
            * relation_normal,
            dim=-1,
            keepdim=True,
        )
        anchor_count = anchors.shape[-2]
        directed = torch.cat(
            (
                omega_height.unsqueeze(-2).expand(*omega_height.shape[:-2], omega_height.shape[-2], anchor_count, 1),
                dot / self.se3_config.frontend.length_scale_m,
                cross / self.se3_config.frontend.length_scale_m,
            ),
            dim=-1,
        )
        tokens = self.screw_relation_projection(torch.cat((scalars, directed), dim=-1))
        logits = self.screw_attention_score(tokens)
        if anchor_valid is not None:
            mask = anchor_valid
            while mask.ndim < logits.ndim - 1:
                mask = mask.unsqueeze(-2)
            logits = logits.masked_fill(~mask.unsqueeze(-1), torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-2)
        return self.screw_projection(torch.sum(weights * tokens, dim=-2))

    def screw_features(
        self,
        evidence: StaticGeometryEvidence,
        evidence_row_index: torch.Tensor | None = None,
        joint_coordinate_sign: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""公开 f_screw audit surface，不改变 forward 的统一 Z 合同。"""

        return self._screw_features(evidence, evidence_row_index, joint_coordinate_sign)


__all__ = [
    "SE3InvariantAnchorFrontendCfg",
    "SE3InvariantGeometryEncoder",
    "SE3InvariantGeometryEncoderCfg",
]
