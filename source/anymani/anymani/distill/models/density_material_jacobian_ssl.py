r"""Gaussian density + anchor-relational Material-point Jacobian 的 retained/disposable 模型组装。

部署保留路径：

```text
StaticGeometryEvidence + current q
  -> ImplicitGeometryEncoder
  -> unified Z [B,G,D]
```

训练期 disposable 路径：

```text
query point + owner Z + sigma -> density reader -> rho [B,G,N_Q,N_sigma]
fixed home material identity + anchor -> per-anchor static feature
owner Z + selected JOINT Z + static feature -> Gamma reader -> [B,E,K,4]
```

Gamma reader 不读取当前物质点位置、raw Jacobian、ancestor mask 或 target relation；当前 q 的运动学信息只能
经 unified Z 进入预测。训练完成后 density/Gamma readers 均删除，schema-5 artifact 只保留 encoder。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .decoders.representations.implicit_field import (
    ConditionalDensityDecoder,
    ScalarSigmaFiLMDensityDecoderCfg,
)
from .decoders.representations.material_point_jacobian import (
    AnchorRelationalJacobianDecoder,
    AnchorRelationalJacobianDecoderCfg,
)
from .input_adapters.geometry import (
    GeometryEncoderCfg,
    GeometryLatents,
    ImplicitGeometryEncoder,
    StaticGeometryEvidence,
)


@dataclass(frozen=True)
class DensityMaterialJacobianModelCfg:
    r"""Unified encoder 与两个 SSL-only readers 的容量配置。"""

    encoder: GeometryEncoderCfg = field(default_factory=GeometryEncoderCfg)  # retained $Z$ 生成器
    density: ScalarSigmaFiLMDensityDecoderCfg = field(default_factory=ScalarSigmaFiLMDensityDecoderCfg)
    material_jacobian: AnchorRelationalJacobianDecoderCfg = field(default_factory=AnchorRelationalJacobianDecoderCfg)

    def __post_init__(self) -> None:
        r"""验证 readers 的 latent/query width 与 encoder 输出一致。"""

        entity_width = self.encoder.backbone.hidden_width  # unified entity token width $D$
        relation_width = self.encoder.frontend.relation_width  # per-anchor relation width $D_r$
        if self.material_jacobian.latent_width != entity_width:
            raise ValueError("material Jacobian reader latent_width must match encoder hidden_width")
        if self.material_jacobian.relation_width != relation_width:
            raise ValueError("material Jacobian reader relation_width must match encoder relation_width")


@dataclass(frozen=True)
class DensityMaterialJacobianForward:
    r"""一次联合前向的 retained latent 与两个 disposable predictions。"""

    latents: GeometryLatents  # `[B,G,D]`，唯一 retained learned state
    query_features: torch.Tensor  # `[B,G,N_Q,D_r]`，density query condition
    material_pair_features: torch.Tensor  # `[B,E,K,D_r]`，固定 material/anchor query condition
    density: torch.Tensor  # `[B,G,N_Q,N_sigma]`
    material_jacobian: torch.Tensor  # `[B,E,K,4]`，rad$^{-1}$


class DensityMaterialJacobianSSLModel(nn.Module):
    r"""共享 unified Z 的 Gaussian density 与 per-anchor relation-Jacobian 模型。"""

    def __init__(self, config: DensityMaterialJacobianModelCfg = DensityMaterialJacobianModelCfg()) -> None:
        r"""构造一个 retained encoder 与两个完整可删除 readers。"""

        super().__init__()
        self.config = config  # resolved model identity 随 full checkpoint 保存
        self.encoder = ImplicitGeometryEncoder(config.encoder)  # schema-5 retained namespace
        entity_width = config.encoder.backbone.hidden_width  # $D$
        relation_width = config.encoder.frontend.relation_width  # $D_r$
        self.density_decoder = ConditionalDensityDecoder(
            config.density,
            entity_width=entity_width,
            query_width=relation_width,
        )
        self.material_jacobian_decoder = AnchorRelationalJacobianDecoder(config.material_jacobian)

    @staticmethod
    def _route_rows(value: torch.Tensor, row_index: torch.Tensor | None) -> torch.Tensor:
        r"""把按资产去重的 static evidence 路由到每个 q row。"""

        return value[row_index] if row_index is not None else value

    def decode_features(
        self,
        latents: GeometryLatents,
        query_features: torch.Tensor,
        material_pair_features: torch.Tensor,
        bandwidths: torch.Tensor,
        evidence: StaticGeometryEvidence,
        owner_index: torch.Tensor,
        joint_index: torch.Tensor,
        *,
        evidence_row_index: torch.Tensor | None,
        entity_valid_mask: torch.Tensor | None,
    ) -> DensityMaterialJacobianForward:
        r"""从显式 Z 与固定 query features 重放两个 readers，供 matched latent interventions 使用。"""

        entities = latents.entities  # `[B,G,D]`，可为 full/zero/shuffled intervention
        batch_size = entities.shape[0]
        density = self.density_decoder(entities, query_features, bandwidths)  # `[B,G,N_Q,N_sigma]`
        if entity_valid_mask is not None:
            density = density * entity_valid_mask[:, :, None, None]
        batch_axis = torch.arange(batch_size, device=entities.device).unsqueeze(1)  # `[B,1]`
        owner_latent = entities[batch_axis, owner_index]  # `[B,E,D]`
        joint_entity_by_row = self._route_rows(evidence.joint_entity_index, evidence_row_index)
        if joint_entity_by_row.ndim == 1:
            joint_entity_by_row = joint_entity_by_row.unsqueeze(0).expand(batch_size, -1)  # `[B,N_J]`
        selected_joint_entity = joint_entity_by_row[batch_axis, joint_index]  # `[B,E]`
        joint_latent = entities[batch_axis, selected_joint_entity]  # `[B,E,D]`
        material_jacobian = self.material_jacobian_decoder(
            owner_latent,
            joint_latent,
            material_pair_features,
        )  # `[B,E,K,4]`
        return DensityMaterialJacobianForward(
            latents,
            query_features,
            material_pair_features,
            density,
            material_jacobian,
        )

    def forward(
        self,
        q: torch.Tensor,
        evidence: StaticGeometryEvidence,
        query_points_h: torch.Tensor,
        bandwidths: torch.Tensor,
        owner_index: torch.Tensor,
        joint_index: torch.Tensor,
        material_point_index: torch.Tensor,
        *,
        evidence_row_index: torch.Tensor | None = None,
        joint_coordinate_sign: torch.Tensor | None = None,
    ) -> DensityMaterialJacobianForward:
        r"""完成联合 encoder、density query 与 fixed-material per-anchor readout。

        `owner_index/joint_index/material_point_index` 统一使用 `[B,E]`，使跨 morphology padding 和每 q
        独立 material sampling 不依赖隐式广播。Material point index 只索引 static home-surface evidence，
        不读取当前 teacher material position。
        """

        batch_size = q.shape[0]  # `(asset,q)` row 数 $B$
        if owner_index.ndim != 2 or joint_index.shape != owner_index.shape or material_point_index.shape != owner_index.shape:
            raise ValueError("material selectors must share [B,E] shape")
        if owner_index.shape[0] != batch_size:
            raise ValueError("material selector batch axis must match q")
        owner_count = evidence.entity_role.shape[-1]  # padded unified owner count $G$
        if query_points_h.ndim != 4 or query_points_h.shape[:2] != (batch_size, owner_count):
            raise ValueError("query_points_h must have [B,G,N_Q,3] shape matching q/evidence")

        # Unified encoder 是两个任务唯一共享的可部署表示路径。
        latents = self.encoder(
            q,
            evidence,
            evidence_row_index,
            joint_coordinate_sign,
        )  # `[B,G,D]`
        query_features = self.encoder.encode_points(
            query_points_h.detach(),
            evidence,
            evidence_row_index,
        )  # `[B,G,N_Q,D_r]`，query 不含 teacher labels
        entity_valid = evidence.entity_valid_mask  # `[A,G]`/`[G]`/None
        if entity_valid is not None:
            entity_valid = self._route_rows(entity_valid, evidence_row_index)
            if entity_valid.ndim == 1:
                entity_valid = entity_valid.unsqueeze(0).expand(batch_size, -1)
            query_features = query_features * entity_valid[:, :, None, None]
        # 由 static home point identity 与当前 selected anchor bank 形成 per-anchor query；它不读取当前 q target。
        home_by_row = self._route_rows(evidence.home_surface_points, evidence_row_index)
        anchors_by_row = self._route_rows(evidence.anchors, evidence_row_index)
        normal_by_row = self._route_rows(evidence.palm_normal, evidence_row_index)
        if home_by_row.ndim == 3:
            home_by_row = home_by_row.unsqueeze(0).expand(batch_size, -1, -1, -1)  # `[B,G,M,3]`
        if anchors_by_row.ndim == 2:
            anchors_by_row = anchors_by_row.unsqueeze(0).expand(batch_size, -1, -1)  # `[B,K,3]`
        if normal_by_row.ndim == 1:
            normal_by_row = normal_by_row.unsqueeze(0).expand(batch_size, -1)  # `[B,3]`
        anchor_valid = evidence.anchor_valid_mask
        if anchor_valid is None:
            anchor_valid = torch.ones(anchors_by_row.shape[:-1], device=q.device, dtype=torch.bool)
        else:
            anchor_valid = self._route_rows(anchor_valid, evidence_row_index)
            if anchor_valid.ndim == 1:
                anchor_valid = anchor_valid.unsqueeze(0).expand(batch_size, -1)  # `[B,K]`
        batch_axis = torch.arange(batch_size, device=q.device).unsqueeze(1)  # `[B,1]`
        home_points = home_by_row[
            batch_axis,
            owner_index,
            material_point_index,
        ]  # `[B,E,3]`，固定 identity 的 home `{h}` position
        material_pair_features = self.encoder.point_anchor_encoder.encode_per_anchor(
            home_points,
            anchors_by_row,
            normal_by_row,
            anchor_valid,
        )  # `[B,E,K,D_r]`

        return self.decode_features(
            latents,
            query_features,
            material_pair_features,
            bandwidths,
            evidence,
            owner_index,
            joint_index,
            evidence_row_index=evidence_row_index,
            entity_valid_mask=entity_valid,
        )

    def retained_state_dict(self) -> dict[str, torch.Tensor]:
        r"""只发布 `encoder.` namespace，两个 SSL readers 整体删除。"""

        return {f"encoder.{name}": value for name, value in self.encoder.state_dict().items()}


__all__ = [
    "DensityMaterialJacobianForward",
    "DensityMaterialJacobianModelCfg",
    "DensityMaterialJacobianSSLModel",
]
