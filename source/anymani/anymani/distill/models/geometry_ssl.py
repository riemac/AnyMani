"""Legacy SO(2) model assembly retained because the shared method base declares its configuration type. The published N040 encoder is the proper-SE(3) model."""


from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .decoders.representations.implicit_field import (  # SSL-only readers
    ConditionalDensityDecoder,
    DistanceSensitivityDecoder,
    GeometrySSLDecoderCfg,
)
from .input_adapters.geometry import (
    GeometryEncoderCfg,
    GeometryLatents,
    ImplicitGeometryEncoder,  # task-free hand conditioning encoder
    StaticGeometryEvidence,  # anchors/home/screws/graph/masks
)


@dataclass(frozen=True)
class GeometrySSLModelCfg:


    encoder: GeometryEncoderCfg = field(default_factory=GeometryEncoderCfg)
    ssl_decoders: GeometrySSLDecoderCfg = field(default_factory=GeometrySSLDecoderCfg)  # disposable readers


@dataclass(frozen=True)
class GeometrySSLForward:


    latents: GeometryLatents
    query_features: torch.Tensor
    density: torch.Tensor  # `[B,G,N_Q,N_sigma]`
    kappa: torch.Tensor  # `[B,E]`


class GeometrySSLModel(nn.Module):


    def __init__(self, config: GeometrySSLModelCfg = GeometrySSLModelCfg()) -> None:


        super().__init__()
        self.config = config
        self.encoder = ImplicitGeometryEncoder(config.encoder)
        entity_width = config.encoder.backbone.hidden_width
        query_width = config.encoder.frontend.relation_width
        self.density_decoder = ConditionalDensityDecoder(
            config.ssl_decoders.density,
            entity_width=entity_width,
            query_width=query_width,
        )  # SSL-only FiLM density reader
        self.sensitivity_decoder = DistanceSensitivityDecoder(
            config.ssl_decoders.sensitivity,
            entity_width=entity_width,
            query_width=query_width,
        )  # SSL-only query-main dual-token FiLM sensitivity reader

    def forward(
        self,
        q: torch.Tensor,
        evidence: StaticGeometryEvidence,
        query_points_h: torch.Tensor,
        bandwidths: torch.Tensor,
        owner_index: torch.Tensor,  # `[E]`/`[B,E]`
        query_index: torch.Tensor,  # `[E]`/`[B,E]`
        joint_index: torch.Tensor,  # `[E]`/`[B,E]`
        evidence_row_index: torch.Tensor | None = None,
        joint_coordinate_sign: torch.Tensor | None = None,
    ) -> GeometrySSLForward:


        owner_count = evidence.entity_role.shape[-1]
        if query_points_h.ndim != 4 or query_points_h.shape[:2] != (q.shape[0], owner_count):  # `[B,G]`
            raise ValueError("query_points_h must have shape [B,G,N_Q,3] matching q/evidence")
        latents = self.encoder(
            q,
            evidence,
            evidence_row_index,
            joint_coordinate_sign,
        )
        query_features = self.encoder.encode_points(
            query_points_h.detach(), evidence, evidence_row_index
        )
        entity_valid = evidence.entity_valid_mask
        if entity_valid is not None:
            if evidence_row_index is not None and entity_valid.ndim == 2:
                entity_valid = entity_valid[evidence_row_index]
            if entity_valid.ndim == 1:
                entity_valid = entity_valid.unsqueeze(0).expand(q.shape[0], -1)  # `[B,G]` view
            query_features = query_features * entity_valid.unsqueeze(-1).unsqueeze(-1)
        return self.decode_latents(
            latents,
            query_features,  # `[B,G,N_Q,D_q]`
            bandwidths=bandwidths,
            entity_valid_mask=entity_valid,  # padding owner mask
            joint_entity_index=(
                evidence.joint_entity_index[evidence_row_index]
                if evidence_row_index is not None and evidence.joint_entity_index.ndim == 2
                else evidence.joint_entity_index
            ),
            owner_index=owner_index,  # sampled owner
            query_index=query_index,  # sampled query
            joint_index=joint_index,  # sampled JOINT
        )

    def decode_latents(
        self,
        latents: GeometryLatents,
        query_features: torch.Tensor,
        *,
        bandwidths: torch.Tensor,
        entity_valid_mask: torch.Tensor | None,  # `[B,G]`
        joint_entity_index: torch.Tensor,  # `[N_J]`/`[B,N_J]`
        owner_index: torch.Tensor,  # `[E]`/`[B,E]`
        query_index: torch.Tensor,  # `[E]`/`[B,E]`
        joint_index: torch.Tensor,  # `[E]`/`[B,E]`
    ) -> GeometrySSLForward:


        entities = latents.entities
        density = self.density_decoder(entities, query_features, bandwidths)
        if entity_valid_mask is not None:
            density = density * entity_valid_mask.unsqueeze(-1).unsqueeze(-1)
        if owner_index.ndim not in {1, 2} or query_index.shape != owner_index.shape or joint_index.shape != owner_index.shape:
            raise ValueError("owner/query/joint selectors must share [E] or [B,E] shape")


        def select_entities(index: torch.Tensor) -> torch.Tensor:
            selector = torch.nn.functional.one_hot(index, num_classes=entities.shape[1]).to(dtype=entities.dtype)
            if selector.ndim == 2:
                selector = selector.unsqueeze(0).expand(entities.shape[0], -1, -1)  # `[B,E,G]`
            return torch.bmm(selector, entities)  # `[B,E,D]`

        def select_queries(owner: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
            query_count = query_features.shape[2]
            flat_index = owner * query_count + query
            flat_queries = query_features.flatten(start_dim=1, end_dim=2)  # `[B,G N_Q,D_q]`
            selector = torch.nn.functional.one_hot(flat_index, num_classes=flat_queries.shape[1]).to(
                dtype=query_features.dtype
            )
            if selector.ndim == 2:
                selector = selector.unsqueeze(0).expand(entities.shape[0], -1, -1)  # `[B,E,G N_Q]`
            return torch.bmm(selector, flat_queries)  # `[B,E,D_q]`

        if owner_index.ndim == 1:
            owner_latent = select_entities(owner_index)
            if joint_entity_index.ndim == 1:
                joint_entities = joint_entity_index.index_select(0, joint_index)  # edge JOINT -> entity slot
                joint_latent = select_entities(joint_entities)
            elif joint_entity_index.ndim == 2 and joint_entity_index.shape[0] == entities.shape[0]:
                joint_entities = joint_entity_index[:, joint_index]
                joint_latent = select_entities(joint_entities)
            else:
                raise ValueError("joint_entity_index must have shape [N_J] or [B,N_J]")
            selected_query = select_queries(owner_index, query_index)  # `[B,E,D_q]`
        else:
            if owner_index.shape[0] != entities.shape[0] or joint_entity_index.ndim != 2:
                raise ValueError("batched selectors/routing must share B with entity latents")
            owner_latent = select_entities(owner_index)
            joint_entities = torch.gather(joint_entity_index, 1, joint_index)  # `[B,E]` entity selectors
            joint_latent = select_entities(joint_entities)
            selected_query = select_queries(owner_index, query_index)  # `[B,E,D_q]`
        kappa = self.sensitivity_decoder(owner_latent, joint_latent, selected_query)  # `[B,E]` signed scalar
        return GeometrySSLForward(latents, query_features, density, kappa)

    def retained_state_dict(self) -> dict[str, torch.Tensor]:


        return {
            f"encoder.{key}": value for key, value in self.encoder.state_dict().items()
        }


__all__ = [
    "GeometrySSLForward",  # typed prediction
    "GeometrySSLModel",  # retained+disposable assembly
    "GeometrySSLModelCfg",  # assembly config
]
