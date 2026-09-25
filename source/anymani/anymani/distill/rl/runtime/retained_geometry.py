'Provide frozen N040 owner tokens from ordered static evidence and current joint angles.'

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from anymani.distill.methods.density_material_jacobian.artifact import SE3RetainedEncoderArtifact
from anymani.distill.models.input_adapters.evidence import StaticGeometryEvidence
from anymani.distill.rl.canonical_evidence import (
    CANONICAL_JOINT_COUNT,
    CANONICAL_OWNER_COUNT,
    CanonicalEvidenceBank,
)


def _stable_digest(payload: Mapping[str, Any]) -> str:
    'Hash a JSON-safe identity with canonical key ordering.'

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _evidence_digest(evidence: StaticGeometryEvidence) -> str:
    'Hash each static evidence tensor by field name, dtype, shape, and bytes.'

    digest = hashlib.sha256()
    for name in (
        "anchors",
        "home_surface_points",
        "home_surface_mask",
        "palm_normal",
        "space_screws",
        "q_home",
        "entity_role",
        "entity_joint_index",
        "joint_entity_index",
        "shortest_path",
        "parent_direction",
        "child_direction",
        "entity_valid_mask",
        "joint_valid_mask",
        "anchor_valid_mask",
    ):
        value = getattr(evidence, name)
        if value is None:
            digest.update(f"{name}:none".encode())
            continue
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class RetainedGeometryBatch:
    'Per-step N040 tokens, masks, and graph relations for one policy batch.'

    geometry_entities: torch.Tensor
    'Final-normalized owner tokens on the [B,21,128] PALM/JOINT/TIP axis.'

    owner_valid_mask: torch.Tensor
    'Boolean owner mask on the [B,21] PALM/JOINT/TIP axis.'

    joint_valid_mask: torch.Tensor
    'Boolean active-joint mask on the [B,16] canonical axis.'

    shortest_path: torch.Tensor
    'Undirected owner graph distances on [B,21,21].'

    parent_direction: torch.Tensor
    'Parent-direction buckets on [B,21,21].'

    child_direction: torch.Tensor
    'Child-direction buckets on [B,21,21].'


class RetainedGeometryProvider(nn.Module):
    'Compute frozen N040 owner tokens from current joint angles in radians and ordered static evidence.'

    _EVIDENCE_NAMES = (
        "anchors",
        "home_surface_points",
        "home_surface_mask",
        "palm_normal",
        "space_screws",
        "q_home",
        "entity_role",
        "entity_joint_index",
        "joint_entity_index",
        "shortest_path",
        "parent_direction",
        "child_direction",
        "entity_valid_mask",
        "joint_valid_mask",
        "anchor_valid_mask",
    )
    _STATIC_CACHE_CHUNK_ASSETS = 16
    'Definition for static cache chunk assets (16).'

    def __init__(
        self,
        *,
        artifact: SE3RetainedEncoderArtifact,
        evidence_bank: CanonicalEvidenceBank,
        dataset_digest: str,
        manifest_digest: str,
        canonical_schema_digest: str,
        evidence_source_config: Mapping[str, Any],
    ) -> None:
        'Initialize the instance.'

        super().__init__()
        if not dataset_digest or not manifest_digest or not canonical_schema_digest:
            raise ValueError("retained geometry runtime identity digests must be non-empty")
        if artifact.feature_spec.get("entity_width") != 128:
            raise ValueError("retained N040 feature_spec entity_width must be 128")

        raw_evidence = evidence_bank.evidence
        asset_count = raw_evidence.anchors.shape[0]

        evidence = StaticGeometryEvidence(
            anchors=raw_evidence.anchors,
            home_surface_points=raw_evidence.home_surface_points,
            home_surface_mask=raw_evidence.home_surface_mask,
            palm_normal=raw_evidence.palm_normal,
            space_screws=raw_evidence.space_screws,
            q_home=raw_evidence.q_home,
            entity_role=raw_evidence.entity_role,
            entity_joint_index=raw_evidence.entity_joint_index,
            joint_entity_index=raw_evidence.joint_entity_index,
            shortest_path=raw_evidence.shortest_path,
            parent_direction=raw_evidence.parent_direction,
            child_direction=raw_evidence.child_direction,
            entity_valid_mask=(
                raw_evidence.entity_valid_mask
                if raw_evidence.entity_valid_mask is not None
                else torch.ones(asset_count, CANONICAL_OWNER_COUNT, dtype=torch.bool, device=raw_evidence.anchors.device)
            ),
            joint_valid_mask=(
                raw_evidence.joint_valid_mask
                if raw_evidence.joint_valid_mask is not None
                else torch.ones(asset_count, CANONICAL_JOINT_COUNT, dtype=torch.bool, device=raw_evidence.anchors.device)
            ),
            anchor_valid_mask=(
                raw_evidence.anchor_valid_mask
                if raw_evidence.anchor_valid_mask is not None
                else torch.ones(
                    asset_count,
                    raw_evidence.anchors.shape[1],
                    dtype=torch.bool,
                    device=raw_evidence.anchors.device,
                )
            ),
        )
        if evidence.home_surface_points.shape[1] != CANONICAL_OWNER_COUNT:
            raise ValueError("retained geometry evidence must contain 21 canonical owners")
        if evidence.space_screws.shape[1] != CANONICAL_JOINT_COUNT:
            raise ValueError("retained geometry evidence must contain 16 canonical joints")
        self.asset_ids = tuple(evidence_bank.asset_ids)
        self.physical_geometry_hashes = tuple(evidence_bank.physical_geometry_hashes)
        self.width = 128


        self.encoder = artifact.encoder
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self._cached_home: torch.Tensor | None = None  # `[A,G,D_h]` frozen learned static activation
        self._cached_screw: torch.Tensor | None = None  # `[A,N_J,D_s]` frozen learned line-anchor activation
        self._cached_role: torch.Tensor | None = None  # `[A,G,D_r]` frozen role embedding
        self._cached_graph_bias: torch.Tensor | None = None  # `[A,H,G,G]` frozen N040 graph-bias activation


        for name in self._EVIDENCE_NAMES:
            value = getattr(evidence, name)
            self.register_buffer(f"evidence_{name}", value.clone() if value is not None else None, persistent=False)

        evidence_tensor_digest = _evidence_digest(evidence)
        identity_payload = {
            "identity_schema_version": "1.0.0",
            "provider_type": "retained_se3_geometry_encoder",
            "provider_algorithm": "frozen-static-frontend-graph-cache-v2",
            "static_cache_chunk_assets": self._STATIC_CACHE_CHUNK_ASSETS,
            "width": self.width,
            "dataset_digest": str(dataset_digest),
            "manifest_digest": str(manifest_digest),
            "canonical_schema_digest": str(canonical_schema_digest),
            "asset_ids": list(self.asset_ids),
            "physical_geometry_hashes": list(self.physical_geometry_hashes),
            "evidence_source_config": dict(evidence_source_config),
            "evidence_tensor_digest": evidence_tensor_digest,
            "retained_artifact": {
                "path": str(artifact.path),
                "sha256": artifact.artifact_sha256,
                "schema_version": "5.0.0",
                "artifact_type": "retained_geometry_encoder",
                "encoder_type": "se3_invariant",
                "feature_spec": dict(artifact.feature_spec),
                "lineage": dict(artifact.lineage),
            },
        }
        self._identity_payload = {**identity_payload, "identity_digest": _stable_digest(identity_payload)}

    @property
    def identity(self) -> dict[str, Any]:
        'Handle identity.'

        artifact = dict(self._identity_payload["retained_artifact"])
        artifact["feature_spec"] = dict(artifact["feature_spec"])
        artifact["lineage"] = dict(artifact["lineage"])
        return {
            **self._identity_payload,
            "asset_ids": list(self.asset_ids),
            "physical_geometry_hashes": list(self.physical_geometry_hashes),
            "evidence_source_config": dict(self._identity_payload["evidence_source_config"]),
            "retained_artifact": artifact,
        }

    def train(self, mode: bool = True) -> RetainedGeometryProvider:
        'Train the declared contract.'

        super().train(mode)
        self.encoder.eval()
        return self

    def _apply(self, fn):
        'Handle apply.'

        result = super()._apply(fn)
        self._cached_home = None
        self._cached_screw = None
        self._cached_role = None
        self._cached_graph_bias = None
        return result

    def _evidence(self) -> StaticGeometryEvidence:
        'Hash the evidence.'

        def required(name: str) -> torch.Tensor:
            value = getattr(self, f"evidence_{name}")
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(f"retained geometry evidence buffer {name!r} is missing")
            return value

        return StaticGeometryEvidence(
            anchors=required("anchors"),
            home_surface_points=required("home_surface_points"),
            home_surface_mask=required("home_surface_mask"),
            palm_normal=required("palm_normal"),
            space_screws=required("space_screws"),
            q_home=required("q_home"),
            entity_role=required("entity_role"),
            entity_joint_index=required("entity_joint_index"),
            joint_entity_index=required("joint_entity_index"),
            shortest_path=required("shortest_path"),
            parent_direction=required("parent_direction"),
            child_direction=required("child_direction"),
            entity_valid_mask=required("entity_valid_mask"),
            joint_valid_mask=required("joint_valid_mask"),
            anchor_valid_mask=required("anchor_valid_mask"),
        )

    def _static_features(
        self, evidence: StaticGeometryEvidence
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        'Handle static features.'

        if (
            self._cached_home is None
            or self._cached_screw is None
            or self._cached_role is None
            or self._cached_graph_bias is None
        ):
            with torch.no_grad():
                home_chunks = []
                screw_chunks = []
                role_chunks = []
                graph_chunks = []
                asset_count = evidence.anchors.shape[0]
                for start in range(0, asset_count, self._STATIC_CACHE_CHUNK_ASSETS):
                    rows = torch.arange(
                        start,
                        min(start + self._STATIC_CACHE_CHUNK_ASSETS, asset_count),
                        device=evidence.anchors.device,
                    )
                    chunk = self._slice_evidence(evidence, rows)
                    home_chunks.append(self.encoder._home_features(chunk).detach())
                    screw_chunks.append(self.encoder._screw_features(chunk).detach())
                    role_one_hot = F.one_hot(chunk.entity_role, num_classes=3).to(
                        dtype=self.encoder.role_embedding.weight.dtype
                    )
                    role_chunks.append((role_one_hot @ self.encoder.role_embedding.weight).detach())
                    graph_chunks.append(
                        self.encoder.backbone._graph_bias(
                            chunk.shortest_path,
                            chunk.parent_direction,
                            chunk.child_direction,
                        ).detach()
                    )
                self._cached_home = torch.cat(home_chunks, dim=0)
                self._cached_screw = torch.cat(screw_chunks, dim=0)
                self._cached_role = torch.cat(role_chunks, dim=0)
                self._cached_graph_bias = torch.cat(graph_chunks, dim=0)
        return self._cached_home, self._cached_screw, self._cached_role, self._cached_graph_bias

    @staticmethod
    def _slice_evidence(evidence: StaticGeometryEvidence, rows: torch.Tensor) -> StaticGeometryEvidence:
        'Handle slice evidence.'

        return StaticGeometryEvidence(
            anchors=evidence.anchors[rows],
            home_surface_points=evidence.home_surface_points[rows],
            home_surface_mask=evidence.home_surface_mask[rows],
            palm_normal=evidence.palm_normal[rows],
            space_screws=evidence.space_screws[rows],
            q_home=evidence.q_home[rows],
            entity_role=evidence.entity_role[rows],
            entity_joint_index=evidence.entity_joint_index[rows],
            joint_entity_index=evidence.joint_entity_index[rows],
            shortest_path=evidence.shortest_path[rows],
            parent_direction=evidence.parent_direction[rows],
            child_direction=evidence.child_direction[rows],
            entity_valid_mask=evidence.entity_valid_mask[rows] if evidence.entity_valid_mask is not None else None,
            joint_valid_mask=evidence.joint_valid_mask[rows] if evidence.joint_valid_mask is not None else None,
            anchor_valid_mask=evidence.anchor_valid_mask[rows] if evidence.anchor_valid_mask is not None else None,
        )

    def _forward_cached(
        self,
        q_rad: torch.Tensor,
        evidence: StaticGeometryEvidence,
        asset_row: torch.Tensor,
    ) -> torch.Tensor:
        'Handle forward cached.'

        home_static, screw_static, role_static, graph_bias_static = self._static_features(evidence)
        entity_valid_source = evidence.entity_valid_mask
        joint_valid_source = evidence.joint_valid_mask
        if entity_valid_source is None or joint_valid_source is None:
            raise RuntimeError("retained provider requires explicit normalized owner/joint masks")
        entity_valid = entity_valid_source[asset_row]
        joint_valid = joint_valid_source[asset_row]
        q_home = evidence.q_home[asset_row]
        screw_batch = screw_static[asset_row] * joint_valid.unsqueeze(-1)
        theta = ((q_rad - q_home) / torch.pi) * joint_valid
        joint_motion = self.encoder.joint_motion_projection(torch.cat((theta.unsqueeze(-1), screw_batch), dim=-1))


        joint_entities = evidence.joint_entity_index[asset_row]
        entity_axis = torch.arange(CANONICAL_OWNER_COUNT, device=q_rad.device).view(1, CANONICAL_OWNER_COUNT, 1)
        joint_to_entity = (joint_entities.unsqueeze(1) == entity_axis).to(dtype=joint_motion.dtype)
        entity_motion = torch.bmm(joint_to_entity, joint_motion * joint_valid.unsqueeze(-1))
        entity_screw = torch.bmm(joint_to_entity, screw_batch)
        entity_input = torch.cat(
            (entity_motion, home_static[asset_row], entity_screw, role_static[asset_row]),
            dim=-1,
        )
        tokens = self.encoder.entity_projection(entity_input) * entity_valid.unsqueeze(-1)
        graph_bias = graph_bias_static[asset_row]  # shapes [B,H,G,G]
        for layer in self.encoder.backbone.layers:
            tokens = layer(tokens, graph_bias, entity_valid)
        entities = self.encoder.backbone.final_norm(tokens)
        return entities * entity_valid.unsqueeze(-1)

    def resolve(self, asset_row: torch.Tensor, q_rad: torch.Tensor) -> RetainedGeometryBatch:
        'Resolve the declared contract; shapes [B], [B,16], [B,21,128]; units rad.'

        if asset_row.ndim != 1 or asset_row.dtype not in {torch.int32, torch.int64, torch.long}:
            raise ValueError("asset_row must be a rank-1 integer tensor")
        if q_rad.ndim != 2 or q_rad.shape != (asset_row.shape[0], CANONICAL_JOINT_COUNT):
            raise ValueError(f"q_rad must have shape [{asset_row.shape[0]},{CANONICAL_JOINT_COUNT}]")
        evidence = self._evidence()
        if asset_row.device != evidence.anchors.device:
            asset_row = asset_row.to(device=evidence.anchors.device)
        if q_rad.device != evidence.anchors.device:
            raise ValueError("q_rad and retained geometry evidence must share a device")
        torch._assert_async(
            torch.all((asset_row >= 0) & (asset_row < evidence.anchors.shape[0])),
            "asset_row contains a row outside retained geometry evidence",
        )

        self.encoder.eval()
        with torch.no_grad():
            geometry_entities = self._forward_cached(q_rad, evidence, asset_row).detach()
        owner_valid_mask = evidence.entity_valid_mask
        joint_valid_mask = evidence.joint_valid_mask
        if owner_valid_mask is None or joint_valid_mask is None:
            raise RuntimeError("retained provider requires explicit normalized owner/joint masks")

        return RetainedGeometryBatch(
            geometry_entities=geometry_entities,
            owner_valid_mask=owner_valid_mask[asset_row],
            joint_valid_mask=joint_valid_mask[asset_row],
            shortest_path=evidence.shortest_path[asset_row],
            parent_direction=evidence.parent_direction[asset_row],
            child_direction=evidence.child_direction[asset_row],
        )


__all__ = ["RetainedGeometryBatch", "RetainedGeometryProvider"]
