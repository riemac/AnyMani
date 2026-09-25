"""Five-mother typed geometry source to canonical policy evidence bank."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from anymani.assets.bank import HandBank
from anymani.assets.bank.hand_container import HandContainer
from anymani.assets.canonical_runtime import CanonicalHandArtifact
from anymani.distill.models.input_adapters.geometry import (
    GeometryPaddingCfg,
    StaticGeometryEvidence,
    build_static_geometry_evidence,
    canonicalize_static_geometry_evidence,
    pad_static_geometry_evidence,
)
from anymani.distill.representations.sources.geometry_source import GeometrySource, GeometrySourceCfg

if TYPE_CHECKING:
    from anymani.robots.hand_spawn import HandSpawnCfg

CANONICAL_OWNER_COUNT = 21
CANONICAL_JOINT_COUNT = 16


@dataclass(frozen=True)
class CanonicalEvidenceBank:
    """Ordered source-backed geometry evidence for 16 joints and 21 owners."""

    evidence: StaticGeometryEvidence
    asset_ids: tuple[str, ...]
    physical_geometry_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.evidence.anchors.ndim != 3:
            raise ValueError("canonical evidence bank must be batched by asset row")
        row_count = self.evidence.anchors.shape[0]
        if len(self.asset_ids) != row_count or len(self.physical_geometry_hashes) != row_count:
            raise ValueError("canonical evidence provenance must align with asset rows")
        if self.evidence.home_surface_points.shape[1] != CANONICAL_OWNER_COUNT:
            raise ValueError("canonical evidence must contain 21 owner slots")
        if self.evidence.space_screws.shape[1] != CANONICAL_JOINT_COUNT:
            raise ValueError("canonical evidence must contain 16 joint slots")

    def gather(self, asset_row: torch.Tensor) -> StaticGeometryEvidence:
        """Gather all evidence fields by a one-dimensional batch of asset rows."""
        if asset_row.ndim != 1 or asset_row.dtype not in {torch.int32, torch.int64, torch.long}:
            raise ValueError("asset_row must be a rank-1 integer tensor")
        if torch.any(asset_row < 0) or torch.any(asset_row >= self.evidence.anchors.shape[0]):
            raise IndexError("asset_row contains an evidence-bank row outside the bank")
        evidence = self.evidence
        return StaticGeometryEvidence(
            anchors=evidence.anchors[asset_row],
            home_surface_points=evidence.home_surface_points[asset_row],
            home_surface_mask=evidence.home_surface_mask[asset_row],
            palm_normal=evidence.palm_normal[asset_row],
            space_screws=evidence.space_screws[asset_row],
            q_home=evidence.q_home[asset_row],
            entity_role=evidence.entity_role[asset_row],
            entity_joint_index=evidence.entity_joint_index[asset_row],
            joint_entity_index=evidence.joint_entity_index[asset_row],
            shortest_path=evidence.shortest_path[asset_row],
            parent_direction=evidence.parent_direction[asset_row],
            child_direction=evidence.child_direction[asset_row],
            entity_valid_mask=evidence.entity_valid_mask[asset_row] if evidence.entity_valid_mask is not None else None,
            joint_valid_mask=evidence.joint_valid_mask[asset_row] if evidence.joint_valid_mask is not None else None,
            anchor_valid_mask=evidence.anchor_valid_mask[asset_row] if evidence.anchor_valid_mask is not None else None,
        )

    def to(self, device: torch.device | str) -> "CanonicalEvidenceBank":
        """Move tensors to the policy device while keeping provenance on the host."""
        target = torch.device(device)
        evidence = self.evidence

        def move(value: torch.Tensor | None) -> torch.Tensor | None:
            return value.to(target) if value is not None else None

        return CanonicalEvidenceBank(
            evidence=StaticGeometryEvidence(
                anchors=evidence.anchors.to(target),
                home_surface_points=evidence.home_surface_points.to(target),
                home_surface_mask=evidence.home_surface_mask.to(target),
                palm_normal=evidence.palm_normal.to(target),
                space_screws=evidence.space_screws.to(target),
                q_home=evidence.q_home.to(target),
                entity_role=evidence.entity_role.to(target),
                entity_joint_index=evidence.entity_joint_index.to(target),
                joint_entity_index=evidence.joint_entity_index.to(target),
                shortest_path=evidence.shortest_path.to(target),
                parent_direction=evidence.parent_direction.to(target),
                child_direction=evidence.child_direction.to(target),
                entity_valid_mask=move(evidence.entity_valid_mask),
                joint_valid_mask=move(evidence.joint_valid_mask),
                anchor_valid_mask=move(evidence.anchor_valid_mask),
            ),
            asset_ids=self.asset_ids,
            physical_geometry_hashes=self.physical_geometry_hashes,
        )


def build_canonical_evidence_bank(
    hand_spawn_cfg: HandSpawnCfg,
    artifacts: Sequence[CanonicalHandArtifact],
    *,
    source_assets: Sequence[HandContainer] | None = None,
    source_cfg: GeometrySourceCfg = GeometrySourceCfg(),
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> CanonicalEvidenceBank:
    'Build canonical evidence bank; shapes [CanonicalHandArtifact], [HandContainer].'

    if source_assets is None and not hand_spawn_cfg.bank.require_geometry_semantics:
        raise ValueError("canonical evidence requires HandBankCfg.require_geometry_semantics=True")
    ordered_sources = (
        tuple(source_assets) if source_assets is not None else HandBank(hand_spawn_cfg.bank).resolve().assets
    )
    if len(ordered_sources) != len(artifacts):
        raise ValueError("canonical source selection and artifact rows must have equal length")
    if any(container.geometry_semantics is None for container in ordered_sources):
        raise ValueError("canonical evidence source_assets must all contain typed geometry semantics")

    canonical_evidences = []
    asset_ids: list[str] = []
    physical_hashes: list[str] = []
    for expected_row, (container, artifact) in enumerate(zip(ordered_sources, artifacts)):
        if artifact.asset_id != container.asset_id or artifact.routing.asset_row != expected_row:
            raise ValueError("canonical artifact row does not match source HandBank ordering")
        semantics = container.geometry_semantics
        if semantics is None:
            raise ValueError(f"asset {container.asset_id!r} lacks typed geometry semantics")
        source = GeometrySource.materialize(container, config=source_cfg)
        source_evidence = build_static_geometry_evidence(
            semantics,
            source.spec_cpu,
            source.home_surface,
            source.anchors,
            device="cpu",
            dtype=dtype,
        )
        canonical_evidences.append(canonicalize_static_geometry_evidence(source_evidence, semantics, artifact.routing))
        asset_ids.append(container.asset_id)
        physical_hashes.append(source.identity.physical_geometry_hash)

    evidence = pad_static_geometry_evidence(
        canonical_evidences,
        config=GeometryPaddingCfg(max_joint_count=16, max_tip_count=4, max_graph_distance=8),
    )
    return CanonicalEvidenceBank(
        evidence=evidence,
        asset_ids=tuple(asset_ids),
        physical_geometry_hashes=tuple(physical_hashes),
    ).to(device)


__all__ = [
    "CANONICAL_JOINT_COUNT",
    "CANONICAL_OWNER_COUNT",
    "CanonicalEvidenceBank",
    "build_canonical_evidence_bank",
]
