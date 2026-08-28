"""Five-mother typed geometry source to canonical policy evidence bank."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from anymani.assets.bank import HandBank
from anymani.assets.canonical_runtime import CanonicalHandArtifact
from anymani.distill.models.input_adapters.geometry import (
    GeometryPaddingCfg,
    build_static_geometry_evidence,
    canonicalize_static_geometry_evidence,
    pad_static_geometry_evidence,
)
from anymani.distill.models.policy import CanonicalEvidenceBank
from anymani.distill.representations.sources.geometry_source import GeometrySource, GeometrySourceCfg
from anymani.robots.hand_spawn import HandSpawnCfg


def build_canonical_evidence_bank(
    hand_spawn_cfg: HandSpawnCfg,
    artifacts: Sequence[CanonicalHandArtifact],
    *,
    source_cfg: GeometrySourceCfg = GeometrySourceCfg(),
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> CanonicalEvidenceBank:
    r"""Materialize real source geometry and scatter it to fixed canonical owner/joint axes.

    The source selection and artifact sequence must share the same asset-row order. Geometry is built
    from ``HandContainer.geometry_semantics`` and original collision meshes; canonical ghost links are
    deliberately absent from surfaces, anchors, screws, and physical identity.
    """

    if not hand_spawn_cfg.bank.require_geometry_semantics:
        raise ValueError("canonical evidence requires HandBankCfg.require_geometry_semantics=True")
    source_assets = HandBank(hand_spawn_cfg.bank).resolve().assets
    if len(source_assets) != len(artifacts):
        raise ValueError("canonical source selection and artifact rows must have equal length")

    canonical_evidences = []
    asset_ids: list[str] = []
    physical_hashes: list[str] = []
    for expected_row, (container, artifact) in enumerate(zip(source_assets, artifacts)):
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


__all__ = ["build_canonical_evidence_bank"]
