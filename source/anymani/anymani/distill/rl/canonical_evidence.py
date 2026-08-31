"""Five-mother typed geometry source to canonical policy evidence bank."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from anymani.assets.bank import HandBank
from anymani.assets.bank.hand_container import HandContainer
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
    source_assets: Sequence[HandContainer] | None = None,
    source_cfg: GeometrySourceCfg = GeometrySourceCfg(),
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> CanonicalEvidenceBank:
    r"""Materialize real source geometry and scatter it to fixed canonical owner/joint axes.

    The source selection and artifact sequence must share the same asset-row order. Geometry is built
    from ``HandContainer.geometry_semantics`` and original collision meshes; canonical ghost links are
    deliberately absent from surfaces, anchors, screws, and physical identity. Heterogeneous runtime 已在
    dataset/prepared-cache 阶段冻结 ordered source assets，因此可通过 ``source_assets`` 交付同一对象，
    避免本函数以空/不同的 ``HandBankCfg`` 二次解析实验 split。

    Args:
        hand_spawn_cfg (HandSpawnCfg): spawn/frame与默认bank合同；必须要求typed geometry semantics。
        artifacts (Sequence[CanonicalHandArtifact]): 与source assets同序的canonical routing artifacts。
        source_assets (Sequence[HandContainer] | None): 可选已解析ordered source containers；省略时
            才按``hand_spawn_cfg.bank``解析旧five-mother route。
        source_cfg (GeometrySourceCfg): anchors/home sampling与物理source配置。
        device (torch.device | str): 最终static evidence bank驻留设备。
        dtype (torch.dtype): 连续几何tensor dtype，N040正式值为FP32。

    Returns:
        CanonicalEvidenceBank: 固定16-JOINT/21-owner axes的source-backed evidence与provenance。
    """

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


__all__ = ["build_canonical_evidence_bank"]
