r"""Geometry SSL runtime 的 dataset resolve、CPU 物化与 expanded asset manifest。

本模块是 ``assets -> representations.sources -> ssl`` 的运行边界：只通过 ``HandBank`` 获取类型化静态语义，
再物化 CPU geometry runtime、物理映射身份与 manifest。它不创建模型、GPU resident window、
optimizer 或 validation metric，也不解析 URDF/hand.yaml。
"""

from __future__ import annotations

import hashlib  # static anchor/home-surface realization 的 byte-level fingerprint
from dataclasses import dataclass

import torch  # official identity-only lowering 使用 CPU float64

from anymani.assets.bank.dataset import (
    HandAssetDataset,
    HandAssetProvenance,
    ResolvedHandAssetDataset,
    ResolvedHandAssetPartition,
)
from anymani.assets.bank.hand_container import HandContainer
from anymani.distill.representations.queries.spatial_sampling import SURFACE_QUERY_SAMPLING_VERSION
from anymani.distill.representations.sources.collision_geometry import (
    AnchorSamples,
    GeometryIdentity,
    HomeSurfaceSamples,
    OwnerGeometryCache,
    geometry_identity,
    materialize_owner_geometry_cache,
)
from anymani.distill.representations.sources.geometry_source import GeometrySource
from anymani.distill.representations.sources.kinematics import lower_hand_geometry_semantics
from anymani.distill.ssl.config import GeometrySSLAssetManifest, GeometrySSLExperimentCfg


@dataclass(frozen=True)
class GeometrySSLResolvedAssets:
    r"""通用 dataset 进入 SSL 后的 train/validation runtime 与 evaluation identities。"""

    dataset: ResolvedHandAssetDataset  # 原始 YAML identity 与全部 partition provenance
    train: tuple[GeometrySource, ...]  # optimizer/calibration 使用的完整 geometry sources
    validation: tuple[GeometrySource, ...]  # checkpoint selection 使用的完整 held-out sources
    evaluation: dict[str, tuple[tuple[HandContainer, GeometryIdentity], ...]]  # 仅 identity，不执行 forward


def anchor_realization_record(anchors: AnchorSamples | None) -> dict[str, str]:
    r"""把实际 anchor 点集及其采样语义规约成可供 resume 比对的 manifest 字段。"""

    if anchors is None:  # official identity-only 资产不生成训练 anchor
        return {
            "anchor_realization_hash": "",
            "anchor_sampling_version": "",
            "anchor_sampling_seed": "",
            "anchor_count": "",
            "anchor_support_radius_m": "",
            "anchor_radial_decay_scale_m": "",
            "anchor_surface_fraction": "",
        }
    digest = hashlib.sha256()
    digest.update(b"anymani-anchor-realization-v1\0")
    for array in (anchors.anchors_hand_m, anchors.surface_mask):
        contiguous = array.copy(order="C")
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
    for values in (anchors.finger_names, anchors.seed_ids):
        for value in values:
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(4, "little"))
            digest.update(encoded)
    scalar_provenance = (
        anchors.algorithm_version,
        str(anchors.sampling_seed),
        repr(anchors.radial_support_radius_m),
        repr(anchors.radial_decay_scale_m),
        repr(anchors.surface_fraction),
    )
    for value in scalar_provenance:
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return {
        "anchor_realization_hash": digest.hexdigest(),
        "anchor_sampling_version": anchors.algorithm_version,
        "anchor_sampling_seed": str(anchors.sampling_seed),
        "anchor_count": str(len(anchors.anchors_hand_m)),
        "anchor_support_radius_m": repr(anchors.radial_support_radius_m),
        "anchor_radial_decay_scale_m": repr(anchors.radial_decay_scale_m),
        "anchor_surface_fraction": repr(anchors.surface_fraction),
    }


def home_surface_realization_record(
    home_surface: HomeSurfaceSamples | None,
    geometry_cache: OwnerGeometryCache | None,
) -> dict[str, str]:
    r"""记录 retained home points 与其真实 surface/Boolean 生产语义。"""

    if home_surface is None or geometry_cache is None:  # official identity-only 路径不生成 retained samples
        return {
            "home_surface_realization_hash": "",
            "home_surface_sampling_seed": "",
            "home_surface_oversample_factor": "",
            "boolean_backend": "",
            "surface_geometry_hash": "",
            "surface_processing_version": "",
            "surface_query_sampling_version": "",
        }
    digest = hashlib.sha256()
    digest.update(b"anymani-home-surface-realization-v1\0")
    for array in (home_surface.points_owner_local_m, home_surface.face_indices, home_surface.barycentric):
        contiguous = array.copy(order="C")
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
    for owner_id in home_surface.owner_ids:
        encoded = owner_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    for value in (str(home_surface.sampling_seed), str(home_surface.oversample_factor)):
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return {
        "home_surface_realization_hash": digest.hexdigest(),
        "home_surface_sampling_seed": str(home_surface.sampling_seed),
        "home_surface_oversample_factor": str(home_surface.oversample_factor),
        "boolean_backend": geometry_cache.boolean_backend,
        "surface_geometry_hash": geometry_cache.surface_geometry_hash,
        "surface_processing_version": geometry_cache.surface_processing_version,
        "surface_query_sampling_version": SURFACE_QUERY_SAMPLING_VERSION,
    }


def manifest_record(
    container: HandContainer,
    identity: GeometryIdentity,
    anchors: AnchorSamples | None = None,
    home_surface: HomeSurfaceSamples | None = None,
    geometry_cache: OwnerGeometryCache | None = None,
    provenance: HandAssetProvenance | None = None,
) -> dict[str, str]:
    r"""提取 content、physical mapping 与 configuration-domain 三层身份。"""

    semantics = container.geometry_semantics  # bank 已验证的静态语义真源
    if semantics is None:
        raise ValueError("manifest asset is missing geometry semantics")
    lineage = provenance or HandAssetProvenance(
        partition="",
        run_alias="",
        run_dir="",
        collection_kind="official",
        group_name="",
        mother_name="",
        mother_path="",
        variant_set="",
        asset_role="official",
    )
    return {
        "asset_id": container.asset_id,
        "content_hash": semantics.content_hash,
        "physical_geometry_hash": identity.physical_geometry_hash,
        "configuration_domain_hash": identity.configuration_domain_hash,
        "source_kind": semantics.source_kind,
        "topology_key": semantics.topology_key or "",
        "family": semantics.family,
        "handedness": semantics.handedness,
        "joint_count": str(len(semantics.active_joint_names)),
        "owner_count": str(len(semantics.owners)),
        "dataset_partition": lineage.partition,
        "run_alias": lineage.run_alias,
        "run_dir": lineage.run_dir,
        "collection_kind": lineage.collection_kind,
        "group_name": lineage.group_name,
        "mother_name": lineage.mother_name,
        "mother_path": lineage.mother_path,
        "variant_set": lineage.variant_set,
        "asset_role": lineage.asset_role,
        **anchor_realization_record(anchors),
        **home_surface_realization_record(home_surface, geometry_cache),
    }


def build_manifest(resolved: GeometrySSLResolvedAssets) -> GeometrySSLAssetManifest:
    r"""把 dataset provenance 与 geometry identity 合并成唯一 expanded asset manifest。"""

    train_provenance = _provenance_by_asset_id(resolved.dataset.train)
    validation_provenance = _provenance_by_asset_id(resolved.dataset.validation)
    evaluation_provenance = {
        name: _provenance_by_asset_id(partition) for name, partition in resolved.dataset.evaluation.items()
    }
    return GeometrySSLAssetManifest(
        schema_version="2.0.0",
        dataset_source_path=str(resolved.dataset.source_path),
        dataset_source_sha256=resolved.dataset.source_sha256,
        train=tuple(
            manifest_record(
                asset.container,
                asset.identity,
                asset.anchors,
                asset.home_surface,
                asset.geometry_cache,
                train_provenance[asset.asset_id],
            )
            for asset in resolved.train
        ),
        validation=tuple(
            manifest_record(
                asset.container,
                asset.identity,
                asset.anchors,
                asset.home_surface,
                asset.geometry_cache,
                validation_provenance[asset.asset_id],
            )
            for asset in resolved.validation
        ),
        evaluation={
            name: tuple(
                manifest_record(container, identity, provenance=evaluation_provenance[name][container.asset_id])
                for container, identity in identities
            )
            for name, identities in resolved.evaluation.items()
        },
    )


def _provenance_by_asset_id(partition: ResolvedHandAssetPartition) -> dict[str, HandAssetProvenance]:
    r"""把 dataset partition 的稳定 asset axis 转成 manifest join index。"""

    return {record.container.asset_id: record.provenance for record in partition.records}


def materialize_assets(
    assets: tuple[HandContainer, ...],
    *,
    config: GeometrySSLExperimentCfg,
) -> tuple[GeometrySource, ...]:
    r"""按 resolved 静态采样配置物化 CPU geometry runtime 与双重 identity。"""

    return tuple(
        GeometrySource.materialize(
            asset,
            config=config.representation.source,
        )
        for asset in assets
    )


def materialize_identity_only(
    assets: tuple[HandContainer, ...],
) -> tuple[tuple[HandContainer, GeometryIdentity], ...]:
    r"""为 official 隔离 manifest 只物化 identity，不构造 home/anchor/workspace/encoder evidence。"""

    records: list[tuple[HandContainer, GeometryIdentity]] = []
    for asset in assets:
        semantics = asset.geometry_semantics
        if semantics is None:
            raise ValueError("identity-only asset is missing geometry semantics")
        spec = lower_hand_geometry_semantics(semantics, dtype=torch.float64)  # 物理 hash 使用 CPU float64 真值
        cache = materialize_owner_geometry_cache(asset, spec)  # 实际 owner surface 是 mapping identity 的一部分
        records.append((asset, geometry_identity(semantics, spec, cache)))
    return tuple(records)


def resolve_geometry_ssl_assets(config: GeometrySSLExperimentCfg) -> GeometrySSLResolvedAssets:
    r"""加载通用 dataset YAML，并按 SSL 生命周期物化 train/validation/evaluation。"""

    if not config.asset_dataset_manifest:
        raise ValueError("geometry SSL requires asset_dataset_manifest")
    dataset = HandAssetDataset.from_yaml(config.asset_dataset_manifest).resolve(require_geometry_semantics=True)
    train = materialize_assets(dataset.train.assets, config=config)
    validation = materialize_assets(dataset.validation.assets, config=config)
    evaluation = {name: materialize_identity_only(partition.assets) for name, partition in dataset.evaluation.items()}
    return GeometrySSLResolvedAssets(
        dataset=dataset,
        train=train,
        validation=validation,
        evaluation=evaluation,
    )


__all__ = [
    "GeometrySSLResolvedAssets",
    "anchor_realization_record",
    "build_manifest",
    "home_surface_realization_record",
    "materialize_identity_only",
    "resolve_geometry_ssl_assets",
]
