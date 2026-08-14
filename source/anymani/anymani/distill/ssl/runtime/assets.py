r"""Geometry SSL runtime 的资产解析、CPU 物化与 physical-group split。

本模块是 ``assets -> robots -> ssl`` 的运行边界：只通过 ``HandBank`` 获取类型化静态语义，
再物化 CPU geometry runtime、物理映射身份与 manifest。它不创建模型、GPU resident window、
optimizer 或 validation metric，也不解析 URDF/hand.yaml。
"""

from __future__ import annotations

import hashlib  # static anchor/home-surface realization 的 byte-level fingerprint
from typing import Literal  # generated/official 决定 bank 的迁移与 fail-closed 路由

import torch  # official identity-only lowering 使用 CPU float64

from anymani.assets.bank.hand_bank import HandBank, HandBankCfg  # 资产集合唯一入口
from anymani.assets.bank.hand_container import HandContainer, HandContainerCfg  # 显式 bundle 选择
from anymani.distill.representations.queries.spatial_sampling import SURFACE_QUERY_SAMPLING_VERSION
from anymani.distill.ssl.config import GeometrySSLAssetManifest, GeometrySSLExperimentCfg
from anymani.distill.ssl.dataset import GeometryAssetRuntime, materialize_geometry_asset_runtime
from anymani.distill.ssl.split import (
    GeometryAssetIdentityRecord,
    GroupedGeometryAssetSplit,
    split_geometry_asset_groups,
)
from anymani.robots.geometry_kinematics import lower_hand_geometry_semantics
from anymani.robots.owner_geometry import (
    AnchorSamples,
    GeometryIdentity,
    HomeSurfaceSamples,
    OwnerGeometryCache,
    geometry_identity,
    materialize_owner_geometry_cache,
)


def resolve_assets(
    paths: tuple[str, ...],
    *,
    source_kind: Literal["generated", "official"],
) -> tuple[HandContainer, ...]:
    r"""通过 HandBank explicit route 解析资产，不在 SSL 重读 sidecar/URDF 细节。

    Args:
        paths (tuple[str, ...]): 已由 resolved experiment 冻结的 bundle roots。
        source_kind (Literal["generated", "official"]): generated 允许版本化迁移；official
            缺人工核验几何语义时严格失败。

    Returns:
        tuple[HandContainer, ...]: 与声明路径同序、包含 ``geometry_semantics`` 的资产。
    """

    if not paths:  # validation/official split 可以为空
        return ()
    selection = HandBank(
        HandBankCfg(
            source_mode="post_mutate",  # explicit route 下只作 provenance，不触发目录 discovery
            selection_mode="explicit",  # 资产身份由 resolved config 精确冻结
            containers=tuple(HandContainerCfg(path=path, source_kind=source_kind) for path in paths),
            require_geometry_semantics=True,  # owner/运动学/anchor 语义必须由 assets 层交付
        )
    ).resolve()
    return selection.assets  # HandBank 保持配置声明顺序


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
) -> dict[str, str]:
    r"""提取 content、physical mapping 与 configuration-domain 三层身份。"""

    semantics = container.geometry_semantics  # bank 已验证的静态语义真源
    if semantics is None:
        raise ValueError("manifest asset is missing geometry semantics")
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
        **anchor_realization_record(anchors),
        **home_surface_realization_record(home_surface, geometry_cache),
    }


def build_manifest(
    train_assets: tuple[GeometryAssetRuntime, ...],
    validation_assets: tuple[GeometryAssetRuntime, ...],
    official_assets: tuple[tuple[HandContainer, GeometryIdentity], ...],
    *,
    grouped_split: GroupedGeometryAssetSplit | None = None,
) -> GeometrySSLAssetManifest:
    r"""冻结三类 split，并通过 manifest 构造拒绝 content/physical identity 泄漏。"""

    return GeometrySSLAssetManifest(
        schema_version="1.1.0",
        train=tuple(
            manifest_record(
                asset.container,
                asset.identity,
                asset.anchors,
                asset.home_surface,
                asset.geometry_cache,
            )
            for asset in train_assets
        ),
        validation=tuple(
            manifest_record(
                asset.container,
                asset.identity,
                asset.anchors,
                asset.home_surface,
                asset.geometry_cache,
            )
            for asset in validation_assets
        ),
        official_evaluation=tuple(manifest_record(container, identity) for container, identity in official_assets),
        split_strategy="physical_group" if grouped_split is not None else "explicit",
        split_seed=0 if grouped_split is None else grouped_split.split_seed,
        requested_validation_asset_count=(
            0 if grouped_split is None else grouped_split.requested_validation_asset_count
        ),
        actual_validation_asset_count=(
            len(validation_assets) if grouped_split is None else grouped_split.actual_validation_asset_count
        ),
    )


def identity_record(runtime: GeometryAssetRuntime) -> GeometryAssetIdentityRecord:
    r"""把 CPU runtime 规约成 physical-group split 所需的最小身份记录。"""

    semantics = runtime.container.geometry_semantics
    if semantics is None:
        raise ValueError("identity record asset is missing geometry semantics")
    return GeometryAssetIdentityRecord(
        asset_id=runtime.asset_id,
        path=str(runtime.container.urdf_path.parent),
        content_hash=semantics.content_hash,
        physical_geometry_hash=runtime.identity.physical_geometry_hash,
        configuration_domain_hash=runtime.identity.configuration_domain_hash,
    )


def materialize_assets(
    assets: tuple[HandContainer, ...],
    *,
    config: GeometrySSLExperimentCfg,
) -> tuple[GeometryAssetRuntime, ...]:
    r"""按 resolved 静态采样配置物化 CPU geometry runtime 与双重 identity。"""

    return tuple(
        materialize_geometry_asset_runtime(
            asset,
            query_config=config.query,
            config=config.materialization,
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


def resolve_generated_runtime_splits(
    config: GeometrySSLExperimentCfg,
) -> tuple[tuple[GeometryAssetRuntime, ...], tuple[GeometryAssetRuntime, ...], GroupedGeometryAssetSplit | None]:
    r"""解析显式 split，或从完整 family catalog 构造 physical-geometry grouped split。"""

    if config.assets.family_paths:
        family_assets = resolve_assets(config.assets.family_paths, source_kind="generated")
        family_runtime = materialize_assets(family_assets, config=config)
        grouped = split_geometry_asset_groups(
            tuple(identity_record(runtime) for runtime in family_runtime),
            mother_asset_id=config.assets.mother_asset_id,
            validation_asset_count=config.assets.validation_asset_count,
            split_seed=config.assets.split_seed,
        )
        runtime_by_id = {runtime.asset_id: runtime for runtime in family_runtime}
        train = tuple(runtime_by_id[record.asset_id] for record in grouped.train)
        validation = tuple(runtime_by_id[record.asset_id] for record in grouped.validation)
        return train, validation, grouped

    train_assets = resolve_assets(config.assets.train_paths, source_kind="generated")
    validation_assets = resolve_assets(config.assets.validation_paths, source_kind="generated")
    return (
        materialize_assets(train_assets, config=config),
        materialize_assets(validation_assets, config=config),
        None,
    )


__all__ = [
    "anchor_realization_record",
    "build_manifest",
    "home_surface_realization_record",
    "materialize_identity_only",
    "resolve_assets",
    "resolve_generated_runtime_splits",
]
