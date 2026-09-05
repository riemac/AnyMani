r"""Member-level hand cohort lock built on published dataset manifests.

``HandAssetDataset`` owns complete train/validation/evaluation partitions whose generated-lineage
selection atom is a whole variant set. Policy scaling needs a narrower immutable view: one experiment
may select the mother and three variants from each of many lineages, while preserving the exact
source-manifest identity of every member. This module resolves that view without rescanning the
generated tree or redefining bundle semantics.

A cohort lock uses two independent coordinates:

* ``cohort_index`` is the dense local tensor axis $k\in\{0,\ldots,A-1\}$ used by simulation and
  learning;
* ``source_alias + source_row`` is provenance into one parent dataset manifest and is never treated
  as a globally unique physical identity.

The lock validates parent YAML bytes, asset ID, content hash and full lineage provenance before it
returns :class:`ResolvedHandAssetRecord` objects. Canonical physical hashes remain a downstream
lowering certificate and are attached to the run identity rather than guessed at this source layer.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from .dataset import HandAssetDataset, HandAssetProvenance, ResolvedHandAssetPartition, ResolvedHandAssetRecord
from .path_utils import resolve_bank_path
from .prepared_train import resolve_prepared_train
from .yaml_utils import safe_load

HAND_ASSET_COHORT_SCHEMA_VERSION = "1.1.0"
"""Resolved member-level cohort lock schema."""

HAND_ASSET_CANONICAL_COHORT_SCHEMA_VERSION = "1.2.0"
"""Canonical-final cohort lock schema；在1.1 source lock之上冻结lowering后的物理身份。"""

_SUPPORTED_HAND_ASSET_COHORT_SCHEMA_VERSIONS = {
    "1.0.0",
    HAND_ASSET_COHORT_SCHEMA_VERSION,
    HAND_ASSET_CANONICAL_COHORT_SCHEMA_VERSION,
}
"""1.0只读保留已产生A16证据；所有新writer均发布含mutation digest的1.1。"""


@dataclass(frozen=True)
class HandAssetCohortSource:
    r"""One parent train manifest whose byte identity is sealed by the cohort lock."""

    alias: str  # Short source-qualified member prefix, for example ``ppo`` or ``ssl``.
    manifest_path: Path  # Resolved parent dataset YAML path.
    manifest_sha256: str  # Exact parent YAML byte digest.


@dataclass(frozen=True)
class HandAssetCohortMember:
    r"""One cohort-axis member and its verified parent-manifest coordinate."""

    cohort_index: int  # Dense local axis $k\in[0,A)$; list order must equal this value.
    source_alias: str  # Key into :attr:`ResolvedHandAssetCohort.sources`.
    source_row: int  # Row within the selected source manifest's resolved train partition.
    asset_id: str  # Sidecar asset identity verified against the resolved container.
    content_hash: str  # Typed source semantics identity; empty only for legacy fixtures.
    provenance: HandAssetProvenance  # Exact run/group/mother/variant role from the source resolver.
    mutation_descriptor: Mapping[str, Any]  # Variant mutation payload digest/modes；1.0历史lock为空
    configuration_domain_hash: str = ""  # 1.2冻结canonical materializer输入的完整配置域
    physical_geometry_hash: str = ""  # 1.2由canonical lowering产生的物理身份SHA-256
    canonical_schema_digest: str = ""  # 1.2冻结canonical storage/routing schema

    @property
    def source_key(self) -> str:
        r"""Return the source-qualified row key, which is unique even when numeric rows collide."""

        return f"{self.source_alias}#{self.source_row}"


@dataclass(frozen=True)
class ResolvedHandAssetCohort:
    r"""Verified member-level lock plus the existing resolved asset records it selects."""

    cohort_id: str  # Human-readable stable experiment-support identifier.
    lock_path: Path  # Exact persisted lock path.
    lock_sha256: str  # Exact lock YAML bytes.
    selection: Mapping[str, Any]  # Auditable selector recipe/result metadata; never policy input.
    canonical_binding: Mapping[str, Any]  # 1.2的source-lock与lowering协议证书；旧schema为空
    sources: Mapping[str, HandAssetCohortSource]  # Parent manifests keyed by lock alias.
    members: tuple[HandAssetCohortMember, ...]  # Ordered local member axis.
    partition: ResolvedHandAssetPartition  # Existing records in exactly the same local order.

    @property
    def assets(self):
        r"""Return the ordered :class:`HandContainer` axis consumed by canonical lowering."""

        return self.partition.assets

    @property
    def source_keys(self) -> tuple[str, ...]:
        r"""Return stable human/machine provenance keys aligned with the local cohort axis."""

        return tuple(member.source_key for member in self.members)


def _validate_lock_mapping(document: Mapping[str, Any]) -> tuple[str, str, Mapping[str, Any], list[Any]]:
    r"""Validate non-I/O cohort fields and return the three structured roots."""

    schema_version = str(document.get("schema_version", ""))
    if schema_version not in _SUPPORTED_HAND_ASSET_COHORT_SCHEMA_VERSIONS:
        raise ValueError(
            f"hand asset cohort schema must be one of {sorted(_SUPPORTED_HAND_ASSET_COHORT_SCHEMA_VERSIONS)!r}"
        )
    cohort_id = str(document.get("cohort_id", "")).strip()
    sources = document.get("sources")
    members = document.get("members")
    if not cohort_id or not isinstance(sources, Mapping) or not sources:
        raise ValueError("cohort lock requires a non-empty cohort_id and sources mapping")
    if not isinstance(members, list) or not members:
        raise ValueError("cohort lock requires a non-empty ordered members list")
    selection = document.get("selection", {})
    if not isinstance(selection, Mapping):
        raise TypeError("cohort selection metadata must be a mapping")
    return schema_version, cohort_id, cast(Mapping[str, Any], selection), members


def _mutation_descriptor(record: ResolvedHandAssetRecord) -> dict[str, Any]:
    r"""把variant的完整mutation payload规约为可审计且紧凑的lock字段。

    ``post_mutate_samples``可能包含每个joint/link的数百个连续采样值；lock保存其canonical SHA-256作为完整
    身份，并额外展开各mutator的``resolved_self_mode``供人工核对。Mother没有mutation，使用显式kind区分。
    """

    if record.provenance.asset_role == "mother":
        return {"kind": "mother"}  # mother本身没有post-mutate随机变量
    samples = record.container.sidecar.get("post_mutate_samples")
    if not isinstance(samples, Mapping) or not samples:
        raise ValueError(f"variant asset {record.container.asset_id!r} lacks post_mutate_samples")
    payload = json.dumps(samples, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    mutator_modes = {
        str(name): str(values.get("resolved_self_mode", ""))
        for name, values in sorted(samples.items())
        if isinstance(values, Mapping)
    }  # 人类可读的mutator activation摘要；完整连续值由下方digest冻结
    return {
        "kind": "variant",
        "post_mutate_samples_sha256": hashlib.sha256(payload).hexdigest(),
        "mutator_modes": mutator_modes,
        "source_origin_sample_id": str(record.container.sidecar.get("source_origin_sample_id", "")),
        "source_origin_topology_dir": str(record.container.sidecar.get("source_origin_topology_dir", "")),
    }


def _member_document(
    *,
    cohort_index: int,
    source_alias: str,
    source_row: int,
    record: ResolvedHandAssetRecord,
) -> dict[str, Any]:
    r"""Serialize one already-resolved record without duplicating bundle parsing logic."""

    return {
        "cohort_index": int(cohort_index),
        "source_alias": source_alias,
        "source_row": int(source_row),
        "asset_id": record.container.asset_id,
        "content_hash": record.content_hash,
        "provenance": asdict(record.provenance),
        "mutation_descriptor": _mutation_descriptor(record),
    }


def load_hand_asset_cohort(
    path: str | Path,
    *,
    require_geometry_semantics: bool = True,
    allow_legacy_left_handedness: bool = False,
) -> ResolvedHandAssetCohort:
    r"""Load a member-level lock and verify every selected record against its parent manifest.

    Args:
        path (str | Path): Cohort lock YAML path, absolute or relative to the AnyMani root.
        require_geometry_semantics (bool): Require typed geometry on every selected source record.
        allow_legacy_left_handedness (bool): Explicit legacy escape hatch inherited from dataset resolution.

    Returns:
        ResolvedHandAssetCohort: Exact local member axis ready for canonical lowering.
    """

    lock_path = resolve_bank_path(path)
    raw_bytes = lock_path.read_bytes()
    raw_document = safe_load(raw_bytes) or {}
    if not isinstance(raw_document, Mapping):
        raise TypeError("cohort lock YAML root must be a mapping")
    schema_version, cohort_id, selection, raw_members = _validate_lock_mapping(raw_document)

    # Each source is resolved through the canonical dataset/prepared-cache path, then sealed against lock bytes.
    sources: dict[str, HandAssetCohortSource] = {}
    partitions: dict[str, ResolvedHandAssetPartition] = {}
    raw_sources = cast(Mapping[str, Any], raw_document["sources"])
    for alias, raw_source in raw_sources.items():
        alias = str(alias).strip()
        if not alias or "#" in alias or not isinstance(raw_source, Mapping):
            raise ValueError("cohort source aliases must be non-empty, '#' free mappings")
        dataset = HandAssetDataset.from_yaml(str(raw_source.get("manifest_path", "")))
        expected_sha = str(raw_source.get("manifest_sha256", ""))
        if dataset.source_sha256 != expected_sha:
            raise ValueError(
                f"cohort source {alias!r} manifest SHA mismatch: expected={expected_sha}, "
                f"actual={dataset.source_sha256}"
            )
        partition, _cache_hit = resolve_prepared_train(
            dataset,
            require_geometry_semantics=require_geometry_semantics,
            allow_legacy_left_handedness=allow_legacy_left_handedness,
        )
        sources[alias] = HandAssetCohortSource(alias, dataset.source_path, dataset.source_sha256)
        partitions[alias] = partition

    # Numeric source rows may overlap across manifests; source-qualified keys and physical assets may not.
    members: list[HandAssetCohortMember] = []
    records: list[ResolvedHandAssetRecord] = []
    source_keys: set[str] = set()
    asset_ids: set[str] = set()
    nonempty_content_hashes: set[str] = set()
    bundle_paths: set[Path] = set()
    for expected_index, raw_member in enumerate(raw_members):
        if not isinstance(raw_member, Mapping):
            raise TypeError(f"cohort member {expected_index} must be a mapping")
        cohort_index = int(raw_member.get("cohort_index", -1))
        source_alias = str(raw_member.get("source_alias", ""))
        source_row = int(raw_member.get("source_row", -1))
        if cohort_index != expected_index:
            raise ValueError(f"cohort member order/index mismatch: expected {expected_index}, got {cohort_index}")
        if source_alias not in partitions or source_row < 0 or source_row >= len(partitions[source_alias].records):
            raise ValueError(f"cohort member {cohort_index} has invalid source coordinate {source_alias}#{source_row}")
        record = partitions[source_alias].records[source_row]
        expected_asset_id = str(raw_member.get("asset_id", ""))
        expected_content_hash = str(raw_member.get("content_hash", ""))
        expected_provenance = raw_member.get("provenance")
        if expected_asset_id != record.container.asset_id or expected_content_hash != record.content_hash:
            raise ValueError(f"cohort member {cohort_index} asset/content identity disagrees with its source record")
        if not isinstance(expected_provenance, Mapping) or dict(expected_provenance) != asdict(record.provenance):
            raise ValueError(f"cohort member {cohort_index} lineage provenance disagrees with its source record")
        actual_mutation_descriptor = _mutation_descriptor(record)
        expected_mutation_descriptor = raw_member.get("mutation_descriptor")
        if schema_version in {HAND_ASSET_COHORT_SCHEMA_VERSION, HAND_ASSET_CANONICAL_COHORT_SCHEMA_VERSION} and (
            not isinstance(expected_mutation_descriptor, Mapping)
            or dict(expected_mutation_descriptor) != actual_mutation_descriptor
        ):
            raise ValueError(f"cohort member {cohort_index} mutation descriptor disagrees with its source record")
        configuration_domain_hash = str(raw_member.get("configuration_domain_hash", ""))
        physical_geometry_hash = str(raw_member.get("physical_geometry_hash", ""))
        canonical_schema_digest = str(raw_member.get("canonical_schema_digest", ""))
        if schema_version == HAND_ASSET_CANONICAL_COHORT_SCHEMA_VERSION:
            for field_name, value in (
                ("configuration_domain_hash", configuration_domain_hash),
                ("physical_geometry_hash", physical_geometry_hash),
                ("canonical_schema_digest", canonical_schema_digest),
            ):
                if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                    raise ValueError(f"cohort member {cohort_index} {field_name} must be lowercase SHA-256")
        member = HandAssetCohortMember(
            cohort_index=cohort_index,
            source_alias=source_alias,
            source_row=source_row,
            asset_id=record.container.asset_id,
            content_hash=record.content_hash,
            provenance=record.provenance,
            mutation_descriptor=(
                actual_mutation_descriptor
                if schema_version in {HAND_ASSET_COHORT_SCHEMA_VERSION, HAND_ASSET_CANONICAL_COHORT_SCHEMA_VERSION}
                else {}
            ),
            configuration_domain_hash=configuration_domain_hash,
            physical_geometry_hash=physical_geometry_hash,
            canonical_schema_digest=canonical_schema_digest,
        )
        bundle_path = record.container.urdf_path.parent.resolve(strict=True)
        if member.source_key in source_keys or member.asset_id in asset_ids or bundle_path in bundle_paths:
            raise ValueError(f"cohort member {cohort_index} duplicates a source key, asset ID or bundle path")
        if member.content_hash and member.content_hash in nonempty_content_hashes:
            raise ValueError(f"cohort member {cohort_index} duplicates content hash {member.content_hash}")
        source_keys.add(member.source_key)
        asset_ids.add(member.asset_id)
        bundle_paths.add(bundle_path)
        if member.content_hash:
            nonempty_content_hashes.add(member.content_hash)
        members.append(member)
        records.append(record)

    if schema_version == HAND_ASSET_CANONICAL_COHORT_SCHEMA_VERSION and len(
        {member.physical_geometry_hash for member in members}
    ) != len(members):
        raise ValueError("canonical cohort lock contains duplicate physical geometry hashes")
    raw_canonical_binding = raw_document.get("canonical_binding", {})
    if not isinstance(raw_canonical_binding, Mapping):
        raise TypeError("cohort canonical_binding must be a mapping")
    if schema_version == HAND_ASSET_CANONICAL_COHORT_SCHEMA_VERSION:
        source_lock_sha256 = str(raw_canonical_binding.get("source_lock_sha256", ""))
        if len(source_lock_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in source_lock_sha256
        ):
            raise ValueError("canonical cohort binding requires source_lock_sha256")
    return ResolvedHandAssetCohort(
        cohort_id=cohort_id,
        lock_path=lock_path,
        lock_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        selection=selection,
        canonical_binding=cast(Mapping[str, Any], raw_canonical_binding),
        sources=sources,
        members=tuple(members),
        partition=ResolvedHandAssetPartition(name=f"cohort:{cohort_id}", records=tuple(records)),
    )


def write_hand_asset_cohort_lock(
    path: str | Path,
    *,
    cohort_id: str,
    source_manifests: Mapping[str, str | Path],
    member_coordinates: Sequence[tuple[str, int]],
    selection: Mapping[str, Any],
    require_geometry_semantics: bool = True,
) -> Path:
    r"""Resolve explicit source coordinates and atomically publish their immutable cohort lock.

    This writer is the final publication boundary for a selector. Selection algorithms remain free to
    construct ``member_coordinates`` from morphology descriptors, while every published member is
    re-resolved and sealed here through the same dataset runtime used by downstream consumers.

    Returns:
        Path: Absolute published lock path.
    """

    cohort_id = str(cohort_id).strip()
    if not cohort_id or not source_manifests or not member_coordinates:
        raise ValueError("cohort writer requires cohort_id, source manifests and member coordinates")
    datasets: dict[str, HandAssetDataset] = {}
    partitions: dict[str, ResolvedHandAssetPartition] = {}
    for raw_alias, manifest_path in source_manifests.items():
        alias = str(raw_alias).strip()
        if not alias or "#" in alias or alias in datasets:
            raise ValueError(f"invalid or duplicate cohort source alias {raw_alias!r}")
        dataset = HandAssetDataset.from_yaml(manifest_path)
        partition, _cache_hit = resolve_prepared_train(
            dataset,
            require_geometry_semantics=require_geometry_semantics,
        )
        datasets[alias] = dataset
        partitions[alias] = partition

    members: list[dict[str, Any]] = []
    for cohort_index, (source_alias, source_row) in enumerate(member_coordinates):
        if source_alias not in partitions or source_row < 0 or source_row >= len(partitions[source_alias].records):
            raise ValueError(f"invalid cohort source coordinate {source_alias}#{source_row}")
        members.append(
            _member_document(
                cohort_index=cohort_index,
                source_alias=source_alias,
                source_row=source_row,
                record=partitions[source_alias].records[source_row],
            )
        )
    document = {
        "schema_version": HAND_ASSET_COHORT_SCHEMA_VERSION,
        "cohort_id": cohort_id,
        "selection": dict(selection),
        "sources": {
            alias: {
                "manifest_path": str(dataset.source_path),
                "manifest_sha256": dataset.source_sha256,
            }
            for alias, dataset in datasets.items()
        },
        "members": members,
    }
    # Canonical compact JSON is valid YAML and gives the lock a stable byte identity across writer runs.
    payload = (json.dumps(document, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode()
    output = resolve_bank_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(output)
    load_hand_asset_cohort(output, require_geometry_semantics=require_geometry_semantics)
    return output


def finalize_hand_asset_cohort_lock(
    source_lock_path: str | Path,
    output_path: str | Path,
    *,
    canonical_identities: Sequence[tuple[str, str, str]],
    canonical_schema_version: str,
    require_geometry_semantics: bool = True,
) -> Path:
    r"""把已lower的physical/schema identities原子写入schema-1.2 canonical-final lock。

    Args:
        source_lock_path (str | Path): 已通过source验证的1.0/1.1 lock；其byte SHA进入父证书。
        output_path (str | Path): 新1.2 lock路径，必须与source lock不同以保留已有证据链。
        canonical_identities (Sequence[tuple[str, str, str]]): 与cohort轴同序的
            ``(configuration_domain_hash, physical_geometry_hash, canonical_schema_digest)`` SHA-256 triples。
        canonical_schema_version (str): Canonical runtime schema版本，例如``1.0.0``。
        require_geometry_semantics (bool): 正式路径必须为true；false仅供最小文件合同fixture。

    Returns:
        Path: 已重新加载并验证source/configuration/physical字段的absolute 1.2 lock路径。
    """

    source = load_hand_asset_cohort(source_lock_path, require_geometry_semantics=require_geometry_semantics)
    source_path = source.lock_path.resolve()
    output = resolve_bank_path(output_path)
    if output.resolve() == source_path:
        raise ValueError("canonical cohort finalization must preserve the source lock at a distinct path")
    identities = tuple(
        (str(configuration), str(physical), str(schema))
        for configuration, physical, schema in canonical_identities
    )
    if len(identities) != len(source.members) or not canonical_schema_version.strip():
        raise ValueError("canonical identities must align with every cohort member and declare a schema version")
    for member_index, (configuration_hash, physical_hash, schema_digest) in enumerate(identities):
        for field_name, value in (
            ("configuration hash", configuration_hash),
            ("physical hash", physical_hash),
            ("schema digest", schema_digest),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"canonical member {member_index} {field_name} must be lowercase SHA-256")
    if len({physical for _configuration, physical, _schema in identities}) != len(identities):
        raise ValueError("canonical cohort finalization rejects duplicate physical geometry hashes")

    raw_document = safe_load(source_path.read_bytes())
    if not isinstance(raw_document, Mapping):
        raise TypeError("source cohort lock must contain a mapping")
    document = dict(raw_document)
    raw_members = document.get("members")
    if not isinstance(raw_members, list) or len(raw_members) != len(identities):
        raise ValueError("source cohort members changed during canonical finalization")
    finalized_members = []
    for member, source_member, (configuration_hash, physical_hash, schema_digest) in zip(
        source.members, raw_members, identities, strict=True
    ):
        if not isinstance(source_member, Mapping):
            raise TypeError("source cohort member must be a mapping")
        finalized = dict(source_member)
        finalized["mutation_descriptor"] = dict(_mutation_descriptor(source.partition.records[member.cohort_index]))
        finalized["configuration_domain_hash"] = configuration_hash
        finalized["physical_geometry_hash"] = physical_hash
        finalized["canonical_schema_digest"] = schema_digest
        finalized_members.append(finalized)
    document["schema_version"] = HAND_ASSET_CANONICAL_COHORT_SCHEMA_VERSION
    document["members"] = finalized_members
    document["canonical_binding"] = {
        "source_lock_path": str(source_path),
        "source_lock_sha256": source.lock_sha256,
        "canonical_schema_version": canonical_schema_version,
        "physical_identity_algorithm": "canonical-runtime-lowering",
    }
    payload = (json.dumps(document, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(output)
    load_hand_asset_cohort(output, require_geometry_semantics=require_geometry_semantics)
    return output


__all__ = [
    "HAND_ASSET_CANONICAL_COHORT_SCHEMA_VERSION",
    "HAND_ASSET_COHORT_SCHEMA_VERSION",
    "HandAssetCohortMember",
    "HandAssetCohortSource",
    "ResolvedHandAssetCohort",
    "finalize_hand_asset_cohort_lock",
    "load_hand_asset_cohort",
    "write_hand_asset_cohort_lock",
]
