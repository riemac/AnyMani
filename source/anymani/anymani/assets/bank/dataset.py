"Resolves typed asset manifests into train, validation, and evaluation partitions without creating runtime environments."

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal, TypeAlias, cast

from .hand_bank import HandBank, HandBankCfg
from .hand_container import HandContainer, HandContainerCfg
from .path_utils import resolve_bank_path
from .yaml_utils import safe_load

HAND_ASSET_DATASET_SCHEMA_VERSION = "2.0.0"
"Version of the train, validation, and evaluation manifest schema."

HandAssetCollectionKind: TypeAlias = Literal["groups", "mixed", "official"]
"Declared source organization for a generated hand collection."


@dataclass(frozen=True)
class HandAssetLineageCfg:
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    include_mother: bool
    variant_sets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        "Normalizes typed inputs and enforces the class invariants."

        if not self.include_mother and not self.variant_sets:
            raise ValueError("one lineage must include its mother or at least one variant set")
        if len(set(self.variant_sets)) != len(self.variant_sets):
            raise ValueError("variant set names must be unique within one mother lineage")
        for name in self.variant_sets:
            _require_relative_component(name, label="variant set")


HandAssetMotherMap: TypeAlias = Mapping[str, HandAssetLineageCfg]
HandAssetGroupMap: TypeAlias = Mapping[str, HandAssetMotherMap]


@dataclass(frozen=True)
class HandAssetRunCfg:
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    run_dir: str = ""
    groups: HandAssetGroupMap = field(default_factory=dict)  # `<run>/<production_group>/<mother>`
    mixed: HandAssetGroupMap = field(default_factory=dict)  # `<run>/mixed/<composition_group>/<mother>`

    def __post_init__(self) -> None:
        "Normalizes typed inputs and enforces the class invariants."

        if not self.groups and not self.mixed:
            raise ValueError("one dataset run block must contain groups or mixed lineages")
        for collection_kind, group_map in (("groups", self.groups), ("mixed", self.mixed)):
            for group_name, mothers in group_map.items():
                _require_relative_component(group_name, label=f"{collection_kind} group")
                if not mothers:
                    raise ValueError(f"dataset {collection_kind} group {group_name!r} cannot be empty")
                for mother_name in mothers:
                    _require_relative_component(mother_name, label="mother")


@dataclass(frozen=True)
class HandAssetPartitionCfg:
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    runs: Mapping[str, HandAssetRunCfg] = field(default_factory=dict)

    def __post_init__(self) -> None:
        "Normalizes typed inputs and enforces the class invariants."

        for run_alias in self.runs:
            if not str(run_alias).strip():
                raise ValueError("dataset run alias cannot be empty")


@dataclass(frozen=True)
class HandAssetOfficialPartitionCfg:
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    assets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        "Normalizes typed inputs and enforces the class invariants."

        if any(not path.strip() for path in self.assets):
            raise ValueError("official asset paths cannot be empty")
        if len(set(self.assets)) != len(self.assets):
            raise ValueError("official asset paths must be unique")


@dataclass(frozen=True)
class HandAssetValidationCfg:
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    unseen_variant_set: HandAssetPartitionCfg = field(default_factory=HandAssetPartitionCfg)
    unseen_mother: HandAssetPartitionCfg = field(default_factory=HandAssetPartitionCfg)


@dataclass(frozen=True)
class HandAssetEvaluationCfg:
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    unseen_variant_set: HandAssetPartitionCfg = field(default_factory=HandAssetPartitionCfg)
    unseen_mother: HandAssetPartitionCfg = field(default_factory=HandAssetPartitionCfg)
    official_zero_shot: HandAssetOfficialPartitionCfg = field(default_factory=HandAssetOfficialPartitionCfg)


@dataclass(frozen=True)
class HandAssetDatasetCfg:
    "Partition manifest config for generated lineages, validation cohorts, evaluation cohorts, and optional official hands."

    schema_version: str = HAND_ASSET_DATASET_SCHEMA_VERSION  # persisted YAML contract
    default_run_dir: str = ""
    train: HandAssetPartitionCfg = field(default_factory=HandAssetPartitionCfg)
    validation: HandAssetValidationCfg = field(default_factory=HandAssetValidationCfg)
    evaluation: HandAssetEvaluationCfg = field(default_factory=HandAssetEvaluationCfg)

    def __post_init__(self) -> None:
        "Normalizes typed inputs and enforces the class invariants."

        if self.schema_version != HAND_ASSET_DATASET_SCHEMA_VERSION:
            raise ValueError(f"hand asset dataset schema must be exactly {HAND_ASSET_DATASET_SCHEMA_VERSION!r}")
        if not self.train.runs:
            raise ValueError("hand asset dataset requires a non-empty train partition")


@dataclass(frozen=True)
class HandAssetProvenance:
    "Source partition, run, group, mother, variant set, and asset role for one hand."

    partition: str  # train / validation / evaluation suite name
    run_alias: str
    run_dir: str
    collection_kind: HandAssetCollectionKind  # groups / mixed / official
    group_name: str
    mother_name: str
    mother_path: str
    variant_set: str
    asset_role: Literal["mother", "variant", "official"]


@dataclass(frozen=True)
class ResolvedHandAssetRecord:
    "Validated hand container joined with source provenance and a content hash."

    container: HandContainer
    provenance: HandAssetProvenance
    content_hash: str = ""


@dataclass(frozen=True)
class ResolvedHandAssetPartition:
    "Ordered validated records for one named train, validation, or evaluation partition."

    name: str
    records: tuple[ResolvedHandAssetRecord, ...]

    @property
    def assets(self) -> tuple[HandContainer, ...]:
        "Returns the ordered validated hand assets for this partition."

        return tuple(record.container for record in self.records)


@dataclass(frozen=True)
class ResolvedHandAssetDataset:
    "Resolved dataset partitions after duplicate and holdout-isolation checks."

    source_path: Path
    source_sha256: str
    config: HandAssetDatasetCfg
    train: ResolvedHandAssetPartition
    validation: Mapping[str, ResolvedHandAssetPartition]
    evaluation: Mapping[str, ResolvedHandAssetPartition]

    def config_dict(self) -> dict[str, Any]:
        "Returns the resolved dataset configuration as a JSON-safe mapping."

        return asdict(self.config)


class HandAssetDataset:
    "Typed manifest loader and resolver for generated and official hand assets."

    def __init__(self, config: HandAssetDatasetCfg, *, source_path: Path, source_sha256: str) -> None:
        "Stores the typed manifest config and exact source-file hash for later partition resolution."

        self.config = config
        self.source_path = source_path
        self.source_sha256 = source_sha256

    @classmethod
    def from_yaml(cls, path: str | Path) -> HandAssetDataset:
        'Constructs yaml.'

        resolved_path = resolve_bank_path(path)
        if not resolved_path.is_file():
            raise FileNotFoundError(f"hand asset dataset manifest does not exist: {resolved_path}")
        raw_bytes = resolved_path.read_bytes()
        document = safe_load(raw_bytes) or {}
        if not isinstance(document, Mapping):
            raise TypeError("hand asset dataset YAML root must be a mapping")
        config = _dataset_cfg_from_mapping(document)
        return cls(
            config,
            source_path=resolved_path,
            source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        )

    def resolve(
        self,
        *,
        require_geometry_semantics: bool = False,
        allow_legacy_left_handedness: bool = False,
    ) -> ResolvedHandAssetDataset:
        "Resolves declared source paths, typed sidecars, and the required mesh closure."

        train = self._resolve_generated_partition(
            self.config.train,
            partition_name="train",
            require_geometry_semantics=require_geometry_semantics,
            allow_legacy_left_handedness=allow_legacy_left_handedness,
        )
        validation_unseen_variant_set = self._resolve_generated_partition(
            self.config.validation.unseen_variant_set,
            partition_name="validation.unseen_variant_set",
            require_geometry_semantics=require_geometry_semantics,
            allow_legacy_left_handedness=allow_legacy_left_handedness,
        )
        validation_unseen_mother = self._resolve_generated_partition(
            self.config.validation.unseen_mother,
            partition_name="validation.unseen_mother",
            require_geometry_semantics=require_geometry_semantics,
            allow_legacy_left_handedness=allow_legacy_left_handedness,
        )
        validation = {
            "unseen_variant_set": validation_unseen_variant_set,
            "unseen_mother": validation_unseen_mother,
        }
        unseen_variant_set = self._resolve_generated_partition(
            self.config.evaluation.unseen_variant_set,
            partition_name="evaluation.unseen_variant_set",
            require_geometry_semantics=require_geometry_semantics,
            allow_legacy_left_handedness=allow_legacy_left_handedness,
        )
        unseen_mother = self._resolve_generated_partition(
            self.config.evaluation.unseen_mother,
            partition_name="evaluation.unseen_mother",
            require_geometry_semantics=require_geometry_semantics,
            allow_legacy_left_handedness=allow_legacy_left_handedness,
        )
        official = self._resolve_official_partition(
            self.config.evaluation.official_zero_shot,
            require_geometry_semantics=require_geometry_semantics,
        )
        evaluation = {
            "unseen_variant_set": unseen_variant_set,
            "unseen_mother": unseen_mother,
            "official_zero_shot": official,
        }



        all_partitions = (train, *validation.values(), *evaluation.values())
        _validate_unique_asset_records(all_partitions)
        _validate_named_suite_relations(
            train,
            validation_unseen_variant_set=validation_unseen_variant_set,
            validation_unseen_mother=validation_unseen_mother,
            evaluation_unseen_variant_set=unseen_variant_set,
            evaluation_unseen_mother=unseen_mother,
        )
        return ResolvedHandAssetDataset(
            source_path=self.source_path,
            source_sha256=self.source_sha256,
            config=self.config,
            train=train,
            validation=validation,
            evaluation=evaluation,
        )

    def resolve_train(
        self,
        *,
        require_geometry_semantics: bool = False,
        allow_legacy_left_handedness: bool = False,
        max_assets: int | None = None,
    ) -> ResolvedHandAssetPartition:
        'Resolves train.'

        if max_assets is not None and max_assets < 1:
            raise ValueError("max_assets must be positive when provided")
        train = self._resolve_generated_partition(
            self.config.train,
            partition_name="train",
            require_geometry_semantics=require_geometry_semantics,
            allow_legacy_left_handedness=allow_legacy_left_handedness,
            max_records=max_assets,
        )
        _validate_unique_asset_records((train,))
        return train

    def resolve_evaluation(
        self,
        *,
        require_geometry_semantics: bool = False,
        allow_legacy_left_handedness: bool = False,
    ) -> Mapping[str, ResolvedHandAssetPartition]:
        'Resolves evaluation.'

        unseen_variant_set = self._resolve_generated_partition(
            self.config.evaluation.unseen_variant_set,
            partition_name="evaluation.unseen_variant_set",
            require_geometry_semantics=require_geometry_semantics,
            allow_legacy_left_handedness=allow_legacy_left_handedness,
        )
        unseen_mother = self._resolve_generated_partition(
            self.config.evaluation.unseen_mother,
            partition_name="evaluation.unseen_mother",
            require_geometry_semantics=require_geometry_semantics,
            allow_legacy_left_handedness=allow_legacy_left_handedness,
        )
        official = self._resolve_official_partition(
            self.config.evaluation.official_zero_shot,
            require_geometry_semantics=require_geometry_semantics,
        )
        evaluation = {
            "unseen_variant_set": unseen_variant_set,
            "unseen_mother": unseen_mother,
            "official_zero_shot": official,
        }


        train_mothers = _declared_generated_mothers(self.config.train, default_run_dir=self.config.default_run_dir)
        validation_seen = _declared_generated_mothers(
            self.config.validation.unseen_variant_set,
            default_run_dir=self.config.default_run_dir,
        )
        validation_unseen = _declared_generated_mothers(
            self.config.validation.unseen_mother,
            default_run_dir=self.config.default_run_dir,
        )
        evaluation_seen = _validate_unseen_variant_set(unseen_variant_set, train_mothers=train_mothers)
        overlap = evaluation_seen & validation_seen
        if overlap:
            raise ValueError(f"validation/evaluation unseen_variant_set mothers overlap: {tuple(sorted(overlap))}")
        _validate_unseen_mother(
            unseen_mother,
            forbidden_mothers=train_mothers | validation_unseen,
        )
        _validate_unique_asset_records(tuple(evaluation.values()))
        return evaluation

    def _resolve_generated_partition(
        self,
        config: HandAssetPartitionCfg,
        *,
        partition_name: str,
        require_geometry_semantics: bool,
        allow_legacy_left_handedness: bool,
        max_records: int | None = None,
    ) -> ResolvedHandAssetPartition:

        jobs: list[dict[str, Any]] = []
        for run_alias, run_config in config.runs.items():
            run_dir = run_config.run_dir or self.config.default_run_dir
            if not run_dir:
                raise ValueError(f"dataset run {run_alias!r} requires run_dir or default_run_dir")
            run_root = resolve_bank_path(run_dir)
            _validate_generation_run(run_root)
            for raw_collection_kind, group_map in (("groups", run_config.groups), ("mixed", run_config.mixed)):
                collection_kind = cast(Literal["groups", "mixed"], raw_collection_kind)
                for group_name, mothers in group_map.items():
                    group_root = (
                        run_root / group_name if collection_kind == "groups" else run_root / "mixed" / group_name
                    )
                    for mother_name, lineage in mothers.items():
                        mother_root = (group_root / mother_name).resolve(strict=False)
                        jobs.append(
                            {
                                "mother_root": mother_root,
                                "lineage": lineage,
                                "partition_name": partition_name,
                                "run_alias": str(run_alias),
                                "run_root": run_root,
                                "collection_kind": collection_kind,
                                "group_name": str(group_name),
                                "mother_name": str(mother_name),
                                "require_geometry_semantics": require_geometry_semantics,
                                "allow_legacy_left_handedness": allow_legacy_left_handedness,
                            }
                        )
        if max_records is not None:

            resolved_lineages = []
            resolved_count = 0
            for job in jobs:
                lineage_records = _resolve_generated_lineage_job(job)
                resolved_lineages.append(lineage_records)
                resolved_count += len(lineage_records)
                if resolved_count >= max_records:
                    break
        elif len(jobs) < 2:
            resolved_lineages = [_resolve_generated_lineage_job(jobs[0])] if jobs else []
        else:
            worker_count = min(8, max(1, (os.cpu_count() or 2) // 2), len(jobs))
            print(
                f"[Assets] Resolving partition={partition_name!r}: "
                f"{len(jobs)} lineages with {worker_count} CPU workers"
            )
            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="asset-resolve") as executor:
                futures = [executor.submit(_resolve_generated_lineage_job, job) for job in jobs]
                resolved_lineages = [future.result() for future in futures]
        records = [record for lineage_records in resolved_lineages for record in lineage_records]
        if max_records is not None:
            records = records[:max_records]
        return ResolvedHandAssetPartition(name=partition_name, records=tuple(records))

    def _resolve_official_partition(
        self,
        config: HandAssetOfficialPartitionCfg,
        *,
        require_geometry_semantics: bool,
    ) -> ResolvedHandAssetPartition:

        records: list[ResolvedHandAssetRecord] = []
        for path in config.assets:
            container = _resolve_one_container(
                HandContainerCfg(path=path, source_kind="official"),
                require_geometry_semantics=require_geometry_semantics,
                allow_legacy_left_handedness=False,
            )
            records.append(
                ResolvedHandAssetRecord(
                    container=container,
                    provenance=HandAssetProvenance(
                        partition="evaluation.official_zero_shot",
                        run_alias="official",
                        run_dir="",
                        collection_kind="official",
                        group_name="",
                        mother_name="",
                        mother_path="",
                        variant_set="",
                        asset_role="official",
                    ),
                    content_hash=_container_content_hash(container),
                )
            )
        return ResolvedHandAssetPartition(name="evaluation.official_zero_shot", records=tuple(records))


def _resolve_generated_lineage_job(job: Mapping[str, Any]) -> tuple[ResolvedHandAssetRecord, ...]:

    return _resolve_generated_lineage(
        job["mother_root"],
        job["lineage"],
        partition_name=job["partition_name"],
        run_alias=job["run_alias"],
        run_root=job["run_root"],
        collection_kind=job["collection_kind"],
        group_name=job["group_name"],
        mother_name=job["mother_name"],
        require_geometry_semantics=job["require_geometry_semantics"],
        allow_legacy_left_handedness=job["allow_legacy_left_handedness"],
    )


def _resolve_generated_lineage(
    mother_root: Path,
    lineage: HandAssetLineageCfg,
    *,
    partition_name: str,
    run_alias: str,
    run_root: Path,
    collection_kind: Literal["groups", "mixed"],
    group_name: str,
    mother_name: str,
    require_geometry_semantics: bool,
    allow_legacy_left_handedness: bool,
) -> tuple[ResolvedHandAssetRecord, ...]:

    if not (mother_root / "hand.urdf").is_file() or not (mother_root / "hand.yaml").is_file():
        raise FileNotFoundError(f"dataset mother bundle is incomplete: {mother_root}")
    records: list[ResolvedHandAssetRecord] = []
    if lineage.include_mother:
        mother_container = _resolve_one_container(
            HandContainerCfg(path=mother_root, source_kind="generated"),
            require_geometry_semantics=require_geometry_semantics,
            allow_legacy_left_handedness=allow_legacy_left_handedness,
        )
        records.append(
            _resolved_record(
                mother_container,
                partition_name=partition_name,
                run_alias=run_alias,
                run_root=run_root,
                collection_kind=collection_kind,
                group_name=group_name,
                mother_name=mother_name,
                mother_root=mother_root,
                variant_set="",
                asset_role="mother",
            )
        )


    for variant_set_name in lineage.variant_sets:
        variant_set_root = (mother_root / variant_set_name).resolve(strict=False)
        _validate_variant_set(variant_set_root, mother_root)
        selection = HandBank(
            HandBankCfg(
                source_mode="post_mutate",
                selection_mode="all",
                post_mutate_path=variant_set_root,
                include_source_topology=False,
                require_geometry_semantics=require_geometry_semantics,
                allow_legacy_left_handedness=allow_legacy_left_handedness,
            )
        ).resolve()
        if not selection.assets:
            raise ValueError(f"dataset variant set contains no hand variants: {variant_set_root}")
        records.extend(
            _resolved_record(
                container,
                partition_name=partition_name,
                run_alias=run_alias,
                run_root=run_root,
                collection_kind=collection_kind,
                group_name=group_name,
                mother_name=mother_name,
                mother_root=mother_root,
                variant_set=variant_set_name,
                asset_role="variant",
            )
            for container in selection.assets
        )
    return tuple(records)


def _resolve_one_container(
    config: HandContainerCfg,
    *,
    require_geometry_semantics: bool,
    allow_legacy_left_handedness: bool,
) -> HandContainer:

    return (
        HandBank(
            HandBankCfg(
                source_mode="mixed",
                selection_mode="explicit",
                containers=(config,),
                require_geometry_semantics=require_geometry_semantics,
                allow_legacy_left_handedness=allow_legacy_left_handedness,
            )
        )
        .resolve()
        .assets[0]
    )


def _resolved_record(
    container: HandContainer,
    *,
    partition_name: str,
    run_alias: str,
    run_root: Path,
    collection_kind: Literal["groups", "mixed"],
    group_name: str,
    mother_name: str,
    mother_root: Path,
    variant_set: str,
    asset_role: Literal["mother", "variant"],
) -> ResolvedHandAssetRecord:

    return ResolvedHandAssetRecord(
        container=container,
        provenance=HandAssetProvenance(
            partition=partition_name,
            run_alias=run_alias,
            run_dir=str(run_root.resolve(strict=False)),
            collection_kind=collection_kind,
            group_name=group_name,
            mother_name=mother_name,
            mother_path=str(mother_root.resolve(strict=False)),
            variant_set=variant_set,
            asset_role=asset_role,
        ),
        content_hash=_container_content_hash(container),
    )


def _validate_generation_run(run_root: Path) -> None:

    summary_path = run_root / "summary.yaml"
    summary = _load_yaml_mapping(summary_path, label="generation run summary")
    run = summary.get("run")
    if not isinstance(run, Mapping) or run.get("mode") != "made":
        raise ValueError(f"dataset run_dir must contain a mode='made' summary: {summary_path}")


def _validate_variant_set(variant_set_root: Path, mother_root: Path) -> None:

    summary_path = variant_set_root / "summary.yaml"
    summary = _load_yaml_mapping(summary_path, label="variant set summary")
    run = summary.get("run")
    if not isinstance(run, Mapping) or run.get("mode") != "mutate":
        raise ValueError(f"variant set must contain a mode='mutate' summary: {summary_path}")
    config = summary.get("config")
    source = config.get("source_topology_dir") if isinstance(config, Mapping) else None
    if not isinstance(source, str) or resolve_bank_path(source) != mother_root.resolve(strict=False):
        raise ValueError(
            "variant set source_topology_dir does not match its declared mother: "
            f"source={source!r}, mother={mother_root}"
        )
    variant_dirs = tuple(
        child for child in variant_set_root.iterdir() if child.is_dir() and (child / "hand.urdf").is_file()
    )
    stats = summary.get("stats")
    succeeded = stats.get("succeeded") if isinstance(stats, Mapping) else None
    if succeeded is not None and int(succeeded) != len(variant_dirs):
        raise ValueError(
            f"variant set summary succeeded={succeeded} does not match discovered variants={len(variant_dirs)}"
        )


def _validate_unique_asset_records(partitions: Sequence[ResolvedHandAssetPartition]) -> None:

    seen_paths: dict[Path, str] = {}
    seen_ids: dict[str, str] = {}
    seen_content: dict[str, str] = {}
    for partition in partitions:
        for record in partition.records:
            bundle_path = record.container.urdf_path.parent.resolve(strict=False)
            _record_unique_identity(bundle_path, seen_paths, label="bundle path", partition=partition.name)
            _record_unique_identity(record.container.asset_id, seen_ids, label="asset ID", partition=partition.name)
            if record.content_hash:
                _record_unique_identity(
                    record.content_hash,
                    seen_content,
                    label="content hash",
                    partition=partition.name,
                )


def _record_unique_identity(
    identity: str | Path,
    seen: dict[str, str] | dict[Path, str],
    *,
    label: str,
    partition: str,
) -> None:

    if isinstance(identity, Path):
        path_seen = cast(dict[Path, str], seen)
        previous = path_seen.get(identity)
        if previous is not None:
            raise ValueError(
                f"hand asset dataset {label} leaks across roles: {identity!r} in {previous!r} and {partition!r}"
            )
        path_seen[identity] = partition
        return
    string_seen = cast(dict[str, str], seen)
    previous = string_seen.get(identity)
    if previous is not None:
        raise ValueError(
            f"hand asset dataset {label} leaks across roles: {identity!r} in {previous!r} and {partition!r}"
        )
    string_seen[identity] = partition


def _validate_named_suite_relations(
    train: ResolvedHandAssetPartition,
    *,
    validation_unseen_variant_set: ResolvedHandAssetPartition,
    validation_unseen_mother: ResolvedHandAssetPartition,
    evaluation_unseen_variant_set: ResolvedHandAssetPartition,
    evaluation_unseen_mother: ResolvedHandAssetPartition,
) -> None:

    train_mothers = {record.provenance.mother_path for record in train.records if record.provenance.mother_path}
    validation_seen_mothers = _validate_unseen_variant_set(
        validation_unseen_variant_set,
        train_mothers=train_mothers,
    )
    evaluation_seen_mothers = _validate_unseen_variant_set(
        evaluation_unseen_variant_set,
        train_mothers=train_mothers,
    )
    overlap = validation_seen_mothers & evaluation_seen_mothers
    if overlap:
        raise ValueError(f"validation/evaluation unseen_variant_set mothers overlap: {tuple(sorted(overlap))}")

    validation_unseen_mothers = _validate_unseen_mother(
        validation_unseen_mother,
        forbidden_mothers=train_mothers,
    )
    _validate_unseen_mother(
        evaluation_unseen_mother,
        forbidden_mothers=train_mothers | validation_unseen_mothers,
    )


def _declared_generated_mothers(
    partition: HandAssetPartitionCfg,
    *,
    default_run_dir: str,
) -> set[str]:

    mothers: set[str] = set()
    for run_config in partition.runs.values():
        run_dir = run_config.run_dir or default_run_dir
        if not run_dir:
            raise ValueError("dataset run requires run_dir or default_run_dir")
        run_root = resolve_bank_path(run_dir)
        for collection_kind, group_map in (("groups", run_config.groups), ("mixed", run_config.mixed)):
            for group_name, lineages in group_map.items():
                group_root = run_root / group_name if collection_kind == "groups" else run_root / "mixed" / group_name
                mothers.update(str((group_root / mother_name).resolve(strict=False)) for mother_name in lineages)
    return mothers


def _validate_unseen_variant_set(
    partition: ResolvedHandAssetPartition,
    *,
    train_mothers: set[str],
) -> set[str]:

    mothers: set[str] = set()
    for record in partition.records:
        provenance = record.provenance
        if provenance.asset_role != "variant":
            raise ValueError("unseen_variant_set may contain variants only; mother inclusion must be false")
        if provenance.mother_path not in train_mothers:
            raise ValueError(f"unseen_variant_set mother is absent from train: {provenance.mother_path}")
        mothers.add(provenance.mother_path)
    return mothers


def _validate_unseen_mother(
    partition: ResolvedHandAssetPartition,
    *,
    forbidden_mothers: set[str],
) -> set[str]:

    mothers = {record.provenance.mother_path for record in partition.records if record.provenance.mother_path}
    overlap = mothers & forbidden_mothers
    if overlap:
        raise ValueError(f"unseen_mother already appears in train or validation: {tuple(sorted(overlap))}")
    return mothers


def _container_content_hash(container: HandContainer) -> str:

    if container.geometry_semantics is not None:
        return container.geometry_semantics.content_hash
    raw_semantics = container.sidecar.get("geometry_semantics")
    if isinstance(raw_semantics, Mapping):
        return str(raw_semantics.get("content_hash") or "")
    return ""


def _dataset_cfg_from_mapping(document: Mapping[str, Any]) -> HandAssetDatasetCfg:

    _require_keys(
        document,
        allowed={"schema_version", "default_run_dir", "train", "validation", "evaluation"},
        required={"schema_version", "default_run_dir", "train", "evaluation"},
        context="dataset",
    )
    schema_version = str(document["schema_version"])
    if schema_version != HAND_ASSET_DATASET_SCHEMA_VERSION:
        raise ValueError(f"hand asset dataset schema must be exactly {HAND_ASSET_DATASET_SCHEMA_VERSION!r}")

    validation_raw = _as_mapping(document.get("validation", {}), context="validation")
    _require_keys(
        validation_raw,
        allowed={"unseen_variant_set", "unseen_mother"},
        required=set(),
        context="validation",
    )
    evaluation_raw = _as_mapping(document.get("evaluation", {}), context="evaluation")
    _require_keys(
        evaluation_raw,
        allowed={"unseen_variant_set", "unseen_mother", "official_zero_shot"},
        required=set(),
        context="evaluation",
    )
    official_raw = _as_mapping(evaluation_raw.get("official_zero_shot", {}), context="official_zero_shot")
    _require_keys(official_raw, allowed={"assets"}, required=set(), context="official_zero_shot")
    official_assets = _string_tuple(official_raw.get("assets", ()), context="official_zero_shot.assets")
    return HandAssetDatasetCfg(
        schema_version=schema_version,
        default_run_dir=str(document["default_run_dir"]),
        train=_partition_cfg_from_mapping(document["train"], context="train"),
        validation=HandAssetValidationCfg(
            unseen_variant_set=_partition_cfg_from_mapping(
                validation_raw.get("unseen_variant_set", {}), context="validation.unseen_variant_set"
            ),
            unseen_mother=_partition_cfg_from_mapping(
                validation_raw.get("unseen_mother", {}), context="validation.unseen_mother"
            ),
        ),
        evaluation=HandAssetEvaluationCfg(
            unseen_variant_set=_partition_cfg_from_mapping(
                evaluation_raw.get("unseen_variant_set", {}), context="evaluation.unseen_variant_set"
            ),
            unseen_mother=_partition_cfg_from_mapping(
                evaluation_raw.get("unseen_mother", {}), context="evaluation.unseen_mother"
            ),
            official_zero_shot=HandAssetOfficialPartitionCfg(assets=official_assets),
        ),
    )


def _partition_cfg_from_mapping(value: Any, *, context: str) -> HandAssetPartitionCfg:

    payload = _as_mapping(value, context=context)
    _require_keys(payload, allowed={"runs"}, required=set(), context=context)
    runs_raw = _as_mapping(payload.get("runs", {}), context=f"{context}.runs")
    return HandAssetPartitionCfg(
        runs={
            str(alias): _run_cfg_from_mapping(run, context=f"{context}.runs.{alias}") for alias, run in runs_raw.items()
        }
    )


def _run_cfg_from_mapping(value: Any, *, context: str) -> HandAssetRunCfg:

    payload = _as_mapping(value, context=context)
    _require_keys(payload, allowed={"run_dir", "groups", "mixed"}, required=set(), context=context)
    return HandAssetRunCfg(
        run_dir=str(payload.get("run_dir", "")),
        groups=_group_map_from_mapping(payload.get("groups", {}), context=f"{context}.groups"),
        mixed=_group_map_from_mapping(payload.get("mixed", {}), context=f"{context}.mixed"),
    )


def _group_map_from_mapping(value: Any, *, context: str) -> dict[str, dict[str, HandAssetLineageCfg]]:

    groups = _as_mapping(value, context=context)
    parsed: dict[str, dict[str, HandAssetLineageCfg]] = {}
    for group_name, mothers_value in groups.items():
        mothers = _as_mapping(mothers_value, context=f"{context}.{group_name}")
        parsed[str(group_name)] = {
            str(mother_name): _lineage_cfg_from_mapping(
                lineage_value,
                context=f"{context}.{group_name}.{mother_name}",
            )
            for mother_name, lineage_value in mothers.items()
        }
    return parsed


def _lineage_cfg_from_mapping(value: Any, *, context: str) -> HandAssetLineageCfg:

    payload = _as_mapping(value, context=context)
    _require_keys(payload, allowed={"include_mother", "variant_sets"}, required={"include_mother"}, context=context)
    include_mother = payload["include_mother"]
    if not isinstance(include_mother, bool):
        raise TypeError(f"{context}.include_mother must be bool")
    return HandAssetLineageCfg(
        include_mother=include_mother,
        variant_sets=_string_tuple(payload.get("variant_sets", ()), context=f"{context}.variant_sets"),
    )


def _as_mapping(value: Any, *, context: str) -> Mapping[str, Any]:

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    return value


def _string_tuple(value: Any, *, context: str) -> tuple[str, ...]:

    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{context} must be a sequence of strings")
    result = tuple(str(item) for item in value)
    if any(not item.strip() for item in result):
        raise ValueError(f"{context} cannot contain empty strings")
    return result


def _require_keys(
    payload: Mapping[str, Any],
    *,
    allowed: set[str],
    required: set[str],
    context: str,
) -> None:

    keys = {str(key) for key in payload}
    unknown = keys - allowed
    missing = required - keys
    if unknown:
        raise ValueError(f"{context} contains unknown fields: {tuple(sorted(unknown))}")
    if missing:
        raise ValueError(f"{context} is missing required fields: {tuple(sorted(missing))}")


def _require_relative_component(value: str, *, label: str) -> None:

    path = PurePosixPath(str(value))
    if not value or path.is_absolute() or len(path.parts) != 1 or path.parts[0] in {".", ".."}:
        raise ValueError(f"dataset {label} must be one relative path component: {value!r}")


def _load_yaml_mapping(path: Path, *, label: str) -> Mapping[str, Any]:

    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    document = safe_load(path.read_bytes()) or {}
    if not isinstance(document, Mapping):
        raise TypeError(f"{label} must be a mapping: {path}")
    return document


__all__ = [
    "HAND_ASSET_DATASET_SCHEMA_VERSION",
    "HandAssetDataset",
    "HandAssetDatasetCfg",
    "HandAssetEvaluationCfg",
    "HandAssetLineageCfg",
    "HandAssetOfficialPartitionCfg",
    "HandAssetPartitionCfg",
    "HandAssetProvenance",
    "HandAssetRunCfg",
    "HandAssetValidationCfg",
    "ResolvedHandAssetDataset",
    "ResolvedHandAssetPartition",
    "ResolvedHandAssetRecord",
]
