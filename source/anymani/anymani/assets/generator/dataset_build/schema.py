"Defines the strict typed YAML schema for mother selection, lineage counts, partitions, and retry policy."

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import yaml

DATASET_BUILD_TEMPLATE_SCHEMA_VERSION = "1.0.0"
"Version of the typed dataset-generation template contract."


@dataclass(frozen=True)
class DatasetInventoryCfg:
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    run_dir: str

    def __post_init__(self) -> None:
        "Normalizes typed inputs and enforces the class invariants."

        if not self.run_dir.strip():
            raise ValueError("dataset build inventory.run_dir cannot be empty")


@dataclass(frozen=True)
class DatasetBuildSeedsCfg:
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    selection: int
    mutation: int


@dataclass(frozen=True)
class DatasetBalanceCfg:
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    selection_unit: Literal["canonical_mirror_pair"] = "canonical_mirror_pair"
    macro_family: Mapping[str, float] = field(default_factory=dict)
    topology_shape: Mapping[str, float] = field(default_factory=dict)
    missing_slot: Literal["uniform"] = "uniform"
    mixed_composition_group: Literal["uniform"] = "uniform"
    dof: Literal["uniform_available"] = "uniform_available"

    def __post_init__(self) -> None:
        "Normalizes typed inputs and enforces the class invariants."

        if self.selection_unit != "canonical_mirror_pair":
            raise ValueError("balance.selection_unit must be canonical_mirror_pair")
        if self.missing_slot != "uniform" or self.mixed_composition_group != "uniform":
            raise ValueError("missing-slot and mixed-composition balancing must be uniform")
        if self.dof != "uniform_available":
            raise ValueError("balance.dof must be uniform_available")
        expected_macro = {
            "single_allegro",
            "single_leap",
            "mixed_allegro_base",
            "mixed_leap_base",
        }
        if set(self.macro_family) != expected_macro:
            raise ValueError(f"balance.macro_family must contain exactly {tuple(sorted(expected_macro))}")
        if set(self.topology_shape) != {"full", "missing"}:
            raise ValueError("balance.topology_shape must contain exactly full and missing")
        for name, weight in (*self.macro_family.items(), *self.topology_shape.items()):
            if float(weight) <= 0.0:
                raise ValueError(f"dataset balance weight must be positive: {name}={weight}")


@dataclass(frozen=True)
class DatasetRoleCfg:
    "Mirror-pair count and final number of assets contributed by each selected lineage."

    mother_count: int
    assets_per_lineage: int

    def __post_init__(self) -> None:
        "Normalizes typed inputs and enforces the class invariants."

        if self.mother_count < 2 or self.mother_count % 2 != 0:
            raise ValueError("dataset role mother_count must be a positive even number")
        if self.assets_per_lineage < 1:
            raise ValueError("dataset role assets_per_lineage must be >= 1")


@dataclass(frozen=True)
class DatasetValidationTemplateCfg:
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    unseen_variant_set: DatasetRoleCfg
    unseen_mother: DatasetRoleCfg


@dataclass(frozen=True)
class DatasetEvaluationTemplateCfg:
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    unseen_variant_set: DatasetRoleCfg
    unseen_mother: DatasetRoleCfg
    official_zero_shot: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        "Normalizes typed inputs and enforces the class invariants."

        if any(not path.strip() for path in self.official_zero_shot):
            raise ValueError("official_zero_shot asset paths cannot be empty")
        if len(set(self.official_zero_shot)) != len(self.official_zero_shot):
            raise ValueError("official_zero_shot asset paths must be unique")


@dataclass(frozen=True)
class DatasetPartitionsTemplateCfg:
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    train: DatasetRoleCfg
    validation: DatasetValidationTemplateCfg
    evaluation: DatasetEvaluationTemplateCfg


@dataclass(frozen=True)
class DatasetGenerationPolicyCfg:
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    dataset_retry_rounds: int = 3
    uniqueness: Literal["resample"] = "resample"
    failed_run_policy: Literal["quarantine"] = "quarantine"

    def __post_init__(self) -> None:
        "Normalizes typed inputs and enforces the class invariants."

        if self.dataset_retry_rounds < 1:
            raise ValueError("generation_policy.dataset_retry_rounds must be >= 1")
        if self.uniqueness != "resample" or self.failed_run_policy != "quarantine":
            raise ValueError("generation policy requires uniqueness=resample and failed_run_policy=quarantine")


@dataclass(frozen=True)
class DatasetPpoManifestCfg:
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    enabled: bool = True
    train_mother_count: int = 128
    reuse_ssl_holdouts: bool = True

    def __post_init__(self) -> None:
        "Normalizes typed inputs and enforces the class invariants."

        if self.enabled and (self.train_mother_count < 2 or self.train_mother_count % 2 != 0):
            raise ValueError("ppo train_mother_count must be a positive even number")


@dataclass(frozen=True)
class DatasetManifestsCfg:
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    ssl_enabled: bool = True
    ppo: DatasetPpoManifestCfg = field(default_factory=DatasetPpoManifestCfg)


@dataclass(frozen=True)
class DatasetBuildTemplateCfg:
    "Typed dataset plan containing a pre-made inventory, deterministic seeds, morphology balance, partitions, and retry rules."

    schema_version: str
    template_id: str
    inventory: DatasetInventoryCfg
    seeds: DatasetBuildSeedsCfg
    balance: DatasetBalanceCfg
    partitions: DatasetPartitionsTemplateCfg
    generation_policy: DatasetGenerationPolicyCfg
    manifests: DatasetManifestsCfg

    def __post_init__(self) -> None:
        "Normalizes typed inputs and enforces the class invariants."

        if self.schema_version != DATASET_BUILD_TEMPLATE_SCHEMA_VERSION:
            raise ValueError(f"dataset build template schema must be exactly {DATASET_BUILD_TEMPLATE_SCHEMA_VERSION!r}")
        if not self.template_id.strip():
            raise ValueError("dataset build template_id cannot be empty")
        if self.manifests.ppo.enabled and self.manifests.ppo.train_mother_count > self.partitions.train.mother_count:
            raise ValueError("PPO train mother count cannot exceed SSL train mother count")
        seen_total = (
            self.partitions.validation.unseen_variant_set.mother_count
            + self.partitions.evaluation.unseen_variant_set.mother_count
        )
        if self.manifests.ppo.enabled and self.manifests.ppo.reuse_ssl_holdouts:
            if self.manifests.ppo.train_mother_count < seen_total:
                raise ValueError("PPO train must contain all validation/evaluation seen-mother cohorts")


def load_dataset_build_template(path: str | Path) -> tuple[DatasetBuildTemplateCfg, str]:
    "Loads a strict template and returns the validated typed plan with its raw-file SHA-256."

    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"dataset build template does not exist: {resolved_path}")
    raw_bytes = resolved_path.read_bytes()
    document = yaml.safe_load(raw_bytes) or {}
    if not isinstance(document, Mapping):
        raise TypeError("dataset build template root must be a mapping")
    return _template_from_mapping(document), hashlib.sha256(raw_bytes).hexdigest()


def _template_from_mapping(document: Mapping[str, Any]) -> DatasetBuildTemplateCfg:

    _require_keys(
        document,
        allowed={
            "schema_version",
            "template_id",
            "inventory",
            "seeds",
            "balance",
            "partitions",
            "generation_policy",
            "manifests",
        },
        required={
            "schema_version",
            "template_id",
            "inventory",
            "seeds",
            "balance",
            "partitions",
            "generation_policy",
            "manifests",
        },
        context="dataset build template",
    )
    inventory = _mapping(document["inventory"], context="inventory")
    _require_keys(inventory, allowed={"run_dir"}, required={"run_dir"}, context="inventory")
    seeds = _mapping(document["seeds"], context="seeds")
    _require_keys(seeds, allowed={"selection", "mutation"}, required={"selection", "mutation"}, context="seeds")
    balance = _mapping(document["balance"], context="balance")
    _require_keys(
        balance,
        allowed={
            "selection_unit",
            "macro_family",
            "topology_shape",
            "missing_slot",
            "mixed_composition_group",
            "dof",
        },
        required={
            "selection_unit",
            "macro_family",
            "topology_shape",
            "missing_slot",
            "mixed_composition_group",
            "dof",
        },
        context="balance",
    )
    partitions = _mapping(document["partitions"], context="partitions")
    _require_keys(partitions, allowed={"train", "validation", "evaluation"}, required={"train", "validation", "evaluation"}, context="partitions")
    validation = _mapping(partitions["validation"], context="partitions.validation")
    _require_keys(validation, allowed={"unseen_variant_set", "unseen_mother"}, required={"unseen_variant_set", "unseen_mother"}, context="partitions.validation")
    evaluation = _mapping(partitions["evaluation"], context="partitions.evaluation")
    _require_keys(
        evaluation,
        allowed={"unseen_variant_set", "unseen_mother", "official_zero_shot"},
        required={"unseen_variant_set", "unseen_mother", "official_zero_shot"},
        context="partitions.evaluation",
    )
    generation = _mapping(document["generation_policy"], context="generation_policy")
    _require_keys(generation, allowed={"dataset_retry_rounds", "uniqueness", "failed_run_policy"}, required={"dataset_retry_rounds", "uniqueness", "failed_run_policy"}, context="generation_policy")
    manifests = _mapping(document["manifests"], context="manifests")
    _require_keys(manifests, allowed={"ssl", "ppo"}, required={"ssl", "ppo"}, context="manifests")
    ssl = _mapping(manifests["ssl"], context="manifests.ssl")
    _require_keys(ssl, allowed={"enabled"}, required={"enabled"}, context="manifests.ssl")
    ppo = _mapping(manifests["ppo"], context="manifests.ppo")
    _require_keys(ppo, allowed={"enabled", "train_mother_count", "reuse_ssl_holdouts"}, required={"enabled", "train_mother_count", "reuse_ssl_holdouts"}, context="manifests.ppo")

    return DatasetBuildTemplateCfg(
        schema_version=str(document["schema_version"]),
        template_id=str(document["template_id"]),
        inventory=DatasetInventoryCfg(run_dir=str(inventory["run_dir"])),
        seeds=DatasetBuildSeedsCfg(selection=int(seeds["selection"]), mutation=int(seeds["mutation"])),
        balance=DatasetBalanceCfg(
            selection_unit=cast(Literal["canonical_mirror_pair"], str(balance["selection_unit"])),
            macro_family={str(key): float(value) for key, value in _mapping(balance["macro_family"], context="balance.macro_family").items()},
            topology_shape={str(key): float(value) for key, value in _mapping(balance["topology_shape"], context="balance.topology_shape").items()},
            missing_slot=cast(Literal["uniform"], str(balance["missing_slot"])),
            mixed_composition_group=cast(Literal["uniform"], str(balance["mixed_composition_group"])),
            dof=cast(Literal["uniform_available"], str(balance["dof"])),
        ),
        partitions=DatasetPartitionsTemplateCfg(
            train=_role(partitions["train"], context="partitions.train"),
            validation=DatasetValidationTemplateCfg(
                unseen_variant_set=_role(validation["unseen_variant_set"], context="partitions.validation.unseen_variant_set"),
                unseen_mother=_role(validation["unseen_mother"], context="partitions.validation.unseen_mother"),
            ),
            evaluation=DatasetEvaluationTemplateCfg(
                unseen_variant_set=_role(evaluation["unseen_variant_set"], context="partitions.evaluation.unseen_variant_set"),
                unseen_mother=_role(evaluation["unseen_mother"], context="partitions.evaluation.unseen_mother"),
                official_zero_shot=_string_tuple(evaluation["official_zero_shot"], context="partitions.evaluation.official_zero_shot"),
            ),
        ),
        generation_policy=DatasetGenerationPolicyCfg(
            dataset_retry_rounds=int(generation["dataset_retry_rounds"]),
            uniqueness=cast(Literal["resample"], str(generation["uniqueness"])),
            failed_run_policy=cast(Literal["quarantine"], str(generation["failed_run_policy"])),
        ),
        manifests=DatasetManifestsCfg(
            ssl_enabled=bool(ssl["enabled"]),
            ppo=DatasetPpoManifestCfg(
                enabled=bool(ppo["enabled"]),
                train_mother_count=int(ppo["train_mother_count"]),
                reuse_ssl_holdouts=bool(ppo["reuse_ssl_holdouts"]),
            ),
        ),
    )


def _role(value: Any, *, context: str) -> DatasetRoleCfg:

    payload = _mapping(value, context=context)
    _require_keys(payload, allowed={"mother_count", "assets_per_lineage"}, required={"mother_count", "assets_per_lineage"}, context=context)
    return DatasetRoleCfg(mother_count=int(payload["mother_count"]), assets_per_lineage=int(payload["assets_per_lineage"]))


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:

    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    return value


def _string_tuple(value: Any, *, context: str) -> tuple[str, ...]:

    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{context} must be a sequence")
    return tuple(str(item) for item in value)


def _require_keys(payload: Mapping[str, Any], *, allowed: set[str], required: set[str], context: str) -> None:

    keys = {str(key) for key in payload}
    unknown = keys - allowed
    missing = required - keys
    if unknown:
        raise ValueError(f"{context} contains unknown fields: {tuple(sorted(unknown))}")
    if missing:
        raise ValueError(f"{context} is missing required fields: {tuple(sorted(missing))}")


__all__ = [
    "DATASET_BUILD_TEMPLATE_SCHEMA_VERSION",
    "DatasetBalanceCfg",
    "DatasetBuildTemplateCfg",
    "DatasetEvaluationTemplateCfg",
    "DatasetGenerationPolicyCfg",
    "DatasetInventoryCfg",
    "DatasetManifestsCfg",
    "DatasetPartitionsTemplateCfg",
    "DatasetPpoManifestCfg",
    "DatasetRoleCfg",
    "DatasetValidationTemplateCfg",
    "load_dataset_build_template",
]
