r"""分层资产数据集构建模板的严格 typed YAML schema。

模板只声明“从哪个 pre-made inventory 选择哪些 mother lineages，以及每条 lineage
最终贡献多少项资产”。post-mutate 的几何分布、validator、physics closure 仍由
``HandGeneratorCfg`` 所属 Python 配置拥有；机器 worker 数也不进入科研模板。

数量约定：``mother_count`` 统计 left/right 两只实际 mother，canonical mirror pair
不可拆，因此所有数量必须为偶数。``assets_per_lineage`` 包含可选 mother 本体，planner
按 ``n_variant=n_asset-1[include_mother]`` lower 成 generator 任务。
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import yaml

DATASET_BUILD_TEMPLATE_SCHEMA_VERSION = "1.0.0"
"""首版模板 schema；与最终消费 manifest 的 schema 2.0 相互独立。"""


@dataclass(frozen=True)
class DatasetInventoryCfg:
    r"""唯一 pre-made generation run 来源。"""

    run_dir: str  # 相对 AnyMani 根目录或绝对 generation-run 路径

    def __post_init__(self) -> None:
        r"""拒绝空 inventory 路径。"""

        if not self.run_dir.strip():
            raise ValueError("dataset build inventory.run_dir cannot be empty")


@dataclass(frozen=True)
class DatasetBuildSeedsCfg:
    r"""mother selection 与 post-mutate proposal 的两个独立随机根。"""

    selection: int
    mutation: int


@dataclass(frozen=True)
class DatasetBalanceCfg:
    r"""canonical pair 的分层配额与细层覆盖目标。

    ``macro_family`` 的四个键分别表示 single-Allegro、single-LEAP、以 Allegro
    为 palm/thumb 基座的 mixed，以及以 LEAP 为基座的 mixed。权重只定义相对配额，
    planner 通过容量约束最大余数法得到整数解。
    """

    selection_unit: Literal["canonical_mirror_pair"] = "canonical_mirror_pair"
    macro_family: Mapping[str, float] = field(default_factory=dict)
    topology_shape: Mapping[str, float] = field(default_factory=dict)
    missing_slot: Literal["uniform"] = "uniform"
    mixed_composition_group: Literal["uniform"] = "uniform"
    dof: Literal["uniform_available"] = "uniform_available"

    def __post_init__(self) -> None:
        r"""验证稳定分层轴及正权重。"""

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
    r"""一条 generated dataset role 的 lineage 数量与最终每-lineage 资产数。"""

    mother_count: int
    assets_per_lineage: int

    def __post_init__(self) -> None:
        r"""mirror pair 不可拆，且每条 lineage 至少交付一项资产。"""

        if self.mother_count < 2 or self.mother_count % 2 != 0:
            raise ValueError("dataset role mother_count must be a positive even number")
        if self.assets_per_lineage < 1:
            raise ValueError("dataset role assets_per_lineage must be >= 1")


@dataclass(frozen=True)
class DatasetValidationTemplateCfg:
    r"""checkpoint selection 的 seen-mother 与 unseen-mother 两条通道。"""

    unseen_variant_set: DatasetRoleCfg
    unseen_mother: DatasetRoleCfg


@dataclass(frozen=True)
class DatasetEvaluationTemplateCfg:
    r"""训练冻结后的 generated 与 official evaluation 通道。"""

    unseen_variant_set: DatasetRoleCfg
    unseen_mother: DatasetRoleCfg
    official_zero_shot: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        r"""official 路径必须非空且不重复；首版正式模板可保留空集合。"""

        if any(not path.strip() for path in self.official_zero_shot):
            raise ValueError("official_zero_shot asset paths cannot be empty")
        if len(set(self.official_zero_shot)) != len(self.official_zero_shot):
            raise ValueError("official_zero_shot asset paths must be unique")


@dataclass(frozen=True)
class DatasetPartitionsTemplateCfg:
    r"""train、validation 与 evaluation 的完整生成计划。"""

    train: DatasetRoleCfg
    validation: DatasetValidationTemplateCfg
    evaluation: DatasetEvaluationTemplateCfg


@dataclass(frozen=True)
class DatasetGenerationPolicyCfg:
    r"""dataset-level 补采、唯一性和失败资产保留策略。"""

    dataset_retry_rounds: int = 3
    uniqueness: Literal["resample"] = "resample"
    failed_run_policy: Literal["quarantine"] = "quarantine"

    def __post_init__(self) -> None:
        r"""dataset retry 是 generator 每槽 proposal budget 之外的有限外层预算。"""

        if self.dataset_retry_rounds < 1:
            raise ValueError("generation_policy.dataset_retry_rounds must be >= 1")
        if self.uniqueness != "resample" or self.failed_run_policy != "quarantine":
            raise ValueError("generation policy requires uniqueness=resample and failed_run_policy=quarantine")


@dataclass(frozen=True)
class DatasetPpoManifestCfg:
    r"""从 SSL train lineages 派生 PPO train 子集的声明。"""

    enabled: bool = True
    train_mother_count: int = 128
    reuse_ssl_holdouts: bool = True

    def __post_init__(self) -> None:
        r"""PPO 子集也保持 canonical mirror pairs。"""

        if self.enabled and (self.train_mother_count < 2 or self.train_mother_count % 2 != 0):
            raise ValueError("ppo train_mother_count must be a positive even number")


@dataclass(frozen=True)
class DatasetManifestsCfg:
    r"""一次 build 需要发布的下游消费 manifests。"""

    ssl_enabled: bool = True
    ppo: DatasetPpoManifestCfg = field(default_factory=DatasetPpoManifestCfg)


@dataclass(frozen=True)
class DatasetBuildTemplateCfg:
    r"""一份可审计、可重放的资产数据集选择与生成意图。"""

    schema_version: str
    template_id: str
    inventory: DatasetInventoryCfg
    seeds: DatasetBuildSeedsCfg
    balance: DatasetBalanceCfg
    partitions: DatasetPartitionsTemplateCfg
    generation_policy: DatasetGenerationPolicyCfg
    manifests: DatasetManifestsCfg

    def __post_init__(self) -> None:
        r"""验证 schema identity 与跨区数量约束。"""

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
    r"""严格读取模板并返回 typed config 与原始 YAML SHA-256。

    Args:
        path (str | Path): 模板 YAML 路径。

    Returns:
        tuple[DatasetBuildTemplateCfg, str]: 已验证模板与 byte-level 内容身份。
    """

    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"dataset build template does not exist: {resolved_path}")
    raw_bytes = resolved_path.read_bytes()
    document = yaml.safe_load(raw_bytes) or {}
    if not isinstance(document, Mapping):
        raise TypeError("dataset build template root must be a mapping")
    return _template_from_mapping(document), hashlib.sha256(raw_bytes).hexdigest()


def _template_from_mapping(document: Mapping[str, Any]) -> DatasetBuildTemplateCfg:
    r"""把 YAML 基础容器恢复为冻结 dataclasses，并拒绝未知字段。"""

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
    r"""解析一条 role 的两个显式数量字段。"""

    payload = _mapping(value, context=context)
    _require_keys(payload, allowed={"mother_count", "assets_per_lineage"}, required={"mother_count", "assets_per_lineage"}, context=context)
    return DatasetRoleCfg(mother_count=int(payload["mother_count"]), assets_per_lineage=int(payload["assets_per_lineage"]))


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    r"""要求 YAML 节点是 mapping。"""

    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    return value


def _string_tuple(value: Any, *, context: str) -> tuple[str, ...]:
    r"""把 YAML sequence 规范成字符串 tuple。"""

    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{context} must be a sequence")
    return tuple(str(item) for item in value)


def _require_keys(payload: Mapping[str, Any], *, allowed: set[str], required: set[str], context: str) -> None:
    r"""严格拒绝未知字段与缺失字段。"""

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
