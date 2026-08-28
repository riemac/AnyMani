r"""跨 generation run 的 hand asset dataset manifest 与解析运行时。

``HandBank`` 回答“一个 source root 或显式列表中有哪些可消费 hand bundles”；本模块在其上
增加实验无关的数据集层，回答“哪些 mother/variant sets 属于哪个具名 partition”。YAML 只引用
generation run、production group、mother 与完整 variant set，不列单个 variant ID。SSL、PPO 与
task 可复用同一 resolved dataset，但各自仍拥有 geometry materialization、spawn 与训练语义。

目录合同：

```text
generated/<generation_run>/
  <group>/<mother>/hand.urdf
  <group>/<mother>/<variant_set>/summary.yaml
  <group>/<mother>/<variant_set>/<asset_id>/hand.urdf
  mixed/<composition_group>/<mother>/...
```
"""

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
"""具名 validation/evaluation suites 的严格 run-map dataset YAML schema。"""

HandAssetCollectionKind: TypeAlias = Literal["groups", "mixed", "official"]
"""资产目录分支；``mixed`` 比普通 production group 多一层 composition group。"""


@dataclass(frozen=True)
class HandAssetLineageCfg:
    r"""一个 mother 在某 partition 中被选择的组成。

    ``include_mother`` 必须显式声明，避免同一 mother 的 train/validation variant sets 并存时
    把 mother 本体隐式复制到两个 partition。``variant_sets`` 是配置原子；runtime 会收录 set
    下全部合法 direct-child variants。
    """

    include_mother: bool  # 是否把 mother 根 bundle 本身作为一项数据
    variant_sets: tuple[str, ...] = ()  # mother 下的 mutate-run 目录名，保持 YAML 声明顺序

    def __post_init__(self) -> None:
        r"""拒绝空 lineage、重复 set 和越出 mother 根目录的路径片段。"""

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
    r"""一个 partition 内来自同一 generation run 的选择块。

    ``run_dir`` 为空时继承 dataset 顶层 ``default_run_dir``；路径覆盖只允许发生在这一层，
    不在 group/mother/variant set 间建立多级继承。
    """

    run_dir: str = ""  # 可选 generation run override；相对路径锚定 AnyMani 根
    groups: HandAssetGroupMap = field(default_factory=dict)  # `<run>/<production_group>/<mother>`
    mixed: HandAssetGroupMap = field(default_factory=dict)  # `<run>/mixed/<composition_group>/<mother>`

    def __post_init__(self) -> None:
        r"""run block 至少选择一条 lineage，并验证目录键是安全相对组件。"""

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
    r"""由有序 run aliases 组成的 generated partition。"""

    runs: Mapping[str, HandAssetRunCfg] = field(default_factory=dict)  # alias 只服务 YAML 可读性与 provenance

    def __post_init__(self) -> None:
        r"""拒绝空 alias；空 partition 本身合法，用于尚未声明的 validation/suite。"""

        for run_alias in self.runs:
            if not str(run_alias).strip():
                raise ValueError("dataset run alias cannot be empty")


@dataclass(frozen=True)
class HandAssetOfficialPartitionCfg:
    r"""不服从 generated run 目录层级的 official zero-shot bundle 列表。"""

    assets: tuple[str, ...] = ()  # official bundle/URDF 路径；人工 geometry semantics 在 HandBank 校验

    def __post_init__(self) -> None:
        r"""路径字符串必须非空且不重复。"""

        if any(not path.strip() for path in self.assets):
            raise ValueError("official asset paths cannot be empty")
        if len(set(self.assets)) != len(self.assets):
            raise ValueError("official asset paths must be unique")


@dataclass(frozen=True)
class HandAssetValidationCfg:
    r"""checkpoint selection 使用的两条 generated validation 泛化轴。

    ``unseen_variant_set`` 的 mother 必须已属于 train，因而只测同一 morphology
    下未见 post-mutate realization；``unseen_mother`` 的完整 lineage 必须与 train
    隔离。两者保持具名结构，训练侧才能分别聚合后再决定 checkpoint score。
    """

    unseen_variant_set: HandAssetPartitionCfg = field(default_factory=HandAssetPartitionCfg)
    unseen_mother: HandAssetPartitionCfg = field(default_factory=HandAssetPartitionCfg)


@dataclass(frozen=True)
class HandAssetEvaluationCfg:
    r"""训练冻结后可独立执行的三类 zero-shot 资产选择。"""

    unseen_variant_set: HandAssetPartitionCfg = field(default_factory=HandAssetPartitionCfg)
    unseen_mother: HandAssetPartitionCfg = field(default_factory=HandAssetPartitionCfg)
    official_zero_shot: HandAssetOfficialPartitionCfg = field(default_factory=HandAssetOfficialPartitionCfg)


@dataclass(frozen=True)
class HandAssetDatasetCfg:
    r"""一份下游中立、可由人工或算法生成的 hand asset dataset 声明。"""

    schema_version: str = HAND_ASSET_DATASET_SCHEMA_VERSION  # persisted YAML contract
    default_run_dir: str = ""  # 大多数 run block 共用的 generation root
    train: HandAssetPartitionCfg = field(default_factory=HandAssetPartitionCfg)
    validation: HandAssetValidationCfg = field(default_factory=HandAssetValidationCfg)
    evaluation: HandAssetEvaluationCfg = field(default_factory=HandAssetEvaluationCfg)

    def __post_init__(self) -> None:
        r"""只做无 IO schema 检查；目录存在性在 :meth:`HandAssetDataset.resolve` 验证。"""

        if self.schema_version != HAND_ASSET_DATASET_SCHEMA_VERSION:
            raise ValueError(f"hand asset dataset schema must be exactly {HAND_ASSET_DATASET_SCHEMA_VERSION!r}")
        if not self.train.runs:
            raise ValueError("hand asset dataset requires a non-empty train partition")


@dataclass(frozen=True)
class HandAssetProvenance:
    r"""一项 resolved bundle 在 dataset 命名空间中的完整来源坐标。"""

    partition: str  # train / validation / evaluation suite name
    run_alias: str  # YAML run block 的稳定人类标签
    run_dir: str  # resolved generation run 绝对路径
    collection_kind: HandAssetCollectionKind  # groups / mixed / official
    group_name: str  # production group 或 mixed composition group
    mother_name: str  # mother topology 目录名；official 为空
    mother_path: str  # resolved mother 根；evaluation 关系判据
    variant_set: str  # mutate-run 名；mother/official 为空
    asset_role: Literal["mother", "variant", "official"]  # 当前 bundle 在 lineage 中的角色


@dataclass(frozen=True)
class ResolvedHandAssetRecord:
    r"""HandContainer 与 dataset provenance 的不可分记录。"""

    container: HandContainer  # 下游中立虚拟 bundle
    provenance: HandAssetProvenance  # partition/run/lineage 坐标
    content_hash: str = ""  # typed semantics 内容身份；legacy 无字段时为空


@dataclass(frozen=True)
class ResolvedHandAssetPartition:
    r"""一个 partition 或 evaluation suite 的有序 resolved records。"""

    name: str
    records: tuple[ResolvedHandAssetRecord, ...]

    @property
    def assets(self) -> tuple[HandContainer, ...]:
        r"""返回供 HandSpawn/GeometrySource 直接消费的 container 轴。"""

        return tuple(record.container for record in self.records)


@dataclass(frozen=True)
class ResolvedHandAssetDataset:
    r"""一次 YAML resolve 后的完整 train/validation/evaluation 选择与审计身份。"""

    source_path: Path  # dataset YAML 绝对路径
    source_sha256: str  # 原始 YAML bytes，供 resume/实验产物比对
    config: HandAssetDatasetCfg  # 已类型化且验证的选择声明
    train: ResolvedHandAssetPartition
    validation: Mapping[str, ResolvedHandAssetPartition]
    evaluation: Mapping[str, ResolvedHandAssetPartition]

    def config_dict(self) -> dict[str, Any]:
        r"""返回可嵌入 resolved experiment YAML 的基础容器。"""

        return asdict(self.config)


class HandAssetDataset:
    r"""读取通用 dataset YAML，并通过 HandBank 解析所有 leaf bundles。"""

    def __init__(self, config: HandAssetDatasetCfg, *, source_path: Path, source_sha256: str) -> None:
        r"""保存无副作用配置；实际目录扫描只发生在 :meth:`resolve`。"""

        self.config = config
        self.source_path = source_path
        self.source_sha256 = source_sha256

    @classmethod
    def from_yaml(cls, path: str | Path) -> HandAssetDataset:
        r"""加载并严格解析一份 dataset manifest。

        Args:
            path (str | Path): 绝对路径或相对 AnyMani 根目录的 YAML 路径。

        Returns:
            HandAssetDataset: 尚未扫描 hand bundles 的 dataset runtime。
        """

        resolved_path = resolve_bank_path(path)  # 不受 Hydra 改 cwd 或 shell cwd 影响
        if not resolved_path.is_file():
            raise FileNotFoundError(f"hand asset dataset manifest does not exist: {resolved_path}")
        raw_bytes = resolved_path.read_bytes()  # hash 锚定用户实际提交/生成的 YAML bytes
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
        r"""展开完整 variant sets，并验证 partition 与 evaluation relation。

        Args:
            require_geometry_semantics (bool): 是否要求每个 HandContainer 交付 typed static semantics。
            allow_legacy_left_handedness (bool): 是否显式放行缺严格镜像证书的 legacy generated left。

        Returns:
            ResolvedHandAssetDataset: 有序资产、lineage provenance 与 YAML identity。
        """

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

        # Dataset identity 要求每项物理 bundle 只承担一个 partition/suite 角色；更深的
        # physical-mapping equality 由 SSL geometry materialization 后的 hash gate 负责。
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
        r"""只展开 train partition，不读取 validation/evaluation bundle 路径。

        schema 仍由 :meth:`from_yaml` 按 2.0 完整严格解析；该入口只缩小运行时 IO 边界，服务 PPO
        训练与资产预热。返回顺序完全继承 YAML run/group/mother/variant-set 声明，并由既有 lineage
        resolver 在每个 variant set 内按 asset ID 排序。

        Args:
            require_geometry_semantics (bool): 是否要求每项资产交付 typed static geometry semantics。
            allow_legacy_left_handedness (bool): 是否显式放行缺严格镜像证书的 legacy generated left。
            max_assets (int | None): 仅供 smoke/诊断的有序 train 前缀长度；``None`` 表示完整 partition。

        Returns:
            ResolvedHandAssetPartition: 唯一性验证后的有序 train records/container 轴。
        """

        if max_assets is not None and max_assets < 1:
            raise ValueError("max_assets must be positive when provided")
        train = self._resolve_generated_partition(
            self.config.train,
            partition_name="train",
            require_geometry_semantics=require_geometry_semantics,
            allow_legacy_left_handedness=allow_legacy_left_handedness,
            max_records=max_assets,
        )  # 只对 train.runs 触发 bundle/sidecar/mesh IO
        _validate_unique_asset_records((train,))  # train 内路径、asset ID 与 content hash 仍必须唯一
        return train

    def _resolve_generated_partition(
        self,
        config: HandAssetPartitionCfg,
        *,
        partition_name: str,
        require_geometry_semantics: bool,
        allow_legacy_left_handedness: bool,
        max_records: int | None = None,
    ) -> ResolvedHandAssetPartition:
        r"""按 YAML 声明顺序展开 generated run/group/mother/set 层级。"""

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
            # Smoke prefix 必须保持正式 YAML 顺序，但不应并行启动后续无关 lineage 的 IO。
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
            records = records[:max_records]  # 最后一条 lineage 可能使 records 超过 requested prefix
        return ResolvedHandAssetPartition(name=partition_name, records=tuple(records))

    def _resolve_official_partition(
        self,
        config: HandAssetOfficialPartitionCfg,
        *,
        require_geometry_semantics: bool,
    ) -> ResolvedHandAssetPartition:
        r"""解析 official bundles，不套用 generated run/mother 目录启发式。"""

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
    r"""在线程 worker 中解析一条 lineage，保持主线程负责结果顺序规约。"""

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
    r"""解析一只 mother 及声明的完整 variant sets，并附着同一 lineage provenance。"""

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

    # 每个 variant set 保持配置顺序，set 内由 HandBank 按 asset ID 排序，消除文件系统枚举差异。
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
    r"""通过既有 HandBank explicit route 解析一个 bundle，保持单资产 contract 单一真源。"""

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
    r"""把 container 与不会被 flat asset ID 表达的 lineage 坐标绑定。"""

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
    r"""验证 dataset run block 指向 pre-made generation run，而非任意目录。"""

    summary_path = run_root / "summary.yaml"
    summary = _load_yaml_mapping(summary_path, label="generation run summary")
    run = summary.get("run")
    if not isinstance(run, Mapping) or run.get("mode") != "made":
        raise ValueError(f"dataset run_dir must contain a mode='made' summary: {summary_path}")


def _validate_variant_set(variant_set_root: Path, mother_root: Path) -> None:
    r"""验证 mutate summary 的来源 mother 与实际目录关系，并复核成功资产数。"""

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
    r"""拒绝同一路径、asset ID 或已知 content hash 承担多个 dataset 角色。"""

    seen_paths: dict[Path, str] = {}
    seen_ids: dict[str, str] = {}
    seen_content: dict[str, str] = {}
    for partition in partitions:
        for record in partition.records:
            bundle_path = record.container.urdf_path.parent.resolve(strict=False)
            _record_unique_identity(bundle_path, seen_paths, label="bundle path", partition=partition.name)
            _record_unique_identity(record.container.asset_id, seen_ids, label="asset ID", partition=partition.name)
            if record.content_hash:  # legacy sidecar 没有 content hash 时仍由 path/ID 两层保护
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
    r"""把同类型 identity 写入对应索引，并报告跨角色重复。"""

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
    r"""验证具名 validation/evaluation suites 的 lineage 隔离关系。

    validation 与 evaluation 都包含“同 mother 新 variants”和“全新 mother”两条轴。
    evaluation 发生在 checkpoint selection 之后，因此其 seen-mother cohort 不能与
    validation seen-mother cohort 重合，unseen mother 也不能在任何更早角色出现。
    """

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


def _validate_unseen_variant_set(
    partition: ResolvedHandAssetPartition,
    *,
    train_mothers: set[str],
) -> set[str]:
    r"""验证一条 unseen-variant suite，并返回其 mother lineage 集。"""

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
    r"""验证一条 unseen-mother suite，并返回其完整 lineage 集。"""

    mothers = {record.provenance.mother_path for record in partition.records if record.provenance.mother_path}
    overlap = mothers & forbidden_mothers
    if overlap:
        raise ValueError(f"unseen_mother already appears in train or validation: {tuple(sorted(overlap))}")
    return mothers


def _container_content_hash(container: HandContainer) -> str:
    r"""读取 typed 或 raw geometry semantics 的 content identity，不自行重算资产语义。"""

    if container.geometry_semantics is not None:
        return container.geometry_semantics.content_hash
    raw_semantics = container.sidecar.get("geometry_semantics")
    if isinstance(raw_semantics, Mapping):
        return str(raw_semantics.get("content_hash") or "")
    return ""


def _dataset_cfg_from_mapping(document: Mapping[str, Any]) -> HandAssetDatasetCfg:
    r"""把 YAML 基础容器重建为冻结 dataclasses，并拒绝未知字段。"""

    _require_keys(
        document,
        allowed={"schema_version", "default_run_dir", "train", "validation", "evaluation"},
        required={"schema_version", "default_run_dir", "train", "validation", "evaluation"},
        context="dataset",
    )
    schema_version = str(document["schema_version"])
    if schema_version != HAND_ASSET_DATASET_SCHEMA_VERSION:
        raise ValueError(f"hand asset dataset schema must be exactly {HAND_ASSET_DATASET_SCHEMA_VERSION!r}")

    validation_raw = _as_mapping(document["validation"], context="validation")
    _require_keys(
        validation_raw,
        allowed={"unseen_variant_set", "unseen_mother"},
        required={"unseen_variant_set", "unseen_mother"},
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
                validation_raw["unseen_variant_set"], context="validation.unseen_variant_set"
            ),
            unseen_mother=_partition_cfg_from_mapping(
                validation_raw["unseen_mother"], context="validation.unseen_mother"
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
    r"""解析 ``runs.<alias>`` mapping，并保持 YAML 插入顺序。"""

    payload = _as_mapping(value, context=context)
    _require_keys(payload, allowed={"runs"}, required=set(), context=context)
    runs_raw = _as_mapping(payload.get("runs", {}), context=f"{context}.runs")
    return HandAssetPartitionCfg(
        runs={
            str(alias): _run_cfg_from_mapping(run, context=f"{context}.runs.{alias}") for alias, run in runs_raw.items()
        }
    )


def _run_cfg_from_mapping(value: Any, *, context: str) -> HandAssetRunCfg:
    r"""解析一个 generation run block 的 groups/mixed 两类目录分支。"""

    payload = _as_mapping(value, context=context)
    _require_keys(payload, allowed={"run_dir", "groups", "mixed"}, required=set(), context=context)
    return HandAssetRunCfg(
        run_dir=str(payload.get("run_dir", "")),
        groups=_group_map_from_mapping(payload.get("groups", {}), context=f"{context}.groups"),
        mixed=_group_map_from_mapping(payload.get("mixed", {}), context=f"{context}.mixed"),
    )


def _group_map_from_mapping(value: Any, *, context: str) -> dict[str, dict[str, HandAssetLineageCfg]]:
    r"""解析 production/composition group -> mother -> lineage 三层 mapping。"""

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
    r"""解析一个 mother 的显式本体开关与完整 variant-set 目录名。"""

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
    r"""收窄 YAML mapping 类型并保留字段上下文。"""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    return value


def _string_tuple(value: Any, *, context: str) -> tuple[str, ...]:
    r"""把 YAML sequence 冻结为非空字符串 tuple。"""

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
    r"""严格拒绝拼写错误字段和缺失的身份字段。"""

    keys = {str(key) for key in payload}
    unknown = keys - allowed
    missing = required - keys
    if unknown:
        raise ValueError(f"{context} contains unknown fields: {tuple(sorted(unknown))}")
    if missing:
        raise ValueError(f"{context} is missing required fields: {tuple(sorted(missing))}")


def _require_relative_component(value: str, *, label: str) -> None:
    r"""目录键只能是单个相对组件，防止 manifest 越出声明层级。"""

    path = PurePosixPath(str(value))
    if not value or path.is_absolute() or len(path.parts) != 1 or path.parts[0] in {".", ".."}:
        raise ValueError(f"dataset {label} must be one relative path component: {value!r}")


def _load_yaml_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
    r"""读取 summary YAML，并把缺文件、空文档与非 mapping 分开报告。"""

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
