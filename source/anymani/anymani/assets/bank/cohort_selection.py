r"""从published train manifests确定性选择pure LEAP-right mother-lineage cohorts。

Scale ladder的选择原子是mother lineage，不是manifest row：每条lineage取mother与三个代表variants。离散覆盖
优先于连续几何距离，且source-qualified row只作可追溯坐标，不参与tie-break。A64使用PPO中的全部16条
pure LEAP-right mothers；A128保留该前缀，再从SSL-only topology补16条，使四个$(N_{tip},D_{thumb})$
cell各有8条mothers。

本模块只读取asset bank交付的typed semantics、provenance与静态geometry fingerprint，不启动Isaac。最终
``physical_geometry_hash``仍由canonical lowering给出，不能由source descriptor猜测。
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from anymani.assets.geometry_identity import geometry_fingerprint_from_sidecar

from .cohort import write_hand_asset_cohort_lock
from .dataset import HandAssetDataset, ResolvedHandAssetPartition, ResolvedHandAssetRecord
from .prepared_train import resolve_prepared_train
from .representative_selection import SUPPORTED_CELL_VALUES, RepresentativeAsset, representative_assets

LINEAGE_COHORT_SELECTION_SCHEMA_VERSION = "1.0.0"
"""Pure-lineage recipe/result metadata schema；它嵌入cohort lock的``selection``字段。"""


@dataclass(frozen=True)
class SourceCellMotherQuota:
    r"""一个source manifest在四个固定cell中应贡献的mother数量。"""

    source_alias: str  # 例如``ppo``或``ssl``，同时成为source-qualified key前缀
    counts: tuple[int, int, int, int]  # 顺序与``SUPPORTED_CELL_VALUES``逐项一致

    def __post_init__(self) -> None:
        r"""拒绝空alias、错误cell宽度与负配额。"""

        if not self.source_alias.strip() or "#" in self.source_alias:
            raise ValueError("source quota alias must be non-empty and '#' free")
        if len(self.counts) != len(SUPPORTED_CELL_VALUES) or any(count < 0 for count in self.counts):
            raise ValueError("source quota must contain four non-negative cell counts")

    def for_cell(self, cell: tuple[int, int]) -> int:
        r"""返回指定$(N_{tip},D_{thumb})$ cell的mother配额。"""

        try:
            return self.counts[SUPPORTED_CELL_VALUES.index(cell)]
        except ValueError as error:
            raise ValueError(f"unsupported lineage cohort cell {cell}") from error


@dataclass(frozen=True)
class PureLeapRightLineageRecipe:
    r"""Pure LEAP-right scale cohort的声明式选择合同。"""

    cohort_id: str  # 稳定的人类可读实验支持集ID
    source_quotas: tuple[SourceCellMotherQuota, ...]  # source顺序同时定义cohort prefix顺序
    members_per_lineage: int = 4  # 固定mother + 3 representative variants
    selection_seed: int = 20260904  # 当前算法无随机采样；seed仍冻结未来兼容的选择域
    production_group: str = "single_palm_leap"  # dataset production group过滤条件
    handedness: str = "right"  # 首轮scale ladder只覆盖右手
    family: str = "leap"  # surviving finger slots必须全部属于LEAP
    require_unique_topology: bool = True  # A128的32条mother主轴不得重复topology
    excluded_mother_names: tuple[str, ...] = ()  # 研究留出母体；从所有source的候选池排除整个lineage

    def __post_init__(self) -> None:
        r"""静态验证recipe基数、source唯一性与固定四成员语义。"""

        aliases = tuple(quota.source_alias for quota in self.source_quotas)
        if not self.cohort_id.strip() or not aliases or len(set(aliases)) != len(aliases):
            raise ValueError("lineage recipe requires cohort_id and unique source aliases")
        if self.members_per_lineage != 4:
            raise ValueError("scale lineage recipe requires exactly mother plus three variants")
        if self.handedness not in {"left", "right"} or not self.family.strip() or not self.production_group.strip():
            raise ValueError("lineage recipe has invalid handedness/family/production group")
        if len(set(self.excluded_mother_names)) != len(self.excluded_mother_names) or any(
            not name or name != name.strip() for name in self.excluded_mother_names
        ):
            raise ValueError("excluded mother names must be non-empty, whitespace-free and unique")

    @property
    def mother_count(self) -> int:
        r"""返回所有source/cell配额之和。"""

        return sum(sum(quota.counts) for quota in self.source_quotas)

    @property
    def asset_count(self) -> int:
        r"""返回最终成员数$A=4N_{mother}$。"""

        return self.mother_count * self.members_per_lineage


@dataclass(frozen=True)
class LineageDescriptor:
    r"""用于离散覆盖与连续max-min选择的一条mother物理描述。"""

    key: str  # source alias + mother content identity组成的内部唯一key
    source_alias: str
    mother_row: int  # source train row，只作provenance
    mother_name: str  # handed topology名称，例如``right_t3_m4_r1``
    mother_asset_id: str
    mother_content_hash: str
    static_geometry_fingerprint: str  # build-time静态碰撞几何身份，不等同canonical physical hash
    cell: tuple[int, int]  # $(N_{tip},D_{thumb})$
    topology: str  # handedness-neutral topology
    missing_slots: tuple[str, ...]  # 空tuple表示四指完整
    finger_dofs: tuple[int, int, int, int]  # index/middle/ring/thumb活动DoF
    descriptor: tuple[float, ...]  # representative_selection交付的连续物理描述

    @property
    def identity_tiebreak(self) -> tuple[str, str]:
        r"""返回与manifest row无关的最终稳定tie-break。"""

        return self.mother_content_hash, self.mother_asset_id


@dataclass(frozen=True)
class MutationVariantDescriptor:
    r"""Lineage内一个variant的mutation类别、TIP类别与连续几何描述。"""

    key: str  # source-qualified content identity，不使用manifest row作tie-break
    source_row: int  # 最终lock provenance坐标
    asset_id: str
    content_hash: str
    mutation_mode_tokens: tuple[str, ...]  # ``mutator:resolved_self_mode``集合
    tip_signature_tokens: tuple[str, ...]  # ``finger:tip_type``集合
    descriptor: tuple[float, ...]  # 与mother/cell相同定义的连续物理描述

    @property
    def identity_tiebreak(self) -> tuple[str, str]:
        r"""最终稳定tie-break为content hash再asset ID。"""

        return self.content_hash, self.asset_id


@dataclass(frozen=True)
class ResolvedLineageCohortSelection:
    r"""Selector交付给cohort writer的有序坐标与完整审计文档。"""

    recipe: PureLeapRightLineageRecipe
    member_coordinates: tuple[tuple[str, int], ...]  # source-qualified有序成员轴$[A]$
    selection_document: Mapping[str, Any]  # 写入lock并由其SHA冻结


@dataclass(frozen=True)
class _ResolvedLineage:
    r"""Lineage descriptor与其mother/前三个variant source rows的内部绑定。"""

    descriptor: LineageDescriptor
    member_rows: tuple[int, int, int, int]
    member_records: tuple[ResolvedHandAssetRecord, ...]


PURE_LEAP_RIGHT_A64_RECIPE = PureLeapRightLineageRecipe(
    cohort_id="pure-leap-right-a64",
    source_quotas=(SourceCellMotherQuota("ppo", (2, 3, 7, 4)),),
)
"""PPO train中全部16条pure LEAP-right mothers，每条mother+3 variants。"""

PURE_LEAP_RIGHT_A128_RECIPE = PureLeapRightLineageRecipe(
    cohort_id="pure-leap-right-a128",
    source_quotas=(
        SourceCellMotherQuota("ppo", (2, 3, 7, 4)),
        SourceCellMotherQuota("ssl", (6, 5, 1, 4)),
    ),
)
"""PPO 16条前缀加SSL-only 16条，使四个cell最终各8条mothers。"""


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    r"""返回标准化连续描述符的Euclidean距离。"""

    if len(left) != len(right):
        raise ValueError("lineage descriptor distance requires equal widths")
    return math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def _standardized_vectors(lineages: Sequence[LineageDescriptor]) -> dict[str, tuple[float, ...]]:
    r"""在当前cell候选与既选基线的联合域逐维做population标准化。"""

    if not lineages or len({lineage.key for lineage in lineages}) != len(lineages):
        raise ValueError("standardization requires non-empty unique lineage keys")
    width = len(lineages[0].descriptor)  # 连续物理描述维度$D$
    if width < 1 or any(len(lineage.descriptor) != width for lineage in lineages):
        raise ValueError("lineage descriptors must share one non-empty width")
    columns = tuple(tuple(lineage.descriptor[index] for lineage in lineages) for index in range(width))
    means = tuple(math.fsum(column) / len(column) for column in columns)  # $\mu_d$
    scales = tuple(
        max(math.sqrt(math.fsum((value - mean) ** 2 for value in column) / len(column)), 1.0e-12)
        for column, mean in zip(columns, means, strict=True)
    )  # $\sigma_d$；常数维使用$10^{-12}$防除零
    return {
        lineage.key: tuple(
            (value - mean) / scale for value, mean, scale in zip(lineage.descriptor, means, scales, strict=True)
        )
        for lineage in lineages
    }


def select_diverse_lineages(
    candidates: Sequence[LineageDescriptor],
    *,
    quota: int,
    prior: Sequence[LineageDescriptor] = (),
) -> tuple[LineageDescriptor, ...]:
    r"""按离散覆盖→跨source距离→组内max-min选择一个cell的mother组合。

    穷举的是mother组合而非资产组合；当前最大问题为$\binom{18}{4}=3060$，可在CPU上精确比较。目标按
    lexicographic顺序最大化：missing-slot类别数、finger-DoF向量数、到既选PPO集合的最小距离、组合内部
    最小距离及距离总和。数值目标完全相同时，选择排序后``(content_hash, asset_id)``最小的组合。
    """

    ordered = tuple(sorted(candidates, key=lambda item: item.identity_tiebreak))  # tie时hash小者先被保留
    baseline = tuple(prior)
    if quota < 1 or quota > len(ordered):
        raise ValueError(f"lineage quota {quota} cannot be drawn from {len(ordered)} candidates")
    cells = {lineage.cell for lineage in (*ordered, *baseline)}
    if len(cells) != 1:
        raise ValueError("diverse lineage selection must operate inside exactly one morphology cell")
    vectors = _standardized_vectors((*ordered, *baseline))

    best_combo: tuple[LineageDescriptor, ...] | None = None
    best_score: tuple[float, ...] | None = None
    for combo in combinations(ordered, quota):
        internal_distances = [
            _distance(vectors[left.key], vectors[right.key])
            for left_index, left in enumerate(combo)
            for right in combo[left_index + 1 :]
        ]
        baseline_distances = [
            _distance(vectors[lineage.key], vectors[reference.key]) for lineage in combo for reference in baseline
        ]
        nearest_baseline_sum = (
            sum(
                min(_distance(vectors[lineage.key], vectors[reference.key]) for reference in baseline)
                for lineage in combo
            )
            if baseline
            else 0.0
        )
        score = (
            float(len({lineage.missing_slots for lineage in combo})),  # 首先覆盖不同缺指类别
            float(len({lineage.finger_dofs for lineage in combo})),  # 再覆盖不同逐指DoF向量
            min(baseline_distances) if baseline_distances else 0.0,  # 与PPO最相近项也应尽量远
            min(internal_distances) if internal_distances else 0.0,  # SSL组合内部max-min
            math.fsum(internal_distances) + nearest_baseline_sum,  # 最后扩大总体连续几何跨度
        )
        if best_score is None or score > best_score:
            best_combo, best_score = combo, score  # combinations按hash序；完全同分时自然保留最小tie
    if best_combo is None:
        raise RuntimeError("lineage combination search produced no candidate")
    return best_combo


def select_diverse_variants(
    mother_descriptor: Sequence[float],
    variants: Sequence[MutationVariantDescriptor],
    *,
    count: int = 3,
) -> tuple[MutationVariantDescriptor, ...]:
    r"""先覆盖mutation/TIP类别，再以标准化geometry max-min选择lineage variants。

    每轮lexicographic最大化尚未覆盖的``mutator:mode`` token数、尚未覆盖的``finger:tip_type`` token数、
    整体mutation/TIP signature新颖性，以及到mother和已选variants的最小标准化描述距离。完全同分时按
    ``(content_hash, asset_id)``升序选择，manifest row不影响结果。
    """

    ordered = tuple(sorted(variants, key=lambda item: item.identity_tiebreak))
    if count < 1 or count > len(ordered):
        raise ValueError(f"variant count {count} cannot be drawn from {len(ordered)} candidates")
    width = len(mother_descriptor)
    if width < 1 or any(len(variant.descriptor) != width for variant in ordered):
        raise ValueError("mother and variant descriptors must share one non-empty width")

    # Mother与全部variant共同确定lineage-local标准化，避免某个物理量因单位尺度支配距离。
    raw_vectors = (tuple(float(value) for value in mother_descriptor), *(variant.descriptor for variant in ordered))
    columns = tuple(tuple(vector[index] for vector in raw_vectors) for index in range(width))
    means = tuple(math.fsum(column) / len(column) for column in columns)
    scales = tuple(
        max(math.sqrt(math.fsum((value - mean) ** 2 for value in column) / len(column)), 1.0e-12)
        for column, mean in zip(columns, means, strict=True)
    )
    standardized = tuple(
        tuple((value - mean) / scale for value, mean, scale in zip(vector, means, scales, strict=True))
        for vector in raw_vectors
    )
    mother_vector = standardized[0]
    vectors = {variant.key: standardized[index + 1] for index, variant in enumerate(ordered)}

    selected: list[MutationVariantDescriptor] = []
    covered_modes: set[str] = set()
    covered_tips: set[str] = set()
    covered_mode_signatures: set[tuple[str, ...]] = set()
    covered_tip_signatures: set[tuple[str, ...]] = set()
    while len(selected) < count:
        remaining = tuple(variant for variant in ordered if variant not in selected)

        def score(variant: MutationVariantDescriptor) -> tuple[float, ...]:
            r"""返回当前greedy轮的离散覆盖与连续max-min目标。"""

            references = (mother_vector, *(vectors[item.key] for item in selected))
            diversity = min(_distance(vectors[variant.key], reference) for reference in references)
            return (
                float(len(set(variant.mutation_mode_tokens) - covered_modes)),
                float(len(set(variant.tip_signature_tokens) - covered_tips)),
                float(variant.mutation_mode_tokens not in covered_mode_signatures),
                float(variant.tip_signature_tokens not in covered_tip_signatures),
                diversity,
            )

        chosen = max(remaining, key=score)  # ordered为hash升序；完全同分时max保留首项
        selected.append(chosen)
        covered_modes.update(chosen.mutation_mode_tokens)
        covered_tips.update(chosen.tip_signature_tokens)
        covered_mode_signatures.add(chosen.mutation_mode_tokens)
        covered_tip_signatures.add(chosen.tip_signature_tokens)
    return tuple(selected)


def _variant_descriptor(
    source_alias: str,
    row: int,
    record: ResolvedHandAssetRecord,
    representative: RepresentativeAsset,
) -> MutationVariantDescriptor:
    r"""从sidecar mutation samples提取分类tokens，并绑定连续geometry descriptor。"""

    samples = record.container.sidecar.get("post_mutate_samples")
    if not isinstance(samples, Mapping) or not samples:
        raise ValueError(f"variant asset {record.container.asset_id!r} lacks post_mutate_samples")
    mode_tokens = tuple(
        f"{name}:{values.get('resolved_self_mode', '')}"
        for name, values in sorted(samples.items())
        if isinstance(values, Mapping)
    )
    tip_replace = samples.get("tip_replace", {})
    finger_specs = tip_replace.get("finger_specs", {}) if isinstance(tip_replace, Mapping) else {}
    tip_tokens = tuple(
        f"{finger}:{spec.get('tip_type', 'base')}"
        for finger, spec in sorted(finger_specs.items())
        if isinstance(spec, Mapping)
    )
    return MutationVariantDescriptor(
        key=f"{source_alias}:{record.content_hash}:{record.container.asset_id}",
        source_row=row,
        asset_id=record.container.asset_id,
        content_hash=record.content_hash,
        mutation_mode_tokens=mode_tokens,
        tip_signature_tokens=tip_tokens,
        descriptor=representative.descriptor,
    )


def _finger_dofs(asset: RepresentativeAsset, record: ResolvedHandAssetRecord) -> tuple[int, int, int, int]:
    r"""按index/middle/ring/thumb返回mother活动关节数。"""

    semantics = record.container.geometry_semantics
    if semantics is None:
        raise ValueError(f"mother asset {asset.asset_id!r} lacks geometry semantics")
    counts = tuple(
        sum(name.startswith(f"{finger}_") for name in semantics.active_joint_names)
        for finger in ("index", "middle", "ring", "thumb")
    )
    return counts[0], counts[1], counts[2], counts[3]  # 固定canonical四指顺序，收窄tuple类型


def _resolved_lineages(
    source_alias: str,
    partition: ResolvedHandAssetPartition,
    recipe: PureLeapRightLineageRecipe,
) -> tuple[_ResolvedLineage, ...]:
    r"""从一个resolved train partition提取满足pure family边界的mother+3 variants。"""

    representative_by_row = {asset.row: asset for asset in representative_assets(partition)}
    grouped: dict[str, list[tuple[int, ResolvedHandAssetRecord]]] = defaultdict(list)
    for row, record in enumerate(partition.records):
        grouped[record.provenance.mother_path].append((row, record))  # provenance定义lineage，不猜目录层级

    lineages: list[_ResolvedLineage] = []
    for mother_path, members in sorted(grouped.items()):
        mother_items = tuple(item for item in members if item[1].provenance.asset_role == "mother")
        if len(mother_items) != 1:
            continue  # official/非lineage records不属于本pure generated选择域
        mother_row, mother_record = mother_items[0]
        provenance = mother_record.provenance
        mother = mother_record.container
        semantics = mother.geometry_semantics
        if provenance.group_name != recipe.production_group or semantics is None:
            continue
        sidecar = mother.sidecar
        slot_map = sidecar.get("slot_family_map")
        if (
            semantics.handedness != recipe.handedness
            or sidecar.get("family") != recipe.family
            or sidecar.get("family_composition") != "single_family"
            or not isinstance(slot_map, Mapping)
        ):
            continue
        finger_dofs = _finger_dofs(representative_by_row[mother_row], mother_record)
        surviving_fingers = tuple(
            finger for finger, dof in zip(("index", "middle", "ring", "thumb"), finger_dofs, strict=True) if dof > 0
        )
        if any(slot_map.get(finger) != recipe.family for finger in surviving_fingers):
            continue  # 顶层family不足以证明pure composition，逐surviving slot复验

        variants = tuple(item for item in members if item[1].provenance.asset_role == "variant")
        if len(variants) < recipe.members_per_lineage - 1:
            raise ValueError(f"lineage {mother_path!r} lacks three representative variants")
        representative = representative_by_row[mother_row]
        variant_descriptors = tuple(
            _variant_descriptor(source_alias, row, record, representative_by_row[row]) for row, record in variants
        )
        chosen_variants = select_diverse_variants(
            representative.descriptor,
            variant_descriptors,
            count=recipe.members_per_lineage - 1,
        )
        record_by_row = {row: record for row, record in variants}
        chosen_members = (
            (mother_row, mother_record),
            *((variant.source_row, record_by_row[variant.source_row]) for variant in chosen_variants),
        )
        missing_slots = tuple(
            finger for finger, dof in zip(("index", "middle", "ring", "thumb"), finger_dofs, strict=True) if dof == 0
        )
        descriptor = LineageDescriptor(
            key=f"{source_alias}:{mother_record.content_hash}:{mother.asset_id}",
            source_alias=source_alias,
            mother_row=mother_row,
            mother_name=provenance.mother_name,
            mother_asset_id=mother.asset_id,
            mother_content_hash=mother_record.content_hash,
            static_geometry_fingerprint=geometry_fingerprint_from_sidecar(mother.sidecar_path),
            cell=representative.cell,
            topology=representative.topology,
            missing_slots=missing_slots,
            finger_dofs=finger_dofs,
            descriptor=representative.descriptor,
        )
        member_rows = tuple(row for row, _record in chosen_members)
        lineages.append(
            _ResolvedLineage(
                descriptor=descriptor,
                member_rows=(member_rows[0], member_rows[1], member_rows[2], member_rows[3]),
                member_records=tuple(record for _row, record in chosen_members),
            )
        )
    return tuple(lineages)


def resolve_lineage_cohort_selection(
    recipe: PureLeapRightLineageRecipe,
    *,
    source_manifests: Mapping[str, str | Path],
) -> ResolvedLineageCohortSelection:
    r"""解析source manifests、执行cell配额选择并形成可持久化审计文档。"""

    required_aliases = tuple(quota.source_alias for quota in recipe.source_quotas)
    if set(source_manifests) != set(required_aliases):
        raise ValueError("source manifests must exactly match recipe source aliases")
    partitions: dict[str, ResolvedHandAssetPartition] = {}
    source_sha256s: dict[str, str] = {}
    lineages_by_source: dict[str, tuple[_ResolvedLineage, ...]] = {}
    for alias in required_aliases:
        dataset = HandAssetDataset.from_yaml(source_manifests[alias])
        partition, _cache_hit = resolve_prepared_train(dataset, require_geometry_semantics=True)
        partitions[alias] = partition
        source_sha256s[alias] = dataset.source_sha256
        lineages_by_source[alias] = _resolved_lineages(alias, partition, recipe)

    # 留出针对运动学母体，而不是某几个variant。先核对名称存在，再在选择前移除，仍满足原cell配额。
    excluded = set(recipe.excluded_mother_names)  # 本轮研究的policy-unseen lineage集合
    available_mothers = {
        lineage.descriptor.mother_name for lineages in lineages_by_source.values() for lineage in lineages
    }
    if unknown := excluded - available_mothers:
        raise ValueError(f"excluded mother names were not found in source train candidates: {sorted(unknown)}")

    selected: list[_ResolvedLineage] = []
    selected_topologies: set[str] = set()
    for source_quota in recipe.source_quotas:
        for cell in SUPPORTED_CELL_VALUES:
            quota = source_quota.for_cell(cell)
            if quota == 0:
                continue
            pool = [
                lineage
                for lineage in lineages_by_source[source_quota.source_alias]
                if lineage.descriptor.cell == cell
                and lineage.descriptor.mother_name not in excluded
                and (not recipe.require_unique_topology or lineage.descriptor.topology not in selected_topologies)
            ]
            prior = tuple(lineage.descriptor for lineage in selected if lineage.descriptor.cell == cell)
            chosen_descriptors = select_diverse_lineages(
                tuple(lineage.descriptor for lineage in pool), quota=quota, prior=prior
            )
            by_key = {lineage.descriptor.key: lineage for lineage in pool}
            chosen = tuple(by_key[descriptor.key] for descriptor in chosen_descriptors)
            selected.extend(chosen)  # source→cell→hash-tie顺序定义stable cohort prefix
            selected_topologies.update(lineage.descriptor.topology for lineage in chosen)

    if len(selected) != recipe.mother_count:
        raise RuntimeError(f"selector produced {len(selected)} mothers, expected {recipe.mother_count}")
    coordinates = tuple((lineage.descriptor.source_alias, row) for lineage in selected for row in lineage.member_rows)
    records = tuple(record for lineage in selected for record in lineage.member_records)
    asset_ids = tuple(record.container.asset_id for record in records)
    content_hashes = tuple(record.content_hash for record in records)
    static_fingerprints = tuple(geometry_fingerprint_from_sidecar(record.container.sidecar_path) for record in records)
    if len(set(coordinates)) != recipe.asset_count or len(set(asset_ids)) != recipe.asset_count:
        raise ValueError("selected lineage cohort duplicates source coordinates or asset IDs")
    if len(set(content_hashes)) != recipe.asset_count or len(set(static_fingerprints)) != recipe.asset_count:
        raise ValueError("selected lineage cohort duplicates content or build-time static geometry identity")

    selected_documents = []
    member_cursor = 0
    for lineage in selected:
        descriptor = lineage.descriptor
        member_documents = []
        for row, record in zip(lineage.member_rows, lineage.member_records, strict=True):
            member_documents.append(
                {
                    "cohort_index": member_cursor,
                    "source_key": f"{descriptor.source_alias}#{row}",
                    "asset_id": record.container.asset_id,
                    "content_hash": record.content_hash,
                    "static_geometry_fingerprint": static_fingerprints[member_cursor],
                    "asset_role": record.provenance.asset_role,
                }
            )
            member_cursor += 1
        selected_documents.append(
            {
                "source_alias": descriptor.source_alias,
                "mother_name": descriptor.mother_name,
                "mother_source_key": f"{descriptor.source_alias}#{descriptor.mother_row}",
                "mother_asset_id": descriptor.mother_asset_id,
                "mother_content_hash": descriptor.mother_content_hash,
                "mother_static_geometry_fingerprint": descriptor.static_geometry_fingerprint,
                "cell": {"tip_count": descriptor.cell[0], "thumb_dof": descriptor.cell[1]},
                "topology": descriptor.topology,
                "missing_slots": list(descriptor.missing_slots),
                "finger_dofs": list(descriptor.finger_dofs),
                "members": member_documents,
            }
        )
    recipe_document = asdict(recipe)  # 新排除名单随resolved lock冻结，避免重新选择时忘记研究留出。
    if not excluded:
        recipe_document.pop("excluded_mother_names")  # 已发布的无排除recipe保留原有序列化身份。
    selection_document = {
        "schema_version": LINEAGE_COHORT_SELECTION_SCHEMA_VERSION,
        "algorithm": "pure-lineage-cell-combinatorial-plus-variant-greedy-diversity-v2",
        "selection_seed": recipe.selection_seed,
        "recipe": recipe_document,
        "source_manifest_sha256s": source_sha256s,
        "tie_break": "ascending-(mother-content-hash,mother-asset-id); manifest-row-excluded",
        "distance": "cell-local-population-standardized-representative-physical-descriptor-l2",
        "selected_lineages": selected_documents,
        "overlap_certificate": {
            "asset_ids_unique": True,
            "content_hashes_unique": True,
            "static_geometry_fingerprints_unique": True,
            "canonical_physical_geometry_hashes": "pending-canonical-lowering",
        },
    }
    return ResolvedLineageCohortSelection(
        recipe=recipe,
        member_coordinates=coordinates,
        selection_document=selection_document,
    )


def write_lineage_cohort_lock(
    path: str | Path,
    *,
    recipe: PureLeapRightLineageRecipe,
    source_manifests: Mapping[str, str | Path],
) -> Path:
    r"""执行确定性lineage selector并通过统一writer原子发布schema-1.1 lock。"""

    resolved = resolve_lineage_cohort_selection(recipe, source_manifests=source_manifests)
    return write_hand_asset_cohort_lock(
        path,
        cohort_id=recipe.cohort_id,
        source_manifests=source_manifests,
        member_coordinates=resolved.member_coordinates,
        selection=resolved.selection_document,
        require_geometry_semantics=True,
    )


__all__ = [
    "LINEAGE_COHORT_SELECTION_SCHEMA_VERSION",
    "PURE_LEAP_RIGHT_A64_RECIPE",
    "PURE_LEAP_RIGHT_A128_RECIPE",
    "LineageDescriptor",
    "MutationVariantDescriptor",
    "PureLeapRightLineageRecipe",
    "ResolvedLineageCohortSelection",
    "SourceCellMotherQuota",
    "resolve_lineage_cohort_selection",
    "select_diverse_lineages",
    "select_diverse_variants",
    "write_lineage_cohort_lock",
]
