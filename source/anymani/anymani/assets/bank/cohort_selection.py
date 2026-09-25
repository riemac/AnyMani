"Selects balanced lineages and variants with deterministic morphology rules and writes source-qualified locks."

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
"Version of the typed lineage-selection lock."


@dataclass(frozen=True)
class SourceCellMotherQuota:
    "Capacity-constrained quota for canonical mirror pairs in one morphology cell."

    source_alias: str
    counts: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        "Normalizes typed inputs and enforces the class invariants."

        if not self.source_alias.strip() or "#" in self.source_alias:
            raise ValueError("source quota alias must be non-empty and '#' free")
        if len(self.counts) != len(SUPPORTED_CELL_VALUES) or any(count < 0 for count in self.counts):
            raise ValueError("source quota must contain four non-negative cell counts")

    def for_cell(self, cell: tuple[int, int]) -> int:
        "Returns the quota and selected mirror pairs for one morphology balance cell."

        try:
            return self.counts[SUPPORTED_CELL_VALUES.index(cell)]
        except ValueError as error:
            raise ValueError(f"unsupported lineage cohort cell {cell}") from error


@dataclass(frozen=True)
class PureFamilyLineageRecipe:
    "Family-only lineage selection recipe with explicit topology and degree-of-freedom balance."

    cohort_id: str
    source_quotas: tuple[SourceCellMotherQuota, ...]
    members_per_lineage: int = 4
    selection_seed: int = 20260904
    production_group: str = "single_palm_leap"
    handedness: str = "right"
    family: str = "leap"
    require_unique_topology: bool = True
    excluded_mother_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        "Normalizes typed inputs and enforces the class invariants."

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
        "Returns the number of distinct mother assets represented in the selection."

        return sum(sum(quota.counts) for quota in self.source_quotas)

    @property
    def asset_count(self) -> int:
        "Returns the number of members contributed by the selected lineages."

        return self.mother_count * self.members_per_lineage


PureLeapRightLineageRecipe = PureFamilyLineageRecipe
"Typed balance recipe for selecting LEAP-right source lineages."


@dataclass(frozen=True)
class LineageDescriptor:
    "Morphology features used to assign one source mother lineage to a balance cell."

    key: str
    source_alias: str
    mother_row: int
    mother_name: str
    mother_asset_id: str
    mother_content_hash: str
    static_geometry_fingerprint: str
    cell: tuple[int, int]
    topology: str  # handedness-neutral topology
    missing_slots: tuple[str, ...]
    finger_dofs: tuple[int, int, int, int]
    descriptor: tuple[float, ...]

    @property
    def identity_tiebreak(self) -> tuple[str, str]:
        "Returns a stable source-identity key for deterministic selection ties."

        return self.mother_content_hash, self.mother_asset_id


@dataclass(frozen=True)
class MutationVariantDescriptor:
    "Stable identity for mutation modes and samples that distinguish a variant set."

    key: str
    source_row: int
    asset_id: str
    content_hash: str
    mutation_mode_tokens: tuple[str, ...]
    tip_signature_tokens: tuple[str, ...]
    descriptor: tuple[float, ...]

    @property
    def identity_tiebreak(self) -> tuple[str, str]:
        "Returns a stable source-identity key for deterministic selection ties."

        return self.content_hash, self.asset_id


@dataclass(frozen=True)
class ResolvedLineageCohortSelection:
    "Ordered source rows and balance evidence for a resolved lineage cohort."

    recipe: PureFamilyLineageRecipe
    member_coordinates: tuple[tuple[str, int], ...]
    selection_document: Mapping[str, Any]


@dataclass(frozen=True)
class _ResolvedLineage:
    "Internal lineage rows prepared for deterministic cohort assembly."

    descriptor: LineageDescriptor
    member_rows: tuple[int, int, int, int]
    member_records: tuple[ResolvedHandAssetRecord, ...]


PURE_LEAP_RIGHT_A64_RECIPE = PureLeapRightLineageRecipe(
    cohort_id="pure-leap-right-a64",
    source_quotas=(SourceCellMotherQuota("ppo", (2, 3, 7, 4)),),
)
"Balanced recipe for selecting 64 LEAP-right assets."

PURE_LEAP_RIGHT_A128_RECIPE = PureLeapRightLineageRecipe(
    cohort_id="pure-leap-right-a128",
    source_quotas=(
        SourceCellMotherQuota("ppo", (2, 3, 7, 4)),
        SourceCellMotherQuota("ssl", (6, 5, 1, 4)),
    ),
)
"Balanced recipe for selecting 128 LEAP-right assets."


PURE_ALLEGRO_RIGHT_A128_RECIPE = PureFamilyLineageRecipe(
    cohort_id="pure-allegro-right-a128",
    source_quotas=(
        SourceCellMotherQuota("ppo", (1, 3, 4, 7)),
        SourceCellMotherQuota("ssl", (7, 5, 4, 1)),
    ),
    production_group="single_palm_allegro",
    family="allegro",
    excluded_mother_names=("right_t4_m4_r3",),
)
"Balanced recipe for selecting 128 Allegro-right assets."


def _distance(left: Sequence[float], right: Sequence[float]) -> float:

    if len(left) != len(right):
        raise ValueError("lineage descriptor distance requires equal widths")
    return math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def _standardized_vectors(lineages: Sequence[LineageDescriptor]) -> dict[str, tuple[float, ...]]:

    if not lineages or len({lineage.key for lineage in lineages}) != len(lineages):
        raise ValueError("standardization requires non-empty unique lineage keys")
    width = len(lineages[0].descriptor)
    if width < 1 or any(len(lineage.descriptor) != width for lineage in lineages):
        raise ValueError("lineage descriptors must share one non-empty width")
    columns = tuple(tuple(lineage.descriptor[index] for lineage in lineages) for index in range(width))
    means = tuple(math.fsum(column) / len(column) for column in columns)
    scales = tuple(
        max(math.sqrt(math.fsum((value - mean) ** 2 for value in column) / len(column)), 1.0e-12)
        for column, mean in zip(columns, means, strict=True)
    )
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
    'Selects diverse lineages.'

    ordered = tuple(sorted(candidates, key=lambda item: item.identity_tiebreak))
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
            float(len({lineage.missing_slots for lineage in combo})),
            float(len({lineage.finger_dofs for lineage in combo})),
            min(baseline_distances) if baseline_distances else 0.0,
            min(internal_distances) if internal_distances else 0.0,
            math.fsum(internal_distances) + nearest_baseline_sum,
        )
        if best_score is None or score > best_score:
            best_combo, best_score = combo, score
    if best_combo is None:
        raise RuntimeError("lineage combination search produced no candidate")
    return best_combo


def select_diverse_variants(
    mother_descriptor: Sequence[float],
    variants: Sequence[MutationVariantDescriptor],
    *,
    count: int = 3,
) -> tuple[MutationVariantDescriptor, ...]:
    'Selects diverse variants.'

    ordered = tuple(sorted(variants, key=lambda item: item.identity_tiebreak))
    if count < 1 or count > len(ordered):
        raise ValueError(f"variant count {count} cannot be drawn from {len(ordered)} candidates")
    width = len(mother_descriptor)
    if width < 1 or any(len(variant.descriptor) != width for variant in ordered):
        raise ValueError("mother and variant descriptors must share one non-empty width")


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
            r"""Scores this greedy candidate by discrete morphology coverage and continuous max-min separation."""

            references = (mother_vector, *(vectors[item.key] for item in selected))
            diversity = min(_distance(vectors[variant.key], reference) for reference in references)
            return (
                float(len(set(variant.mutation_mode_tokens) - covered_modes)),
                float(len(set(variant.tip_signature_tokens) - covered_tips)),
                float(variant.mutation_mode_tokens not in covered_mode_signatures),
                float(variant.tip_signature_tokens not in covered_tip_signatures),
                diversity,
            )

        chosen = max(remaining, key=score)
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

    semantics = record.container.geometry_semantics
    if semantics is None:
        raise ValueError(f"mother asset {asset.asset_id!r} lacks geometry semantics")
    counts = tuple(
        sum(name.startswith(f"{finger}_") for name in semantics.active_joint_names)
        for finger in ("index", "middle", "ring", "thumb")
    )
    return counts[0], counts[1], counts[2], counts[3]


def _resolved_lineages(
    source_alias: str,
    partition: ResolvedHandAssetPartition,
    recipe: PureFamilyLineageRecipe,
) -> tuple[_ResolvedLineage, ...]:

    representative_by_row = {asset.row: asset for asset in representative_assets(partition)}
    grouped: dict[str, list[tuple[int, ResolvedHandAssetRecord]]] = defaultdict(list)
    for row, record in enumerate(partition.records):
        grouped[record.provenance.mother_path].append((row, record))

    lineages: list[_ResolvedLineage] = []
    for mother_path, members in sorted(grouped.items()):
        mother_items = tuple(item for item in members if item[1].provenance.asset_role == "mother")
        if len(mother_items) != 1:
            continue
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
            continue

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
    recipe: PureFamilyLineageRecipe,
    *,
    source_manifests: Mapping[str, str | Path],
) -> ResolvedLineageCohortSelection:
    'Resolves lineage cohort selection.'

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


    excluded = set(recipe.excluded_mother_names)
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
            selected.extend(chosen)
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
    recipe_document = asdict(recipe)
    if not excluded:
        recipe_document.pop("excluded_mother_names")
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
    recipe: PureFamilyLineageRecipe,
    source_manifests: Mapping[str, str | Path],
) -> Path:
    'Writes lineage cohort lock.'

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
    "PURE_ALLEGRO_RIGHT_A128_RECIPE",
    "PURE_LEAP_RIGHT_A64_RECIPE",
    "PURE_LEAP_RIGHT_A128_RECIPE",
    "LineageDescriptor",
    "MutationVariantDescriptor",
    "PureFamilyLineageRecipe",
    "PureLeapRightLineageRecipe",
    "ResolvedLineageCohortSelection",
    "SourceCellMotherQuota",
    "resolve_lineage_cohort_selection",
    "select_diverse_lineages",
    "select_diverse_variants",
    "write_lineage_cohort_lock",
]
