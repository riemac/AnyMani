"Plans deterministic mirror-pair, family, topology, and degree-of-freedom allocations from available inventory."

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from ...bank.path_utils import resolve_bank_path
from .schema import DatasetBuildTemplateCfg, DatasetRoleCfg

SELECTION_LOCK_SCHEMA_VERSION = "1.0.0"
"Version of the source-qualified cohort-selection lock."

_MACRO_ORDER = (
    "single_allegro",
    "single_leap",
    "mixed_allegro_base",
    "mixed_leap_base",
)
_SHAPE_ORDER = ("full", "missing")


@dataclass(frozen=True)
class MotherInventoryRecord:
    "Inventory facts for one generated mother and its canonical mirror partner."

    asset_id: str
    relative_dir: str
    collection_kind: Literal["groups", "mixed"]
    group_name: str
    mother_name: str
    handedness: Literal["left", "right"]
    base_family: Literal["allegro", "leap"]
    family_composition: Literal["single_family", "mixed"]
    macro_family: str
    topology_shape: Literal["full", "missing"]
    missing_slots: tuple[str, ...]
    dof: int
    finger_count: int
    slot_family_map: tuple[tuple[str, str], ...]
    selected_slot_recipes: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class CanonicalMotherPair:
    "Left-right mirror pair kept together during partition selection."

    pair_key: str
    left: MotherInventoryRecord
    right: MotherInventoryRecord

    @property
    def macro_family(self) -> str:
        "Returns the single-family or mixed-family balance stratum for one topology."

        return self.left.macro_family

    @property
    def topology_shape(self) -> str:
        "Returns whether a topology is complete or has one or more missing slots."

        return self.left.topology_shape

    @property
    def dof(self) -> int:
        "Returns the active-DoF stratum for one mother pair."

        return self.left.dof

    def members(self) -> tuple[MotherInventoryRecord, MotherInventoryRecord]:
        "Returns the source rows contributed by one planned lineage."

        return self.left, self.right


@dataclass(frozen=True)
class PlannedLineage:
    "Selected lineage with its partition, mutation budget, and deterministic retry seeds."

    role: str
    pair_key: str
    mother: MotherInventoryRecord
    include_mother: bool
    assets_per_lineage: int
    variant_count: int
    mutation_seed: int

    @property
    def task_id(self) -> str:
        "Builds a stable identifier for one partition, suite, lineage, and retry."

        return f"{self.role}:{self.mother.asset_id}"


@dataclass(frozen=True)
class DatasetSelectionPlan:
    "Ordered selection plan with capacity and balance evidence."

    template_id: str
    template_sha256: str
    inventory_run_dir: str
    inventory_summary_sha256: str
    selection_seed: int
    mutation_seed: int
    pairs: Mapping[str, tuple[CanonicalMotherPair, ...]]
    lineages: Mapping[str, tuple[PlannedLineage, ...]]
    ppo_train_pair_keys: tuple[str, ...]
    quota_report: Mapping[str, Any]
    generator_config_module: str = ""
    generator_config_sha256: str = ""
    generator_config_snapshot: Mapping[str, Any] | None = None
    git_commit: str = ""
    git_dirty: bool = False

    def to_lock_document(self) -> dict[str, Any]:
        'Serializes lock document.'

        return {
            "schema_version": SELECTION_LOCK_SCHEMA_VERSION,
            "template_id": self.template_id,
            "template_sha256": self.template_sha256,
            "inventory": {
                "run_dir": self.inventory_run_dir,
                "summary_sha256": self.inventory_summary_sha256,
            },
            "seeds": {"selection": self.selection_seed, "mutation": self.mutation_seed},
            "generator": {
                "config_module": self.generator_config_module,
                "config_sha256": self.generator_config_sha256,
                "config_snapshot": dict(self.generator_config_snapshot or {}),
            },
            "code": {"git_commit": self.git_commit, "git_dirty": self.git_dirty},
            "quota_report": dict(self.quota_report),
            "cohorts": {
                role: [pair.pair_key for pair in pairs]
                for role, pairs in self.pairs.items()
            },
            "ppo_train_pair_keys": list(self.ppo_train_pair_keys),
            "lineages": {
                role: [
                    {
                        "task_id": lineage.task_id,
                        "pair_key": lineage.pair_key,
                        "mother": asdict(lineage.mother),
                        "include_mother": lineage.include_mother,
                        "assets_per_lineage": lineage.assets_per_lineage,
                        "variant_count": lineage.variant_count,
                        "mutation_seed": lineage.mutation_seed,
                    }
                    for lineage in role_lineages
                ]
                for role, role_lineages in self.lineages.items()
            },
        }


def build_dataset_selection_plan(
    template: DatasetBuildTemplateCfg,
    *,
    template_sha256: str,
    generator_config_module: str = "",
    generator_config_snapshot: Mapping[str, Any] | None = None,
    git_commit: str = "",
    git_dirty: bool = False,
) -> DatasetSelectionPlan:
    'Builds dataset selection plan.'

    run_root = resolve_bank_path(template.inventory.run_dir)
    config_snapshot = dict(generator_config_snapshot or {})
    config_sha256 = hashlib.sha256(
        yaml.safe_dump(config_snapshot, allow_unicode=True, sort_keys=True).encode()
    ).hexdigest()
    records, summary_sha256 = scan_mother_inventory(run_root)
    pair_inventory = build_canonical_mirror_pairs(records)


    new_role_cfg = {
        "train": template.partitions.train,
        "validation.unseen_mother": template.partitions.validation.unseen_mother,
        "evaluation.unseen_mother": template.partitions.evaluation.unseen_mother,
    }
    new_pairs, new_quota_report = _select_disjoint_new_mother_cohorts(
        pair_inventory,
        role_cfg=new_role_cfg,
        template=template,
    )


    train_pairs = new_pairs["train"]
    ppo_pair_count = template.manifests.ppo.train_mother_count // 2
    ppo_pairs = _select_one_cohort(
        train_pairs,
        pair_count=ppo_pair_count,
        template=template,
        domain="ppo.train",
    )
    validation_seen_count = template.partitions.validation.unseen_variant_set.mother_count // 2
    evaluation_seen_count = template.partitions.evaluation.unseen_variant_set.mother_count // 2
    if validation_seen_count + evaluation_seen_count > len(ppo_pairs):
        raise ValueError("PPO train cohort cannot cover both seen-mother validation/evaluation suites")
    validation_seen, evaluation_seen = _split_seen_mother_cohorts(
        ppo_pairs,
        validation_pair_count=validation_seen_count,
        evaluation_pair_count=evaluation_seen_count,
        template=template,
    )

    pairs: dict[str, tuple[CanonicalMotherPair, ...]] = {
        "train": new_pairs["train"],
        "validation.unseen_variant_set": validation_seen,
        "validation.unseen_mother": new_pairs["validation.unseen_mother"],
        "evaluation.unseen_variant_set": evaluation_seen,
        "evaluation.unseen_mother": new_pairs["evaluation.unseen_mother"],
    }
    role_cfg = {
        "train": template.partitions.train,
        "validation.unseen_variant_set": template.partitions.validation.unseen_variant_set,
        "validation.unseen_mother": template.partitions.validation.unseen_mother,
        "evaluation.unseen_variant_set": template.partitions.evaluation.unseen_variant_set,
        "evaluation.unseen_mother": template.partitions.evaluation.unseen_mother,
    }
    include_mother = {
        "train": True,
        "validation.unseen_variant_set": False,
        "validation.unseen_mother": True,
        "evaluation.unseen_variant_set": False,
        "evaluation.unseen_mother": True,
    }
    lineages = {
        role: _planned_lineages(
            selected_pairs,
            role=role,
            cfg=role_cfg[role],
            include_mother=include_mother[role],
            mutation_root_seed=template.seeds.mutation,
        )
        for role, selected_pairs in pairs.items()
    }
    quota_report = {
        "inventory": _pair_distribution(pair_inventory),
        "new_mother_cohorts": new_quota_report,
        "selected": {role: _pair_distribution(selected) for role, selected in pairs.items()},
        "ppo_train": _pair_distribution(ppo_pairs),
        "planned_variants": sum(
            lineage.variant_count for role_lineages in lineages.values() for lineage in role_lineages
        ),
        "planned_assets": sum(
            lineage.assets_per_lineage for role_lineages in lineages.values() for lineage in role_lineages
        ),
    }
    return DatasetSelectionPlan(
        template_id=template.template_id,
        template_sha256=template_sha256,
        inventory_run_dir=template.inventory.run_dir,
        inventory_summary_sha256=summary_sha256,
        selection_seed=template.seeds.selection,
        mutation_seed=template.seeds.mutation,
        pairs=pairs,
        lineages=lineages,
        ppo_train_pair_keys=tuple(pair.pair_key for pair in ppo_pairs),
        quota_report=quota_report,
        generator_config_module=generator_config_module,
        generator_config_sha256=config_sha256,
        generator_config_snapshot=config_snapshot,
        git_commit=git_commit,
        git_dirty=git_dirty,
    )


def scan_mother_inventory(run_root: Path) -> tuple[tuple[MotherInventoryRecord, ...], str]:
    "Reads mother sidecars and forms deterministic morphology inventory records."

    summary_path = run_root / "summary.yaml"
    if not summary_path.is_file():
        raise FileNotFoundError(f"generation run lacks summary.yaml: {run_root}")
    summary_bytes = summary_path.read_bytes()
    summary = yaml.safe_load(summary_bytes) or {}
    if not isinstance(summary, Mapping) or summary.get("run", {}).get("mode") != "made":
        raise ValueError(f"dataset inventory must be a pre-made generation run: {run_root}")

    records: list[MotherInventoryRecord] = []
    for sidecar_path in sorted(run_root.rglob("hand.yaml")):
        relative = sidecar_path.relative_to(run_root)
        is_group_mother = len(relative.parts) == 3 and relative.parts[0] != "mixed"
        is_mixed_mother = len(relative.parts) == 4 and relative.parts[0] == "mixed"
        if not (is_group_mother or is_mixed_mother):
            continue
        sidecar = _load_selection_metadata(sidecar_path)
        if not isinstance(sidecar, Mapping):
            raise TypeError(f"mother sidecar must be a mapping: {sidecar_path}")
        records.append(_mother_record(run_root, sidecar_path.parent, sidecar))

    expected = int(summary.get("stats", {}).get("succeeded", -1))
    if expected != len(records):
        raise ValueError(f"inventory summary succeeded={expected} does not match mother bundles={len(records)}")
    return tuple(records), hashlib.sha256(summary_bytes).hexdigest()


def _load_selection_metadata(sidecar_path: Path) -> Mapping[str, Any]:

    prefix: list[str] = []
    with sidecar_path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("per_finger_connectivity:"):
                break
            prefix.append(line)
    document = yaml.safe_load("".join(prefix)) or {}
    if not isinstance(document, Mapping):
        raise TypeError(f"mother selection metadata must be a mapping: {sidecar_path}")
    return document


def build_canonical_mirror_pairs(
    records: Sequence[MotherInventoryRecord],
) -> tuple[CanonicalMotherPair, ...]:
    'Builds canonical mirror pairs.'

    grouped: dict[str, dict[str, MotherInventoryRecord]] = defaultdict(dict)
    for record in records:
        canonical_name = _canonical_mother_name(record.mother_name, record.handedness)
        key = f"{record.collection_kind}/{record.group_name}/{canonical_name}"
        if record.handedness in grouped[key]:
            raise ValueError(f"canonical pair {key!r} contains duplicate {record.handedness} mother")
        grouped[key][record.handedness] = record

    pairs: list[CanonicalMotherPair] = []
    for pair_key, members in sorted(grouped.items()):
        if set(members) != {"left", "right"}:
            raise ValueError(f"canonical pair {pair_key!r} must contain exactly left and right mothers")
        left = members["left"]
        right = members["right"]
        if _pair_semantics(left) != _pair_semantics(right):
            raise ValueError(f"canonical pair {pair_key!r} disagrees beyond handedness")
        pairs.append(CanonicalMotherPair(pair_key=pair_key, left=left, right=right))
    return tuple(pairs)


def write_selection_lock(plan: DatasetSelectionPlan, path: str | Path) -> Path:
    'Writes selection lock.'

    output_path = Path(path)
    payload = yaml.safe_dump(plan.to_lock_document(), allow_unicode=True, sort_keys=False)
    if output_path.exists():
        existing = output_path.read_text(encoding="utf-8")
        if existing != payload:
            raise FileExistsError(f"selection lock already exists with different content: {output_path}")
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(output_path)
    return output_path


def _select_disjoint_new_mother_cohorts(
    inventory: Sequence[CanonicalMotherPair],
    *,
    role_cfg: Mapping[str, DatasetRoleCfg],
    template: DatasetBuildTemplateCfg,
) -> tuple[dict[str, tuple[CanonicalMotherPair, ...]], dict[str, Any]]:

    role_pair_counts = {role: cfg.mother_count // 2 for role, cfg in role_cfg.items()}
    macro_capacity = Counter(pair.macro_family for pair in inventory)
    macro_total = _allocate_capped(
        total=sum(role_pair_counts.values()),
        weights=template.balance.macro_family,
        capacities=macro_capacity,
        seed=template.seeds.selection,
        domain="new-mothers/macro-total",
    )
    role_macro: dict[str, dict[str, int]] = {role: {} for role in role_cfg}
    for macro in _MACRO_ORDER:
        allocation = _allocate_capped(
            total=macro_total[macro],
            weights={role: float(count) for role, count in role_pair_counts.items()},
            capacities={role: count for role, count in role_pair_counts.items()},
            seed=template.seeds.selection,
            domain=f"new-mothers/{macro}/roles",
        )
        for role, count in allocation.items():
            role_macro[role][macro] = count
    if any(sum(macros.values()) != role_pair_counts[role] for role, macros in role_macro.items()):
        raise RuntimeError("macro apportionment did not preserve role pair totals")

    role_cell: dict[str, dict[tuple[str, str], int]] = {role: {} for role in role_cfg}
    for macro in _MACRO_ORDER:
        shape_capacity = Counter(pair.topology_shape for pair in inventory if pair.macro_family == macro)
        shape_total = _allocate_capped(
            total=macro_total[macro],
            weights=template.balance.topology_shape,
            capacities=shape_capacity,
            seed=template.seeds.selection,
            domain=f"new-mothers/{macro}/shape-total",
        )
        missing_by_role = _allocate_capped(
            total=shape_total["missing"],
            weights={role: float(role_macro[role][macro]) for role in role_cfg},
            capacities={role: role_macro[role][macro] for role in role_cfg},
            seed=template.seeds.selection,
            domain=f"new-mothers/{macro}/missing-by-role",
        )
        for role in role_cfg:
            missing = missing_by_role[role]
            role_cell[role][(macro, "missing")] = missing
            role_cell[role][(macro, "full")] = role_macro[role][macro] - missing

    selected: dict[str, list[CanonicalMotherPair]] = {role: [] for role in role_cfg}
    used: set[str] = set()
    for role in role_cfg:
        for macro in _MACRO_ORDER:
            for shape in _SHAPE_ORDER:
                count = role_cell[role][(macro, shape)]
                candidates = tuple(
                    pair
                    for pair in inventory
                    if pair.pair_key not in used
                    and pair.macro_family == macro
                    and pair.topology_shape == shape
                )
                chosen = _select_balanced_candidates(
                    candidates,
                    count=count,
                    template=template,
                    domain=f"{role}/{macro}/{shape}",
                )
                selected[role].extend(chosen)
                used.update(pair.pair_key for pair in chosen)
    frozen = {role: tuple(pairs) for role, pairs in selected.items()}
    report = {
        "macro_total": dict(macro_total),
        "role_macro": role_macro,
        "role_cells": {
            role: {f"{macro}/{shape}": count for (macro, shape), count in cells.items()}
            for role, cells in role_cell.items()
        },
    }
    return frozen, report


def _select_one_cohort(
    pool: Sequence[CanonicalMotherPair],
    *,
    pair_count: int,
    template: DatasetBuildTemplateCfg,
    domain: str,
) -> tuple[CanonicalMotherPair, ...]:

    macro_capacity = Counter(pair.macro_family for pair in pool)
    macro_quota = _allocate_capped(
        total=pair_count,
        weights=template.balance.macro_family,
        capacities=macro_capacity,
        seed=template.seeds.selection,
        domain=f"{domain}/macro",
    )
    selected: list[CanonicalMotherPair] = []
    for macro in _MACRO_ORDER:
        macro_pool = tuple(pair for pair in pool if pair.macro_family == macro)
        shape_capacity = Counter(pair.topology_shape for pair in macro_pool)
        shape_quota = _allocate_capped(
            total=macro_quota[macro],
            weights=template.balance.topology_shape,
            capacities=shape_capacity,
            seed=template.seeds.selection,
            domain=f"{domain}/{macro}/shape",
        )
        for shape in _SHAPE_ORDER:
            cell_pool = tuple(pair for pair in macro_pool if pair.topology_shape == shape)
            selected.extend(
                _select_balanced_candidates(
                    cell_pool,
                    count=shape_quota[shape],
                    template=template,
                    domain=f"{domain}/{macro}/{shape}",
                )
            )
    if len(selected) != pair_count:
        raise RuntimeError(f"cohort {domain!r} selected {len(selected)} pairs, expected {pair_count}")
    return tuple(selected)


def _split_seen_mother_cohorts(
    ppo_pairs: Sequence[CanonicalMotherPair],
    *,
    validation_pair_count: int,
    evaluation_pair_count: int,
    template: DatasetBuildTemplateCfg,
) -> tuple[tuple[CanonicalMotherPair, ...], tuple[CanonicalMotherPair, ...]]:

    joint_count = validation_pair_count + evaluation_pair_count
    if joint_count > len(ppo_pairs):
        raise ValueError("seen-mother suites exceed PPO train pair count")
    joint_pool = _select_one_cohort(
        ppo_pairs,
        pair_count=joint_count,
        template=template,
        domain="seen-mother-union",
    )
    macro_capacity = Counter(pair.macro_family for pair in joint_pool)
    validation_macro = _allocate_capped(
        total=validation_pair_count,
        weights=template.balance.macro_family,
        capacities=macro_capacity,
        seed=template.seeds.selection,
        domain="validation-seen/macro",
    )
    shape_capacity = Counter(pair.topology_shape for pair in joint_pool)
    validation_shape = _allocate_capped(
        total=validation_pair_count,
        weights={shape: float(capacity) for shape, capacity in shape_capacity.items()},
        capacities=shape_capacity,
        seed=template.seeds.selection,
        domain="validation-seen/shape",
    )
    missing_capacity_by_macro = Counter(
        pair.macro_family for pair in joint_pool if pair.topology_shape == "missing"
    )
    validation_missing = _allocate_capped(
        total=validation_shape.get("missing", 0),
        weights={macro: float(max(missing_capacity_by_macro.get(macro, 0), 1)) for macro in _MACRO_ORDER},
        capacities={
            macro: min(missing_capacity_by_macro.get(macro, 0), validation_macro.get(macro, 0))
            for macro in _MACRO_ORDER
        },
        seed=template.seeds.selection,
        domain="validation-seen/missing-by-macro",
    )

    validation: list[CanonicalMotherPair] = []
    for macro in _MACRO_ORDER:
        cell_quota = {
            "missing": validation_missing[macro],
            "full": validation_macro[macro] - validation_missing[macro],
        }
        for shape in _SHAPE_ORDER:
            candidates = tuple(
                pair
                for pair in joint_pool
                if pair.macro_family == macro and pair.topology_shape == shape
            )
            validation.extend(
                _select_balanced_candidates(
                    candidates,
                    count=cell_quota[shape],
                    template=template,
                    domain=f"validation.unseen_variant_set/{macro}/{shape}",
                )
            )
    validation_keys = {pair.pair_key for pair in validation}
    evaluation = tuple(pair for pair in joint_pool if pair.pair_key not in validation_keys)
    if len(validation) != validation_pair_count or len(evaluation) != evaluation_pair_count:
        raise RuntimeError("seen-mother split did not preserve validation/evaluation pair totals")
    return tuple(validation), evaluation


def _select_balanced_candidates(
    candidates: Sequence[CanonicalMotherPair],
    *,
    count: int,
    template: DatasetBuildTemplateCfg,
    domain: str,
) -> tuple[CanonicalMotherPair, ...]:

    if count == 0:
        return ()
    if count > len(candidates):
        raise ValueError(f"selection cell {domain!r} requires {count} pairs but only {len(candidates)} are available")
    grouped: dict[str, list[CanonicalMotherPair]] = defaultdict(list)
    for pair in candidates:
        grouped[_secondary_group(pair)].append(pair)
    group_quota = _allocate_capped(
        total=count,
        weights={name: 1.0 for name in grouped},
        capacities={name: len(items) for name, items in grouped.items()},
        seed=template.seeds.selection,
        domain=f"{domain}/secondary",
    )

    selected: list[CanonicalMotherPair] = []
    for group_name in sorted(grouped):
        group_candidates = grouped[group_name]
        dof_groups: dict[int, list[CanonicalMotherPair]] = defaultdict(list)
        for pair in group_candidates:
            dof_groups[pair.dof].append(pair)
        dof_quota = _allocate_capped(
            total=group_quota[group_name],
            weights={str(dof): 1.0 for dof in dof_groups},
            capacities={str(dof): len(items) for dof, items in dof_groups.items()},
            seed=template.seeds.selection,
            domain=f"{domain}/{group_name}/dof",
        )
        for dof, bucket in sorted(dof_groups.items()):
            ranked = sorted(
                bucket,
                key=lambda pair: _stable_rank(template.seeds.selection, f"{domain}/{group_name}/{dof}", pair.pair_key),
            )
            selected.extend(ranked[: dof_quota[str(dof)]])
    if len(selected) != count:
        raise RuntimeError(f"balanced selector {domain!r} selected {len(selected)} pairs, expected {count}")
    return tuple(selected)


def _allocate_capped(
    *,
    total: int,
    weights: Mapping[str, float],
    capacities: Mapping[str, int],
    seed: int,
    domain: str,
) -> dict[str, int]:

    keys = tuple(str(key) for key in weights)
    capacity_keys = {str(key) for key in capacities}
    if capacity_keys - set(keys):
        raise ValueError(f"allocation {domain!r} capacities contain undeclared weight keys")
    normalized_weights = {str(key): float(value) for key, value in weights.items()}
    normalized_capacities = {key: int(capacities.get(key, 0)) for key in keys}
    if total < 0 or sum(normalized_capacities.values()) < total:
        raise ValueError(f"allocation {domain!r} capacity cannot satisfy total={total}")
    result = {key: 0 for key in keys}
    remaining = total
    while remaining > 0:
        active = [key for key in keys if result[key] < normalized_capacities[key]]
        if not active:
            raise ValueError(f"allocation {domain!r} exhausted all capacities")
        weight_sum = sum(normalized_weights[key] for key in active)
        raw = {key: remaining * normalized_weights[key] / weight_sum for key in active}
        progressed = 0
        for key in active:
            increment = min(int(raw[key]), normalized_capacities[key] - result[key])
            result[key] += increment
            progressed += increment
        remaining -= progressed
        if remaining == 0:
            break
        ranked = sorted(
            (key for key in active if result[key] < normalized_capacities[key]),
            key=lambda key: (
                -(raw[key] - int(raw[key])),
                _stable_rank(seed, domain, key),
            ),
        )
        if not ranked:
            continue
        for key in ranked:
            if remaining == 0:
                break
            result[key] += 1
            remaining -= 1
    return result


def _planned_lineages(
    pairs: Sequence[CanonicalMotherPair],
    *,
    role: str,
    cfg: DatasetRoleCfg,
    include_mother: bool,
    mutation_root_seed: int,
) -> tuple[PlannedLineage, ...]:

    variant_count = cfg.assets_per_lineage - int(include_mother)
    lineages: list[PlannedLineage] = []
    for pair in pairs:
        for mother in pair.members():
            lineages.append(
                PlannedLineage(
                    role=role,
                    pair_key=pair.pair_key,
                    mother=mother,
                    include_mother=include_mother,
                    assets_per_lineage=cfg.assets_per_lineage,
                    variant_count=variant_count,
                    mutation_seed=_derive_seed(mutation_root_seed, role, mother.asset_id, retry_round=0),
                )
            )
    if len(lineages) != cfg.mother_count:
        raise RuntimeError(f"role {role!r} planned {len(lineages)} mothers, expected {cfg.mother_count}")
    return tuple(lineages)


def derive_retry_seed(root_seed: int, role: str, asset_id: str, retry_round: int) -> int:
    'Derives retry seed.'

    if retry_round < 0:
        raise ValueError("retry_round must be non-negative")
    return _derive_seed(root_seed, role, asset_id, retry_round=retry_round)


def select_pair_subset(
    pool: Sequence[CanonicalMotherPair],
    *,
    mother_count: int,
    template: DatasetBuildTemplateCfg,
    domain: str,
) -> tuple[CanonicalMotherPair, ...]:
    'Selects pair subset.'

    if mother_count < 0 or mother_count % 2 != 0:
        raise ValueError("pair subset mother_count must be a non-negative even number")
    return _select_one_cohort(pool, pair_count=mother_count // 2, template=template, domain=domain)


def _derive_seed(root_seed: int, role: str, asset_id: str, *, retry_round: int) -> int:

    payload = f"{root_seed}\0{role}\0{asset_id}\0{retry_round}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1) + 1


def _mother_record(run_root: Path, mother_root: Path, sidecar: Mapping[str, Any]) -> MotherInventoryRecord:

    relative_dir = mother_root.relative_to(run_root)
    collection_kind: Literal["groups", "mixed"] = "mixed" if relative_dir.parts[0] == "mixed" else "groups"
    group_name = relative_dir.parts[1] if collection_kind == "mixed" else relative_dir.parts[0]
    mother_name = relative_dir.parts[-1]
    handedness = str(sidecar.get("handedness"))
    if handedness not in {"left", "right"}:
        raise ValueError(f"mother {relative_dir} has invalid handedness={handedness!r}")
    base_family = str(sidecar.get("family"))
    family_composition = str(sidecar.get("family_composition"))
    if base_family not in {"allegro", "leap"} or family_composition not in {"single_family", "mixed"}:
        raise ValueError(f"mother {relative_dir} lacks normalized family composition provenance")
    missing_slots = tuple(str(slot) for slot in sidecar.get("missing_slots", ()))
    macro = f"single_{base_family}" if family_composition == "single_family" else f"mixed_{base_family}_base"
    return MotherInventoryRecord(
        asset_id=str(sidecar["id"]),
        relative_dir=relative_dir.as_posix(),
        collection_kind=collection_kind,
        group_name=group_name,
        mother_name=mother_name,
        handedness=cast(Literal["left", "right"], handedness),
        base_family=cast(Literal["allegro", "leap"], base_family),
        family_composition=cast(Literal["single_family", "mixed"], family_composition),
        macro_family=macro,
        topology_shape="missing" if missing_slots else "full",
        missing_slots=missing_slots,
        dof=int(sidecar["dof"]),
        finger_count=int(sidecar["finger_count"]),
        slot_family_map=tuple(sorted((str(key), str(value)) for key, value in sidecar.get("slot_family_map", {}).items())),
        selected_slot_recipes=tuple(sorted((str(key), str(value)) for key, value in sidecar.get("selected_slot_recipes", {}).items())),
    )


def _canonical_mother_name(name: str, handedness: str) -> str:

    prefix = f"{handedness}_"
    if not name.startswith(prefix):
        raise ValueError(f"mother name {name!r} does not start with handedness prefix {prefix!r}")
    return name[len(prefix) :]


def _pair_semantics(record: MotherInventoryRecord) -> tuple[Any, ...]:

    return (
        record.collection_kind,
        record.group_name,
        record.base_family,
        record.family_composition,
        record.macro_family,
        record.topology_shape,
        record.missing_slots,
        record.dof,
        record.finger_count,
        record.slot_family_map,
        record.selected_slot_recipes,
    )


def _secondary_group(pair: CanonicalMotherPair) -> str:

    if pair.left.family_composition == "mixed":
        return pair.left.group_name
    if pair.left.missing_slots:
        return "+".join(pair.left.missing_slots)
    return "full"


def _stable_rank(seed: int, domain: str, identity: str) -> str:

    return hashlib.sha256(f"{seed}\0{domain}\0{identity}".encode()).hexdigest()


def _pair_distribution(pairs: Sequence[CanonicalMotherPair]) -> dict[str, Any]:

    return {
        "pair_count": len(pairs),
        "mother_count": 2 * len(pairs),
        "macro_family": dict(sorted(Counter(pair.macro_family for pair in pairs).items())),
        "topology_shape": dict(sorted(Counter(pair.topology_shape for pair in pairs).items())),
        "dof": dict(sorted(Counter(pair.dof for pair in pairs).items())),
        "secondary_group": dict(sorted(Counter(_secondary_group(pair) for pair in pairs).items())),
    }


__all__ = [
    "CanonicalMotherPair",
    "DatasetSelectionPlan",
    "MotherInventoryRecord",
    "PlannedLineage",
    "SELECTION_LOCK_SCHEMA_VERSION",
    "build_canonical_mirror_pairs",
    "build_dataset_selection_plan",
    "derive_retry_seed",
    "select_pair_subset",
    "scan_mother_inventory",
    "write_selection_lock",
]
