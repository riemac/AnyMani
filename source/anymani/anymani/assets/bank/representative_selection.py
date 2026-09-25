"Selects representative cohort members using the declared morphology and physical identities."

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .dataset import ResolvedHandAssetPartition
from .hand_container import HandContainer

REPRESENTATIVE_SELECTION_SCHEMA_VERSION = "1.0.0"
"Version of the deterministic representative-selection record."

PHYSX_FINGER_ORDER = ("index", "middle", "ring", "thumb")
"Finger ordering used by the downstream articulation schema."

SUPPORTED_CELL_VALUES = ((3, 3), (3, 4), (4, 3), (4, 4))
"Discrete morphology-cell values accepted by the selector."


@dataclass(frozen=True)
class RepresentativeAsset:
    "Candidate asset with morphology features used for representative selection."

    row: int
    asset_id: str
    geometry_identity: str
    handedness: str
    tip_count: int
    thumb_dof: int
    active_dof: int
    topology: str
    family_signature: str
    asset_role: str
    descriptor: tuple[float, ...]

    @property
    def cell(self) -> tuple[int, int]:
        "Returns the morphology balance cell assigned to this representative asset."

        return self.tip_count, self.thumb_dof


@dataclass(frozen=True)
class RepresentativePair:
    "Two canonical mirror partners that remain one selection unit."

    cell: tuple[int, int]
    topology: str  # handedness-neutral topology
    family_signature: str
    left: RepresentativeAsset  # left asset
    right: RepresentativeAsset  # right asset
    descriptor: tuple[float, ...]
    reflection_distance: float


def _neutral_topology(value: str) -> str:

    for prefix in ("left_", "right_"):
        if value.startswith(prefix):
            return value[len(prefix) :]
    raise ValueError(f"topology name must start with left_/right_, got {value!r}")


def _payload_numbers(value: Any) -> tuple[float, ...]:

    if isinstance(value, bool) or value is None or isinstance(value, str):
        return ()
    if isinstance(value, int | float):
        parsed = float(value)
        return (parsed,) if math.isfinite(parsed) else ()
    if isinstance(value, Mapping):
        return tuple(number for key in sorted(value) for number in _payload_numbers(value[key]))
    if isinstance(value, Sequence):
        return tuple(number for item in value for number in _payload_numbers(item))
    return ()


def _summary(values: Sequence[float]) -> tuple[float, float, float, float]:

    if not values:
        return (0.0, 0.0, 0.0, 0.0)
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)  # population variance
    return mean, math.sqrt(variance), min(values), max(values)


def _canonical_position(position: Sequence[float], handedness: str) -> tuple[float, float, float]:

    if len(position) != 3:
        raise ValueError("hand-frame position must contain three coordinates")
    x, y, z = (float(value) for value in position)
    return (-x if handedness == "left" else x), y, z


def _family_signature(container: HandContainer) -> str:

    slot_map = container.sidecar.get("slot_family_map", {})
    if not isinstance(slot_map, Mapping):
        raise ValueError(f"asset {container.asset_id!r} has invalid slot_family_map")
    return "|".join(f"{finger}:{slot_map.get(finger, 'missing')}" for finger in PHYSX_FINGER_ORDER)


def _physical_descriptor(container: HandContainer) -> tuple[float, ...]:

    semantics = container.geometry_semantics
    if semantics is None:
        raise ValueError(f"asset {container.asset_id!r} lacks typed geometry semantics")
    handedness = semantics.handedness
    if handedness not in {"left", "right"}:
        raise ValueError(f"asset {container.asset_id!r} has unsupported handedness={handedness!r}")


    limits_by_finger: dict[str, list[tuple[float, float]]] = {finger: [] for finger in PHYSX_FINGER_ORDER}
    for joint_name, (lower, upper) in zip(
        semantics.active_joint_names,
        semantics.joint_limits_rad,
        strict=True,
    ):
        finger = joint_name.split("_", 1)[0]  # canonical finger role
        if finger in limits_by_finger:
            limits_by_finger[finger].append((float(lower), float(upper)))


    chain_length_by_finger = {finger: 0.0 for finger in PHYSX_FINGER_ORDER}
    for joint in semantics.kinematic_joints:
        finger = joint.joint_name.split("_", 1)[0]
        if finger in chain_length_by_finger:
            chain_length_by_finger[finger] += math.sqrt(math.fsum(value * value for value in joint.origin_pos_m))


    anchor_by_finger = {
        seed.finger_name: _canonical_position(seed.position_a_m, handedness)
        for seed in semantics.anchor_seeds
        if seed.finger_name in PHYSX_FINGER_ORDER
    }
    descriptor: list[float] = []
    for finger in PHYSX_FINGER_ORDER:
        limits = limits_by_finger[finger]
        centers = [(lower + upper) * 0.5 for lower, upper in limits]
        spans = [upper - lower for lower, upper in limits]
        descriptor.extend(
            (
                len(limits) / 4.0,
                *_summary(centers),
                *_summary(spans),
                chain_length_by_finger[finger],
                *anchor_by_finger.get(finger, (0.0, 0.0, 0.0)),
            )
        )


    owner_role = {owner.owner_id: owner.role for owner in semantics.owners}
    palm_numbers: list[float] = []
    tip_numbers: list[float] = []
    for component in semantics.components:
        numbers = _payload_numbers(component.geometry_payload)
        if owner_role[component.owner_id] == "palm":
            palm_numbers.extend(numbers)
        elif owner_role[component.owner_id] == "tip":
            tip_numbers.extend(numbers)
    descriptor.extend((*_summary(palm_numbers), *_summary(tip_numbers)))
    if not descriptor or not all(math.isfinite(value) for value in descriptor):
        raise ValueError(f"asset {container.asset_id!r} produced a non-finite representative descriptor")
    return tuple(descriptor)


def representative_assets(partition: ResolvedHandAssetPartition) -> tuple[RepresentativeAsset, ...]:
    "Returns the selected source rows in cohort-axis order."

    assets: list[RepresentativeAsset] = []
    for row, record in enumerate(partition.records):
        container = record.container
        semantics = container.geometry_semantics
        if semantics is None:
            raise ValueError(f"asset {container.asset_id!r} lacks geometry semantics")
        tip_count = sum(owner.role == "tip" for owner in semantics.owners)
        thumb_dof = sum(name.startswith("thumb_") for name in semantics.active_joint_names)
        cell = (tip_count, thumb_dof)
        if cell not in SUPPORTED_CELL_VALUES:
            raise ValueError(f"asset {container.asset_id!r} lies outside MVP cells: {cell}")
        assets.append(
            RepresentativeAsset(
                row=row,
                asset_id=container.asset_id,
                geometry_identity=semantics.content_hash,
                handedness=semantics.handedness,
                tip_count=tip_count,
                thumb_dof=thumb_dof,
                active_dof=len(semantics.active_joint_names),
                topology=_neutral_topology(str(semantics.topology_key or container.sidecar["topology_name"])),
                family_signature=_family_signature(container),
                asset_role=record.provenance.asset_role,
                descriptor=_physical_descriptor(container),
            )
        )
    if len({asset.row for asset in assets}) != len(assets) or len({asset.asset_id for asset in assets}) != len(assets):
        raise ValueError("representative selection requires unique formal rows and asset IDs")
    return tuple(assets)


def _standardized_descriptors(assets: Sequence[RepresentativeAsset]) -> dict[int, tuple[float, ...]]:

    if not assets:
        raise ValueError("cannot standardize an empty asset cell")
    width = len(assets[0].descriptor)
    if width < 1 or any(len(asset.descriptor) != width for asset in assets):
        raise ValueError("all representative descriptors must share one non-empty width")
    columns = tuple(tuple(asset.descriptor[index] for asset in assets) for index in range(width))
    means = tuple(math.fsum(column) / len(column) for column in columns)
    scales = tuple(
        max(math.sqrt(math.fsum((value - mean) ** 2 for value in column) / len(column)), 1.0e-12)
        for column, mean in zip(columns, means, strict=True)
    )
    return {
        asset.row: tuple(
            (value - mean) / scale
            for value, mean, scale in zip(asset.descriptor, means, scales, strict=True)
        )
        for asset in assets
    }


def _distance(left: Sequence[float], right: Sequence[float]) -> float:

    if len(left) != len(right):
        raise ValueError("descriptor distance requires equal widths")
    return math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def _pair_cell_assets(assets: Sequence[RepresentativeAsset]) -> list[RepresentativePair]:

    standardized = _standardized_descriptors(assets)
    groups: dict[tuple[str, str], list[RepresentativeAsset]] = defaultdict(list)
    for asset in assets:
        groups[(asset.topology, asset.family_signature)].append(asset)
    pairs: list[RepresentativePair] = []
    for (topology, family_signature), group in sorted(groups.items()):
        left = sorted((asset for asset in group if asset.handedness == "left"), key=lambda item: item.row)
        right_remaining = sorted((asset for asset in group if asset.handedness == "right"), key=lambda item: item.row)
        for left_asset in left:
            if not right_remaining:
                break
            right_asset = min(
                right_remaining,
                key=lambda item: (_distance(standardized[left_asset.row], standardized[item.row]), item.row),
            )
            right_remaining.remove(right_asset)
            left_vector = standardized[left_asset.row]  # canonicalized left descriptor
            right_vector = standardized[right_asset.row]  # right descriptor
            pairs.append(
                RepresentativePair(
                    cell=left_asset.cell,
                    topology=topology,
                    family_signature=family_signature,
                    left=left_asset,
                    right=right_asset,
                    descriptor=tuple(
                        0.5 * (a + b) for a, b in zip(left_vector, right_vector, strict=True)
                    ),
                    reflection_distance=_distance(left_vector, right_vector),
                )
            )
    return pairs


def _rank_pairs(pairs: Sequence[RepresentativePair]) -> tuple[RepresentativePair, ...]:

    remaining = list(pairs)
    selected: list[RepresentativePair] = []
    covered_topologies: set[str] = set()
    covered_families: set[str] = set()
    covered_dofs: set[tuple[int, int]] = set()
    while remaining:
        def score(pair: RepresentativePair) -> tuple[float, ...]:
            diversity = (
                min(_distance(pair.descriptor, chosen.descriptor) for chosen in selected)
                if selected
                else 0.0
            )
            mother_pair = float(pair.left.asset_role == "mother" and pair.right.asset_role == "mother")
            return (
                float(pair.topology not in covered_topologies),
                float(pair.family_signature not in covered_families),
                float((pair.left.active_dof, pair.right.active_dof) not in covered_dofs),
                diversity,
                mother_pair,
                -pair.reflection_distance,
                -float(pair.left.row),
                -float(pair.right.row),
            )

        chosen = max(remaining, key=score)
        remaining.remove(chosen)
        selected.append(chosen)
        covered_topologies.add(chosen.topology)
        covered_families.add(chosen.family_signature)
        covered_dofs.add((chosen.left.active_dof, chosen.right.active_dof))
    return tuple(selected)


def ranked_representative_pairs(
    assets: Sequence[RepresentativeAsset],
) -> Mapping[tuple[int, int], tuple[RepresentativePair, ...]]:
    "Ranks complete mirror pairs by morphology coverage and stable geometry identity."

    grouped: dict[tuple[int, int], list[RepresentativeAsset]] = defaultdict(list)
    for asset in assets:
        grouped[asset.cell].append(asset)
    missing = [cell for cell in SUPPORTED_CELL_VALUES if cell not in grouped]
    if missing:
        raise ValueError(f"formal train lacks representative MVP cells: {missing}")
    return {
        cell: _rank_pairs(_pair_cell_assets(tuple(grouped[cell])))
        for cell in SUPPORTED_CELL_VALUES
    }


def _asset_document(asset: RepresentativeAsset) -> dict[str, Any]:

    return {
        "row": asset.row,
        "asset_id": asset.asset_id,
        "geometry_identity": asset.geometry_identity,
        "handedness": asset.handedness,
        "active_dof": asset.active_dof,
        "asset_role": asset.asset_role,
    }


def representative_selection_document(
    partition: ResolvedHandAssetPartition,
    *,
    parent_dataset_path: str,
    parent_dataset_sha256: str,
    pairs_per_cell: int = 10,
    candidate_pairs_per_cell: int = 32,
) -> dict[str, Any]:
    "Serializes the selector recipe and its ordered representative members."

    if pairs_per_cell < 1 or candidate_pairs_per_cell < pairs_per_cell:
        raise ValueError("candidate pair count must be at least pairs_per_cell > 0")
    assets = representative_assets(partition)
    ranked = ranked_representative_pairs(assets)
    cells: list[dict[str, Any]] = []
    selected_rows: list[int] = []
    for tip_count, thumb_dof in SUPPORTED_CELL_VALUES:
        pairs = ranked[(tip_count, thumb_dof)]
        if len(pairs) < candidate_pairs_per_cell:
            raise ValueError(
                f"cell tips{tip_count}/thumb{thumb_dof} has {len(pairs)} pairs, "
                f"needs {candidate_pairs_per_cell}"
            )
        pair_documents = []
        for rank, pair in enumerate(pairs[:candidate_pairs_per_cell]):
            pair_documents.append(
                {
                    "rank": rank,
                    "topology": pair.topology,
                    "family_signature": pair.family_signature,
                    "reflection_distance": pair.reflection_distance,
                    "left": _asset_document(pair.left),
                    "right": _asset_document(pair.right),
                }
            )
            if rank < pairs_per_cell:
                selected_rows.extend((pair.left.row, pair.right.row))
        cells.append(
            {
                "label": f"tips{tip_count}_thumb{thumb_dof}dof",
                "tip_count": tip_count,
                "thumb_dof": thumb_dof,
                "selected_pair_count": pairs_per_cell,
                "candidate_pairs": pair_documents,
            }
        )
    if len(selected_rows) != pairs_per_cell * len(SUPPORTED_CELL_VALUES) * 2:
        raise RuntimeError("representative selection produced an unexpected hand count")
    return {
        "artifact_type": "anymani.hand_asset_representative_selection",
        "schema_version": REPRESENTATIVE_SELECTION_SCHEMA_VERSION,
        "selection_name": "heterogeneous_rotation_mvp80_v1",
        "parent_dataset_path": parent_dataset_path,
        "parent_dataset_sha256": parent_dataset_sha256,
        "selection_algorithm": "cell-paired-categorical-coverage-farthest-v1",
        "pairs_per_cell": pairs_per_cell,
        "candidate_pairs_per_cell": candidate_pairs_per_cell,
        "selected_asset_count": len(selected_rows),
        "initial_selected_rows": selected_rows,
        "cells": cells,
    }


def finalize_representative_selection(
    candidate_document: Mapping[str, Any],
    *,
    passed_rows: Sequence[int],
    pregrasp_catalog_root: str,
    pregrasp_summary_paths: Sequence[str],
) -> dict[str, Any]:
    "Freezes ordered representatives and mirror-pair evidence in a canonical selection lock."

    if candidate_document.get("artifact_type") != "anymani.hand_asset_representative_selection":
        raise ValueError("unexpected representative candidate artifact_type")
    if candidate_document.get("schema_version") != REPRESENTATIVE_SELECTION_SCHEMA_VERSION:
        raise ValueError("unsupported representative candidate schema_version")
    passed = {int(row) for row in passed_rows}
    if len(passed) != len(tuple(passed_rows)):
        raise ValueError("passed_rows must not contain duplicates")
    pairs_per_cell = int(candidate_document["pairs_per_cell"])
    selected_rows: list[int] = []
    selected_cells: list[dict[str, Any]] = []
    rejected_pairs: list[dict[str, Any]] = []
    for cell in candidate_document["cells"]:
        selected_pairs: list[dict[str, Any]] = []
        for pair in cell["candidate_pairs"]:
            left_row = int(pair["left"]["row"])
            right_row = int(pair["right"]["row"])
            if left_row in passed and right_row in passed and len(selected_pairs) < pairs_per_cell:
                selected_pairs.append(dict(pair))
                selected_rows.extend((left_row, right_row))
            elif left_row not in passed or right_row not in passed:
                rejected_pairs.append(
                    {
                        "cell": cell["label"],
                        "pair_rank": int(pair["rank"]),
                        "left_row": left_row,
                        "right_row": right_row,
                        "left_passed": left_row in passed,
                        "right_passed": right_row in passed,
                    }
                )
            if len(selected_pairs) == pairs_per_cell:
                break
        if len(selected_pairs) != pairs_per_cell:
            raise ValueError(
                f"cell {cell['label']!r} has only {len(selected_pairs)} passing pairs; needs {pairs_per_cell}"
            )
        selected_cells.append(
            {
                "label": cell["label"],
                "tip_count": int(cell["tip_count"]),
                "thumb_dof": int(cell["thumb_dof"]),
                "pairs": selected_pairs,
            }
        )
    if len(selected_rows) != pairs_per_cell * len(SUPPORTED_CELL_VALUES) * 2 or len(set(selected_rows)) != len(
        selected_rows
    ):
        raise RuntimeError("final representative selection must contain 80 unique rows")
    return {
        "artifact_type": "anymani.hand_asset_mvp_selection",
        "schema_version": REPRESENTATIVE_SELECTION_SCHEMA_VERSION,
        "selection_name": candidate_document["selection_name"],
        "parent_dataset_path": candidate_document["parent_dataset_path"],
        "parent_dataset_sha256": candidate_document["parent_dataset_sha256"],
        "candidate_selection_algorithm": candidate_document["selection_algorithm"],
        "candidate_manifest": "ppo_mvp80_candidates.yaml",
        "pregrasp_catalog_root": pregrasp_catalog_root,
        "pregrasp_summary_paths": list(pregrasp_summary_paths),
        "selected_asset_count": len(selected_rows),
        "selected_rows": selected_rows,
        "cells": selected_cells,
        "rejected_pairs": rejected_pairs,
    }


__all__ = [
    "PHYSX_FINGER_ORDER",
    "REPRESENTATIVE_SELECTION_SCHEMA_VERSION",
    "SUPPORTED_CELL_VALUES",
    "RepresentativeAsset",
    "RepresentativePair",
    "ranked_representative_pairs",
    "finalize_representative_selection",
    "representative_assets",
    "representative_selection_document",
]
