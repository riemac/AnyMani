r"""从完整PPO资产轴构造handedness-paired、形态分层的确定性候选顺序。

本模块只读取 :class:`ResolvedHandAssetPartition` 已交付的sidecar、lineage与typed geometry semantics，
不启动Isaac Sim，也不读取任何RL checkpoint。选择对象是left/right topology-paired候选；四个不含handedness
的粗粒度cell分别由TIP数量$N_{tip}\in\{3,4\}$和thumb DoF $D_t\in\{3,4\}$定义。每个cell的
前10对形成80手MVP初始候选，后续pair只在good-pregrasp生成失败时按既定顺序替补。

连续描述符全部来自资产真值：每指关节数、关节range、finger chain长度、palm anchor位置以及collision
payload尺度统计。Left hand的palm-frame $x$ 分量先按YZ镜像变换到canonical right-hand比较域；该变换只服务
代表性距离，不修改资产、控制坐标或训练输入。Learned $Z$、reward与PPO表现都不参与选择。
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .dataset import ResolvedHandAssetPartition
from .hand_container import HandContainer

REPRESENTATIVE_SELECTION_SCHEMA_VERSION = "1.0.0"
"""当前pair-aware selection artifact的持久化schema。"""

PHYSX_FINGER_ORDER = ("index", "middle", "ring", "thumb")
"""与canonical 16-slot depth-major action轴一致的finger顺序。"""

SUPPORTED_CELL_VALUES = ((3, 3), (3, 4), (4, 3), (4, 4))
"""Handedness-neutral的$(N_{tip},D_t)$四个粗粒度cell。"""


@dataclass(frozen=True)
class RepresentativeAsset:
    r"""一项formal train asset的分层标签与可解释物理描述。

    ``descriptor``只服务同一cell内的标准化距离。它不作为policy输入，也不定义完整morphology space。
    ``geometry_identity``使用typed geometry semantics内容身份；canonical/PhysX physical hash在scene lowering
    后另行验证。
    """

    row: int  # formal ``ppo.yaml`` train轴行号
    asset_id: str  # 人类可读稳定asset ID
    geometry_identity: str  # typed geometry semantics完整内容身份
    handedness: str  # ``left``或``right``
    tip_count: int  # 有效TIP数量$N_{tip}$
    thumb_dof: int  # thumb活动关节数$D_t$
    active_dof: int  # 整手活动关节数$n_i$
    topology: str  # 去除left/right前缀后的handedness-neutral topology
    family_signature: str  # slot-level LEAP/Allegro/mixed组成
    asset_role: str  # mother或variant，仅用于tie-break/审计
    descriptor: tuple[float, ...]  # 固定宽度连续物理描述

    @property
    def cell(self) -> tuple[int, int]:
        r"""返回handedness-neutral $(N_{tip},D_t)$ cell。"""

        return self.tip_count, self.thumb_dof


@dataclass(frozen=True)
class RepresentativePair:
    r"""同cell、同topology/family signature的一组left/right训练候选。"""

    cell: tuple[int, int]  # $(N_{tip},D_t)$
    topology: str  # handedness-neutral topology
    family_signature: str  # 两侧相同slot family组成
    left: RepresentativeAsset  # left asset
    right: RepresentativeAsset  # right asset
    descriptor: tuple[float, ...]  # 两侧canonicalized descriptor均值
    reflection_distance: float  # 两侧标准化描述距离；只作代表性诊断


def _neutral_topology(value: str) -> str:
    r"""去除唯一允许的handedness前缀，保留joint-depth/missing-slot拓扑。"""

    for prefix in ("left_", "right_"):
        if value.startswith(prefix):
            return value[len(prefix) :]
    raise ValueError(f"topology name must start with left_/right_, got {value!r}")


def _payload_numbers(value: Any) -> tuple[float, ...]:
    r"""递归提取collision geometry payload中的有限数值叶节点。

    Boolean属于JSON整数子类，但不代表长度/尺度，因而排除。字符串路径与mesh身份由geometry hash负责，
    不进入欧氏描述符。
    """

    if isinstance(value, bool) or value is None or isinstance(value, str):
        return ()
    if isinstance(value, int | float):
        parsed = float(value)  # geometry payload长度通常为m，无量纲scale保持原值
        return (parsed,) if math.isfinite(parsed) else ()
    if isinstance(value, Mapping):
        return tuple(number for key in sorted(value) for number in _payload_numbers(value[key]))
    if isinstance(value, Sequence):
        return tuple(number for item in value for number in _payload_numbers(item))
    return ()


def _summary(values: Sequence[float]) -> tuple[float, float, float, float]:
    r"""把变长物理尺度序列规约为mean/std/min/max四个稳定统计。"""

    if not values:
        return (0.0, 0.0, 0.0, 0.0)
    mean = math.fsum(values) / len(values)  # 序列均值，单位继承输入
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)  # population variance
    return mean, math.sqrt(variance), min(values), max(values)


def _canonical_position(position: Sequence[float], handedness: str) -> tuple[float, float, float]:
    r"""把left-hand YZ反射到canonical right-hand比较域。"""

    if len(position) != 3:
        raise ValueError("hand-frame position must contain three coordinates")
    x, y, z = (float(value) for value in position)  # hand-frame位置，单位m
    return (-x if handedness == "left" else x), y, z  # YZ reflection只翻转$x$


def _family_signature(container: HandContainer) -> str:
    r"""按canonical finger order序列化实际slot family组成。"""

    slot_map = container.sidecar.get("slot_family_map", {})
    if not isinstance(slot_map, Mapping):
        raise ValueError(f"asset {container.asset_id!r} has invalid slot_family_map")
    return "|".join(f"{finger}:{slot_map.get(finger, 'missing')}" for finger in PHYSX_FINGER_ORDER)


def _physical_descriptor(container: HandContainer) -> tuple[float, ...]:
    r"""从typed semantics构造固定宽度、handedness-canonicalized物理描述。

    每根finger贡献：活动joint count、joint-limit center/span统计、完整kinematic origin链长度，以及palm
    anchor位置。末尾追加PALM/TIP collision payload尺度统计。描述符不尝试成为morphology的完备参数化；
    它只在离散topology/family覆盖之后打破同类variant选择的任意性。
    """

    semantics = container.geometry_semantics
    if semantics is None:
        raise ValueError(f"asset {container.asset_id!r} lacks typed geometry semantics")
    handedness = semantics.handedness  # left/right物理标签
    if handedness not in {"left", "right"}:
        raise ValueError(f"asset {container.asset_id!r} has unsupported handedness={handedness!r}")

    # Active joint names在generated schema中保留``<finger>_j<depth>``角色，可直接形成每指range统计。
    limits_by_finger: dict[str, list[tuple[float, float]]] = {finger: [] for finger in PHYSX_FINGER_ORDER}
    for joint_name, (lower, upper) in zip(
        semantics.active_joint_names,
        semantics.joint_limits_rad,
        strict=True,
    ):
        finger = joint_name.split("_", 1)[0]  # canonical finger role
        if finger in limits_by_finger:
            limits_by_finger[finger].append((float(lower), float(upper)))

    # Fixed与revolute joint origin共同决定整根finger的几何链长；这里使用局部translation范数的总和。
    chain_length_by_finger = {finger: 0.0 for finger in PHYSX_FINGER_ORDER}
    for joint in semantics.kinematic_joints:
        finger = joint.joint_name.split("_", 1)[0]
        if finger in chain_length_by_finger:
            chain_length_by_finger[finger] += math.sqrt(math.fsum(value * value for value in joint.origin_pos_m))

    # Anchor seed位于palm solid内、靠近各finger root；它能稳定表达mount横向/纵向布局。
    anchor_by_finger = {
        seed.finger_name: _canonical_position(seed.position_a_m, handedness)
        for seed in semantics.anchor_seeds
        if seed.finger_name in PHYSX_FINGER_ORDER
    }
    descriptor: list[float] = []
    for finger in PHYSX_FINGER_ORDER:
        limits = limits_by_finger[finger]  # 当前finger活动joint合法区间，单位rad
        centers = [(lower + upper) * 0.5 for lower, upper in limits]  # joint-range中心，rad
        spans = [upper - lower for lower, upper in limits]  # joint-range宽度，rad
        descriptor.extend(
            (
                len(limits) / 4.0,  # 每指DoF归一到canonical最大4
                *_summary(centers),  # 4项joint center统计
                *_summary(spans),  # 4项joint span统计
                chain_length_by_finger[finger],  # 运动学链总translation长度，m
                *anchor_by_finger.get(finger, (0.0, 0.0, 0.0)),  # canonicalized mount anchor，m
            )
        )

    # Collision payload只以role分组做尺度统计，避免mesh路径或component数量直接成为距离主轴。
    owner_role = {owner.owner_id: owner.role for owner in semantics.owners}
    palm_numbers: list[float] = []  # PALM collision尺寸/scale叶节点
    tip_numbers: list[float] = []  # TIP collision尺寸/scale叶节点
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
    r"""把resolved formal train partition转换为selection-only资产表。"""

    assets: list[RepresentativeAsset] = []
    for row, record in enumerate(partition.records):
        container = record.container  # bundle与lineage不可分record中的资产视图
        semantics = container.geometry_semantics
        if semantics is None:
            raise ValueError(f"asset {container.asset_id!r} lacks geometry semantics")
        tip_count = sum(owner.role == "tip" for owner in semantics.owners)  # $N_{tip}\in\{3,4\}$
        thumb_dof = sum(name.startswith("thumb_") for name in semantics.active_joint_names)  # $D_t$
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
    r"""在一个cell内对连续描述逐维做population标准化。"""

    if not assets:
        raise ValueError("cannot standardize an empty asset cell")
    width = len(assets[0].descriptor)  # 固定物理描述宽度$D$
    if width < 1 or any(len(asset.descriptor) != width for asset in assets):
        raise ValueError("all representative descriptors must share one non-empty width")
    columns = tuple(tuple(asset.descriptor[index] for asset in assets) for index in range(width))
    means = tuple(math.fsum(column) / len(column) for column in columns)  # $\mu_d$
    scales = tuple(
        max(math.sqrt(math.fsum((value - mean) ** 2 for value in column) / len(column)), 1.0e-12)
        for column, mean in zip(columns, means, strict=True)
    )  # $\sigma_d$；constant列以1e-12防止除零
    return {
        asset.row: tuple(
            (value - mean) / scale
            for value, mean, scale in zip(asset.descriptor, means, scales, strict=True)
        )
        for asset in assets
    }


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    r"""返回标准化物理描述的Euclidean距离。"""

    if len(left) != len(right):
        raise ValueError("descriptor distance requires equal widths")
    return math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def _pair_cell_assets(assets: Sequence[RepresentativeAsset]) -> list[RepresentativePair]:
    r"""在同cell/topology/family内贪心匹配描述最接近的left/right资产。

    Formal train的左右variant并非逐项严格镜像，因此这里只构造topology-paired cohort，而不伪造physical
    mirror identity。每项asset最多进入一个pair；两侧数量不等时，多余项留在候选池外并由artifact计数暴露。
    """

    standardized = _standardized_descriptors(assets)  # cell内统一标准化距离域
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
            )  # 最接近的未使用right variant，row作确定性tie-break
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
                    ),  # pair-level中心描述
                    reflection_distance=_distance(left_vector, right_vector),
                )
            )
    return pairs


def _rank_pairs(pairs: Sequence[RepresentativePair]) -> tuple[RepresentativePair, ...]:
    r"""按离散覆盖优先、连续max-min次之生成完整确定性候选顺序。"""

    remaining = list(pairs)
    selected: list[RepresentativePair] = []
    covered_topologies: set[str] = set()  # 已覆盖handedness-neutral topology
    covered_families: set[str] = set()  # 已覆盖slot family signature
    covered_dofs: set[tuple[int, int]] = set()  # 已覆盖(left DoF,right DoF)
    while remaining:
        def score(pair: RepresentativePair) -> tuple[float, ...]:
            diversity = (
                min(_distance(pair.descriptor, chosen.descriptor) for chosen in selected)
                if selected
                else 0.0
            )  # greedy farthest-point到已选集合的最小距离
            mother_pair = float(pair.left.asset_role == "mother" and pair.right.asset_role == "mother")
            return (
                float(pair.topology not in covered_topologies),  # 首先最大化topology覆盖
                float(pair.family_signature not in covered_families),  # 再覆盖family composition
                float((pair.left.active_dof, pair.right.active_dof) not in covered_dofs),  # 再覆盖DoF组合
                diversity,  # 同离散层内最大化物理多样性
                mother_pair,  # 完全同分时优先可审计mother pair
                -pair.reflection_distance,  # 再优先左右描述接近
                -float(pair.left.row),  # formal row作为最终稳定tie-break
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
    r"""为四个MVP cells分别生成left/right pair候选顺序。"""

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
    r"""返回selection artifact中的单资产审计字段。"""

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
    r"""构造可持久化的80手初始选择与确定性fallback pair队列。"""

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
    r"""按既定pair顺序用good-pregrasp通过集发布最终80-row manifest。

    每个pair只有left/right两侧都通过时才可进入训练集；失败pair完整保存在``rejected_pairs``。本函数不按
    pregrasp score重新排序，也不读取rotation/PPO表现，因而物理生成只充当宽松reset可行性门。

    Args:
        candidate_document: :func:`representative_selection_document` 生成的候选artifact。
        passed_rows: 已发布Top-8 good-pregrasp entry的formal row集合。
        pregrasp_catalog_root: 相对AnyMani根的scientific catalog路径。
        pregrasp_summary_paths: 形成通过集的一个或多个generation summary路径。

    Returns:
        dict: 每cell恰好10 pairs、总计80项的最终selection manifest。
    """

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
