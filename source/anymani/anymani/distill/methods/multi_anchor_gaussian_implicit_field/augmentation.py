r"""训练期单 JOINT 坐标符号改写。

被选中的样本只做一次主 forward，不另算 paired latent MSE。物理 surface/query/closest point 不变，
因此 density/distance 不变；对应 JOINT 的 $\kappa/g$ 随坐标翻号。
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any

import torch

from anymani.distill.models.input_adapters.geometry import StaticGeometryEvidence
from anymani.distill.objectives.representations.gauge_consistency import rewrite_joint_sign_coordinates
from anymani.distill.representations.queries.spatial_sampling import SpatialQueryBatch
from anymani.distill.representations.targets.field_samples import FieldTargetBatch, SensitivityTargetBatch

from .batch import OnlineGeometrySample, PaddedOnlineGeometryBatch
from .config import EntityPermutationCfg, JointSignRewriteCfg


def sample_entity_permutation(
    entity_count: int,
    *,
    asset_id: str,
    q_block_start: int,
    root_seed: int,
    config: EntityPermutationCfg,
) -> torch.Tensor:
    r"""由稳定资产/q-block 身份采样 ``new_slot -> old_slot`` 双射。

    SHA-256 消除 Python 内置 ``hash`` 的进程随机化；generator 位于 CPU 且不触碰 Torch 全局 RNG。
    同一调用结果供该资产完整 8-q block 共用。
    """

    if entity_count < 1 or q_block_start < 0 or root_seed < 0:
        raise ValueError("entity permutation requires positive entity count and non-negative identities")
    if not config.enabled:
        return torch.arange(entity_count, dtype=torch.long)
    digest = hashlib.sha256()
    digest.update(b"anymani-entity-permutation-v1\0")
    digest.update(asset_id.encode("utf-8"))
    digest.update(int(q_block_start).to_bytes(8, "little", signed=False))
    digest.update(int(root_seed + config.seed_offset).to_bytes(8, "little", signed=False))
    seed = int.from_bytes(digest.digest()[:8], "little", signed=False) % (2**63 - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randperm(entity_count, generator=generator)


def _permute_online_geometry_sample(
    sample: OnlineGeometrySample,
    permutation: torch.Tensor,
) -> OnlineGeometrySample:
    r"""同步重标号一个 q-block 的完整 entity/owner 轴，保持 JOINT coordinate axis 不变。"""

    evidence = sample.evidence
    entity_count = evidence.entity_role.shape[-1]
    if permutation.shape != (entity_count,) or permutation.dtype != torch.long:
        raise ValueError("entity permutation must have long shape [G]")
    permutation = permutation.to(device=evidence.entity_role.device)
    if not torch.equal(permutation.sort().values, torch.arange(entity_count, device=permutation.device)):
        raise ValueError("entity permutation must be a bijection over [0,G)")
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(entity_count, device=permutation.device)

    def owner_axis(value: torch.Tensor, axis: int) -> torch.Tensor:
        """沿调用点明确声明的 owner 轴重排，避免偶然 ``Q == G`` 时猜错轴。"""

        return value.index_select(axis, permutation.to(value.device))

    def graph_axes(value: torch.Tensor) -> torch.Tensor:
        index = permutation.to(value.device)
        return value.index_select(-2, index).index_select(-1, index)

    permuted_evidence = StaticGeometryEvidence(
        anchors=evidence.anchors,
        home_surface_points=owner_axis(evidence.home_surface_points, 0),
        home_surface_mask=owner_axis(evidence.home_surface_mask, 0),
        palm_normal=evidence.palm_normal,
        space_screws=evidence.space_screws,
        q_home=evidence.q_home,
        entity_role=owner_axis(evidence.entity_role, 0),
        entity_joint_index=owner_axis(evidence.entity_joint_index, 0),
        joint_entity_index=inverse.to(evidence.joint_entity_index.device)[evidence.joint_entity_index],
        shortest_path=graph_axes(evidence.shortest_path),
        parent_direction=graph_axes(evidence.parent_direction),
        child_direction=graph_axes(evidence.child_direction),
        entity_valid_mask=(
            owner_axis(evidence.entity_valid_mask, 0) if evidence.entity_valid_mask is not None else None
        ),
        joint_valid_mask=evidence.joint_valid_mask,
        anchor_valid_mask=evidence.anchor_valid_mask,
    )

    adjacent = sample.queries.adjacent_owner_index.clone()
    adjacent_valid = adjacent >= 0
    adjacent[adjacent_valid] = inverse.to(adjacent.device)[adjacent[adjacent_valid]]
    queries = SpatialQueryBatch(
        owner_axis(sample.queries.query_points_h, 1),
        owner_axis(sample.queries.query_stratum, 1),
        owner_axis(adjacent, 1),
        owner_axis(sample.queries.workspace_anchor_index, 1),
    )
    field = sample.field_targets
    field_targets = FieldTargetBatch(
        query_points=owner_axis(field.query_points, 1),
        query_stratum=owner_axis(field.query_stratum, 1),
        distance=owner_axis(field.distance, 1),
        density=owner_axis(field.density, 1),
        valid_mask=owner_axis(field.valid_mask, 1),
        owner_role=owner_axis(field.owner_role, 0 if field.owner_role.ndim == 1 else 1),
        bandwidths=field.bandwidths,
        provenance=field.provenance,
    )
    sensitivity = sample.sensitivity_targets
    remapped_owner = inverse.to(sensitivity.owner_index.device)[sensitivity.owner_index]
    low_face_bits = sensitivity.closest_source.bitwise_and(0xFFFFFFFF)
    closest_source = remapped_owner.to(torch.int64).bitwise_left_shift(32) | low_face_bits
    sensitivity_targets = SensitivityTargetBatch(
        owner_index=remapped_owner,
        query_index=sensitivity.query_index,
        joint_index=sensitivity.joint_index,
        ancestor_mask=sensitivity.ancestor_mask,
        active_mask=sensitivity.active_mask,
        closest_point=sensitivity.closest_point,
        closest_source=closest_source,
        uniqueness_margin=sensitivity.uniqueness_margin,
        kappa=sensitivity.kappa,
        field_sensitivity=sensitivity.field_sensitivity,
        valid_mask=sensitivity.valid_mask,
        owner_category=sensitivity.owner_category,
        query_stratum=sensitivity.query_stratum,
        fallback_category=sensitivity.fallback_category,
        sampling_role=sensitivity.sampling_role,
        central_difference=sensitivity.central_difference,
        central_difference_valid_mask=sensitivity.central_difference_valid_mask,
        central_difference_plus_face=sensitivity.central_difference_plus_face,
        central_difference_minus_face=sensitivity.central_difference_minus_face,
        central_difference_elapsed_seconds=sensitivity.central_difference_elapsed_seconds,
        provenance=sensitivity.provenance,
    )
    return OnlineGeometrySample(
        asset_id=sample.asset_id,
        q=sample.q,
        evidence=permuted_evidence,
        queries=queries,
        field_targets=field_targets,
        sensitivity_targets=sensitivity_targets,
        anchor_index=sample.anchor_index,
        q_index=sample.q_index,
    )


def permute_online_geometry_sample(
    sample: OnlineGeometrySample,
    permutation: torch.Tensor,
) -> OnlineGeometrySample:
    r"""同步重标号一个 q-block 的完整 entity/owner 轴，保持 JOINT coordinate axis 不变。"""

    return _permute_online_geometry_sample(sample, permutation)


def _first_value_mismatch(expected: Any, actual: Any, path: str = "sample") -> str | None:
    r"""递归定位两个 typed sample 在哪个字段上违反了同一个 entity 变换。"""

    if isinstance(expected, torch.Tensor) or isinstance(actual, torch.Tensor):
        if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
            return path
        if expected.shape != actual.shape or expected.dtype != actual.dtype or not torch.equal(expected, actual):
            return path
        return None
    if is_dataclass(expected) or is_dataclass(actual):
        if type(expected) is not type(actual):
            return path
        for field in fields(expected):
            mismatch = _first_value_mismatch(
                getattr(expected, field.name),
                getattr(actual, field.name),
                f"{path}.{field.name}",
            )
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(expected, Mapping) or isinstance(actual, Mapping):
        if not isinstance(expected, Mapping) or not isinstance(actual, Mapping) or expected.keys() != actual.keys():
            return path
        for key in expected:
            mismatch = _first_value_mismatch(expected[key], actual[key], f"{path}[{key!r}]")
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(expected, (tuple, list)) or isinstance(actual, (tuple, list)):
        if not isinstance(expected, type(actual)) or len(expected) != len(actual):
            return path
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=True)):
            mismatch = _first_value_mismatch(expected_item, actual_item, f"{path}[{index}]")
            if mismatch is not None:
                return mismatch
        return None
    return None if expected == actual else path


def validate_entity_permutation_transform(
    reference: OnlineGeometrySample,
    candidate: OnlineGeometrySample,
    permutation: torch.Tensor,
) -> None:
    r"""验证 candidate 是否完整执行了同一个 entity permutation。

    该校验以 reference 重新生成理论变换结果，并递归比较 evidence、graph、query、field target、
    sensitivity target 和所有 scalar metadata。它专门用于合同测试与离线语义审计；训练热路径只调用
    ``permute_online_geometry_sample``，不会为每个 q-block 复制一次完整 sample 做双份比较。

    Args:
        reference: permutation 前的完整在线 geometry sample。
        candidate: 声称由同一 permutation 得到的 sample。
        permutation: ``new_slot -> old_slot`` 的 entity 双射。

    Raises:
        ValueError: candidate 任一字段没有遵循同一个同步 entity 变换。
    """

    expected = _permute_online_geometry_sample(reference, permutation)
    mismatch = _first_value_mismatch(expected, candidate)
    if mismatch is not None:
        raise ValueError(f"entity permutation transform is not synchronized at {mismatch}")


def maybe_rewrite_batch(
    batch: PaddedOnlineGeometryBatch,
    *,
    config: JointSignRewriteCfg,
    step: int,
    seed: int,
) -> PaddedOnlineGeometryBatch:
    r"""按 20% 概率、每个选中样本恰好一个有效 JOINT 改写输入与一阶 target。"""

    evidence = batch.evidence
    joint_valid = evidence.joint_valid_mask
    if batch.evidence_row_index is not None and joint_valid is not None and joint_valid.ndim == 2:
        joint_valid = joint_valid[batch.evidence_row_index]
    if joint_valid is None:
        joint_valid = torch.ones_like(batch.q, dtype=torch.bool)
    if joint_valid.ndim == 1:
        joint_valid = joint_valid.unsqueeze(0).expand(batch.q.shape[0], -1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(config.seed_offset) + int(step))
    selected = torch.rand(batch.q.shape[0], generator=generator) < config.probability
    if not bool(selected.any()):
        return batch  # 默认未改写行保持 unique evidence table，不复制静态张量
    joint_sign = torch.ones_like(batch.q)
    for batch_index, is_selected in enumerate(selected.tolist()):
        if not is_selected:
            continue
        valid_indices = torch.where(joint_valid[batch_index])[0]
        if len(valid_indices) == 0:
            raise ValueError("joint-sign rewrite requires at least one valid JOINT")
        cursor = 0
        if batch.q_index is not None:
            cursor = int(batch.q_index[batch_index].item())
        chosen = valid_indices[(int(step) + cursor + batch_index) % len(valid_indices)]
        joint_sign[batch_index, chosen] = -1.0
    return rewrite_batch_joint_sign_coordinates(batch, joint_sign)


def rewrite_batch_joint_sign_coordinates(
    batch: PaddedOnlineGeometryBatch,
    joint_sign: torch.Tensor,
) -> PaddedOnlineGeometryBatch:
    r"""按显式 `[B,N_J]` 符号同步改写 q/evidence 与全部一阶 truth/provenance。"""

    if joint_sign.shape != batch.q.shape:
        raise ValueError("joint_sign must have shape [B,N_J] matching batch.q")
    evidence = batch.evidence
    if batch.evidence_row_index is not None:
        evidence = _expand_evidence_rows(evidence, batch.evidence_row_index)
    rewritten_q, rewritten_evidence, joint_sign = rewrite_joint_sign_coordinates(
        batch.q,
        evidence,
        joint_sign=joint_sign,
    )
    sensitivity = _rewrite_sensitivity_targets(batch.sensitivity_targets, joint_sign)
    return PaddedOnlineGeometryBatch(
        asset_ids=batch.asset_ids,
        q=rewritten_q,
        evidence=rewritten_evidence,
        evidence_row_index=torch.arange(batch.q.shape[0], device=batch.q.device, dtype=torch.long),
        queries=batch.queries,
        field_targets=batch.field_targets,
        sensitivity_targets=sensitivity,
        anchor_index=batch.anchor_index,
        q_index=batch.q_index,
    )


def _expand_evidence_rows(
    evidence: StaticGeometryEvidence,
    row_index: torch.Tensor,
) -> StaticGeometryEvidence:
    r"""把 unique evidence table 展开到 q-row，供逐行 joint-sign 坐标改写。

    同一 q-block 通常只有少数行被改写；第一版为保持公式清晰展开完整 microbatch。未发生任何改写时
    ``maybe_rewrite_batch`` 已直接返回原 table，不承担该复制成本。
    """

    if evidence.anchors.ndim != 3 or row_index.ndim != 1:
        raise ValueError("evidence row expansion requires batched evidence and [B] row_index")

    def route(value: torch.Tensor) -> torch.Tensor:
        return value[row_index] if value.ndim > 0 and value.shape[0] == evidence.anchors.shape[0] else value

    def route_optional(value: torch.Tensor | None) -> torch.Tensor | None:
        return None if value is None else route(value)

    return StaticGeometryEvidence(
        anchors=route(evidence.anchors),
        home_surface_points=route(evidence.home_surface_points),
        home_surface_mask=route(evidence.home_surface_mask),
        palm_normal=route(evidence.palm_normal),
        space_screws=route(evidence.space_screws),
        q_home=route(evidence.q_home),
        entity_role=route(evidence.entity_role),
        entity_joint_index=route(evidence.entity_joint_index),
        joint_entity_index=route(evidence.joint_entity_index),
        shortest_path=route(evidence.shortest_path),
        parent_direction=route(evidence.parent_direction),
        child_direction=route(evidence.child_direction),
        entity_valid_mask=route_optional(evidence.entity_valid_mask),
        joint_valid_mask=route_optional(evidence.joint_valid_mask),
        anchor_valid_mask=route_optional(evidence.anchor_valid_mask),
    )


def _rewrite_sensitivity_targets(
    targets: SensitivityTargetBatch,
    joint_sign: torch.Tensor,
) -> SensitivityTargetBatch:
    r"""翻转被改写 JOINT 的一阶量，并交换该坐标下 q+/q- face provenance。"""

    if targets.joint_index.ndim == 1:
        sign = joint_sign[:, targets.joint_index]
    else:
        sign = torch.gather(joint_sign, 1, targets.joint_index)
    kappa = targets.kappa * sign
    field_sensitivity = targets.field_sensitivity * sign.unsqueeze(-1)
    central_difference = (
        targets.central_difference * sign if targets.central_difference is not None else None
    )
    plus_face = targets.central_difference_plus_face
    minus_face = targets.central_difference_minus_face
    if plus_face is not None and minus_face is not None:
        flipped = sign < 0.0
        plus_face, minus_face = (
            torch.where(flipped, minus_face, plus_face),
            torch.where(flipped, plus_face, minus_face),
        )
    return SensitivityTargetBatch(
        owner_index=targets.owner_index,
        query_index=targets.query_index,
        joint_index=targets.joint_index,
        ancestor_mask=targets.ancestor_mask,
        active_mask=targets.active_mask,
        closest_point=targets.closest_point,
        closest_source=targets.closest_source,
        uniqueness_margin=targets.uniqueness_margin,
        kappa=kappa,
        field_sensitivity=field_sensitivity,
        valid_mask=targets.valid_mask,
        owner_category=targets.owner_category,
        query_stratum=targets.query_stratum,
        fallback_category=targets.fallback_category,
        sampling_role=targets.sampling_role,
        central_difference=central_difference,
        central_difference_valid_mask=targets.central_difference_valid_mask,
        central_difference_plus_face=plus_face,
        central_difference_minus_face=minus_face,
        central_difference_elapsed_seconds=targets.central_difference_elapsed_seconds,
        provenance=targets.provenance,
    )


__all__ = [
    "maybe_rewrite_batch",
    "permute_online_geometry_sample",
    "rewrite_batch_joint_sign_coordinates",
    "sample_entity_permutation",
    "validate_entity_permutation_transform",
]
