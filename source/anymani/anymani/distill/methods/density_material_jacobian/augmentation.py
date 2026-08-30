r"""Density + Gamma method 的 entity permutation 与 joint-coordinate sign rewrite。"""

from __future__ import annotations

import torch

from anymani.distill.methods.multi_anchor_gaussian_implicit_field.augmentation import sample_entity_permutation
from anymani.distill.models.input_adapters.geometry import StaticGeometryEvidence
from anymani.distill.representations.queries.spatial_sampling import SpatialQueryBatch
from anymani.distill.representations.targets.field_samples import FieldTargetBatch
from anymani.distill.representations.targets.material_point_jacobian import MaterialPointRelationJacobianTarget

from .batch import DensityGammaOnlineSample, PaddedDensityGammaBatch
from .config import JointSignRewriteCfg


def permute_density_gamma_sample(
    sample: DensityGammaOnlineSample,
    permutation: torch.Tensor,
) -> DensityGammaOnlineSample:
    r"""同步重标号完整 owner/entity 轴，保持 JOINT coordinate、edge 和 anchor 轴不变。"""

    evidence = sample.evidence
    entity_count = evidence.entity_role.shape[-1]
    if permutation.shape != (entity_count,) or permutation.dtype != torch.long:
        raise ValueError("entity permutation must have long shape [G]")
    permutation = permutation.to(evidence.entity_role.device)
    if not torch.equal(permutation.sort().values, torch.arange(entity_count, device=permutation.device)):
        raise ValueError("entity permutation must be a bijection")
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(entity_count, device=permutation.device)

    def owner_axis(value: torch.Tensor, axis: int) -> torch.Tensor:
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
        entity_valid_mask=None if evidence.entity_valid_mask is None else owner_axis(evidence.entity_valid_mask, 0),
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
    target = sample.material_targets
    remapped_owner = inverse.to(target.owner_index.device)[target.owner_index]
    material_targets = MaterialPointRelationJacobianTarget(
        distance_m=target.distance_m,
        distance_sensitivity_m_per_rad=target.distance_sensitivity_m_per_rad,
        relation_values=target.relation_values,
        relation_sensitivity_per_rad=target.relation_sensitivity_per_rad,
        distance_valid_mask=target.distance_valid_mask,
        radius_valid_mask=target.radius_valid_mask,
        material_points_h_m=target.material_points_h_m,
        point_jacobian_h_m_per_rad=target.point_jacobian_h_m_per_rad,
        owner_index=remapped_owner,
        joint_index=target.joint_index,
        ancestor_mask=target.ancestor_mask,
        provenance=target.provenance,
    )
    return DensityGammaOnlineSample(
        asset_id=sample.asset_id,
        q=sample.q,
        evidence=permuted_evidence,
        queries=queries,
        field_targets=field_targets,
        material_targets=material_targets,
        material_point_index=sample.material_point_index,
        edge_valid_mask=sample.edge_valid_mask,
        anchor_index=sample.anchor_index,
        q_index=sample.q_index,
    )


def rewrite_density_gamma_batch_joint_sign(
    batch: PaddedDensityGammaBatch,
    joint_sign: torch.Tensor,
) -> PaddedDensityGammaBatch:
    r"""同步改写 q/encoder screw gauge，并翻转 selected Gamma Jacobian columns。"""

    if joint_sign.shape != batch.q.shape or not torch.all((joint_sign == 1.0) | (joint_sign == -1.0)):
        raise ValueError("joint_sign must have q shape with entries in {-1,+1}")
    prior = batch.joint_coordinate_sign
    coordinate_sign = joint_sign if prior is None else prior * joint_sign
    target = batch.material_targets
    selected_sign = torch.gather(joint_sign, 1, target.joint_index)  # `[B,E]`
    rewritten_target = MaterialPointRelationJacobianTarget(
        distance_m=target.distance_m,
        distance_sensitivity_m_per_rad=target.distance_sensitivity_m_per_rad * selected_sign.unsqueeze(-1),
        relation_values=target.relation_values,
        relation_sensitivity_per_rad=target.relation_sensitivity_per_rad * selected_sign[:, :, None, None],
        distance_valid_mask=target.distance_valid_mask,
        radius_valid_mask=target.radius_valid_mask,
        material_points_h_m=target.material_points_h_m,
        point_jacobian_h_m_per_rad=target.point_jacobian_h_m_per_rad * selected_sign.unsqueeze(-1),
        owner_index=target.owner_index,
        joint_index=target.joint_index,
        ancestor_mask=target.ancestor_mask,
        provenance=target.provenance,
    )
    return PaddedDensityGammaBatch(
        asset_ids=batch.asset_ids,
        q=batch.q * joint_sign,
        evidence=batch.evidence,
        evidence_row_index=batch.evidence_row_index,
        queries=batch.queries,
        field_targets=batch.field_targets,
        material_targets=rewritten_target,
        material_point_index=batch.material_point_index,
        edge_valid_mask=batch.edge_valid_mask,
        anchor_index=batch.anchor_index,
        q_index=batch.q_index,
        joint_coordinate_sign=coordinate_sign,
    )


def maybe_rewrite_density_gamma_batch(
    batch: PaddedDensityGammaBatch,
    *,
    config: JointSignRewriteCfg,
    step: int,
    seed: int,
) -> PaddedDensityGammaBatch:
    r"""按每 row probability 选择一个有效 JOINT，并执行完整 coordinate rewrite。"""

    if config.probability <= 0.0:
        return batch
    joint_valid = batch.evidence.joint_valid_mask
    if joint_valid is None:
        joint_valid = torch.ones_like(batch.q, dtype=torch.bool)
    elif joint_valid.ndim == 2:
        joint_valid = joint_valid[batch.evidence_row_index]
    if joint_valid.ndim == 1:
        joint_valid = joint_valid.unsqueeze(0).expand_as(batch.q)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(config.seed_offset) + int(step))
    selected = torch.rand(batch.q.shape[0], generator=generator) < config.probability
    joint_sign = torch.ones_like(batch.q)
    for row, enabled in enumerate(selected.tolist()):
        if not enabled:
            continue
        valid = torch.where(joint_valid[row])[0]
        cursor = int(batch.q_index[row]) if batch.q_index is not None else row
        chosen = valid[(int(step) + cursor + row) % len(valid)]
        joint_sign[row, chosen] = -1.0
    return rewrite_density_gamma_batch_joint_sign(batch, joint_sign)


__all__ = [
    "maybe_rewrite_density_gamma_batch",
    "permute_density_gamma_sample",
    "rewrite_density_gamma_batch_joint_sign",
    "sample_entity_permutation",
]
