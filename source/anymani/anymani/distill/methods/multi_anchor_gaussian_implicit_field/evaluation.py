r"""多锚点 Gaussian Method 固定评估中的配对 latent ablation。"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Protocol, cast

import torch

from .augmentation import rewrite_batch_joint_sign_coordinates
from .batch import PaddedOnlineGeometryBatch, method_batch_views

if TYPE_CHECKING:
    from anymani.distill.models.geometry_ssl import GeometrySSLModel

geometry_ssl_ablation_forward: Any = None
geometry_ssl_reconstruction_metrics_per_sample: Any = None
same_asset_q_permutation: Any = None
cross_asset_permutation: Any = None


class _HashWriter(Protocol):
    r"""固定 evaluation bank digest 所需的最小增量 hash 接口。"""

    def update(self, data: bytes, /) -> object:
        r"""把一段稳定字节序列纳入摘要。"""
        ...


def update_evaluation_digest(digest: _HashWriter, batch: PaddedOnlineGeometryBatch) -> None:
    r"""把固定 bank 的资产、q、query、sigma、routing、mask 与 teacher 纳入 SHA-256。"""

    for asset_id in batch.asset_ids:
        encoded = asset_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    tensors = (
        batch.q_index,
        batch.anchor_index,
        batch.q,
        batch.evidence_row_index,
        batch.evidence.anchors,
        batch.evidence.home_surface_points,
        batch.evidence.home_surface_mask,
        batch.evidence.palm_normal,
        batch.evidence.space_screws,
        batch.evidence.q_home,
        batch.evidence.entity_role,
        batch.evidence.entity_joint_index,
        batch.evidence.joint_entity_index,
        batch.evidence.shortest_path,
        batch.evidence.parent_direction,
        batch.evidence.child_direction,
        batch.evidence.entity_valid_mask,
        batch.evidence.joint_valid_mask,
        batch.evidence.anchor_valid_mask,
        batch.queries.query_points_h,
        batch.queries.query_stratum,
        batch.queries.workspace_anchor_index,
        batch.queries.adjacent_owner_index,
        batch.field_targets.bandwidths,
        batch.field_targets.valid_mask,
        batch.field_targets.owner_role,
        batch.sensitivity_targets.owner_index,
        batch.sensitivity_targets.query_index,
        batch.sensitivity_targets.joint_index,
        batch.sensitivity_targets.ancestor_mask,
        batch.sensitivity_targets.active_mask,
        batch.sensitivity_targets.closest_point,
        batch.sensitivity_targets.closest_source,
        batch.sensitivity_targets.uniqueness_margin,
        batch.sensitivity_targets.valid_mask,
        batch.field_targets.distance,
        batch.field_targets.density,
        batch.sensitivity_targets.kappa,
        batch.sensitivity_targets.field_sensitivity,
    )
    for tensor in tensors:
        if tensor is None:
            digest.update(b"none\0")
            continue
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.numpy().tobytes(order="C"))


def fixed_evaluation_ablation_evidence(
    model: GeometrySSLModel,
    batches: tuple[PaddedOnlineGeometryBatch, ...],
) -> dict[str, object]:
    r"""在固定 Method 测度上生成可按 ``(asset_id,q_index)`` 配对的 ablation evidence。"""

    # diagnostics.evaluation.geometry_ssl 反向依赖 methods.batch；延迟导入避免 methods package 初始化环。
    global cross_asset_permutation, geometry_ssl_ablation_forward
    global geometry_ssl_reconstruction_metrics_per_sample, same_asset_q_permutation
    if any(
        operation is None
        for operation in (
            cross_asset_permutation,
            geometry_ssl_ablation_forward,
            geometry_ssl_reconstruction_metrics_per_sample,
            same_asset_q_permutation,
        )
    ):
        from anymani.distill.diagnostics.evaluation.geometry_ssl import (
            cross_asset_permutation as _cross_asset_permutation,
        )
        from anymani.distill.diagnostics.evaluation.geometry_ssl import (
            geometry_ssl_ablation_forward as _geometry_ssl_ablation_forward,
        )
        from anymani.distill.diagnostics.evaluation.geometry_ssl import (
            geometry_ssl_reconstruction_metrics_per_sample as _geometry_ssl_reconstruction_metrics_per_sample,
        )
        from anymani.distill.diagnostics.evaluation.geometry_ssl import (
            same_asset_q_permutation as _same_asset_q_permutation,
        )

        if cross_asset_permutation is None:
            cross_asset_permutation = _cross_asset_permutation
        if geometry_ssl_ablation_forward is None:
            geometry_ssl_ablation_forward = _geometry_ssl_ablation_forward
        if geometry_ssl_reconstruction_metrics_per_sample is None:
            geometry_ssl_reconstruction_metrics_per_sample = _geometry_ssl_reconstruction_metrics_per_sample
        if same_asset_q_permutation is None:
            same_asset_q_permutation = _same_asset_q_permutation
    records: list[dict[str, object]] = []
    ablations = (
        "query_only",
        "same_asset_q_shuffle",
        "cross_asset_shuffle",
        "joint_token_shuffle",
    )
    for block_index, batch in enumerate(batches):
        q = batch.q.detach()
        common = {
            "owner_index": batch.sensitivity_targets.owner_index,
            "query_index": batch.sensitivity_targets.query_index,
            "joint_index": batch.sensitivity_targets.joint_index,
        }
        full = model(
            q,
            batch.evidence,
            batch.queries.query_points_h,
            batch.field_targets.bandwidths,
            evidence_row_index=batch.evidence_row_index,
            **common,
        )
        # CUDA Graph 复用 compiled output storage；每次 forward 后立即归约，避免后续调用覆盖旧预测。
        per_ablation: dict[str, dict[str, tuple[float | None, ...]] | None] = {
            "full": geometry_ssl_reconstruction_metrics_per_sample(full, batch)
        }
        query_only = geometry_ssl_ablation_forward(
            model,
            q,
            batch.evidence,
            batch.queries.query_points_h,
            batch.field_targets.bandwidths,
            ablation="query_only",
            evidence_row_index=batch.evidence_row_index,
            **common,
        )
        per_ablation["query_only"] = geometry_ssl_reconstruction_metrics_per_sample(query_only, batch)
        try:
            same_asset_permutation = same_asset_q_permutation(batch.asset_ids, device=q.device)
        except ValueError:
            per_ablation["same_asset_q_shuffle"] = None
        else:
            same_asset_q_shuffle = geometry_ssl_ablation_forward(
                model,
                q,
                batch.evidence,
                batch.queries.query_points_h,
                batch.field_targets.bandwidths,
                ablation="latent_shuffle",
                batch_permutation=same_asset_permutation,
                evidence_row_index=batch.evidence_row_index,
                **common,
            )
            per_ablation["same_asset_q_shuffle"] = geometry_ssl_reconstruction_metrics_per_sample(
                same_asset_q_shuffle, batch
            )
        try:
            cross_permutation = cross_asset_permutation(batch.asset_ids, device=q.device)
        except ValueError:
            per_ablation["cross_asset_shuffle"] = None
        else:
            cross_asset_shuffle = geometry_ssl_ablation_forward(
                model,
                q,
                batch.evidence,
                batch.queries.query_points_h,
                batch.field_targets.bandwidths,
                ablation="latent_shuffle",
                batch_permutation=cross_permutation,
                evidence_row_index=batch.evidence_row_index,
                **common,
            )
            per_ablation["cross_asset_shuffle"] = geometry_ssl_reconstruction_metrics_per_sample(
                cross_asset_shuffle, batch
            )
        joint_token_shuffle = geometry_ssl_ablation_forward(
            model,
            q,
            batch.evidence,
            batch.queries.query_points_h,
            batch.field_targets.bandwidths,
            ablation="joint_token_shuffle",
            evidence_row_index=batch.evidence_row_index,
            **common,
        )
        per_ablation["joint_token_shuffle"] = geometry_ssl_reconstruction_metrics_per_sample(
            joint_token_shuffle, batch
        )
        q_indices = batch.q_index.tolist() if batch.q_index is not None else [-1] * len(batch.asset_ids)
        for sample_index, (asset_id, q_index) in enumerate(zip(batch.asset_ids, q_indices)):
            records.append(
                {
                    "block_index": block_index,
                    "asset_id": asset_id,
                    "q_index": int(q_index),
                    "metrics": {
                        name: (
                            None
                            if values is None
                            else {metric: per_sample[sample_index] for metric, per_sample in values.items()}
                        )
                        for name, values in per_ablation.items()
                    },
                }
            )
    return {
        "pairing_key": ["asset_id", "q_index"],
        "ablations": ("full", *ablations),
        "records": records,
    }


def evaluate_method_session(
    method: Any,
    session: Any,
    schedule: Any,
    *,
    include_ablations: bool = False,
) -> Any:
    r"""流式执行固定 $A^{(0)}$/4-16-64 mm/4+4 edge 测度。

    Trainer 只决定何时评估；Method evaluation 拥有固定 sigma、anchor、edge、joint-sign audit、分层轴与
    ablation。所有 objective 只累计 detached numerator/denominator，不保留跨 batch graph。
    """

    from anymani.distill.diagnostics.evaluation.geometry_ssl import (
        aggregate_geometry_ssl_stratified_components,
        geometry_ssl_stratified_components_per_sample,
        joint_sign_observable_metrics,
    )
    from anymani.distill.methods.contracts import MethodEvaluationReport

    method.eval_mode()
    totals: dict[str, list[float]] = {}
    stratified_blocks: list[tuple[tuple[str, ...], Any]] = []
    ablation_records: list[dict[str, object]] = []
    ablation_names: tuple[str, ...] | None = None
    baseline_statistics: dict[str, torch.Tensor] | None = None
    joint_sign_totals = {
        "density_invariance_mse": [0.0, 0.0],
        "kappa_sign_equivariance_mse": [0.0, 0.0],
    }
    digest = hashlib.sha256(b"multi-anchor-fixed-evaluation-v1\0")
    block_index = 0
    with torch.no_grad():
        while not schedule.complete:
            batch = session.realize(schedule.next(), schedule=schedule, step=block_index)
            baseline_statistics = method.merge_teacher_baseline_statistics(
                baseline_statistics,
                method.teacher_baseline_statistics(batch),
            )  # suite teacher bank 自己定义 baseline，不读取 train-run 统计
            step_result, prediction = method._forward_with_prediction(batch, step=block_index, mode="eval")
            for objective in step_result.objectives.values():
                for component in objective.components:
                    current = totals.setdefault(component.name, [0.0, 0.0])
                    current[0] += float(component.numerator.detach())
                    current[1] += float(component.denominator.detach())
            stratified_blocks.append(
                (batch.asset_ids, geometry_ssl_stratified_components_per_sample(prediction, batch))
            )
            update_evaluation_digest(digest, batch)
            if include_ablations:
                # joint-sign audit 在全部 ablations 之后使用 reference observable；先复制 CUDA Graph 输出。
                parity_reference = replace(
                    prediction,
                    density=prediction.density.clone(),
                    kappa=prediction.kappa.clone(),
                )
                evidence = fixed_evaluation_ablation_evidence(method.require_model(), (batch,))
                raw_names = evidence.get("ablations")
                if not isinstance(raw_names, (tuple, list)):
                    raise ValueError("method ablation evidence names must be a sequence")
                names = tuple(str(name) for name in raw_names)
                ablation_names = names if ablation_names is None else ablation_names
                if names != ablation_names:
                    raise ValueError("method ablation names changed within one evaluation session")
                raw_records = evidence.get("records")
                if not isinstance(raw_records, list):
                    raise ValueError("method ablation evidence records must be a list")
                for record in raw_records:
                    copied = dict(record)
                    copied["block_index"] = block_index
                    ablation_records.append(copied)

                # 每行确定性选择一个有效 JOINT；只验证 observable density/$\kappa$ 等变性，不修改 teacher。
                joint_valid = batch.evidence.joint_valid_mask
                if joint_valid is None:
                    joint_valid = torch.ones_like(batch.q, dtype=torch.bool)
                elif batch.evidence_row_index is not None and joint_valid.ndim == 2:
                    joint_valid = joint_valid[batch.evidence_row_index]
                elif joint_valid.ndim == 1:
                    joint_valid = joint_valid.unsqueeze(0).expand_as(batch.q)
                joint_sign = torch.ones_like(batch.q)
                for row in range(batch.q.shape[0]):
                    valid_slots = torch.where(joint_valid[row])[0]
                    if valid_slots.numel() == 0:
                        raise ValueError("joint-sign audit requires at least one valid JOINT per q row")
                    cursor = int(batch.q_index[row]) if batch.q_index is not None else row
                    selected = valid_slots[(block_index + cursor + row) % valid_slots.numel()]
                    joint_sign[row, selected] = -1.0
                rewritten_batch = rewrite_batch_joint_sign_coordinates(batch, joint_sign)
                rewritten_prediction = method._forward_with_prediction(
                    rewritten_batch,
                    step=block_index,
                    mode="eval",
                    apply_augmentation=False,
                )[1]
                parity = joint_sign_observable_metrics(
                    parity_reference,
                    rewritten_prediction,
                    joint_sign=joint_sign,
                    joint_index=batch.sensitivity_targets.joint_index,
                    density_valid_mask=batch.field_targets.valid_mask,
                    edge_valid_mask=batch.sensitivity_targets.valid_mask,
                )
                weights = {
                    "density_invariance_mse": float(
                        batch.field_targets.valid_mask.sum() * prediction.density.shape[-1]
                    ),
                    "kappa_sign_equivariance_mse": float(batch.sensitivity_targets.valid_mask.sum()),
                }
                for name, value in parity.items():
                    joint_sign_totals[name][0] += value * weights[name]
                    joint_sign_totals[name][1] += weights[name]
            block_index += 1

    strata = aggregate_geometry_ssl_stratified_components(tuple(stratified_blocks))
    raw_metrics = strata.get("metric_scores")
    if not isinstance(raw_metrics, dict):
        raise ValueError("method evaluation strata lack metric_scores")
    metrics = {str(name): float(value) for name, value in raw_metrics.items()}
    strata["objective_terms"] = {name: numerator / denominator for name, (numerator, denominator) in totals.items()}
    strata["bank_digest_sha256"] = digest.hexdigest()
    if include_ablations:
        strata["joint_sign_observable_audit"] = {
            name: numerator / max(denominator, 1.0) for name, (numerator, denominator) in joint_sign_totals.items()
        }
    ablations = None
    if include_ablations:
        ablations = {
            "split": f"{session.role}.{session.suite}" if session.suite else session.role,
            "pairing_key": ["asset_id", "q_index"],
            "ablations": ablation_names or (),
            "records": ablation_records,
        }
    if baseline_statistics is None:
        raise RuntimeError("fixed evaluation session produced no teacher baseline statistics")
    return MethodEvaluationReport(
        metrics=metrics,
        strata=strata,
        teacher_baselines=method.finalize_teacher_baselines(baseline_statistics),
        ablations=ablations,
    )


def fit_z_compression_basis(method: Any, session: Any, schedule: Any) -> Any:
    r"""在独立 training-q bank 上流式拟合一个统一 PALM/JOINT/TIP PCA basis。"""

    from anymani.distill.diagnostics.evaluation.z_compression import UnifiedPCAAccumulator

    method.eval_mode()
    accumulator = UnifiedPCAAccumulator(method.feature_spec().entity_width)
    block_index = 0
    with torch.no_grad():
        while not schedule.complete:
            batch = session.realize(schedule.next(), schedule=schedule, step=block_index)
            q, evidence, row_index = method_batch_views(batch).model_input
            latents = method.require_model().encoder(q, evidence, row_index)
            valid = evidence.entity_valid_mask
            if valid is None:
                valid = torch.ones(latents.entities.shape[:2], device=q.device, dtype=torch.bool)
            elif row_index is not None and valid.ndim == 2:
                valid = valid[row_index]
            elif valid.ndim == 1:
                valid = valid.unsqueeze(0).expand(q.shape[0], -1)
            accumulator.update(latents.entities, valid)
            block_index += 1
    return accumulator.finalize()


def evaluate_z_compression_session(
    method: Any,
    session: Any,
    schedule: Any,
    *,
    basis: Any,
    ranks: tuple[int, ...],
) -> dict[str, object]:
    r"""在固定 suite bank 上以原 readers 重放 32/64/96/128 维 $Z$。"""

    from anymani.distill.diagnostics.evaluation.geometry_ssl import (
        aggregate_geometry_ssl_stratified_components,
        geometry_ssl_stratified_components_per_sample,
    )
    from anymani.distill.diagnostics.evaluation.z_compression import decode_z_compression_ranks

    blocks: dict[int, list[tuple[tuple[str, ...], Any]]] = {rank: [] for rank in ranks}
    baseline_statistics: dict[str, torch.Tensor] | None = None
    model = method.require_model()
    block_index = 0
    with torch.no_grad():
        while not schedule.complete:
            batch = session.realize(schedule.next(), schedule=schedule, step=block_index)
            baseline_statistics = method.merge_teacher_baseline_statistics(
                baseline_statistics,
                method.teacher_baseline_statistics(batch),
            )
            reference = method._forward_with_prediction(batch, step=block_index, mode="eval")[1]
            valid = batch.evidence.entity_valid_mask
            if valid is None:
                valid = torch.ones(reference.latents.entities.shape[:2], device=batch.q.device, dtype=torch.bool)
            elif batch.evidence_row_index is not None and valid.ndim == 2:
                valid = valid[batch.evidence_row_index]
            elif valid.ndim == 1:
                valid = valid.unsqueeze(0).expand(batch.q.shape[0], -1)
            joint_routing = batch.evidence.joint_entity_index
            if batch.evidence_row_index is not None and joint_routing.ndim == 2:
                joint_routing = joint_routing[batch.evidence_row_index]
            targets = batch.sensitivity_targets
            predictions = decode_z_compression_ranks(
                model,
                reference,
                basis=basis,
                ranks=ranks,
                entity_valid_mask=valid,
                bandwidths=batch.field_targets.bandwidths,
                joint_entity_index=joint_routing,
                owner_index=targets.owner_index,
                query_index=targets.query_index,
                joint_index=targets.joint_index,
            )
            for rank, prediction in predictions.items():
                blocks.setdefault(rank, []).append(
                    (batch.asset_ids, geometry_ssl_stratified_components_per_sample(prediction, batch))
                )
            block_index += 1
    if baseline_statistics is None:
        raise ValueError("Z compression evaluation requires at least one validation block")
    baseline = method.finalize_teacher_baselines(baseline_statistics)
    density_record = cast(Mapping[str, object], baseline["density"])
    kappa_record = cast(Mapping[str, object], baseline["kappa"])
    density_baseline = float(cast(float, density_record["baseline_mse"]))
    kappa_baseline = float(cast(float, kappa_record["physical_baseline_mse"]))
    results: dict[str, object] = {}
    for rank in sorted(blocks):
        if not blocks[rank]:
            continue
        stratified = aggregate_geometry_ssl_stratified_components(tuple(blocks[rank]))
        metrics = cast(Mapping[str, float], stratified["metric_scores"])
        normalized = {
            "density": float(metrics["density"]) / density_baseline,
            "kappa": float(metrics["kappa"]) / kappa_baseline,
        }
        results[str(rank)] = {
            "stratified": stratified,
            "normalized_metric_scores": normalized,
            "skill": {name: 1.0 - value for name, value in normalized.items()},
        }
    return {"teacher_baselines": baseline, "ranks": results}


def analyze_ablations(
    evidence: Mapping[str, Any],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    r"""执行 morphology/q 两级配对 bootstrap。"""

    from anymani.distill.diagnostics.analysis.geometry_ssl import analyze_geometry_ssl_ablation_evidence

    return analyze_geometry_ssl_ablation_evidence(
        dict(evidence),
        bootstrap_samples=bootstrap_replicates,
        seed=seed,
    )


__all__ = [
    "analyze_ablations",
    "evaluate_method_session",
    "evaluate_z_compression_session",
    "fit_z_compression_basis",
    "fixed_evaluation_ablation_evidence",
    "update_evaluation_digest",
]
