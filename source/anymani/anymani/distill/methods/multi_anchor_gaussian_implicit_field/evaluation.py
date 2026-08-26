r"""多锚点 Gaussian Method 固定评估中的配对 latent ablation。"""

from __future__ import annotations

from typing import Protocol

from anymani.distill.diagnostics.evaluation.geometry_ssl import (
    cross_asset_permutation,
    geometry_ssl_ablation_forward,
    geometry_ssl_reconstruction_metrics_per_sample,
    same_asset_q_permutation,
)
from anymani.distill.models.geometry_ssl import GeometrySSLForward, GeometrySSLModel

from .batch import PaddedOnlineGeometryBatch


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
        batch.q,
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
        full = model(q, batch.evidence, batch.queries.query_points_h, batch.field_targets.bandwidths, **common)
        predictions: dict[str, GeometrySSLForward | None] = {"full": full}
        predictions["query_only"] = geometry_ssl_ablation_forward(
            model,
            q,
            batch.evidence,
            batch.queries.query_points_h,
            batch.field_targets.bandwidths,
            ablation="query_only",
            **common,
        )
        try:
            same_asset_permutation = same_asset_q_permutation(batch.asset_ids, device=q.device)
        except ValueError:
            predictions["same_asset_q_shuffle"] = None
        else:
            predictions["same_asset_q_shuffle"] = geometry_ssl_ablation_forward(
                model,
                q,
                batch.evidence,
                batch.queries.query_points_h,
                batch.field_targets.bandwidths,
                ablation="latent_shuffle",
                batch_permutation=same_asset_permutation,
                **common,
            )
        try:
            cross_permutation = cross_asset_permutation(batch.asset_ids, device=q.device)
        except ValueError:
            predictions["cross_asset_shuffle"] = None
        else:
            predictions["cross_asset_shuffle"] = geometry_ssl_ablation_forward(
                model,
                q,
                batch.evidence,
                batch.queries.query_points_h,
                batch.field_targets.bandwidths,
                ablation="latent_shuffle",
                batch_permutation=cross_permutation,
                **common,
            )
        predictions["joint_token_shuffle"] = geometry_ssl_ablation_forward(
            model,
            q,
            batch.evidence,
            batch.queries.query_points_h,
            batch.field_targets.bandwidths,
            ablation="joint_token_shuffle",
            **common,
        )
        per_ablation = {
            name: geometry_ssl_reconstruction_metrics_per_sample(prediction, batch) if prediction is not None else None
            for name, prediction in predictions.items()
        }
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


__all__ = ["fixed_evaluation_ablation_evidence", "update_evaluation_digest"]
