r"""Geometry SSL runtime 对固定 validation 协议的执行层。

分层 metric 与 ablation 数学定义由 ``diagnostics.evaluation`` 拥有；本模块负责在固定 bank 上执行
模型、组织 `(asset_id,q_index)` evidence、流式重放训练形态独立 q bank，并验证初始/最终 bank
的 q/query/teacher SHA-256 完全一致。checkpoint promotion 仍由 trainer 决定。
"""

from __future__ import annotations

import hashlib  # 固定 q bank 的 byte-level identity
from typing import Protocol  # 增量 hash writer 的最小接口

import torch  # validation forward、有限性检查与 RNG 保存/恢复

from anymani.distill.diagnostics.evaluation.geometry_ssl import (
    aggregate_geometry_ssl_stratified_components,
    cross_asset_permutation,
    geometry_ssl_ablation_forward,
    geometry_ssl_reconstruction_metrics_per_sample,
    geometry_ssl_stratified_components_per_sample,
    same_asset_q_permutation,
)
from anymani.distill.models.geometry_ssl import GeometrySSLForward, GeometrySSLModel
from anymani.distill.representations.geometry import (
    GeometryRepresentation,
    GeometryRepresentationCfg,
    PaddedOnlineGeometryBatch,
)
from anymani.distill.representations.sources.geometry_source import GeometrySource
from anymani.distill.representations.targets.geometry_field import fixed_validation_gaussian_field_config
from anymani.distill.ssl.runtime import (
    GeometrySSLRuntimeCfg,
    ResidentGeometryAssetWindow,
    WindowedOnlineGeometryBatcher,
)


def normalized_validation_score(metrics: dict[str, float], initial_metrics: dict[str, float]) -> float:
    r"""以初始化误差归一化 density、κ 与 derived-g 后等权求 checkpoint score。"""

    ratios = []
    for name in ("density", "kappa", "derived_field"):
        denominator = initial_metrics[name]
        if denominator <= 0.0:
            raise FloatingPointError(f"initial validation metric {name!r} must be positive")
        ratios.append(metrics[name] / denominator)
    return sum(ratios) / len(ratios)


def validation_stratified_evidence(
    predictions: tuple[GeometrySSLForward, ...],
    batches: tuple[PaddedOnlineGeometryBatch, ...],
) -> dict[str, object]:
    r"""聚合固定 held-out morphology bank 的分层、形态等权 selection evidence。"""

    if len(predictions) != len(batches) or not batches:
        raise ValueError("validation predictions and batches must be aligned and non-empty")
    blocks = tuple(
        (batch.asset_ids, geometry_ssl_stratified_components_per_sample(prediction, batch))
        for prediction, batch in zip(predictions, batches)
    )
    return aggregate_geometry_ssl_stratified_components(blocks)


def stratified_metric_scores(evidence: dict[str, object]) -> dict[str, float]:
    r"""从动态分层 evidence 严格提取三个有限、非负 checkpoint metrics。"""

    raw = evidence.get("metric_scores")
    if not isinstance(raw, dict) or set(raw) != {"density", "kappa", "derived_field"}:
        raise ValueError("stratified validation evidence has invalid metric_scores keys")
    result: dict[str, float] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not isinstance(value, (int, float)):
            raise ValueError("stratified validation metric scores must be numeric")
        score = float(value)
        if not torch.isfinite(torch.tensor(score)) or score < 0.0:
            raise ValueError("stratified validation metric scores must be finite and non-negative")
        result[name] = score
    return result


def fixed_validation_ablation_evidence(
    model: GeometrySSLModel,
    batches: tuple[PaddedOnlineGeometryBatch, ...],
) -> dict[str, object]:
    r"""在 held-out morphology bank 上生成可做 `(asset,q)` 配对 bootstrap 的 ablation evidence。"""

    records: list[dict[str, object]] = []
    ablations = (
        "query_only",
        "same_asset_q_shuffle",
        "cross_asset_shuffle",
        "first_order_zero",
        "first_order_joint_shuffle",
        "first_order_sign_flip",
    )
    for block_index, batch in enumerate(batches):
        q = batch.q.detach()  # ablation 不重新采样物理构型
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
            predictions["same_asset_q_shuffle"] = None  # 单 q 尾 block 不补造同资产样本
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
            predictions["cross_asset_shuffle"] = None  # 单资产尾 block 不伪造跨手来源
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
        for name in ("first_order_zero", "first_order_joint_shuffle", "first_order_sign_flip"):
            predictions[name] = geometry_ssl_ablation_forward(
                model,
                q,
                batch.evidence,
                batch.queries.query_points_h,
                batch.field_targets.bandwidths,
                ablation=name,
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
        "split": "held_out_morphology",
        "pairing_key": ["asset_id", "q_index"],
        "ablations": ("full", *ablations),
        "records": records,
    }


def stream_training_morphology_q_bank(
    model: GeometrySSLModel,
    runtimes: tuple[GeometrySource, ...],
    *,
    representation_config: GeometryRepresentationCfg,
    seed: int,
    q_per_asset: int,
    assets_per_minibatch: int,
    q_per_asset_per_minibatch: int,
    max_resident_assets: int,
    device: torch.device,
    dtype: torch.dtype,
    phase: str,
) -> dict[str, object]:
    r"""流式评估训练形态上的独立 q bank，并保存可确定性重放的逐样本 evidence。

    初始与最终评估从独立 seed 的 cursor 0 重建同一序列；q、query、density teacher 与一阶 teacher
    共同进入 SHA-256。bank 不常驻训练期 GPU，只在 initial/final 各流式执行一次。
    """

    bank_seed = int(seed)  # 与 train/held-out/bootstrap seed 空间分离
    runtime_config = GeometrySSLRuntimeCfg(
        max_resident_assets=max_resident_assets,
        assets_per_minibatch=min(assets_per_minibatch, len(runtimes)),
        q_per_asset_per_minibatch=q_per_asset_per_minibatch,
        q_per_asset_per_epoch=q_per_asset,
        epochs=1,
    )
    representation = GeometryRepresentation(representation_config)
    window = ResidentGeometryAssetWindow(
        runtimes,
        device=str(device),
        dtype=dtype,
        max_resident_assets=runtime_config.max_resident_assets,
        loader=representation.to_device,
    )
    batcher = WindowedOnlineGeometryBatcher(
        runtimes,
        window,
        seed=bank_seed,
        runtime_config=runtime_config,
        field_config=fixed_validation_gaussian_field_config(representation_config.field),
        query_config=representation_config.query,
        target_config=representation_config.target,
        padding=representation_config.layout,
    )
    digest = hashlib.sha256()  # q/query/teacher byte-level identity
    digest.update(b"geometry-ssl-train-morphology-q-bank-v2\0")
    records: list[dict[str, object]] = []
    cpu_rng_state = torch.get_rng_state()  # validation 不推进正式训练 RNG
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    model.eval()
    try:
        with torch.no_grad():
            while batcher.epoch < 1:
                batch = batcher.sample()
                prediction = model(
                    batch.q,
                    batch.evidence,
                    batch.queries.query_points_h,
                    batch.field_targets.bandwidths,
                    owner_index=batch.sensitivity_targets.owner_index,
                    query_index=batch.sensitivity_targets.query_index,
                    joint_index=batch.sensitivity_targets.joint_index,
                )
                metrics = geometry_ssl_reconstruction_metrics_per_sample(prediction, batch)
                q_indices = batch.q_index.tolist() if batch.q_index is not None else [-1] * len(batch.asset_ids)
                _update_training_q_bank_digest(digest, batch)
                for sample_index, (asset_id, q_index) in enumerate(zip(batch.asset_ids, q_indices)):
                    records.append(
                        {
                            "asset_id": asset_id,
                            "q_index": int(q_index),
                            "metrics": {metric: per_sample[sample_index] for metric, per_sample in metrics.items()},
                        }
                    )
                window.drain_telemetry_events()  # q-bank evidence 不混入 optimizer runtime 日志
    finally:
        window.release_all()
        window.drain_telemetry_events()
        torch.set_rng_state(cpu_rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_rng_state)
    return {
        "split": "training_morphology_independent_q",
        "phase": phase,
        "seed": bank_seed,
        "q_per_asset": q_per_asset,
        "asset_count": len(runtimes),
        "record_count": len(records),
        "bank_digest_sha256": digest.hexdigest(),
        "records": records,
    }


def compare_training_q_banks(
    initial: dict[str, object],
    final: dict[str, object],
) -> dict[str, dict[str, float | bool]]:
    r"""验证两次 replay identity，并返回 `initial-final` 的 morphology-balanced 改善量。"""

    if initial.get("bank_digest_sha256") != final.get("bank_digest_sha256"):
        raise RuntimeError("training morphology q bank replay digest changed between initial and final evaluation")
    initial_summary = _summarize_training_q_bank(initial)
    final_summary = _summarize_training_q_bank(final)
    return {
        metric: {
            "initial": initial_summary[metric]["asset_balanced_mean"],
            "final": final_summary[metric]["asset_balanced_mean"],
            "improvement_initial_minus_final": (
                initial_summary[metric]["asset_balanced_mean"] - final_summary[metric]["asset_balanced_mean"]
            ),
            "improved": final_summary[metric]["asset_balanced_mean"] < initial_summary[metric]["asset_balanced_mean"],
        }
        for metric in ("density", "kappa", "derived_field")
    }


def _update_training_q_bank_digest(digest: _HashWriter, batch: PaddedOnlineGeometryBatch) -> None:
    r"""把固定 bank 的路由、q、query 与 teacher 张量按稳定顺序写入 SHA-256。"""

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


def _summarize_training_q_bank(evidence: dict[str, object]) -> dict[str, dict[str, float]]:
    r"""先在每项训练 morphology 内 q-平均，再跨 morphology 等权平均 raw MSE。"""

    records = evidence.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("training morphology q bank requires non-empty records")
    asset_ids = tuple(dict.fromkeys(str(record["asset_id"]) for record in records))
    summary: dict[str, dict[str, float]] = {}
    for metric in ("density", "kappa", "derived_field"):
        by_asset: dict[str, float] = {}
        for asset_id in asset_ids:
            values = [
                float(record["metrics"][metric])
                for record in records
                if record["asset_id"] == asset_id and record["metrics"][metric] is not None
            ]
            if values:
                by_asset[asset_id] = sum(values) / len(values)
        if not by_asset:
            raise ValueError(f"training morphology q bank has no valid metric={metric!r}")
        summary[metric] = {
            "asset_balanced_mean": sum(by_asset.values()) / len(by_asset),
            "asset_count": float(len(by_asset)),
        }
    return summary


class _HashWriter(Protocol):
    r"""固定 bank digest 所需的最小增量 hash 接口。"""

    def update(self, data: bytes, /) -> object:
        r"""把一段稳定字节序列纳入摘要。"""
        ...


__all__ = [
    "compare_training_q_banks",
    "fixed_validation_ablation_evidence",
    "normalized_validation_score",
    "stratified_metric_scores",
    "stream_training_morphology_q_bank",
    "validation_stratified_evidence",
]
