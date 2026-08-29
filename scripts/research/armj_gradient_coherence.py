r"""AR-MPJ-004：Material-point relation Jacobian 的跨 morphology shared-gradient coherence。

N020 direct κ 的最终 shared-encoder gradient 在 32 个独立 batches 上只有 raw coherence 0.287、unit
coherence 0.255，off-diagonal cosine mean 0.035。该 probe 在相同因果层级检查新单目标：固定一个已完成
tiny-overfit 的 width-64/layers-2 模型，在未参与 tiny bank 的 asset batches 上分别计算总 objective 与四个
relation channels 对全部 retained encoder 参数的梯度方向。

Reader private parameters不进入 coherence matrix；目标是判断不同 morphology/q batches 是否对共享表示提出
相容的更新方向，而不是报告 reader 自身容易拟合。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from anymani.distill.ssl.config_store import compose_evaluation_cfg
from anymani.distill.ssl.contracts import build_runtime
from anymani.distill.ssl.runtime.sampling import FixedAssetQSchedule
from anymani.distill.ssl.runtime.scheduler import ResidentGeometryAssetWindow
from armj_tiny_overfit import (
    CHANNEL_SCALE,
    TinyRelationJacobianModel,
    _build_fixed_relation_batch,
    _loss_and_metrics,
)


def _flatten_gradients(
    gradients: tuple[torch.Tensor | None, ...],
    parameters: tuple[torch.nn.Parameter, ...],
) -> torch.Tensor:
    r"""按稳定 parameter 顺序把全 encoder 梯度拼成 CPU FP32 向量。"""

    values = [
        torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.detach().reshape(-1)
        for parameter, gradient in zip(parameters, gradients, strict=True)
    ]
    return torch.cat(values).to(device="cpu", dtype=torch.float32)


def _direction_report(matrix: torch.Tensor) -> dict[str, float | int]:
    r"""报告 raw/unit coherence、batch cosine 分布与 gradient-matrix entropy rank。"""

    x = matrix.double()  # `[T,P]`，T 个独立 morphology batches、P 个 encoder parameters
    norms = torch.linalg.vector_norm(x, dim=1)  # 每个 batch 的完整 encoder gradient norm
    unit = x / norms[:, None].clamp_min(1.0e-30)  # 只保留方向，删除 batch gradient scale
    cosine = unit @ unit.T  # `[T,T]` pairwise batch cosine
    off_diagonal = cosine[~torch.eye(len(x), dtype=torch.bool)]  # 排除恒为 1 的自相似
    gram = x @ x.T  # 小型 `[T,T]` Gram，避免物化 parameter-space SVD
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    probability = eigenvalues / eigenvalues.sum().clamp_min(1.0e-30)
    positive = probability > 0.0
    return {
        "batch_count": len(x),
        "parameter_count": x.shape[1],
        "norm_mean": float(norms.mean()),
        "norm_median": float(norms.median()),
        "raw_coherence": float(torch.linalg.vector_norm(x.sum(dim=0)) / norms.sum().clamp_min(1.0e-30)),
        "unit_coherence": float(torch.linalg.vector_norm(unit.sum(dim=0)) / len(x)),
        "offdiag_cosine_mean": float(off_diagonal.mean()),
        "offdiag_cosine_median": float(off_diagonal.median()),
        "offdiag_cosine_q05": float(torch.quantile(off_diagonal, 0.05)),
        "offdiag_cosine_q95": float(torch.quantile(off_diagonal, 0.95)),
        "positive_cosine_fraction": float((off_diagonal > 0.0).double().mean()),
        "gradient_entropy_rank": float(torch.exp(-(probability[positive] * probability[positive].log()).sum())),
    }


def _channel_losses(prediction: torch.Tensor, batch: Any) -> dict[str, torch.Tensor]:
    r"""返回与 tiny objective 相同尺度的总 loss 和逐 channel loss。"""

    total, _metrics = _loss_and_metrics(prediction, batch)  # 2 active + 1 zero、四通道共同 objective
    scale = torch.tensor(CHANNEL_SCALE, device=prediction.device, dtype=prediction.dtype)
    channel_valid = torch.ones_like(batch.target, dtype=torch.bool)
    channel_valid[..., 1] = batch.radius_valid_mask  # radius channel 的唯一额外奇点 mask
    valid = batch.edge_valid_mask[:, :, None, None] & batch.anchor_valid_mask[:, None, :, None] & channel_valid
    residual = (prediction - batch.target) / scale  # 与训练完全一致的无量纲残差
    names = ("height", "radius", "dot", "chirality")
    losses = {"total": total}
    for channel, name in enumerate(names):
        losses[name] = residual[..., channel].square()[valid[..., channel]].mean()
    return losses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure cross-batch relation-Jacobian encoder gradients.")
    parser.add_argument("--assets", type=int, default=128)
    parser.add_argument("--q-per-asset", type=int, default=2)
    parser.add_argument("--assets-per-batch", type=int, default=8)
    parser.add_argument("--skip-batches", type=int, default=1)
    parser.add_argument("--points-per-edge", type=int, default=1)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("logs/autoresearch/material_point_jacobian/AR-MPJ-003-ablations/models.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/autoresearch/material_point_jacobian/AR-MPJ-004/report.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.assets % args.assets_per_batch:
        raise ValueError("assets must be divisible by assets_per_batch")
    if not 0 <= args.skip_batches < args.assets // args.assets_per_batch:
        raise ValueError("skip_batches must leave at least one measured batch")
    device = torch.device("cuda:0")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "armj-tiny-overfit-v1":
        raise ValueError("checkpoint is not an AR-MPJ-003 tiny-overfit artifact")
    model = TinyRelationJacobianModel().to(device=device, dtype=torch.float32)
    model.load_state_dict(payload["full_state"])
    model.eval()
    parameters = tuple(model.encoder.parameters())  # coherence 只覆盖 retained/shared encoder

    cfg = compose_evaluation_cfg(config_ref="geometry_ssl_multitask_representation_v0_7_5")
    catalog = build_runtime(cfg.data).resolve_evaluation()
    method = build_runtime(cfg.method)
    method.configure_source_artifacts(
        root=cfg.evaluation.source_cache_root,
        mode="readonly",
        dataset_manifest_sha256=str(catalog.dataset.source_sha256),
        producer_device=str(device),
        role="evaluation",
    )
    method.prepare(catalog, role="evaluation", device=device, dtype=torch.float32)
    session = method.open_session(
        "evaluation",
        suite="unseen_variant_set",
        seed=cfg.run.seed + cfg.evaluation.evaluation_seed_offset,
        device=device,
        dtype=torch.float32,
        max_resident_assets=args.assets_per_batch,
        window_factory=ResidentGeometryAssetWindow,
    )
    schedule = FixedAssetQSchedule(
        args.assets,
        q_per_asset=args.q_per_asset,
        assets_per_minibatch=args.assets_per_batch,
        q_per_asset_per_minibatch=args.q_per_asset,
        max_resident_assets=args.assets_per_batch,
    )
    gradient_blocks: dict[str, list[torch.Tensor]] = {
        name: [] for name in ("total", "height", "radius", "dot", "chirality")
    }
    batch_metrics: list[dict[str, Any]] = []
    batch_index = 0
    torch.cuda.synchronize()
    started = perf_counter()
    try:
        while not schedule.complete:
            realized = session.realize(schedule.next(), schedule=schedule, step=batch_index)
            fixed_batch = _build_fixed_relation_batch(
                realized,
                session,
                points_per_edge=args.points_per_edge,
            )
            if batch_index >= args.skip_batches:
                prediction = model(fixed_batch, use_latent=True)
                losses = _channel_losses(prediction, fixed_batch)
                loss_names = tuple(losses)
                for loss_index, name in enumerate(loss_names):
                    gradients = torch.autograd.grad(
                        losses[name],
                        parameters,
                        retain_graph=loss_index + 1 < len(loss_names),
                        allow_unused=True,
                    )
                    gradient_blocks[name].append(_flatten_gradients(gradients, parameters))
                _loss, metrics = _loss_and_metrics(prediction, fixed_batch)
                batch_metrics.append(
                    {
                        "batch_index": batch_index,
                        "asset_ids": list(dict.fromkeys(fixed_batch.asset_ids)),
                        "skill": metrics["skill"],
                        "objective": metrics["objective"],
                        "zero_baseline": metrics["zero_baseline"],
                    }
                )
            batch_index += 1
    finally:
        session.close()
        method.close()
    torch.cuda.synchronize()
    elapsed = perf_counter() - started

    matrices = {name: torch.stack(values) for name, values in gradient_blocks.items()}
    reports = {name: _direction_report(matrix) for name, matrix in matrices.items()}
    channel_names = ("height", "radius", "dot", "chirality")
    mean_gradients = {name: matrices[name].double().mean(dim=0) for name in channel_names}
    mean_cosine: dict[str, float] = {}
    for left_index, left in enumerate(channel_names):
        for right in channel_names[left_index + 1 :]:
            numerator = torch.dot(mean_gradients[left], mean_gradients[right])
            denominator = torch.linalg.vector_norm(mean_gradients[left]) * torch.linalg.vector_norm(mean_gradients[right])
            mean_cosine[f"{left}__{right}"] = float(numerator / denominator.clamp_min(1.0e-30))
    report = {
        "case": "AR-MPJ-004",
        "population": {
            "assets_realized": args.assets,
            "assets_skipped_as_tiny_train_overlap": args.skip_batches * args.assets_per_batch,
            "measured_assets": args.assets - args.skip_batches * args.assets_per_batch,
            "q_per_asset": args.q_per_asset,
            "measured_batches": len(batch_metrics),
            "assets_per_batch": args.assets_per_batch,
        },
        "runtime_seconds": elapsed,
        "gradient_reports": reports,
        "population_mean_channel_gradient_cosine": mean_cosine,
        "batch_metrics": batch_metrics,
        "reference": {
            "n020_kappa_raw_coherence": 0.2874,
            "n020_kappa_unit_coherence": 0.2550,
            "n020_kappa_offdiag_cosine_mean": 0.03484,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
