r"""AR-MPJ-005：小规模 morphology-disjoint relation-Jacobian 单目标泛化实验。

实验使用同一 evaluation catalog 的不重叠资产区间形成 64-asset train 与 32-asset validation，并用
32 个 unseen-mother assets 形成更强外部测试。每项资产固定 4 个 Sobol q、每 joint 固定 2 active + 1
PALM structural-zero material edges。模型仍为 width-64/layers-2，仅训练 anchor-relational Material-point
Jacobian，不联合 density 或 κ。

该 probe 只判断目标是否具备小样本跨 morphology 可学性。固定 q bank 结果不能替代正式 online 训练，
但足以证伪“tiny overfit 只是在记忆 8 个 assets”的替代解释。
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
    FixedRelationBatch,
    TinyRelationJacobianModel,
    _build_fixed_relation_batch,
    _loss_and_metrics,
)


def _build_suite_bank(
    method: Any,
    cfg: Any,
    *,
    suite: str,
    asset_count: int,
    q_per_asset: int,
    assets_per_batch: int,
    points_per_edge: int,
) -> tuple[list[FixedRelationBatch], tuple[str, ...]]:
    r"""按 suite 的稳定 catalog 前缀构造 GPU-resident fixed target batches。"""

    device = torch.device("cuda:0")
    suite_index = 0 if suite == "unseen_variant_set" else 1  # 与正式 evaluation seed 分离
    session = method.open_session(
        "evaluation",
        suite=suite,
        seed=cfg.run.seed + cfg.evaluation.evaluation_seed_offset + suite_index * 1_000_003,
        device=device,
        dtype=torch.float32,
        max_resident_assets=assets_per_batch,
        window_factory=ResidentGeometryAssetWindow,
    )
    schedule = FixedAssetQSchedule(
        asset_count,
        q_per_asset=q_per_asset,
        assets_per_minibatch=assets_per_batch,
        q_per_asset_per_minibatch=2,
        max_resident_assets=assets_per_batch,
    )
    batches: list[FixedRelationBatch] = []
    asset_ids: list[str] = []
    step = 0
    try:
        while not schedule.complete:
            realized = session.realize(schedule.next(), schedule=schedule, step=step)
            fixed = _build_fixed_relation_batch(realized, session, points_per_edge=points_per_edge)
            batches.append(fixed)
            asset_ids.extend(fixed.asset_ids)
            step += 1
    finally:
        session.close()
    return batches, tuple(dict.fromkeys(asset_ids))


def _aggregate_metrics(
    model: TinyRelationJacobianModel,
    batches: list[FixedRelationBatch],
    *,
    use_latent: bool,
) -> dict[str, Any]:
    r"""按实际 valid scalar 数聚合 objective、zero baseline、四通道 skill 与 zero leakage。"""

    device = batches[0].q.device
    scale = torch.tensor(CHANNEL_SCALE, device=device, dtype=torch.float64)  # `[4]` 固定数值尺度
    objective_sum = torch.zeros((), device=device, dtype=torch.float64)
    baseline_sum = torch.zeros_like(objective_sum)
    valid_count = 0
    active_error = torch.zeros(4, device=device, dtype=torch.float64)
    active_baseline = torch.zeros_like(active_error)
    active_count = torch.zeros(4, device=device, dtype=torch.long)
    sign_correct = torch.zeros(4, device=device, dtype=torch.long)
    sign_count = torch.zeros(4, device=device, dtype=torch.long)
    zero_square = torch.zeros((), device=device, dtype=torch.float64)
    zero_count = 0
    model.eval()
    with torch.no_grad():
        for batch in batches:
            prediction = model(batch, use_latent=use_latent)
            target = batch.target
            channel_valid = torch.ones_like(target, dtype=torch.bool)
            channel_valid[..., 1] = batch.radius_valid_mask
            valid = batch.edge_valid_mask[:, :, None, None] & batch.anchor_valid_mask[:, None, :, None] & channel_valid
            residual = (prediction.double() - target.double()) / scale
            normalized_target = target.double() / scale
            objective_sum += residual.square()[valid].sum()
            baseline_sum += normalized_target.square()[valid].sum()
            valid_count += int(valid.sum())
            active = valid & batch.active_mask[:, :, None, None]
            structural_zero = valid & batch.edge_valid_mask[:, :, None, None] & ~batch.active_mask[:, :, None, None]
            zero_square += prediction.double().square()[structural_zero].sum()
            zero_count += int(structural_zero.sum())
            for channel in range(4):
                channel_active = active[..., channel]
                error = prediction[..., channel].double() - target[..., channel].double()
                active_error[channel] += error.square()[channel_active].sum()
                active_baseline[channel] += target[..., channel].double().square()[channel_active].sum()
                active_count[channel] += channel_active.sum()
                nonzero = channel_active & (target[..., channel].abs() >= 1.0e-5)
                sign_correct[channel] += (
                    torch.sign(prediction[..., channel][nonzero]) == torch.sign(target[..., channel][nonzero])
                ).sum()
                sign_count[channel] += nonzero.sum()
    objective = objective_sum / valid_count
    zero_baseline = baseline_sum / valid_count
    channel_names = ("height", "radius", "dot", "chirality")
    channels = {
        name: {
            "active_mse": float(active_error[channel] / active_count[channel]),
            "active_zero_baseline": float(active_baseline[channel] / active_count[channel]),
            "active_skill": float(1.0 - active_error[channel] / active_baseline[channel].clamp_min(1.0e-30)),
            "active_sign_accuracy": float(sign_correct[channel].double() / sign_count[channel].clamp_min(1)),
        }
        for channel, name in enumerate(channel_names)
    }
    return {
        "objective": float(objective),
        "zero_baseline": float(zero_baseline),
        "skill": float(1.0 - objective / zero_baseline.clamp_min(1.0e-30)),
        "structural_zero_prediction_rms": float(torch.sqrt(zero_square / max(1, zero_count))),
        "valid_scalar_count": valid_count,
        "channels": channels,
    }


def _train(
    train_batches: list[FixedRelationBatch],
    validation_batches: list[FixedRelationBatch],
    mother_batches: list[FixedRelationBatch],
    *,
    updates: int,
    learning_rate: float,
    use_latent: bool,
    seed: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    r"""在 train fixed bank 循环更新，并按 250-update cadence 检查 morphology-disjoint validation。"""

    torch.manual_seed(seed)
    device = train_batches[0].q.device
    model = TinyRelationJacobianModel().to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 17)
    order: list[int] = []
    trajectory: list[dict[str, Any]] = []
    torch.cuda.synchronize()
    started = perf_counter()
    for update in range(1, updates + 1):
        if not order:
            order = torch.randperm(len(train_batches), generator=generator).tolist()  # 每个 bank cycle 重排 batch
        batch = train_batches[order.pop()]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch, use_latent=use_latent)
        loss, train_metrics = _loss_and_metrics(prediction, batch)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        if update == 1 or update % 250 == 0 or update == updates:
            validation = _aggregate_metrics(model, validation_batches, use_latent=use_latent)
            trajectory.append(
                {
                    "update": update,
                    "batch_train_skill": train_metrics["skill"],
                    "gradient_norm": float(gradient_norm),
                    "validation": validation,
                }
            )
    torch.cuda.synchronize()
    elapsed = perf_counter() - started
    final = {
        "train": _aggregate_metrics(model, train_batches, use_latent=use_latent),
        "validation_variant": _aggregate_metrics(model, validation_batches, use_latent=use_latent),
        "test_mother": _aggregate_metrics(model, mother_batches, use_latent=use_latent),
    }
    report = {
        "use_latent": use_latent,
        "updates": updates,
        "learning_rate": learning_rate,
        "elapsed_seconds": elapsed,
        "updates_per_second": updates / elapsed,
        "final": final,
        "trajectory": trajectory,
    }
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    return report, state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small morphology-disjoint relation-Jacobian pilot.")
    parser.add_argument("--train-assets", type=int, default=64)
    parser.add_argument("--validation-assets", type=int, default=32)
    parser.add_argument("--mother-assets", type=int, default=32)
    parser.add_argument("--q-per-asset", type=int, default=4)
    parser.add_argument("--assets-per-batch", type=int, default=8)
    parser.add_argument("--points-per-edge", type=int, default=1)
    parser.add_argument("--full-updates", type=int, default=2000)
    parser.add_argument("--query-only-updates", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/autoresearch/material_point_jacobian/AR-MPJ-005"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = (
        args.train_assets,
        args.validation_assets,
        args.mother_assets,
        args.q_per_asset,
        args.assets_per_batch,
        args.points_per_edge,
        args.full_updates,
        args.query_only_updates,
    )
    if min(counts) < 1:
        raise ValueError("all count arguments must be positive")
    if any(count % args.assets_per_batch for count in (args.train_assets, args.validation_assets, args.mother_assets)):
        raise ValueError("all asset split counts must be divisible by assets_per_batch")
    device = torch.device("cuda:0")
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
    variant_count = args.train_assets + args.validation_assets
    variant_batches, variant_ids = _build_suite_bank(
        method,
        cfg,
        suite="unseen_variant_set",
        asset_count=variant_count,
        q_per_asset=args.q_per_asset,
        assets_per_batch=args.assets_per_batch,
        points_per_edge=args.points_per_edge,
    )
    # Fixed schedule 是 asset-group major；每组完整 q blocks 连续，因此按稳定 asset ID 切分 batch。
    train_id_set = set(variant_ids[: args.train_assets])
    validation_id_set = set(variant_ids[args.train_assets :])
    train_batches = [batch for batch in variant_batches if set(batch.asset_ids).issubset(train_id_set)]
    validation_batches = [batch for batch in variant_batches if set(batch.asset_ids).issubset(validation_id_set)]
    if len(train_batches) + len(validation_batches) != len(variant_batches):
        raise RuntimeError("variant batch crosses the declared morphology split")
    mother_batches, mother_ids = _build_suite_bank(
        method,
        cfg,
        suite="unseen_mother",
        asset_count=args.mother_assets,
        q_per_asset=args.q_per_asset,
        assets_per_batch=args.assets_per_batch,
        points_per_edge=args.points_per_edge,
    )
    method.close()
    if train_id_set & validation_id_set or train_id_set & set(mother_ids) or validation_id_set & set(mother_ids):
        raise RuntimeError("morphology-disjoint pilot asset IDs overlap")

    full_report, full_state = _train(
        train_batches,
        validation_batches,
        mother_batches,
        updates=args.full_updates,
        learning_rate=args.learning_rate,
        use_latent=True,
        seed=20260830,
    )
    query_report, query_state = _train(
        train_batches,
        validation_batches,
        mother_batches,
        updates=args.query_only_updates,
        learning_rate=args.learning_rate,
        use_latent=False,
        seed=20260830,
    )
    report = {
        "case": "AR-MPJ-005",
        "population": {
            "train_assets": args.train_assets,
            "validation_variant_assets": args.validation_assets,
            "test_mother_assets": args.mother_assets,
            "q_per_asset": args.q_per_asset,
            "train_batches": len(train_batches),
            "validation_batches": len(validation_batches),
            "mother_batches": len(mother_batches),
            "asset_overlap": False,
        },
        "model": {
            "encoder_hidden_width": 64,
            "encoder_layers": 2,
            "relation_width": 32,
            "channel_scale": list(CHANNEL_SCALE),
        },
        "full": full_report,
        "query_only": query_report,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    torch.save(
        {
            "schema": "armj-small-generalization-v1",
            "report": report,
            "full_state": full_state,
            "query_only_state": query_state,
        },
        args.output_dir / "models.pt",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
