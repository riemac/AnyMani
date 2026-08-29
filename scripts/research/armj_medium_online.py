r"""AR-MPJ-006：512-asset online fresh-q 中型 relation-Jacobian 单目标实验。

训练 split 使用 unseen-variant catalog 前 512 项；每个 epoch 完整遍历一次 morphology catalog，每项资产
产生 2 个新的 scrambled-Sobol q，因此 16 epochs 共形成 16,384 个 `(asset,q)` samples。验证 split 使用
variant indices 512–575 的固定 4-q bank，外部测试使用 64 个 unseen-mother assets 的固定 4-q bank。

模型恢复到 N020 量级的 width-128/layers-4 trunk，但只保留 anchor-relational Material-point Jacobian
objective。训练过程不物化跨 epoch target bank，不联合 density/κ/FairGrad；每个 update 的 physical target
在当前 resident 8-asset window 上在线生成并在 backward 后释放。
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
from anymani.distill.ssl.runtime.sampling import (
    OnlineMinibatchSchedule,
    OnlineSamplingCfg,
    ScheduledMinibatch,
)
from anymani.distill.ssl.runtime.scheduler import ResidentGeometryAssetWindow
from armj_small_generalization import _aggregate_metrics, _build_suite_bank
from armj_tiny_overfit import (
    FixedRelationBatch,
    TinyRelationJacobianModel,
    _build_fixed_relation_batch,
    _loss_and_metrics,
)


def _build_offset_variant_bank(
    method: Any,
    cfg: Any,
    *,
    asset_start: int,
    asset_count: int,
    q_per_asset: int,
    assets_per_batch: int,
    points_per_edge: int,
) -> tuple[list[FixedRelationBatch], tuple[str, ...]]:
    r"""只 realization 指定 variant catalog 区间，避免为 held-out offset 构造前缀 assets。"""

    if asset_count % assets_per_batch or q_per_asset % 2:
        raise ValueError("offset validation bank requires complete 8-asset and 2-q blocks")
    device = torch.device("cuda:0")
    session = method.open_session(
        "evaluation",
        suite="unseen_variant_set",
        seed=cfg.run.seed + cfg.evaluation.evaluation_seed_offset + 2_000_006,
        device=device,
        dtype=torch.float32,
        max_resident_assets=assets_per_batch,
        window_factory=ResidentGeometryAssetWindow,
    )
    batches: list[FixedRelationBatch] = []
    ids: list[str] = []
    minibatch_index = 0
    try:
        for group_start in range(asset_start, asset_start + asset_count, assets_per_batch):
            asset_indices = tuple(range(group_start, group_start + assets_per_batch))
            for q_block in range(q_per_asset // 2):
                item = ScheduledMinibatch(
                    minibatch_index=minibatch_index,
                    epoch_index=-1,
                    minibatch_index_in_epoch=minibatch_index,
                    q_block_index=q_block,
                    asset_group=group_start // assets_per_batch,
                    asset_indices=asset_indices,
                    q_per_asset=2,
                    resident_asset_indices=asset_indices,
                    window_index=(group_start - asset_start) // assets_per_batch,
                )
                realized = session.realize(item, schedule=None, step=minibatch_index)
                fixed = _build_fixed_relation_batch(realized, session, points_per_edge=points_per_edge)
                batches.append(fixed)
                ids.extend(fixed.asset_ids)
                minibatch_index += 1
    finally:
        session.close()
    return batches, tuple(dict.fromkeys(ids))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Medium online relation-Jacobian experiment.")
    parser.add_argument("--train-assets", type=int, default=512)
    parser.add_argument("--validation-assets", type=int, default=64)
    parser.add_argument("--mother-assets", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--assets-per-batch", type=int, default=8)
    parser.add_argument("--q-per-asset-per-batch", type=int, default=2)
    parser.add_argument("--validation-q-per-asset", type=int, default=4)
    parser.add_argument("--points-per-edge", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/autoresearch/material_point_jacobian/AR-MPJ-006"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = (
        args.train_assets,
        args.validation_assets,
        args.mother_assets,
        args.epochs,
        args.assets_per_batch,
        args.q_per_asset_per_batch,
        args.validation_q_per_asset,
        args.points_per_edge,
        args.checkpoint_every_epochs,
    )
    if min(counts) < 1:
        raise ValueError("all count arguments must be positive")
    if args.train_assets % args.assets_per_batch or args.validation_assets % args.assets_per_batch:
        raise ValueError("train/validation assets must divide into complete minibatches")
    if args.mother_assets % args.assets_per_batch:
        raise ValueError("mother assets must divide into complete minibatches")
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    # 两个验证 bank 在训练前固定，确保每个 epoch 比较完全相同的 morphology/q/material identities。
    validation_batches, validation_ids = _build_offset_variant_bank(
        method,
        cfg,
        asset_start=args.train_assets,
        asset_count=args.validation_assets,
        q_per_asset=args.validation_q_per_asset,
        assets_per_batch=args.assets_per_batch,
        points_per_edge=args.points_per_edge,
    )
    mother_batches, mother_ids = _build_suite_bank(
        method,
        cfg,
        suite="unseen_mother",
        asset_count=args.mother_assets,
        q_per_asset=args.validation_q_per_asset,
        assets_per_batch=args.assets_per_batch,
        points_per_edge=args.points_per_edge,
    )
    if set(validation_ids) & set(mother_ids):
        raise RuntimeError("variant validation and mother test asset IDs overlap")

    # Online schedule 每 epoch 恰好一轮 512-asset catalog，下一轮重新打乱并为每 asset 继续 Sobol cursor。
    minibatches_per_epoch = args.train_assets // args.assets_per_batch
    schedule = OnlineMinibatchSchedule(
        args.train_assets,
        OnlineSamplingCfg(
            assets_per_minibatch=args.assets_per_batch,
            q_per_asset_per_minibatch=args.q_per_asset_per_batch,
            shuffle_assets=True,
            seed=args.seed,
        ),
        max_epochs=args.epochs,
        num_minibatches=minibatches_per_epoch,
        max_resident_assets=args.assets_per_batch,
    )
    train_session = method.open_session(
        "evaluation",
        suite="unseen_variant_set",
        seed=cfg.run.seed + cfg.evaluation.evaluation_seed_offset + 3_000_009,
        device=device,
        dtype=torch.float32,
        max_resident_assets=args.assets_per_batch,
        window_factory=ResidentGeometryAssetWindow,
    )

    model = TinyRelationJacobianModel(hidden_width=128, layers=4, relation_width=64).to(
        device=device,
        dtype=torch.float32,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4)
    trajectory: list[dict[str, Any]] = []
    teacher_seconds = 0.0
    target_seconds = 0.0
    model_seconds = 0.0
    processed_pairs = 0
    processed_target_scalars = 0
    epoch_skill_sum = 0.0
    epoch_batch_count = 0

    # Epoch 0 记录随机网络 baseline；后续每个完整 catalog boundary 重算同一 held-out banks。
    trajectory.append(
        {
            "epoch": 0,
            "optimizer_updates": 0,
            "validation_variant": _aggregate_metrics(model, validation_batches, use_latent=True),
            "test_mother": _aggregate_metrics(model, mother_batches, use_latent=True),
        }
    )
    update = 0
    torch.cuda.synchronize()
    overall_started = perf_counter()
    try:
        while not schedule.complete:
            item = schedule.next()
            torch.cuda.synchronize()
            teacher_started = perf_counter()
            realized = train_session.realize(item, schedule=schedule, step=update)
            torch.cuda.synchronize()
            teacher_seconds += perf_counter() - teacher_started

            target_started = perf_counter()
            fixed_batch = _build_fixed_relation_batch(
                realized,
                train_session,
                points_per_edge=args.points_per_edge,
            )
            torch.cuda.synchronize()
            target_seconds += perf_counter() - target_started

            model_started = perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(fixed_batch, use_latent=True)
            loss, metrics = _loss_and_metrics(prediction, fixed_batch)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            torch.cuda.synchronize()
            model_seconds += perf_counter() - model_started

            update += 1
            processed_pairs += fixed_batch.q.shape[0]
            valid_scalars = (
                fixed_batch.edge_valid_mask[:, :, None]
                & fixed_batch.anchor_valid_mask[:, None, :]
            ).sum() * 4
            processed_target_scalars += int(valid_scalars)
            epoch_skill_sum += float(metrics["skill"])
            epoch_batch_count += 1

            if schedule.epoch_boundary:
                epoch = schedule.completed_epochs
                validation = _aggregate_metrics(model, validation_batches, use_latent=True)
                mother = _aggregate_metrics(model, mother_batches, use_latent=True)
                trajectory.append(
                    {
                        "epoch": epoch,
                        "optimizer_updates": update,
                        "mean_online_train_batch_skill": epoch_skill_sum / epoch_batch_count,
                        "last_gradient_norm": float(gradient_norm),
                        "validation_variant": validation,
                        "test_mother": mother,
                    }
                )
                epoch_skill_sum = 0.0
                epoch_batch_count = 0
                if epoch % args.checkpoint_every_epochs == 0 or epoch == args.epochs:
                    torch.save(
                        {
                            "schema": "armj-medium-online-v1",
                            "epoch": epoch,
                            "optimizer_updates": update,
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "trajectory": trajectory,
                        },
                        args.output_dir / f"epoch_{epoch:04d}.pt",
                    )
    finally:
        train_session.close()
        method.close()
    torch.cuda.synchronize()
    overall_seconds = perf_counter() - overall_started

    final = {
        "validation_variant": _aggregate_metrics(model, validation_batches, use_latent=True),
        "test_mother": _aggregate_metrics(model, mother_batches, use_latent=True),
    }
    report = {
        "case": "AR-MPJ-006",
        "population": {
            "train_assets": args.train_assets,
            "validation_variant_assets": args.validation_assets,
            "test_mother_assets": args.mother_assets,
            "epochs": args.epochs,
            "fresh_q_per_asset_per_epoch": args.q_per_asset_per_batch,
            "processed_asset_q_pairs": processed_pairs,
            "processed_target_scalars": processed_target_scalars,
            "optimizer_updates": update,
            "asset_overlap": False,
        },
        "model": {
            "hidden_width": 128,
            "layers": 4,
            "relation_width": 64,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "learning_rate": args.learning_rate,
        },
        "runtime": {
            "overall_seconds": overall_seconds,
            "teacher_realization_seconds": teacher_seconds,
            "relation_target_seconds": target_seconds,
            "model_update_seconds": model_seconds,
            "asset_q_pairs_per_second": processed_pairs / overall_seconds,
            "target_scalars_per_second_overall": processed_target_scalars / overall_seconds,
            "target_scalars_per_second_target_stage": processed_target_scalars / max(target_seconds, 1.0e-12),
        },
        "final": final,
        "trajectory": trajectory,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
