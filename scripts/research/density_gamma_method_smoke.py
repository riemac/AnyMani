r"""Density + Gamma formal method 的真实 source/forward/backward smoke。"""

from __future__ import annotations

import json
from argparse import ArgumentParser
from dataclasses import replace

import torch
from anymani.distill.ssl.config_store import compose_evaluation_cfg
from anymani.distill.ssl.contracts import build_runtime
from anymani.distill.ssl.runtime.sampling import FixedAssetQSchedule
from anymani.distill.ssl.runtime.scheduler import ResidentGeometryAssetWindow


def main() -> None:
    r"""在两个 held-out assets 上闭合 source cache、联合 target、forward 与 FairGrad backward。"""

    parser = ArgumentParser()
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    config = compose_evaluation_cfg(config_ref="geometry_ssl_density_material_jacobian_v0_8_0")
    if args.compile:
        config = replace(
            config,
            evaluation=replace(
                config.evaluation,
                execution=replace(config.evaluation.execution, compile_enabled=True),
            ),
        )
    device = torch.device("cuda:0")
    catalog = build_runtime(config.data).resolve_evaluation()
    method = build_runtime(config.method)
    method.configure_source_artifacts(
        root=config.evaluation.source_cache_root,
        mode="readonly",
        dataset_manifest_sha256=str(catalog.dataset.source_sha256),
        producer_device=str(device),
        role="evaluation",
    )
    method.prepare(catalog, role="evaluation", device=device, dtype=torch.float32)
    method.configure_execution(config.evaluation.execution)
    method.initialize_model(device=device, dtype=torch.float32)
    session = method.open_session(
        "evaluation",
        suite="unseen_variant_set",
        seed=config.run.seed + config.evaluation.evaluation_seed_offset,
        device=device,
        dtype=torch.float32,
        max_resident_assets=2,
        window_factory=ResidentGeometryAssetWindow,
    )
    schedule = FixedAssetQSchedule(
        2,
        q_per_asset=2,
        assets_per_minibatch=2,
        q_per_asset_per_minibatch=2,
        max_resident_assets=2,
    )
    try:
        batch = session.realize(schedule.next(), schedule=schedule, step=0)
        method.train_mode()
        step = method.forward_objectives(batch, step=0, mode="train")
        for parameter in method.parameters():
            parameter.grad = None
        update = method.backward_update_units(
            (batch,),
            forward_step=0,
            logical_sample_count=batch.q.shape[0],
            microbatch_size=batch.q.shape[0],
        )
        group_norms = {}
        for group in method.optimizer_parameter_groups():
            square = sum(
                float(parameter.grad.detach().double().square().sum())
                for parameter in group.parameters
                if parameter.grad is not None
            )
            group_norms[group.name] = square**0.5
        report = {
            "asset_ids": list(batch.asset_ids),
            "q_shape": list(batch.q.shape),
            "density_shape": list(batch.field_targets.density.shape),
            "gamma_shape": list(batch.material_targets.relation_sensitivity_per_rad.shape),
            "material_point_index_shape": list(batch.material_point_index.shape),
            "objective_names": list(step.objectives),
            "update_terms": update.terms,
            "gradient_group_norms": group_norms,
            "anchor_bank": batch.anchor_index.tolist(),
            "kappa_present": hasattr(batch, "sensitivity_targets"),
            "compile_enabled": bool(config.evaluation.execution.compile_enabled),
        }
        print(json.dumps(report, indent=2))
    finally:
        session.close()
        method.close()


if __name__ == "__main__":
    main()
