r"""Density + Gamma full-batch 与 8-asset streaming units 的梯度 parity admission。"""

from __future__ import annotations

import json
from dataclasses import replace

import torch
from anymani.distill.ssl.config_store import compose_evaluation_cfg
from anymani.distill.ssl.contracts import build_runtime
from anymani.distill.ssl.runtime.sampling import FixedAssetQSchedule
from anymani.distill.ssl.runtime.scheduler import ResidentGeometryAssetWindow


def _build_method(config, catalog, *, device: torch.device):
    r"""构造使用同一 config/source cache 的独立 method/session。"""

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
        max_resident_assets=16,
        window_factory=ResidentGeometryAssetWindow,
    )
    return method, session


def main() -> None:
    r"""比较相同 16-assets × 2-q realization 的 teacher、terms 与全部 parameter gradients。"""

    torch.manual_seed(20260830)
    config = compose_evaluation_cfg(config_ref="geometry_ssl_density_material_jacobian_v0_8_0")
    config = replace(
        config,
        method=replace(
            config.method,
            joint_sign_rewrite=replace(config.method.joint_sign_rewrite, probability=0.0),
        ),
        evaluation=replace(
            config.evaluation,
            execution=replace(config.evaluation.execution, model_autocast_dtype="float32"),
        ),
    )  # admission 先隔离 streaming 数学，augmentation 另做 parity
    device = torch.device("cuda:0")
    catalog = build_runtime(config.data).resolve_evaluation()
    full_method, full_session = _build_method(config, catalog, device=device)
    stream_method, stream_session = _build_method(config, catalog, device=device)
    stream_method.load_training_state_dict(full_method.training_state_dict())
    full_schedule = FixedAssetQSchedule(
        16,
        q_per_asset=2,
        assets_per_minibatch=16,
        q_per_asset_per_minibatch=2,
        max_resident_assets=16,
    )
    stream_schedule = FixedAssetQSchedule(
        16,
        q_per_asset=2,
        assets_per_minibatch=16,
        q_per_asset_per_minibatch=2,
        max_resident_assets=16,
    )
    try:
        full_item = full_schedule.next()
        stream_item = stream_schedule.next()
        full_batch = full_session.realize(full_item, schedule=full_schedule, step=0)
        stream_units = tuple(stream_session.realize_units(stream_item, schedule=stream_schedule, step=0))
        for method in (full_method, stream_method):
            for parameter in method.parameters():
                parameter.grad = None
        full_update = full_method.backward_update(
            full_batch,
            forward_step=0,
            microbatch_size=32,
        )
        stream_update = stream_method.backward_update_units(
            iter(stream_units),
            forward_step=0,
            logical_sample_count=32,
            microbatch_size=16,
        )
        gradient_errors = {}
        gradient_relative_l2 = {}
        for full_group, stream_group in zip(
            full_method.optimizer_parameter_groups(),
            stream_method.optimizer_parameter_groups(),
            strict=True,
        ):
            errors = []
            difference_square = 0.0
            reference_square = 0.0
            for full_parameter, stream_parameter in zip(full_group.parameters, stream_group.parameters, strict=True):
                full_grad = full_parameter.grad
                stream_grad = stream_parameter.grad
                if full_grad is None or stream_grad is None:
                    if full_grad is not stream_grad:
                        raise RuntimeError("full/stream gradient None layout differs")
                    continue
                errors.append(float((full_grad - stream_grad).abs().max()))
                difference_square += float((full_grad - stream_grad).double().square().sum())
                reference_square += float(full_grad.double().square().sum())
            gradient_errors[full_group.name] = max(errors, default=0.0)
            gradient_relative_l2[full_group.name] = (difference_square / max(reference_square, 1.0e-30)) ** 0.5
        report = {
            "full_terms": full_update.terms,
            "stream_terms": stream_update.terms,
            "full_denominators": full_update.denominators,
            "stream_denominators": stream_update.denominators,
            "stream_unit_count": len(stream_units),
            "unit_sample_counts": [unit.q.shape[0] for unit in stream_units],
            "gradient_max_abs_error": gradient_errors,
            "gradient_relative_l2_error": gradient_relative_l2,
            "model_autocast_dtype": config.evaluation.execution.model_autocast_dtype,
        }
        print(json.dumps(report, indent=2))
    finally:
        full_session.close()
        stream_session.close()
        full_method.close()
        stream_method.close()


if __name__ == "__main__":
    main()
