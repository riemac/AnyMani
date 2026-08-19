r"""Schema 3 online procedural supervised pretraining lifecycle.

该模块是新的最高级训练内核：Data runtime 解析 catalog，Method runtime 产生共享计算图，Trainer
拥有 schedule/backward/update；Evaluation 只在固定 held-out batches 上读取 method。schema 3
生命周期只依赖五个 role runtime，不读取集中式 experiment 字段。
"""

from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from statistics import median
from typing import Any

import torch
import yaml

from anymani.assets.asset_schema_geometry import SEMANTICS_SCHEMA_VERSION
from anymani.distill.representations.geometry import (
    GeometryRepresentation,
    PaddedOnlineGeometryBatch,
    SobolJointSampler,
    split_online_geometry_sample,
)
from anymani.distill.representations.sources.collision_geometry import (
    geometry_identity,
    materialize_owner_geometry_cache,
)
from anymani.distill.representations.sources.kinematics import lower_hand_geometry_semantics
from anymani.distill.representations.targets.geometry_field import fixed_validation_gaussian_field_config
from anymani.distill.ssl.checkpoint import load_geometry_ssl_checkpoint, load_geometry_ssl_runtime_state
from anymani.distill.ssl.runtime.assets import (
    anchor_realization_record,
    home_surface_realization_record,
    validate_asset_manifest_isolation,
)
from anymani.distill.ssl.runtime.checkpointing import (
    publish_best_checkpoint,
    require_resume_scientific_config,
    restore_validation_selection_state,
)
from anymani.distill.ssl.runtime.sampling import OnlineMinibatchSchedule


def _torch_dtype(name: str) -> torch.dtype:
    r"""把配置字符串映射为当前在线 Warp 路径允许的 dtype。"""

    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported embodiment pretraining dtype={name!r}")


def _plain(value: Any) -> Any:
    r"""把 dataclass/tuple/scalar 递归转成 safe YAML/torch checkpoint 基础类型。"""

    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return _plain(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _write_yaml(path: Path, value: Any) -> None:
    r"""原子写出一个只含基础类型的可审计 YAML artifact。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(_plain(value), sort_keys=False), encoding="utf-8")
    temporary.replace(path)


def _manifest_record(asset: Any, source: Any, *, partition: str, provenance: Any) -> dict[str, Any]:
    r"""写出单 asset 的 content/physical/configuration-domain 与 lineage identity。"""

    semantics = asset.geometry_semantics
    if semantics is None:
        raise ValueError(f"asset {asset.asset_id!r} is missing geometry semantics")
    identity = source.identity
    return {
        "asset_id": asset.asset_id,
        "content_hash": semantics.content_hash,
        "physical_geometry_hash": identity.physical_geometry_hash,
        "configuration_domain_hash": identity.configuration_domain_hash,
        "partition": partition,
        "source_kind": semantics.source_kind,
        "topology_key": semantics.topology_key or "",
        "family": semantics.family,
        "handedness": semantics.handedness,
        "joint_count": len(semantics.active_joint_names),
        "owner_count": len(semantics.owners),
        **anchor_realization_record(source.anchors),
        **home_surface_realization_record(source.home_surface, source.geometry_cache),
        **_plain(provenance),
    }


def _build_manifest(
    catalog: Any, train_sources: tuple[Any, ...], validation_sources: tuple[Any, ...]
) -> dict[str, Any]:
    r"""构造 schema 3 expanded physical manifest，evaluation 只做 identity lowering。"""

    train_by_id = {source.asset_id: source for source in train_sources}
    validation_by_id = {source.asset_id: source for source in validation_sources}
    train = tuple(
        _manifest_record(
            record.container,
            train_by_id[record.container.asset_id],
            partition="train",
            provenance=record.provenance,
        )
        for record in catalog.dataset.train.records
    )
    validation = tuple(
        _manifest_record(
            record.container,
            validation_by_id[record.container.asset_id],
            partition="validation",
            provenance=record.provenance,
        )
        for record in catalog.dataset.validation.records
    )
    evaluation: dict[str, list[dict[str, Any]]] = {}
    for name, partition in catalog.dataset.evaluation.items():
        records: list[dict[str, Any]] = []
        for record in partition.records:
            semantics = record.container.geometry_semantics
            if semantics is None:
                raise ValueError(f"evaluation asset {record.container.asset_id!r} is missing geometry semantics")
            spec = lower_hand_geometry_semantics(semantics, dtype=torch.float64)
            cache = materialize_owner_geometry_cache(record.container, spec)
            identity = geometry_identity(semantics, spec, cache)
            # Evaluation suites are not materialized into train GPU state; identity lowering still closes leakage audit.
            records.append(
                {
                    "asset_id": record.container.asset_id,
                    "content_hash": semantics.content_hash,
                    "physical_geometry_hash": identity.physical_geometry_hash,
                    "configuration_domain_hash": identity.configuration_domain_hash,
                    "partition": name,
                    **_plain(record.provenance),
                }
            )
        evaluation[name] = records
    manifest = {
        "schema_version": "3.0.0",
        "dataset_source_path": str(catalog.dataset.source_path),
        "dataset_source_sha256": catalog.dataset.source_sha256,
        "train": list(train),
        "validation": list(validation),
        "evaluation": evaluation,
    }
    validate_asset_manifest_isolation(manifest)
    return manifest


def _build_batch(
    schedule_item: Any,
    *,
    sources: tuple[Any, ...],
    states_by_id: dict[str, Any],
    samplers: tuple[SobolJointSampler, ...],
    representation: Any,
    window: Any,
    padding: Any,
    seed: int,
    schedule: OnlineMinibatchSchedule,
) -> PaddedOnlineGeometryBatch:
    r"""由 schedule item realization 一次同资产 q block，并合并为 padded model batch。"""

    asset_ids = tuple(sources[index].asset_id for index in schedule_item.asset_indices)
    states = window.ensure(asset_ids)
    states_by_id.update({state.source.asset_id: state for state in states})
    samples = []
    for asset_index in schedule_item.asset_indices:
        state = states_by_id[sources[asset_index].asset_id]
        q_count = schedule_item.q_per_asset
        q = samplers[asset_index].draw(
            q_count, device=state.spec.space_screws.device, dtype=state.spec.space_screws.dtype
        )
        q_start = samplers[asset_index].cursor - q_count
        schedule_index = (
            schedule_item.epoch * schedule.minibatches_per_epoch
            + schedule_item.q_round * schedule.asset_groups_per_round
            + schedule_item.asset_group
        )
        sample = representation.sample(
            state,
            q,
            sampling_seed=seed + schedule_index,
            q_index=torch.arange(q_start, q_start + q_count, device=q.device, dtype=torch.long),
        )
        samples.extend(split_online_geometry_sample(sample))
    from anymani.distill.representations.geometry import pad_online_geometry_samples

    return pad_online_geometry_samples(samples, padding=padding)


def _sampling_state(
    schedule: OnlineMinibatchSchedule, samplers: tuple[SobolJointSampler, ...], sources: tuple[Any, ...]
) -> dict[str, Any]:
    r"""合并 schedule permutation 与每资产 Sobol cursor，作为 optimizer boundary state。"""

    return {
        "schedule": schedule.state_dict(),
        "asset_ids": tuple(source.asset_id for source in sources),
        "samplers": tuple(sampler.state_dict() for sampler in samplers),
    }


def _restore_sampling_state(
    payload: dict[str, Any],
    schedule: OnlineMinibatchSchedule,
    samplers: tuple[SobolJointSampler, ...],
    sources: tuple[Any, ...],
) -> None:
    r"""严格恢复 schedule、asset order 与各自 q cursor。"""

    if tuple(payload.get("asset_ids", ())) != tuple(source.asset_id for source in sources):
        raise ValueError("checkpoint asset axis does not match resolved train catalog")
    raw_schedule = payload.get("schedule")
    if not isinstance(raw_schedule, dict):
        raise ValueError("checkpoint lacks schema 3 online schedule state")
    schedule.load_state_dict(raw_schedule)
    raw_samplers = payload.get("samplers")
    if not isinstance(raw_samplers, (tuple, list)) or len(raw_samplers) != len(samplers):
        raise ValueError("checkpoint Sobol sampler count does not match resolved train catalog")
    for sampler, state in zip(samplers, raw_samplers):
        if not isinstance(state, dict):
            raise ValueError("checkpoint Sobol sampler state must be a mapping")
        sampler.load_state_dict(state)


def _term_denominators(result_blocks: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    r"""按 component 名合并本次 optimizer update 的有效标量分母。"""

    totals: dict[str, torch.Tensor] = {}
    for results in result_blocks:
        for result in results.values():
            for component in result.components:
                value = component.denominator.detach()
                if component.name in totals:
                    totals[component.name] = totals[component.name] + value
                else:
                    totals[component.name] = value
    if not totals or any(float(value) <= 0.0 for value in totals.values()):
        raise ValueError("optimizer update contains no positive objective component denominators")
    return totals


def _loss_for_results(results: dict[str, Any], totals: dict[str, torch.Tensor], method: Any) -> torch.Tensor:
    r"""把 term-owned additive numerators按 update-wide denominator 组成一个 scalar。"""

    loss: torch.Tensor | None = None
    for name, result in results.items():
        weight = float(method.objective_weights[name])
        for component in result.components:
            value = weight * component.numerator / totals[component.name]
            loss = value if loss is None else loss + value
    if loss is None:
        raise ValueError("method returned no objective components")
    return loss


def _mean_terms(result_blocks: list[dict[str, Any]]) -> dict[str, float]:
    r"""记录本 update 各 term 的 numerator/denominator 聚合均值。"""

    values: dict[str, dict[str, tuple[float, float]]] = {}
    for results in result_blocks:
        for name, result in results.items():
            by_component = values.setdefault(name, {})
            for component in result.components:
                numerator, denominator = by_component.get(component.name, (0.0, 0.0))
                by_component[component.name] = (
                    numerator + float(component.numerator.detach()),
                    denominator + float(component.denominator.detach()),
                )
    return {
        name: sum(numerator / denominator for numerator, denominator in by_component.values())
        for name, by_component in values.items()
    }


def _evaluate_validation(method: Any, batches: tuple[PaddedOnlineGeometryBatch, ...]) -> dict[str, float]:
    r"""在固定 validation bank 上聚合六项 term；JVP 需要启用 autograd 但不更新参数。"""

    totals: dict[str, dict[str, tuple[float, float]]] = {}
    method.require_model().eval()
    with torch.enable_grad():
        for index, batch in enumerate(batches):
            results = method.forward_objectives(batch, pair_step=index).objectives
            for name, result in results.items():
                by_component = totals.setdefault(name, {})
                for component in result.components:
                    numerator, denominator = by_component.get(component.name, (0.0, 0.0))
                    by_component[component.name] = (
                        numerator + float(component.numerator.detach()),
                        denominator + float(component.denominator.detach()),
                    )
    return {
        name: sum(numerator / denominator for numerator, denominator in by_component.values())
        for name, by_component in totals.items()
    }


def _calibrate_objective_weights(method: Any, batches: tuple[PaddedOnlineGeometryBatch, ...], output: Path) -> None:
    r"""测量每项 shared-encoder gradient median，并冻结一次 runtime 权重。"""

    if not batches:
        raise ValueError("objective calibration requires at least one generated train minibatch")
    model = method.require_model()
    encoder_parameters = tuple(parameter for parameter in model.encoder.parameters() if parameter.requires_grad)
    measurements: dict[str, list[float]] = {name: [] for name in method.objectives}
    for batch_index, batch in enumerate(batches):
        for name in method.objectives:
            model.zero_grad(set_to_none=True)
            results = method.forward_objectives(batch, pair_step=batch_index).objectives
            term = results[name]
            term_loss = term.components[0].mean
            for component in term.components[1:]:
                term_loss = term_loss + component.mean
            term_loss.backward()
            squared_norm = torch.zeros((), device=next(model.parameters()).device)
            for parameter in encoder_parameters:
                if parameter.grad is not None:
                    squared_norm = squared_norm + parameter.grad.detach().square().sum()
            measurements[name].append(float(torch.sqrt(squared_norm)))
    medians = {name: float(median(values)) for name, values in measurements.items()}
    reference_name = method.config.calibration.reference_term
    reference = medians.get(reference_name, 0.0)
    if reference <= 0.0:
        raise FloatingPointError(f"objective calibration reference {reference_name!r} gradient must be positive")
    weights: dict[str, float] = {}
    for name, value in medians.items():
        declared = float(method.config.objectives[name].weight)
        if declared == 0.0 or not method.config.objectives[name].calibrate:
            weights[name] = declared
        else:
            ratio = method.config.calibration.max_weight if value <= 0.0 else reference / value
            weights[name] = min(max(ratio, method.config.calibration.min_weight), method.config.calibration.max_weight)
    method.set_objective_weights(weights)
    _write_yaml(
        output,
        {
            "source": "generated_train_fixed_calibration_minibatches",
            "minibatch_count": len(batches),
            "gradient_norms": measurements,
            "median_gradient_norms": medians,
            "reference": reference_name,
            "weights": weights,
            "clip": {
                "min": method.config.calibration.min_weight,
                "max": method.config.calibration.max_weight,
            },
        },
    )


def _write_metrics(path: Path, record: dict[str, Any]) -> None:
    r"""追加 JSONL 训练事实，不把日志写入训练状态。"""

    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_plain(record), sort_keys=True) + "\n")


def fit_embodiment_pretrain(
    *,
    trainer: Any,
    data: Any,
    method: Any,
    evaluation: Any,
    run: Any,
    output_dir_override: Path | None,
    resolved_config: dict[str, Any],
) -> Path:
    r"""执行 setup → calibration/resume → train → validation/checkpoint → retained export → teardown。"""

    if run.config.deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    torch.manual_seed(run.config.seed)
    device = torch.device(trainer.config.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"configured CUDA device is unavailable: {device}")
    dtype = _torch_dtype(trainer.config.dtype)
    output_dir = run.prepare_output_dir(output_dir_override)
    catalog = data.resolve()
    train_sources = method.materialize_sources(catalog.train)
    validation_sources = method.materialize_sources(catalog.validation)
    manifest = _build_manifest(catalog, train_sources, validation_sources)
    _write_yaml(output_dir / "resolved_config.yaml", resolved_config)
    _write_yaml(output_dir / "asset_dataset.yaml", catalog.dataset.config_dict())
    _write_yaml(output_dir / "asset_manifest.yaml", manifest)

    representation = method.representation
    validation_representation = GeometryRepresentation(
        replace(
            representation.config,
            field=fixed_validation_gaussian_field_config(representation.config.field),
        )
    )
    train_window = None
    validation_window = None
    model = method.initialize_model(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=trainer.config.optimizer.learning_rate,
        weight_decay=trainer.config.optimizer.weight_decay,
    )
    train_schedule = OnlineMinibatchSchedule(len(train_sources), trainer.config.sampling)
    train_samplers = tuple(
        SobolJointSampler(source.spec_cpu, seed=trainer.config.sampling.seed + index)
        for index, source in enumerate(train_sources)
    )
    validation_schedule = (
        OnlineMinibatchSchedule(
            len(validation_sources),
            evaluation.validation_sampling(
                trainer_sampling=trainer.config.sampling,
                run_seed=run.config.seed,
                asset_count=len(validation_sources),
            ),
        )
        if validation_sources
        else None
    )
    validation_samplers = tuple(
        SobolJointSampler(source.spec_cpu, seed=run.config.seed + evaluation.validation_seed + index)
        for index, source in enumerate(validation_sources)
    )
    updates_per_epoch = math.ceil(train_schedule.minibatches_per_epoch / trainer.config.gradient_accumulation_steps)
    required_updates = updates_per_epoch * trainer.config.sampling.epochs
    if trainer.config.run_safety_step_limit < required_updates:
        raise ValueError(
            f"run_safety_step_limit={trainer.config.run_safety_step_limit} cannot cover "
            f"required optimizer updates={required_updates}"
        )
    initial_validation: dict[str, float] | None = None
    best_score = float("inf")
    selection_history: list[dict[str, Any]] = []
    step = 0
    resume_path: Path | None = None
    if run.config.resume_checkpoint:
        resume_path = Path(run.config.resume_checkpoint).expanduser().resolve()
        step, loaded_metadata = load_geometry_ssl_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            map_location=device,
        )
        if loaded_metadata.get("asset_manifest") != manifest:
            raise ValueError("resume checkpoint asset manifest does not match resolved dataset roles")
        checkpoint_resolved = loaded_metadata.get("resolved_config")
        calibrated_weights = loaded_metadata.get("calibrated_objective")
        if not isinstance(checkpoint_resolved, dict) or not isinstance(calibrated_weights, dict):
            raise ValueError("resume checkpoint lacks resolved config or calibrated objective evidence")
        require_resume_scientific_config(resolved_config, checkpoint_resolved)
        method.set_objective_weights({str(name): float(value) for name, value in calibrated_weights.items()})
        runtime_state = load_geometry_ssl_runtime_state(resume_path, map_location="cpu")
        sampling_state = runtime_state.get("sampling")
        if not isinstance(sampling_state, dict):
            raise ValueError("resume checkpoint lacks schema 3 trainer sampling state")
        _restore_sampling_state(sampling_state, train_schedule, train_samplers, train_sources)
        initial_validation, _initial_strata, best_score, selection_history = restore_validation_selection_state(
            runtime_state
        )
        torch_rng_state = runtime_state.get("torch_rng_state")
        cuda_rng_state = runtime_state.get("cuda_rng_state_all")
        if not isinstance(torch_rng_state, torch.Tensor):
            raise ValueError("resume checkpoint lacks torch RNG state")
        if not isinstance(cuda_rng_state, list) or not all(isinstance(item, torch.Tensor) for item in cuda_rng_state):
            raise ValueError("resume checkpoint lacks CUDA RNG states")
        torch.set_rng_state(torch_rng_state.cpu())
        torch.cuda.set_rng_state_all(cuda_rng_state)
    try:
        from anymani.distill.ssl.runtime.scheduler import ResidentGeometryAssetWindow

        train_window = ResidentGeometryAssetWindow(
            train_sources,
            device=str(device),
            dtype=dtype,
            max_resident_assets=trainer.config.max_resident_assets,
            loader=representation.to_device,
        )
        if resume_path is None:
            if method.config.calibration.minibatches > train_schedule.minibatches_remaining_in_epoch:
                raise ValueError("objective calibration minibatches exceed one train coverage epoch")
            initial_sampling_state = _sampling_state(train_schedule, train_samplers, train_sources)
            calibration_batches: list[PaddedOnlineGeometryBatch] = []
            calibration_state_cache: dict[str, Any] = {}
            for _ in range(method.config.calibration.minibatches):
                item = train_schedule.next()
                calibration_batches.append(
                    _build_batch(
                        item,
                        sources=train_sources,
                        states_by_id=calibration_state_cache,
                        samplers=train_samplers,
                        representation=representation,
                        window=train_window,
                        padding=representation.config.layout,
                        seed=trainer.config.sampling.seed,
                        schedule=train_schedule,
                    )
                )
            _calibrate_objective_weights(method, tuple(calibration_batches), output_dir / "loss_calibration.yaml")
            _restore_sampling_state(initial_sampling_state, train_schedule, train_samplers, train_sources)
            model.zero_grad(set_to_none=True)
            train_window.release_all()
        elif selection_history:
            historical_best_step = int(min(selection_history, key=lambda item: float(item["score"]))["step"])
            source_best = resume_path.parent / f"best_step_{historical_best_step:08d}.pt"
            if not source_best.is_file():
                raise ValueError("resume source run lacks immutable historical best checkpoint")
            inherited_best = output_dir / "checkpoints" / source_best.name
            inherited_best.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_best, inherited_best)
            publish_best_checkpoint(output_dir / "checkpoints" / "best.pt", inherited_best)
        if validation_sources:
            validation_window = ResidentGeometryAssetWindow(
                validation_sources,
                device=str(device),
                dtype=dtype,
                max_resident_assets=trainer.config.max_resident_assets,
                loader=validation_representation.to_device,
            )
        validation_batches: list[PaddedOnlineGeometryBatch] = []
        if validation_schedule is not None and validation_window is not None:
            validation_state_cache: dict[str, Any] = {}
            while not validation_schedule.complete:
                item = validation_schedule.next()
                validation_batches.append(
                    _build_batch(
                        item,
                        sources=validation_sources,
                        states_by_id=validation_state_cache,
                        samplers=validation_samplers,
                        representation=validation_representation,
                        window=validation_window,
                        padding=representation.config.layout,
                        seed=run.config.seed + evaluation.validation_seed,
                        schedule=validation_schedule,
                    )
                )
            validation_window.release_all()
            if initial_validation is None:
                initial_metrics = _evaluate_validation(method, tuple(validation_batches))
                initial_validation = evaluation.selection_baseline(initial_metrics)

        from anymani.distill.ssl.runtime.validation import (
            compare_training_q_banks,
            stream_training_morphology_q_bank,
        )

        training_q_bank_path = output_dir / "training_morphology_q_bank.yaml"
        if resume_path is None:
            initial_training_q_bank = stream_training_morphology_q_bank(
                model,
                train_sources,
                representation_config=representation.config,
                seed=run.config.seed + evaluation.q_bank_seed,
                q_per_asset=evaluation.config.q_per_asset,
                assets_per_minibatch=trainer.config.sampling.assets_per_minibatch,
                q_per_asset_per_minibatch=trainer.config.sampling.q_per_asset_per_minibatch,
                max_resident_assets=trainer.config.max_resident_assets,
                device=device,
                dtype=dtype,
                phase="initial",
            )
        else:
            source_q_bank_path = resume_path.parent.parent / "training_morphology_q_bank.yaml"
            if not source_q_bank_path.is_file():
                raise ValueError("resume source run lacks training_morphology_q_bank.yaml")
            source_q_bank = yaml.safe_load(source_q_bank_path.read_text(encoding="utf-8"))
            if not isinstance(source_q_bank, dict) or not isinstance(source_q_bank.get("initial"), dict):
                raise ValueError("resume source training morphology q bank lacks initial evidence")
            initial_training_q_bank = dict(source_q_bank["initial"])
        _write_yaml(
            training_q_bank_path,
            {"initial": initial_training_q_bank, "final": None, "comparison": None},
        )

        while not train_schedule.complete:
            if step >= trainer.config.run_safety_step_limit:
                raise RuntimeError("run_safety_step_limit exhausted before configured coverage completed")
            model.train()
            optimizer.zero_grad(set_to_none=True)
            update_batches: list[PaddedOnlineGeometryBatch] = []
            update_results: list[dict[str, Any]] = []
            remaining = min(trainer.config.gradient_accumulation_steps, train_schedule.minibatches_remaining_in_epoch)
            train_state_cache: dict[str, Any] = {}
            for _ in range(remaining):
                item = train_schedule.next()
                batch = _build_batch(
                    item,
                    sources=train_sources,
                    states_by_id=train_state_cache,
                    samplers=train_samplers,
                    representation=representation,
                    window=train_window,
                    padding=representation.config.layout,
                    seed=trainer.config.sampling.seed,
                    schedule=train_schedule,
                )
                step_result = method.forward_objectives(batch, pair_step=step + len(update_batches))
                update_batches.append(batch)
                update_results.append(step_result.objectives)
            denominators = _term_denominators(update_results)
            for results in update_results:
                _loss_for_results(results, denominators, method).backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.config.max_gradient_norm)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"non-finite gradient norm at optimizer step {step + 1}")
            optimizer.step()
            step += 1
            means = _mean_terms(update_results)
            _write_metrics(
                output_dir / "metrics.jsonl",
                {
                    "step": step,
                    "split": "train",
                    "epoch": train_schedule.epoch,
                    "minibatches": len(update_batches),
                    "asset_ids": [asset_id for batch in update_batches for asset_id in batch.asset_ids],
                    "terms": means,
                    "gradient_norm": float(gradient_norm.detach()),
                },
            )
            if validation_batches and (
                step % evaluation.config.every_optimizer_updates == 0 or train_schedule.complete
            ):
                metrics = _evaluate_validation(method, tuple(validation_batches))
                if initial_validation is None:
                    raise RuntimeError("validation baseline was not initialized")
                score = evaluation.normalized_score(metrics, initial_validation)
                selection_history.append({"step": step, "score": score, "metrics": metrics})
                if score < best_score:
                    best_score = score
                    metadata = run.checkpoint_metadata(
                        geometry_semantics_schema=SEMANTICS_SCHEMA_VERSION,
                        asset_manifest=manifest,
                        resolved_config=resolved_config,
                        calibrated_objective=method.objective_weights,
                    )
                    run.save_full_checkpoint(
                        output_dir / "checkpoints" / f"best_step_{step:08d}.pt",
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        metadata=metadata,
                        runtime_state={
                            "sampling": _sampling_state(train_schedule, train_samplers, train_sources),
                            "torch_rng_state": torch.get_rng_state(),
                            "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
                            "selection_history": selection_history,
                            "initial_validation_metrics": initial_validation,
                            "initial_validation_strata": None,
                            "best_validation_score": None if best_score == float("inf") else best_score,
                        },
                    )
                    publish_best_checkpoint(
                        output_dir / "checkpoints" / "best.pt",
                        output_dir / "checkpoints" / f"best_step_{step:08d}.pt",
                    )

            if step % trainer.config.checkpoint_every_updates == 0 or train_schedule.complete:
                metadata = run.checkpoint_metadata(
                    geometry_semantics_schema=SEMANTICS_SCHEMA_VERSION,
                    asset_manifest=manifest,
                    resolved_config=resolved_config,
                    calibrated_objective=method.objective_weights,
                )
                run.save_full_checkpoint(
                    output_dir / "checkpoints" / f"step_{step:08d}.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    metadata=metadata,
                    runtime_state={
                        "sampling": _sampling_state(train_schedule, train_samplers, train_sources),
                        "torch_rng_state": torch.get_rng_state(),
                        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
                        "selection_history": selection_history,
                        "initial_validation_metrics": initial_validation,
                        "initial_validation_strata": None,
                        "best_validation_score": None if best_score == float("inf") else best_score,
                    },
                )

        metadata = run.checkpoint_metadata(
            geometry_semantics_schema=SEMANTICS_SCHEMA_VERSION,
            asset_manifest=manifest,
            resolved_config=resolved_config,
            calibrated_objective=method.objective_weights,
        )
        run.save_full_checkpoint(
            output_dir / "checkpoints" / "last.pt",
            model=model,
            optimizer=optimizer,
            step=step,
            metadata=metadata,
            runtime_state={
                "sampling": _sampling_state(train_schedule, train_samplers, train_sources),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
                "selection_history": selection_history,
                "initial_validation_metrics": initial_validation,
                "initial_validation_strata": None,
                "best_validation_score": None if best_score == float("inf") else best_score,
            },
        )
        final_training_q_bank = stream_training_morphology_q_bank(
            model,
            train_sources,
            representation_config=representation.config,
            seed=run.config.seed + evaluation.q_bank_seed,
            q_per_asset=evaluation.config.q_per_asset,
            assets_per_minibatch=trainer.config.sampling.assets_per_minibatch,
            q_per_asset_per_minibatch=trainer.config.sampling.q_per_asset_per_minibatch,
            max_resident_assets=trainer.config.max_resident_assets,
            device=device,
            dtype=dtype,
            phase="final",
        )
        _write_yaml(
            training_q_bank_path,
            {
                "initial": initial_training_q_bank,
                "final": final_training_q_bank,
                "comparison": compare_training_q_banks(initial_training_q_bank, final_training_q_bank),
            },
        )
        best_source = (
            output_dir
            / "checkpoints"
            / (
                f"best_step_{min(selection_history, key=lambda item: item['score'])['step']:08d}.pt"
                if selection_history
                else "last.pt"
            )
        )
        if selection_history:
            # retained artifact 必须来自 validation-best，而不是最后一个 optimizer state。
            load_geometry_ssl_checkpoint(best_source, model=model, map_location=device)
        if validation_batches:
            from anymani.distill.diagnostics.analysis.geometry_ssl import write_geometry_ssl_ablation_analysis
            from anymani.distill.ssl.runtime.validation import fixed_validation_ablation_evidence

            model.eval()
            with torch.no_grad():
                ablation_evidence = fixed_validation_ablation_evidence(model, tuple(validation_batches))
            raw_ablations = ablation_evidence.get("ablations")
            if not isinstance(raw_ablations, (tuple, list)):
                raise ValueError("multi-anchor evaluator did not report its ablation names")
            supported_ablations = tuple(str(name) for name in raw_ablations)[1:]
            evaluation.require_ablation_contract(supported_ablations)
            ablation_path = output_dir / "validation_ablations.yaml"
            _write_yaml(ablation_path, ablation_evidence)
            write_geometry_ssl_ablation_analysis(
                ablation_path,
                output_dir / "validation_ablation_analysis.yaml",
                bootstrap_samples=evaluation.config.bootstrap_replicates,
                seed=run.config.seed + evaluation.bootstrap_seed,
            )
        run.save_retained_artifact(
            output_dir / "retained_artifact.pt",
            model=model,
            feature_spec=method.feature_spec(),
            metadata=metadata,
            source_checkpoint=best_source,
        )
        _write_yaml(
            output_dir / "checkpoint_selection.yaml",
            {
                "selection_metrics": evaluation.config.selection_metrics,
                "initial_validation": initial_validation,
                "history": selection_history,
                "best_checkpoint": str(best_source.relative_to(output_dir)),
                "retained_artifact": "retained_artifact.pt",
            },
        )
        return output_dir
    finally:
        if train_window is not None:
            train_window.release_all()
        if validation_window is not None:
            validation_window.release_all()


__all__ = ["fit_embodiment_pretrain"]
