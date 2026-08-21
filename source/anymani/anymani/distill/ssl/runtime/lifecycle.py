r"""Schema 4 online procedural supervised pretraining lifecycle.

该模块是最高级训练内核：Data runtime 解析 catalog，Method 封闭产生 batch 与五项 objective，Trainer
拥有 window-major schedule、backward 与 update；Evaluation 只在固定 held-out batches 上读取 method。
生命周期不读取 representation 内部字段，也不解释 owner/query/edge 轴。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from anymani.assets.asset_schema_geometry import SEMANTICS_SCHEMA_VERSION
from anymani.distill.methods.multi_anchor_gaussian_implicit_field.batch import PaddedOnlineGeometryBatch
from anymani.distill.representations.sources.collision_geometry import (
    geometry_identity,
    materialize_owner_geometry_cache,
)
from anymani.distill.representations.sources.kinematics import lower_hand_geometry_semantics
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
from anymani.distill.ssl.runtime.run import PretrainRun
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
    catalog: Any,
    train_sources: tuple[Any, ...],
    validation_sources: dict[str, tuple[Any, ...]],
) -> dict[str, Any]:
    r"""构造 schema 4 expanded physical manifest，保留具名 validation suites。"""

    train_by_id = {source.asset_id: source for source in train_sources}
    train = tuple(
        _manifest_record(
            record.container,
            train_by_id[record.container.asset_id],
            partition="train",
            provenance=record.provenance,
        )
        for record in catalog.dataset.train.records
    )
    validation: dict[str, list[dict[str, Any]]] = {}
    for suite_name, suite_sources in validation_sources.items():
        source_by_id = {source.asset_id: source for source in suite_sources}
        validation[suite_name] = [
            _manifest_record(
                record.container,
                source_by_id[record.container.asset_id],
                partition=f"validation.{suite_name}",
                provenance=record.provenance,
            )
            for record in catalog.dataset.validation[suite_name].records
        ]
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
        "schema_version": "4.0.0",
        "dataset_source_path": str(catalog.dataset.source_path),
        "dataset_source_sha256": catalog.dataset.source_sha256,
        "train": list(train),
        "validation": validation,
        "evaluation": evaluation,
    }
    validate_asset_manifest_isolation(manifest)
    return manifest


def _build_batch(
    schedule_item: Any,
    *,
    method: Any,
    sources: tuple[Any, ...],
    samplers: tuple[Any, ...],
    window: Any,
    seed: int,
    schedule: OnlineMinibatchSchedule,
    mode: str = "train",
) -> PaddedOnlineGeometryBatch:
    r"""把一次 schedule item 交给 method 封闭 realize，trainer 不读 representation 内部字段。"""

    return method.realize_minibatch(
        schedule_item,
        sources=sources,
        samplers=samplers,
        window=window,
        seed=seed,
        schedule=schedule,
        mode=mode,
    )


def _sampling_state(
    schedule: OnlineMinibatchSchedule, samplers: tuple[Any, ...], sources: tuple[Any, ...]
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
    samplers: tuple[Any, ...],
    sources: tuple[Any, ...],
) -> None:
    r"""严格恢复 schedule、asset order 与各自 q cursor。"""

    if tuple(payload.get("asset_ids", ())) != tuple(source.asset_id for source in sources):
        raise ValueError("checkpoint asset axis does not match resolved train catalog")
    raw_schedule = payload.get("schedule")
    if not isinstance(raw_schedule, dict):
        raise ValueError("checkpoint lacks schema 4 online schedule state")
    schedule.load_state_dict(raw_schedule)
    raw_samplers = payload.get("samplers")
    if not isinstance(raw_samplers, (tuple, list)) or len(raw_samplers) != len(samplers):
        raise ValueError("checkpoint Sobol sampler count does not match resolved train catalog")
    for sampler, state in zip(samplers, raw_samplers):
        if not isinstance(state, dict):
            raise ValueError("checkpoint Sobol sampler state must be a mapping")
        sampler.load_state_dict(state)


def _declared_objective_weights(method: Any) -> dict[str, float]:
    r"""读取 method 显式声明的五项权重，不经过自动梯度标定。"""

    if hasattr(method, "declared_objective_weights"):
        return dict(method.declared_objective_weights())
    enabled = method.config.objectives.enabled()
    return {name: float(term.weight) for name, term in enabled.items()}


def _scientific_pretrain_identity(resolved_config: dict[str, Any], *, formula_identity: dict[str, str]) -> dict[str, Any]:
    r"""抽出 pretrain 必须与 calibration 一致的科学身份，排除可事后改写的 objective 权重。"""

    method = resolved_config.get("method")
    trainer = resolved_config.get("trainer")
    if not isinstance(method, dict) or not isinstance(trainer, dict):
        raise ValueError("resolved config must contain method and trainer mappings")
    sampling = trainer.get("sampling")
    if not isinstance(sampling, dict) or not sampling:
        raise ValueError("resolved config lacks trainer sampling semantics")
    if not formula_identity:
        raise ValueError("method formula identity must be non-empty")
    return {
        "formula_identity": dict(formula_identity),
        "state_measure": method.get("state_measure"),
        "representation": method.get("representation"),
        "model": method.get("model"),
        "sampling": sampling,
    }


def _worktree_fingerprint() -> tuple[bool, str]:
    r"""记录 dirty/untracked 指纹；不把大型 diff 写入每个 checkpoint。"""

    import subprocess

    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False, "unknown"
    dirty = bool(status.strip())
    digest = hashlib.sha256(status.encode("utf-8")).hexdigest() if dirty else ""
    return dirty, digest


def _evaluate_validation(method: Any, batches: tuple[PaddedOnlineGeometryBatch, ...]) -> dict[str, float]:
    r"""在固定 validation bank 上按 $(asset,q)$ 等权聚合五项 term。"""

    return method.evaluate(batches)


def _write_calibration_artifact(
    method: Any,
    batches: tuple[PaddedOnlineGeometryBatch, ...],
    output: Path,
    *,
    manifest_hash: str,
    resolved_config: dict[str, Any] | None = None,
) -> str:
    r"""前向预实验：算五项统计，不更新参数，不改权重。"""

    if not batches:
        raise ValueError("objective calibration requires at least one generated train minibatch")
    terms: dict[str, list[float]] = {name: [] for name in _declared_objective_weights(method)}
    method.require_model().eval()
    with torch.enable_grad():
        for index, batch in enumerate(batches):
            step = method.forward_objectives(batch, step=index, mode="train")
            for name, result in step.objectives.items():
                terms.setdefault(name, []).append(float(result.metrics["loss"].detach()))
    formula_identity = dict(method.formula_identity()) if hasattr(method, "formula_identity") else {}
    recorded_config = dict(resolved_config or {})
    payload = {
        "schema_version": "4.0.0",
        "source": "formal_train_forward_preflight",
        "minibatch_count": len(batches),
        "dataset_source_sha256": manifest_hash,
        "declared_objective": _declared_objective_weights(method),
        "formula_identity": formula_identity,
        "method_type": f"{type(method).__module__}.{type(method).__qualname__}",
        "code_revision": PretrainRun.code_revision(),
        "resolved_config": recorded_config,
        "scientific_identity": (
            _scientific_pretrain_identity(recorded_config, formula_identity=formula_identity)
            if recorded_config
            else {}
        ),
        "term_means": {name: float(sum(values) / len(values)) for name, values in terms.items() if values},
        "term_traces": terms,
    }
    _write_yaml(output, payload)
    return hashlib.sha256(output.read_bytes()).hexdigest()


def _require_calibration_identity(
    artifact: Path,
    *,
    method: Any,
    manifest_hash: str,
    resolved_config: dict[str, Any],
) -> str:
    r"""核对 calibration artifact 的数据集、公式身份与采样语义；权重以当前 OBJECTIVES_CFG 为准。"""

    payload = yaml.safe_load(artifact.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("calibration artifact must be a mapping")
    if payload.get("schema_version") != "4.0.0":
        raise ValueError("calibration artifact schema must be 4.0.0")
    if payload.get("dataset_source_sha256") != manifest_hash:
        raise ValueError("calibration artifact dataset hash does not match the formal ssl.yaml")
    expected_formula = dict(method.formula_identity()) if hasattr(method, "formula_identity") else {}
    if not expected_formula:
        raise ValueError("current method lacks objective formula identity")
    recorded_formula = payload.get("formula_identity")
    if not isinstance(recorded_formula, dict) or recorded_formula != expected_formula:
        raise ValueError("calibration artifact objective formula identity does not match current method")
    expected_method_type = f"{type(method).__module__}.{type(method).__qualname__}"
    if payload.get("method_type") != expected_method_type:
        raise ValueError("calibration artifact method type does not match current method")
    recorded_revision = payload.get("code_revision")
    current_revision = PretrainRun.code_revision()
    if not recorded_revision or recorded_revision != current_revision:
        raise ValueError("calibration artifact code revision does not match current HEAD")
    expected_identity = _scientific_pretrain_identity(resolved_config, formula_identity=expected_formula)
    recorded_identity = payload.get("scientific_identity")
    if not isinstance(recorded_identity, dict) or recorded_identity != expected_identity:
        raise ValueError("calibration artifact scientific identity does not match current representation/model/sampling")
    return hashlib.sha256(artifact.read_bytes()).hexdigest()


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
    method.prepare(catalog, device=device, dtype=dtype)
    train_sources = method.train_sources
    validation_sources = method.validation_sources
    manifest = _build_manifest(catalog, train_sources, validation_sources)
    _write_yaml(output_dir / "resolved_config.yaml", resolved_config)
    _write_yaml(output_dir / "asset_dataset.yaml", catalog.dataset.config_dict())
    _write_yaml(output_dir / "asset_manifest.yaml", manifest)
    dirty, fingerprint = _worktree_fingerprint()
    declared_weights = _declared_objective_weights(method)
    calibration_hash = ""

    train_window = None
    validation_windows: dict[str, Any] = {}
    model = method.initialize_model(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=trainer.config.optimizer.learning_rate,
        weight_decay=trainer.config.optimizer.weight_decay,
    )
    train_schedule = OnlineMinibatchSchedule(
        len(train_sources),
        trainer.config.sampling,
        max_resident_assets=trainer.config.max_resident_assets,
    )
    method.initialize_samplers(
        train_seed=trainer.config.sampling.seed,
        validation_seeds={
            suite_name: run.config.seed + evaluation.validation_seed + suite_index * 1_000_003
            for suite_index, suite_name in enumerate(validation_sources)
        },
    )
    train_samplers = method.train_samplers
    validation_schedules = {
        suite_name: OnlineMinibatchSchedule(
            len(suite_sources),
            evaluation.validation_sampling(
                trainer_sampling=trainer.config.sampling,
                run_seed=run.config.seed,
                asset_count=len(suite_sources),
            ),
            max_resident_assets=trainer.config.max_resident_assets,
        )
        for suite_name, suite_sources in validation_sources.items()
        if suite_sources
    }
    validation_samplers = method.validation_samplers
    updates_per_epoch = math.ceil(train_schedule.minibatches_per_epoch / trainer.config.gradient_accumulation_steps)
    required_updates = updates_per_epoch * trainer.config.sampling.epochs
    if trainer.config.run_safety_step_limit < required_updates:
        raise ValueError(
            f"run_safety_step_limit={trainer.config.run_safety_step_limit} cannot cover "
            f"required optimizer updates={required_updates}"
        )
    initial_validation: dict[str, dict[str, float]] | None = None
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
        declared = loaded_metadata.get("declared_objective")
        if not isinstance(checkpoint_resolved, dict) or not isinstance(declared, dict):
            raise ValueError("resume checkpoint lacks resolved config or declared objective evidence")
        require_resume_scientific_config(resolved_config, checkpoint_resolved)
        calibration_hash = str(loaded_metadata.get("calibration_artifact_hash", ""))
        runtime_state = load_geometry_ssl_runtime_state(resume_path, map_location="cpu")
        sampling_state = runtime_state.get("sampling")
        if not isinstance(sampling_state, dict):
            raise ValueError("resume checkpoint lacks schema 4 trainer sampling state")
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
            loader=method.load_device_state,
        )
        if resume_path is None and run.config.phase == "calibrate_objectives":
            from anymani.distill.ssl.runtime.sampling import OnlineSamplingCfg

            calibration_schedule = OnlineMinibatchSchedule(
                len(train_sources),
                OnlineSamplingCfg(
                    epochs=1,
                    q_per_asset_per_epoch=trainer.config.sampling.q_per_asset_per_minibatch,
                    assets_per_minibatch=trainer.config.sampling.assets_per_minibatch,
                    q_per_asset_per_minibatch=trainer.config.sampling.q_per_asset_per_minibatch,
                    shuffle_assets=trainer.config.sampling.shuffle_assets,
                    seed=trainer.config.sampling.seed,
                ),
                max_resident_assets=trainer.config.max_resident_assets,
            )
            calibration_batches: list[PaddedOnlineGeometryBatch] = []
            while not calibration_schedule.complete:
                item = calibration_schedule.next()
                calibration_batches.append(
                    _build_batch(
                        item,
                        method=method,
                        sources=train_sources,
                        samplers=train_samplers,
                        window=train_window,
                        seed=trainer.config.sampling.seed,
                        schedule=calibration_schedule,
                        mode="train",
                    )
                )
            _write_calibration_artifact(
                method,
                tuple(calibration_batches),
                output_dir / "loss_calibration.yaml",
                manifest_hash=str(catalog.dataset.source_sha256),
                resolved_config=resolved_config,
            )
            return output_dir
        if resume_path is None and run.config.phase == "pretrain":
            artifact_path = Path(run.config.calibration_artifact).expanduser() if run.config.calibration_artifact else None
            if artifact_path is None or not artifact_path.is_file():
                raise ValueError("pretrain requires run.calibration_artifact pointing to a schema 4 loss_calibration.yaml")
            calibration_hash = _require_calibration_identity(
                artifact_path,
                method=method,
                manifest_hash=str(catalog.dataset.source_sha256),
                resolved_config=resolved_config,
            )
        elif selection_history:
            if resume_path is None:
                raise RuntimeError("historical best inheritance requires a resume checkpoint path")
            historical_best_step = int(min(selection_history, key=lambda item: float(item["score"]))["step"])
            source_best = resume_path.parent / f"best_step_{historical_best_step:08d}.pt"
            if not source_best.is_file():
                raise ValueError("resume source run lacks immutable historical best checkpoint")
            inherited_best = output_dir / "checkpoints" / source_best.name
            inherited_best.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_best, inherited_best)
            publish_best_checkpoint(output_dir / "checkpoints" / "best.pt", inherited_best)
        validation_batches: dict[str, tuple[PaddedOnlineGeometryBatch, ...]] = {}
        for suite_index, (suite_name, suite_sources) in enumerate(validation_sources.items()):
            if not suite_sources:
                continue
            suite_schedule = validation_schedules[suite_name]
            suite_samplers = validation_samplers[suite_name]
            suite_window = ResidentGeometryAssetWindow(
                suite_sources,
                device=str(device),
                dtype=dtype,
                max_resident_assets=trainer.config.max_resident_assets,
                loader=method.load_validation_device_state,
            )
            validation_windows[suite_name] = suite_window
            suite_batches: list[PaddedOnlineGeometryBatch] = []
            while not suite_schedule.complete:
                item = suite_schedule.next()
                suite_batches.append(
                    _build_batch(
                        item,
                        method=method,
                        sources=suite_sources,
                        samplers=suite_samplers,
                        window=suite_window,
                        seed=run.config.seed + evaluation.validation_seed + suite_index * 1_000_003,
                        schedule=suite_schedule,
                        mode="eval",
                    )
                )
            suite_window.release_all()
            validation_batches[suite_name] = tuple(suite_batches)
        if validation_batches and initial_validation is None:
            initial_metrics = {
                suite_name: _evaluate_validation(method, suite_batches)
                for suite_name, suite_batches in validation_batches.items()
            }
            initial_validation = evaluation.selection_baseline(initial_metrics)

        from anymani.distill.ssl.runtime.validation import (
            compare_training_q_banks,
            stream_training_morphology_q_bank,
        )

        training_q_bank_path = output_dir / "training_morphology_q_bank.yaml"
        if resume_path is None:
            initial_training_q_bank = stream_training_morphology_q_bank(
                method,
                train_sources,
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
            remaining = min(trainer.config.gradient_accumulation_steps, train_schedule.minibatches_remaining_in_epoch)
            update_steps: list[Any] = []
            for _ in range(remaining):
                item = train_schedule.next()
                batch = _build_batch(
                    item,
                    method=method,
                    sources=train_sources,
                    samplers=train_samplers,
                    window=train_window,
                    seed=trainer.config.sampling.seed,
                    schedule=train_schedule,
                    mode="train",
                )
                step_result = method.forward_objectives(batch, step=step + len(update_steps), mode="train")
                update_batches.append(batch)
                update_steps.append(step_result)
            update = method.reduce_update(tuple(update_steps))
            update.loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.config.max_gradient_norm)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"non-finite gradient norm at optimizer step {step + 1}")
            optimizer.step()
            step += 1
            means = update.terms
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
                metrics = {
                    suite_name: _evaluate_validation(method, suite_batches)
                    for suite_name, suite_batches in validation_batches.items()
                }
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
                        declared_objective=declared_weights,
                        calibration_artifact_hash=calibration_hash,
                        worktree_dirty=dirty,
                        worktree_fingerprint=fingerprint,
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
                    declared_objective=declared_weights,
                    calibration_artifact_hash=calibration_hash,
                    worktree_dirty=dirty,
                    worktree_fingerprint=fingerprint,
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
            declared_objective=declared_weights,
            calibration_artifact_hash=calibration_hash,
            worktree_dirty=dirty,
            worktree_fingerprint=fingerprint,
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
            method,
            train_sources,
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
            for suite_index, (suite_name, frozen_batches) in enumerate(validation_batches.items()):
                with torch.no_grad():
                    ablation_evidence = fixed_validation_ablation_evidence(model, frozen_batches)
                raw_ablations = ablation_evidence.get("ablations")
                if not isinstance(raw_ablations, (tuple, list)):
                    raise ValueError("multi-anchor evaluator did not report its ablation names")
                supported_ablations = tuple(str(name) for name in raw_ablations)[1:]
                evaluation.require_ablation_contract(supported_ablations)
                ablation_path = output_dir / f"validation_{suite_name}_ablations.yaml"
                _write_yaml(ablation_path, ablation_evidence)
                write_geometry_ssl_ablation_analysis(
                    ablation_path,
                    output_dir / f"validation_{suite_name}_ablation_analysis.yaml",
                    bootstrap_samples=evaluation.config.bootstrap_replicates,
                    seed=run.config.seed + evaluation.bootstrap_seed + suite_index * 1_000_003,
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
        for validation_window in validation_windows.values():
            validation_window.release_all()


__all__ = ["fit_embodiment_pretrain"]
