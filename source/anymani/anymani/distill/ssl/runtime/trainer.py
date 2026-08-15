r"""Task-free multi-anchor Geometry SSL 的完整运行生命周期。

本模块是训练 façade 的执行内核，只编排已经分离的 runtime 组件：

```text
runtime.assets        HandBank -> CPU runtime -> physical split/manifest
runtime.objective     physical q forward + paired rewrite + accumulation reduction
runtime.validation    fixed morphology bank + ablation + independent-q replay
runtime.checkpointing resume contract + selection lineage + runtime payload
```

命令行解析属于 ``ssl.pretrain``；物理 metric 公式属于 ``objectives`` 与
``diagnostics.evaluation``。本文件不解析 URDF/sidecar，也不重新定义 loss 或 validation metric。
"""

from __future__ import annotations

import os  # cuBLAS deterministic workspace 必须在首次 CUDA 运算前声明
import shutil  # resume 时继承源 run 的 immutable historical best
import subprocess  # Git revision 只读查询
from dataclasses import asdict  # manifest metadata 与 calibrated evidence 冻结
from datetime import UTC, datetime  # run directory 使用 UTC 绝对时间
from importlib.metadata import PackageNotFoundError, version  # installed/editable 包版本证据
from pathlib import Path  # run/checkpoint/evidence 路径
from time import perf_counter  # optimizer-step wall-clock telemetry

import torch  # 模型、optimizer、CUDA 与 autograd
import yaml  # selection/q-bank/ablation evidence

from anymani.assets.asset_schema_geometry import SEMANTICS_SCHEMA_VERSION
from anymani.distill.diagnostics.analysis.geometry_ssl import write_geometry_ssl_ablation_analysis
from anymani.distill.diagnostics.recording.geometry_ssl import GeometrySSLRunLogger
from anymani.distill.models.geometry_ssl import GeometrySSLForward, GeometrySSLModel
from anymani.distill.objectives.representations.field_reconstruction import (
    GeometryFieldObjective,
    GeometryFieldObjectiveCfg,
    GeometryFieldObjectiveTerms,
)
from anymani.distill.representations.geometry import (
    GeometryRepresentation,
    GeometryRepresentationCfg,
    PaddedOnlineGeometryBatch,
)
from anymani.distill.representations.targets.geometry_field import fixed_validation_gaussian_field_config
from anymani.distill.ssl.calibration import calibrate_geometry_ssl_weights
from anymani.distill.ssl.checkpoint import (
    GeometrySSLCheckpointMetadata,
    load_geometry_ssl_checkpoint,
    load_geometry_ssl_runtime_state,
    save_geometry_ssl_checkpoint,
)
from anymani.distill.ssl.config import GeometrySSLExperimentCfg, resolved_config_dict, write_resolved_experiment_files
from anymani.distill.ssl.runtime import (
    GeometrySSLRuntimeCfg,
    ResidentGeometryAssetWindow,
    WindowedOnlineGeometryBatcher,
    runtime_state_from_dict,
)
from anymani.distill.ssl.runtime.assets import (
    build_manifest,
    materialize_identity_only,
    resolve_assets,
    resolve_generated_runtime_splits,
)
from anymani.distill.ssl.runtime.checkpointing import (
    best_step_from_selection_history,
    checkpoint_runtime_payload,
    publish_best_checkpoint,
    require_resume_scientific_config,
    restore_validation_selection_state,
)
from anymani.distill.ssl.runtime.objective import (
    accumulated_objective,
    forward_objective,
    objective_denominators_from_batch,
)
from anymani.distill.ssl.runtime.validation import (
    compare_training_q_banks,
    fixed_validation_ablation_evidence,
    normalized_validation_score,
    stratified_metric_scores,
    stream_training_morphology_q_bank,
    validation_stratified_evidence,
)


def _torch_dtype(name: str) -> torch.dtype:
    r"""把 resolved 字符串限制为显式训练 dtype，不隐式启用 AMP。"""

    if name == "float32":
        return torch.float32  # 正式 Warp/CUDA 主路径
    if name == "float64":
        return torch.float64  # 纯 tensor reference；Warp config 会在更早层拒绝
    raise ValueError(f"unsupported geometry SSL dtype={name!r}")


def _code_revision() -> str:
    r"""尽力记录当前 Git revision；非 Git 安装返回明确 ``unknown``。"""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )  # 不使用 shell，不修改工作树
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _package_version() -> str:
    r"""读取 installed AnyMani distribution version；editable 未登记时显式标注。"""

    try:
        return version("anymani")
    except PackageNotFoundError:
        return "editable-unknown"


def _run_geometry_ssl_lifecycle(
    config: GeometrySSLExperimentCfg,
    *,
    output_dir_override: Path | None = None,
) -> Path:
    r"""执行 asset split、calibration、online train、fixed validation 与 checkpoint 生命周期。

    生命周期固定为：physical split/manifest → resident runtime → held-out teacher bank → model/optimizer →
    train-only calibration 或 strict resume → initialization evidence → online optimizer loop → final ablations、
    independent-q replay、best/last checkpoint 与 cache release。official assets 在 manifest 后不进入任何
    teacher、model 或 optimizer 路径。

    Args:
        config (GeometrySSLExperimentCfg): 完整、冻结、已逐层验证的实验配置。
        output_dir_override (Path | None): contract/smoke 的显式隔离目录；正式运行使用配置中的
            ``output_dir/experiment_name/UTC timestamp``。

    Returns:
        Path: 本次 run 的 artifact 根目录。

    Raises:
        ValueError: 资产、split、配置、resume 或有效监督合同不闭合。
        RuntimeError: CUDA/Warp、epoch budget 或 evidence replay 失败。
        FloatingPointError: 梯度或 validation normalization 非有限。
    """

    # 运行前先冻结可复现性与设备合同；online Warp teacher 不提供 CPU fallback。
    if not config.assets.family_paths and not config.assets.train_paths:
        raise ValueError("geometry SSL requires family_paths or at least one generated train asset path")
    if config.protocol.reproducibility.deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)  # 无确定 CUDA kernel 时 fail-closed
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
    torch.manual_seed(config.protocol.reproducibility.seed)  # model 初始化与 PyTorch 路径复现锚点
    device = torch.device(config.trainer.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"configured CUDA device is unavailable: {device}")
    dtype = _torch_dtype(config.trainer.dtype)

    # 解析 generated physical split 与隔离 official identity，先写 manifest 再初始化 GPU/runtime。
    train_runtime, validation_runtime, grouped_split = resolve_generated_runtime_splits(config)
    official_assets = resolve_assets(config.assets.official_evaluation_paths, source_kind="official")
    official_runtime = materialize_identity_only(official_assets)
    manifest = build_manifest(
        train_runtime,
        validation_runtime,
        official_runtime,
        grouped_split=grouped_split,
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_dir_override or Path(config.run.output_dir) / config.run.experiment_name / timestamp
    write_resolved_experiment_files(output_dir, config=config, manifest=manifest)

    # 训练 runtime 只持有有界 resident window；epoch 由每资产 Sobol q coverage 定义。
    representation = GeometryRepresentation(
        GeometryRepresentationCfg(
            source=config.representation.source,
            field=config.representation.field,
            query=config.representation.query,
            target=config.representation.target,
            layout=config.representation.layout,
        )
    )  # source/query/target/layout 的唯一 runtime consumer
    runtime_config = GeometrySSLRuntimeCfg(
        max_resident_assets=config.trainer.max_resident_assets,
        assets_per_microbatch=config.trainer.assets_per_microbatch,
        q_per_asset_per_microbatch=config.protocol.coverage.q_per_asset_per_realization,
        q_per_asset_per_epoch=config.protocol.coverage.q_per_asset_per_epoch,
        epochs=config.protocol.coverage.epochs,
    )
    train_window = ResidentGeometryAssetWindow(
        train_runtime,
        device=str(device),
        dtype=dtype,
        max_resident_assets=runtime_config.max_resident_assets,
        loader=representation.to_device,
    )
    train_batcher = WindowedOnlineGeometryBatcher(
        train_runtime,
        train_window,
        seed=config.protocol.reproducibility.seed,
        runtime_config=runtime_config,
        field_config=config.representation.field,
        query_config=config.representation.query,
        target_config=config.representation.target,
        padding=config.representation.layout,
    )
    required_steps = (
        train_batcher.blocks_per_epoch * runtime_config.epochs + config.trainer.gradient_accumulation_steps - 1
    ) // config.trainer.gradient_accumulation_steps
    if config.protocol.run_safety_step_limit < required_steps:
        raise ValueError(
            f"run_safety_step_limit={config.protocol.run_safety_step_limit} cannot cover configured epochs; "
            f"required at least {required_steps} optimizer steps for {len(train_runtime)} assets"
        )
    if train_batcher.blocks_per_epoch % config.trainer.gradient_accumulation_steps:
        raise ValueError("blocks_per_epoch must be divisible by gradient_accumulation_steps")

    # Held-out morphology bank 在启动时完整 materialize teacher tensors，随后立即释放 validation BVHs。
    validation_batches: tuple[PaddedOnlineGeometryBatch, ...] = ()
    validation_window: ResidentGeometryAssetWindow | None = None
    if validation_runtime:
        validation_runtime_config = GeometrySSLRuntimeCfg(
            max_resident_assets=runtime_config.max_resident_assets,
            assets_per_microbatch=min(runtime_config.assets_per_microbatch, len(validation_runtime)),
            q_per_asset_per_microbatch=runtime_config.q_per_asset_per_microbatch,
            q_per_asset_per_epoch=config.protocol.validation.q_per_asset,
            epochs=1,
        )
        validation_window = ResidentGeometryAssetWindow(
            validation_runtime,
            device=str(device),
            dtype=dtype,
            max_resident_assets=validation_runtime_config.max_resident_assets,
            loader=representation.to_device,
        )
        validation_batcher = WindowedOnlineGeometryBatcher(
            validation_runtime,
            validation_window,
            seed=config.protocol.reproducibility.seed + 1_000_003,
            runtime_config=validation_runtime_config,
            field_config=fixed_validation_gaussian_field_config(config.representation.field),
            query_config=config.representation.query,
            target_config=config.representation.target,
            padding=config.representation.layout,
        )
        try:
            validation_batches = validation_batcher.sample_epoch()
        finally:
            validation_window.release_all()  # teacher 物化失败也必须释放已上传的 validation BVH lease

    # 模型、optimizer 与 resume/calibration 分支在同一 resolved config 下构造。
    model = GeometrySSLModel(config.model).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.trainer.learning_rate,
        weight_decay=config.trainer.weight_decay,
    )
    start_step = 0
    checkpoint_path: Path | None = None
    initial_validation_metrics: dict[str, float] | None = None
    initial_validation_strata: dict[str, object] | None = None
    best_validation_score = float("inf")
    selection_history: list[dict[str, object]] = []
    if config.run.resume_checkpoint:
        checkpoint_path = Path(config.run.resume_checkpoint).expanduser().resolve()
        start_step, loaded_metadata = load_geometry_ssl_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            map_location=device,
        )
        runtime_payload = load_geometry_ssl_runtime_state(checkpoint_path, map_location="cpu")
        train_batcher.load_state_dict(runtime_state_from_dict(runtime_payload))
        resolved_checkpoint = loaded_metadata.get("resolved_config")
        calibrated_objective = loaded_metadata.get("calibrated_objective")
        if not isinstance(resolved_checkpoint, dict) or not isinstance(calibrated_objective, dict):
            raise ValueError("resume checkpoint lacks resolved config or calibrated objective evidence")
        if loaded_metadata.get("asset_manifest") != asdict(manifest):
            raise ValueError("resume asset manifest does not match current resolved physical split")
        require_resume_scientific_config(config, resolved_checkpoint)
        calibrated_weights = GeometryFieldObjectiveCfg(**calibrated_objective)
        (
            initial_validation_metrics,
            initial_validation_strata,
            best_validation_score,
            selection_history,
        ) = restore_validation_selection_state(runtime_payload)
        torch_rng_state = runtime_payload.get("torch_rng_state")
        if not isinstance(torch_rng_state, torch.Tensor):
            raise ValueError("resume checkpoint lacks torch RNG state")
        torch.set_rng_state(torch_rng_state.cpu())
        cuda_rng_state = runtime_payload.get("cuda_rng_state_all")
        if torch.cuda.is_available():
            if not isinstance(cuda_rng_state, list) or not all(isinstance(item, torch.Tensor) for item in cuda_rng_state):
                raise ValueError("resume checkpoint lacks CUDA RNG states")
            torch.cuda.set_rng_state_all(cuda_rng_state)
    else:
        calibration_state = train_batcher.state_dict()  # calibration 不消费正式 q coverage
        try:
            calibration_batches = tuple(
                train_batcher.sample() for _ in range(config.protocol.calibration.batches)
            )
            calibrated_weights = calibrate_geometry_ssl_weights(
                model,
                GeometryFieldObjective,
                calibration_batches,
                lambda calibration_model, calibration_objective, calibration_batch: forward_objective(
                    calibration_model,
                    calibration_objective,
                    calibration_batch,
                    pair_step=0,
                )[1],
                output_path=output_dir / "loss_calibration.yaml",
                min_weight=config.protocol.calibration.min_weight,
                max_weight=config.protocol.calibration.max_weight,
            )
            train_batcher.load_state_dict(calibration_state)
            del calibration_batches
            model.zero_grad(set_to_none=True)
        finally:
            train_window.release_all()  # calibration 成败都不把临时 resident BVH 带到正式训练生命周期

    # Declared objective 保持不变；校准结果作为 runtime evidence 独立进入 artifact/checkpoint。
    objective = GeometryFieldObjective(calibrated_weights)
    metadata = GeometrySSLCheckpointMetadata(
        code_revision=_code_revision(),
        package_version=_package_version(),
        geometry_semantics_schema=SEMANTICS_SCHEMA_VERSION,
        asset_manifest=asdict(manifest),
        resolved_config=resolved_config_dict(config),
        calibrated_objective=asdict(calibrated_weights),
    )
    logger = GeometrySSLRunLogger(output_dir)
    for event in validation_window.drain_telemetry_events() if validation_window is not None else ():
        logger.log_runtime_event({**event, "phase": "fixed_held_out_morphology_bank"})
    for event in train_window.drain_telemetry_events():
        logger.log_runtime_event(
            {**event, "phase": "resume" if config.run.resume_checkpoint else "loss_calibration"}
        )

    # Training-morphology independent-q bank 的 initialization evidence 在中断前即可审计。
    training_q_bank_path = output_dir / "training_morphology_q_bank.yaml"
    if checkpoint_path is None:
        initial_training_q_bank = stream_training_morphology_q_bank(
            model,
            train_runtime,
            config=config,
            device=device,
            dtype=dtype,
            logger=logger,
            phase="initial",
        )
    else:
        source_q_bank_path = checkpoint_path.parent.parent / "training_morphology_q_bank.yaml"
        if not source_q_bank_path.is_file():
            raise ValueError("resume source run lacks training_morphology_q_bank.yaml initialization evidence")
        source_q_bank = yaml.safe_load(source_q_bank_path.read_text(encoding="utf-8"))
        if not isinstance(source_q_bank, dict) or not isinstance(source_q_bank.get("initial"), dict):
            raise ValueError("resume source training morphology q bank lacks initial evidence")
        initial_training_q_bank = source_q_bank["initial"]
        if (
            initial_training_q_bank.get("seed") != config.protocol.reproducibility.seed + 3_000_003
            or initial_training_q_bank.get("asset_count") != len(train_runtime)
            or initial_training_q_bank.get("q_per_asset") != config.protocol.validation.q_per_asset
        ):
            raise ValueError("resume source training morphology q bank does not match resolved experiment")
    training_q_bank_path.write_text(
        yaml.safe_dump({"initial": initial_training_q_bank, "final": None, "comparison": None}, sort_keys=False),
        encoding="utf-8",
    )

    # Held-out morphology initialization strata 是所有后续 score 的固定 normalization baseline。
    last_batch: PaddedOnlineGeometryBatch | None = None
    last_prediction: GeometrySSLForward | None = None
    if validation_batches and initial_validation_metrics is None:
        model.eval()
        initial_predictions = tuple(
            forward_objective(model, objective, validation_batch, pair_step=index)[0]
            for index, validation_batch in enumerate(validation_batches)
        )
        initial_validation_strata = validation_stratified_evidence(initial_predictions, validation_batches)
        initial_validation_metrics = stratified_metric_scores(initial_validation_strata)
        del initial_predictions
        model.zero_grad(set_to_none=True)
    if checkpoint_path is not None and selection_history:
        best_step = best_step_from_selection_history(selection_history)
        if best_step is None:
            raise RuntimeError("non-empty selection history did not resolve a best step")
        source_best_path = checkpoint_path.parent / f"best_step_{best_step:08d}.pt"
        if not source_best_path.is_file():
            raise ValueError("resume source run lacks the immutable historical best checkpoint")
        inherited_immutable_path = output_dir / "checkpoints" / source_best_path.name
        inherited_immutable_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_best_path, inherited_immutable_path)
        publish_best_checkpoint(output_dir / "checkpoints" / "best.pt", inherited_immutable_path)

    try:
        step = start_step
        while train_batcher.epoch < runtime_config.epochs:
            step += 1
            if step > config.protocol.run_safety_step_limit:
                raise RuntimeError("run_safety_step_limit exhausted before configured coverage epochs completed")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            step_started = perf_counter()
            q_sample_count = 0
            model.train()
            optimizer.zero_grad(set_to_none=True)

            # 先取得本 optimizer step 的全部 masks，再按 global denominator 逐 microbatch backward。
            accumulated_batches = tuple(
                train_batcher.sample() for _ in range(config.trainer.gradient_accumulation_steps)
            )
            denominator_components = tuple(
                objective_denominators_from_batch(batch, model) for batch in accumulated_batches
            )
            denominator_totals = tuple(
                torch.tensor(
                    sum(field_denominators[index] for field_denominators, _paired in denominator_components),
                    device=device,
                    dtype=dtype,
                )
                for index in range(5)
            )
            paired_denominator_totals = (
                torch.tensor(
                    sum(paired[0] for _field, paired in denominator_components),
                    device=device,
                    dtype=dtype,
                ),
                torch.tensor(
                    sum(paired[1] for _field, paired in denominator_components),
                    device=device,
                    dtype=dtype,
                ),
            )
            for accumulation_index, batch in enumerate(accumulated_batches):
                prediction, terms = forward_objective(
                    model,
                    objective,
                    batch,
                    pair_step=(step - 1) * config.trainer.gradient_accumulation_steps + accumulation_index,
                )
                accumulated_objective(
                    terms,
                    denominator_totals,
                    paired_denominator_totals,
                    config.objective,
                ).backward()
                q_sample_count += len(batch.asset_ids)
                last_batch, last_prediction = batch, prediction

            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.trainer.max_gradient_norm)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"non-finite gradient norm at step={step}: {float(gradient_norm)}")
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_seconds = perf_counter() - step_started
            logger.log_runtime_event(
                {
                    "event": "optimizer_step",
                    "phase": "training",
                    "step": step,
                    "epoch": train_batcher.epoch,
                    "block_index": train_batcher.block_index,
                    "q_sample_count": q_sample_count,
                    "step_seconds": step_seconds,
                    "q_samples_per_second": q_sample_count / step_seconds,
                    "cuda_peak_allocated_bytes": (
                        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
                    ),
                    "cuda_peak_reserved_bytes": (
                        int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
                    ),
                    "cuda_end_allocated_bytes": (
                        int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else None
                    ),
                    "cuda_end_reserved_bytes": (
                        int(torch.cuda.memory_reserved(device)) if device.type == "cuda" else None
                    ),
                }
            )
            for event in train_window.drain_telemetry_events():
                logger.log_runtime_event({**event, "phase": "training", "step": step})

            if step % config.trainer.log_every_updates == 0 or step == 1:
                logger.log_terms(
                    step=step,
                    split="train",
                    terms=terms,
                    asset_ids=batch.asset_ids,
                    gradient_norm=float(gradient_norm),
                    batch=batch,
                )

            # Held-out morphology 只使用启动时固定 bank；score 由 morphology/bin/axis/metric 四级等权聚合。
            if validation_batches and (
                step % config.protocol.validation.every_optimizer_updates == 0
                or train_batcher.epoch == runtime_config.epochs
            ):
                model.eval()
                validation_term_blocks: list[GeometryFieldObjectiveTerms] = []
                validation_prediction_blocks: list[GeometrySSLForward] = []
                for validation_index, validation_batch in enumerate(validation_batches):
                    validation_prediction, validation_terms = forward_objective(
                        model,
                        objective,
                        validation_batch,
                        pair_step=validation_index,
                    )
                    validation_term_blocks.append(validation_terms)
                    validation_prediction_blocks.append(validation_prediction)
                    logger.log_terms(
                        step=step,
                        split="validation",
                        terms=validation_terms,
                        asset_ids=validation_batch.asset_ids,
                        batch=validation_batch,
                    )
                    if validation_index == len(validation_batches) - 1:
                        logger.save_dense_snapshot(
                            step=step,
                            split="validation",
                            prediction=validation_prediction,
                            batch=validation_batch,
                        )
                validation_strata = validation_stratified_evidence(
                    tuple(validation_prediction_blocks),
                    validation_batches,
                )
                validation_metrics = stratified_metric_scores(validation_strata)
                if initial_validation_metrics is None:
                    raise RuntimeError("validation selection baseline was not initialized")
                validation_score = normalized_validation_score(validation_metrics, initial_validation_metrics)
                selection_history.append(
                    {
                        "step": step,
                        "score": validation_score,
                        "metrics": validation_metrics,
                        "axis_scores": validation_strata["axis_scores"],
                    }
                )
                if validation_score < best_validation_score:
                    best_validation_score = validation_score
                    immutable_best_path = output_dir / "checkpoints" / f"best_step_{step:08d}.pt"
                    save_geometry_ssl_checkpoint(
                        immutable_best_path,
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        metadata=metadata,
                        runtime_state=checkpoint_runtime_payload(
                            train_batcher,
                            train_window,
                            initial_validation_metrics=initial_validation_metrics,
                            initial_validation_strata=initial_validation_strata,
                            best_validation_score=best_validation_score,
                            selection_history=selection_history,
                        ),
                    )
                    publish_best_checkpoint(output_dir / "checkpoints" / "best.pt", immutable_best_path)
                del validation_term_blocks, validation_prediction_blocks

            if step % config.trainer.checkpoint_every_updates == 0 or train_batcher.epoch == runtime_config.epochs:
                save_geometry_ssl_checkpoint(
                    output_dir / "checkpoints" / f"step_{step:08d}.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    metadata=metadata,
                    runtime_state=checkpoint_runtime_payload(
                        train_batcher,
                        train_window,
                        initial_validation_metrics=initial_validation_metrics,
                        initial_validation_strata=initial_validation_strata,
                        best_validation_score=best_validation_score,
                        selection_history=selection_history,
                    ),
                )

        # Final frozen-model diagnostics 不参与 optimizer 或 checkpoint promotion。
        if validation_batches:
            model.eval()
            with torch.no_grad():
                ablation_evidence = fixed_validation_ablation_evidence(model, validation_batches)
            ablation_path = output_dir / "validation_ablations.yaml"
            ablation_path.write_text(yaml.safe_dump(ablation_evidence, sort_keys=False), encoding="utf-8")
            write_geometry_ssl_ablation_analysis(
                ablation_path,
                output_dir / "validation_ablation_analysis.yaml",
                bootstrap_samples=config.protocol.validation.bootstrap_replicates,
                seed=config.protocol.reproducibility.seed + 2_000_003,
            )

        final_training_q_bank = stream_training_morphology_q_bank(
            model,
            train_runtime,
            config=config,
            device=device,
            dtype=dtype,
            logger=logger,
            phase="final",
        )
        training_q_bank_path.write_text(
            yaml.safe_dump(
                {
                    "initial": initial_training_q_bank,
                    "final": final_training_q_bank,
                    "comparison": compare_training_q_banks(initial_training_q_bank, final_training_q_bank),
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        if last_batch is not None and last_prediction is not None:
            logger.save_dense_snapshot(step=step, split="train", prediction=last_prediction, batch=last_batch)
            save_geometry_ssl_checkpoint(
                output_dir / "checkpoints" / "last.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                metadata=metadata,
                runtime_state=checkpoint_runtime_payload(
                    train_batcher,
                    train_window,
                    initial_validation_metrics=initial_validation_metrics,
                    initial_validation_strata=initial_validation_strata,
                    best_validation_score=best_validation_score,
                    selection_history=selection_history,
                ),
            )
            (output_dir / "checkpoint_selection.yaml").write_text(
                yaml.safe_dump(
                    {
                        "split": "held_out_morphology",
                        "metrics": ["density", "kappa", "derived_field"],
                        "normalization": "divide_by_initial_model_error_then_equal_mean",
                        "aggregation": "morphology_equal_then_bin_equal_then_axis_equal_then_metric_equal",
                        "initial_metrics": initial_validation_metrics,
                        "initial_strata": initial_validation_strata,
                        "best_score": None if best_validation_score == float("inf") else best_validation_score,
                        "history": selection_history,
                        "best_checkpoint": "checkpoints/best.pt" if selection_history else None,
                        "last_checkpoint": "checkpoints/last.pt",
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
    finally:
        train_window.release_all()
        if validation_window is not None:
            validation_window.release_all()
        for event in train_window.drain_telemetry_events():
            logger.log_runtime_event({**event, "phase": "shutdown"})
        for event in validation_window.drain_telemetry_events() if validation_window is not None else ():
            logger.log_runtime_event({**event, "phase": "shutdown"})
        logger.close()
    return output_dir


__all__: list[str] = []
