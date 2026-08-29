r"""Schema-8 full checkpoint 的独立 validation 与 evaluation 执行内核。"""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
import torch

from anymani.assets.asset_schema_geometry import SEMANTICS_SCHEMA_VERSION
from anymani.distill.diagnostics.evaluation.z_compression import UnifiedPCABasis, unified_pca_basis_digest
from anymani.distill.ssl.checkpoint import load_pretrain_checkpoint
from anymani.distill.ssl.runtime.checkpointing import publish_checkpoint_alias
from anymani.distill.ssl.runtime.lifecycle import _plain, _process_memory_evidence, _torch_dtype, _write_yaml
from anymani.distill.ssl.runtime.sampling import FixedAssetQSchedule
from anymani.distill.ssl.runtime.scheduler import ResidentGeometryAssetWindow


def selection_baseline(
    metrics: dict[str, dict[str, float]],
    selection_metrics: tuple[str, ...],
    *,
    teacher_baselines: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, dict[str, float]]:
    r"""为每条 validation suite 复制固定 teacher-only rho/kappa baseline。

    ``metrics`` 只提供 suite 轴和字段完整性；epoch-0 网络仍可单独记录，但不定义 normalization。
    """

    if not metrics:
        raise ValueError("validation selection requires at least one non-empty named suite")
    baseline: dict[str, dict[str, float]] = {}
    for suite_name, suite_metrics in metrics.items():
        missing = set(selection_metrics) - suite_metrics.keys()
        if missing:
            raise ValueError(f"validation suite {suite_name!r} lacks selection terms: {sorted(missing)}")
        source = teacher_baselines[suite_name] if teacher_baselines is not None else suite_metrics
        missing_baselines = set(selection_metrics) - source.keys()
        if missing_baselines:
            raise ValueError(f"teacher baseline lacks selection terms: {sorted(missing_baselines)}")
        baseline[suite_name] = {name: float(source[name]) for name in selection_metrics}
    values = torch.tensor([value for suite in baseline.values() for value in suite.values()])
    if not bool(torch.isfinite(values).all()) or bool((values <= 0.0).any()):
        raise FloatingPointError("initial validation selection metrics must be finite and positive")
    return baseline


def normalized_validation_score(
    metrics: dict[str, dict[str, float]],
    baseline: dict[str, dict[str, float]],
    selection_metrics: tuple[str, ...],
) -> float:
    r"""先对 rho/kappa teacher-baseline-normalized error 等权，再对 validation suites 等权。"""

    if set(metrics) != set(baseline):
        raise ValueError("validation metrics and teacher-baseline suites do not match")
    suite_scores = [
        sum(metrics[suite_name][name] / suite_baseline[name] for name in selection_metrics)
        / len(selection_metrics)
        for suite_name, suite_baseline in baseline.items()
    ]
    score = sum(suite_scores) / len(suite_scores)
    if not torch.isfinite(torch.tensor(score)):
        raise FloatingPointError("normalized validation score must be finite")
    return score


def _configure_execution(*, deterministic_algorithms: bool, seed: int, device_name: str, dtype_name: str) -> tuple[torch.device, torch.dtype]:
    r"""建立与训练一致的 CUDA、RNG 和 dtype 执行边界。"""

    if deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(bool(deterministic_algorithms))
    torch.manual_seed(seed)
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"configured CUDA device is unavailable: {device}")
    return device, _torch_dtype(dtype_name)


def _checkpoint_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    r"""提取跨 baseline/candidate 必须严格相同的训练科学身份。"""

    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("full checkpoint lacks metadata mapping")
    resolved = metadata.get("resolved_config")
    if not isinstance(resolved, Mapping):
        raise ValueError("full checkpoint lacks resolved training config")
    run = resolved.get("run")
    if not isinstance(run, Mapping):
        raise ValueError("full checkpoint resolved config lacks run mapping")
    return {
        "dataset_identity": _plain(metadata.get("dataset_identity")),
        "data": _plain(resolved.get("data")),
        "method": _plain(resolved.get("method")),
        "trainer": _plain(resolved.get("trainer")),
        "seed": run.get("seed"),
        "declared_objective": _plain(metadata.get("declared_objective")),
        "objective_formula": _plain(metadata.get("objective_formula")),
        "fairgrad_formula": _plain(metadata.get("fairgrad_formula")),
        "parameter_partition": _plain(metadata.get("parameter_partition")),
        "source_artifact": _plain(metadata.get("source_artifact")),
        "code_revision": metadata.get("code_revision"),
        "package_version": metadata.get("package_version"),
        "geometry_semantics_schema": metadata.get("geometry_semantics_schema"),
        "worktree_dirty": metadata.get("worktree_dirty"),
        "worktree_fingerprint": metadata.get("worktree_fingerprint"),
    }


def _require_checkpoint_for_stage(
    path: Path,
    *,
    dataset_identity: Mapping[str, Any],
    current_data: Any,
    current_method: Any,
    seed: int,
    current_source_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    r"""加载 full checkpoint，并核对当前 stage 的 dataset/method/seed 身份。"""

    if not path.is_file():
        raise FileNotFoundError(f"full checkpoint does not exist: {path}")
    payload = load_pretrain_checkpoint(path, map_location="cpu")
    identity = _checkpoint_identity(payload)
    if identity["dataset_identity"] != _plain(dataset_identity):
        raise ValueError("checkpoint dataset identity does not match the resolved catalog")
    if identity["data"] != _plain(current_data) or identity["method"] != _plain(current_method):
        raise ValueError("checkpoint data/method config does not match the post-training preset")
    if identity["seed"] != seed:
        raise ValueError("checkpoint training seed does not match the post-training run seed")
    if identity["geometry_semantics_schema"] != SEMANTICS_SCHEMA_VERSION:
        raise ValueError("checkpoint geometry semantics schema does not match the current evaluator")
    if current_source_artifact is not None and identity["source_artifact"] != _plain(current_source_artifact):
        raise ValueError("checkpoint source artifact identity does not match the current post-training source")
    return payload


def _method_source_artifact_identity(method: Any) -> Mapping[str, Any]:
    """读取 runtime method 的 source identity，禁止无法比较的隐式空值。"""

    builder = getattr(method, "source_artifact_identity", None)
    identity = builder() if callable(builder) else {}
    if not isinstance(identity, Mapping):
        raise TypeError("method source_artifact_identity() must return a mapping")
    return identity


def _checkpoint_run_root(path: Path) -> Path:
    r"""返回 full checkpoint 所属 artifact root；普通复制文件退化为其父目录。"""

    resolved = path.expanduser().resolve(strict=False)
    return resolved.parent.parent if resolved.parent.name == "checkpoints" else resolved.parent


def _require_independent_output_dir(output_dir: Path, checkpoint_paths: tuple[Path, ...]) -> None:
    r"""拒绝事后阶段把任何 artifact 写回输入 checkpoint 所属 run。"""

    resolved_output = output_dir.expanduser().resolve(strict=False)
    for checkpoint_path in checkpoint_paths:
        source_root = _checkpoint_run_root(checkpoint_path)
        if resolved_output == source_root or source_root in resolved_output.parents:
            raise ValueError(
                "post-training output directory must remain outside every source checkpoint run: "
                f"output={resolved_output}, source_run={source_root}"
            )


def _run_physical_audit(method: Any, catalog: Any, output_dir: Path) -> dict[str, Any]:
    r"""在显式事后阶段执行一次完整 physical provenance gate。"""

    audit_starter = getattr(method, "start_physical_audit", None)
    if callable(audit_starter):
        handle = audit_starter(catalog)
        wait = getattr(handle, "wait", None)
        if not callable(wait):
            raise TypeError("physical audit handle must expose wait()")
        manifest = wait()
    else:
        manifest = method.asset_manifest(catalog)
    if not isinstance(manifest, Mapping):
        raise TypeError("physical audit must return an asset manifest mapping")
    result = {str(name): value for name, value in manifest.items()}
    _write_yaml(output_dir / "asset_manifest.yaml", result)
    return result


def _run_suites(
    *,
    role: str,
    method: Any,
    config: Any,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    include_ablations: bool,
) -> dict[str, Any]:
    r"""在具名 held-out suites 上执行固定 Method q-bank。"""

    reports: dict[str, Any] = {}
    offset = config.seed_offset if role == "validation" else config.evaluation_seed_offset
    for suite_index, suite_name in enumerate(method.split_names(role)):
        asset_count = method.split_asset_count(role, suite=suite_name)
        if asset_count == 0:
            reports[suite_name] = {"status": "empty", "asset_count": 0}
            continue
        session = method.open_session(
            role,
            suite=suite_name,
            seed=seed + offset + suite_index * 1_000_003,
            device=device,
            dtype=dtype,
            max_resident_assets=config.max_resident_assets,
            window_factory=ResidentGeometryAssetWindow,
        )
        schedule = FixedAssetQSchedule(
            session.asset_count,
            q_per_asset=config.q_per_asset,
            assets_per_minibatch=config.assets_per_minibatch,
            q_per_asset_per_minibatch=config.q_per_asset_per_minibatch,
            max_resident_assets=config.max_resident_assets,
        )
        try:
            reports[suite_name] = method.evaluate_session(
                session,
                schedule,
                include_ablations=include_ablations,
            )
        finally:
            session.close()
    return reports


def _run_training_q_bank(
    *,
    method: Any,
    config: Any,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Any:
    r"""从独立 cursor 0 在训练 morphology 上执行固定 Method 测度。"""

    session = method.open_session(
        "training_evaluation",
        seed=seed + config.training_q_bank_seed_offset,
        device=device,
        dtype=dtype,
        max_resident_assets=config.max_resident_assets,
        window_factory=ResidentGeometryAssetWindow,
    )
    schedule = FixedAssetQSchedule(
        session.asset_count,
        q_per_asset=config.q_per_asset,
        assets_per_minibatch=config.assets_per_minibatch,
        q_per_asset_per_minibatch=config.q_per_asset_per_minibatch,
        max_resident_assets=config.max_resident_assets,
    )
    try:
        return method.evaluate_session(session, schedule, include_ablations=False)
    finally:
        session.close()


def _run_z_compression(
    *,
    method: Any,
    config: Any,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    output_dir: Path,
) -> dict[str, object] | None:
    """拟合 training-q unified PCA，并在 validation fixed banks 上重放原 readers。"""

    fit = getattr(method, "fit_z_compression_basis", None)
    evaluate = getattr(method, "evaluate_z_compression_session", None)
    if not config.z_compression_ranks or not callable(fit) or not callable(evaluate):
        return None
    training_session = method.open_session(
        "training_evaluation",
        seed=seed + config.training_q_bank_seed_offset,
        device=device,
        dtype=dtype,
        max_resident_assets=config.max_resident_assets,
        window_factory=ResidentGeometryAssetWindow,
    )
    training_schedule = FixedAssetQSchedule(
        training_session.asset_count,
        q_per_asset=config.q_per_asset,
        assets_per_minibatch=config.assets_per_minibatch,
        q_per_asset_per_minibatch=config.q_per_asset_per_minibatch,
        max_resident_assets=config.max_resident_assets,
    )
    try:
        basis = cast(UnifiedPCABasis, fit(training_session, training_schedule))
    finally:
        training_session.close()
    suites: dict[str, object] = {}
    for suite_index, suite_name in enumerate(method.split_names("validation")):
        session = method.open_session(
            "validation",
            suite=suite_name,
            seed=seed + config.evaluation_seed_offset + suite_index * 1_000_003,
            device=device,
            dtype=dtype,
            max_resident_assets=config.max_resident_assets,
            window_factory=ResidentGeometryAssetWindow,
        )
        schedule = FixedAssetQSchedule(
            session.asset_count,
            q_per_asset=config.q_per_asset,
            assets_per_minibatch=config.assets_per_minibatch,
            q_per_asset_per_minibatch=config.q_per_asset_per_minibatch,
            max_resident_assets=config.max_resident_assets,
        )
        try:
            suites[suite_name] = evaluate(
                session,
                schedule,
                basis=basis,
                ranks=config.z_compression_ranks,
            )
        finally:
            session.close()
    basis_path = output_dir / "z_compression_basis.npz"
    temporary = basis_path.with_suffix(basis_path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            mean=basis.mean.detach().cpu().numpy(),
            components=basis.components.detach().cpu().numpy(),
            eigenvalues=basis.eigenvalues.detach().cpu().numpy(),
        )
    temporary.replace(basis_path)
    return {
        "schema_version": "1.0.0",
        "basis": {
            "sample_count": basis.sample_count,
            "basis_sha256": unified_pca_basis_digest(basis),
            "artifact": basis_path.name,
            "eigenvalues": basis.eigenvalues,
        },
        "ranks": config.z_compression_ranks,
        "suites": suites,
    }


def _prepare_stage(
    *,
    data: Any,
    method: Any,
    config: Any,
    run: Any,
    output_dir: Path,
    resolved_config: dict[str, Any],
) -> tuple[Path, Any, dict[str, Any], torch.device, torch.dtype, float]:
    r"""解析 catalog、初始化 Method 并写出独立 stage 的配置身份。"""

    started = perf_counter()
    device, dtype = _configure_execution(
        deterministic_algorithms=run.config.deterministic_algorithms,
        seed=run.config.seed,
        device_name=config.device,
        dtype_name=config.dtype,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        catalog = data.resolve()
        configure_source_artifacts = getattr(method, "configure_source_artifacts", None)
        if callable(configure_source_artifacts):
            configure_source_artifacts(
                root=config.source_cache_root,
                mode=config.source_cache_mode,
                dataset_manifest_sha256=str(catalog.dataset.source_sha256),
                producer_device=str(device),
            )
        method.prepare(catalog, device=device, dtype=dtype)
        method.initialize_model(device=device, dtype=dtype)
        identity_builder = getattr(catalog, "training_dataset_identity", None)
        if not callable(identity_builder):
            raise TypeError("resolved catalog must expose training_dataset_identity()")
        dataset_identity = identity_builder()
        if not isinstance(dataset_identity, Mapping):
            raise TypeError("training_dataset_identity() must return a mapping")
        dataset_identity = {str(name): value for name, value in dataset_identity.items()}
        _write_yaml(output_dir / "resolved_config.yaml", resolved_config)
        _write_yaml(output_dir / "asset_dataset.yaml", catalog.dataset.config_dict())
        _write_yaml(output_dir / "training_dataset_identity.yaml", dataset_identity)
        return output_dir, catalog, dataset_identity, device, dtype, started
    except BaseException:
        method.close()
        raise


def _write_stage_resources(output_dir: Path, method: Any, *, started: float) -> None:
    r"""写出事后阶段的资源高水位，不改变评估统计。"""

    evidence = getattr(method, "runtime_resource_evidence", None)
    payload: dict[str, Any] = {}
    if callable(evidence):
        raw = evidence()
        if not isinstance(raw, Mapping):
            raise TypeError("runtime_resource_evidence must return a mapping")
        payload.update({str(name): value for name, value in raw.items()})
    payload["process_memory"] = _process_memory_evidence()
    payload["elapsed_seconds"] = perf_counter() - started
    _write_yaml(output_dir / "runtime_resources.yaml", payload)


def validate_checkpoints(
    *,
    data: Any,
    method: Any,
    config: Any,
    run: Any,
    output_dir_override: Path | None,
    resolved_config: dict[str, Any],
) -> Path:
    r"""记录显式 epoch-0 网络证据，并用每条固定 validation suite 自身 teacher baseline 选择候选。"""

    run.config.validate_inputs()
    baseline_path = Path(run.config.baseline_checkpoint).expanduser().resolve()
    candidate_paths = tuple(Path(path).expanduser().resolve() for path in run.config.checkpoints)
    output_dir = run.resolve_output_dir(output_dir_override)
    _require_independent_output_dir(output_dir, (baseline_path, *candidate_paths))
    output_dir, catalog, dataset_identity, device, dtype, started = _prepare_stage(
        data=data,
        method=method,
        config=config,
        run=run,
        output_dir=output_dir,
        resolved_config=resolved_config,
    )
    try:
        current_source_artifact = _method_source_artifact_identity(method)
        baseline_payload = _require_checkpoint_for_stage(
            baseline_path,
            dataset_identity=dataset_identity,
            current_data=resolved_config["data"],
            current_method=resolved_config["method"],
            seed=run.config.seed,
            current_source_artifact=current_source_artifact,
        )
        if int(baseline_payload["epoch"]) != 0 or int(baseline_payload["optimizer_update"]) != 0:
            raise ValueError("validation baseline checkpoint must be the unupdated epoch_000000 state")
        baseline_identity = _checkpoint_identity(baseline_payload)
        candidates: list[tuple[Path, int]] = []
        candidate_epochs: set[int] = set()
        for path in candidate_paths:
            payload = _require_checkpoint_for_stage(
                path,
                dataset_identity=dataset_identity,
                current_data=resolved_config["data"],
                current_method=resolved_config["method"],
                seed=run.config.seed,
                current_source_artifact=current_source_artifact,
            )
            if _checkpoint_identity(payload) != baseline_identity:
                raise ValueError("validation baseline and candidates do not share one training lineage")
            epoch = int(payload["epoch"])
            if epoch <= 0:
                raise ValueError("validation candidates must be post-update epoch checkpoints")
            if epoch in candidate_epochs:
                raise ValueError("validation candidates must have distinct completed epochs")
            candidate_epochs.add(epoch)
            candidates.append((path, epoch))
            del payload  # candidate optimizer/method state 留在 CPU，预检后立即释放

        _run_physical_audit(method, catalog, output_dir)
        method.eval_mode()
        method.load_training_state_dict(baseline_payload["method_state"])
        baseline_reports = _run_suites(
            role="validation",
            method=method,
            config=config,
            seed=run.config.seed,
            device=device,
            dtype=dtype,
            include_ablations=False,
        )
        baseline_metrics = {
            name: dict(report.metrics) for name, report in baseline_reports.items() if hasattr(report, "metrics")
        }
        suite_teacher_baselines = {
            name: {
                metric: float(
                    report.teacher_baselines[metric][
                        "physical_baseline_mse"
                        if metric == "kappa"
                        else "baseline_mse"
                    ]
                )
                for metric in config.selection_metrics
            }
            for name, report in baseline_reports.items()
            if hasattr(report, "teacher_baselines")
        }
        baseline = selection_baseline(
            baseline_metrics,
            config.selection_metrics,
            teacher_baselines=suite_teacher_baselines,
        )
        _write_yaml(output_dir / "validation_baseline.yaml", baseline_reports)

        history: list[dict[str, Any]] = []
        for path, epoch in candidates:
            payload = _require_checkpoint_for_stage(
                path,
                dataset_identity=dataset_identity,
                current_data=resolved_config["data"],
                current_method=resolved_config["method"],
                seed=run.config.seed,
                current_source_artifact=current_source_artifact,
            )
            method.load_training_state_dict(payload["method_state"])
            reports = _run_suites(
                role="validation",
                method=method,
                config=config,
                seed=run.config.seed,
                device=device,
                dtype=dtype,
                include_ablations=False,
            )
            metrics = {name: dict(report.metrics) for name, report in reports.items() if hasattr(report, "metrics")}
            score = normalized_validation_score(metrics, baseline, config.selection_metrics)
            _write_yaml(output_dir / f"validation_epoch_{epoch:06d}.yaml", reports)
            trainer_state = payload["trainer_state"]
            history.append(
                {
                    "source_checkpoint": str(path),
                    "epoch": epoch,
                    "optimizer_update": int(payload["optimizer_update"]),
                    "new_pairs_seen": int(trainer_state.get("new_pairs_seen", 0)),
                    "score": score,
                    "metrics": metrics,
                }
            )
            del payload  # 下一候选加载前释放当前 full checkpoint 与 AdamW state

        best = min(history, key=lambda item: float(item["score"]))
        source = Path(str(best["source_checkpoint"]))
        immutable = output_dir / "checkpoints" / f"selected_epoch_{int(best['epoch']):06d}.pt"
        immutable.parent.mkdir(parents=True, exist_ok=True)
        if immutable.exists():
            raise FileExistsError(f"immutable validation checkpoint already exists: {immutable}")
        shutil.copy2(source, immutable)
        publish_checkpoint_alias(output_dir / "checkpoints" / "best.pt", immutable)
        _write_yaml(
            output_dir / "checkpoint_selection.yaml",
            {
                "schema_version": "1.0.0",
                "selection_metrics": config.selection_metrics,
                "baseline_checkpoint": str(baseline_path),
                "baseline_kind": "teacher_only_naive",
                "baseline": baseline,
                "history": history,
                "best_source_checkpoint": str(source),
                "best_checkpoint": str(immutable.relative_to(output_dir)),
            },
        )
        return output_dir
    finally:
        try:
            _write_stage_resources(output_dir, method, started=started)
        finally:
            method.close()


def evaluate_checkpoint(
    *,
    data: Any,
    method: Any,
    config: Any,
    run: Any,
    output_dir_override: Path | None,
    resolved_config: dict[str, Any],
) -> Path:
    r"""对一个显式 full checkpoint 运行 held-out suites；baseline 只控制 q-bank 对比。"""

    run.config.validate_inputs()
    checkpoint_path = Path(run.config.checkpoint).expanduser().resolve()
    baseline_path = (
        Path(run.config.baseline_checkpoint).expanduser().resolve() if run.config.baseline_checkpoint else None
    )
    output_dir = run.resolve_output_dir(output_dir_override)
    source_paths = (checkpoint_path,) if baseline_path is None else (checkpoint_path, baseline_path)
    _require_independent_output_dir(output_dir, source_paths)
    output_dir, catalog, dataset_identity, device, dtype, started = _prepare_stage(
        data=data,
        method=method,
        config=config,
        run=run,
        output_dir=output_dir,
        resolved_config=resolved_config,
    )
    try:
        current_source_artifact = _method_source_artifact_identity(method)
        payload = _require_checkpoint_for_stage(
            checkpoint_path,
            dataset_identity=dataset_identity,
            current_data=resolved_config["data"],
            current_method=resolved_config["method"],
            seed=run.config.seed,
            current_source_artifact=current_source_artifact,
        )
        baseline_payload: dict[str, Any] | None = None
        if baseline_path is not None:
            baseline_payload = _require_checkpoint_for_stage(
                baseline_path,
                dataset_identity=dataset_identity,
                current_data=resolved_config["data"],
                current_method=resolved_config["method"],
                seed=run.config.seed,
                current_source_artifact=current_source_artifact,
            )
            if _checkpoint_identity(payload) != _checkpoint_identity(baseline_payload):
                raise ValueError("evaluation checkpoint and optional baseline do not share one training lineage")
            if int(baseline_payload["epoch"]) != 0 or int(baseline_payload["optimizer_update"]) != 0:
                raise ValueError("evaluation baseline checkpoint must be the unupdated epoch_000000 state")

        _run_physical_audit(method, catalog, output_dir)
        method.eval_mode()
        if baseline_payload is not None:
            method.load_training_state_dict(baseline_payload["method_state"])
            initial_q_bank = _plain(
                _run_training_q_bank(method=method, config=config, seed=run.config.seed, device=device, dtype=dtype)
            )
            method.load_training_state_dict(payload["method_state"])
            final_q_bank = _plain(
                _run_training_q_bank(method=method, config=config, seed=run.config.seed, device=device, dtype=dtype)
            )
            if initial_q_bank.get("strata", {}).get("bank_digest_sha256") != final_q_bank.get("strata", {}).get(
                "bank_digest_sha256"
            ):
                raise RuntimeError("training morphology q-bank identity changed between baseline and checkpoint")
            comparison = {
                name: {
                    "initial": float(initial_q_bank["metrics"][name]),
                    "final": float(final_q_bank["metrics"][name]),
                    "improvement_initial_minus_final": (
                        float(initial_q_bank["metrics"][name]) - float(final_q_bank["metrics"][name])
                    ),
                }
                for name in config.selection_metrics
            }
            _write_yaml(
                output_dir / "training_morphology_q_bank.yaml",
                {
                    "baseline_checkpoint": str(baseline_path),
                    "checkpoint": str(checkpoint_path),
                    "initial": initial_q_bank,
                    "final": final_q_bank,
                    "comparison": comparison,
                },
            )

        method.load_training_state_dict(payload["method_state"])
        reports = _run_suites(
            role="evaluation",
            method=method,
            config=config,
            seed=run.config.seed,
            device=device,
            dtype=dtype,
            include_ablations=True,
        )
        summary: dict[str, Any] = {}
        for suite_index, (suite_name, report) in enumerate(reports.items()):
            if not hasattr(report, "metrics"):
                summary[suite_name] = report
                continue
            suite_payload = {
                "metrics": report.metrics,
                "strata": report.strata,
                "teacher_baselines": report.teacher_baselines,
                "ablations": report.ablations,
            }
            if report.ablations is not None:
                actual = tuple(str(name) for name in report.ablations.get("ablations", ()))[1:]
                if actual != config.final_ablations:
                    raise ValueError("Method final ablations do not match evaluation config")
                suite_payload["ablation_analysis"] = method.analyze_ablations(
                    report.ablations,
                    bootstrap_replicates=config.bootstrap_replicates,
                    seed=run.config.seed + config.bootstrap_seed_offset + suite_index * 1_000_003,
                )
            summary[suite_name] = suite_payload
        _write_yaml(
            output_dir / "evaluation.yaml",
            {
                "schema_version": "1.0.0",
                "source_checkpoint": str(checkpoint_path),
                "baseline_checkpoint": str(baseline_path) if baseline_path is not None else None,
                "suites": summary,
            },
        )
        z_compression = _run_z_compression(
            method=method,
            config=config,
            seed=run.config.seed,
            device=device,
            dtype=dtype,
            output_dir=output_dir,
        )
        if z_compression is not None:
            _write_yaml(output_dir / "z_compression.yaml", z_compression)
        return output_dir
    finally:
        try:
            _write_stage_resources(output_dir, method, started=started)
        finally:
            method.close()


__all__ = [
    "evaluate_checkpoint",
    "normalized_validation_score",
    "selection_baseline",
    "validate_checkpoints",
]
