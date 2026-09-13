r"""Schema-9 full checkpoint 的显式 evaluation 执行内核。"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from anymani.assets.asset_schema_geometry import SEMANTICS_SCHEMA_VERSION
from anymani.distill.diagnostics.evaluation.z_compression import UnifiedPCABasis, unified_pca_basis_digest
from anymani.distill.ssl.checkpoint import load_pretrain_checkpoint
from anymani.distill.ssl.runtime.lifecycle import _plain, _process_memory_evidence, _torch_dtype, _write_yaml
from anymani.distill.ssl.runtime.sampling import FixedAssetQSchedule
from anymani.distill.ssl.runtime.scheduler import ResidentGeometryAssetWindow


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
    checkpoint_dataset = identity["dataset_identity"]
    if not isinstance(checkpoint_dataset, Mapping) or checkpoint_dataset.get("source_sha256") != dataset_identity.get(
        "source_sha256"
    ):
        raise ValueError("checkpoint train manifest identity does not match the evaluation manifest bytes")
    if identity["data"] != _plain(current_data) or identity["method"] != _plain(current_method):
        raise ValueError("checkpoint data/method config does not match the post-training preset")
    if identity["seed"] != seed:
        raise ValueError("checkpoint training seed does not match the post-training run seed")
    if identity["geometry_semantics_schema"] != SEMANTICS_SCHEMA_VERSION:
        raise ValueError("checkpoint geometry semantics schema does not match the current evaluator")
    if current_source_artifact is not None:
        checkpoint_source = identity["source_artifact"]
        current_source = _plain(current_source_artifact)
        if not isinstance(checkpoint_source, Mapping) or not isinstance(current_source, Mapping):
            raise ValueError("checkpoint/current source artifact identities must be mappings")
        stable_fields = ("schema_version", "algorithms")
        if any(checkpoint_source.get(name) != current_source.get(name) for name in stable_fields):
            raise ValueError("checkpoint source artifact schema/algorithm identity does not match evaluation")
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
    if role != "evaluation":
        raise ValueError(f"post-training suite role must be evaluation, got {role!r}")
    offset = config.evaluation_seed_offset
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


def _run_z_compression(
    *,
    method: Any,
    config: Any,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    output_dir: Path,
    basis: UnifiedPCABasis,
) -> dict[str, Any] | None:
    r"""消费 train-derived basis，并在 evaluation fixed banks 上重放原 readers。

    basis 的拟合必须发生在训练角色，evaluation 进程只解析 held-out catalog，因而这里不打开 train
    provider，也不把 8192 项 train source 重新展开。低秩结果只属于显式 compression analysis，核心
    evaluation 的指标和 teacher baseline 不被替换。
    """

    evaluate = getattr(method, "evaluate_z_compression_session", None)
    if not config.z_compression_ranks or not callable(evaluate):
        return None
    suites: dict[str, object] = {}
    for suite_index, suite_name in enumerate(method.split_names("evaluation")):
        session = method.open_session(
            "evaluation",
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
            sample_count=np.asarray(basis.sample_count, dtype=np.int64),
        )
    temporary.replace(basis_path)
    return {
        "schema_version": "2.0.0",
        "basis": {
            "sample_count": basis.sample_count,
            "basis_sha256": unified_pca_basis_digest(basis),
            "artifact": basis_path.name,
            "eigenvalues": basis.eigenvalues,
        },
        "ranks": config.z_compression_ranks,
        "suites": suites,
    }


def _load_z_compression_basis(path: Path, *, width: int) -> UnifiedPCABasis:
    r"""读取并验证训练阶段发布的统一 PCA basis，不允许 evaluation 临时改拟合数据。"""

    if not path.is_file():
        raise FileNotFoundError(
            "compression analysis requires the train-derived basis artifact; "
            f"expected {path}. Run training to completion or pass --compression_basis."
        )
    with np.load(path, allow_pickle=False) as arrays:
        required = {"mean", "components", "eigenvalues", "sample_count"}
        missing = required - set(arrays.files)
        if missing:
            raise ValueError(f"compression basis is missing arrays: {sorted(missing)}")
        mean = torch.from_numpy(np.array(arrays["mean"], copy=True)).to(torch.float64)
        components = torch.from_numpy(np.array(arrays["components"], copy=True)).to(torch.float64)
        eigenvalues = torch.from_numpy(np.array(arrays["eigenvalues"], copy=True)).to(torch.float64)
        sample_count = int(np.asarray(arrays["sample_count"]).item())
    if mean.shape != (width,) or components.shape != (width, width) or eigenvalues.shape != (width,):
        raise ValueError(
            "compression basis shape does not match evaluation encoder width: "
            f"mean={tuple(mean.shape)}, components={tuple(components.shape)}, eigenvalues={tuple(eigenvalues.shape)}, "
            f"width={width}"
        )
    if sample_count < 2 or not all(bool(torch.isfinite(value).all()) for value in (mean, components, eigenvalues)):
        raise ValueError("compression basis must contain finite tensors and at least two fitted tokens")
    return UnifiedPCABasis(mean, components, eigenvalues, sample_count)


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
        catalog = data.resolve_evaluation()
        configure_source_artifacts = getattr(method, "configure_source_artifacts", None)
        if callable(configure_source_artifacts):
            configure_source_artifacts(
                root=config.source_cache_root,
                mode=config.source_cache_mode,
                dataset_manifest_sha256=str(catalog.dataset.source_sha256),
                producer_device=str(device),
                role="evaluation",
            )
        method.prepare(catalog, role="evaluation", device=device, dtype=dtype)
        torch.backends.cuda.matmul.allow_tf32 = bool(config.execution.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(config.execution.allow_tf32)
        configure_execution = getattr(method, "configure_execution", None)
        if callable(configure_execution):
            configure_execution(config.execution)
        method.initialize_model(device=device, dtype=dtype)
        dataset_identity = {"source_sha256": str(catalog.dataset.source_sha256)}
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


def evaluate_checkpoint(
    *,
    data: Any,
    method: Any,
    config: Any,
    run: Any,
    output_dir_override: Path | None,
    resolved_config: dict[str, Any],
) -> Path:
    r"""对一个显式 full checkpoint 运行 held-out core 与用户点名的昂贵分析。"""

    run.config.validate_inputs()
    checkpoint_path = Path(run.config.checkpoint).expanduser().resolve()
    output_dir = run.resolve_output_dir(output_dir_override)
    _require_independent_output_dir(output_dir, (checkpoint_path,))
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
        _run_physical_audit(method, catalog, output_dir)
        method.eval_mode()
        method.load_training_state_dict(payload["method_state"])
        reports = _run_suites(
            role="evaluation",
            method=method,
            config=config,
            seed=run.config.seed,
            device=device,
            dtype=dtype,
            include_ablations="ablations" in run.config.analyses,
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
                "analyses": run.config.analyses,
                "suites": summary,
            },
        )
        if "compression" in run.config.analyses:
            basis_path = (
                Path(run.config.compression_basis).expanduser().resolve()
                if run.config.compression_basis
                else _checkpoint_run_root(checkpoint_path) / "z_compression_basis.npz"
            )
            width = method.feature_spec().entity_width
            basis = _load_z_compression_basis(basis_path, width=width)
            compression = _run_z_compression(
                method=method,
                config=config,
                seed=run.config.seed,
                device=device,
                dtype=dtype,
                output_dir=output_dir,
                basis=basis,
            )
            if compression is None:
                raise RuntimeError("compression analysis was requested but Method does not expose its replay API")
            basis_record = compression.get("basis")
            if not isinstance(basis_record, dict):
                raise TypeError("compression report lacks a basis mapping")
            basis_record["artifact"] = str(basis_path)
            _write_yaml(output_dir / "z_compression.yaml", compression)
        return output_dir
    finally:
        try:
            _write_stage_resources(output_dir, method, started=started)
        finally:
            method.close()


__all__ = [
    "evaluate_checkpoint",
]
