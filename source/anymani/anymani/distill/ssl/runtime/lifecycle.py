r"""Schema 9 online procedural supervised pure-pretraining lifecycle.

该模块是最高级训练内核：Data runtime 解析 catalog，Method session 封闭产生 batch，Trainer 拥有
epoch/minibatch schedule、分组 optimizer、backward 与 full checkpoint。Evaluation 使用独立入口，
不会从本模块被隐式启动。
生命周期不读取 representation 内部字段，也不解释 owner/query/edge 轴。
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
import torch
import yaml

from anymani.assets.asset_schema_geometry import SEMANTICS_SCHEMA_VERSION
from anymani.distill.diagnostics.evaluation.z_compression import UnifiedPCABasis, unified_pca_basis_digest
from anymani.distill.diagnostics.recording.geometry_ssl import GeometrySSLRunLogger
from anymani.distill.ssl.checkpoint import load_pretrain_checkpoint
from anymani.distill.ssl.runtime.checkpointing import (
    publish_checkpoint_alias,
    require_resume_metadata_identity,
    require_resume_scientific_config,
)
from anymani.distill.ssl.runtime.sampling import OnlineMinibatchSchedule


def _torch_dtype(name: str) -> torch.dtype:
    r"""把配置字符串映射为当前 Geometry SSL execution policy 的 dtype。"""

    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
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


def _publish_train_compression_basis(
    *,
    method: Any,
    train_session: Any,
    device: torch.device,
    dtype: torch.dtype,
    config: Any,
    seed: int,
    output_dir: Path,
) -> None:
    r"""在最终 train 参数上拟合并发布 unified-$Z$ PCA basis。

    该阶段只重放 train role 的固定 q-bank，不执行 optimizer、不读取 evaluation、也不改变正式
    `1024` update 预算。PCA 的均值和主方向以 FP64 充分统计写入独立 NPZ，evaluation 随后只消费
    该文件并在 held-out role replay，不会在 evaluation 进程中重新拟合训练分布。
    """

    from anymani.distill.ssl.runtime.sampling import FixedAssetQSchedule
    from anymani.distill.ssl.runtime.scheduler import ResidentGeometryAssetWindow

    fit = getattr(method, "fit_z_compression_basis", None)
    if not callable(fit):
        raise TypeError("compression basis publication requires Method.fit_z_compression_basis()")
    q_per_asset = int(config.compression_q_per_asset)
    # 训练更新已完成；释放其 resident lease 后再打开固定 q-bank，避免两套 source window 叠加显存。
    train_session.close()
    basis_session = method.open_session(
        "train",
        seed=seed + 3_000_003,
        device=device,
        dtype=dtype,
        max_resident_assets=min(8, train_session.asset_count),
        window_factory=ResidentGeometryAssetWindow,
    )
    schedule = FixedAssetQSchedule(
        basis_session.asset_count,
        q_per_asset=q_per_asset,
        assets_per_minibatch=min(8, basis_session.asset_count),
        q_per_asset_per_minibatch=min(8, q_per_asset),
        max_resident_assets=min(8, basis_session.asset_count),
    )
    try:
        basis = cast(UnifiedPCABasis, fit(basis_session, schedule))
    finally:
        basis_session.close()
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
    _write_yaml(
        output_dir / "z_compression_basis.yaml",
        {
            "schema_version": "2.0.0",
            "source": "train_role_fixed_q_bank",
            "q_per_asset": q_per_asset,
            "sample_count": basis.sample_count,
            "basis_sha256": unified_pca_basis_digest(basis),
            "artifact": basis_path.name,
        },
    )


def _build_batch(
    schedule_item: Any,
    *,
    session: Any,
    schedule: OnlineMinibatchSchedule,
    step: int,
) -> Any:
    r"""把离散 schedule item 交给 opaque Method session realization。"""

    return session.realize(schedule_item, schedule=schedule, step=step)


def _sampling_state(
    schedule: OnlineMinibatchSchedule, session: Any
) -> dict[str, Any]:
    r"""合并 Trainer schedule 与 Method session cursor，作为 optimizer-boundary state。"""

    return {
        "schedule": schedule.state_dict(),
        "method_session": session.state_dict(),
    }


def _restore_sampling_state(
    payload: dict[str, Any],
    schedule: OnlineMinibatchSchedule,
    session: Any,
) -> None:
    r"""严格恢复 Trainer schedule 与 opaque Method session state。"""

    raw_schedule = payload.get("schedule")
    if not isinstance(raw_schedule, dict):
        raise ValueError("checkpoint lacks schema 9 epoch/minibatch state")
    schedule.load_state_dict(raw_schedule)
    raw_session = payload.get("method_session")
    if not isinstance(raw_session, dict):
        raise ValueError("checkpoint lacks method session state")
    session.load_state_dict(raw_session)


def _declared_objective_weights(method: Any) -> dict[str, float]:
    r"""读取 method 显式声明的 objective 权重；baseline 归一化不改写该声明。"""

    return dict(method.declared_objective_weights())


def _format_objective_terms(terms: Mapping[str, float]) -> str:
    r"""按 method 声明顺序格式化任意 objective terms，不让 Trainer 绑定具体物理目标名。"""

    if not terms:
        raise ValueError("objective term summary requires at least one named term")
    return " ".join(f"{name}={float(value):.6e}" for name, value in terms.items())


def _mini_epoch_order(
    num_minibatches: int,
    *,
    seed: int,
    epoch_index: int,
    mini_epoch_index: int,
) -> tuple[int, ...]:
    r"""由训练身份确定性重排当前 epoch buffer 的 minibatch 访问顺序。"""

    if min(num_minibatches, seed + 1, epoch_index + 1, mini_epoch_index + 1) < 1:
        raise ValueError("mini-epoch ordering inputs must be non-negative and num_minibatches positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + epoch_index * 1_000_003 + mini_epoch_index * 10_007)
    return tuple(int(index) for index in torch.randperm(num_minibatches, generator=generator).tolist())


def _scientific_pretrain_identity(resolved_config: dict[str, Any], *, formula_identity: dict[str, str]) -> dict[str, Any]:
    r"""记录本次预实验的完整方法与采样配置，供研究者比较而不强制 preset 相同。"""

    method = resolved_config.get("method")
    trainer = resolved_config.get("trainer")
    if not isinstance(method, dict) or not isinstance(trainer, dict):
        raise ValueError("resolved config must contain method and trainer mappings")
    if not formula_identity:
        raise ValueError("method formula identity must be non-empty")
    return {
        "formula_identity": dict(formula_identity),
        "state_measure": method.get("state_measure"),
        "representation": method.get("representation"),
        "model": method.get("model"),
        "joint_sign_rewrite": method.get("joint_sign_rewrite"),
        "sampling": trainer.get("sampling"),
        "max_epochs": trainer.get("max_epochs"),
        "num_minibatches": trainer.get("num_minibatches"),
        "mini_epochs": trainer.get("mini_epochs"),
        "microbatch_size": trainer.get("microbatch_size"),
    }


def _worktree_fingerprint() -> tuple[bool, str]:
    r"""对 tracked diff 与 untracked 文件内容做 SHA-256；checkpoint 只保存摘要。"""

    import subprocess

    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        if not status.strip():
            return False, ""
        digest = hashlib.sha256(b"anymani-worktree-v2\0")
        tracked_diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--"],
            check=True,
            capture_output=True,
            timeout=30,
        ).stdout
        digest.update(tracked_diff)
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            check=True,
            capture_output=True,
            timeout=30,
        ).stdout.split(b"\0")
        for raw_relative in untracked:
            if not raw_relative:
                continue
            relative = os.fsdecode(raw_relative)
            path = Path(relative)
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            if path.is_file():
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
    except (OSError, subprocess.SubprocessError):
        return True, "unknown"
    return True, digest.hexdigest()


def _process_memory_evidence() -> dict[str, int]:
    r"""读取当前进程的 Linux RSS 高水位、当前 RSS 与 swap，统一换算为 bytes。

    ``ru_maxrss`` 与 ``/proc/self/status`` 的 Linux 单位均为 KiB。当前值和高水位同时记录，
    用于区分 batch 处理中可释放的临时几何数组与 lifecycle 结束时仍被持有的引用。
    """

    import resource

    fields: dict[str, int] = {}  # Linux status 名称到 KiB 数值；缺失字段按零处理
    status_path = Path("/proc/self/status")
    if status_path.is_file():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            name, separator, remainder = line.partition(":")
            if separator and name in {"VmRSS", "VmHWM", "VmSwap"}:
                fields[name] = int(remainder.strip().split()[0])
    resource_peak_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)  # 当前进程历史 RSS 峰值，KiB
    return {
        "current_rss_bytes": fields.get("VmRSS", 0) * 1024,
        "peak_rss_bytes": max(fields.get("VmHWM", 0), resource_peak_kib) * 1024,
        "current_swap_bytes": fields.get("VmSwap", 0) * 1024,
    }


def fit_embodiment_pretrain(
    *,
    trainer: Any,
    data: Any,
    method: Any,
    run: Any,
    output_dir_override: Path | None,
    resolved_config: dict[str, Any],
) -> Path:
    r"""执行 schema-9 pure-pretrain phase，并在同一 run 内累计 teacher baseline。

    ``pretrain`` 只产生在线监督、执行参数更新、记录训练曲线并保存 full checkpoint。固定 q-bank、
    held-out evaluation、physical audit 与 retained export 均不属于该进程。
    """

    from anymani.distill.ssl.runtime.scheduler import ResidentGeometryAssetWindow

    lifecycle_started = perf_counter()  # catalog/method/model/session 的统一 wall-time 原点
    runtime_timing: dict[str, float] = {}  # 只写 runtime_resources，不进入科学 artifact identity
    print("[SSL] Schema 9 pure pretraining")
    if run.config.deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(bool(run.config.deterministic_algorithms))
    torch.manual_seed(run.config.seed)
    device = torch.device(trainer.config.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"configured CUDA device is unavailable: {device}")
    teacher_dtype = _torch_dtype(trainer.config.execution.teacher_dtype)
    parameter_dtype = _torch_dtype(trainer.config.execution.parameter_dtype)
    torch.backends.cuda.matmul.allow_tf32 = bool(trainer.config.execution.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(trainer.config.execution.allow_tf32)
    print("[SSL] Preparing output directory...")
    output_dir = run.prepare_output_dir(output_dir_override)
    print(f"[SSL] Output: {output_dir}")
    logger = GeometrySSLRunLogger(output_dir)
    print("[SSL] Resolving asset catalog (this may take 1-2 minutes for 8k assets)...")
    stage_started = perf_counter()
    catalog = data.resolve_train()
    runtime_timing["catalog_resolve_seconds"] = perf_counter() - stage_started
    train_count = len(catalog.train) if hasattr(catalog, "train") else "synthetic"
    print(f"[SSL] Catalog resolved: train={train_count} assets in {runtime_timing['catalog_resolve_seconds']:.2f}s")
    print("[SSL] Preparing method (computing FK/Jacobian templates)...")
    stage_started = perf_counter()
    configure_source_artifacts = getattr(method, "configure_source_artifacts", None)
    source_mode = run.config.source_cache_mode
    effective_source_mode = "read-write" if source_mode == "auto" else source_mode
    if callable(configure_source_artifacts):
        configure_source_artifacts(
            root=run.config.source_cache_root,
            mode=effective_source_mode,
            dataset_manifest_sha256=str(catalog.dataset.source_sha256),
            producer_device=str(device),
            role="train",
        )
    method.prepare(catalog, role="train", device=device, dtype=teacher_dtype)
    runtime_timing["method_prepare_seconds"] = perf_counter() - stage_started
    print(f"[SSL] Method prepared in {runtime_timing['method_prepare_seconds']:.2f}s")
    print(f"[SSL] Initializing model on {device}...")
    stage_started = perf_counter()
    configure_execution = getattr(method, "configure_execution", None)
    if callable(configure_execution):
        configure_execution(trainer.config.execution)
    method.initialize_model(device=device, dtype=parameter_dtype)
    runtime_timing["model_initialize_seconds"] = perf_counter() - stage_started
    print(f"[SSL] Model initialized in {runtime_timing['model_initialize_seconds']:.2f}s")
    _write_yaml(output_dir / "resolved_config.yaml", resolved_config)
    _write_yaml(output_dir / "asset_dataset.yaml", catalog.dataset.config_dict())
    dirty, fingerprint = _worktree_fingerprint()
    declared_weights = _declared_objective_weights(method)
    identity_builder = getattr(catalog, "training_dataset_identity", None)
    if not callable(identity_builder):
        raise TypeError("resolved catalog must expose training_dataset_identity() for schema 9 checkpoints")
    dataset_identity = identity_builder()
    if not isinstance(dataset_identity, Mapping):
        raise TypeError("training_dataset_identity() must return a mapping")
    dataset_identity = {str(name): value for name, value in dataset_identity.items()}
    _write_yaml(output_dir / "training_dataset_identity.yaml", dataset_identity)

    def open_session(role: str, *, suite: str = "", seed: int) -> Any:
        r"""以统一资源上限打开 opaque Method split session。"""

        return method.open_session(
            role,
            suite=suite,
            seed=seed,
            device=device,
            dtype=teacher_dtype,
            max_resident_assets=trainer.config.device_window_assets,
            window_factory=ResidentGeometryAssetWindow,
            resource_profile=trainer.config.resource_profile,
        )

    def write_resource_evidence() -> None:
        r"""在 arena clear 前记录 Method 暴露的有界资源事实；不要求所有 Method 实现该可选诊断。"""

        evidence = getattr(method, "runtime_resource_evidence", None)
        if callable(evidence):
            raw_evidence = evidence()
            if not isinstance(raw_evidence, Mapping):
                raise TypeError("runtime_resource_evidence must return a mapping")
            payload: dict[str, object] = {str(name): value for name, value in raw_evidence.items()}
            payload["process_memory"] = _process_memory_evidence()
            runtime_timing["lifecycle_elapsed_seconds"] = perf_counter() - lifecycle_started
            payload["lifecycle_timing"] = dict(runtime_timing)
            _write_yaml(output_dir / "runtime_resources.yaml", payload)

    print("[SSL] Opening training session (train partition)...")
    train_session = open_session("train", seed=trainer.config.sampling.seed)
    print(f"[SSL] Training session: {train_session.asset_count} assets")
    train_schedule = OnlineMinibatchSchedule(
        train_session.asset_count,
        trainer.config.sampling,
        max_epochs=trainer.config.max_epochs,
        num_minibatches=trainer.config.num_minibatches,
        max_resident_assets=trainer.config.sampling.assets_per_minibatch,
    )
    method_parameter_groups = method.optimizer_parameter_groups()
    optimizer = torch.optim.AdamW(
        [
            {"name": group.name, "params": group.parameters}
            for group in method_parameter_groups
        ],
        lr=trainer.config.optimizer.learning_rate,
        weight_decay=trainer.config.optimizer.weight_decay,
    )
    completed_epochs = 0
    optimizer_update = 0
    new_pairs_seen = 0
    pair_uses = 0
    teacher_pairs_realized = 0
    forward_index = 0  # 含 mini-epoch 复用的全局前向序号，决定 augmentation seed
    baseline_statistics: dict[str, torch.Tensor] | None = None  # 每个新 teacher batch 只累计一次的 CPU FP64 统计
    source_ref_count = 0
    source_ref_digest = hashlib.sha256(b"anymani-source-artifact-ref-log-v1\0")
    source_ref_path = output_dir / "source_artifacts.jsonl"
    automatic_recovery = output_dir / "checkpoints" / "recovery.pt"
    resume_path = (
        Path(run.config.resume_checkpoint).expanduser().resolve()
        if run.config.resume_checkpoint
        else (automatic_recovery if automatic_recovery.is_file() and not run.config.new_run else None)
    )
    log_continuation_offsets = {"metrics_jsonl_bytes": 0, "runtime_jsonl_bytes": 0}
    source_ref_byte_offset = 0
    if resume_path is not None and resume_path.parent.name != "checkpoints":
        raise ValueError("resume checkpoint must remain under its source run's checkpoints directory")

    def metadata() -> Any:
        r"""构造不触发 physical audit 的 pure-pretrain checkpoint lineage。"""

        source_identity_builder = getattr(method, "source_artifact_identity", None)
        source_identity = source_identity_builder() if callable(source_identity_builder) else {}
        if not isinstance(source_identity, Mapping):
            raise TypeError("method source_artifact_identity() must return a mapping")
        return run.checkpoint_metadata(
            geometry_semantics_schema=SEMANTICS_SCHEMA_VERSION,
            dataset_identity=dataset_identity,
            resolved_config=resolved_config,
            declared_objective=declared_weights,
            objective_formula=method.formula_identity(),
            fairgrad_formula=method.optimization_identity(),
            parameter_partition={
                group.name: sum(parameter.numel() for parameter in group.parameters)
                for group in method_parameter_groups
            },
            source_artifact=source_identity,
            worktree_dirty=dirty,
            worktree_fingerprint=fingerprint,
        )

    def trainer_state() -> dict[str, Any]:
        r"""返回完整 epoch 边界的 schedule/session/RNG 与训练预算状态。"""

        return {
            "sampling": _sampling_state(train_schedule, train_session),
            "completed_epochs": completed_epochs,
            "optimizer_update": optimizer_update,
            "new_pairs_seen": new_pairs_seen,
            "pair_uses": pair_uses,
            "teacher_pairs_realized": teacher_pairs_realized,
            "forward_index": forward_index,
            "teacher_baseline_statistics": baseline_statistics,
            "source_artifact_ref_count": source_ref_count,
            "source_artifact_ref_digest": source_ref_digest.hexdigest(),
            "source_artifact_ref_bytes": source_ref_byte_offset,
            "log_continuation_offsets": dict(log_continuation_offsets),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        }

    def save_checkpoint(path: Path) -> None:
        r"""保存 immutable 通用容器；Method 与 Trainer 分别提供自己的 state。"""

        if path.exists():
            raise FileExistsError(f"immutable epoch checkpoint already exists: {path}")
        run.save_full_checkpoint(
            path,
            method_state=method.training_state_dict(),
            optimizer_state=optimizer.state_dict(),
            epoch=completed_epochs,
            optimizer_update=optimizer_update,
            metadata=metadata(),
            trainer_state=trainer_state(),
        )

    def save_recovery() -> None:
        r"""原子覆盖 epoch-boundary recovery；immutable cycle checkpoints 仍由独立路径保存。"""

        run.save_full_checkpoint(
            automatic_recovery,
            method_state=method.training_state_dict(),
            optimizer_state=optimizer.state_dict(),
            epoch=completed_epochs,
            optimizer_update=optimizer_update,
            metadata=metadata(),
            trainer_state=trainer_state(),
        )

    if resume_path is not None:
        payload = load_pretrain_checkpoint(resume_path, map_location=device)
        method.load_training_state_dict(payload["method_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        completed_epochs = int(payload["epoch"])
        optimizer_update = int(payload["optimizer_update"])
        loaded_metadata = dict(payload["metadata"])
        if loaded_metadata.get("dataset_identity") != dataset_identity:
            raise ValueError("resume checkpoint dataset identity does not match resolved training asset axis")
        checkpoint_resolved = loaded_metadata.get("resolved_config")
        if not isinstance(checkpoint_resolved, dict):
            raise ValueError("resume checkpoint lacks resolved config")
        require_resume_scientific_config(resolved_config, checkpoint_resolved)
        require_resume_metadata_identity(
            asdict(metadata()),
            loaded_metadata,
            allow_worktree_change=bool(run.config.allow_worktree_change),
        )
        state = dict(payload["trainer_state"])
        for name in ("completed_epochs", "optimizer_update", "new_pairs_seen", "pair_uses", "teacher_pairs_realized"):
            value = state.get(name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"resume checkpoint lacks valid progress counter {name}")
        if int(state["completed_epochs"]) != completed_epochs or int(state["optimizer_update"]) != optimizer_update:
            raise ValueError("checkpoint top-level epoch/update disagree with trainer progress state")
        new_pairs_seen = int(state["new_pairs_seen"])
        pair_uses = int(state["pair_uses"])
        teacher_pairs_realized = int(state["teacher_pairs_realized"])
        raw_forward_index = state.get("forward_index")
        if not isinstance(raw_forward_index, int) or raw_forward_index < 0:
            raise ValueError("resume checkpoint lacks a valid global forward_index")
        forward_index = raw_forward_index
        raw_baseline_statistics = state.get("teacher_baseline_statistics")
        if raw_baseline_statistics is not None:
            if not isinstance(raw_baseline_statistics, Mapping) or not all(
                isinstance(value, torch.Tensor) for value in raw_baseline_statistics.values()
            ):
                raise ValueError("resume checkpoint has invalid teacher baseline sufficient statistics")
            baseline_statistics = {
                str(name): value.detach().cpu().to(torch.float64)
                for name, value in raw_baseline_statistics.items()
            }
        expected_ref_count = state.get("source_artifact_ref_count")
        expected_ref_digest = state.get("source_artifact_ref_digest")
        if not isinstance(expected_ref_count, int) or expected_ref_count < 0 or not isinstance(expected_ref_digest, str):
            raise ValueError("resume checkpoint lacks source artifact ref-log prefix identity")
        lineage_ref_path = resume_path.parent.parent / "source_artifacts.jsonl"
        lineage_lines = lineage_ref_path.read_bytes().splitlines(keepends=True) if expected_ref_count else []
        if len(lineage_lines) < expected_ref_count:
            raise ValueError("resume source artifact ref log is shorter than checkpoint prefix")
        prefix = b"".join(lineage_lines[:expected_ref_count])
        source_ref_digest.update(prefix)
        if source_ref_digest.hexdigest() != expected_ref_digest:
            raise ValueError("resume source artifact ref-log prefix digest mismatch")
        source_ref_count = expected_ref_count
        # 即使 prefix 为空也要截断旧的中断尾巴，否则后续 byte offset 会包含未被 digest 覆盖的行。
        source_ref_path.write_bytes(prefix)
        raw_ref_bytes = state.get("source_artifact_ref_bytes")
        if not isinstance(raw_ref_bytes, int) or raw_ref_bytes < 0 or len(prefix) != raw_ref_bytes:
            raise ValueError("resume checkpoint source ref byte offset disagrees with validated prefix")
        source_ref_byte_offset = raw_ref_bytes
        raw_log_offsets = state.get("log_continuation_offsets")
        if not isinstance(raw_log_offsets, Mapping):
            raise ValueError("resume checkpoint lacks logger continuation offsets")
        log_continuation_offsets = {str(name): int(value) for name, value in raw_log_offsets.items()}
        logger.restore_continuation(log_continuation_offsets, purge_step=optimizer_update + 1)
        sampling_state = state.get("sampling")
        if not isinstance(sampling_state, dict):
            raise ValueError("resume checkpoint lacks Trainer sampling state")
        _restore_sampling_state(sampling_state, train_schedule, train_session)
        torch_rng_state = state.get("torch_rng_state")
        cuda_rng_state = state.get("cuda_rng_state_all")
        if not isinstance(torch_rng_state, torch.Tensor):
            raise ValueError("resume checkpoint lacks torch RNG state")
        if not isinstance(cuda_rng_state, list) or not all(isinstance(item, torch.Tensor) for item in cuda_rng_state):
            raise ValueError("resume checkpoint lacks CUDA RNG states")
        torch.set_rng_state(torch_rng_state.cpu())
        # CUDA generator 的 state API 接收 CPU ByteTensor；checkpoint map_location=device 后需显式还原 host state。
        torch.cuda.set_rng_state_all([item.detach().cpu() for item in cuda_rng_state])

    try:
        if resume_path is None:
            save_checkpoint(output_dir / "checkpoints" / "epoch_000000.pt")

        while not train_schedule.complete:
            method.train_mode()
            epoch_index = train_schedule.completed_epochs
            if epoch_index != completed_epochs:
                raise RuntimeError("trainer epoch counter disagrees with sampling schedule")
            epoch_records: list[dict[str, Any]] = []
            epoch_snapshot_batch: Any | None = None

            # 追加 source refs 与 resident telemetry 的动作必须跟随每个 unit，而不是等完整 epoch。
            def drain_runtime_evidence() -> None:
                nonlocal source_ref_count

                drain_source_refs = getattr(method, "drain_source_artifact_references", None)
                if callable(drain_source_refs):
                    references = drain_source_refs()
                    if references:
                        if not isinstance(references, (tuple, list)):
                            raise TypeError("source artifact references must be a sequence")
                        with source_ref_path.open("ab") as stream:
                            for reference in references:
                                if not isinstance(reference, Mapping):
                                    raise TypeError("each source artifact reference must be a mapping")
                                line = json.dumps(reference, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
                                stream.write(line)
                                source_ref_digest.update(line)
                                source_ref_count += 1
                drain_runtime = getattr(train_session, "drain_runtime_events", None)
                if callable(drain_runtime):
                    runtime_events = drain_runtime()
                    if not isinstance(runtime_events, (tuple, list)):
                        raise TypeError("runtime events must be a sequence")
                    for event in runtime_events:
                        if not isinstance(event, Mapping):
                            raise TypeError("each runtime event must be a mapping")
                        logger.log_runtime_event(dict(event))

            def observe_teacher_unit(unit: Any) -> None:
                nonlocal baseline_statistics, teacher_pairs_realized, epoch_snapshot_batch

                drain_runtime_evidence()
                baseline_block = {
                    name: value.detach().cpu().to(torch.float64)
                    for name, value in method.teacher_baseline_statistics(unit).items()
                }
                baseline_statistics = method.merge_teacher_baseline_statistics(baseline_statistics, baseline_block)
                teacher_pairs_realized += int(unit.q.shape[0])
                if epoch_snapshot_batch is None:
                    epoch_snapshot_batch = unit

            def finish_update(
                update: Any,
                *,
                schedule_item: Any,
                mini_epoch_index: int,
                diagnostic_seconds: float,
                log_batch: Any,
                asset_ids: tuple[str, ...],
                update_started: float,
            ) -> None:
                nonlocal optimizer_update, new_pairs_seen, pair_uses

                from anymani.distill.methods.multi_anchor_gaussian_implicit_field.training import (
                    clip_parameter_groups,
                )

                gradient_groups = clip_parameter_groups(
                    method_parameter_groups,
                    max_norm=trainer.config.max_gradient_norm_per_group,
                )
                optimizer.step()
                optimizer_update += 1
                step_seconds = perf_counter() - update_started
                if mini_epoch_index == 0:
                    new_pairs_seen += int(update.sample_count)
                pair_uses += int(update.sample_count)
                record = {
                    "terms": update.terms,
                    "gradient_groups": {name: asdict(value) for name, value in gradient_groups.items()},
                    "gradient_evidence": update.gradient_evidence,
                    "diagnostic_seconds": diagnostic_seconds,
                    "diagnostics": update.diagnostics,
                }
                epoch_records.append(record)
                logger.log_terms(
                    optimizer_update=optimizer_update,
                    epoch=epoch_index + 1,
                    mini_epoch=mini_epoch_index,
                    minibatch_in_epoch=schedule_item.minibatch_index_in_epoch,
                    global_minibatch=schedule_item.minibatch_index,
                    new_pairs_seen=new_pairs_seen,
                    pair_uses=pair_uses,
                    teacher_pairs_realized=teacher_pairs_realized,
                    microbatches_consumed=int(update.sample_count) // trainer.config.microbatch_size,
                    wall_time_seconds=perf_counter() - lifecycle_started,
                    split="train",
                    terms=update.terms,
                    denominators=update.denominators,
                    asset_ids=asset_ids,
                    gradient_groups={name: asdict(value) for name, value in gradient_groups.items()},
                    batch=log_batch if hasattr(log_batch, "q_index") else None,
                    gradient_evidence=update.gradient_evidence,
                    diagnostic_seconds=diagnostic_seconds,
                    diagnostics=update.diagnostics,
                )
                logger.log_runtime_event(
                    {
                        "event": "optimizer_update",
                        "optimizer_update": optimizer_update,
                        "epoch": epoch_index + 1,
                        "step_seconds": step_seconds,
                        "q_samples_per_second": float(update.sample_count) / max(step_seconds, 1.0e-12),
                        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                        "z_gradient_diagnostic_seconds": diagnostic_seconds,
                    }
                )

            if trainer.config.mini_epochs == 1:
                # 正式 fast path：realize -> teacher -> backward 按 64-pair unit 立即消费，历史 unit 不驻留 GPU。
                for minibatch_index_in_epoch in range(trainer.config.num_minibatches):
                    schedule_item = train_schedule.next()
                    if (
                        schedule_item.epoch_index != epoch_index
                        or schedule_item.minibatch_index_in_epoch != minibatch_index_in_epoch
                    ):
                        raise RuntimeError("training schedule epoch/minibatch identity drifted")
                    raw_units = train_session.realize_units(
                        schedule_item,
                        schedule=train_schedule,
                        step=schedule_item.minibatch_index,
                    )
                    observed_units: list[Any] = []  # 只保存当前 yield 的 Python 引用；yield 后立即 pop
                    asset_ids: list[str] = []

                    def units():
                        r"""把 source refs/baseline 与同一 unit 的首次消费绑定。"""

                        for unit in raw_units:
                            observe_teacher_unit(unit)
                            asset_ids.extend(unit.asset_ids)
                            observed_units.append(unit)
                            yield unit
                            observed_units.pop()

                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.reset_peak_memory_stats(device)
                    collect_z_gradients = (epoch_index + 1) % 4 == 0 and minibatch_index_in_epoch == (
                        trainer.config.num_minibatches - 1
                    )
                    update_started = perf_counter()
                    diagnostic_started = perf_counter()
                    update = method.backward_update_units(
                        units(),
                        forward_step=forward_index,
                        logical_sample_count=schedule_item.sample_count,
                        microbatch_size=trainer.config.microbatch_size,
                        collect_z_gradients=collect_z_gradients,
                    )
                    diagnostic_seconds = perf_counter() - diagnostic_started if collect_z_gradients else 0.0
                    forward_index += 1
                    finish_update(
                        update,
                        schedule_item=schedule_item,
                        mini_epoch_index=0,
                        diagnostic_seconds=diagnostic_seconds,
                        log_batch=epoch_snapshot_batch,
                        asset_ids=tuple(asset_ids),
                        update_started=update_started,
                    )
            else:
                # 复用路径只在 pinned CPU 保存 detached teacher units；每次反传恢复一个 64-pair unit，
                # learned activation 始终按当前参数重算，不把完整 epoch 或 512-pair batch 常驻 GPU。
                replay_units: list[tuple[Any, ...]] = []
                replay_asset_ids: list[tuple[str, ...]] = []
                schedule_items: list[Any] = []
                for minibatch_index_in_epoch in range(trainer.config.num_minibatches):
                    schedule_item = train_schedule.next()
                    if (
                        schedule_item.epoch_index != epoch_index
                        or schedule_item.minibatch_index_in_epoch != minibatch_index_in_epoch
                    ):
                        raise RuntimeError("training schedule epoch/minibatch identity drifted")
                    raw_units = train_session.realize_units(
                        schedule_item,
                        schedule=train_schedule,
                        step=schedule_item.minibatch_index,
                    )
                    staged_units: list[Any] = []
                    asset_ids: list[str] = []
                    for unit in raw_units:
                        observe_teacher_unit(unit)
                        staged = method.stage_replay_unit(unit)
                        if epoch_snapshot_batch is unit:
                            epoch_snapshot_batch = staged  # checkpoint snapshot 不保留首个 CUDA teacher unit
                        staged_units.append(staged)
                        asset_ids.extend(unit.asset_ids)
                        del unit
                    if sum(int(unit.q.shape[0]) for unit in staged_units) != schedule_item.sample_count:
                        raise RuntimeError("pinned replay units do not cover the logical minibatch sample count")
                    replay_units.append(tuple(staged_units))
                    replay_asset_ids.append(tuple(asset_ids))
                    schedule_items.append(schedule_item)
                for mini_epoch_index in range(trainer.config.mini_epochs):
                    order = _mini_epoch_order(
                        len(replay_units),
                        seed=run.config.seed,
                        epoch_index=epoch_index,
                        mini_epoch_index=mini_epoch_index,
                    )
                    for order_position, buffer_index in enumerate(order):
                        schedule_item = schedule_items[buffer_index]
                        optimizer.zero_grad(set_to_none=True)
                        torch.cuda.reset_peak_memory_stats(device)
                        collect_z_gradients = (
                            (epoch_index + 1) % 4 == 0
                            and mini_epoch_index == trainer.config.mini_epochs - 1
                            and order_position == len(order) - 1
                        )
                        update_started = perf_counter()
                        diagnostic_started = perf_counter()

                        def restored_units():
                            r"""按 replay 顺序一次恢复一个 opaque unit；yield 后当前 CUDA unit 可立即释放。"""

                            for staged in replay_units[buffer_index]:
                                yield method.restore_replay_unit(staged, device=device)

                        update = method.backward_update_units(
                            restored_units(),
                            forward_step=forward_index,
                            logical_sample_count=schedule_item.sample_count,
                            microbatch_size=trainer.config.microbatch_size,
                            collect_z_gradients=collect_z_gradients,
                        )
                        diagnostic_seconds = perf_counter() - diagnostic_started if collect_z_gradients else 0.0
                        forward_index += 1
                        finish_update(
                            update,
                            schedule_item=schedule_item,
                            mini_epoch_index=mini_epoch_index,
                            diagnostic_seconds=diagnostic_seconds,
                            log_batch=replay_units[buffer_index][0],
                            asset_ids=replay_asset_ids[buffer_index],
                            update_started=update_started,
                        )
                del replay_units, replay_asset_ids

            completed_epochs = epoch_index + 1
            epoch_terms = {
                name: sum(float(record["terms"][name]) for record in epoch_records) / len(epoch_records)
                for name in declared_weights
            }
            epoch_payload = {
                **{f"raw/{name}": value for name, value in epoch_terms.items()},
            }  # TensorBoard、终端与 JSONL update 均由同一批 update facts 聚合
            logger.log_epoch_terms(
                epoch=completed_epochs,
                new_pairs_seen=new_pairs_seen,
                pair_uses=pair_uses,
                optimizer_updates=optimizer_update,
                terms=epoch_payload,
            )
            print(
                f"[SSL] epoch {completed_epochs:03d}/{trainer.config.max_epochs} "
                f"{_format_objective_terms(epoch_terms)} "
                f"updates={optimizer_update} pairs={new_pairs_seen}",
                flush=True,
            )
            checkpoint_due = completed_epochs % trainer.config.checkpoint_every_epochs == 0 or train_schedule.complete
            if checkpoint_due:
                dense_snapshot = getattr(method, "dense_snapshot", None)
                if callable(dense_snapshot) and epoch_snapshot_batch is not None:
                    snapshot_input = epoch_snapshot_batch
                    if trainer.config.mini_epochs > 1:
                        snapshot_input = method.restore_replay_unit(snapshot_input, device=device)
                    snapshot_result: Any = dense_snapshot(
                        snapshot_input,
                        microbatch_size=trainer.config.microbatch_size,
                    )
                    if not isinstance(snapshot_result, tuple) or len(snapshot_result) != 2:
                        raise TypeError("method dense_snapshot() must return (prediction,batch)")
                    snapshot_prediction, snapshot_batch = snapshot_result
                    snapshot_path = logger.save_dense_snapshot(
                        optimizer_update=optimizer_update,
                        split="train",
                        prediction=snapshot_prediction,
                        batch=snapshot_batch,
                    )
                    logger.log_runtime_event(
                        {
                            "event": "dense_snapshot",
                            "optimizer_update": optimizer_update,
                            "relative_path": str(snapshot_path.relative_to(output_dir)),
                            "sample_count": len(snapshot_batch.asset_ids),
                        }
                    )
                    del snapshot_input, snapshot_prediction, snapshot_batch
                save_checkpoint(output_dir / "checkpoints" / f"epoch_{completed_epochs:06d}.pt")
                print(
                    "[SSL checkpoint]\n"
                    f"  epoch={completed_epochs} optimizer_updates={optimizer_update}\n"
                    f"  {_format_objective_terms(epoch_terms)}\n"
                    f"  new_pairs_seen={new_pairs_seen} pair_uses={pair_uses}",
                    flush=True,
                )

            # Recovery barrier 位于完整 epoch 的日志与可选 immutable checkpoint 之后；最多重做一个 epoch。
            log_continuation_offsets = logger.continuation_offsets()
            source_ref_byte_offset = source_ref_path.stat().st_size if source_ref_path.is_file() else 0
            save_recovery()

        final_checkpoint = output_dir / "checkpoints" / f"epoch_{completed_epochs:06d}.pt"
        if not final_checkpoint.is_file():
            save_checkpoint(final_checkpoint)
        publish_checkpoint_alias(output_dir / "checkpoints" / "last.pt", final_checkpoint)
        if baseline_statistics is None:
            raise RuntimeError("training completed without run-local teacher baseline statistics")
        teacher_baselines = method.finalize_teacher_baselines(baseline_statistics)
        _write_yaml(
            output_dir / "run_teacher_baselines.yaml",
            {
                "schema_version": "1.0.0",
                "source": "complete_run_teacher_distribution",
                "formula_identity": method.formula_identity(),
                "optimizer_updates": optimizer_update,
                "teacher_pairs_realized": teacher_pairs_realized,
                "teacher_baselines": teacher_baselines,
            },
        )
        logger.finalize_training_metrics(
            teacher_baselines=teacher_baselines,
            expected_optimizer_updates=optimizer_update,
            lineage_metrics_path=(
                resume_path.parent.parent / "metrics.jsonl"
                if resume_path is not None
                else None
            ),
        )
        _write_yaml(
            output_dir / "training_summary.yaml",
            {
                "schema_version": "1.0.0",
                "completed_epochs": completed_epochs,
                "optimizer_updates": optimizer_update,
                "new_pairs_seen": new_pairs_seen,
                "pair_uses": pair_uses,
                "teacher_pairs_realized": teacher_pairs_realized,
                "teacher_baselines": teacher_baselines,
                "final_checkpoint": str(final_checkpoint),
            },
        )
        if trainer.config.emit_compression_basis:
            _publish_train_compression_basis(
                method=method,
                train_session=train_session,
                device=device,
                dtype=teacher_dtype,
                config=trainer.config,
                seed=run.config.seed,
                output_dir=output_dir,
            )
        retained_payload = method.retained_artifact_payload(
            metadata=asdict(metadata()),
            source_checkpoint=final_checkpoint,
        )
        run.save_retained_artifact(output_dir / "retained_encoder.pt", retained_payload)
        automatic_recovery.unlink(missing_ok=True)
        (output_dir / "INCOMPLETE").unlink(missing_ok=True)
        completion_temporary = output_dir / "COMPLETE.tmp"
        completion_temporary.write_text(
            f"schema=9.0.0\nepoch={completed_epochs}\noptimizer_update={optimizer_update}\n",
            encoding="ascii",
        )
        completion_temporary.replace(output_dir / "COMPLETE")
        return output_dir
    finally:
        train_session.close()
        drain_runtime = getattr(train_session, "drain_runtime_events", None)
        if callable(drain_runtime):
            runtime_events = drain_runtime()
            if not isinstance(runtime_events, (tuple, list)):
                raise TypeError("runtime events must be a sequence")
            for event in runtime_events:
                if not isinstance(event, Mapping):
                    raise TypeError("each runtime event must be a mapping")
                logger.log_runtime_event(dict(event))
        write_resource_evidence()
        method.close()
        logger.close()


__all__ = ["fit_embodiment_pretrain"]
