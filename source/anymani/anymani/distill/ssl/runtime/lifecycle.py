r"""Schema 8 online procedural supervised pure-pretraining lifecycle.

该模块是最高级训练内核：Data runtime 解析 catalog，Method session 封闭产生 batch，Trainer 拥有
epoch/minibatch schedule、分组 optimizer、backward 与 full checkpoint。Validation/evaluation 使用独立入口，
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
from typing import Any

import torch
import yaml

from anymani.assets.asset_schema_geometry import SEMANTICS_SCHEMA_VERSION
from anymani.distill.diagnostics.recording.geometry_ssl import GeometrySSLRunLogger
from anymani.distill.ssl.checkpoint import load_pretrain_checkpoint
from anymani.distill.ssl.runtime.checkpointing import (
    publish_checkpoint_alias,
    require_resume_metadata_identity,
    require_resume_scientific_config,
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
        raise ValueError("checkpoint lacks schema 8 epoch/minibatch state")
    schedule.load_state_dict(raw_schedule)
    raw_session = payload.get("method_session")
    if not isinstance(raw_session, dict):
        raise ValueError("checkpoint lacks method session state")
    session.load_state_dict(raw_session)


def _declared_objective_weights(method: Any) -> dict[str, float]:
    r"""读取 method 显式声明的 rho/kappa 权重；baseline 归一化不改写该声明。"""

    return dict(method.declared_objective_weights())


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
    r"""执行 schema-8 pure-pretrain phase，并在同一 run 内累计 teacher baseline。

    ``pretrain`` 只产生在线监督、执行参数更新、记录训练曲线并保存 full checkpoint。固定 q-bank、
    validation、held-out evaluation、best selection、physical audit 与 retained export 均不属于该进程。
    """

    from anymani.distill.ssl.runtime.scheduler import ResidentGeometryAssetWindow

    lifecycle_started = perf_counter()  # catalog/method/model/session 的统一 wall-time 原点
    runtime_timing: dict[str, float] = {}  # 只写 runtime_resources，不进入科学 artifact identity
    print("[SSL] Schema 8 pure pretraining")
    if run.config.deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(bool(run.config.deterministic_algorithms))
    torch.manual_seed(run.config.seed)
    device = torch.device(trainer.config.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"configured CUDA device is unavailable: {device}")
    dtype = _torch_dtype(trainer.config.dtype)
    print("[SSL] Preparing output directory...")
    output_dir = run.prepare_output_dir(output_dir_override)
    print(f"[SSL] Output: {output_dir}")
    logger = GeometrySSLRunLogger(output_dir)
    print("[SSL] Resolving asset catalog (this may take 1-2 minutes for 8k assets)...")
    stage_started = perf_counter()
    catalog = data.resolve()
    runtime_timing["catalog_resolve_seconds"] = perf_counter() - stage_started
    if hasattr(catalog, "train"):
        train_count = len(catalog.train)
        validation_count = sum(len(partition) for partition in catalog.validation.values())
    else:
        partitions = getattr(catalog.dataset, "partitions", None)
        if isinstance(partitions, dict):
            train_count = len(partitions.get("train", ()))
            validation_count = len(partitions.get("validation", ()))
        else:
            train_count = "synthetic"
            validation_count = "synthetic"
    print(
        f"[SSL] Catalog resolved: train={train_count} validation={validation_count} assets "
        f"in {runtime_timing['catalog_resolve_seconds']:.2f}s"
    )
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
        )
    method.prepare(catalog, device=device, dtype=dtype)
    preflight_source_artifacts = getattr(method, "preflight_source_artifacts", None)
    prepare_source_artifacts = getattr(method, "prepare_source_artifacts", None)
    if source_mode == "auto" and callable(preflight_source_artifacts):
        source_ready = True
        try:
            preflight = preflight_source_artifacts()
        except (FileNotFoundError, ValueError):
            source_ready = False
            if not callable(prepare_source_artifacts):
                raise TypeError("auto source cache mode requires prepare_source_artifacts()")
            print("[SSL] Geometry Source Artifact Cache is incomplete; building missing artifacts...")
            source_started = perf_counter()
            preparation = prepare_source_artifacts(device=device, dtype=dtype)
            if not isinstance(preparation, Mapping):
                raise TypeError("source artifact preparation must return a mapping")
            runtime_timing["source_prepare_seconds"] = perf_counter() - source_started
            print(f"[SSL] Source artifacts prepared in {runtime_timing['source_prepare_seconds']:.2f}s")
        if not source_ready:
            preflight = preflight_source_artifacts()
        if not isinstance(preflight, Mapping):
            raise TypeError("source artifact preflight must return a mapping")
        runtime_timing.update({f"source_preflight_{name}": float(value) for name, value in preflight.items()})
        if callable(configure_source_artifacts):
            configure_source_artifacts(
                root=run.config.source_cache_root,
                mode="readonly",
                dataset_manifest_sha256=str(catalog.dataset.source_sha256),
                producer_device=str(device),
            )
    elif callable(preflight_source_artifacts) and source_mode == "readonly":
        preflight = preflight_source_artifacts()
        if not isinstance(preflight, Mapping):
            raise TypeError("source artifact preflight must return a mapping")
        runtime_timing.update({f"source_preflight_{name}": float(value) for name, value in preflight.items()})
    runtime_timing["method_prepare_seconds"] = perf_counter() - stage_started
    print(f"[SSL] Method prepared in {runtime_timing['method_prepare_seconds']:.2f}s")
    print(f"[SSL] Initializing model on {device}...")
    stage_started = perf_counter()
    method.initialize_model(device=device, dtype=dtype)
    runtime_timing["model_initialize_seconds"] = perf_counter() - stage_started
    print(f"[SSL] Model initialized in {runtime_timing['model_initialize_seconds']:.2f}s")
    _write_yaml(output_dir / "resolved_config.yaml", resolved_config)
    _write_yaml(output_dir / "asset_dataset.yaml", catalog.dataset.config_dict())
    dirty, fingerprint = _worktree_fingerprint()
    declared_weights = _declared_objective_weights(method)
    identity_builder = getattr(catalog, "training_dataset_identity", None)
    if not callable(identity_builder):
        raise TypeError("resolved catalog must expose training_dataset_identity() for schema 8 checkpoints")
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
            dtype=dtype,
            max_resident_assets=trainer.config.max_resident_assets,
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
        max_resident_assets=trainer.config.max_resident_assets,
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
    resume_path = Path(run.config.resume_checkpoint).expanduser().resolve() if run.config.resume_checkpoint else None
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
        require_resume_metadata_identity(asdict(metadata()), loaded_metadata)
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
        if prefix:
            source_ref_path.write_bytes(prefix)
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
        torch.cuda.set_rng_state_all(cuda_rng_state)

    try:
        if resume_path is None:
            save_checkpoint(output_dir / "checkpoints" / "epoch_000000.pt")

        while not train_schedule.complete:
            method.train_mode()
            epoch_index = train_schedule.completed_epochs
            if epoch_index != completed_epochs:
                raise RuntimeError("trainer epoch counter disagrees with sampling schedule")
            # 本 epoch 的全部新 teacher minibatches 先完成 realization，mini-epoch 只重排并复用该 buffer。
            batches: list[Any] = []
            schedule_items: list[Any] = []
            for minibatch_index_in_epoch in range(trainer.config.num_minibatches):
                schedule_item = train_schedule.next()
                if (
                    schedule_item.epoch_index != epoch_index
                    or schedule_item.minibatch_index_in_epoch != minibatch_index_in_epoch
                ):
                    raise RuntimeError("training schedule epoch/minibatch identity drifted")
                schedule_items.append(schedule_item)
                realized_batch = _build_batch(
                        schedule_item,
                        session=train_session,
                        schedule=train_schedule,
                        step=schedule_item.minibatch_index,
                    )
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
                baseline_block = {
                    name: value.detach().cpu().to(torch.float64)
                    for name, value in method.teacher_baseline_statistics(realized_batch).items()
                }
                baseline_statistics = method.merge_teacher_baseline_statistics(
                    baseline_statistics,
                    baseline_block,
                )  # mini-epoch 后续复用不重复计入 teacher distribution
                batches.append(realized_batch)
                teacher_pairs_realized += schedule_item.sample_count
            epoch_records: list[dict[str, Any]] = []
            for mini_epoch_index in range(trainer.config.mini_epochs):
                order = _mini_epoch_order(
                    len(batches),
                    seed=run.config.seed,
                    epoch_index=epoch_index,
                    mini_epoch_index=mini_epoch_index,
                )
                for order_position, buffer_index in enumerate(order):
                    update_started = perf_counter()  # 覆盖 zero_grad、forward/backward、clip 与 optimizer.step
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    batch = batches[buffer_index]
                    schedule_item = schedule_items[buffer_index]
                    optimizer.zero_grad(set_to_none=True)
                    update: Any
                    diagnostic_seconds = 0.0  # 非 cadence update 不承担 proxy 开销
                    backward_update = getattr(method, "backward_update", None)
                    if callable(backward_update):
                        collect_z_gradients = (
                            (epoch_index + 1) % 4 == 0
                            and mini_epoch_index == trainer.config.mini_epochs - 1
                            and order_position == len(order) - 1
                        )  # 每 4 epochs 只采最后一个 update，且必须发生在 optimizer.step 前
                        diagnostic_started = perf_counter()
                        update = backward_update(
                            batch,
                            forward_step=forward_index,
                            microbatch_size=trainer.config.microbatch_size,
                            collect_z_gradients=collect_z_gradients,
                        )
                        diagnostic_seconds = perf_counter() - diagnostic_started if collect_z_gradients else 0.0
                        forward_index += 1
                    else:
                        raise TypeError("schema-8 trainer requires Method.backward_update() for separate task gradients")
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
                    microbatches_consumed = int(update.sample_count) // trainer.config.microbatch_size
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
                        microbatches_consumed=microbatches_consumed,
                        wall_time_seconds=perf_counter() - lifecycle_started,
                        split="train",
                        terms=update.terms,
                        denominators=update.denominators,
                        asset_ids=tuple(getattr(batch, "asset_ids", ())),
                        gradient_groups={name: asdict(value) for name, value in gradient_groups.items()},
                        batch=batch if hasattr(batch, "q_index") else None,
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
                            "cuda_peak_allocated_bytes": (
                                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
                            ),
                            "cuda_peak_reserved_bytes": (
                                int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
                            ),
                            "z_gradient_diagnostic_seconds": diagnostic_seconds,
                        }
                    )

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
                f"rho_mse={epoch_terms['density']:.4e} kappa_scaled_mse={epoch_terms['kappa']:.4e} "
                f"updates={optimizer_update} pairs={new_pairs_seen}",
                flush=True,
            )
            checkpoint_due = completed_epochs % trainer.config.checkpoint_every_epochs == 0 or train_schedule.complete
            if checkpoint_due:
                dense_snapshot = getattr(method, "dense_snapshot", None)
                if callable(dense_snapshot):
                    snapshot_result: Any = dense_snapshot(
                        batches[0],
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
                del batches
                save_checkpoint(output_dir / "checkpoints" / f"epoch_{completed_epochs:06d}.pt")
                print(
                    "[SSL checkpoint]\n"
                    f"  epoch={completed_epochs} optimizer_updates={optimizer_update}\n"
                    f"  density_mse={epoch_terms['density']:.6e} "
                    f"kappa_scaled_mse={epoch_terms['kappa']:.6e}\n"
                    f"  new_pairs_seen={new_pairs_seen} pair_uses={pair_uses}",
                    flush=True,
                )
            else:
                del batches

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
