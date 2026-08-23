r"""Schema 5 online procedural supervised pretraining lifecycle.

该模块是最高级训练内核：Data runtime 解析 catalog，Method session 封闭产生 batch、评估与 artifact，
Trainer 拥有 phase、window-major schedule、backward、validation promotion 与 final evaluation。
生命周期不读取 representation 内部字段，也不解释 owner/query/edge 轴。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from anymani.assets.asset_schema_geometry import SEMANTICS_SCHEMA_VERSION
from anymani.distill.ssl.checkpoint import load_pretrain_checkpoint
from anymani.distill.ssl.runtime.checkpointing import (
    publish_best_checkpoint,
    require_resume_scientific_config,
    restore_validation_selection_state,
)
from anymani.distill.ssl.runtime.run import PretrainRun
from anymani.distill.ssl.runtime.sampling import FixedAssetQSchedule, OnlineMinibatchSchedule


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
        raise ValueError("checkpoint lacks schema 5 online minibatch state")
    schedule.load_state_dict(raw_schedule)
    raw_session = payload.get("method_session")
    if not isinstance(raw_session, dict):
        raise ValueError("checkpoint lacks method session state")
    session.load_state_dict(raw_session)


def _declared_objective_weights(method: Any) -> dict[str, float]:
    r"""读取 method 显式声明的五项权重，不经过自动梯度标定。"""

    return dict(method.declared_objective_weights())


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
        "num_minibatches": trainer.get("num_minibatches"),
        "mini_epochs": trainer.get("mini_epochs"),
        "gradient_accumulation_steps": trainer.get("gradient_accumulation_steps"),
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


def _write_calibration_artifact(
    method: Any,
    session: Any,
    schedule: OnlineMinibatchSchedule,
    output: Path,
    *,
    mini_epochs: int,
    gradient_accumulation_steps: int,
    manifest_hash: str,
    resolved_config: dict[str, Any] | None = None,
    after_first_forward: Callable[[], object] | None = None,
    before_write: Callable[[], object] | None = None,
) -> str:
    r"""按正式训练的数据复用顺序运行预实验，不 backward、不更新参数或权重。

    每组只 realization 一次新 minibatch 数据，再循环 ``mini_epochs`` 次 forward。由此产物同时区分
    新生成样本数与循环利用后的样本前向次数，避免把数据规模和训练计算量混为一谈。
    """

    traces: dict[str, list[float]] = {name: [] for name in _declared_objective_weights(method)}
    totals: dict[str, list[float]] = {}
    minibatch_count = 0  # 已生成的新 minibatch 数
    forward_count = 0  # 含 mini-epoch 复用的模型前向次数
    new_sample_count = 0  # 互不重复 realization 的 $(asset,q)$ 数
    forward_sample_count = 0  # 含复用的累计 $(asset,q)$ 前向次数
    asset_indices_seen: set[int] = set()  # 本次预实验实际接触的互异训练资产
    audit_started = False  # 首个 forward 完成后才允许后台物理审计争抢 CPU
    method.train_mode()  # 保留正式训练的 dropout/normalization 行为，只取消参数更新
    total_minibatches = schedule.num_minibatches
    partial_output = output.with_name(f"{output.stem}.partial{output.suffix}")  # 中断后最近完整 group 证据
    print(f"[Calibration] Starting: {total_minibatches} minibatches, "
          f"{mini_epochs} mini-epochs each, {gradient_accumulation_steps}-batch groups")
    with torch.enable_grad():
        while not schedule.complete:
            group_size = min(gradient_accumulation_steps, schedule.minibatches_remaining)
            batches: list[Any] = []  # 当前组的物理 teacher realization，只构造一次
            for _ in range(group_size):
                schedule_item = schedule.next()
                batches.append(
                    _build_batch(
                        session=session,
                        schedule=schedule,
                        schedule_item=schedule_item,
                        step=minibatch_count,
                    )
                )
                minibatch_count += 1
                new_sample_count += schedule_item.sample_count
                asset_indices_seen.update(schedule_item.asset_indices)
            # 同一 q/query/teacher batch 循环利用；forward_index 使每遍重新抽 joint-sign rewrite。
            for mini_epoch_index in range(mini_epochs):
                for batch_idx, batch in enumerate(batches):
                    result = method.forward_objectives(batch, step=forward_count, mode="calibration")
                    forward_count += 1
                    forward_sample_count += int(result.sample_count)
                    if forward_count == 1 and after_first_forward is not None and not audit_started:
                        after_first_forward()
                        audit_started = True
                    for name, objective in result.objectives.items():
                        for component in objective.components:
                            current = totals.setdefault(component.name, [0.0, 0.0])
                            current[0] += float(component.numerator.detach())
                            current[1] += float(component.denominator.detach())
                        traces.setdefault(name, []).append(float(objective.metrics["loss"].detach()))
                    # 每完成一个 mini-epoch 组打印一次进度
                    if batch_idx == len(batches) - 1:
                        progress_pct = 100.0 * minibatch_count / total_minibatches
                        print(f"[Calibration] Progress: {minibatch_count}/{total_minibatches} minibatches "
                              f"({progress_pct:.1f}%), mini-epoch {mini_epoch_index + 1}/{mini_epochs}, "
                               f"{forward_count} forward passes completed")
            # 一个 accumulation group 的全部 mini-epoch 均完成后才发布 partial，避免记录半组统计。
            _write_yaml(
                partial_output,
                {
                    "schema_version": "5.0.0",
                    "status": "in_progress",
                    "source": "formal_train_forward_preflight",
                    "dataset_source_sha256": manifest_hash,
                    "execution": {
                        "asset_count": session.asset_count,
                        "distinct_asset_count": len(asset_indices_seen),
                        "asset_use_count": minibatch_count * schedule.config.assets_per_minibatch,
                        "new_sample_count": new_sample_count,
                        "forward_sample_count": forward_sample_count,
                        "minibatch_count": minibatch_count,
                        "forward_count": forward_count,
                        "mini_epochs": mini_epochs,
                        "gradient_accumulation_steps": gradient_accumulation_steps,
                        "sampling": asdict(schedule.config),
                    },
                    "formula_identity": dict(method.formula_identity()),
                    "term_means": {
                        name: numerator / denominator
                        for name, (numerator, denominator) in totals.items()
                    },
                    "term_traces": traces,
                },
            )
    if minibatch_count < 1:
        raise ValueError("objective calibration requires at least one generated train minibatch")
    print(f"[Calibration] Completed: {minibatch_count} minibatches, {forward_count} forward passes, "
          f"{len(asset_indices_seen)} unique assets sampled")
    print("[Calibration] Writing artifact...")
    if before_write is not None:
        before_write()  # 完整 physical isolation audit 必须先通过，再发布 calibration artifact
    formula_identity = dict(method.formula_identity())
    recorded_config = dict(resolved_config or {})
    worktree_dirty, worktree_fingerprint = _worktree_fingerprint()
    payload = {
        "schema_version": "5.0.0",
        "status": "complete",
        "source": "formal_train_forward_preflight",
        "execution": {
            "asset_count": session.asset_count,
            "distinct_asset_count": len(asset_indices_seen),
            "asset_use_count": minibatch_count * schedule.config.assets_per_minibatch,
            "new_sample_count": new_sample_count,
            "forward_sample_count": forward_sample_count,
            "minibatch_count": minibatch_count,
            "forward_count": forward_count,
            "mini_epochs": mini_epochs,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "sampling": asdict(schedule.config),
        },
        "dataset_source_sha256": manifest_hash,
        "declared_objective": _declared_objective_weights(method),
        "formula_identity": formula_identity,
        "method_type": f"{type(method).__module__}.{type(method).__qualname__}",
        "code_revision": PretrainRun.code_revision(),
        "worktree_dirty": worktree_dirty,
        "worktree_fingerprint": worktree_fingerprint,
        "resolved_config": recorded_config,
        "scientific_identity": (
            _scientific_pretrain_identity(recorded_config, formula_identity=formula_identity)
            if recorded_config
            else {}
        ),
        "term_means": {
            name: numerator / denominator for name, (numerator, denominator) in totals.items()
        },
        "term_traces": traces,
    }
    _write_yaml(output, payload)
    partial_output.unlink(missing_ok=True)  # 最终 artifact 已原子发布，中间态不再具有独立语义
    artifact_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"[Calibration] Artifact written: {output}")
    print("[Calibration] Loss scale summary:")
    for name, mean_value in sorted(payload["term_means"].items()):
        print(f"  {name}: {mean_value:.6e}")
    return artifact_hash


def _require_calibration_identity(
    artifact: Path,
    *,
    method: Any,
    manifest_hash: str,
) -> str:
    r"""核对预实验产物的数据集、公式、方法类型和代码身份，不强制两个 preset 相同。"""

    payload = yaml.safe_load(artifact.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("calibration artifact must be a mapping")
    if payload.get("schema_version") != "5.0.0":
        raise ValueError("calibration artifact schema must be 5.0.0")
    if payload.get("status") != "complete":
        raise ValueError("calibration artifact must have status='complete'")
    if payload.get("dataset_source_sha256") != manifest_hash:
        raise ValueError("calibration artifact dataset hash does not match the formal ssl.yaml")
    expected_formula = dict(method.formula_identity())
    if not expected_formula:
        raise ValueError("current method lacks objective formula identity")
    recorded_formula = payload.get("formula_identity")
    if not isinstance(recorded_formula, dict) or recorded_formula != expected_formula:
        raise ValueError("calibration artifact objective formula identity does not match current method")
    expected_method_type = f"{type(method).__module__}.{type(method).__qualname__}"
    if payload.get("method_type") != expected_method_type:
        raise ValueError("calibration artifact method type does not match current method")
    recorded_revision = payload.get("code_revision")
    if not isinstance(recorded_revision, str) or not recorded_revision:
        raise ValueError("calibration artifact lacks code revision provenance")
    recorded_dirty = payload.get("worktree_dirty")
    recorded_fingerprint = payload.get("worktree_fingerprint")
    if not isinstance(recorded_dirty, bool) or not isinstance(recorded_fingerprint, str):
        raise ValueError("calibration artifact lacks worktree provenance")
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
    run: Any,
    output_dir_override: Path | None,
    resolved_config: dict[str, Any],
) -> Path:
    r"""执行显式 calibration/pretrain phase，并由 Trainer 统筹 validation 与冻结后 evaluation。"""

    from anymani.distill.ssl.runtime.scheduler import ResidentGeometryAssetWindow

    print(f"[SSL] Phase: {run.config.phase}")
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
    print("[SSL] Resolving asset catalog (this may take 1-2 minutes for 8k assets)...")
    catalog = data.resolve()
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
    print(f"[SSL] Catalog resolved: train={train_count} validation={validation_count} assets")
    print("[SSL] Preparing method (computing FK/Jacobian templates)...")
    method.prepare(catalog, device=device, dtype=dtype)
    print(f"[SSL] Initializing model on {device}...")
    method.initialize_model(device=device, dtype=dtype)
    audit_handle: Any | None = None
    audit_supported = hasattr(method, "start_physical_audit")
    manifest: dict[str, Any] | None = None

    def start_audit() -> None:
        r"""在首个 forward/首个训练组之后启动后台完整 physical audit。"""

        nonlocal audit_handle
        if audit_handle is None and audit_supported:
            print("[SSL] Starting background physical asset audit...")
            audit_handle = method.start_physical_audit(catalog)

    def await_manifest() -> dict[str, Any]:
        r"""等待完整 physical manifest，并在首次通过后写出唯一审计文件。"""

        nonlocal manifest
        if manifest is None:
            start_audit()
            print("[SSL] Awaiting complete physical asset audit...")
            manifest = audit_handle.wait() if audit_handle is not None else method.asset_manifest(catalog)
            _write_yaml(output_dir / "asset_manifest.yaml", manifest)
            print("[SSL] Physical asset audit passed")
        if manifest is None:
            raise RuntimeError("physical asset audit completed without a manifest")
        return manifest

    _write_yaml(output_dir / "resolved_config.yaml", resolved_config)
    _write_yaml(output_dir / "asset_dataset.yaml", catalog.dataset.config_dict())
    dirty, fingerprint = _worktree_fingerprint()
    declared_weights = _declared_objective_weights(method)

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
        )

    def write_resource_evidence() -> None:
        r"""在 arena clear 前记录 Method 暴露的有界资源事实；不要求所有 Method 实现该可选诊断。"""

        evidence = getattr(method, "runtime_resource_evidence", None)
        if callable(evidence):
            _write_yaml(output_dir / "runtime_resources.yaml", evidence())

    def cancel_unpublished_audit() -> None:
        r"""异常或提前返回时停止尚未写入 ``asset_manifest.yaml`` 的后台 audit。"""

        if manifest is None and audit_handle is not None:
            cancel = getattr(audit_handle, "cancel", None)
            if callable(cancel):
                cancel()

    def run_suites(role: str, config: Any, *, include_ablations: bool) -> dict[str, Any]:
        r"""在每条具名 suite 上重建固定 bank；空 suite 显式报告，不伪造成功。"""

        reports: dict[str, Any] = {}
        base_offset = config.seed_offset if role == "validation" else config.evaluation_seed_offset
        for suite_index, suite_name in enumerate(method.split_names(role)):
            asset_count = method.split_asset_count(role, suite=suite_name)
            if asset_count == 0:
                reports[suite_name] = {"status": "empty", "asset_count": 0}
                continue
            seed = run.config.seed + base_offset + suite_index * 1_000_003
            session = open_session(role, suite=suite_name, seed=seed)
            schedule = FixedAssetQSchedule(
                session.asset_count,
                q_per_asset=config.q_per_asset,
                assets_per_minibatch=config.assets_per_minibatch,
                q_per_asset_per_minibatch=config.q_per_asset_per_minibatch,
                max_resident_assets=trainer.config.max_resident_assets,
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

    def run_training_q_bank() -> Any:
        r"""在训练 morphology 上从独立 cursor 0 重放同一固定 Method 测度。"""

        config = trainer.config.final_evaluation
        seed = run.config.seed + config.training_q_bank_seed_offset
        session = open_session("training_evaluation", seed=seed)
        schedule = FixedAssetQSchedule(
            session.asset_count,
            q_per_asset=config.q_per_asset,
            assets_per_minibatch=config.assets_per_minibatch,
            q_per_asset_per_minibatch=config.q_per_asset_per_minibatch,
            max_resident_assets=trainer.config.max_resident_assets,
        )
        try:
            return method.evaluate_session(session, schedule, include_ablations=False)
        finally:
            session.close()

    print("[SSL] Opening training session (train partition)...")
    train_session = open_session("train", seed=trainer.config.sampling.seed)
    print(f"[SSL] Training session: {train_session.asset_count} assets")
    if run.config.phase == "calibrate_objectives":
        if run.config.resume_checkpoint:
            raise ValueError("calibrate_objectives does not resume an optimizer checkpoint")
        schedule = OnlineMinibatchSchedule(
            train_session.asset_count,
            trainer.config.sampling,
            num_minibatches=trainer.config.num_minibatches,
            max_resident_assets=trainer.config.max_resident_assets,
        )
        print(f"[SSL] Starting calibration: {trainer.config.num_minibatches} minibatches × "
              f"{trainer.config.mini_epochs} mini-epochs = "
              f"{trainer.config.num_minibatches * trainer.config.mini_epochs} forward passes")
        try:
            _write_calibration_artifact(
                method,
                train_session,
                schedule,
                output_dir / "loss_calibration.yaml",
                mini_epochs=trainer.config.mini_epochs,
                gradient_accumulation_steps=trainer.config.gradient_accumulation_steps,
                manifest_hash=str(catalog.dataset.source_sha256),
                resolved_config=resolved_config,
            )
            return output_dir
        finally:
            train_session.close()
            write_resource_evidence()
            cancel_unpublished_audit()
            method.close()

    calibration_hash = ""  # 正式训练可独立运行；提供预实验产物时只记录可审计 lineage
    if run.config.calibration_artifact:
        calibration_path = Path(run.config.calibration_artifact).expanduser()
        if not calibration_path.is_file():
            raise ValueError("run.calibration_artifact does not point to an existing loss_calibration.yaml")
        calibration_hash = _require_calibration_identity(
            calibration_path,
            method=method,
            manifest_hash=str(catalog.dataset.source_sha256),
        )
    train_schedule = OnlineMinibatchSchedule(
        train_session.asset_count,
        trainer.config.sampling,
        num_minibatches=trainer.config.num_minibatches,
        max_resident_assets=trainer.config.max_resident_assets,
    )
    update_groups = math.ceil(trainer.config.num_minibatches / trainer.config.gradient_accumulation_steps)
    required_updates = update_groups * trainer.config.mini_epochs
    if trainer.config.run_safety_step_limit < required_updates:
        raise ValueError(
            f"run_safety_step_limit={trainer.config.run_safety_step_limit} cannot cover "
            f"required optimizer updates={required_updates}"
        )
    optimizer = torch.optim.AdamW(
        method.parameters(),
        lr=trainer.config.optimizer.learning_rate,
        weight_decay=trainer.config.optimizer.weight_decay,
    )
    initial_validation: dict[str, dict[str, float]] | None = None
    best_score = float("inf")
    selection_history: list[dict[str, Any]] = []
    step = 0
    forward_index = 0  # 含 mini-epoch 复用的全局前向序号，决定 augmentation seed
    resume_path = Path(run.config.resume_checkpoint).expanduser().resolve() if run.config.resume_checkpoint else None

    def metadata() -> Any:
        r"""构造本次 run 共用的通用 checkpoint lineage。"""

        return run.checkpoint_metadata(
            geometry_semantics_schema=SEMANTICS_SCHEMA_VERSION,
            asset_manifest=await_manifest(),
            resolved_config=resolved_config,
            declared_objective=declared_weights,
            calibration_artifact_hash=calibration_hash,
            worktree_dirty=dirty,
            worktree_fingerprint=fingerprint,
        )

    def trainer_state() -> dict[str, Any]:
        r"""返回 schedule/session/RNG/selection 的完整 optimizer-boundary 状态。"""

        return {
            "sampling": _sampling_state(train_schedule, train_session),
            "forward_index": forward_index,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
            "selection_history": selection_history,
            "initial_validation_metrics": initial_validation,
            "initial_validation_strata": None,
            "best_validation_score": None if best_score == float("inf") else best_score,
        }

    def save_checkpoint(path: Path) -> None:
        r"""保存通用容器；Method 与 Trainer 分别提供自己的 state。"""

        run.save_full_checkpoint(
            path,
            method_state=method.training_state_dict(),
            optimizer_state=optimizer.state_dict(),
            step=step,
            metadata=metadata(),
            trainer_state=trainer_state(),
        )

    if resume_path is not None:
        payload = load_pretrain_checkpoint(resume_path, map_location=device)
        method.load_training_state_dict(payload["method_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        step = int(payload["step"])
        loaded_metadata = dict(payload["metadata"])
        if loaded_metadata.get("asset_manifest") != await_manifest():
            raise ValueError("resume checkpoint asset manifest does not match resolved dataset roles")
        checkpoint_resolved = loaded_metadata.get("resolved_config")
        if not isinstance(checkpoint_resolved, dict):
            raise ValueError("resume checkpoint lacks resolved config")
        require_resume_scientific_config(resolved_config, checkpoint_resolved)
        state = dict(payload["trainer_state"])
        raw_forward_index = state.get("forward_index")
        if not isinstance(raw_forward_index, int) or raw_forward_index < 0:
            raise ValueError("resume checkpoint lacks a valid global forward_index")
        forward_index = raw_forward_index
        sampling_state = state.get("sampling")
        if not isinstance(sampling_state, dict):
            raise ValueError("resume checkpoint lacks Trainer sampling state")
        _restore_sampling_state(sampling_state, train_schedule, train_session)
        initial_validation, _initial_strata, best_score, selection_history = restore_validation_selection_state(state)
        torch_rng_state = state.get("torch_rng_state")
        cuda_rng_state = state.get("cuda_rng_state_all")
        if not isinstance(torch_rng_state, torch.Tensor):
            raise ValueError("resume checkpoint lacks torch RNG state")
        if not isinstance(cuda_rng_state, list) or not all(isinstance(item, torch.Tensor) for item in cuda_rng_state):
            raise ValueError("resume checkpoint lacks CUDA RNG states")
        torch.set_rng_state(torch_rng_state.cpu())
        torch.cuda.set_rng_state_all(cuda_rng_state)
        if selection_history:
            historical_step = int(min(selection_history, key=lambda item: float(item["score"]))["step"])
            source_best = resume_path.parent / f"best_step_{historical_step:08d}.pt"
            if not source_best.is_file():
                raise ValueError("resume source run lacks immutable historical best checkpoint")
            inherited_best = output_dir / "checkpoints" / source_best.name
            inherited_best.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_best, inherited_best)
            publish_best_checkpoint(output_dir / "checkpoints" / "best.pt", inherited_best)

    try:
        q_bank_path = output_dir / "training_morphology_q_bank.yaml"
        if resume_path is None:
            initial_q_bank = run_training_q_bank()
        else:
            source_q_bank = resume_path.parent.parent / "training_morphology_q_bank.yaml"
            if not source_q_bank.is_file():
                raise ValueError("resume source run lacks training_morphology_q_bank.yaml")
            source_payload = yaml.safe_load(source_q_bank.read_text(encoding="utf-8"))
            if not isinstance(source_payload, dict) or not isinstance(source_payload.get("initial"), dict):
                raise ValueError("resume source q-bank artifact lacks initial evidence")
            initial_q_bank = source_payload["initial"]
        _write_yaml(q_bank_path, {"initial": initial_q_bank, "final": None, "comparison": None})

        if initial_validation is None:
            initial_reports = run_suites("validation", trainer.config.validation, include_ablations=False)
            initial_metrics = {
                suite: report.metrics
                for suite, report in initial_reports.items()
                if hasattr(report, "metrics")
            }
            initial_validation = trainer.selection_baseline(initial_metrics)
            _write_yaml(output_dir / "validation_initial.yaml", initial_reports)

        while not train_schedule.complete:
            if step + trainer.config.mini_epochs > trainer.config.run_safety_step_limit:
                raise RuntimeError("run_safety_step_limit exhausted before configured minibatches completed")
            method.train_mode()
            # 一组新 minibatch 只 realization 一次；五次 mini-epoch 均复用这些 q/query/teacher tensors。
            group_size = min(trainer.config.gradient_accumulation_steps, train_schedule.minibatches_remaining)
            batches: list[Any] = []
            for _ in range(group_size):
                schedule_item = train_schedule.next()
                batches.append(
                    _build_batch(
                        schedule_item,
                        session=train_session,
                        schedule=train_schedule,
                        step=schedule_item.minibatch_index,
                    )
                )
            group_start_step = step  # 用于检测本组是否跨过 validation/checkpoint cadence
            mini_epoch_records: list[dict[str, Any]] = []  # 当前数据组五次参数更新的审计记录
            for mini_epoch_index in range(trainer.config.mini_epochs):
                optimizer.zero_grad(set_to_none=True)
                update_steps: list[Any] = []
                for batch in batches:
                    update_steps.append(method.forward_objectives(batch, step=forward_index, mode="train"))
                    forward_index += 1
                update = method.reduce_update(tuple(update_steps))
                update.loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(method.parameters(), trainer.config.max_gradient_norm)
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError(f"non-finite gradient norm at optimizer step {step + 1}")
                optimizer.step()
                step += 1
                mini_epoch_records.append(
                    {
                        "step": step,
                        "mini_epoch": mini_epoch_index,
                        "terms": update.terms,
                        "gradient_norm": float(gradient_norm.detach()),
                    }
                )
                if step % trainer.config.log_every_updates == 0:
                    _write_metrics(
                        output_dir / "metrics.jsonl",
                        {
                            "step": step,
                            "split": "train",
                            "new_minibatches_consumed": train_schedule.minibatch_cursor,
                            "minibatches_reused": len(update_steps),
                            "mini_epoch": mini_epoch_index,
                            "terms": update.terms,
                            "gradient_norm": float(gradient_norm.detach()),
                        },
                    )
            # 完成整组复用后临时 batch 才可丢弃；checkpoint 因而无需保存巨大的 teacher realization。
            if train_schedule.complete and step % trainer.config.log_every_updates != 0:
                final_record = mini_epoch_records[-1]
                _write_metrics(
                    output_dir / "metrics.jsonl",
                    {
                        "step": step,
                        "split": "train",
                        "new_minibatches_consumed": train_schedule.minibatch_cursor,
                        "minibatches_reused": group_size,
                        "mini_epoch": final_record["mini_epoch"],
                        "terms": final_record["terms"],
                        "gradient_norm": final_record["gradient_norm"],
                    },
                )
            if audit_supported and audit_handle is None:
                start_audit()  # 首个训练组已完成，后台审计不再阻塞首个 resident window
            validation_cadence = trainer.config.validation.every_optimizer_updates
            validation_due = (
                group_start_step // validation_cadence < step // validation_cadence or train_schedule.complete
            )
            if validation_due:
                reports = run_suites("validation", trainer.config.validation, include_ablations=False)
                metrics = {suite: report.metrics for suite, report in reports.items() if hasattr(report, "metrics")}
                if initial_validation is None:
                    raise RuntimeError("validation baseline was not initialized")
                score = trainer.normalized_validation_score(metrics, initial_validation)
                selection_history.append({"step": step, "score": score, "metrics": metrics})
                _write_yaml(output_dir / f"validation_step_{step:08d}.yaml", reports)
                if score < best_score:
                    best_score = score
                    immutable = output_dir / "checkpoints" / f"best_step_{step:08d}.pt"
                    save_checkpoint(immutable)
                    publish_best_checkpoint(output_dir / "checkpoints" / "best.pt", immutable)
            checkpoint_cadence = trainer.config.checkpoint_every_updates
            checkpoint_due = (
                group_start_step // checkpoint_cadence < step // checkpoint_cadence or train_schedule.complete
            )
            if checkpoint_due:
                save_checkpoint(output_dir / "checkpoints" / f"step_{step:08d}.pt")

        save_checkpoint(output_dir / "checkpoints" / "last.pt")
        best_source = output_dir / "checkpoints" / (
            f"best_step_{min(selection_history, key=lambda item: item['score'])['step']:08d}.pt"
            if selection_history
            else "last.pt"
        )
        best_payload = load_pretrain_checkpoint(best_source, map_location=device)
        method.load_training_state_dict(best_payload["method_state"])

        final_q_bank = run_training_q_bank()
        initial_q_bank_payload = _plain(initial_q_bank)
        final_q_bank_payload = _plain(final_q_bank)
        initial_strata = initial_q_bank_payload.get("strata", {})
        final_strata = final_q_bank_payload.get("strata", {})
        if initial_strata.get("bank_digest_sha256") != final_strata.get("bank_digest_sha256"):
            raise RuntimeError("training morphology q-bank identity changed between initial and final evaluation")
        q_bank_comparison = {
            name: {
                "initial": float(initial_q_bank_payload["metrics"][name]),
                "final": float(final_q_bank_payload["metrics"][name]),
                "improvement_initial_minus_final": (
                    float(initial_q_bank_payload["metrics"][name])
                    - float(final_q_bank_payload["metrics"][name])
                ),
            }
            for name in trainer.config.validation.selection_metrics
        }
        _write_yaml(
            q_bank_path,
            {"initial": initial_q_bank_payload, "final": final_q_bank_payload, "comparison": q_bank_comparison},
        )

        final_reports = run_suites("evaluation", trainer.config.final_evaluation, include_ablations=True)
        final_summary: dict[str, Any] = {}
        for suite_index, (suite_name, report) in enumerate(final_reports.items()):
            if not hasattr(report, "metrics"):
                final_summary[suite_name] = report
                continue
            suite_payload = {"metrics": report.metrics, "strata": report.strata, "ablations": report.ablations}
            if report.ablations is not None:
                actual = tuple(str(name) for name in report.ablations.get("ablations", ()))[1:]
                if actual != trainer.config.final_evaluation.final_ablations:
                    raise ValueError("Method final ablations do not match Trainer final_evaluation config")
                suite_payload["ablation_analysis"] = method.analyze_ablations(
                    report.ablations,
                    bootstrap_replicates=trainer.config.final_evaluation.bootstrap_replicates,
                    seed=(
                        run.config.seed
                        + trainer.config.final_evaluation.bootstrap_seed_offset
                        + suite_index * 1_000_003
                    ),
                )
            final_summary[suite_name] = suite_payload
        _write_yaml(output_dir / "final_evaluation.yaml", final_summary)

        current_metadata = metadata()
        retained_payload = method.retained_artifact_payload(
            metadata=asdict(current_metadata),
            source_checkpoint=best_source,
        )
        run.save_retained_artifact(output_dir / "retained_artifact.pt", retained_payload)
        _write_yaml(
            output_dir / "checkpoint_selection.yaml",
            {
                "selection_metrics": trainer.config.validation.selection_metrics,
                "initial_validation": initial_validation,
                "history": selection_history,
                "best_checkpoint": str(best_source.relative_to(output_dir)),
                "retained_artifact": "retained_artifact.pt",
                "final_evaluation": "final_evaluation.yaml",
            },
        )
        return output_dir
    finally:
        train_session.close()
        write_resource_evidence()
        cancel_unpublished_audit()
        method.close()


__all__ = ["fit_embodiment_pretrain"]
