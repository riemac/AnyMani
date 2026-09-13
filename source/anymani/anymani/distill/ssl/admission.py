r"""Geometry SSL 正式训练前的真实 fixed-batch 数值准入入口。

本模块不启动训练 lifecycle，也不发布 checkpoint。它只从正式 train catalog 确定性实现一个
``64 assets × 8 q = 512 pairs`` teacher batch，并让候选执行路径消费同一批物理真值。这样比较的是
梯度归约或 precision 本身，而不是两次独立采样的随机差异。

运行入口：``python -m anymani.distill.ssl.admission streamed-parity``。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import fields, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
import yaml

from anymani.distill.methods.multi_anchor_gaussian_implicit_field.batch import (
    split_padded_online_geometry_batch,
)
from anymani.distill.methods.multi_anchor_gaussian_implicit_field.training import clip_parameter_groups
from anymani.distill.representations.sources.artifacts import (
    _decode_anchor,
    _decode_base,
    _encode_anchor,
    _encode_base,
)
from anymani.distill.representations.sources.geometry_source import GeometrySource
from anymani.distill.ssl.config_store import compose_pretrain_cfg
from anymani.distill.ssl.contracts import build_runtime
from anymani.distill.ssl.experiments import DEFAULT_EXPERIMENT_NAME
from anymani.distill.ssl.runtime.sampling import OnlineMinibatchSchedule
from anymani.distill.ssl.runtime.scheduler import ResidentGeometryAssetWindow


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    r"""原子写出只含基础类型的 admission evidence。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")
    temporary.replace(path)


def _formal_shape_guard(config: Any) -> None:
    r"""拒绝把缩小 batch 的快速测试误标为正式 512-pair parity。"""

    sampling = config.trainer.sampling
    observed = (
        int(sampling.assets_per_minibatch),
        int(sampling.q_per_asset_per_minibatch),
        int(config.trainer.microbatch_size),
    )
    if observed != (64, 8, 64):
        raise ValueError(f"admission requires formal shape (64 assets,8 q,microbatch 64), got {observed}")


def _configure_method(
    config: Any,
    catalog: Any,
    *,
    device: torch.device,
    execution: Any,
) -> Any:
    r"""构造一个独立 Method/model，但复用同一 train source object store。"""

    method = build_runtime(config.method)
    method.configure_source_artifacts(
        root=config.run.source_cache_root,
        mode="readonly",
        dataset_manifest_sha256=str(catalog.dataset.source_sha256),
        producer_device=str(device),
        role="train",
    )
    method.prepare(catalog, role="train", device=device, dtype=torch.float32)
    method.configure_execution(execution)
    method.initialize_model(device=device, dtype=torch.float32)
    return method


def _realize_fixed_batch(config: Any, catalog: Any, method: Any, *, device: torch.device) -> Any:
    r"""实现 formal schedule 的第一个完整 logical minibatch，随后关闭 source session。"""

    schedule = OnlineMinibatchSchedule(
        len(catalog.train),
        config.trainer.sampling,
        max_epochs=1,
        num_minibatches=1,
        max_resident_assets=config.trainer.sampling.assets_per_minibatch,
    )
    session = method.open_session(
        "train",
        seed=config.trainer.sampling.seed,
        device=device,
        dtype=torch.float32,
        max_resident_assets=config.trainer.device_window_assets,
        window_factory=ResidentGeometryAssetWindow,
        resource_profile=False,
    )
    try:
        item = schedule.next()
        batch = session.realize(item, schedule=schedule, step=0)
    finally:
        session.close()
    if tuple(batch.q.shape)[:1] != (512,):
        raise RuntimeError(f"fixed admission batch must contain 512 pairs, got q shape={tuple(batch.q.shape)}")
    return batch


def _optimizer(method: Any, config: Any) -> torch.optim.AdamW:
    r"""按正式 shared/density/kappa 参数分组构造独立 AdamW。"""

    groups = method.optimizer_parameter_groups()
    return torch.optim.AdamW(
        [{"name": group.name, "params": group.parameters} for group in groups],
        lr=config.trainer.optimizer.learning_rate,
        weight_decay=config.trainer.optimizer.weight_decay,
    )


def _clone_parameters(method: Any) -> dict[str, torch.Tensor]:
    r"""保存一次 optimizer step 前的 FP32 master parameters。"""

    return {
        name: parameter.detach().clone()
        for name, parameter in method.require_model().named_parameters()
    }


def _capture_vectors(method: Any, initial: Mapping[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor | None]]:
    r"""按 Method-owned 参数组捕获最终 gradient 与 one-step parameter delta。"""

    names_by_identity = {
        id(parameter): name for name, parameter in method.require_model().named_parameters()
    }
    captured: dict[str, dict[str, torch.Tensor | None]] = {}
    for group in method.optimizer_parameter_groups():
        for parameter in group.parameters:
            name = names_by_identity[id(parameter)]
            captured[f"{group.name}/{name}"] = {
                "gradient": None if parameter.grad is None else parameter.grad.detach().cpu().clone(),
                "delta": (parameter.detach() - initial[name]).cpu().clone(),
            }
    return captured


def _vector_comparison(
    first: Mapping[str, dict[str, torch.Tensor | None]],
    second: Mapping[str, dict[str, torch.Tensor | None]],
    *,
    field: str,
) -> dict[str, Any]:
    r"""在 FP64 中累计全参数 cosine/norm，并保留最坏逐元素绝对误差。"""

    if set(first) != set(second):
        raise ValueError("compared parameter partitions do not share stable names")
    dot = 0.0
    first_square = 0.0
    second_square = 0.0
    difference_square = 0.0
    max_absolute_error = 0.0
    none_mismatch: list[str] = []
    for name in sorted(first):
        left = first[name][field]
        right = second[name][field]
        if left is None or right is None:
            if left is not None or right is not None:
                none_mismatch.append(name)
            continue
        left64 = left.detach().to(torch.float64)
        right64 = right.detach().to(torch.float64)
        difference = left64 - right64
        dot += float((left64 * right64).sum())
        first_square += float(left64.square().sum())
        second_square += float(right64.square().sum())
        difference_square += float(difference.square().sum())
        max_absolute_error = max(max_absolute_error, float(difference.abs().max()))
    denominator = math.sqrt(max(first_square * second_square, 0.0))
    return {
        "cosine": dot / max(denominator, 1.0e-300),
        "first_norm": math.sqrt(max(first_square, 0.0)),
        "second_norm": math.sqrt(max(second_square, 0.0)),
        "relative_l2_error": math.sqrt(max(difference_square, 0.0)) / max(math.sqrt(first_square), 1.0e-300),
        "max_absolute_error": max_absolute_error,
        "none_mismatch": none_mismatch,
    }


def _relative_error(first: float, second: float) -> float:
    r"""返回以 full-reference 为分母的稳定相对误差。"""

    return abs(first - second) / max(abs(first), 1.0e-30)


def run_streamed_parity(config_ref: str, output: Path) -> Path:
    r"""比较真实 512-pair full 与 8×64 streamed FairGrad/AdamW 路径。"""

    config = compose_pretrain_cfg(config_ref=config_ref)
    config.validate_composed()
    _formal_shape_guard(config)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.run.seed)
    device = torch.device(config.trainer.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"streamed parity requires an available CUDA device, got {device}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = build_runtime(config.data)
    catalog = data.resolve_train()
    execution = replace(
        config.trainer.execution,
        model_autocast_dtype="float32",
        allow_tf32=False,
        compile_enabled=False,
    )
    full_method = _configure_method(config, catalog, device=device, execution=execution)
    stream_method = _configure_method(config, catalog, device=device, execution=execution)
    try:
        batch = _realize_fixed_batch(config, catalog, full_method, device=device)
        stream_method.load_training_state_dict(full_method.training_state_dict())
        full_optimizer = _optimizer(full_method, config)
        stream_optimizer = _optimizer(stream_method, config)
        initial_full = _clone_parameters(full_method)
        initial_stream = _clone_parameters(stream_method)

        full_optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = perf_counter()
        full_update = full_method.backward_update(
            batch,
            forward_step=0,
            microbatch_size=64,
            collect_z_gradients=False,
        )
        full_optimizer.step()
        torch.cuda.synchronize(device)
        full_seconds = perf_counter() - started
        full_peak = int(torch.cuda.max_memory_allocated(device))
        full_vectors = _capture_vectors(full_method, initial_full)

        stream_optimizer.zero_grad(set_to_none=True)
        units = split_padded_online_geometry_batch(batch, microbatch_size=64)
        if len(units) != 8 or any(unit.q.shape[0] != 64 for unit in units):
            raise RuntimeError("streamed parity requires exactly eight 64-pair units")
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = perf_counter()
        stream_update = stream_method.backward_update_units(
            iter(units),
            forward_step=0,
            logical_sample_count=512,
            microbatch_size=64,
            collect_z_gradients=False,
        )
        stream_optimizer.step()
        torch.cuda.synchronize(device)
        stream_seconds = perf_counter() - started
        stream_peak = int(torch.cuda.max_memory_allocated(device))
        stream_vectors = _capture_vectors(stream_method, initial_stream)

        gradient = _vector_comparison(full_vectors, stream_vectors, field="gradient")
        delta = _vector_comparison(full_vectors, stream_vectors, field="delta")
        term_errors = {
            name: _relative_error(full_update.terms[name], stream_update.terms[name])
            for name in full_update.terms
        }
        denominator_equal = full_update.denominators == stream_update.denominators
        fairgrad_errors = {
            name: _relative_error(value, stream_update.gradient_evidence[name])
            for name, value in full_update.gradient_evidence.items()
        }
        passed = bool(
            denominator_equal
            and max(term_errors.values(), default=0.0) <= 1.0e-6
            and max(fairgrad_errors.values(), default=0.0) <= 1.0e-6
            and not gradient["none_mismatch"]
            and gradient["cosine"] >= 0.999999
            and gradient["relative_l2_error"] <= 1.0e-5
            and not delta["none_mismatch"]
            and delta["cosine"] >= 0.999999
            and delta["relative_l2_error"] <= 1.0e-5
        )
        payload = {
            "schema_version": "1.0.0",
            "phase": "streamed_parity",
            "config": config_ref,
            "shape": {"assets": 64, "q_per_asset": 8, "pairs": 512, "units": 8, "unit_pairs": 64},
            "full": {
                "terms": full_update.terms,
                "denominators": full_update.denominators,
                "fairgrad": full_update.gradient_evidence,
                "seconds": full_seconds,
                "peak_allocated_bytes": full_peak,
            },
            "streamed": {
                "terms": stream_update.terms,
                "denominators": stream_update.denominators,
                "fairgrad": stream_update.gradient_evidence,
                "seconds": stream_seconds,
                "peak_allocated_bytes": stream_peak,
            },
            "comparison": {
                "term_relative_error": term_errors,
                "denominators_equal": denominator_equal,
                "fairgrad_relative_error": fairgrad_errors,
                "gradient": gradient,
                "parameter_delta": delta,
            },
            "passed": passed,
        }
        _write_yaml(output, payload)
        if not passed:
            raise RuntimeError(f"streamed/full admission failed; inspect {output}")
        return output
    finally:
        full_method.close()
        stream_method.close()


def _read_legacy_directory(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    r"""只读 schema-1 object 并执行 manifest、COMPLETE 与全 payload SHA-256 审计。"""

    manifest_path = path / "manifest.json"
    complete_path = path / "COMPLETE"
    manifest_bytes = manifest_path.read_bytes()
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    if complete_path.read_text(encoding="ascii").strip() != digest:
        raise ValueError(f"legacy source COMPLETE digest mismatch: {path}")
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != "1.0.0":
        raise ValueError(f"legacy source object is not schema 1.0.0: {path}")
    arrays: dict[str, np.ndarray] = {}
    for name, record in manifest["arrays"].items():
        array_path = path / str(record["path"])
        payload = array_path.read_bytes()
        if len(payload) != int(record["byte_count"]) or hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError(f"legacy source payload checksum mismatch: {array_path}")
        array = np.load(array_path, allow_pickle=False)
        if list(array.shape) != record["shape"] or str(array.dtype) != record["dtype"]:
            raise ValueError(f"legacy source payload metadata mismatch: {array_path}")
        arrays[str(name)] = array
    return arrays, dict(manifest["metadata"])


def _decode_legacy_base(container: Any, path: Path) -> tuple[Any, dict[str, np.ndarray]]:
    r"""把 schema-1 重复 surface/solid 布局规范化为当前 compact core。"""

    arrays, metadata = _read_legacy_directory(path)
    normalized = json.loads(json.dumps(metadata))  # 深拷贝，绝不修改磁盘上的历史 manifest
    for index, record in enumerate(normalized["geometry_cache"]["records"]):
        prefix = f"owner_{index:03d}"
        solid_vertices = arrays.get(f"{prefix}_solid_vertices")
        solid_faces = arrays.get(f"{prefix}_solid_faces")
        if solid_vertices is None or solid_faces is None:
            record["solid_storage"] = "none"
        elif np.array_equal(solid_vertices, arrays[f"{prefix}_surface_vertices"]) and np.array_equal(
            solid_faces,
            arrays[f"{prefix}_surface_faces"],
        ):
            record["solid_storage"] = "alias_surface"
        else:
            record["solid_storage"] = "separate"
    return _decode_base(container, arrays, normalized), arrays


def _legacy_asset_paths(root: Path, asset_ids: tuple[str, ...]) -> dict[str, Path]:
    r"""扫描 schema-1 flat key 目录，为指定 64 个资产建立只读 asset-id 索引。"""

    requested = set(asset_ids)
    matched: dict[str, Path] = {}
    for candidate in root.iterdir():
        manifest = candidate / "base" / "manifest.json"
        if not manifest.is_file():
            continue
        metadata = json.loads(manifest.read_bytes())["metadata"]
        asset_id = str(metadata.get("asset_id", ""))
        if asset_id in requested:
            if asset_id in matched:
                raise ValueError(f"legacy v1 cache contains duplicate asset_id={asset_id!r}")
            matched[asset_id] = candidate
            if len(matched) == len(requested):
                break
    missing = requested - matched.keys()
    if missing:
        raise FileNotFoundError(f"legacy v1 cache lacks requested assets: {sorted(missing)}")
    return matched


def _require_numpy_mapping_equal(
    first: Mapping[str, np.ndarray],
    second: Mapping[str, np.ndarray],
    *,
    context: str,
) -> int:
    r"""验证两个规范 array mapping 的 key、shape、dtype-independent values 完全一致。"""

    if set(first) != set(second):
        raise ValueError(f"{context} array keys differ: {sorted(set(first) ^ set(second))}")
    compared = 0
    for name in sorted(first):
        left = np.asarray(first[name])
        right = np.asarray(second[name])
        if left.shape != right.shape or not np.array_equal(left, right):
            raise ValueError(f"{context} array mismatch name={name!r} shapes={left.shape}/{right.shape}")
        compared += 1
    return compared


def _require_sampling_equal(legacy_arrays: Mapping[str, np.ndarray], core: Any, *, context: str) -> int:
    r"""验证 v2 从 canonical mesh 恢复的 normals/CDF/triangles 与 v1 冗余 payload 相同。"""

    sampling = core.surface_sampling_arrays
    if sampling is None:
        raise ValueError(f"{context} v2 core lacks reconstructed surface sampling arrays")
    compared = 0
    for index, (vertices, faces, normals, cdf) in enumerate(
        zip(
            sampling.vertices_owner_local_m,
            sampling.faces,
            sampling.face_normals_owner_local,
            sampling.face_area_cdf,
            strict=True,
        )
    ):
        prefix = f"owner_{index:03d}_query"
        expected_triangles = np.asarray(vertices)[np.asarray(faces)]
        for name, actual in (
            (f"{prefix}_triangles", expected_triangles),
            (f"{prefix}_normals", normals),
            (f"{prefix}_area_cdf", cdf),
        ):
            legacy = legacy_arrays[name]
            if legacy.shape != np.asarray(actual).shape or not np.array_equal(legacy, actual):
                raise ValueError(f"{context} reconstructed sampling mismatch name={name!r}")
            compared += 1
    return compared


def _require_typed_batch_equal(first: Any, second: Any, *, context: str) -> int:
    r"""逐字段比较 query/teacher dataclass；仅排除非科学 wall-time telemetry。"""

    compared = 0
    for field_info in fields(first):
        if field_info.name == "central_difference_elapsed_seconds":
            continue
        left = getattr(first, field_info.name)
        right = getattr(second, field_info.name)
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            if not torch.equal(left, right):
                maximum = float((left.to(torch.float64) - right.to(torch.float64)).abs().max())
                raise ValueError(f"{context}.{field_info.name} tensor mismatch max_abs={maximum}")
        elif left != right:
            raise ValueError(f"{context}.{field_info.name} mismatch")
        compared += 1
    return compared


def run_source_parity(config_ref: str, v1_root: Path, output: Path) -> Path:
    r"""审计 64 个真实资产的 v1/v2 base、8-bank anchors 与固定 teacher outputs。"""

    config = compose_pretrain_cfg(config_ref=config_ref)
    config.validate_composed()
    _formal_shape_guard(config)
    device = torch.device(config.trainer.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"source parity requires an available CUDA device, got {device}")
    data = build_runtime(config.data)
    catalog = data.resolve_train()
    schedule = OnlineMinibatchSchedule(
        len(catalog.train),
        config.trainer.sampling,
        max_epochs=1,
        num_minibatches=1,
        max_resident_assets=config.trainer.sampling.assets_per_minibatch,
    )
    item = schedule.next()
    selected = tuple(catalog.train[index] for index in item.asset_indices)
    if len(selected) != 64:
        raise RuntimeError(f"source parity requires 64 scheduled assets, got {len(selected)}")
    legacy_paths = _legacy_asset_paths(v1_root.resolve(), tuple(container.asset_id for container in selected))
    method = build_runtime(config.method)
    method.configure_source_artifacts(
        root=config.run.source_cache_root,
        mode="read-write",
        dataset_manifest_sha256=str(catalog.dataset.source_sha256),
        producer_device=str(device),
        role="train",
    )
    method.prepare(catalog, role="train", device=device, dtype=torch.float32)
    source_config = config.method.representation.source
    normalized_array_count = 0
    sampling_array_count = 0
    anchor_array_count = 0
    typed_teacher_field_count = 0
    v1_keys: list[str] = []
    started = perf_counter()
    try:
        for position, container in enumerate(selected):
            legacy_root = legacy_paths[container.asset_id]
            legacy_core, legacy_arrays = _decode_legacy_base(container, legacy_root / "base")
            v2_core, _reference = method.source_artifact_store.load_base(container, source_config)
            legacy_normalized_arrays, legacy_metadata = _encode_base(legacy_core)
            v2_normalized_arrays, v2_metadata = _encode_base(v2_core)
            context = f"asset={container.asset_id}"
            if legacy_metadata != v2_metadata:
                raise ValueError(f"{context} normalized base metadata differ")
            normalized_array_count += _require_numpy_mapping_equal(
                legacy_normalized_arrays,
                v2_normalized_arrays,
                context=context,
            )
            sampling_array_count += _require_sampling_equal(legacy_arrays, v2_core, context=context)

            legacy_realizations = []
            v2_realizations = []
            for bank_index in range(source_config.anchors.bank_size):
                legacy_anchor_arrays, legacy_anchor_metadata = _read_legacy_directory(
                    legacy_root / "anchors" / f"bank_{bank_index:06d}"
                )
                legacy_realization, legacy_stats = _decode_anchor(legacy_anchor_arrays, legacy_anchor_metadata)
                state = method._load_device_state_with_artifact(
                    v2_core,
                    representation=method.representation,
                    bank_index=bank_index,
                    device=device,
                    dtype=torch.float32,
                )
                try:
                    v2_realization = state.source.anchor_realization
                    if v2_realization is None:
                        raise RuntimeError(f"{context} bank={bank_index} lacks selected realization identity")
                    v2_stats = state.anchor_classification
                    if v2_stats is None:
                        raise RuntimeError(f"{context} bank={bank_index} lacks classification stats")
                finally:
                    state.device_source.release()
                legacy_encoded_arrays, legacy_encoded_metadata = _encode_anchor(legacy_realization, legacy_stats)
                v2_encoded_arrays, v2_encoded_metadata = _encode_anchor(v2_realization, v2_stats)
                anchor_array_count += _require_numpy_mapping_equal(
                    legacy_encoded_arrays,
                    v2_encoded_arrays,
                    context=f"{context} bank={bank_index}",
                )
                for ignored in ("classifier", "producer"):
                    legacy_encoded_metadata.pop(ignored, None)
                    v2_encoded_metadata.pop(ignored, None)
                if legacy_encoded_metadata != v2_encoded_metadata:
                    raise ValueError(f"{context} bank={bank_index} anchor metadata differ")
                legacy_realizations.append(legacy_realization)
                v2_realizations.append(v2_realization)

            legacy_source = GeometrySource.from_core(
                legacy_core,
                anchor_bank=tuple(realization.samples for realization in legacy_realizations),
            )
            v2_source = GeometrySource.from_core(
                v2_core,
                anchor_bank=tuple(realization.samples for realization in v2_realizations),
            )
            legacy_state = method.representation.to_device(legacy_source, device=device, dtype=torch.float32)
            v2_state = method.representation.to_device(v2_source, device=device, dtype=torch.float32)
            try:
                q = v2_state.spec.q_home.to(device=device, dtype=torch.float32).unsqueeze(0)
                q_index = torch.zeros(1, device=device, dtype=torch.long)
                for bank_index in range(source_config.anchors.bank_size):
                    sampling_seed = int(config.run.seed + position * source_config.anchors.bank_size + bank_index)
                    legacy_sample = method.representation.sample(
                        legacy_state,
                        q,
                        sampling_seed=sampling_seed,
                        q_index=q_index,
                        anchor_index=bank_index,
                        supervision_split="train",
                    )
                    v2_sample = method.representation.sample(
                        v2_state,
                        q,
                        sampling_seed=sampling_seed,
                        q_index=q_index,
                        anchor_index=bank_index,
                        supervision_split="train",
                    )
                    typed_teacher_field_count += _require_typed_batch_equal(
                        legacy_sample.queries,
                        v2_sample.queries,
                        context=f"{context}.bank={bank_index}.queries",
                    )
                    typed_teacher_field_count += _require_typed_batch_equal(
                        legacy_sample.field_targets,
                        v2_sample.field_targets,
                        context=f"{context}.bank={bank_index}.field_targets",
                    )
                    typed_teacher_field_count += _require_typed_batch_equal(
                        legacy_sample.sensitivity_targets,
                        v2_sample.sensitivity_targets,
                        context=f"{context}.bank={bank_index}.sensitivity_targets",
                    )
            finally:
                legacy_state.device_source.release()
                v2_state.device_source.release()
            v1_keys.append(legacy_root.name)
    finally:
        method.close()
    payload = {
        "schema_version": "1.0.0",
        "phase": "source_parity",
        "config": config_ref,
        "v1_root": str(v1_root.resolve()),
        "v2_root": str(Path(config.run.source_cache_root).resolve()),
        "asset_count": len(selected),
        "bank_count_per_asset": source_config.anchors.bank_size,
        "teacher_cases": len(selected) * source_config.anchors.bank_size,
        "normalized_base_arrays_compared": normalized_array_count,
        "reconstructed_sampling_arrays_compared": sampling_array_count,
        "anchor_arrays_compared": anchor_array_count,
        "typed_teacher_fields_compared": typed_teacher_field_count,
        "v1_artifact_keys": v1_keys,
        "elapsed_seconds": perf_counter() - started,
        "exact": True,
        "passed": True,
    }
    _write_yaml(output, payload)
    return output


def _tensor_tree_digest(value: Any, digest: Any) -> None:
    r"""把 fixed teacher batch 的 tensor bytes、shape、dtype 与离散 provenance 写入 SHA-256。"""

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    elif hasattr(value, "__dataclass_fields__"):
        for field_info in fields(value):
            if field_info.name == "central_difference_elapsed_seconds":
                continue
            digest.update(field_info.name.encode("utf-8"))
            _tensor_tree_digest(getattr(value, field_info.name), digest)
    elif isinstance(value, Mapping):
        for name in sorted(value, key=str):
            digest.update(str(name).encode("utf-8"))
            _tensor_tree_digest(value[name], digest)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _tensor_tree_digest(item, digest)
    elif value is not None:
        digest.update(repr(value).encode("utf-8"))


def _fixed_batch_digest(batch: Any) -> str:
    r"""返回排除 wall-time telemetry 后的完整 fixed teacher batch 内容摘要。"""

    digest = hashlib.sha256(b"anymani-geometry-ssl-admission-batch-v1\0")
    _tensor_tree_digest(batch, digest)
    return digest.hexdigest()


def _partition_comparison(
    reference: Mapping[str, dict[str, torch.Tensor | None]],
    candidate: Mapping[str, dict[str, torch.Tensor | None]],
    *,
    field: str,
) -> dict[str, dict[str, Any]]:
    r"""分别报告 shared encoder、density reader 与 kappa reader 的向量一致性。"""

    result: dict[str, dict[str, Any]] = {}
    for group in ("shared_encoder", "density_reader", "kappa_reader"):
        prefix = group + "/"
        left = {name: value for name, value in reference.items() if name.startswith(prefix)}
        right = {name: value for name, value in candidate.items() if name.startswith(prefix)}
        result[group] = _vector_comparison(left, right, field=field)
    return result


def _clear_compiler_runtime() -> None:
    r"""释放上一 precision profile 的 compiled graph/workspace，保留 fixed teacher batch。"""

    gc.collect()
    torch._dynamo.reset()  # pyright: ignore[reportPrivateImportUsage]
    torch.cuda.empty_cache()


def _run_precision_profile(
    config: Any,
    catalog: Any,
    batch: Any,
    initial_state: Mapping[str, torch.Tensor],
    *,
    name: str,
    autocast_dtype: str,
    allow_tf32: bool,
    compile_enabled: bool,
    updates: int,
    device: torch.device,
) -> dict[str, Any]:
    r"""从同一初始 state 运行 16 个 matched updates，并返回内部首步向量与可序列化证据。"""

    execution = replace(
        config.trainer.execution,
        model_autocast_dtype=autocast_dtype,
        allow_tf32=allow_tf32,
        compile_enabled=compile_enabled,
    )
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.manual_seed(config.run.seed)
    method = _configure_method(config, catalog, device=device, execution=execution)
    method.load_training_state_dict(initial_state)
    optimizer = _optimizer(method, config)
    initial_parameters = _clone_parameters(method)
    losses: list[dict[str, float]] = []
    step_seconds: list[float] = []
    peak_allocated = 0
    peak_reserved = 0
    minimum_headroom = math.inf
    first_vectors: dict[str, dict[str, torch.Tensor | None]] | None = None
    finite = True
    try:
        units = split_padded_online_geometry_batch(batch, microbatch_size=64)
        for update_index in range(updates):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            free_before, _total = torch.cuda.mem_get_info(device)
            reserved_before = torch.cuda.memory_reserved(device)
            torch.cuda.synchronize(device)
            started = perf_counter()
            update = method.backward_update_units(
                iter(units),
                forward_step=update_index,
                logical_sample_count=512,
                microbatch_size=64,
                collect_z_gradients=False,
            )
            clip_parameter_groups(
                method.optimizer_parameter_groups(),
                max_norm=config.trainer.max_gradient_norm_per_group,
            )
            optimizer.step()
            torch.cuda.synchronize(device)
            elapsed = perf_counter() - started
            step_seconds.append(elapsed)
            losses.append({name: float(value) for name, value in update.terms.items()})
            current_peak_allocated = int(torch.cuda.max_memory_allocated(device))
            current_peak_reserved = int(torch.cuda.max_memory_reserved(device))
            peak_allocated = max(peak_allocated, current_peak_allocated)
            peak_reserved = max(peak_reserved, current_peak_reserved)
            minimum_headroom = min(
                minimum_headroom,
                float(free_before - max(current_peak_reserved - reserved_before, 0)),
            )
            parameters = tuple(method.parameters())
            finite = finite and all(math.isfinite(value) for value in update.terms.values())
            finite = finite and all(
                torch.isfinite(parameter).all().item()
                and (parameter.grad is None or torch.isfinite(parameter.grad).all().item())
                for parameter in parameters
            )
            if update_index == 0:
                first_vectors = _capture_vectors(method, initial_parameters)
        if first_vectors is None:
            raise RuntimeError(f"precision profile {name!r} produced no optimizer update")
        steady_seconds = step_seconds[1:] if len(step_seconds) > 1 else step_seconds
        mean_steady_seconds = sum(steady_seconds) / len(steady_seconds)
        auc = {
            term: sum(record[term] for record in losses)
            for term in losses[0]
        }
        return {
            "name": name,
            "execution": {
                "model_autocast_dtype": autocast_dtype,
                "allow_tf32": allow_tf32,
                "compile_enabled": compile_enabled,
                "compile_mode": config.trainer.execution.compile_mode,
            },
            "losses": losses,
            "auc": auc,
            "step_seconds": step_seconds,
            "steady_mean_seconds": mean_steady_seconds,
            "steady_pairs_per_second": 512.0 / mean_steady_seconds,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "minimum_driver_headroom_bytes_estimate": int(minimum_headroom),
            "finite": finite,
            "_vectors": first_vectors,
        }
    finally:
        method.close()


def _profile_admission(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    r"""按冻结阈值比较候选 precision 与 FP32 compile reference。"""

    loss_relative_error = {
        name: _relative_error(reference["losses"][0][name], candidate["losses"][0][name])
        for name in reference["losses"][0]
    }
    auc_relative_error = {
        name: _relative_error(reference["auc"][name], candidate["auc"][name])
        for name in reference["auc"]
    }
    gradients = _partition_comparison(reference["_vectors"], candidate["_vectors"], field="gradient")
    deltas = _partition_comparison(reference["_vectors"], candidate["_vectors"], field="delta")
    throughput_ratio = candidate["steady_pairs_per_second"] / reference["steady_pairs_per_second"]
    memory_reduction = 1.0 - candidate["peak_allocated_bytes"] / reference["peak_allocated_bytes"]
    accuracy_passed = bool(
        max(loss_relative_error.values()) <= 0.01
        and max(auc_relative_error.values()) <= 0.05
        and all(value["cosine"] >= 0.99 and not value["none_mismatch"] for value in gradients.values())
        and all(value["cosine"] >= 0.99 and not value["none_mismatch"] for value in deltas.values())
    )
    performance_passed = bool(
        candidate["steady_mean_seconds"] <= reference["steady_mean_seconds"] * 1.03
        and (throughput_ratio >= 1.10 or memory_reduction >= 0.25)
        and candidate["minimum_driver_headroom_bytes_estimate"] >= 2 * 1024**3
    )
    return {
        "loss_relative_error": loss_relative_error,
        "auc_relative_error": auc_relative_error,
        "gradient": gradients,
        "parameter_delta": deltas,
        "throughput_ratio": throughput_ratio,
        "peak_allocated_reduction": memory_reduction,
        "accuracy_passed": accuracy_passed,
        "performance_passed": performance_passed,
        "passed": bool(candidate["finite"] and accuracy_passed and performance_passed),
    }


def _public_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    r"""从 YAML evidence 删除只供进程内比较的逐参数 tensor vectors。"""

    return {str(name): value for name, value in profile.items() if name != "_vectors"}


def run_precision_admission(config_ref: str, output: Path, *, updates: int = 16) -> Path:
    r"""执行 FP32 eager/compile、BF16 及按需 TF32 的 matched 16-update 准入。"""

    if updates != 16:
        raise ValueError("formal precision admission requires exactly 16 updates")
    config = compose_pretrain_cfg(config_ref=config_ref)
    config.validate_composed()
    _formal_shape_guard(config)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.run.seed)
    device = torch.device(config.trainer.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"precision admission requires an available CUDA device, got {device}")
    data = build_runtime(config.data)
    catalog = data.resolve_train()
    eager_execution = replace(
        config.trainer.execution,
        model_autocast_dtype="float32",
        allow_tf32=False,
        compile_enabled=False,
    )
    template = _configure_method(config, catalog, device=device, execution=eager_execution)
    try:
        batch = _realize_fixed_batch(config, catalog, template, device=device)
        initial_state = {
            name: value.detach().cpu().clone()
            for name, value in template.training_state_dict().items()
        }
    finally:
        template.close()
    batch_digest = _fixed_batch_digest(batch)
    _clear_compiler_runtime()

    profiles: dict[str, dict[str, Any]] = {}
    profiles["fp32_eager"] = _run_precision_profile(
        config,
        catalog,
        batch,
        initial_state,
        name="fp32_eager",
        autocast_dtype="float32",
        allow_tf32=False,
        compile_enabled=False,
        updates=updates,
        device=device,
    )
    _clear_compiler_runtime()
    profiles["fp32_compile"] = _run_precision_profile(
        config,
        catalog,
        batch,
        initial_state,
        name="fp32_compile",
        autocast_dtype="float32",
        allow_tf32=False,
        compile_enabled=True,
        updates=updates,
        device=device,
    )
    fp32_compile_parity = _profile_admission(profiles["fp32_eager"], profiles["fp32_compile"])
    # FP32 compile 只需保持 eager 数值并有限；性能收益不是 BF16/TF32 回退判断的一部分。
    fp32_compile_valid = bool(
        profiles["fp32_compile"]["finite"]
        and fp32_compile_parity["accuracy_passed"]
    )
    if not fp32_compile_valid:
        raise RuntimeError("FP32 compile failed eager numerical parity; no precision profile is admissible")

    _clear_compiler_runtime()
    profiles["bf16_compile"] = _run_precision_profile(
        config,
        catalog,
        batch,
        initial_state,
        name="bf16_compile",
        autocast_dtype="bfloat16",
        allow_tf32=False,
        compile_enabled=True,
        updates=updates,
        device=device,
    )
    comparisons: dict[str, dict[str, Any]] = {
        "fp32_compile_vs_eager": fp32_compile_parity,
        "bf16_compile_vs_fp32_compile": _profile_admission(
            profiles["fp32_compile"],
            profiles["bf16_compile"],
        ),
    }
    if comparisons["bf16_compile_vs_fp32_compile"]["passed"]:
        selected = "bf16_compile"
    else:
        _clear_compiler_runtime()
        profiles["tf32_compile"] = _run_precision_profile(
            config,
            catalog,
            batch,
            initial_state,
            name="tf32_compile",
            autocast_dtype="tf32",
            allow_tf32=True,
            compile_enabled=True,
            updates=updates,
            device=device,
        )
        comparisons["tf32_compile_vs_fp32_compile"] = _profile_admission(
            profiles["fp32_compile"],
            profiles["tf32_compile"],
        )
        selected = (
            "tf32_compile"
            if comparisons["tf32_compile_vs_fp32_compile"]["passed"]
            else "fp32_compile"
        )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    selected_profile = profiles[selected]["execution"]
    payload = {
        "schema_version": "1.0.0",
        "phase": "precision_admission",
        "config": config_ref,
        "fixed_teacher_batch_sha256": batch_digest,
        "teacher_query_target_mask_shared_exactly": True,
        "shape": {"assets": 64, "q_per_asset": 8, "pairs": 512, "updates": updates},
        "thresholds": {
            "first_step_loss_relative_error_max": 0.01,
            "gradient_cosine_min": 0.99,
            "parameter_delta_cosine_min": 0.99,
            "auc_relative_error_max": 0.05,
            "candidate_slowdown_max": 0.03,
            "throughput_gain_min": 0.10,
            "peak_allocated_reduction_min": 0.25,
            "driver_headroom_min_bytes": 2 * 1024**3,
        },
        "profiles": {name: _public_profile(profile) for name, profile in profiles.items()},
        "comparisons": comparisons,
        "selected_profile": {
            "name": selected,
            **selected_profile,
        },
        "passed": True,
    }
    _write_yaml(output, payload)
    _clear_compiler_runtime()
    return output


def _build_parser() -> argparse.ArgumentParser:
    r"""构造不暴露科研深层字段的 admission CLI。"""

    parser = argparse.ArgumentParser(description="Run Geometry SSL pretraining admission probes.")
    subparsers = parser.add_subparsers(dest="phase", required=True)
    streamed = subparsers.add_parser("streamed-parity", help="compare full and 8x64 streamed FP32 gradients")
    streamed.add_argument("--config", default=DEFAULT_EXPERIMENT_NAME)
    streamed.add_argument(
        "--output",
        type=Path,
        default=Path("logs/ssl/_admission/streamed_parity.yaml"),
    )
    source = subparsers.add_parser("source-parity", help="compare 64 real v1/v2 sources across all eight banks")
    source.add_argument("--config", default=DEFAULT_EXPERIMENT_NAME)
    source.add_argument("--v1-root", type=Path, default=Path("logs/ssl/_cache/geometry_source/v1"))
    source.add_argument(
        "--output",
        type=Path,
        default=Path("logs/ssl/_admission/source_parity.yaml"),
    )
    precision = subparsers.add_parser("precision", help="run matched FP32/BF16/TF32 16-update admission")
    precision.add_argument("--config", default=DEFAULT_EXPERIMENT_NAME)
    precision.add_argument("--updates", type=int, default=16)
    precision.add_argument(
        "--output",
        type=Path,
        default=Path("logs/ssl/_admission/precision.yaml"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    r"""运行一个显式 admission phase 并打印证据路径。"""

    args = _build_parser().parse_args(argv)
    if args.phase == "streamed-parity":
        result = run_streamed_parity(args.config, args.output)
    elif args.phase == "source-parity":
        result = run_source_parity(args.config, args.v1_root, args.output)
    elif args.phase == "precision":
        result = run_precision_admission(args.config, args.output, updates=args.updates)
    else:  # pragma: no cover - argparse required subcommand 已封闭该分支
        raise ValueError(f"unknown admission phase={args.phase!r}")
    print(result)
    return result


if __name__ == "__main__":
    main()


__all__ = ["main", "run_precision_admission", "run_source_parity", "run_streamed_parity"]
