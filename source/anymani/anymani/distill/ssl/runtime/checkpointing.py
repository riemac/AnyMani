r"""Geometry SSL runtime 的 resume 科学合同与 checkpoint-selection lineage。

底层 tensor payload 的原子读写由 ``ssl.checkpoint`` 拥有；本模块只定义 runtime 必须恢复的
epoch/window/Sobol/RNG/initialization-baseline/historical-best 状态，并拒绝当前 CLI 与 checkpoint
之间的科学配置或 asset manifest 漂移。
"""

from __future__ import annotations

from pathlib import Path  # immutable best checkpoint 与 mutable best.pt 发布路径

import torch  # RNG states 与有限性验证

from anymani.distill.ssl.config import GeometrySSLExperimentCfg, experiment_config_from_dict, resolved_config_dict
from anymani.distill.ssl.runtime import ResidentGeometryAssetWindow, WindowedOnlineGeometryBatcher


def resume_scientific_config(config: GeometrySSLExperimentCfg) -> dict[str, object]:
    r"""返回 resume 必须一致的科学配置，只排除 output/resume 定位。"""

    payload = resolved_config_dict(config)
    run = payload.pop("run", None)  # 整个 run 槽只定位 artifact/resume，不改变科学轨迹
    if not isinstance(run, dict):
        raise ValueError("resolved geometry SSL config lacks run mapping")
    return payload


def require_resume_scientific_config(
    current: GeometrySSLExperimentCfg,
    checkpoint_resolved: dict[str, object],
) -> None:
    r"""拒绝当前 CLI 与 checkpoint 的任一 scientific config 漂移。"""

    checkpoint_protocol = checkpoint_resolved.get("protocol")
    reproducibility = checkpoint_protocol.get("reproducibility") if isinstance(checkpoint_protocol, dict) else None
    if not isinstance(reproducibility, dict) or "deterministic_algorithms" not in reproducibility:
        raise ValueError("resume checkpoint predates the explicit deterministic-algorithm contract")
    checkpoint_config = experiment_config_from_dict(checkpoint_resolved)
    expected = resume_scientific_config(checkpoint_config)
    actual = resume_scientific_config(current)
    if actual != expected:
        changed_sections = tuple(key for key in expected.keys() | actual.keys() if expected.get(key) != actual.get(key))
        raise ValueError(f"resume scientific config mismatch in sections={changed_sections}")


def restore_validation_selection_state(
    runtime_payload: dict[str, object],
) -> tuple[dict[str, float] | None, dict[str, object] | None, float, list[dict[str, object]]]:
    r"""恢复 initialization strata、normalization baseline 与 historical best score/history。"""

    raw_initial = runtime_payload.get("initial_validation_metrics")
    raw_initial_strata = runtime_payload.get("initial_validation_strata")
    raw_best = runtime_payload.get("best_validation_score")
    raw_history = runtime_payload.get("selection_history")
    if raw_initial is None:
        initial = None
    elif isinstance(raw_initial, dict):
        if set(raw_initial) != {"density", "kappa", "derived_field"}:
            raise ValueError("resume checkpoint validation baseline has invalid metric keys")
        initial = {str(name): float(value) for name, value in raw_initial.items()}
        if any(not torch.isfinite(torch.tensor(value)) or value <= 0.0 for value in initial.values()):
            raise ValueError("resume checkpoint validation baseline must be finite and positive")
    else:
        raise ValueError("resume checkpoint validation baseline must be a mapping or null")
    if raw_initial_strata is None:
        initial_strata = None
    elif isinstance(raw_initial_strata, dict):
        initial_strata = dict(raw_initial_strata)
        if initial is None or initial_strata.get("metric_scores") != initial:
            raise ValueError("resume checkpoint initial strata do not match validation baseline metrics")
    else:
        raise ValueError("resume checkpoint initial validation strata must be a mapping or null")
    if raw_best is None:
        best_score = float("inf")
    elif isinstance(raw_best, (int, float)) and torch.isfinite(torch.tensor(float(raw_best))):
        best_score = float(raw_best)
    else:
        raise ValueError("resume checkpoint best validation score must be finite or null")
    if not isinstance(raw_history, list) or not all(isinstance(item, dict) for item in raw_history):
        raise ValueError("resume checkpoint selection history must be a list of mappings")
    history = [dict(item) for item in raw_history]
    if bool(history) != (best_score < float("inf")):
        raise ValueError("resume checkpoint best score and selection history are inconsistent")
    return initial, initial_strata, best_score, history


def checkpoint_runtime_payload(
    batcher: WindowedOnlineGeometryBatcher,
    window: ResidentGeometryAssetWindow,
    *,
    initial_validation_metrics: dict[str, float] | None,
    initial_validation_strata: dict[str, object] | None,
    best_validation_score: float,
    selection_history: list[dict[str, object]],
) -> dict[str, object]:
    r"""构造完整 optimizer-boundary runtime/selection/RNG payload。"""

    state = batcher.state_dict()
    return {
        "epoch": state.epoch,
        "block_index": state.block_index,
        "resident_asset_ids": window.resident_asset_ids,
        "batcher_state": state.batcher_state,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        "initial_validation_metrics": initial_validation_metrics,
        "initial_validation_strata": initial_validation_strata,
        "best_validation_score": None if best_validation_score == float("inf") else best_validation_score,
        "selection_history": selection_history,
    }


def best_step_from_selection_history(history: list[dict[str, object]]) -> int | None:
    r"""返回 historical score 最小的 immutable best checkpoint step。"""

    if not history:
        return None
    candidates: list[tuple[float, int]] = []
    for item in history:
        score = item.get("score")
        step = item.get("step")
        if not isinstance(score, (int, float)) or not isinstance(step, int):
            raise ValueError("selection history entries require numeric score and integer step")
        candidates.append((float(score), step))
    return min(candidates)[1]


def publish_best_checkpoint(best_path: Path, immutable_path: Path) -> None:
    r"""把 immutable `best_step_*.pt` 以原子 hard-link 名 `best.pt` 发布。"""

    temporary = best_path.with_suffix(best_path.suffix + ".link.tmp")
    temporary.unlink(missing_ok=True)
    temporary.hardlink_to(immutable_path)  # 同目录同文件系统，共享 checkpoint inode
    temporary.replace(best_path)


__all__ = [
    "best_step_from_selection_history",
    "checkpoint_runtime_payload",
    "publish_best_checkpoint",
    "require_resume_scientific_config",
    "restore_validation_selection_state",
]
