r"""Geometry SSL runtime 的 resume 科学合同与 checkpoint-selection lineage。

底层 tensor payload 的原子读写由 ``ssl.checkpoint`` 拥有；本模块只定义 runtime 必须恢复的
minibatch/Sobol/RNG/initialization-baseline/historical-best 状态，并拒绝当前 CLI 与 checkpoint
之间的科学配置或 asset manifest 漂移。
"""

from __future__ import annotations

from pathlib import Path  # immutable best checkpoint 与 mutable best.pt 发布路径

import torch  # RNG states 与有限性验证

from anymani.distill.ssl.experiment import EmbodimentPretrainCfg, resolved_config_dict


def resume_scientific_config(config: EmbodimentPretrainCfg | dict[str, object]) -> dict[str, object]:
    r"""返回 resume 必须一致的科学配置，只排除 output/resume 定位。"""

    payload = resolved_config_dict(config) if isinstance(config, EmbodimentPretrainCfg) else dict(config)
    run = payload.get("run")
    if not isinstance(run, dict):
        raise ValueError("resolved geometry SSL config lacks run mapping")
    payload["run"] = {
        key: value for key, value in run.items() if key not in {"output_dir", "experiment_name", "resume_checkpoint"}
    }  # seed/deterministic_algorithms 属于科学轨迹，只排除 artifact 定位字段
    return payload


def require_resume_scientific_config(
    current: EmbodimentPretrainCfg | dict[str, object],
    checkpoint_resolved: dict[str, object],
) -> None:
    r"""拒绝当前 CLI 与 checkpoint 的任一 scientific config 漂移。"""

    schema = checkpoint_resolved.get("schema_version")
    if schema != "6.0.0":
        raise ValueError("resume checkpoint must contain schema 6 resolved configuration")
    expected = resume_scientific_config(checkpoint_resolved)
    actual = resume_scientific_config(current)
    if actual != expected:
        changed_sections = tuple(key for key in expected.keys() | actual.keys() if expected.get(key) != actual.get(key))
        raise ValueError(f"resume scientific config mismatch in sections={changed_sections}")


def require_resume_calibration_hash(current_hash: str, checkpoint_metadata: dict[str, object]) -> None:
    r"""拒绝同一路径内容变化或 CLI calibration artifact 漂移。"""

    recorded_hash = checkpoint_metadata.get("calibration_artifact_hash")
    if not isinstance(recorded_hash, str):
        raise ValueError("resume checkpoint lacks calibration artifact hash lineage")
    if current_hash != recorded_hash:
        raise ValueError("resume calibration artifact hash does not match checkpoint lineage")


def restore_validation_selection_state(
    runtime_payload: dict[str, object],
) -> tuple[dict[str, dict[str, float]] | None, dict[str, object] | None, float, list[dict[str, object]]]:
    r"""恢复 initialization strata、normalization baseline 与 historical best score/history。"""

    raw_initial = runtime_payload.get("initial_validation_metrics")
    raw_initial_strata = runtime_payload.get("initial_validation_strata")
    raw_best = runtime_payload.get("best_validation_score")
    raw_history = runtime_payload.get("selection_history")
    if raw_initial is None:
        initial = None
    elif isinstance(raw_initial, dict):
        expected_metrics = {"density", "kappa", "derived_field"}
        initial = {}
        for suite_name, raw_metrics in raw_initial.items():
            if not isinstance(raw_metrics, dict) or set(raw_metrics) != expected_metrics:
                raise ValueError("resume checkpoint validation baseline has invalid suite metric keys")
            initial[str(suite_name)] = {str(name): float(value) for name, value in raw_metrics.items()}
        if not initial or any(
            not torch.isfinite(torch.tensor(value)) or value <= 0.0
            for suite_metrics in initial.values()
            for value in suite_metrics.values()
        ):
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


def best_epoch_from_selection_history(history: list[dict[str, object]]) -> int | None:
    r"""返回 historical score 最小的 immutable best checkpoint epoch。"""

    if not history:
        return None
    candidates: list[tuple[float, int]] = []
    for item in history:
        score = item.get("score")
        epoch = item.get("epoch")
        if not isinstance(score, (int, float)) or not isinstance(epoch, int):
            raise ValueError("selection history entries require numeric score and integer epoch")
        candidates.append((float(score), epoch))
    return min(candidates)[1]


def publish_best_checkpoint(best_path: Path, immutable_path: Path) -> None:
    r"""把 immutable `best_epoch_*.pt` 以原子 hard-link 名 `best.pt` 发布。"""

    temporary = best_path.with_suffix(best_path.suffix + ".link.tmp")
    temporary.unlink(missing_ok=True)
    temporary.hardlink_to(immutable_path)  # 同目录同文件系统，共享 checkpoint inode
    temporary.replace(best_path)


__all__ = [
    "best_epoch_from_selection_history",
    "publish_best_checkpoint",
    "require_resume_calibration_hash",
    "require_resume_scientific_config",
    "restore_validation_selection_state",
]
