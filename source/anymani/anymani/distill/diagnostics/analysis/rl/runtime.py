r"""从 RL JSONL evidence 生成资源峰值和最后阶段摘要。

本模块不 import IsaacLab、策略或 trainer。输入 artifact 即使来自失败 run，只要 JSONL 中已有完整行，
仍可恢复最后活动阶段、峰值 RSS/swap/显存和阶段完成集合。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    r"""读取 append-only JSONL 的全部完整记录；空文件返回空列表。"""

    if not path.is_file():
        return []  # failed-before-first-event 也是可解释的 artifact 状态
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL record at {path}:{line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"JSONL record at {path}:{line_number} must be an object")
        records.append(record)
    return records


def _optional_max(records: list[dict[str, Any]], field: str) -> int | float | None:
    r"""返回所有非空数值字段的最大值；没有观测时返回 ``None``。"""

    values = [record[field] for record in records if isinstance(record.get(field), (int, float))]
    return max(values) if values else None


def _phase_details(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    r"""配对每个 phase 的 start/complete，保留输入规模、结果指标与阶段墙钟。"""

    starts: dict[str, dict[str, Any]] = {}
    details: dict[str, dict[str, Any]] = {}
    excluded = {"event", "phase", "pid", "schema_version", "utc"}
    for record in records:
        phase = record.get("phase")
        if not isinstance(phase, str):
            continue
        if record.get("event") == "start":
            starts[phase] = record
            continue
        if record.get("event") != "complete":
            continue
        start = starts.get(phase, {})
        detail = {key: value for key, value in start.items() if key not in excluded and key != "elapsed_seconds"}
        detail.update({key: value for key, value in record.items() if key not in excluded and key != "elapsed_seconds"})
        start_elapsed = start.get("elapsed_seconds")
        complete_elapsed = record.get("elapsed_seconds")
        if isinstance(start_elapsed, (int, float)) and isinstance(complete_elapsed, (int, float)):
            detail["duration_seconds"] = float(complete_elapsed) - float(start_elapsed)
        details[phase] = detail
    return details


def summarize_rl_runtime_artifacts(run_dir: Path | str) -> dict[str, Any]:
    r"""汇总一次 RL run 的阶段完成集和资源峰值。

    Args:
        run_dir (Path | str): 包含 ``phase_events.jsonl`` 与 ``resource_samples.jsonl`` 的目录。

    Returns:
        dict[str, Any]: 可直接合入 ``summary.yaml`` 的只读充分统计。
    """

    root = Path(run_dir).expanduser()  # 只读 artifact 根
    phases = _read_jsonl(root / "phase_events.jsonl")
    resources = _read_jsonl(root / "resource_samples.jsonl")
    completed = [
        str(record["phase"])
        for record in phases
        if record.get("event") == "complete" and isinstance(record.get("phase"), str)
    ]  # 保持运行时完成顺序，不做 set 排序
    failures = [record for record in phases if record.get("event") == "failed"]
    last_phase = phases[-1].get("phase") if phases else None  # OOM 前最后主动阶段
    peak_tree_rss = _optional_max(resources, "process_tree_peak_rss_bytes")
    peak_tree_current_rss = _optional_max(resources, "process_tree_rss_bytes")
    peak_tree_swap = _optional_max(resources, "process_tree_swap_bytes")
    return {
        "phase_event_count": len(phases),
        "resource_sample_count": len(resources),
        "completed_phases": completed,
        "completed_phase_details": _phase_details(phases),
        "last_phase": last_phase,
        "failure": failures[-1] if failures else None,
        "peak_process_rss_bytes": (
            peak_tree_rss if peak_tree_rss is not None else _optional_max(resources, "process_peak_rss_bytes")
        ),  # benchmark 优先 shell+Python/Kit 进程树；旧 artifact 保留单 PID fallback
        "peak_process_tree_current_rss_bytes": peak_tree_current_rss,
        "peak_process_tree_hwm_sum_bytes": peak_tree_rss,
        "peak_process_swap_bytes": (
            peak_tree_swap if peak_tree_swap is not None else _optional_max(resources, "process_swap_bytes")
        ),
        "peak_process_tree_pid_count": _optional_max(resources, "process_tree_pid_count"),
        "peak_gpu_process_memory_bytes": _optional_max(resources, "gpu_process_memory_bytes"),
        "minimum_system_available_ram_bytes": (
            min(
                record["system_available_ram_bytes"]
                for record in resources
                if isinstance(record.get("system_available_ram_bytes"), (int, float))
            )
            if any(isinstance(record.get("system_available_ram_bytes"), (int, float)) for record in resources)
            else None
        ),
    }


__all__ = ["summarize_rl_runtime_artifacts"]
