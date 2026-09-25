'Record simulator and PPO phases, resource snapshots, and process failures.'

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

RL_RUNTIME_EVIDENCE_SCHEMA_VERSION = "1.0.0"
'Definition for RL runtime evidence schema version (1.0.0).'

FATAL_RUNTIME_PATTERNS = (
    "Scene state is corrupted",
    "compressContactStage",
    "getRigidDynamicData: CUDA error",
    "CUDA error, code 2",
    "palm-rotation training exhausted CUDA driver headroom:",
    "palm-rotation training exceeded PyTorch allocated safety fraction:",
)
'Definition for fatal runtime patterns.'


def _utc_now() -> str:
    'Handle utc now.'

    return datetime.now(UTC).isoformat()


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    'Append jsonl.'

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(payload), sort_keys=True) + "\n")
        stream.flush()


def record_optional_rl_phase(phase: str, event: str, **fields: Any) -> None:
    'Record optional RL phase.'

    output_dir = os.environ.get("ANYMANI_RL_EVIDENCE_DIR")
    if not output_dir:
        return
    if not phase or not event:
        raise ValueError("phase and event must be non-empty")
    raw_start = os.environ.get("ANYMANI_RL_STARTED_MONOTONIC_NS")
    started_ns = int(raw_start) if raw_start else time.monotonic_ns()  # standalone child fallback
    _append_jsonl(
        Path(output_dir).expanduser() / "phase_events.jsonl",
        {
            "schema_version": RL_RUNTIME_EVIDENCE_SCHEMA_VERSION,
            "utc": _utc_now(),
            "elapsed_seconds": (time.monotonic_ns() - started_ns) / 1.0e9,
            "pid": os.getpid(),
            "phase": phase,
            "event": event,
            **fields,
        },
    )


def scan_appended_fatal_log(path: Path, previous_size: int) -> tuple[int, str | None]:
    'Handle scan appended fatal log.'

    try:
        current_size = path.stat().st_size
        start = max(0, min(previous_size, current_size) - 512)
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            stream.seek(start)
            appended = stream.read()
    except FileNotFoundError:
        return previous_size, None
    for line in appended.splitlines():
        if any(pattern in line for pattern in FATAL_RUNTIME_PATTERNS):
            return current_size, line.strip()
    return current_size, None


def _read_kib_fields(path: Path, names: set[str]) -> dict[str, int]:
    'Read kib fields; shapes [str], [str,int].'

    values: dict[str, int] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return values
    for line in lines:
        name, separator, remainder = line.partition(":")  # ``VmRSS: 1234 kB``
        if not separator or name not in names:
            continue
        tokens = remainder.strip().split()
        if tokens:
            values[name] = int(tokens[0]) * 1024
    return values


def read_linux_process_resources(pid: int) -> dict[str, int | bool | None]:
    'Read linux process resources.'

    if pid <= 0:
        raise ValueError("pid must be a positive integer")
    proc_root = Path("/proc") / str(pid)
    process = _read_kib_fields(proc_root / "status", {"VmRSS", "VmHWM", "VmSwap"})
    system = _read_kib_fields(Path("/proc/meminfo"), {"MemAvailable", "SwapTotal", "SwapFree"})
    io_values: dict[str, int] = {}
    try:
        for line in (proc_root / "io").read_text(encoding="utf-8").splitlines():
            name, separator, value = line.partition(":")
            if separator and name in {"read_bytes", "write_bytes"}:
                io_values[name] = int(value.strip())
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
    return {
        "process_alive": proc_root.exists(),
        "process_rss_bytes": process.get("VmRSS"),
        "process_peak_rss_bytes": process.get("VmHWM"),
        "process_swap_bytes": process.get("VmSwap"),
        "process_read_bytes": io_values.get("read_bytes"),
        "process_write_bytes": io_values.get("write_bytes"),
        "system_available_ram_bytes": system.get("MemAvailable"),
        "system_swap_total_bytes": system.get("SwapTotal"),
        "system_swap_free_bytes": system.get("SwapFree"),
    }


class RlRunRecorder:
    'Contract for RL run recorder.'

    def __init__(self, output_dir: Path | str, identity: Mapping[str, Any]) -> None:
        'Initialize the instance.'

        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.phase_path = self.output_dir / "phase_events.jsonl"
        self.resource_path = self.output_dir / "resource_samples.jsonl"
        self.summary_path = self.output_dir / "summary.yaml"
        self._started_monotonic_ns = time.monotonic_ns()
        identity_payload = {
            "schema_version": RL_RUNTIME_EVIDENCE_SCHEMA_VERSION,
            "created_utc": _utc_now(),
            **dict(identity),
        }
        (self.output_dir / "identity.yaml").write_text(
            yaml.safe_dump(identity_payload, sort_keys=True, allow_unicode=True),
            encoding="utf-8",
        )

    @property
    def elapsed_seconds(self) -> float:
        'Handle elapsed seconds.'

        return (time.monotonic_ns() - self._started_monotonic_ns) / 1.0e9  # ns -> s

    @property
    def started_monotonic_ns(self) -> int:
        'Handle started monotonic ns.'

        return self._started_monotonic_ns

    def record_phase(self, phase: str, event: str, **fields: Any) -> None:
        'Record phase.'

        if not phase or not event:
            raise ValueError("phase and event must be non-empty")
        _append_jsonl(
            self.phase_path,
            {
                "schema_version": RL_RUNTIME_EVIDENCE_SCHEMA_VERSION,
                "utc": _utc_now(),
                "elapsed_seconds": self.elapsed_seconds,
                "pid": os.getpid(),
                "phase": phase,
                "event": event,
                **fields,
            },
        )

    def record_resources(
        self,
        pid: int,
        *,
        phase: str | None = None,
        gpu_process_memory_bytes: int | None = None,
        **fields: Any,
    ) -> None:
        'Record resources.'

        resources = read_linux_process_resources(pid)
        _append_jsonl(
            self.resource_path,
            {
                "schema_version": RL_RUNTIME_EVIDENCE_SCHEMA_VERSION,
                "utc": _utc_now(),
                "elapsed_seconds": self.elapsed_seconds,
                "pid": int(pid),
                "phase": phase,
                "gpu_process_memory_bytes": gpu_process_memory_bytes,
                **resources,
                **fields,
            },
        )

    def write_summary(self, summary: Mapping[str, Any]) -> Path:
        'Write summary.'

        payload = {
            "schema_version": RL_RUNTIME_EVIDENCE_SCHEMA_VERSION,
            "finalized_utc": _utc_now(),
            "elapsed_seconds": self.elapsed_seconds,
            **dict(summary),
        }
        temporary = self.summary_path.with_suffix(".yaml.tmp")
        temporary.write_text(yaml.safe_dump(payload, sort_keys=True, allow_unicode=True), encoding="utf-8")
        temporary.replace(self.summary_path)
        return self.summary_path


__all__ = [
    "FATAL_RUNTIME_PATTERNS",
    "RL_RUNTIME_EVIDENCE_SCHEMA_VERSION",
    "RlRunRecorder",
    "record_optional_rl_phase",
    "read_linux_process_resources",
    "scan_appended_fatal_log",
]
