r"""RL 基础设施 benchmark 的 append-only 运行证据。

一次 heterogeneous RL run 同时涉及 Python/Kit 启动、资产准备、USD cache、Stage、PhysX、
MDP rollout 与 PPO update。单一总耗时无法定位失败，因此本模块保存两个互不混淆的时间轴：

* ``phase_events.jsonl``：被测进程主动声明当前阶段和阶段耗时；
* ``resource_samples.jsonl``：父进程按固定低频率采样目标 PID 的 RSS、swap、I/O 与显存。

采样代码不进入 policy/physics hot path。即使被测进程被 OOM killer、timeout 或 driver 终止，
父进程仍可写出最后资源水位和退出状态，形成可审计的失败包络。
"""

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
"""RL runtime evidence 的结构版本；与 SSL artifact schema 相互独立。"""


def _utc_now() -> str:
    r"""返回带 UTC 时区的 ISO-8601 墙钟，供跨进程事件对齐。"""

    return datetime.now(UTC).isoformat()  # 人类可读墙钟；性能差值始终使用 monotonic time


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    r"""把一条 JSON 基础类型记录原子追加为单行。

    Args:
        path (Path): append-only JSONL 路径。
        payload (Mapping[str, Any]): 已规约为 JSON-compatible 类型的记录。
    """

    path.parent.mkdir(parents=True, exist_ok=True)  # recorder 可以在 run directory 创建前实例化
    with path.open("a", encoding="utf-8") as stream:  # 单 writer 逐行追加，保留异常前完整记录
        stream.write(json.dumps(dict(payload), sort_keys=True) + "\n")  # 每行可独立恢复与审计
        stream.flush()  # OOM/timeout 前尽量把最后阶段交给文件系统


def record_optional_rl_phase(phase: str, event: str, **fields: Any) -> None:
    r"""若父 benchmark 注入 evidence directory，则由 child 追加阶段事件。

    普通训练未设置 ``ANYMANI_RL_EVIDENCE_DIR`` 时该函数是无副作用空操作。benchmark 父进程同时
    注入 ``ANYMANI_RL_STARTED_MONOTONIC_NS``，使 shell、Python/Kit child 的 elapsed time 共用
    同一原点；子进程异常退出前已 flush 的最后阶段仍可恢复。

    Args:
        phase (str): ``asset_prepare``、``environment_construct``、``frozen_z``、``ppo_train`` 等稳定阶段。
        event (str): ``start``、``complete`` 或 ``failed``。
        **fields: JSON-safe shape、count、duration 或 identity 摘要。
    """

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


def _read_kib_fields(path: Path, names: set[str]) -> dict[str, int]:
    r"""读取 Linux ``key: value kB`` 文件并统一换算为 bytes。

    Args:
        path (Path): 例如 ``/proc/<pid>/status`` 或 ``/proc/meminfo``。
        names (set[str]): 需要的字段名，不含冒号。

    Returns:
        dict[str, int]: 字段到 bytes 的映射；进程退出或字段缺失时返回已读子集。
    """

    values: dict[str, int] = {}  # Linux kB 字段到 bytes；不把缺失误写成真实零值
    try:
        lines = path.read_text(encoding="utf-8").splitlines()  # 低频父进程采样，不进入 CUDA hot path
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return values  # 目标进程可能恰好退出；调用方仍保存 sample timestamp
    for line in lines:
        name, separator, remainder = line.partition(":")  # ``VmRSS: 1234 kB``
        if not separator or name not in names:
            continue
        tokens = remainder.strip().split()  # 第一项为数值，第二项通常为 kB
        if tokens:
            values[name] = int(tokens[0]) * 1024  # Linux proc 的 kB 口径换算为 bytes
    return values


def read_linux_process_resources(pid: int) -> dict[str, int | bool | None]:
    r"""读取一个 Linux 进程及系统内存的无 CUDA 同步资源快照。

    本函数只读取 ``/proc``，不调用 PyTorch、Warp、NVML 或 ``nvidia-smi``。GPU 进程显存由
    benchmark 父进程按其可用 backend 解析后传给 :meth:`RlRunRecorder.record_resources`，
    避免把可选 GPU 依赖写入通用 evidence recorder。

    Args:
        pid (int): 被测 Isaac/rl_games 进程 PID，必须为正整数。

    Returns:
        dict[str, int | bool | None]: 当前/峰值 RSS、进程 swap、I/O、系统 available RAM 与 swap。
    """

    if pid <= 0:
        raise ValueError("pid must be a positive integer")
    proc_root = Path("/proc") / str(pid)  # 目标进程的 Linux proc 根
    process = _read_kib_fields(proc_root / "status", {"VmRSS", "VmHWM", "VmSwap"})
    system = _read_kib_fields(Path("/proc/meminfo"), {"MemAvailable", "SwapTotal", "SwapFree"})
    io_values: dict[str, int] = {}  # 进程累计读写 bytes；缺权限或已退出时保持空
    try:
        for line in (proc_root / "io").read_text(encoding="utf-8").splitlines():
            name, separator, value = line.partition(":")
            if separator and name in {"read_bytes", "write_bytes"}:
                io_values[name] = int(value.strip())  # Linux block-I/O 真实 bytes，不是 syscall 字节
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
    return {
        "process_alive": proc_root.exists(),  # sample 时 PID 是否仍有 proc entry
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
    r"""记录一次 RL benchmark/train run 的 identity、阶段与资源事实。

    ``started_monotonic_ns`` 只属于当前进程，用于精确计算相对耗时；``utc`` 用于父子进程
    对齐。Recorder 不决定 run 是否成功，调用方在 shutdown 或异常边界通过 ``write_summary``
    写入最终结论。
    """

    def __init__(self, output_dir: Path | str, identity: Mapping[str, Any]) -> None:
        r"""创建 run directory 并写入不可变 identity。

        Args:
            output_dir (Path | str): 当前 run 的唯一证据目录。
            identity (Mapping[str, Any]): 代码、数据集、route、环境数、seed 与版本身份。
        """

        self.output_dir = Path(output_dir).expanduser()  # run-local evidence 根，不依赖 shell cwd
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.phase_path = self.output_dir / "phase_events.jsonl"  # 被测进程主动阶段时间轴
        self.resource_path = self.output_dir / "resource_samples.jsonl"  # 父进程低频资源时间轴
        self.summary_path = self.output_dir / "summary.yaml"  # agent/人类优先读取的一页结果
        self._started_monotonic_ns = time.monotonic_ns()  # 进程内单调时钟原点，不受墙钟校时影响
        identity_payload = {
            "schema_version": RL_RUNTIME_EVIDENCE_SCHEMA_VERSION,
            "created_utc": _utc_now(),
            **dict(identity),
        }
        (self.output_dir / "identity.yaml").write_text(
            yaml.safe_dump(identity_payload, sort_keys=True, allow_unicode=True),
            encoding="utf-8",
        )  # identity 在首个 expensive phase 前落盘

    @property
    def elapsed_seconds(self) -> float:
        r"""返回 recorder 创建以来的单调墙钟秒数。"""

        return (time.monotonic_ns() - self._started_monotonic_ns) / 1.0e9  # ns -> s

    @property
    def started_monotonic_ns(self) -> int:
        r"""返回父子进程共享 evidence 时间轴的单调时钟原点。"""

        return self._started_monotonic_ns

    def record_phase(self, phase: str, event: str, **fields: Any) -> None:
        r"""追加一次阶段开始、完成或失败事件。

        Args:
            phase (str): 稳定阶段名，例如 ``asset_prepare``、``physx_start``、``ppo_update``。
            event (str): ``start``、``complete``、``failed`` 或调用方定义的窄事件名。
            **fields: 阶段耗时、资产数、cache hit/miss、异常类型等 JSON 基础字段。
        """

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
        r"""追加目标进程的低频资源快照。

        Args:
            pid (int): 被测进程 PID；父进程可以记录与自身不同的 PID。
            phase (str | None): 采样时最近活动阶段。
            gpu_process_memory_bytes (int | None): NVML/nvidia-smi 口径的目标进程显存 bytes。
            **fields: GPU utilization、温度或父进程判定等可选基础字段。
        """

        resources = read_linux_process_resources(pid)  # 仅 /proc，目标退出时返回 partial sample
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
        r"""原子发布一次 run 的一页式 YAML 结论。

        Args:
            summary (Mapping[str, Any]): 成功层级、失败阶段、容量、耗时、峰值资源与吞吐结论。

        Returns:
            Path: 已发布的 ``summary.yaml`` 路径。
        """

        payload = {
            "schema_version": RL_RUNTIME_EVIDENCE_SCHEMA_VERSION,
            "finalized_utc": _utc_now(),
            "elapsed_seconds": self.elapsed_seconds,
            **dict(summary),
        }
        temporary = self.summary_path.with_suffix(".yaml.tmp")  # 同文件系统临时文件，支持原子 replace
        temporary.write_text(yaml.safe_dump(payload, sort_keys=True, allow_unicode=True), encoding="utf-8")
        temporary.replace(self.summary_path)  # 读者只会看到完整旧版或完整新版
        return self.summary_path


__all__ = [
    "RL_RUNTIME_EVIDENCE_SCHEMA_VERSION",
    "RlRunRecorder",
    "record_optional_rl_phase",
    "read_linux_process_resources",
]
