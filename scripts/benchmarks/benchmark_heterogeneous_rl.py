#!/usr/bin/env python3
r"""以独立父进程运行 heterogeneous RL 命令并保存失败包络。

本脚本不 import IsaacLab，也不重写环境或 PPO。``--`` 后的命令是唯一被测对象；父进程只负责：

1. 保存实际 argv、工作目录与采样配置；
2. 把子进程 stdout/stderr 原样写入 run directory；
3. 低频采样 ``/proc/<pid>`` 与 ``nvidia-smi`` 的 per-process 显存；
4. 在成功、非零退出或 timeout 后发布结构化 ``summary.yaml``。

示例：

```bash
python scripts/benchmarks/benchmark_heterogeneous_rl.py \
  --output_dir logs/benchmarks/heterogeneous_rl/smoke \
  --timeout_s 1800 -- \
  /home/hac/isaac/IsaacLab/isaaclab.sh -p scripts/research/train_hetero_structured_ppo.py \
  --tier support_basin --num-envs 8 --updates 1 --horizon 16 --eval-steps 20
```
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
from pathlib import Path

from anymani.distill.diagnostics.analysis.rl import summarize_rl_runtime_artifacts
from anymani.distill.diagnostics.recording.rl import RlRunRecorder, read_linux_process_resources


def _descendant_pids(root_pid: int) -> set[int]:
    r"""读取 Linux ``/proc`` 并返回 root shell 的存活进程树 PID 集合。"""

    parent_by_pid: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text(encoding="utf-8").split()
            parent_by_pid[int(entry.name)] = int(fields[3])  # `/proc/pid/stat` 第四字段 ppid
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
            continue
    descendants = {int(root_pid)}
    changed = True
    while changed:
        changed = False
        for pid, parent in parent_by_pid.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return descendants


def _gpu_process_memory_bytes(pids: set[int]) -> int | None:
    r"""聚合目标进程树的 NVML 显存；查询失败时返回 ``None``。

    Args:
        pids (set[int]): ``isaaclab.sh`` shell 与其 Python/Kit descendants。

    Returns:
        int | None: ``nvidia-smi`` 报告的 MiB 换算为 bytes；进程尚未建立 CUDA context 时为 ``None``。
    """

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )  # 父进程低频查询，不进入被测 CUDA stream
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    total_mib = 0  # 同一 PID 可能由驱动报告多条 device/context 记录
    matched = False
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            record_pid, used_mib = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        if record_pid in pids:
            total_mib += used_mib  # MiB，保持 nvidia-smi 进程口径
            matched = True
    return total_mib * 1024 * 1024 if matched else None  # MiB -> bytes


def _parse_args() -> argparse.Namespace:
    r"""解析父进程采样参数和 ``--`` 后的原始子命令。"""

    parser = argparse.ArgumentParser(description="Record a heterogeneous RL command and its resource envelope.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Unique benchmark evidence directory.")
    parser.add_argument("--sample_period_s", type=float, default=1.0, help="Parent resource sample period.")
    parser.add_argument("--timeout_s", type=float, default=0.0, help="Wall timeout; 0 disables timeout.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command after --.")
    args = parser.parse_args()
    if args.sample_period_s <= 0.0:
        parser.error("--sample_period_s must be positive")
    if args.timeout_s < 0.0:
        parser.error("--timeout_s must be non-negative")
    args.command = list(args.command)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]  # argparse REMAINDER 可能保留分隔符
    if not args.command:
        parser.error("a child command is required after --")
    return args


def main() -> int:
    r"""运行子命令、采样资源并返回相同的成功/失败退出语义。

    Returns:
        int: 子进程退出码；timeout 使用 124，与常用 shell ``timeout`` 语义一致。
    """

    args = _parse_args()
    output_dir = args.output_dir.expanduser().resolve()  # benchmark 证据使用绝对路径锚定
    recorder = RlRunRecorder(
        output_dir,
        {
            "benchmark": "heterogeneous_rl_parent",
            "command": args.command,
            "cwd": os.getcwd(),
            "sample_period_s": float(args.sample_period_s),
            "timeout_s": float(args.timeout_s),
        },
    )
    stdout_path = output_dir / "stdout.log"  # 被测程序原始 stdout，不在 recorder 内重新解释
    stderr_path = output_dir / "stderr.log"  # PhysX/Kit warning 与 traceback 的事实源
    recorder.record_phase("child_process", "start", command=args.command)
    started = time.monotonic()  # timeout 和总运行时间使用父进程单调时钟
    timed_out = False

    # 新 process group 允许 timeout 时同时终止 isaaclab.sh 派生的 Python/Kit 进程。
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        child_env = {
            **os.environ,
            "ANYMANI_RL_EVIDENCE_DIR": str(output_dir),
            "ANYMANI_RL_STARTED_MONOTONIC_NS": str(recorder.started_monotonic_ns),
        }  # child phase events 与 parent resource samples 共用同一 run/time origin
        process = subprocess.Popen(
            args.command,
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=True,
            env=child_env,
        )
        while process.poll() is None:
            process_tree = _descendant_pids(process.pid)  # shell + Python/Kit 当前存活 descendants
            tree_resources = [read_linux_process_resources(pid) for pid in process_tree]
            recorder.record_resources(
                process.pid,
                phase="child_process",
                gpu_process_memory_bytes=_gpu_process_memory_bytes(process_tree),
                process_tree_pid_count=len(process_tree),
                process_tree_rss_bytes=sum(int(item.get("process_rss_bytes") or 0) for item in tree_resources),
                process_tree_peak_rss_bytes=sum(
                    int(item.get("process_peak_rss_bytes") or 0) for item in tree_resources
                ),
                process_tree_swap_bytes=sum(int(item.get("process_swap_bytes") or 0) for item in tree_resources),
            )
            if args.timeout_s > 0.0 and time.monotonic() - started >= args.timeout_s:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)  # 先给 Kit 正常关闭窗口
                try:
                    process.wait(timeout=15.0)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)  # 超过关闭窗口后回收整个 group
                break
            time.sleep(args.sample_period_s)  # 父进程低频采样；不轮询 CUDA event
        return_code = process.wait()

    # 子进程退出后的最后 partial sample 保留 process_alive=False 与系统余量。
    recorder.record_resources(process.pid, phase="child_process", gpu_process_memory_bytes=None)
    effective_code = 124 if timed_out else int(return_code)  # 常用 timeout 可识别退出码
    event = "complete" if effective_code == 0 else "failed"
    recorder.record_phase(
        "child_process",
        event,
        return_code=effective_code,
        timed_out=timed_out,
        wall_seconds=time.monotonic() - started,
    )
    artifact_summary = summarize_rl_runtime_artifacts(output_dir)  # 只读已落盘 JSONL
    recorder.write_summary(
        {
            "status": "passed" if effective_code == 0 else "failed",
            "return_code": effective_code,
            "timed_out": timed_out,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            **artifact_summary,
        }
    )
    return effective_code


if __name__ == "__main__":
    raise SystemExit(main())
