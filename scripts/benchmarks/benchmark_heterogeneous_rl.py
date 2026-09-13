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
  /home/hac/isaac/IsaacLab/isaaclab.sh -p -m anymani.distill.rl.scripts.train_hetero_structured_ppo \
  --tier support_basin --num-envs 8 --updates 1 --horizon 16 --eval-steps 20
```
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from anymani.distill.diagnostics.analysis.rl import summarize_rl_runtime_artifacts
from anymani.distill.diagnostics.recording.rl import (
    FATAL_RUNTIME_PATTERNS,
    RlRunRecorder,
    read_linux_process_resources,
    scan_appended_fatal_log,
)


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


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    r"""先SIGTERM再有界SIGKILL回收整个Isaac/Kit process group。"""

    try:
        os.killpg(process.pid, signal.SIGTERM)  # 给Kit最多15秒释放scene与CUDA context
    except ProcessLookupError:
        return  # 子进程可能在日志采样与发送信号之间退出；仍由调用方wait并发布失败。
    try:
        process.wait(timeout=15.0)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)  # C++/driver损坏后不得无限等待析构


def _parse_args() -> argparse.Namespace:
    r"""解析父进程采样参数和 ``--`` 后的原始子命令。"""

    parser = argparse.ArgumentParser(description="Record a heterogeneous RL command and its resource envelope.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Unique benchmark evidence directory.")
    parser.add_argument("--sample_period_s", type=float, default=1.0, help="Parent resource sample period.")
    parser.add_argument("--timeout_s", type=float, default=0.0, help="Wall timeout; 0 disables timeout.")
    parser.add_argument("--live_output", action="store_true", help="Relay durable child logs to the controlling terminal.")
    parser.add_argument(
        "--disable_fatal_watchdog",
        action="store_true",
        help="Disable built-in PhysX/CUDA log termination; intended only for watchdog contract probes.",
    )
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


def _relay_log(path: Path, stop: threading.Event) -> None:
    r"""把原始落盘日志实时送到控制终端；完整配置仍保留在原文件中。

    Python child使用无缓冲输出，终端读取与资源采样互不阻塞。
    启动配置的机器指纹不显示在终端，只显示运行名、规模和奖励核。
    """
    with path.open(encoding="utf-8", errors="replace") as stream:
        while True:
            line = stream.readline()  # 每次发送完整可读行，保留原有FPS与错误信息。
            if line:
                if line.lstrip().startswith('{"actor_init_checkpoint_sha256"'):
                    data = json.loads(line)
                    goal = data.get("orientation_goal") or {}
                    line = f"[CONFIG] N={data.get('num_envs')} H={data.get('horizon')} epochs={data.get('max_updates')} kernel={goal.get('kernel', 'inverse')} run={data.get('run_dir')}\n"
                sys.stdout.write(line)
                sys.stdout.flush()
            elif stop.is_set():
                break  # child已退出且文件尾已排空。
            else:
                stop.wait(.1)  # 仅终端流的低频等待，不占用训练线程。


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
            "fatal_runtime_patterns": [] if args.disable_fatal_watchdog else list(FATAL_RUNTIME_PATTERNS),
        },
    )
    stdout_path = output_dir / "stdout.log"  # 被测程序原始 stdout，不在 recorder 内重新解释
    stderr_path = output_dir / "stderr.log"  # PhysX/Kit warning 与 traceback 的事实源
    recorder.record_phase("child_process", "start", command=args.command)
    started = time.monotonic()  # timeout 和总运行时间使用父进程单调时钟
    timed_out = False
    fatal_runtime_line: str | None = None  # 首个不可恢复C++/CUDA日志；不在损坏后请求checkpoint
    log_sizes = {"stdout": 0, "stderr": 0}  # 两个append-only文件的增量扫描cursor
    interrupts: list[int] = []  # 父进程拥有child的新process group，必须显式转发用户中断。
    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}

    def request_stop(signum: int, _frame: object) -> None:
        r"""只登记中断请求，由监控循环转发并保留最后的资源/退出记录。"""
        interrupts.append(signum)

    for sig in previous_handlers:
        signal.signal(sig, request_stop)
    stop_relay = threading.Event()
    relay_threads: list[threading.Thread] = []
    interrupt_deadline: float | None = None  # 给child退出清理的有界时间。

    # 新 process group 允许 timeout 时同时终止 isaaclab.sh 派生的 Python/Kit 进程。
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        child_env = {
            **os.environ,
            "ANYMANI_RL_EVIDENCE_DIR": str(output_dir),
            "ANYMANI_RL_STARTED_MONOTONIC_NS": str(recorder.started_monotonic_ns),
            "PYTHONUNBUFFERED": "1",  # 文件日志也按行实时落盘，终端可直接查看进度。
        }  # child phase events 与 parent resource samples 共用同一 run/time origin
        process = subprocess.Popen(
            args.command,
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=True,
            env=child_env,
        )
        (output_dir / "supervisor.json").write_text(
            json.dumps({"wrapper_pid": os.getpid(), "child_pid": process.pid}) + "\n"
        )  # 独立终端与外层PTY通过同一监督进程控制唯一训练实例。
        if args.live_output:
            for path in (stdout_path, stderr_path):
                thread = threading.Thread(target=_relay_log, args=(path, stop_relay), daemon=True)
                thread.start()
                relay_threads.append(thread)
            print("[CONTROL] Ctrl+C stops the training process group; completed checkpoints are retained.", flush=True)
        while process.poll() is None:
            if interrupts and interrupt_deadline is None:
                print("\n[CONTROL] Stop requested. Cleaning up training; a second Ctrl+C forces termination.", flush=True)
                recorder.record_phase("user_interrupt", "start", signal=int(interrupts[0]))
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGINT)  # 进入child的finally，避免留下孤儿GPU进程。
                interrupt_deadline = time.monotonic() + 30.0
            if interrupt_deadline is not None and (len(interrupts) > 1 or time.monotonic() >= interrupt_deadline):
                _terminate_process_group(process)
                break
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
            if not args.disable_fatal_watchdog:
                for label, path in (("stdout", stdout_path), ("stderr", stderr_path)):
                    log_sizes[label], matched_line = scan_appended_fatal_log(path, log_sizes[label])
                    if matched_line is not None:
                        fatal_runtime_line = f"{label}: {matched_line}"  # stream provenance + 原始故障行
                        recorder.record_phase(
                            "runtime_watchdog",
                            "failed",
                            fatal_runtime_line=fatal_runtime_line,
                        )
                        _terminate_process_group(process)  # 不让corrupted scene继续产生rollout或optimizer update
                        break
                if fatal_runtime_line is not None:
                    break
            if args.timeout_s > 0.0 and time.monotonic() - started >= args.timeout_s:
                timed_out = True
                _terminate_process_group(process)
                break
            time.sleep(args.sample_period_s)  # 父进程低频采样；不轮询 CUDA event
        return_code = process.wait()
        stop_relay.set()
        for thread in relay_threads:
            thread.join(timeout=3.0)  # 退出后排空终端中的最后错误/统计行。

    failure_path = output_dir / "python_failure.json"
    python_failure = json.loads(failure_path.read_text()) if failure_path.is_file() else None
    if python_failure is not None:
        fatal_runtime_line = f"python: {python_failure['exception_type']}: {python_failure['message']}"
        recorder.record_phase("python_exception", "failed", **python_failure)  # Kit的exit(0)不能覆盖已捕获的异常。

    # 快速失败可能在两次采样之间完成；关闭日志writer后补扫durable尾部，退出码不是唯一证据。
    if not args.disable_fatal_watchdog and fatal_runtime_line is None:
        for label, path in (("stdout", stdout_path), ("stderr", stderr_path)):
            log_sizes[label], matched_line = scan_appended_fatal_log(path, log_sizes[label])
            if matched_line is not None:
                fatal_runtime_line = f"{label}: {matched_line}"
                recorder.record_phase("runtime_watchdog", "failed", fatal_runtime_line=fatal_runtime_line)
                break  # 已退出的进程无需再发信号；保持首个故障的流与原文。

    # 子进程退出后的最后 partial sample 保留 process_alive=False 与系统余量。
    recorder.record_resources(process.pid, phase="child_process", gpu_process_memory_bytes=None)
    effective_code = 128 + interrupts[0] if interrupts else (125 if fatal_runtime_line is not None else (124 if timed_out else int(return_code)))
    event = "complete" if effective_code == 0 else "failed"
    recorder.record_phase(
        "child_process",
        event,
        return_code=effective_code,
        timed_out=timed_out,
        fatal_runtime_line=fatal_runtime_line,
        wall_seconds=time.monotonic() - started,
    )
    artifact_summary = summarize_rl_runtime_artifacts(output_dir)  # 只读已落盘 JSONL
    recorder.write_summary(
        {
            "status": "interrupted" if interrupts else ("passed" if effective_code == 0 else "failed"),
            "return_code": effective_code,
            "timed_out": timed_out,
            "fatal_runtime_line": fatal_runtime_line,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "user_interrupted": bool(interrupts),
            "python_failure": python_failure,
            **artifact_summary,
        }
    )
    for sig, handler in previous_handlers.items():
        signal.signal(sig, handler)
    if args.live_output:
        print(f"[FINISHED] exit={effective_code}; evidence={output_dir}", flush=True)
    return effective_code


if __name__ == "__main__":
    raise SystemExit(main())
