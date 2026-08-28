r"""RL runtime evidence 的纯文件合同；不启动 Isaac Sim 或 CUDA。"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from anymani.distill.diagnostics.analysis.rl import summarize_rl_runtime_artifacts
from anymani.distill.diagnostics.recording.rl import RlRunRecorder, read_linux_process_resources


def test_recorder_writes_identity_events_resources_and_atomic_summary(tmp_path: Path) -> None:
    r"""一次成功 run 应交付四种可独立读取的结构化证据。"""

    recorder = RlRunRecorder(tmp_path, {"task": "heterogeneous", "num_envs": 8})
    recorder.record_phase("asset_prepare", "start", unique_assets=8)
    recorder.record_phase("asset_prepare", "complete", cache_hits=8, cache_misses=0)
    recorder.record_resources(recorder_pid := __import__("os").getpid(), phase="asset_prepare", gpu_process_memory_bytes=7)
    summary_path = recorder.write_summary({"status": "passed", "pid": recorder_pid})

    identity = yaml.safe_load((tmp_path / "identity.yaml").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (tmp_path / "phase_events.jsonl").read_text().splitlines()]
    resources = [json.loads(line) for line in (tmp_path / "resource_samples.jsonl").read_text().splitlines()]
    summary = yaml.safe_load(summary_path.read_text(encoding="utf-8"))

    assert identity["task"] == "heterogeneous"
    assert [event["event"] for event in events] == ["start", "complete"]
    assert resources[0]["pid"] == recorder_pid
    assert resources[0]["gpu_process_memory_bytes"] == 7
    assert summary["status"] == "passed"


def test_artifact_analysis_preserves_last_failure_and_resource_peaks(tmp_path: Path) -> None:
    r"""失败 run 也必须保留最后阶段和各资源口径的峰值。"""

    (tmp_path / "phase_events.jsonl").write_text(
        "\n".join(
            (
                json.dumps({"phase": "stage", "event": "complete"}),
                json.dumps({"phase": "physx_start", "event": "failed", "error_type": "RuntimeError"}),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "resource_samples.jsonl").write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "process_tree_rss_bytes": 6,
                        "process_tree_peak_rss_bytes": 12,
                        "process_tree_swap_bytes": 1,
                        "process_peak_rss_bytes": 10,
                        "process_swap_bytes": 2,
                        "gpu_process_memory_bytes": 4,
                        "system_available_ram_bytes": 100,
                    }
                ),
                json.dumps(
                    {
                        "process_tree_rss_bytes": 9,
                        "process_tree_peak_rss_bytes": 24,
                        "process_tree_swap_bytes": 4,
                        "process_peak_rss_bytes": 20,
                        "process_swap_bytes": 3,
                        "gpu_process_memory_bytes": 8,
                        "system_available_ram_bytes": 90,
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_rl_runtime_artifacts(tmp_path)

    assert summary["completed_phases"] == ["stage"]
    assert summary["last_phase"] == "physx_start"
    assert summary["failure"]["error_type"] == "RuntimeError"
    assert summary["peak_process_rss_bytes"] == 24
    assert summary["peak_process_tree_current_rss_bytes"] == 9
    assert summary["peak_process_tree_hwm_sum_bytes"] == 24
    assert summary["peak_process_swap_bytes"] == 4
    assert summary["peak_gpu_process_memory_bytes"] == 8
    assert summary["minimum_system_available_ram_bytes"] == 90


def test_artifact_analysis_pairs_phase_duration_and_metrics(tmp_path: Path) -> None:
    r"""一页 summary 应直接暴露 phase 耗时、规模与 throughput，而非要求人工重算 JSONL。"""

    (tmp_path / "phase_events.jsonl").write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "phase": "runtime_step",
                        "event": "start",
                        "elapsed_seconds": 10.0,
                        "num_envs": 4096,
                        "policy_steps": 8,
                    }
                ),
                json.dumps(
                    {
                        "phase": "runtime_step",
                        "event": "complete",
                        "elapsed_seconds": 12.0,
                        "environment_steps": 32768,
                        "environment_steps_per_second": 16384.0,
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_rl_runtime_artifacts(tmp_path)

    assert summary["completed_phase_details"]["runtime_step"] == {
        "num_envs": 4096,
        "policy_steps": 8,
        "environment_steps": 32768,
        "environment_steps_per_second": 16384.0,
        "duration_seconds": 2.0,
    }


def test_linux_resource_reader_rejects_invalid_pid_and_reads_current_process() -> None:
    r"""Linux reader 对非法 PID fail-closed，对当前进程至少返回存活状态。"""

    try:
        read_linux_process_resources(0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("invalid PID must be rejected")

    snapshot = read_linux_process_resources(__import__("os").getpid())
    assert snapshot["process_alive"] is True
    assert isinstance(snapshot["process_rss_bytes"], int)
    assert isinstance(snapshot["system_available_ram_bytes"], int)
