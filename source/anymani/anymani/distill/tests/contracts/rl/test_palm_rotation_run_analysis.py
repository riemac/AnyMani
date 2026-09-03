r"""MVP80 Parquet-first run analyzer的identity、89-row与触发器合同。"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from anymani.distill.diagnostics.analysis.rl.palm_rotation import analyze_palm_rotation_run
from anymani.distill.diagnostics.recording.rl.palm_rotation import PALM_ROTATION_METRICS_SCHEMA


def test_run_analysis_reads_89_row_updates_and_ranks_assets(tmp_path: Path) -> None:
    r"""Analyzer必须按完整update分组，并从Parquet而非TensorBoard恢复cell/asset弱项。"""

    rows: list[dict[str, object]] = []
    identity = "d" * 64
    for update in (1, 2):
        shared = {
            "schema_version": "2.0.0",
            "identity_digest": identity,
            "update": update,
            "transitions": update * 76800,
            "reward_mean": float(update),
            "net_turns_mean": 0.01 * update,
            "value_error_mean": 1.0,
            "action_clamp_fraction": 0.1,
            "residual_rms": 0.01,
            "film_modulation_rms": 0.02,
        }
        rows.append({**shared, "scope": "global", "scope_index": 0})
        rows.extend(
            {**shared, "scope": "cell", "scope_index": cell, "cell_id": cell}
            for cell in range(8)
        )
        rows.extend(
            {
                **shared,
                "scope": "asset",
                "scope_index": asset,
                "dataset_row": 1000 + asset,
                "cell_id": asset // 10,
                "net_turns_mean": 0.001 * asset,
            }
            for asset in range(80)
        )
    normalized = [{name: row.get(name) for name in PALM_ROTATION_METRICS_SCHEMA} for row in rows]
    frame = pl.DataFrame(normalized, schema=PALM_ROTATION_METRICS_SCHEMA)
    frame.write_parquet(tmp_path / "metrics.parquet", compression="zstd")

    report = analyze_palm_rotation_run(tmp_path, window_updates=2)
    assert report["audit"]["rows_per_update_89"]
    assert report["audit"]["update_axis_contiguous"]
    assert report["weakest_assets"][0]["dataset_row"] == 1000
    assert report["strongest_assets"][0]["dataset_row"] == 1079
    assert report["triggers"]["non_finite"] is False


def test_run_analysis_prefers_newer_shards_over_stale_final(tmp_path: Path) -> None:
    r"""正常停止后续训中断时，旧final不能遮蔽更晚的immutable checkpoint shards。"""

    rows = []
    for update in (1, 2):
        shared = {
            "schema_version": "2.0.0",
            "identity_digest": "e" * 64,
            "update": update,
            "transitions": update * 76800,
            "reward_mean": float(update),
            "net_turns_mean": 0.01 * update,
        }
        rows.append({**shared, "scope": "global", "scope_index": 0})
        rows.extend({**shared, "scope": "cell", "scope_index": cell, "cell_id": cell} for cell in range(8))
        rows.extend(
            {
                **shared,
                "scope": "asset",
                "scope_index": asset,
                "dataset_row": 2000 + asset,
                "cell_id": asset // 10,
            }
            for asset in range(80)
        )
    normalized = [{name: row.get(name) for name in PALM_ROTATION_METRICS_SCHEMA} for row in rows]
    frame = pl.DataFrame(normalized, schema=PALM_ROTATION_METRICS_SCHEMA)
    frame.filter(pl.col("update") == 1).write_parquet(tmp_path / "metrics.parquet")  # stale first stop
    shard_dir = tmp_path / "metrics_shards"
    shard_dir.mkdir()
    frame.write_parquet(shard_dir / "metrics-000000-u00000001-u00000002.parquet")  # resumed durable prefix

    report = analyze_palm_rotation_run(tmp_path)
    assert report["audit"]["updates"] == [1, 2]
    assert report["last_global"]["update"] == 2
