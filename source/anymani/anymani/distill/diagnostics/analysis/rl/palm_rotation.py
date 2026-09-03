r"""MVP80 run目录的Parquet-first只读学习诊断。

Parquet是global/8-cell/80-asset标量事实源；TensorBoard只服务在线查看。本模块验证identity、update连续性与
89-row几何，再报告末窗口、cell失衡、最弱资产、探索裁剪和base/FiLM/residual启用程度。它不import task、
model或teacher，也不修改训练状态。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

import polars as pl


def _sha256(path: Path) -> str:
    r"""流式计算checkpoint或Parquet artifact摘要。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_metrics(run_dir: Path) -> tuple[pl.DataFrame, list[Path]]:
    r"""读取覆盖最新update的事实表，处理中断续训留下的stale final table。

    一个run在首次正常停止时会发布``metrics.parquet``，随后从checkpoint续训会继续追加immutable shards；若
    新进程被中断，旧final仍存在但只覆盖较早update。此时必须比较两条轴的最大update，而不能机械优先final。
    """

    final_path = run_dir / "metrics.parquet"
    shards = sorted((run_dir / "metrics_shards").glob("metrics-*.parquet"))
    shard_frame = pl.concat([pl.read_parquet(path) for path in shards], how="diagonal_relaxed") if shards else None
    final_frame = pl.read_parquet(final_path) if final_path.is_file() else None
    if final_frame is None and shard_frame is None:
        raise FileNotFoundError(f"MVP80 run contains no Parquet metrics: {run_dir}")
    shard_max = int(cast(Any, shard_frame["update"].max())) if shard_frame is not None else -1
    final_max = int(cast(Any, final_frame["update"].max())) if final_frame is not None else -1
    if shard_frame is not None and shard_max > final_max:
        return shard_frame, shards  # interrupted continuation的最新durable prefix
    if final_frame is None:
        raise RuntimeError("unreachable Parquet source state")
    return final_frame, [final_path]


def _finite_column(frame: pl.DataFrame, name: str) -> bool:
    r"""Nullable float列只检查非null样本；NaN/Inf均使run audit失败。"""

    if name not in frame.columns:
        return False
    values = frame[name].drop_nulls().to_list()
    return all(math.isfinite(float(value)) for value in values)


def _window_rows(frame: pl.DataFrame, updates: int) -> pl.DataFrame:
    r"""选择最后$K$个不同update，不按物理row数截断89-row group。"""

    update_axis = frame["update"].unique().sort().to_list()
    selected = update_axis[-min(updates, len(update_axis)) :]
    return frame.filter(pl.col("update").is_in(selected))


def _scope_summary(frame: pl.DataFrame, *, group_key: str, metrics: tuple[str, ...]) -> list[dict[str, Any]]:
    r"""对最后窗口按cell/asset等权求均值，并保留identity labels。"""

    expressions = [pl.col(metric).drop_nulls().mean().alias(metric) for metric in metrics if metric in frame.columns]
    labels = [name for name in ("dataset_row", "cell_id") if name in frame.columns and name != group_key]
    return frame.group_by(group_key).agg([pl.col(name).drop_nulls().first().alias(name) for name in labels] + expressions).sort(group_key).to_dicts()


def analyze_palm_rotation_run(run_dir: Path | str, *, window_updates: int = 20) -> dict[str, Any]:
    r"""验证一个MVP80 run并返回JSON-safe global/cell/asset诊断。

    Args:
        run_dir (Path | str): `logs/distill/rl_games/.../<run-name>`目录。
        window_updates (int): 末窗口update数；默认20，不对曲线做平滑插值。

    Returns:
        dict[str, Any]: identity/update audit、global末值、cell/asset窗口均值与触发信号。
    """

    root = Path(run_dir).expanduser().resolve()
    frame, sources = _load_metrics(root)
    required = {"identity_digest", "update", "scope", "scope_index", "dataset_row", "cell_id"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"MVP80 metrics miss required columns: {sorted(required - set(frame.columns))}")
    identity_values = frame["identity_digest"].unique().to_list()
    if len(identity_values) != 1:
        raise RuntimeError("MVP80 run mixes multiple method identities")
    updates = [int(value) for value in frame["update"].unique().sort().to_list()]
    expected_updates = list(range(updates[0], updates[-1] + 1))
    counts = frame.group_by("update").len().sort("update")
    row_geometry_valid = counts["len"].to_list() == [89] * len(updates)
    update_axis_valid = updates == expected_updates
    scope_counts = _scope_counts(frame["scope"].to_list())
    numeric_columns = [name for name, dtype in frame.schema.items() if dtype.is_numeric()]
    non_finite = sorted(name for name in numeric_columns if not _finite_column(frame, name))

    window = _window_rows(frame, window_updates)
    global_rows = frame.filter(pl.col("scope") == "global").sort("update")
    cell_window = window.filter(pl.col("scope") == "cell")
    asset_window = window.filter(pl.col("scope") == "asset")
    metrics = (
        "reward_mean",
        "goal_count_mean",
        "net_turns_mean",
        "drop_rate",
        "terminal_goal_count_mean",
        "terminal_net_turns_mean",
        "terminal_directional_consistency_mean",
        "value_error_mean",
        "kl_per_active_dof",
        "action_clamp_fraction",
        "action_rms",
        "physical_action_rms",
        "residual_rms",
        "film_modulation_rms",
        "completed_episode_count",
    )
    cells = _scope_summary(cell_window, group_key="cell_id", metrics=metrics)
    assets = _scope_summary(asset_window, group_key="scope_index", metrics=metrics)
    ranking_metric = "terminal_net_turns_mean"
    if not any(float(row.get("completed_episode_count") or 0.0) > 0.0 for row in assets):
        ranking_metric = "net_turns_mean"  # 120 s episode尚未结束时使用明确标注的rollout proxy
    ranked_assets = sorted(assets, key=lambda row: float(row.get(ranking_metric) or 0.0))

    last_global = global_rows.tail(1).to_dicts()[0]
    value_errors = [float(row["value_error_mean"]) for row in cells if row.get("value_error_mean") is not None]
    positive_errors = [value for value in value_errors if value > 0.0]
    critic_imbalance = max(positive_errors) / max(_median(positive_errors), 1.0e-12) if positive_errors else 0.0
    triggers = {
        "non_finite": bool(non_finite),
        "action_clamp_high": float(last_global.get("action_clamp_fraction") or 0.0) > 0.20,
        "critic_cell_imbalance_over_2x": critic_imbalance > 2.0,
        "residual_inactive": float(last_global.get("residual_rms") or 0.0) < 1.0e-4,
        "film_inactive": float(last_global.get("film_modulation_rms") or 0.0) < 1.0e-4,
    }

    checkpoints = []
    for path in sorted((root / "nn").glob("*.pth")):
        checkpoints.append({"name": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size})
    return {
        "artifact_type": "anymani.palm_rotation_mvp80.run_analysis",
        "schema_version": "1.0.0",
        "run_dir": str(root),
        "identity_digest": identity_values[0],
        "audit": {
            "updates": updates,
            "update_axis_contiguous": update_axis_valid,
            "rows_per_update_89": row_geometry_valid,
            "scope_counts": scope_counts,
            "non_finite_columns": non_finite,
            "source_files": [{"path": str(path), "sha256": _sha256(path)} for path in sources],
        },
        "last_global": last_global,
        "cell_window": cells,
        "weakest_assets": ranked_assets[:10],
        "strongest_assets": list(reversed(ranked_assets[-10:])),
        "ranking_metric": ranking_metric,
        "critic_cell_value_error_ratio": critic_imbalance,
        "triggers": triggers,
        "checkpoints": checkpoints,
    }


def _scope_counts(values: list[str]) -> dict[str, int]:
    r"""返回排序后的scope频数，避免analysis依赖动态Counter序列化细节。"""

    return {value: values.count(value) for value in sorted(set(values))}


def _median(values: list[float]) -> float:
    r"""计算小型cell列表中位数，不引入NumPy运行时。"""

    ordered = sorted(values)
    if not ordered:
        raise ValueError("median requires values")
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else 0.5 * (ordered[middle - 1] + ordered[middle])


def main() -> None:
    r"""CLI：读取一个run目录并原子发布JSON诊断。"""

    parser = argparse.ArgumentParser(description="Analyze one MVP80 palm-rotation PPO run directory.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--window_updates", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = analyze_palm_rotation_run(args.run_dir, window_updates=args.window_updates)
    output = args.output or args.run_dir / "analysis.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "triggers": report["triggers"]}, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["analyze_palm_rotation_run"]
