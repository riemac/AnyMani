r"""从TensorBoard raw scalars审计异构N040 PPO学习与八组结果。

本模块只读event artifact，不import task/model/checkpoint。Reward、episode length、task rotation/success、actor
KL/entropy、central critic loss与cell terminal metrics共同解释；正reward不能替代物理旋转能力。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any

import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

CELL_LABELS = (
    "left_tips3_thumb3dof",
    "left_tips3_thumb4dof",
    "left_tips4_thumb3dof",
    "left_tips4_thumb4dof",
    "right_tips3_thumb3dof",
    "right_tips3_thumb4dof",
    "right_tips4_thumb3dof",
    "right_tips4_thumb4dof",
)


def _series_summary(events, *, window: int = 20) -> dict[str, Any]:
    r"""以raw step/value序列报告首尾窗口，不做平滑或重采样。"""

    values = [float(event.value) for event in events]
    steps = [int(event.step) for event in events]
    finite = [value for value in values if math.isfinite(value)]
    if not values:
        return {"count": 0}
    first = values[: min(window, len(values))]
    last = values[-min(window, len(values)) :]
    return {
        "count": len(values),
        "first_step": steps[0],
        "last_step": steps[-1],
        "first_value": values[0],
        "last_value": values[-1],
        "first_window_mean": mean(first),
        "last_window_mean": mean(last),
        "min_value": min(values),
        "max_value": max(values),
        "finite_count": len(finite),
        "non_finite_count": len(values) - len(finite),
    }


def _cell_series_summary(
    metric_events,
    count_events,
    *,
    window: int = 20,
    metric_is_sum: bool = False,
) -> dict[str, Any]:
    r"""按episode_count过滤无样本cohort，并恢复cell内均值。

    新固定-key logger把cell numerator/mean与count以同一reset-cohort权重写入TensorBoard，二者比值恢复
    cell内mean。旧dynamic-key run的count可大于1且metric已是raw mean，此时不做比值变换。
    """

    counts = {int(event.step): float(event.value) for event in count_events}
    old_dynamic_schema = any(value > 1.0 for value in counts.values()) and not metric_is_sum
    recovered = []
    for event in metric_events:
        count = counts.get(int(event.step), 0.0)
        if count <= 0.0:
            continue
        value = float(event.value) if old_dynamic_schema else float(event.value) / count
        recovered.append(SimpleNamespace(step=int(event.step), value=value))
    summary = _series_summary(recovered, window=window)
    summary["observed_reset_cohorts"] = len(recovered)
    summary["logging_schema"] = (
        "sum_over_episode_count"
        if metric_is_sum
        else "dynamic_raw_mean"
        if old_dynamic_schema
        else "fixed_key_legacy_mean_over_count"
    )
    return summary


def analyze_heterogeneous_ppo(event_path: Path) -> dict[str, Any]:
    r"""分析一个event segment并区分学习、稳定持物与真实旋转。

    Args:
        event_path (Path): 单个TensorBoard event文件；多segment run应由上层先审计resume lineage。

    Returns:
        dict[str, Any]: JSON/YAML-safe audit、global curves、task metrics与八组terminal evidence。
    """

    resolved = event_path.expanduser().resolve()
    accumulator = EventAccumulator(str(resolved), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = tuple(accumulator.Tags().get("scalars", ()))

    decisive_tags = {
        "reward": "rewards/iter",
        "episode_length": "episode_lengths/iter",
        "actor_kl": "info/kl",
        "actor_entropy": "losses/entropy",
        "central_value_loss": "losses/cval_loss",
        "goal_success": "Episode/Metrics/goal_pose/goal_success_count",
        "net_rotation_turns": "Episode/Metrics/goal_pose/net_rotation_turns",
        "net_rotation_rad": "Episode/Metrics/goal_pose/net_rotation_rad",
        "tip_active_count": "Episode/Metrics/goal_pose/contact/tip_active_count_mean",
        "palm_occupancy": "Episode/Metrics/goal_pose/contact/palm_occupancy_fraction",
        "object_out": "Episode/Metrics/goal_pose/termination/object_out_of_anchor_fraction",
        "time_out": "Episode/Metrics/goal_pose/termination/time_out_fraction",
    }
    global_metrics = {
        name: _series_summary(accumulator.Scalars(tag)) if tag in tags else {"count": 0, "missing_tag": tag}
        for name, tag in decisive_tags.items()
    }

    cell_metrics: dict[str, Any] = {}
    for label in CELL_LABELS:
        prefix = f"Episode/Metrics/goal_pose/cell/{label}/"
        count_tag = prefix + "episode_count"
        count_events = accumulator.Scalars(count_tag) if count_tag in tags else []
        metrics = {"episode_count": _series_summary(count_events)}
        for metric_name in (
            "goal_success_count",
            "net_rotation_turns",
            "position_error",
            "contact/tip_active_count_mean",
            "contact/finger_non_tip_occupancy_fraction",
            "termination/object_out_of_anchor_fraction",
        ):
            sum_tag = prefix + metric_name + "_sum"
            legacy_tag = prefix + metric_name
            tag = sum_tag if sum_tag in tags else legacy_tag
            metrics[metric_name] = (
                _cell_series_summary(
                    accumulator.Scalars(tag),
                    count_events,
                    metric_is_sum=tag == sum_tag,
                )
                if tag in tags and count_events
                else {"count": 0}
            )
        cell_metrics[label] = metrics

    required = ("reward", "episode_length", "actor_kl", "central_value_loss", "goal_success", "net_rotation_turns")
    missing_required = tuple(name for name in required if global_metrics[name].get("count", 0) == 0)
    non_finite_required = tuple(name for name in required if global_metrics[name].get("non_finite_count", 0) > 0)
    cell_coverage = sum(cell_metrics[label]["episode_count"].get("count", 0) > 0 for label in CELL_LABELS)
    audit_status = "usable"
    caveats = []
    if missing_required or non_finite_required:
        audit_status = "unidentifiable"
    elif cell_coverage < 8:
        audit_status = "usable_with_caveat"
        caveats.append(f"only {cell_coverage}/8 morphology cells have terminal metric records")
    if global_metrics["goal_success"].get("count", 0) < 5:
        caveats.append("episode-level physical metrics contain fewer than five reset cohorts")

    return {
        "schema_version": "1.0.0",
        "artifact_type": "anymani.heterogeneous_ppo.analysis",
        "event_path": str(resolved),
        "audit": {
            "status": audit_status,
            "scalar_tag_count": len(tags),
            "missing_required": list(missing_required),
            "non_finite_required": list(non_finite_required),
            "cell_coverage": cell_coverage,
            "caveats": caveats,
        },
        "global_metrics": global_metrics,
        "cell_metrics": cell_metrics,
    }


def main() -> None:
    r"""CLI：读取一个event文件并原子语义写出YAML分析。"""

    parser = argparse.ArgumentParser(description="Analyze AnyMani heterogeneous PPO TensorBoard evidence.")
    parser.add_argument("event_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    report = analyze_heterogeneous_ppo(args.event_path)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(yaml.safe_dump(report, sort_keys=False, allow_unicode=True), encoding="utf-8")


if __name__ == "__main__":
    main()
