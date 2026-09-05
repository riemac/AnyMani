r"""把fixed evaluation terminal sums整理成等资产权重的任务表现分布。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import yaml
from anymani.assets.bank.dataset import HandAssetDataset
from anymani.assets.bank.prepared_train import resolve_prepared_train

DEFAULT_DATASET = Path("source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1/ppo.yaml")

REPORT_METRICS = {
    "subgoals_per_episode": "goal_success_count",
    "episode_any_subgoal_fraction": "episode_with_goal_success_count",
    "terminal_positive_30deg_fraction": "episode_positive_30deg_count",
    "terminal_negative_30deg_fraction": "episode_negative_30deg_count",
    "terminal_positive_one_turn_fraction": "episode_positive_one_turn_count",
    "positive_net_turns_per_episode": "net_rotation_turns",
    "signed_net_degrees_per_episode": "derived/net_rotation_deg_per_episode",
    "time_weighted_signed_speed_rad_s": "derived/time_weighted_signed_speed_rad_s",
    "episode_mean_abs_axis_speed_rad_s": "rotation/axis_speed_abs_mean_rad_s",
    "off_axis_ang_vel_rms_rad_s": "rotation/off_axis_ang_vel_rms_rad_s",
    "episode_duration_s": "task/episode_duration_s",
    "object_out_fraction": "termination/object_out_of_anchor_fraction",
    "axis_misaligned_fraction": "termination/goal_axis_misaligned_fraction",
    "timeout_fraction": "termination/time_out_fraction",
    "active_tips_mean": "contact/tip_active_count_mean",
    "palm_occupancy_fraction": "contact/palm_occupancy_fraction",
    "finger_non_tip_occupancy_fraction": "contact/finger_non_tip_occupancy_fraction",
}


def _distribution(values: list[float]) -> dict[str, float]:
    r"""对16个asset-level estimands等权汇总，不按episode count二次加权。"""

    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "q10": float(np.quantile(array, 0.10)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "q90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
    }


def _topology_metadata(topology_key: str) -> tuple[int, int]:
    r"""由typed topology key恢复active fingertip数与thumb DoF。"""

    parts = topology_key.split("_")[1:]
    dofs = {match.group(1): int(match.group(2)) for part in parts if (match := re.fullmatch(r"([timr])(\d+)", part))}
    if "t" not in dofs:
        raise ValueError(f"topology lacks thumb segment: {topology_key}")
    return len(dofs), dofs["t"]


def analyze(input_path: Path, dataset_path: Path) -> dict:
    r"""连接评估artifact与正式dataset identity，生成逐资产与macro分布。"""

    evaluation = json.loads(input_path.read_text())
    raw_assets = evaluation.get("asset_metrics", {})
    if not raw_assets:
        raise ValueError("evaluation artifact does not contain asset_metrics; rerun play with --asset-metrics")
    dataset = HandAssetDataset.from_yaml(dataset_path)
    partition, _ = resolve_prepared_train(dataset, require_geometry_semantics=True)
    records = []
    for raw_row, raw_metrics in sorted(raw_assets.items(), key=lambda item: int(item[0])):
        dataset_row = int(raw_row)
        asset = partition.assets[dataset_row]
        geometry = asset.geometry_semantics
        if geometry is None or geometry.topology_key is None:
            raise ValueError(f"dataset row {dataset_row} lacks required geometry topology semantics")
        topology_key = geometry.topology_key
        tip_count, thumb_dof = _topology_metadata(topology_key)
        metrics = {name: float(raw_metrics[source]) for name, source in REPORT_METRICS.items()}
        records.append(
            {
                "dataset_row": dataset_row,
                "asset_id": asset.asset_id,
                "topology_key": topology_key,
                "handedness": geometry.handedness,
                "family": geometry.family,
                "active_dof": len(geometry.active_joint_names),
                "active_fingertips": tip_count,
                "thumb_dof": thumb_dof,
                "asset_role": "mother" if asset.urdf_path.parent.name == topology_key else "variant",
                "episode_count": float(raw_metrics["episode_count"]),
                **metrics,
            }
        )
    distributions = {
        metric_name: _distribution([record[metric_name] for record in records]) for metric_name in REPORT_METRICS
    }
    totals = {
        "subgoal_pulses": sum(record["subgoals_per_episode"] * record["episode_count"] for record in records),
        "episodes_with_any_subgoal": sum(
            record["episode_any_subgoal_fraction"] * record["episode_count"] for record in records
        ),
        "episodes_terminal_positive_30deg": sum(
            record["terminal_positive_30deg_fraction"] * record["episode_count"] for record in records
        ),
        "episodes_terminal_negative_30deg": sum(
            record["terminal_negative_30deg_fraction"] * record["episode_count"] for record in records
        ),
        "episodes_terminal_positive_one_turn": sum(
            record["terminal_positive_one_turn_fraction"] * record["episode_count"] for record in records
        ),
        "completed_episode_seconds": sum(record["episode_duration_s"] * record["episode_count"] for record in records),
    }
    return {
        "schema_version": "1.0.0",
        "artifact_type": "anymani.heterogeneous_ppo.asset_distribution_analysis",
        "source_evaluation": str(input_path.resolve()),
        "checkpoint": evaluation["checkpoint"],
        "seed": evaluation["seed"],
        "completed_policy_steps": evaluation["completed_policy_steps"],
        "asset_count": len(records),
        "episode_count": sum(record["episode_count"] for record in records),
        "totals": totals,
        "aggregation": {
            "asset_distribution": "equal weight over unique assets",
            "within_asset": evaluation["aggregation"],
            "pooled_episode_metrics": evaluation["global_metrics"],
        },
        "coverage": {
            "assets_with_any_subgoal_pulse": sum(record["episode_any_subgoal_fraction"] > 0.0 for record in records),
            "assets_with_terminal_positive_30deg": sum(
                record["terminal_positive_30deg_fraction"] > 0.0 for record in records
            ),
            "assets_with_terminal_positive_one_turn": sum(
                record["terminal_positive_one_turn_fraction"] > 0.0 for record in records
            ),
            "assets_with_positive_signed_net_angle": sum(
                record["signed_net_degrees_per_episode"] > 0.0 for record in records
            ),
        },
        "asset_distributions": distributions,
        "assets": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    analysis = analyze(args.input_path, args.dataset)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(yaml.safe_dump(analysis, sort_keys=False, allow_unicode=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
