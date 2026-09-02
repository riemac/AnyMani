r"""Post-physics snapshots到fixed-duration trajectory distributions的纯Torch reducer。

IsaacLab在``env.step``返回前自动reset done rows，因此evaluation不能在返回后直接读取command/contact buffers。本模块
只接受task在pre-reset时冻结的full-env snapshot；逐环境累计speed/contact/reward samples，并在natural terminal或
evaluation window结束时生成一个trajectory record。Trajectory共享同一policy/seed，统计只作描述性分布，不伪造独立CI。
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

REQUIRED_SNAPSHOT_FIELDS = (
    "valid",
    "step",
    "dataset_row",
    "axis_speed_rad_s",
    "net_rotation_rad",
    "completed_subgoals",
    "goal_success_pulse",
    "episode_duration_s",
    "tip_active_count",
    "palm_contact",
    "finger_non_tip_contact",
    "orientation_keypoint_error_m",
    "position_error_m",
    "termination_object_out_of_anchor",
    "termination_goal_axis_misaligned",
    "termination_time_out",
)


def _distribution(values: list[float]) -> dict[str, float]:
    r"""返回trajectory-level mean/median/q10/lower-CVaR10与极值。"""

    if not values:
        raise ValueError("trajectory distribution requires at least one value")
    tensor = torch.tensor(values, dtype=torch.float64)
    q10 = torch.quantile(tensor, 0.1)
    lower_tail = tensor[tensor <= q10]
    return {
        "mean": float(tensor.mean().item()),
        "median": float(torch.quantile(tensor, 0.5).item()),
        "q10": float(q10.item()),
        "cvar10_lower": float(lower_tail.mean().item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


class FixedDurationTrajectoryAccumulator:
    r"""把pre-reset snapshots规约为完整与right-censored window trajectories。

    每个environment有独立running sums。Natural terminal在对应step加入当前sample后立即finalize；随后running state清零，
    与Isaac自动开始的新episode对齐。Evaluation最后只finalize至少包含一个sample的active rows，不把刚reset的零长度episode
    计入分母。
    """

    def __init__(self, dataset_row_by_env: torch.Tensor, *, step_dt: float) -> None:
        r"""分配per-env speed/contact/reward sums。

        Args:
            dataset_row_by_env (torch.Tensor): formal dataset labels，形状$[N]$，只作分层统计。
            step_dt (float): policy step时长，单位s；当前baseline为0.05 s。
        """

        if dataset_row_by_env.ndim != 1 or dataset_row_by_env.dtype != torch.long:
            raise ValueError("evaluation dataset rows must be a rank-1 LongTensor")
        if not math.isfinite(step_dt) or step_dt <= 0.0:
            raise ValueError("evaluation step_dt must be finite and positive")
        self.dataset_row_by_env = dataset_row_by_env.clone()
        self.num_envs = dataset_row_by_env.numel()
        self.device = dataset_row_by_env.device
        self.step_dt = float(step_dt)
        self._sample_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float64, device=self.device)
            for name in (
                "reward",
                "axis_speed",
                "absolute_axis_speed",
                "tip_active_count",
                "palm_contact",
                "finger_non_tip_contact",
            )
        }
        self.records: list[dict[str, Any]] = []

    def _validate_snapshot(self, snapshot: Mapping[str, torch.Tensor]) -> None:
        r"""验证snapshot完整、finite且dataset routing与evaluation固定轴一致。"""

        if set(snapshot) != set(REQUIRED_SNAPSHOT_FIELDS):
            missing = sorted(set(REQUIRED_SNAPSHOT_FIELDS) - set(snapshot))
            extra = sorted(set(snapshot) - set(REQUIRED_SNAPSHOT_FIELDS))
            raise ValueError(f"evaluation snapshot fields disagree: missing={missing}, extra={extra}")
        if any(value.shape != (self.num_envs,) for value in snapshot.values()):
            raise ValueError("all evaluation snapshot tensors must share the rank-1 environment axis")
        if not bool(snapshot["valid"].all().item()):
            raise RuntimeError("evaluation snapshot contains rows not captured before reset")
        if not torch.equal(snapshot["dataset_row"], self.dataset_row_by_env):
            raise RuntimeError("evaluation snapshot dataset routing changed")
        floating = [value for value in snapshot.values() if value.is_floating_point()]
        if not all(bool(torch.isfinite(value).all().item()) for value in floating):
            raise RuntimeError("evaluation snapshot contains non-finite physical values")

    def add_step(
        self,
        snapshot: Mapping[str, torch.Tensor],
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        r"""加入一个post-physics sample，并finalize当前natural terminal rows。"""

        self._validate_snapshot(snapshot)
        if rewards.shape != (self.num_envs,) or dones.shape != (self.num_envs,) or dones.dtype != torch.bool:
            raise ValueError("evaluation reward/done tensors must share the environment axis")
        if not bool(torch.isfinite(rewards).all().item()):
            raise RuntimeError("evaluation reward contains non-finite values")
        self._sample_count += 1
        self._sums["reward"] += rewards.to(dtype=torch.float64)
        self._sums["axis_speed"] += snapshot["axis_speed_rad_s"].to(dtype=torch.float64)
        self._sums["absolute_axis_speed"] += snapshot["axis_speed_rad_s"].abs().to(dtype=torch.float64)
        self._sums["tip_active_count"] += snapshot["tip_active_count"].to(dtype=torch.float64)
        self._sums["palm_contact"] += snapshot["palm_contact"].to(dtype=torch.float64)
        self._sums["finger_non_tip_contact"] += snapshot["finger_non_tip_contact"].to(dtype=torch.float64)
        done_ids = dones.nonzero(as_tuple=False).flatten()
        if done_ids.numel() > 0:
            self._finalize(snapshot, done_ids, natural_terminal=True)

    def finish_window(self, snapshot: Mapping[str, torch.Tensor]) -> None:
        r"""把至少执行一步的active rows记为fixed-window censored trajectories。"""

        self._validate_snapshot(snapshot)
        active_ids = (self._sample_count > 0).nonzero(as_tuple=False).flatten()
        if active_ids.numel() > 0:
            self._finalize(snapshot, active_ids, natural_terminal=False)

    def _finalize(
        self,
        snapshot: Mapping[str, torch.Tensor],
        env_ids: torch.Tensor,
        *,
        natural_terminal: bool,
    ) -> None:
        r"""冻结selected trajectories到CPU JSON records并清对应running sums。"""

        counts = self._sample_count[env_ids]
        if bool((counts < 1).any().item()):
            raise RuntimeError("cannot finalize a zero-sample evaluation trajectory")
        selected_snapshot = {name: value[env_ids].detach().cpu() for name, value in snapshot.items()}
        selected_sums = {name: value[env_ids].detach().cpu() for name, value in self._sums.items()}
        env_ids_cpu = env_ids.detach().cpu()
        counts_cpu = counts.detach().cpu()
        for local_index, env_id_tensor in enumerate(env_ids_cpu):
            env_id = int(env_id_tensor.item())
            sample_count = int(counts_cpu[local_index].item())
            net_rotation = float(selected_snapshot["net_rotation_rad"][local_index].item())
            duration = float(selected_snapshot["episode_duration_s"][local_index].item())
            completed_subgoals = float(selected_snapshot["completed_subgoals"][local_index].item())
            termination = {
                name: bool(selected_snapshot[f"termination_{name}"][local_index].item())
                for name in ("object_out_of_anchor", "goal_axis_misaligned", "time_out")
            }
            end_reasons = [name for name, active in termination.items() if active]
            if not natural_terminal:
                end_reasons = ["evaluation_window"]
                termination = {name: False for name in termination}
            record = {
                "trajectory_index": len(self.records),
                "env_id": env_id,
                "dataset_row": int(selected_snapshot["dataset_row"][local_index].item()),
                "sample_count": sample_count,
                "duration_s": duration,
                "natural_terminal": natural_terminal,
                "end_reasons": end_reasons,
                "reward_mean_per_step": float(selected_sums["reward"][local_index].item()) / sample_count,
                "signed_axis_speed_sample_mean_rad_s": float(
                    selected_sums["axis_speed"][local_index].item()
                )
                / sample_count,
                "absolute_axis_speed_sample_mean_rad_s": float(
                    selected_sums["absolute_axis_speed"][local_index].item()
                )
                / sample_count,
                "time_weighted_signed_speed_rad_s": net_rotation / max(duration, torch.finfo(torch.float32).eps),
                "signed_net_rotation_rad": net_rotation,
                "signed_net_rotation_turns": net_rotation / (2.0 * math.pi),
                "completed_subgoals": completed_subgoals,
                "episode_any_success_pulse": completed_subgoals > 0.0,
                "reached_positive_30deg": net_rotation >= math.pi / 6.0,
                "reached_negative_30deg": net_rotation <= -math.pi / 6.0,
                "reached_positive_full_turn": net_rotation >= 2.0 * math.pi,
                "reached_negative_full_turn": net_rotation <= -2.0 * math.pi,
                "tip_active_count_mean": float(selected_sums["tip_active_count"][local_index].item()) / sample_count,
                "palm_occupancy_fraction": float(selected_sums["palm_contact"][local_index].item()) / sample_count,
                "finger_non_tip_occupancy_fraction": float(
                    selected_sums["finger_non_tip_contact"][local_index].item()
                )
                / sample_count,
                "terminal_orientation_keypoint_error_m": float(
                    selected_snapshot["orientation_keypoint_error_m"][local_index].item()
                ),
                "terminal_position_error_m": float(selected_snapshot["position_error_m"][local_index].item()),
                "termination_object_out_of_anchor": termination["object_out_of_anchor"],
                "termination_goal_axis_misaligned": termination["goal_axis_misaligned"],
                "termination_time_out": termination["time_out"],
            }
            self.records.append(record)
        self._sample_count[env_ids] = 0
        for running_sum in self._sums.values():
            running_sum[env_ids] = 0.0

    def summary(self, *, requested_steps: int) -> dict[str, Any]:
        r"""生成trajectory、per-asset与equal-asset描述性统计。"""

        if requested_steps < 1 or not self.records:
            raise ValueError("evaluation summary requires positive steps and finalized trajectories")
        records = self.records
        total_samples = sum(int(record["sample_count"]) for record in records)
        expected_samples = self.num_envs * requested_steps
        if total_samples != expected_samples:
            raise RuntimeError(f"trajectory samples {total_samples} disagree with fixed window {expected_samples}")

        def trajectory_mean(name: str) -> float:
            return sum(float(record[name]) for record in records) / len(records)

        def sample_weighted_mean(name: str) -> float:
            return sum(float(record[name]) * int(record["sample_count"]) for record in records) / total_samples

        by_asset: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            by_asset[int(record["dataset_row"])].append(record)
        per_asset: dict[str, Any] = {}
        for dataset_row, asset_records in sorted(by_asset.items()):
            signed_values = [float(record["signed_net_rotation_rad"]) for record in asset_records]
            per_asset[str(dataset_row)] = {
                "trajectory_count": len(asset_records),
                "signed_net_rotation_rad": _distribution(signed_values),
                "subgoals_per_trajectory": sum(float(record["completed_subgoals"]) for record in asset_records)
                / len(asset_records),
                "episode_any_success_pulse_fraction": sum(
                    bool(record["episode_any_success_pulse"]) for record in asset_records
                )
                / len(asset_records),
                "positive_30deg_fraction": sum(bool(record["reached_positive_30deg"]) for record in asset_records)
                / len(asset_records),
                "negative_30deg_fraction": sum(bool(record["reached_negative_30deg"]) for record in asset_records)
                / len(asset_records),
                "positive_full_turn_fraction": sum(
                    bool(record["reached_positive_full_turn"]) for record in asset_records
                )
                / len(asset_records),
                "negative_full_turn_fraction": sum(
                    bool(record["reached_negative_full_turn"]) for record in asset_records
                )
                / len(asset_records),
                "drop_fraction": sum(bool(record["termination_object_out_of_anchor"]) for record in asset_records)
                / len(asset_records),
                "axis_failure_fraction": sum(
                    bool(record["termination_goal_axis_misaligned"]) for record in asset_records
                )
                / len(asset_records),
                "timeout_fraction": sum(bool(record["termination_time_out"]) for record in asset_records)
                / len(asset_records),
            }
        distribution_names = ("mean", "median", "q10", "cvar10_lower", "min", "max")
        equal_asset_signed_distribution = {
            name: sum(float(asset["signed_net_rotation_rad"][name]) for asset in per_asset.values()) / len(per_asset)
            for name in distribution_names
        }
        total_subgoals = sum(float(record["completed_subgoals"]) for record in records)
        return {
            "trajectory_count": float(len(records)),
            "raw_done_count": float(sum(bool(record["natural_terminal"]) for record in records)),
            "reward_mean_per_step": sample_weighted_mean("reward_mean_per_step"),
            "signed_axis_speed_mean_rad_s": sample_weighted_mean("signed_axis_speed_sample_mean_rad_s"),
            "absolute_axis_speed_mean_rad_s": sample_weighted_mean("absolute_axis_speed_sample_mean_rad_s"),
            "time_weighted_signed_speed_rad_s": trajectory_mean("time_weighted_signed_speed_rad_s"),
            "signed_net_rotation_rad_mean": trajectory_mean("signed_net_rotation_rad"),
            "signed_net_rotation_turns_mean": trajectory_mean("signed_net_rotation_turns"),
            "subgoals_per_trajectory": total_subgoals / len(records),
            "subgoal_throughput_per_env_s": total_subgoals / (total_samples * self.step_dt),
            "episode_any_success_pulse_fraction": trajectory_mean("episode_any_success_pulse"),
            "positive_30deg_fraction": trajectory_mean("reached_positive_30deg"),
            "negative_30deg_fraction": trajectory_mean("reached_negative_30deg"),
            "positive_full_turn_fraction": trajectory_mean("reached_positive_full_turn"),
            "negative_full_turn_fraction": trajectory_mean("reached_negative_full_turn"),
            "drop_fraction": trajectory_mean("termination_object_out_of_anchor"),
            "axis_failure_fraction": trajectory_mean("termination_goal_axis_misaligned"),
            "timeout_fraction": trajectory_mean("termination_time_out"),
            "tip_active_count_mean": sample_weighted_mean("tip_active_count_mean"),
            "palm_occupancy_fraction": sample_weighted_mean("palm_occupancy_fraction"),
            "finger_non_tip_occupancy_fraction": sample_weighted_mean("finger_non_tip_occupancy_fraction"),
            "trajectory_distribution": {
                "signed_net_rotation_rad": _distribution(
                    [float(record["signed_net_rotation_rad"]) for record in records]
                ),
                "duration_s": _distribution([float(record["duration_s"]) for record in records]),
                "time_weighted_signed_speed_rad_s": _distribution(
                    [float(record["time_weighted_signed_speed_rad_s"]) for record in records]
                ),
            },
            "per_asset": per_asset,
            "equal_asset": {
                "asset_count": len(per_asset),
                "signed_net_rotation_rad": equal_asset_signed_distribution,
                "subgoals_per_trajectory": sum(
                    float(asset["subgoals_per_trajectory"]) for asset in per_asset.values()
                )
                / len(per_asset),
                "drop_fraction": sum(float(asset["drop_fraction"]) for asset in per_asset.values()) / len(per_asset),
                "axis_failure_fraction": sum(float(asset["axis_failure_fraction"]) for asset in per_asset.values())
                / len(per_asset),
            },
            "uncertainty": {
                "independent_training_seed_count": 1,
                "confidence_interval": None,
                "interpretation": (
                    "single-seed descriptive trajectories; rows share one policy and are correlated, so trajectory "
                    "count is not an independent estimate of seed-level uncertainty"
                ),
            },
        }


def write_trajectory_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    r"""写完整trajectory distribution并返回文件SHA-256。"""

    import hashlib

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records).encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


__all__ = ["FixedDurationTrajectoryAccumulator", "REQUIRED_SNAPSHOT_FIELDS", "write_trajectory_jsonl"]
