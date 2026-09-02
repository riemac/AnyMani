r"""Per-asset terminal sufficient statistics与equal-asset聚合的纯Torch helpers。"""

from __future__ import annotations

from collections.abc import Mapping

import torch


def asset_episode_sufficient_statistics(
    *,
    dataset_row_by_env: torch.Tensor,
    reset_env_ids: torch.Tensor,
    goal_success_count: torch.Tensor,
    goal_success_pulse: torch.Tensor,
    net_rotation_rad: torch.Tensor,
    positive_net_rotation_turns: torch.Tensor,
    episode_duration_s: torch.Tensor,
    termination_bits: Mapping[str, torch.Tensor],
    horizon_s: float,
) -> dict[str, float]:
    r"""只对当前terminal/reset rows生成per-asset sum/count。

    ``goal_success_pulse``显式加入completed count，覆盖“同一步success+failure但command尚未update”的生命周期。
    Termination tensors必须是当前compute snapshot；函数从不读取manager aggregate或其它rows。
    """

    num_envs = dataset_row_by_env.numel()
    vectors = (
        goal_success_count,
        goal_success_pulse,
        net_rotation_rad,
        positive_net_rotation_turns,
        episode_duration_s,
        *termination_bits.values(),
    )
    if dataset_row_by_env.ndim != 1 or any(vector.shape != (num_envs,) for vector in vectors):
        raise ValueError("diagnostic inputs must share full rank-1 environment axis")
    if reset_env_ids.ndim != 1 or reset_env_ids.numel() < 1:
        raise ValueError("reset_env_ids must be a non-empty rank-1 subset")
    completed_subgoals = goal_success_count + goal_success_pulse.to(dtype=goal_success_count.dtype)
    signed_turns = net_rotation_rad / (2.0 * torch.pi)
    reached_full_turn = net_rotation_rad >= 2.0 * torch.pi
    reached_negative_full_turn = net_rotation_rad <= -2.0 * torch.pi
    reached_positive_30deg = net_rotation_rad >= torch.pi / 6.0
    reached_negative_30deg = net_rotation_rad <= -torch.pi / 6.0
    extras: dict[str, float] = {}
    selected_rows = dataset_row_by_env[reset_env_ids]
    for dataset_row in sorted(set(int(value) for value in selected_rows.tolist())):
        member_ids = reset_env_ids[selected_rows == dataset_row]
        prefix = f"asset/{dataset_row}"
        extras[f"{prefix}/episode_count"] = float(member_ids.numel())
        extras[f"{prefix}/goal_success_count_sum"] = float(completed_subgoals[member_ids].sum().item())
        extras[f"{prefix}/episode_any_success_pulse_sum"] = float(
            (completed_subgoals[member_ids] > 0.0).to(dtype=torch.float32).sum().item()
        )
        extras[f"{prefix}/subgoal_throughput_fixed_horizon_sum"] = float(
            (completed_subgoals[member_ids] / horizon_s).sum().item()
        )
        extras[f"{prefix}/net_rotation_rad_signed_sum"] = float(net_rotation_rad[member_ids].sum().item())
        extras[f"{prefix}/net_rotation_turns_signed_sum"] = float(signed_turns[member_ids].sum().item())
        extras[f"{prefix}/positive_net_rotation_turns_sum"] = float(
            positive_net_rotation_turns[member_ids].sum().item()
        )
        extras[f"{prefix}/reached_positive_full_turn_sum"] = float(
            reached_full_turn[member_ids].to(dtype=torch.float32).sum().item()
        )
        extras[f"{prefix}/reached_negative_full_turn_sum"] = float(
            reached_negative_full_turn[member_ids].to(dtype=torch.float32).sum().item()
        )
        extras[f"{prefix}/reached_positive_30deg_sum"] = float(
            reached_positive_30deg[member_ids].to(dtype=torch.float32).sum().item()
        )
        extras[f"{prefix}/reached_negative_30deg_sum"] = float(
            reached_negative_30deg[member_ids].to(dtype=torch.float32).sum().item()
        )
        safe_duration = episode_duration_s[member_ids].clamp_min(torch.finfo(torch.float32).eps)
        extras[f"{prefix}/time_weighted_signed_speed_rad_s_sum"] = float(
            (net_rotation_rad[member_ids] / safe_duration).sum().item()
        )
        extras[f"{prefix}/episode_duration_s_sum"] = float(episode_duration_s[member_ids].sum().item())
        for term_name, term_bits in termination_bits.items():
            extras[f"{prefix}/termination_{term_name}_sum"] = float(
                term_bits[member_ids].to(dtype=torch.float32).sum().item()
            )
    return extras


def equal_asset_metric_from_extras(extras: Mapping[str, float], metric_sum_name: str) -> float:
    r"""从``asset/<row>/<metric>``与episode_count恢复unique-asset等权均值。"""

    prefixes = sorted(
        key[: -len("/episode_count")]
        for key in extras
        if key.startswith("asset/") and key.endswith("/episode_count") and extras[key] > 0.0
    )
    if not prefixes:
        raise ValueError("equal-asset aggregation requires positive episode counts")
    per_asset = [
        float(extras[f"{prefix}/{metric_sum_name}"]) / float(extras[f"{prefix}/episode_count"])
        for prefix in prefixes
    ]
    return sum(per_asset) / len(per_asset)


__all__ = ["asset_episode_sufficient_statistics", "equal_asset_metric_from_extras"]
