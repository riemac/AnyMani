(
    'Pure-Torch state for per-asset EMA and 8-cell median reward release over 80 '
    'hands. Each formal asset tracks its own EMA and counterfactual coefficient. '
    'The deployed env coefficient uses the ordinary median of the ten asset EMAs '
    'in its cell: G_c = median(EMA_i), lambda_c = clip(G_c - 1, 0, 1).'
)

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

HETERO_REWARD_RELEASE_STATE_ATTR = "_anymani_hetero_reward_release_state"


def release_from_net_turns(
    net_turns: torch.Tensor,
    *,
    release_start_turns: float,
    release_end_turns: float,
) -> torch.Tensor:
    'Map non-negative net turns linearly to a [0,1] reward-release coefficient.'

    if release_end_turns <= release_start_turns:
        raise ValueError("reward release end must exceed start")
    return torch.clamp(
        (net_turns - float(release_start_turns)) / float(release_end_turns - release_start_turns),
        min=0.0,
        max=1.0,
    )


def even_median(values: torch.Tensor) -> torch.Tensor:
    'Return the ordinary median of a 1D tensor; average the two middle values for even counts.'

    if values.ndim != 1 or values.numel() < 1:
        raise ValueError("median requires a non-empty rank-1 tensor")
    ordered = torch.sort(values).values
    middle = ordered.numel() // 2
    return ordered[middle] if ordered.numel() % 2 else 0.5 * (ordered[middle - 1] + ordered[middle])


class HeterogeneousRewardReleaseState:
    'Persist all 80 asset EMAs, 8 cell states, and per-env coefficients.'

    def __init__(
        self,
        *,
        dataset_rows_by_asset: Sequence[int],
        cell_ids_by_asset: Sequence[int],
        asset_index_by_env: Sequence[int],
        device: torch.device | str,
    ) -> None:
        'Validate static routing and initialize all curriculum state to zero.'

        rows = tuple(int(row) for row in dataset_rows_by_asset)
        cells = tuple(int(cell) for cell in cell_ids_by_asset)
        routing = tuple(int(index) for index in asset_index_by_env)
        if not rows or len(rows) != len(cells) or len(set(rows)) != len(rows):
            raise ValueError("reward release requires unique asset rows aligned with cell IDs")
        if set(cells) - set(range(8)):
            raise ValueError("reward release cell IDs must lie in [0,7]")
        if not routing or any(index < 0 or index >= len(rows) for index in routing):
            raise ValueError("reward release env routing references a missing asset")
        self.dataset_rows_by_asset = rows
        self.cell_ids_by_asset = torch.tensor(cells, dtype=torch.long, device=device)  # `[A]`
        self.asset_index_by_env = torch.tensor(routing, dtype=torch.long, device=device)  # `[N]`
        self.asset_net_turns_ema = torch.zeros(len(rows), device=device)  # Per-asset EMA, shape [A].
        self.asset_episode_updates = torch.zeros(len(rows), dtype=torch.long, device=device)  # Reset-cohort counts.
        self.asset_candidate_lambda = torch.zeros(len(rows), device=device)  # Counterfactual per-asset coefficient lambda_i.
        self.cell_net_turns_median = torch.zeros(8, device=device)  # Per-cell returns, shape [8].
        self.cell_lambda = torch.zeros(8, device=device)  # Deployed per-cell coefficient, shape [8].
        self.env_lambda = torch.zeros(len(routing), device=device)  # Per-env reward/critic view, shape [N].

    def state_dict(self) -> dict[str, Any]:
        (
            'Export complete training state for exact PPO resume. Include dataset rows '
            'and cell/env routing so an EMA cannot be silently assigned to another hand '
            'ordering. Return CPU tensors and JSON-safe routing only.'
        )

        return {
            "schema_version": "1.0.0",  # Curriculum checkpoint ABI.
            "dataset_rows_by_asset": list(self.dataset_rows_by_asset),  # Ordered formal asset rows, shape [A].
            "cell_ids_by_asset": self.cell_ids_by_asset.detach().cpu(),  # Handedness x tip x thumb cell for each asset, shape [A].
            "asset_index_by_env": self.asset_index_by_env.detach().cpu(),  # Round-robin env routing, shape [N].
            "asset_net_turns_ema": self.asset_net_turns_ema.detach().cpu(),  # Per-asset EMA.
            "asset_episode_updates": self.asset_episode_updates.detach().cpu(),  # per-asset update count
            "asset_candidate_lambda": self.asset_candidate_lambda.detach().cpu(),  # Counterfactual per-asset coefficient.
            "cell_net_turns_median": self.cell_net_turns_median.detach().cpu(),  # Cell score G_c.
            "cell_lambda": self.cell_lambda.detach().cpu(),  # Deployed cell coefficient lambda_c.
            "env_lambda": self.env_lambda.detach().cpu(),  # deployed per-env coefficient
        }

    def load_state_dict(self, state: object) -> None:
        (
            'Validate static routing, then restore all curriculum tensors in place. Raise '
            'RuntimeError for incompatible schema, rows, routing, shape, or dtype.'
        )

        if not isinstance(state, dict) or state.get("schema_version") != "1.0.0":
            raise RuntimeError("heterogeneous reward-release checkpoint state is missing or incompatible")
        rows = tuple(int(value) for value in state.get("dataset_rows_by_asset", ()))  # checkpoint asset axis
        if rows != self.dataset_rows_by_asset:
            raise RuntimeError("reward-release checkpoint dataset rows disagree with runtime")

        # Validate static routes before changing any dynamic tensor, so failures cannot partially overwrite runtime state.
        expected_static = {
            "cell_ids_by_asset": self.cell_ids_by_asset,
            "asset_index_by_env": self.asset_index_by_env,
        }
        for name, expected in expected_static.items():
            actual = torch.as_tensor(state.get(name), device=expected.device, dtype=expected.dtype)  # exact route
            if actual.shape != expected.shape or not torch.equal(actual, expected):
                raise RuntimeError(f"reward-release checkpoint {name} disagrees with runtime")

        # Copy dynamic tensors in place using target dtype/device; preserve state references held by reward and critic terms.
        dynamic = {
            "asset_net_turns_ema": self.asset_net_turns_ema,
            "asset_episode_updates": self.asset_episode_updates,
            "asset_candidate_lambda": self.asset_candidate_lambda,
            "cell_net_turns_median": self.cell_net_turns_median,
            "cell_lambda": self.cell_lambda,
            "env_lambda": self.env_lambda,
        }
        restored: dict[str, torch.Tensor] = {}  # Restore in two stages so shape errors cause no partial mutation.
        for name, target in dynamic.items():
            value = torch.as_tensor(state.get(name), device=target.device, dtype=target.dtype)  # checkpoint -> runtime
            if value.shape != target.shape or not bool(torch.isfinite(value.float()).all().item()):
                raise RuntimeError(f"reward-release checkpoint {name} is malformed")
            restored[name] = value
        for name, target in dynamic.items():
            target.copy_(restored[name])  # exact optimizer-resume curriculum continuity

    def update(
        self,
        *,
        reset_env_ids: torch.Tensor,
        positive_net_turns_by_env: torch.Tensor,
        ema_alpha: float,
        release_start_turns: float,
        release_end_turns: float,
    ) -> None:
        'Update per-asset EMA from completed episodes, then publish cell medians and per-env coefficients.'

        if reset_env_ids.ndim != 1 or positive_net_turns_by_env.shape != self.asset_index_by_env.shape:
            raise ValueError("reward release update tensors disagree with environment axis")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("reward release ema_alpha must lie in (0,1]")
        selected_assets = self.asset_index_by_env[reset_env_ids]
        for asset_index in torch.unique(selected_assets).tolist():
            member_ids = reset_env_ids[selected_assets == int(asset_index)]
            batch_mean = positive_net_turns_by_env[member_ids].mean()  # G_i for the current asset reset cohort.
            self.asset_net_turns_ema[asset_index] = (
                (1.0 - ema_alpha) * self.asset_net_turns_ema[asset_index] + ema_alpha * batch_mean.detach()
            )
            self.asset_episode_updates[asset_index] += 1
        self.asset_candidate_lambda.copy_(
            release_from_net_turns(
                self.asset_net_turns_ema,
                release_start_turns=release_start_turns,
                release_end_turns=release_end_turns,
            )
        )
        for cell_id in range(8):
            members = self.cell_ids_by_asset == cell_id
            if bool(members.any().item()):
                self.cell_net_turns_median[cell_id] = even_median(self.asset_net_turns_ema[members])
        self.cell_lambda.copy_(
            release_from_net_turns(
                self.cell_net_turns_median,
                release_start_turns=release_start_turns,
                release_end_turns=release_end_turns,
            )
        )
        self.env_lambda.copy_(self.cell_lambda[self.cell_ids_by_asset[self.asset_index_by_env]])


__all__ = [
    "HETERO_REWARD_RELEASE_STATE_ATTR",
    "HeterogeneousRewardReleaseState",
    "even_median",
    "release_from_net_turns",
]
