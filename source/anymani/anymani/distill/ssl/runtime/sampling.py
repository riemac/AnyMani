"""Deterministic asset and joint-state minibatch schedule."""


from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OnlineSamplingCfg:


    assets_per_minibatch: int = 64
    q_per_asset_per_minibatch: int = 8
    shuffle_assets: bool = True
    seed: int = 0

    def __post_init__(self) -> None:


        if min(self.assets_per_minibatch, self.q_per_asset_per_minibatch) < 1 or self.seed < 0:
            raise ValueError("online sampling batch axes must be positive and seed must be non-negative")


@dataclass(frozen=True)
class ScheduledMinibatch:


    minibatch_index: int
    epoch_index: int
    minibatch_index_in_epoch: int
    q_block_index: int
    asset_group: int
    asset_indices: tuple[int, ...]
    q_per_asset: int
    resident_asset_indices: tuple[int, ...]
    window_index: int = 0

    @property
    def sample_count(self) -> int:


        return len(self.asset_indices) * self.q_per_asset


@dataclass(frozen=True)
class OnlineSamplingState:


    minibatch_cursor: int


class OnlineMinibatchSchedule:


    def __init__(
        self,
        asset_count: int,
        config: OnlineSamplingCfg,
        *,
        max_epochs: int,
        num_minibatches: int,
        max_resident_assets: int | None = None,
    ) -> None:


        if asset_count < 1 or max_epochs < 1 or num_minibatches < 1:
            raise ValueError("online minibatch schedule requires positive asset, epoch and minibatch counts")
        if asset_count % config.assets_per_minibatch != 0:
            raise ValueError("training asset count must be divisible by assets_per_minibatch")
        self.asset_count = int(asset_count)
        self.config = config
        self.max_epochs = int(max_epochs)
        self.num_minibatches = int(num_minibatches)
        self.total_minibatches = self.max_epochs * self.num_minibatches
        requested_resident = int(max_resident_assets or asset_count)
        if requested_resident < config.assets_per_minibatch:
            raise ValueError("max_resident_assets must cover one complete training minibatch")
        self.max_resident_assets = min(requested_resident, self.asset_count)
        self.minibatches_per_cycle = self.asset_count // config.assets_per_minibatch
        self.minibatches_per_window = max(1, self.max_resident_assets // config.assets_per_minibatch)
        self.minibatch_cursor = 0

    @property
    def minibatches_remaining(self) -> int:


        return self.total_minibatches - self.minibatch_cursor

    @property
    def minibatches_remaining_in_epoch(self) -> int:


        if self.complete:
            return 0
        return self.num_minibatches - self.minibatch_cursor % self.num_minibatches

    @property
    def epoch_boundary(self) -> bool:


        return self.minibatch_cursor % self.num_minibatches == 0

    @property
    def completed_epochs(self) -> int:


        return self.minibatch_cursor // self.num_minibatches

    @property
    def complete(self) -> bool:


        return self.minibatch_cursor >= self.total_minibatches

    @property
    def current_permutation(self) -> tuple[int, ...]:


        if self.complete:
            return tuple()
        cycle_index = self.minibatch_cursor // self.minibatches_per_cycle
        return self._permutation_for_cycle(cycle_index)

    def _permutation_for_cycle(self, cycle_index: int) -> tuple[int, ...]:


        if not self.config.shuffle_assets:
            return tuple(range(self.asset_count))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.seed + cycle_index * 1_000_003)
        return tuple(int(index) for index in torch.randperm(self.asset_count, generator=generator).tolist())

    def next(self) -> ScheduledMinibatch:


        if self.complete:
            raise StopIteration("all configured training minibatches are complete")
        minibatch_index = self.minibatch_cursor
        cycle_index, group_index = divmod(minibatch_index, self.minibatches_per_cycle)
        permutation = self._permutation_for_cycle(cycle_index)
        group_start = group_index * self.config.assets_per_minibatch
        group_stop = group_start + self.config.assets_per_minibatch
        window_index = group_index // self.minibatches_per_window
        window_group_start = window_index * self.minibatches_per_window
        window_group_stop = min(window_group_start + self.minibatches_per_window, self.minibatches_per_cycle)
        window_start = window_group_start * self.config.assets_per_minibatch
        window_stop = window_group_stop * self.config.assets_per_minibatch
        result = ScheduledMinibatch(
            minibatch_index=minibatch_index,
            epoch_index=minibatch_index // self.num_minibatches,
            minibatch_index_in_epoch=minibatch_index % self.num_minibatches,
            q_block_index=cycle_index,
            asset_group=group_index,
            asset_indices=permutation[group_start:group_stop],
            q_per_asset=self.config.q_per_asset_per_minibatch,
            resident_asset_indices=permutation[window_start:window_stop],
            window_index=window_index,
        )
        self.minibatch_cursor += 1
        return result

    def state_dict(self) -> dict[str, object]:


        if not self.epoch_boundary:
            raise RuntimeError("sampling checkpoint is only valid at an epoch boundary")

        return {
            "minibatch_cursor": self.minibatch_cursor,
            "max_epochs": self.max_epochs,
            "num_minibatches": self.num_minibatches,
            "permutation": self.current_permutation,
            "seed": self.config.seed,
            "max_resident_assets": self.max_resident_assets,
        }

    def load_state_dict(
        self,
        state: OnlineSamplingState | dict[str, object],
        *,
        allow_completed_budget_extension: bool = False,
    ) -> None:


        if isinstance(state, dict):
            parsed = sampling_state_from_dict(state)
            if state.get("num_minibatches") != self.num_minibatches:
                raise ValueError("sampling checkpoint num_minibatches does not match trainer config")
            stored_max_epochs = state.get("max_epochs")
            extending_budget = stored_max_epochs != self.max_epochs
            if extending_budget:
                if not allow_completed_budget_extension:
                    raise ValueError("sampling checkpoint max_epochs does not match trainer config")
                if not isinstance(stored_max_epochs, int) or stored_max_epochs >= self.max_epochs:
                    raise ValueError("completed budget extension must increase max_epochs")
                old_total_minibatches = stored_max_epochs * self.num_minibatches
                if parsed.minibatch_cursor != old_total_minibatches:
                    raise ValueError("completed budget extension requires a completed source budget")
            if state.get("seed") != self.config.seed:
                raise ValueError("sampling checkpoint seed does not match trainer config")
            if state.get("max_resident_assets") != self.max_resident_assets:
                raise ValueError("sampling checkpoint resident window cap does not match trainer config")
            raw_permutation = state.get("permutation")
            if not isinstance(raw_permutation, (tuple, list)):
                raise ValueError("sampling checkpoint permutation must be an integer sequence")
            expected_permutation = tuple()
            if parsed.minibatch_cursor < self.total_minibatches and not extending_budget:
                cycle_index = parsed.minibatch_cursor // self.minibatches_per_cycle
                expected_permutation = self._permutation_for_cycle(cycle_index)
            if tuple(raw_permutation) != expected_permutation:
                raise ValueError("sampling checkpoint permutation does not match deterministic schedule")
            state = parsed
        if not 0 <= state.minibatch_cursor <= self.total_minibatches:
            raise ValueError("sampling minibatch cursor lies outside configured budget")
        if state.minibatch_cursor % self.num_minibatches != 0:
            raise ValueError("sampling checkpoint cursor must lie on an epoch boundary")
        self.minibatch_cursor = int(state.minibatch_cursor)


class FixedAssetQSchedule:


    def __init__(
        self,
        asset_count: int,
        *,
        q_per_asset: int,
        assets_per_minibatch: int,
        q_per_asset_per_minibatch: int,
        max_resident_assets: int | None = None,
    ) -> None:


        counts = (asset_count, q_per_asset, assets_per_minibatch, q_per_asset_per_minibatch)
        if min(counts) < 1:
            raise ValueError("fixed evaluation schedule counts must be positive")
        self.asset_count = int(asset_count)
        self.q_per_asset = int(q_per_asset)
        self.assets_per_minibatch = int(assets_per_minibatch)
        self.q_per_asset_per_minibatch = int(q_per_asset_per_minibatch)
        self.max_resident_assets = min(int(max_resident_assets or asset_count), self.asset_count)
        if self.max_resident_assets < self.assets_per_minibatch:
            raise ValueError("evaluation resident window must cover one asset minibatch")
        self.q_blocks = math.ceil(self.q_per_asset / self.q_per_asset_per_minibatch)
        self.minibatches_per_q_block = math.ceil(self.asset_count / self.assets_per_minibatch)
        self.minibatches_per_window = max(1, self.max_resident_assets // self.assets_per_minibatch)
        self.num_minibatches = self.minibatches_per_q_block * self.q_blocks
        self.minibatch_cursor = 0

    @property
    def complete(self) -> bool:


        return self.minibatch_cursor >= self.num_minibatches

    def next(self) -> ScheduledMinibatch:


        if self.complete:
            raise StopIteration("fixed evaluation q-bank is complete")
        minibatch_index = self.minibatch_cursor
        remaining = minibatch_index
        window_index = 0
        window_group_start = 0
        while True:
            groups_in_window = min(
                self.minibatches_per_window,
                self.minibatches_per_q_block - window_group_start,
            )
            minibatches_in_window = groups_in_window * self.q_blocks
            if remaining < minibatches_in_window:
                break
            remaining -= minibatches_in_window
            window_index += 1
            window_group_start += groups_in_window
        group_in_window, q_block_index = divmod(remaining, self.q_blocks)
        asset_group = window_group_start + group_in_window
        group_start = asset_group * self.assets_per_minibatch
        group_stop = min(group_start + self.assets_per_minibatch, self.asset_count)
        window_group_stop = min(window_group_start + self.minibatches_per_window, self.minibatches_per_q_block)
        window_start = window_group_start * self.assets_per_minibatch
        window_stop = min(window_group_stop * self.assets_per_minibatch, self.asset_count)
        q_consumed = q_block_index * self.q_per_asset_per_minibatch
        result = ScheduledMinibatch(
            minibatch_index=minibatch_index,
            epoch_index=-1,
            minibatch_index_in_epoch=minibatch_index,
            q_block_index=q_block_index,
            asset_group=asset_group,
            asset_indices=tuple(range(group_start, group_stop)),
            q_per_asset=min(self.q_per_asset_per_minibatch, self.q_per_asset - q_consumed),
            resident_asset_indices=tuple(range(window_start, window_stop)),
            window_index=window_index,
        )
        self.minibatch_cursor += 1
        return result


def sampling_state_from_dict(payload: dict[str, object]) -> OnlineSamplingState:


    minibatch_cursor = payload.get("minibatch_cursor")
    if not isinstance(minibatch_cursor, int):
        raise ValueError("sampling checkpoint requires integer minibatch_cursor")
    return OnlineSamplingState(minibatch_cursor=minibatch_cursor)


__all__ = [
    "FixedAssetQSchedule",
    "OnlineMinibatchSchedule",
    "OnlineSamplingCfg",
    "OnlineSamplingState",
    "ScheduledMinibatch",
    "sampling_state_from_dict",
]
