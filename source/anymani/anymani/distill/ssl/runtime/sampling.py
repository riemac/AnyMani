r"""Trainer-owned 在线资产打乱、minibatch 分组与 coverage epoch 日程。

该层只决定每次选择哪些 catalog rows、每项资产需要多少个新 q。具体 q 分布由 method 提供的
state sampler 实现；query、sigma、edge 与 privileged target 仍由 representation realization 负责。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OnlineSamplingCfg:
    r"""每资产 coverage、minibatch 两个基本轴与 deterministic asset shuffle。"""

    epochs: int = 20  # coverage epochs；每轮每资产都获得同样的新 q 数
    q_per_asset_per_epoch: int = 256  # $N_q^{epoch}$
    assets_per_minibatch: int = 2  # $N_{asset}^{mb}$，尾组可以更小
    q_per_asset_per_minibatch: int = 2  # $N_q^{mb}$，尾 q round 可以更小
    shuffle_assets: bool = True  # 每个 q round 重新打乱全部平等 assets
    seed: int = 0  # asset permutation 与每资产 state sampler 的根 seed

    def __post_init__(self) -> None:
        r"""验证所有 coverage 与 minibatch 轴严格为正。"""

        values = (
            self.epochs,
            self.q_per_asset_per_epoch,
            self.assets_per_minibatch,
            self.q_per_asset_per_minibatch,
        )
        if min(values) < 1 or self.seed < 0:
            raise ValueError("online sampling epochs/batch axes must be positive and seed must be non-negative")


@dataclass(frozen=True)
class ScheduledMinibatch:
    r"""一个尚未 realization 的资产组与每资产 q 数。"""

    epoch: int  # 从 0 开始的当前 coverage epoch
    q_round: int  # 当前 epoch 内第几个 q block
    asset_group: int  # 当前 permutation 内第几个资产 minibatch
    asset_indices: tuple[int, ...]  # catalog row indices；尾组保持真实长度
    q_per_asset: int  # 当前 round 每项资产需要的新 q 数

    @property
    def sample_count(self) -> int:
        r"""返回模型 forward 的 realization batch size。"""

        return len(self.asset_indices) * self.q_per_asset


@dataclass(frozen=True)
class OnlineSamplingState:
    r"""checkpoint optimizer boundary 可恢复的显式 schedule cursor。"""

    epoch: int
    q_round: int
    asset_group: int


class OnlineMinibatchSchedule:
    r"""按最新 Research 合同遍历平等资产轴，不以缓存窗口改变统计顺序。"""

    def __init__(self, asset_count: int, config: OnlineSamplingCfg) -> None:
        r"""保存 catalog 长度和纯离散日程；不创建 q sampler 或设备状态。"""

        if asset_count < 1:
            raise ValueError("online minibatch schedule requires at least one train asset")
        self.asset_count = int(asset_count)
        self.config = config
        self.epoch = 0
        self.q_round = 0
        self.asset_group = 0

    @property
    def q_rounds_per_epoch(self) -> int:
        r"""返回覆盖每资产 q budget 所需 round 数，最后一轮可较小。"""

        return math.ceil(self.config.q_per_asset_per_epoch / self.config.q_per_asset_per_minibatch)

    @property
    def asset_groups_per_round(self) -> int:
        r"""返回一次全资产 permutation 的 minibatch 数，最后一组可较小。"""

        return math.ceil(self.asset_count / self.config.assets_per_minibatch)

    @property
    def minibatches_per_epoch(self) -> int:
        r"""返回由两个基本轴推导的完整 epoch minibatch 数。"""

        return self.q_rounds_per_epoch * self.asset_groups_per_round

    @property
    def minibatches_remaining_in_epoch(self) -> int:
        r"""返回当前 epoch 还未消费的真实 minibatch 数。"""

        if self.complete:
            return 0
        return (self.q_rounds_per_epoch - self.q_round) * self.asset_groups_per_round - self.asset_group

    @property
    def current_permutation(self) -> tuple[int, ...]:
        r"""返回当前 q round 的确定性资产 permutation，供 checkpoint 审计。"""

        return self._permutation() if not self.complete else tuple()

    @property
    def complete(self) -> bool:
        r"""返回全部 coverage epochs 是否完成。"""

        return self.epoch >= self.config.epochs

    def _permutation(self) -> tuple[int, ...]:
        r"""由 `(seed,epoch,q_round)` 无状态重建当前全资产顺序。"""

        if not self.config.shuffle_assets:
            return tuple(range(self.asset_count))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.seed + self.epoch * 1_000_003 + self.q_round)
        return tuple(int(index) for index in torch.randperm(self.asset_count, generator=generator).tolist())

    def _q_count(self) -> int:
        r"""返回当前 q round 的真实每资产 q 数，不重复样本补齐尾块。"""

        consumed = self.q_round * self.config.q_per_asset_per_minibatch
        remaining = self.config.q_per_asset_per_epoch - consumed
        return min(self.config.q_per_asset_per_minibatch, remaining)

    def next(self) -> ScheduledMinibatch:
        r"""返回下一资产组并推进 schedule cursor。"""

        if self.complete:
            raise StopIteration("all configured coverage epochs are complete")
        permutation = self._permutation()
        start = self.asset_group * self.config.assets_per_minibatch
        stop = min(start + self.config.assets_per_minibatch, self.asset_count)
        result = ScheduledMinibatch(
            epoch=self.epoch,
            q_round=self.q_round,
            asset_group=self.asset_group,
            asset_indices=permutation[start:stop],
            q_per_asset=self._q_count(),
        )
        self.asset_group += 1
        if self.asset_group >= self.asset_groups_per_round:
            self.asset_group = 0
            self.q_round += 1
            if self.q_round >= self.q_rounds_per_epoch:
                self.q_round = 0
                self.epoch += 1
        return result

    def state_dict(self) -> dict[str, object]:
        r"""返回 cursor、当前 permutation 与采样 seed 的可序列化状态。"""

        return {
            "epoch": self.epoch,
            "q_round": self.q_round,
            "asset_group": self.asset_group,
            "permutation": self.current_permutation,
            "seed": self.config.seed,
        }

    def load_state_dict(self, state: OnlineSamplingState | dict[str, object]) -> None:
        r"""恢复 cursor，并拒绝越出当前 config 预算的状态。"""

        if isinstance(state, dict):
            parsed = sampling_state_from_dict(state)
            raw_permutation = state.get("permutation")
            if raw_permutation is not None:
                if not isinstance(raw_permutation, (tuple, list)) or not all(
                    isinstance(index, int) for index in raw_permutation
                ):
                    raise ValueError("sampling checkpoint permutation must be an integer sequence")
                if tuple(raw_permutation) != self._permutation_for(parsed):
                    raise ValueError("sampling checkpoint permutation does not match deterministic schedule")
            if state.get("seed") != self.config.seed:
                raise ValueError("sampling checkpoint seed does not match trainer config")
            state = parsed
        if state.epoch < 0 or state.epoch > self.config.epochs:
            raise ValueError("sampling state epoch lies outside configured coverage")
        if state.epoch == self.config.epochs:
            if state.q_round != 0 or state.asset_group != 0:
                raise ValueError("completed sampling state must have zero inner cursors")
        elif not (0 <= state.q_round < self.q_rounds_per_epoch):
            raise ValueError("sampling state q_round lies outside one epoch")
        elif not (0 <= state.asset_group < self.asset_groups_per_round):
            raise ValueError("sampling state asset_group lies outside one q round")
        self.epoch = int(state.epoch)
        self.q_round = int(state.q_round)
        self.asset_group = int(state.asset_group)

    def _permutation_for(self, state: OnlineSamplingState) -> tuple[int, ...]:
        r"""按任意 checkpoint epoch/round 重建 permutation，不修改当前 cursor。"""

        if state.epoch >= self.config.epochs:
            return tuple()  # 完成态没有下一 q round，和 current_permutation 的审计语义一致
        if not self.config.shuffle_assets:
            return tuple(range(self.asset_count))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.seed + state.epoch * 1_000_003 + state.q_round)
        return tuple(int(index) for index in torch.randperm(self.asset_count, generator=generator).tolist())


def sampling_state_from_dict(payload: dict[str, object]) -> OnlineSamplingState:
    r"""把 checkpoint 基础 mapping 重建为严格 schedule state。"""

    epoch = payload.get("epoch")
    q_round = payload.get("q_round")
    asset_group = payload.get("asset_group")
    if not isinstance(epoch, int) or not isinstance(q_round, int) or not isinstance(asset_group, int):
        raise ValueError("sampling checkpoint requires integer epoch/q_round/asset_group")
    return OnlineSamplingState(epoch, q_round, asset_group)


__all__ = [
    "OnlineMinibatchSchedule",
    "OnlineSamplingCfg",
    "OnlineSamplingState",
    "ScheduledMinibatch",
    "sampling_state_from_dict",
]
