r"""Trainer-owned 在线 epoch/minibatch 日程与固定评估 q-bank 日程。

训练预算由 ``max_epochs × num_minibatches`` 给出，其中 ``num_minibatches`` 是每个 epoch
新生成的批数。每个 minibatch 包含固定数量的资产，每项资产产生固定数量的新 Sobol 构型；
训练集走完后以新的确定性 permutation 继续，epoch 本身不表示完整 catalog 遍历。
独立 validation/evaluation 仍按每资产固定 q-bank 完整遍历，因此使用与训练解耦的评估日程。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OnlineSamplingCfg:
    r"""一次新训练 minibatch 的资产轴、q 轴与确定性资产打乱。

    数值锚点 ``64 assets × 8 q/asset`` 对应一次模型 forward 的 512 个等权
    ``(asset,q)`` 样本。总共生成多少批由 Trainer 的 ``num_minibatches`` 声明。
    """

    assets_per_minibatch: int = 64  # 每批互异训练资产数 $N_{asset}^{mb}$
    q_per_asset_per_minibatch: int = 8  # 每项资产新生成的 Sobol 构型数 $N_q^{mb}$
    shuffle_assets: bool = True  # 每次走完 train catalog 后生成新的确定性排列
    seed: int = 0  # asset permutation 与每资产 state sampler 的共同根 seed

    def __post_init__(self) -> None:
        r"""验证训练 minibatch 的两个统计轴严格为正。"""

        if min(self.assets_per_minibatch, self.q_per_asset_per_minibatch) < 1 or self.seed < 0:
            raise ValueError("online sampling batch axes must be positive and seed must be non-negative")


@dataclass(frozen=True)
class ScheduledMinibatch:
    r"""一个尚未 realization 的资产组及其完整 resident window。

    ``q_block_index`` 是同一资产第几次获得新 Sobol q-block。训练时它等于 catalog
    permutation 的轮次；固定评估时它等于该资产的 q-bank block 序号。
    """

    minibatch_index: int  # 当前日程中的全局 minibatch 序号，从 0 开始
    epoch_index: int  # 训练 epoch 序号；固定评估使用 -1
    minibatch_index_in_epoch: int  # 当前 epoch 内序号；固定评估等于全局评估批序号
    q_block_index: int  # 当前资产组的 q-block 序号，用于 anchor bank 轮换
    asset_group: int  # 当前 catalog permutation 或固定 q-round 内的资产组序号
    asset_indices: tuple[int, ...]  # catalog row indices；训练批固定长度，评估尾批可较短
    q_per_asset: int  # 当前组中每项资产需要生成的新 q 数
    resident_asset_indices: tuple[int, ...]  # 当前 GPU window 的完整 catalog 下标
    window_index: int = 0  # 当前 catalog permutation 中的 resident window 序号

    @property
    def sample_count(self) -> int:
        r"""返回本次模型 forward 的等权 ``(asset,q)`` 样本数。"""

        return len(self.asset_indices) * self.q_per_asset


@dataclass(frozen=True)
class OnlineSamplingState:
    r"""checkpoint optimizer boundary 可恢复的全局训练 minibatch 游标。"""

    minibatch_cursor: int  # 已经完成 realization 的新 minibatch 数


class OnlineMinibatchSchedule:
    r"""生成恰好 ``max_epochs × num_minibatches`` 个固定形状训练批。

    训练资产数必须被 ``assets_per_minibatch`` 整除，以保证显式预算恒等式
    $N_{asset-use}=N_{mb}N_{asset}^{mb}$ 不被隐式尾批改变。``max_resident_assets``
    只决定多少个完整 minibatch 同时驻留，不改变资产顺序或统计预算。
    """

    def __init__(
        self,
        asset_count: int,
        config: OnlineSamplingCfg,
        *,
        max_epochs: int,
        num_minibatches: int,
        max_resident_assets: int | None = None,
    ) -> None:
        r"""保存训练 catalog、显式批数、设备窗口上限和确定性游标。"""

        if asset_count < 1 or max_epochs < 1 or num_minibatches < 1:
            raise ValueError("online minibatch schedule requires positive asset, epoch and minibatch counts")
        if asset_count % config.assets_per_minibatch != 0:
            raise ValueError("training asset count must be divisible by assets_per_minibatch")
        self.asset_count = int(asset_count)  # 当前 train split 的互异资产总数
        self.config = config  # 每个新 minibatch 的固定资产/q 轴
        self.max_epochs = int(max_epochs)  # 训练回合上限，不等于 catalog cycle
        self.num_minibatches = int(num_minibatches)  # 每个 epoch 需要生成的新数据批数
        self.total_minibatches = self.max_epochs * self.num_minibatches  # 全运行新批预算
        requested_resident = int(max_resident_assets or asset_count)  # 用户声明的设备资产容量
        if requested_resident < config.assets_per_minibatch:
            raise ValueError("max_resident_assets must cover one complete training minibatch")
        self.max_resident_assets = min(requested_resident, self.asset_count)  # 不超过真实 catalog
        self.minibatches_per_cycle = self.asset_count // config.assets_per_minibatch  # 一次全 catalog 的完整批数
        self.minibatches_per_window = max(1, self.max_resident_assets // config.assets_per_minibatch)
        self.minibatch_cursor = 0  # 下一个尚未 realization 的全局 minibatch 序号

    @property
    def minibatches_remaining(self) -> int:
        r"""返回整个运行尚未生成的新训练 minibatch 数。"""

        return self.total_minibatches - self.minibatch_cursor

    @property
    def minibatches_remaining_in_epoch(self) -> int:
        r"""返回当前 epoch 尚未生成的新 minibatch 数。"""

        if self.complete:
            return 0
        return self.num_minibatches - self.minibatch_cursor % self.num_minibatches

    @property
    def epoch_boundary(self) -> bool:
        r"""返回游标是否位于完整 epoch 边界。"""

        return self.minibatch_cursor % self.num_minibatches == 0

    @property
    def completed_epochs(self) -> int:
        r"""返回已经完整生成并可安全 checkpoint 的 epoch 数。"""

        return self.minibatch_cursor // self.num_minibatches

    @property
    def complete(self) -> bool:
        r"""返回显式 epoch/minibatch 预算是否已经耗尽。"""

        return self.minibatch_cursor >= self.total_minibatches

    @property
    def current_permutation(self) -> tuple[int, ...]:
        r"""返回下一批所属的确定性 train-catalog permutation，供 checkpoint 审计。"""

        if self.complete:
            return tuple()
        cycle_index = self.minibatch_cursor // self.minibatches_per_cycle
        return self._permutation_for_cycle(cycle_index)

    def _permutation_for_cycle(self, cycle_index: int) -> tuple[int, ...]:
        r"""由 ``(seed,cycle_index)`` 无状态重建一轮资产排列。"""

        if not self.config.shuffle_assets:
            return tuple(range(self.asset_count))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.seed + cycle_index * 1_000_003)
        return tuple(int(index) for index in torch.randperm(self.asset_count, generator=generator).tolist())

    def next(self) -> ScheduledMinibatch:
        r"""返回下一完整训练批，并把全局新数据游标推进一位。"""

        if self.complete:
            raise StopIteration("all configured training minibatches are complete")
        minibatch_index = self.minibatch_cursor  # 当前批的稳定全局身份
        cycle_index, group_index = divmod(minibatch_index, self.minibatches_per_cycle)
        permutation = self._permutation_for_cycle(cycle_index)  # 当前 catalog 轮次的资产排列
        group_start = group_index * self.config.assets_per_minibatch  # 当前批在排列中的左边界
        group_stop = group_start + self.config.assets_per_minibatch  # 训练批禁止隐式尾组
        window_index = group_index // self.minibatches_per_window  # 当前批所属的 GPU 驻留窗口
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
        self.minibatch_cursor += 1  # realization 后下一次读取下一批新资产/q
        return result

    def state_dict(self) -> dict[str, object]:
        r"""在完整 epoch 边界返回预算、游标、下一轮排列与采样 seed。"""

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
        r"""恢复全局游标；extension 只承接已完全耗尽的较小旧预算。"""

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
    r"""完整遍历每项评估资产的固定 q-bank，保留真实资产尾批与 q 尾块。

    该日程只服务独立 validation、可选 training-q-bank 和 evaluation，不参与训练预算或 resume。
    """

    def __init__(
        self,
        asset_count: int,
        *,
        q_per_asset: int,
        assets_per_minibatch: int,
        q_per_asset_per_minibatch: int,
        max_resident_assets: int | None = None,
    ) -> None:
        r"""保存固定评估 bank 的三个离散轴和流式游标。"""

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
        r"""返回固定评估 q-bank 是否已经完整遍历。"""

        return self.minibatch_cursor >= self.num_minibatches

    def next(self) -> ScheduledMinibatch:
        r"""按 resident-window → asset-group → q-block 顺序返回下一评估批。

        同一资产组先连续完成全部固定 q-bank，再切换设备资产。该顺序不改变每资产 Sobol cursor、
        q 数或统计测度，却让真实 8-asset device subwindow 可复用 CPU source 与 Warp BVH，避免每个
        q-block 都重新物化同一 owner geometry。
        """

        if self.complete:
            raise StopIteration("fixed evaluation q-bank is complete")
        minibatch_index = self.minibatch_cursor
        remaining = minibatch_index  # 在 window-major 展开序列中的局部游标
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
    r"""把 checkpoint 基础 mapping 重建为严格的全局 minibatch 游标。"""

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
