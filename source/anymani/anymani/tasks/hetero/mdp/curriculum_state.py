r"""80手MVP per-asset EMA与8-cell median reward-release纯Torch状态。

该模块不导入Isaac/Omni，可由默认pytest、离线diagnostics与训练adapter共同消费。每个formal asset独立维护
$\bar G_i$与counterfactual$\lambda_i$；实际环境reward使用同cell十项资产EMA的普通中位数：

$$
G_c=\operatorname{median}_{i\in c}\bar G_i,
\qquad
\lambda_c=\operatorname{clip}(G_c-1,0,1).
$$
"""

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
    r"""把non-negative net turns线性映射为$[0,1]$ reward-release coefficient。"""

    if release_end_turns <= release_start_turns:
        raise ValueError("reward release end must exceed start")
    return torch.clamp(
        (net_turns - float(release_start_turns)) / float(release_end_turns - release_start_turns),
        min=0.0,
        max=1.0,
    )


def even_median(values: torch.Tensor) -> torch.Tensor:
    r"""返回一维tensor的普通中位数；偶数项取中间两项均值。"""

    if values.ndim != 1 or values.numel() < 1:
        raise ValueError("median requires a non-empty rank-1 tensor")
    ordered = torch.sort(values).values
    middle = ordered.numel() // 2
    return ordered[middle] if ordered.numel() % 2 else 0.5 * (ordered[middle - 1] + ordered[middle])


class HeterogeneousRewardReleaseState:
    r"""持久化80个asset EMA、8个cell状态与per-env实际系数。"""

    def __init__(
        self,
        *,
        dataset_rows_by_asset: Sequence[int],
        cell_ids_by_asset: Sequence[int],
        asset_index_by_env: Sequence[int],
        device: torch.device | str,
    ) -> None:
        r"""验证静态routing并初始化全零课程状态。"""

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
        self.asset_net_turns_ema = torch.zeros(len(rows), device=device)  # `[A]`，$\bar G_i$
        self.asset_episode_updates = torch.zeros(len(rows), dtype=torch.long, device=device)  # reset cohort计数
        self.asset_candidate_lambda = torch.zeros(len(rows), device=device)  # counterfactual$\lambda_i$
        self.cell_net_turns_median = torch.zeros(8, device=device)  # `[8]`，$G_c$
        self.cell_lambda = torch.zeros(8, device=device)  # `[8]`，实际$\lambda_c$
        self.env_lambda = torch.zeros(len(routing), device=device)  # `[N]`，reward/critic视图

    def state_dict(self) -> dict[str, Any]:
        r"""导出完整训练期课程状态，供PPO checkpoint逐值恢复。

        Static dataset rows、cell routing和env routing同时进入state：仅保存EMA tensor而不保存routing会允许
        另一种80-row顺序静默加载，导致$\bar G_i$被赋给错误手型。

        Returns:
            dict[str, Any]: CPU tensors与JSON-safe static routing；不包含设备相关对象。
        """

        return {
            "schema_version": "1.0.0",  # 课程checkpoint ABI
            "dataset_rows_by_asset": list(self.dataset_rows_by_asset),  # formal rows，有序$[A]$
            "cell_ids_by_asset": self.cell_ids_by_asset.detach().cpu(),  # handedness×tip×thumb cell$[A]$
            "asset_index_by_env": self.asset_index_by_env.detach().cpu(),  # round-robin routing$[N]$
            "asset_net_turns_ema": self.asset_net_turns_ema.detach().cpu(),  # $\bar G_i$
            "asset_episode_updates": self.asset_episode_updates.detach().cpu(),  # per-asset update count
            "asset_candidate_lambda": self.asset_candidate_lambda.detach().cpu(),  # counterfactual$\lambda_i$
            "cell_net_turns_median": self.cell_net_turns_median.detach().cpu(),  # $G_c$
            "cell_lambda": self.cell_lambda.detach().cpu(),  # actual$\lambda_c$
            "env_lambda": self.env_lambda.detach().cpu(),  # deployed per-env coefficient
        }

    def load_state_dict(self, state: object) -> None:
        r"""验证static routing后原位恢复全部课程tensor。

        Args:
            state (object): :meth:`state_dict`产生的mapping。

        Raises:
            RuntimeError: schema、dataset rows、cell/env routing、tensor shape或dtype不兼容。
        """

        if not isinstance(state, dict) or state.get("schema_version") != "1.0.0":
            raise RuntimeError("heterogeneous reward-release checkpoint state is missing or incompatible")
        rows = tuple(int(value) for value in state.get("dataset_rows_by_asset", ()))  # checkpoint asset axis
        if rows != self.dataset_rows_by_asset:
            raise RuntimeError("reward-release checkpoint dataset rows disagree with runtime")

        # Static routes先核对再修改任一dynamic tensor，保证失败时当前runtime状态不被部分覆盖。
        expected_static = {
            "cell_ids_by_asset": self.cell_ids_by_asset,
            "asset_index_by_env": self.asset_index_by_env,
        }
        for name, expected in expected_static.items():
            actual = torch.as_tensor(state.get(name), device=expected.device, dtype=expected.dtype)  # exact route
            if actual.shape != expected.shape or not torch.equal(actual, expected):
                raise RuntimeError(f"reward-release checkpoint {name} disagrees with runtime")

        # Dynamic tensors按目标dtype/device原位copy，保留reward/critic已经持有的state对象引用。
        dynamic = {
            "asset_net_turns_ema": self.asset_net_turns_ema,
            "asset_episode_updates": self.asset_episode_updates,
            "asset_candidate_lambda": self.asset_candidate_lambda,
            "cell_net_turns_median": self.cell_net_turns_median,
            "cell_lambda": self.cell_lambda,
            "env_lambda": self.env_lambda,
        }
        restored: dict[str, torch.Tensor] = {}  # 两阶段恢复，shape错误时不产生partial mutation
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
        r"""由刚结束episodes更新per-asset EMA，再发布cell median与per-env coefficient。"""

        if reset_env_ids.ndim != 1 or positive_net_turns_by_env.shape != self.asset_index_by_env.shape:
            raise ValueError("reward release update tensors disagree with environment axis")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("reward release ema_alpha must lie in (0,1]")
        selected_assets = self.asset_index_by_env[reset_env_ids]
        for asset_index in torch.unique(selected_assets).tolist():
            member_ids = reset_env_ids[selected_assets == int(asset_index)]
            batch_mean = positive_net_turns_by_env[member_ids].mean()  # 当前资产reset cohort的$G_i$
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
