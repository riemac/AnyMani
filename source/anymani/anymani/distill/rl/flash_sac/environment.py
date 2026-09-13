r"""SAC专用的终止状态桥接，复用现有任务和冻结几何transport而不修改PPO路径。

IsaacLab step会在返回观测前自动reset。回放所需next必须是该转移的物理终点，
因此仅在本SAC环境实例的reset入口捕获紧凑帧；实时Actor仍消费正常reset后的观测。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from .observations import compact_observation, compact_raw_groups


@dataclass
class SACStep:
    r"""实时下一观测与回放终点分开命名，防止把新回合初态误作旧转移的next。"""

    observation: dict[str, torch.Tensor]  # 实时Actor下一次控制使用，done行已经reset。
    reward: torch.Tensor  # [N]原始任务奖励，不含学习器归一化。
    terminated: torch.Tensor  # [N]真实物理失败。
    truncated: torch.Tensor  # [N]当前120秒有限时域结束，学习器也停止bootstrap。
    next_compact: dict[str, torch.Tensor]  # done行是reset之前的物理终点，不含完整历史/几何。
    extras: dict[str, Any]


class FlashSACEnvironment:
    r"""只接管新创建SAC实例的reset回调；传入的transport必须提供原结构化任务接口。"""

    def __init__(self, transport: Any):
        self.transport = transport
        self.raw = transport.unwrapped
        self.num_envs = int(transport.num_envs)
        self._original_reset = self.raw._reset_idx
        self._capture_enabled = False
        self._captured: dict[str, torch.Tensor] = {}
        self._captured_mask: torch.Tensor | None = None
        self.raw._reset_idx = self._capture_and_reset  # 仅此环境实例，不改类或在训PPO的对象。

    def _capture_and_reset(self, env_ids: torch.Tensor) -> None:
        r"""在物理状态被新预抓取覆盖前读取单步帧；历史由回放按原episode重建。"""
        if self._capture_enabled and env_ids.numel():
            groups = self.raw.observation_manager.compute(update_history=False)
            compact = compact_raw_groups(groups)
            compact = {name: self.transport._float(value) for name, value in compact.items()}  # 与普通next使用同一FP32/clip100传输。
            if self._captured_mask is None:
                first = next(iter(compact.values()))
                self._captured_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=first.device)
            for name, value in compact.items():
                if name not in self._captured:
                    self._captured[name] = torch.empty_like(value)  # 只在被标记的done行读取。
                self._captured[name][env_ids] = value[env_ids].detach()
            self._captured_mask[env_ids] = True
        self._original_reset(env_ids)

    def reset(self) -> dict[str, torch.Tensor]:
        r"""初始reset不是训练转移；所有Actor历史由原任务正常初始化。"""
        self._capture_enabled = False
        result = self.transport.reset()["obs"]
        self._capture_enabled = True
        return result

    def step(self, actions: torch.Tensor) -> SACStep:
        r"""推进原20Hz策略步，分别交付控制观测和不跨reset的回放终点。"""
        if self._captured_mask is not None:
            self._captured_mask.zero_()
        observation, reward, done, extras = self.transport.step(actions)
        live = observation["obs"]
        terminated = self.raw.reset_terminated.detach().to(reward.device).bool().clone()
        truncated = self.raw.reset_time_outs.detach().to(reward.device).bool().clone()
        expected_done = terminated | truncated
        if not torch.equal(done.bool(), expected_done):
            raise RuntimeError("transport done disagrees with physical terminated/truncated flags")
        next_compact = {name: value.clone() for name, value in compact_observation(live).items()}
        if bool(expected_done.any()):
            if self._captured_mask is None or not bool(self._captured_mask.to(expected_done.device)[expected_done].all()):
                raise RuntimeError("SAC transition is missing its pre-reset terminal observation")
            for name, value in next_compact.items():
                value[expected_done] = self._captured[name].to(value.device)[expected_done]
        return SACStep(live, reward.reshape(-1), terminated, truncated, next_compact, dict(extras))

    def close(self) -> None:
        r"""恢复原实例回调并关闭transport，已结束/删失回合仍由原记录器写出。"""
        self._capture_enabled = False
        self.raw._reset_idx = self._original_reset
        self._captured.clear()
        self._captured_mask = None
        self.transport.close()

    def compact(self, observation: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        r"""对实时观测只提取允许进入回放的动态字段，接口供采样循环统一使用。"""
        return compact_observation(observation)
