r"""在声明的控制步边界交接两个冻结Actor，保持连续轨迹的其他状态。

设边界为k，前k个动作来自初始Actor，第k+1个动作起来自接替Actor。
只替换同一module的参数/buffer，不重建环境、控制器、观察历史或几何缓存。
replacement=None用于全程同策略的参照，仍执行相同的边界和状态检查。
此模块不持有环境或optimizer；调用方提供需要证明连续的实际状态tensor引用。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


class FrozenActorSwitch:
    r"""仅允许一次定时交接，并分别验证前后两个参数阶段保持冻结。

    Args:
        actor: 已严格载入初始checkpoint并处于eval模式的Actor。
        replacement: 接替checkpoint的完整Actor state_dict，None代表不切换参照。
        boundary_step: 已执行的旧动作数；20Hz下300对应15秒。
    """

    def __init__(self, actor: nn.Module, replacement: Mapping[str, torch.Tensor] | None, *, boundary_step: int):
        if type(boundary_step) is not int or boundary_step < 1:
            raise ValueError('boundary_step must be a positive completed-step count')
        if actor.training:
            raise ValueError('frozen Actor must be in eval mode')  # 避免随机层或训练统计介入。
        self.actor = actor  # 保持同一对象，不影响调用方已绑定的forward。
        self.initial = {key: value.detach().clone() for key, value in actor.state_dict().items()}
        self.replacement = None  # 完整校验之后才保留候选参数。
        if replacement is not None:
            if set(replacement) != set(self.initial):
                raise ValueError('replacement Actor keys do not match')  # 禁止部分载入。
            for key, value in replacement.items():
                original = self.initial[key]  # 当前module的形状/dtype合同。
                if value.shape != original.shape or value.dtype != original.dtype:
                    raise ValueError(f'replacement Actor tensor contract mismatch: {key}')
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f'nonfinite replacement Actor tensor: {key}')
            self.replacement = {key: value.detach().clone() for key, value in replacement.items()}
        self.boundary_step = boundary_step  # 全局控制步边界，不随单个环境reset而改变。
        self.phase = 0  # 0为初始checkpoint，1为接替checkpoint。
        self.boundary_reached = False  # 全部首轨迹提前结束时边界可以没有到达。
        self.event: dict[str, Any] = {}  # 保存实际检查结果，不预填“通过”。

    def _check_actor(self, expected: Mapping[str, torch.Tensor]) -> None:
        r"""逐值检查实际Actor参数与该阶段的冻结参照。"""
        current = self.actor.state_dict()  # 包含参数及持久buffer。
        for key, value in expected.items():
            if not torch.equal(current[key], value.to(current[key].device)):
                raise RuntimeError(f'frozen Actor parameters changed: {key}')

    @torch.no_grad()
    def apply(self, completed_steps: int, continuity_tensors: Mapping[str, torch.Tensor]) -> None:
        r"""在完成第k个动作之后、计算第k+1个动作之前交接。

        continuity_tensors来自实际观察历史、控制目标及物理状态，不接受空见证。
        在参数copy前后逐值核对这些tensor，并检查Torch CPU/CUDA随机状态。
        """
        if self.boundary_reached:
            raise ValueError('actor boundary already applied')
        if completed_steps != self.boundary_step:
            raise ValueError('actor switch called outside its declared boundary')
        if not continuity_tensors:
            raise ValueError('state continuity requires nonempty tensor witnesses')
        self._check_actor(self.initial)  # 旧策略在交接之前完整冻结。
        snapshots = {key: value.detach().clone() for key, value in continuity_tensors.items()}
        rng = torch.get_rng_state().clone()  # 参数copy不应改变后续随机流。
        devices = sorted({value.device.index for value in self.initial.values() if value.is_cuda})
        cuda_rng = {device: torch.cuda.get_rng_state(device).clone() for device in devices}
        if self.replacement is not None:
            self.actor.load_state_dict(self.replacement, strict=True)  # 唯一主动变化。
        expected = self.replacement if self.replacement is not None else self.initial
        self._check_actor(expected)  # copy结束立即核对完整接替权重。
        for key, snapshot in snapshots.items():
            if not torch.equal(continuity_tensors[key], snapshot):
                raise RuntimeError(f'state continuity violated during Actor switch: {key}')
        if not torch.equal(rng, torch.get_rng_state()) or any(
            not torch.equal(state, torch.cuda.get_rng_state(device)) for device, state in cuda_rng.items()
        ):
            raise RuntimeError('Torch random state changed during Actor switch')
        self.boundary_reached = True  # 只有参数与连续性都通过才发布交接完成事件。
        self.phase = int(self.replacement is not None)  # 不切换参照全程保持0。
        self.event = {
            'state_continuity_check': 'bitwise-equal', 'torch_rng_check': 'bitwise-equal',
            'prefix_actor_check': 'bitwise-equal', 'replacement_actor_check': 'bitwise-equal',
            'checked_tensor_names': sorted(continuity_tensors),  # 精确说明检查了哪些连续状态。
        }

    def finish(self) -> dict[str, Any]:
        r"""结束时检查实际所处阶段的Actor，返回真实边界是否到达的证据。"""
        expected = self.replacement if self.phase == 1 else self.initial  # 未交接时仍为初始策略。
        assert expected is not None  # phase=1只在存在replacement且copy成功之后出现。
        self._check_actor(expected)  # 接替之后没有发生优化或隐藏参数修改。
        return {
            'boundary_step': self.boundary_step, 'boundary_reached': self.boundary_reached,
            'replacement_requested': self.replacement is not None, 'performed': self.phase == 1,
            'final_actor_check': 'bitwise-equal', **self.event,
        }  # 工程见证不自动晋升为旋转能力结论。
