'Runtime contracts for frozen Actor switch.'

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


class FrozenActorSwitch:
    'Contract for frozen Actor switch; values 20Hz; units Hz.'

    def __init__(self, actor: nn.Module, replacement: Mapping[str, torch.Tensor] | None, *, boundary_step: int):
        if type(boundary_step) is not int or boundary_step < 1:
            raise ValueError('boundary_step must be a positive completed-step count')
        if actor.training:
            raise ValueError('frozen Actor must be in eval mode')
        self.actor = actor
        self.initial = {key: value.detach().clone() for key, value in actor.state_dict().items()}
        self.replacement = None
        if replacement is not None:
            if set(replacement) != set(self.initial):
                raise ValueError('replacement Actor keys do not match')
            for key, value in replacement.items():
                original = self.initial[key]
                if value.shape != original.shape or value.dtype != original.dtype:
                    raise ValueError(f'replacement Actor tensor contract mismatch: {key}')
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f'nonfinite replacement Actor tensor: {key}')
            self.replacement = {key: value.detach().clone() for key, value in replacement.items()}
        self.boundary_step = boundary_step
        self.phase = 0
        self.boundary_reached = False
        self.event: dict[str, Any] = {}

    def _check_actor(self, expected: Mapping[str, torch.Tensor]) -> None:
        'Check Actor.'
        current = self.actor.state_dict()
        for key, value in expected.items():
            if not torch.equal(current[key], value.to(current[key].device)):
                raise RuntimeError(f'frozen Actor parameters changed: {key}')

    @torch.no_grad()
    def apply(self, completed_steps: int, continuity_tensors: Mapping[str, torch.Tensor]) -> None:
        'Handle apply.'
        if self.boundary_reached:
            raise ValueError('actor boundary already applied')
        if completed_steps != self.boundary_step:
            raise ValueError('actor switch called outside its declared boundary')
        if not continuity_tensors:
            raise ValueError('state continuity requires nonempty tensor witnesses')
        self._check_actor(self.initial)
        snapshots = {key: value.detach().clone() for key, value in continuity_tensors.items()}
        rng = torch.get_rng_state().clone()
        devices = sorted({value.device.index for value in self.initial.values() if value.is_cuda})
        cuda_rng = {device: torch.cuda.get_rng_state(device).clone() for device in devices}
        if self.replacement is not None:
            self.actor.load_state_dict(self.replacement, strict=True)
        expected = self.replacement if self.replacement is not None else self.initial
        self._check_actor(expected)
        for key, snapshot in snapshots.items():
            if not torch.equal(continuity_tensors[key], snapshot):
                raise RuntimeError(f'state continuity violated during Actor switch: {key}')
        if not torch.equal(rng, torch.get_rng_state()) or any(
            not torch.equal(state, torch.cuda.get_rng_state(device)) for device, state in cuda_rng.items()
        ):
            raise RuntimeError('Torch random state changed during Actor switch')
        self.boundary_reached = True
        self.phase = int(self.replacement is not None)
        self.event = {
            'state_continuity_check': 'bitwise-equal', 'torch_rng_check': 'bitwise-equal',
            'prefix_actor_check': 'bitwise-equal', 'replacement_actor_check': 'bitwise-equal',
            'checked_tensor_names': sorted(continuity_tensors),
        }

    def finish(self) -> dict[str, Any]:
        'Finish the declared contract.'
        expected = self.replacement if self.phase == 1 else self.initial
        assert expected is not None
        self._check_actor(expected)
        return {
            'boundary_step': self.boundary_step, 'boundary_reached': self.boundary_reached,
            'replacement_requested': self.replacement is not None, 'performed': self.phase == 1,
            'final_actor_check': 'bitwise-equal', **self.event,
        }
