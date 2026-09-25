'Store named rollout tensors and asset-balanced PPO experience.'

from __future__ import annotations

from typing import Any

import torch
from rl_games.common.experience import ExperienceBuffer


class EnvMajorExperienceBuffer(ExperienceBuffer):
    'Contract for env major experience buffer; shapes [H,N].'

    def _create_tensor_from_space(self, space: Any, base_shape: tuple[int, ...]) -> Any:
        'Create tensor from space.'
        if type(space).__name__ == "Dict":
            return {key: self._create_tensor_from_space(value, base_shape) for key, value in space.spaces.items()}
        if len(base_shape) != 2:
            raise ValueError("env-major rollout storage requires a [horizon,environments] prefix")
        horizon, environments = base_shape
        backing = super()._create_tensor_from_space(space, (environments, horizon))
        assert isinstance(backing, torch.Tensor)
        return backing.transpose(0, 1)

    def side_channel(self, width: int) -> torch.Tensor:
        'Handle side channel.'
        horizon, environments = self.obs_base_shape
        return torch.zeros(environments, horizon, width, dtype=torch.float32, device=self.device).transpose(0, 1)
