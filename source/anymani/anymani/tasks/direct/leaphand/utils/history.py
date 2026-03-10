from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.utils.buffers import CircularBuffer


class ObservationHistoryBuffer:
    """Observation history helper built on Isaac Lab's CircularBuffer."""

    def __init__(self, num_envs: int, frame_dim: int, history_length: int, device: str, extra_length: int = 0):
        self._history_length = history_length
        self._buffer = CircularBuffer(max_len=history_length + extra_length, batch_size=num_envs, device=device)

    def append(self, frame: torch.Tensor):
        self._buffer.append(frame)

    def reset(self, env_ids: Sequence[int] | None = None):
        self._buffer.reset(env_ids)

    def get(self) -> torch.Tensor:
        history = self._buffer.buffer.flip(1)[:, : self._history_length]
        return history.reshape(history.shape[0], -1).clone()

    def get_with_latency(self, latencies: torch.Tensor, num_steps: int) -> torch.Tensor:
        history = self._buffer.buffer.flip(1)
        delayed_frames = []
        for i in range(num_steps):
            delayed_frames.append(
                history.gather(1, (latencies + i).unsqueeze(1).expand(-1, 1, history.shape[2])).squeeze(1)
            )
        return torch.stack(delayed_frames, dim=1).reshape(history.shape[0], -1).clone()


class ActionLatencyBuffer:
    """Action history helper using CircularBuffer while preserving zero-fill after env reset."""

    def __init__(self, num_envs: int, action_dim: int, max_latency: int, device: str):
        self._buffer = CircularBuffer(max_len=max_latency + 1, batch_size=num_envs, device=device)
        self._just_reset = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._action_dim = action_dim

    def reset(self, env_ids: Sequence[int] | None = None):
        self._buffer.reset(env_ids)
        if env_ids is None:
            self._just_reset[:] = True
        else:
            self._just_reset[env_ids] = True

    def append_and_get(self, actions: torch.Tensor, latencies: torch.Tensor) -> torch.Tensor:
        self._buffer.append(actions)
        if torch.any(self._just_reset):
            reset_ids = self._just_reset.nonzero(as_tuple=False).squeeze(-1)
            self._buffer._buffer[:, reset_ids] = 0.0
            self._buffer._buffer[self._buffer._pointer, reset_ids] = actions[reset_ids]
            self._just_reset[reset_ids] = False

        history = self._buffer.buffer.flip(1)
        return history.gather(1, latencies.unsqueeze(1).expand(-1, 1, self._action_dim)).squeeze(1)
