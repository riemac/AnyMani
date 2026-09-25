(
    'Per-env initial-state ADR randomizes only object position in the palm plane. '
    'Keep the original 25-level half-width w(k)=0.01*k/25 m; max_level caps reach '
    'without rescaling (0.4 mm at level 1, 2 mm at level 5). Levels are hidden '
    'per-env state. Sample once per reset and score each env on its first 30 '
    'seconds of net turns. After 3 episodes, EMA >= 1 plus a safe latest window '
    'advances one level; EMA < 0.5 rolls back one. Scores saturate at 2 turns.'
)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .task_math import quaternion_apply_wxyz


@dataclass
class ObjectPositionAdrCfg:
    'Position amplitude spectrum and per-env schedule; length in meters, time in seconds.'

    enabled: bool = False  # Disabled components consume no random numbers and change no state.
    reference_levels: int = 25  # Keep the original amplitude-spectrum denominator independent of max_level.
    max_level: int = 5  # Expose at most the first five levels of the original spectrum this round.
    initial_level: int = 1  # Start at level 1 (0.4 mm half-width); weak envs may return to level 0.
    full_half_width_m: float = 0.01  # Original level-25 position half-width.
    reference_seconds: float = 30.0  # Match the main 1-turn/30-second window.
    promote_turns: float = 1.0  # Advance only after one full turn at the current perturbation.
    demote_turns: float = 0.5  # Conservative rollback avoids trapping weak envs at high levels.
    score_cap_turns: float = 2.0  # Scores above two turns do not accelerate level increases.
    ema_alpha: float = 0.5  # Update from each env's own episodes, without first-sample zero bias.
    min_episodes: int = 3  # Count episodes at this level per env, not global reset-hook calls.

    def __post_init__(self) -> None:
        'Reject invalid levels, units, or schedules without a hysteresis interval.'
        if not 0 <= self.initial_level <= self.max_level <= self.reference_levels or self.reference_levels < 1:
            raise ValueError('ADR levels must preserve 0 <= initial <= max <= reference')
        if not 0 < self.ema_alpha <= 1 or self.min_episodes < 1:
            raise ValueError('ADR requires a positive episode window and valid EMA alpha')
        if not 0 <= self.demote_turns < self.promote_turns <= self.score_cap_turns:
            raise ValueError('ADR turn thresholds need a nonempty hysteresis interval')
        if self.full_half_width_m < 0 or self.reference_seconds <= 0:
            raise ValueError('ADR position/time ranges are invalid')


@dataclass
class HeterogeneousAdrCfg:
    'Nested config declares only implemented position randomization; no empty component is created.'

    object_position: ObjectPositionAdrCfg = field(default_factory=ObjectPositionAdrCfg)


class PerEnvironmentPositionAdr:
    'Each env owns its level, episode EMA, and applied perturbation.'

    def __init__(self, cfg: ObjectPositionAdrCfg, num_envs: int, device: torch.device | str):
        self.cfg = cfg  # Validated component configuration.
        self.level = torch.full((num_envs,), cfg.initial_level, dtype=torch.long, device=device)
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.ema = torch.zeros(num_envs, device=device)  # Per-env net turns over the first 30 seconds.
        self.trials = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.offset_h = torch.zeros(num_envs, 3, device=device)  # Applied palm-plane offset; z stays zero.

    @property
    def half_width(self) -> torch.Tensor:
        'Compute half-width from the original 25-level spectrum, independent of max_level.'
        return self.level.float() * (self.cfg.full_half_width_m / self.cfg.reference_levels)

    def observe(self, ids: torch.Tensor, turns: torch.Tensor, safe_window: torch.Tensor) -> None:
        'Consume only completed episodes for selected envs; preserve all other curriculum state.'
        valid = self.initialized[ids]  # The first reset is not a completed episode.
        selected = ids[valid]
        if selected.numel() == 0:
            return
        score = turns[valid].clamp(0, self.cfg.score_cap_turns)  # Do not reward reverse turns or excessive upper-tail speed.
        first = self.trials[selected] == 0
        updated = torch.where(first, score, (1 - self.cfg.ema_alpha) * self.ema[selected] + self.cfg.ema_alpha * score)
        self.ema[selected] = updated
        self.trials[selected] += 1
        ready = self.trials[selected] >= self.cfg.min_episodes
        up = ready & (updated >= self.cfg.promote_turns) & safe_window[valid] & (self.level[selected] < self.cfg.max_level)
        down = ready & (updated < self.cfg.demote_turns) & (self.level[selected] > 0)
        self.level[selected] += up.long() - down.long()  # Change at most one level per env per update.
        changed = selected[up | down]
        self.trials[changed] = 0
        self.ema[changed] = 0  # Accumulate new evidence after a level change.

    def sample(self, ids: torch.Tensor) -> torch.Tensor:
        'Sample in the semantic palm xy plane; leave height and initial orientation unchanged.'
        noise = torch.rand((ids.numel(), 3), device=self.level.device) * 2 - 1
        noise[:, 2] = 0
        self.offset_h[ids] = noise * self.half_width[ids, None]
        self.initialized[ids] = True
        return self.offset_h[ids]

    def state_dict(self) -> dict[str, torch.Tensor]:
        'Save actual curriculum state for exact resume.'
        return {name: getattr(self, name).detach().clone() for name in ('level', 'initialized', 'ema', 'trials', 'offset_h')}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        "Restore by env axis so one env's difficulty is never broadcast to others."
        for name, value in state.items():
            target = getattr(self, name)
            if value.shape != target.shape:
                raise ValueError(f'ADR state shape mismatch: {name}')
            target.copy_(value.to(target.device))


def get_position_adr(env: Any) -> PerEnvironmentPositionAdr | None:
    'Instantiate enabled components only; disabled components have no runtime state.'
    cfg = getattr(env.cfg, 'adr', None)
    if cfg is None or not cfg.object_position.enabled:
        return None
    runtime = getattr(env, '_hetero_position_adr', None)
    if runtime is None:
        runtime = PerEnvironmentPositionAdr(cfg.object_position, env.num_envs, env.device)
        env._hetero_position_adr = runtime
    return runtime


def perturb_reset_position(env: Any, ids: torch.Tensor, position_w: torch.Tensor, hand_quat_w: torch.Tensor) -> torch.Tensor:
    'Apply the episode offset after baseline pregrasp and before physics writes and anchor capture.'
    runtime = get_position_adr(env)
    if runtime is None:
        return position_w
    if bool(runtime.initialized[ids].any()):
        command = env.command_manager.get_term('goal_pose')
        duration = env.episode_length_buf[ids].float() * float(env.step_dt)
        complete = command.reference_window_complete[ids]
        turns = torch.where(complete, command.reference_window_net_turns[ids], command.net_rotation_turns[ids])
        failed = env.termination_manager.get_term('object_out_of_anchor')[ids] | env.termination_manager.get_term('goal_axis_misaligned')[ids]
        safe_window = complete & ((duration > runtime.cfg.reference_seconds + 1e-4) | ~failed)
        runtime.observe(ids, turns, safe_window)
    offset = runtime.sample(ids)  # Leave unrelated envs unchanged.
    return position_w + quaternion_apply_wxyz(hand_quat_w, offset)  # Transform semantic palm-frame offsets to world coordinates.
