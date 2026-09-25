'Record task and training evidence supplied at rollout boundaries.'

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from .episode_evidence import write_episode_evidence, write_first_window_evidence
from .first_window import FirstWindowStatistics


class TrainingEvidence:
    'Contract for training evidence.'

    def __init__(
        self,
        root: Path,
        identity_digest: str,
        asset_index: torch.Tensor,
        reward_names: Sequence[str],
        *,
        policy_dt_s: float = 0.05,
        flush_rows: int = 4096,
        first_window_root: Path | None = None,
    ) -> None:
        'Accumulate per-environment rewards and first-window facts until rollout flush.'

        if asset_index.ndim != 1 or asset_index.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("training evidence asset indices must be a one-dimensional integer tensor")  # shapes [N]
        if asset_index.numel() == 0 or bool((asset_index < 0).any()):
            raise ValueError("training evidence requires nonempty nonnegative asset indices")
        self.root = Path(root)
        self.identity_digest = identity_digest
        self.segment_id = uuid.uuid4().hex
        self.asset_index = asset_index.detach().long().clone()  # shapes [N]
        self.reward_names = tuple(reward_names)
        self.policy_dt_s = policy_dt_s
        self.flush_rows = flush_rows
        self.policy_version = 0
        self._next_shard = 0
        self._pending: list[dict[str, np.ndarray]] = []
        self._gpu_rows: list[dict[str, torch.Tensor]] = []
        self._last_snapshot: dict[str, torch.Tensor] | None = None
        n = asset_index.numel()
        self._episode_id = torch.zeros(n, dtype=torch.long, device=asset_index.device)
        self._episode_start = torch.zeros_like(self._episode_id)
        self._steps = torch.zeros_like(self._episode_id)
        self._reward_sums = torch.zeros(n, len(reward_names), device=asset_index.device)  # shapes [N,R]
        self._asset_reward_sums = torch.zeros(int(asset_index.max()) + 1, len(reward_names), device=asset_index.device)  # shapes [A,R]
        self._asset_samples = torch.zeros(self._asset_reward_sums.shape[0], device=asset_index.device)  # shapes [A]


        self.first_window_root = Path(first_window_root) if first_window_root is not None else None
        if self.first_window_root is not None and self.first_window_root.resolve() == self.root.resolve():
            raise ValueError("first-window and episode evidence need separate directories")
        self.first30_statistics = FirstWindowStatistics(self._asset_samples.numel()) if first_window_root is not None else None
        self._next_window_shard = 0
        self._gpu_window_rows: list[dict[str, torch.Tensor]] = []
        self._pending_windows: list[dict[str, np.ndarray]] = []
        self._first30_prefix: dict[str, torch.Tensor] = {}
        if self.first30_statistics is not None:
            for name in ("net_turns_first30", "absolute_path_turns_first30"):
                self._first30_prefix[name] = torch.zeros(n, device=asset_index.device)
            self._first30_prefix["goal_count_first30"] = torch.zeros_like(self._episode_id)
            self._first30_prefix["first30_complete"] = torch.zeros(n, dtype=torch.bool, device=asset_index.device)  # shapes [N]
            self._first30_prefix["first30_safe"] = torch.zeros(n, dtype=torch.bool, device=asset_index.device)  # shapes [N]

    def capture(self, snapshot: Mapping[str, torch.Tensor], weighted_step_rewards: torch.Tensor) -> None:
        'Handle capture; shapes [N,R].'

        if weighted_step_rewards.shape != self._reward_sums.shape:  # shapes [N,R]
            raise ValueError("reward evidence matrix must match [environments, reward terms]")
        first = self._steps == 0  # shapes [N]
        self._episode_start[first] = self.policy_version
        self._steps.add_(1)
        self._reward_sums.add_(weighted_step_rewards.detach())  # Weighted reward per policy step.
        self._asset_reward_sums.index_add_(0, self.asset_index, weighted_step_rewards.detach())  # shapes [A,R]
        self._asset_samples.index_add_(0, self.asset_index, torch.ones_like(self.asset_index, dtype=torch.float32))  # shapes [A]
        if self.first30_statistics is not None:
            self._capture_first_window(snapshot)
        terminal = (  # shapes [N]
            snapshot["termination_object_out_of_anchor"]
            | snapshot["termination_goal_axis_misaligned"]
            | snapshot["termination_time_out"]
        )
        ids = terminal.nonzero(as_tuple=False).reshape(-1)  # shapes [E]
        self._gpu_rows.append(self._rows(snapshot, ids, censored=False))
        self._steps[ids] = 0
        self._reward_sums[ids] = 0
        for value in self._first30_prefix.values():
            value[ids] = 0
        self._episode_id[ids] += 1
        self._last_snapshot = dict(snapshot)

    def _rows(
        self, snapshot: Mapping[str, torch.Tensor], ids: torch.Tensor, *, censored: bool
    ) -> dict[str, torch.Tensor]:
        'Handle rows.'
        duration = snapshot["episode_duration_s"][ids]  # shapes [E]
        result = {  # shapes [E]
            "env_id": ids,
            "episode_id": self._episode_id[ids],
            "asset_index": self.asset_index[ids],
            "policy_version_start": self._episode_start[ids],
            "policy_version_end": torch.full_like(ids, self.policy_version),
            "policy_steps": torch.round(duration / self.policy_dt_s).long(),
            "duration_s": duration,
            "net_turns": snapshot["net_rotation_rad"][ids] / (2 * torch.pi),
            "absolute_path_turns": snapshot["absolute_path_rotation_rad"][ids] / (2 * torch.pi),
            "max_positive_net_turns": snapshot["max_positive_net_rotation_rad"][ids] / (2 * torch.pi),
            "goal_count": snapshot["completed_subgoals"][ids].long(),
            "frontier_count": snapshot["rotation_frontier_count"][ids].long(),
            "termination_drop": snapshot["termination_object_out_of_anchor"][ids].bool(),
            "termination_axis": snapshot["termination_goal_axis_misaligned"][ids].bool(),
            "termination_timeout": snapshot["termination_time_out"][ids].bool(),
            "censored": torch.full_like(ids, censored, dtype=torch.bool),
        }
        result.update({f"reward/{name}": self._reward_sums[ids, i] for i, name in enumerate(self.reward_names)})  # shapes [E]
        if "completed_orientation_subgoals" in snapshot:
            result["orientation_goal_count"] = snapshot["completed_orientation_subgoals"][ids].long()  # shapes [E]
        for name in ("adr_position_level", "adr_position_offset_x_h_m", "adr_position_offset_y_h_m", "net_turns_first30", "first30_complete"):
            if name in snapshot:
                result[name] = snapshot[name][ids]
        result.update({name: value[ids] for name, value in self._first30_prefix.items()})
        return {key: value.detach().clone() for key, value in result.items()}

    def _capture_first_window(self, snapshot: Mapping[str, torch.Tensor]) -> None:
        'Handle capture first window.'
        previous_complete = self._first30_prefix["first30_complete"]  # shapes [N]
        fresh = ~previous_complete
        duration = snapshot["episode_duration_s"]
        complete = duration >= 30.0  # units Hz
        failure = snapshot["termination_object_out_of_anchor"] | snapshot["termination_goal_axis_misaligned"]  # shapes [N]
        values = {
            "net_turns_first30": snapshot["net_rotation_rad"] / (2 * torch.pi),
            "absolute_path_turns_first30": snapshot["absolute_path_rotation_rad"] / (2 * torch.pi),
            "goal_count_first30": snapshot["completed_subgoals"].long(),
            "first30_complete": complete,  # shapes [N]
            "first30_safe": complete & ~failure,
        }
        for name, current in values.items():
            self._first30_prefix[name] = torch.where(fresh, current, self._first30_prefix[name]).detach()
        ids = (fresh & (complete | failure)).nonzero(as_tuple=False).reshape(-1)
        result = {  # shapes [E]
            "env_id": ids, "episode_id": self._episode_id[ids], "asset_index": self.asset_index[ids],
            "policy_version_start": self._episode_start[ids],
            "policy_version_end": torch.full_like(ids, self.policy_version),
            "policy_steps": torch.round(duration[ids] / self.policy_dt_s).long(),  # values 20Hz; units Hz
            "duration_s": duration[ids],
            "net_turns": self._first30_prefix["net_turns_first30"][ids],
            "absolute_path_turns": self._first30_prefix["absolute_path_turns_first30"][ids],
            "goal_count": self._first30_prefix["goal_count_first30"][ids],
            "complete": complete[ids], "safe": self._first30_prefix["first30_safe"][ids],
            "termination_drop": snapshot["termination_object_out_of_anchor"][ids],
            "termination_axis": snapshot["termination_goal_axis_misaligned"][ids],
            "termination_timeout": snapshot["termination_time_out"][ids],
        }
        self._gpu_window_rows.append({name: value.detach().clone() for name, value in result.items()})

    def _drain_first_windows(self, *, force: bool) -> None:
        'Handle drain first windows.'
        if self.first30_statistics is None:
            return
        if self._gpu_window_rows:
            columns = {name: torch.cat([row[name] for row in self._gpu_window_rows]).cpu().numpy() for name in self._gpu_window_rows[0]}  # shapes [E]
            self._gpu_window_rows.clear()
            if columns["env_id"].size:
                self.first30_statistics.add_batch(columns)
                self._pending_windows.append(columns)
        if self._pending_windows and (force or sum(row["env_id"].size for row in self._pending_windows) >= self.flush_rows):
            assert self.first_window_root is not None
            columns = {name: np.concatenate([row[name] for row in self._pending_windows]) for name in self._pending_windows[0]}
            destination = self.first_window_root / f"windows-{self.segment_id}-{self._next_window_shard:06d}.parquet"
            write_first_window_evidence(
                destination, columns, identity_digest=self.identity_digest, segment_id=self.segment_id,
                policy_dt_s=self.policy_dt_s,
            )
            self._pending_windows.clear()
            self._next_window_shard += 1

    def drain(self, *, force: bool = False) -> dict[str, torch.Tensor]:
        'Handle drain; shapes [A,terms], [A,R], [A].'
        self._drain_first_windows(force=force)
        if self._gpu_rows:
            columns = {key: torch.cat([row[key] for row in self._gpu_rows]).cpu().numpy() for key in self._gpu_rows[0]}  # shapes [E]
            self._gpu_rows.clear()
            if columns["env_id"].size:
                self._pending.append(columns)
        result = {
            "reward_terms": (self._asset_reward_sums / self._asset_samples.clamp_min(1)[:, None]).detach().cpu(),  # shapes [A,R], [A,1]
            "sample_count": self._asset_samples.detach().cpu().clone(),  # shapes [A]
        }
        self._asset_reward_sums.zero_()
        self._asset_samples.zero_()
        if self._pending and (force or sum(row["env_id"].size for row in self._pending) >= self.flush_rows):
            columns = {key: np.concatenate([row[key] for row in self._pending]) for key in self._pending[0]}
            rewards = {name: columns.pop(f"reward/{name}") for name in self.reward_names}  # shapes [E]
            destination = self.root / f"episodes-{self.segment_id}-{self._next_shard:06d}.parquet"
            write_episode_evidence(
                destination,
                columns,  # shapes [E]
                reward_sums=rewards,
                identity_digest=self.identity_digest,
                segment_id=self.segment_id,
                policy_dt_s=self.policy_dt_s,
            )
            self._pending.clear()
            self._next_shard += 1
        return result

    def close(self) -> None:
        'Close the declared contract.'

        if self._last_snapshot is not None:
            ids = (self._steps > 0).nonzero(as_tuple=False).reshape(-1)  # shapes [E]
            self._gpu_rows.append(self._rows(self._last_snapshot, ids, censored=True))
        self.drain(force=True)
        self._last_snapshot = None
