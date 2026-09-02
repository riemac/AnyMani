r"""ManagerBased nested observation到one-level RL storage dictionaries的可逆transport。

Task仍拥有named scientific terms。Transport只移除顶层``policy/critic`` group，不flatten term axes。
``prototype_index``是LongTensor routing side-channel：随rollout/minibatch切片，但不进入normalization或模型投影。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StructuredRlTransport:
    r"""Actor/critic one-level term dictionaries与opaque geometry routing。"""

    policy_terms: dict[str, torch.Tensor]
    critic_terms: dict[str, torch.Tensor]
    prototype_index: torch.Tensor  # long$[B]$ selection-local rows

    def __post_init__(self) -> None:
        r"""验证batch/device和Long routing，不解释prototype数值。"""

        if self.prototype_index.ndim != 1 or self.prototype_index.dtype != torch.long:
            raise ValueError("prototype_index must be long [B]")
        batch = self.prototype_index.shape[0]
        leaves = (*self.policy_terms.values(), *self.critic_terms.values())
        if not leaves or any(value.shape[0] != batch for value in leaves):
            raise ValueError("all structured transport leaves must share batch axis")
        if len({value.device for value in (*leaves, self.prototype_index)}) != 1:
            raise ValueError("structured transport leaves/routing must share device")

    @classmethod
    def from_nested_observation(
        cls,
        observation: Mapping[str, object],
        prototype_index: torch.Tensor,
        *,
        floating_clip: float | None = None,
    ) -> StructuredRlTransport:
        r"""解开顶层policy/critic groups并可选裁剪floating leaves。"""

        policy = observation.get("policy")
        critic = observation.get("critic")
        if not isinstance(policy, Mapping) or not isinstance(critic, Mapping):
            raise ValueError("nested observation must contain policy/critic mappings")

        def prepare(group: Mapping[object, object], name: str) -> dict[str, torch.Tensor]:
            r"""验证一层tensor leaves，保持原rank。"""

            prepared: dict[str, torch.Tensor] = {}
            for raw_key, raw_value in group.items():
                if not isinstance(raw_key, str) or not isinstance(raw_value, torch.Tensor):
                    raise TypeError(f"{name} group must contain string->Tensor leaves")
                value = raw_value
                if floating_clip is not None and torch.is_floating_point(value):
                    value = torch.clamp(value, min=-floating_clip, max=floating_clip)
                prepared[raw_key] = value
            return prepared

        return cls(
            policy_terms=prepare(policy, "policy"),
            critic_terms=prepare(critic, "critic"),
            prototype_index=prototype_index,
        )

    def policy_storage(self) -> dict[str, torch.Tensor]:
        r"""返回one-level rollout actor dict，并加入opaque routing leaf。"""

        return {**self.policy_terms, "prototype_index": self.prototype_index}

    def critic_storage(self) -> dict[str, torch.Tensor]:
        r"""返回one-level rollout critic dict，并加入同一routing leaf。"""

        return {**self.critic_terms, "prototype_index": self.prototype_index}

    def select(self, indices: torch.Tensor) -> StructuredRlTransport:
        r"""按rollout/minibatch indices一致切片全部terms与routing。"""

        if indices.ndim != 1 or indices.dtype != torch.long:
            raise ValueError("transport selection indices must be long [K]")
        return StructuredRlTransport(
            policy_terms={name: value[indices] for name, value in self.policy_terms.items()},
            critic_terms={name: value[indices] for name, value in self.critic_terms.items()},
            prototype_index=self.prototype_index[indices],
        )


__all__ = ["StructuredRlTransport"]
