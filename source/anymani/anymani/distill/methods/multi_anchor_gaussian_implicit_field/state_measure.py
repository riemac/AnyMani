"""Deterministic joint-limit sampling for geometry pretraining."""


from __future__ import annotations

import torch

from anymani.distill.representations.sources.kinematics import EmbodimentGeometrySpec


class SobolJointSampler:


    def __init__(self, spec: EmbodimentGeometrySpec, *, seed: int) -> None:


        if spec.joint_limits is None:
            raise ValueError("EmbodimentGeometrySpec must contain joint_limits for q sampling")
        self.limits = spec.joint_limits.detach().cpu().to(torch.float64)
        self.seed = int(seed)
        self.cursor = 0
        self.engine = torch.quasirandom.SobolEngine(
            dimension=self.limits.shape[0],
            scramble=True,
            seed=self.seed,
        )

    def draw(
        self,
        count: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:


        if count < 1:
            raise ValueError("Sobol draw count must be positive")
        unit = self.engine.draw(count, dtype=torch.float64)
        q = self.limits[:, 0] + unit * (self.limits[:, 1] - self.limits[:, 0])  # rad
        self.cursor += int(count)
        return q.to(device=device, dtype=dtype)

    def state_dict(self) -> dict[str, int]:


        return {"seed": self.seed, "cursor": self.cursor, "dimension": int(self.limits.shape[0])}

    def load_state_dict(self, state: dict[str, int]) -> None:


        if int(state.get("seed", -1)) != self.seed:
            raise ValueError("Sobol checkpoint seed does not match asset sampler")
        if int(state.get("dimension", -1)) != self.limits.shape[0]:
            raise ValueError("Sobol checkpoint dimension does not match asset joint count")
        cursor = int(state.get("cursor", -1))
        if cursor < 0:
            raise ValueError("Sobol checkpoint cursor must be non-negative")
        self.engine = torch.quasirandom.SobolEngine(
            dimension=self.limits.shape[0],
            scramble=True,
            seed=self.seed,
        )
        if cursor:
            self.engine.fast_forward(cursor)
        self.cursor = cursor


__all__ = ["SobolJointSampler"]
