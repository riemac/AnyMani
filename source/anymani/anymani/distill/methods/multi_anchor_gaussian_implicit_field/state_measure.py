r"""关节构型测度：完整 joint-limit 超矩形上的独立 scrambled Sobol。

$$
q\sim\operatorname{Sobol}\left(\prod_{i=1}^{N_J}[q_i^{\min},q_i^{\max}]\right).
$$

该测度包含自碰撞构型。limits 只定义采样域，不进入 encoder。每资产拥有独立 engine 与
cursor，resume 后下一个 $q$ 必须逐元素一致。
"""

from __future__ import annotations

import torch

from anymani.distill.representations.sources.kinematics import EmbodimentGeometrySpec


class SobolJointSampler:
    r"""在每项资产完整 joint-limit 超矩形中连续产生 scrambled Sobol $q$。

    对单位 Sobol 样本 $u\in[0,1]^{N_J}$ 使用 $q_i=l_i+u_i(q_i^{\max}-l_i)$，单位 rad。
    """

    def __init__(self, spec: EmbodimentGeometrySpec, *, seed: int) -> None:
        r"""保存 CPU rad limits 并初始化 $N_J$ 维独立 scrambled SobolEngine。"""

        if spec.joint_limits is None:
            raise ValueError("EmbodimentGeometrySpec must contain joint_limits for q sampling")
        self.limits = spec.joint_limits.detach().cpu().to(torch.float64)  # `[N_J,2]`，rad
        self.seed = int(seed)  # engine 重建与 checkpoint resume 的 deterministic identity
        self.cursor = 0  # 已消费的 Sobol q 数；不是 optimizer step
        self.engine = torch.quasirandom.SobolEngine(
            dimension=self.limits.shape[0],  # $N_J$ 随资产变化
            scramble=True,  # Owen scrambling 提供 seed 可复现随机化
            seed=self.seed,
        )

    def draw(
        self,
        count: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        r"""返回 `[count,N_J]` 合法 $q$，完整域包含 self-collision 构型。"""

        if count < 1:
            raise ValueError("Sobol draw count must be positive")
        unit = self.engine.draw(count, dtype=torch.float64)  # $[0,1]^{N_J}$
        q = self.limits[:, 0] + unit * (self.limits[:, 1] - self.limits[:, 0])  # rad
        self.cursor += int(count)
        return q.to(device=device, dtype=dtype)

    def state_dict(self) -> dict[str, int]:
        r"""返回可写入 checkpoint 的低差异序列状态。"""

        return {"seed": self.seed, "cursor": self.cursor, "dimension": int(self.limits.shape[0])}

    def load_state_dict(self, state: dict[str, int]) -> None:
        r"""从 seed+cursor 重建 Sobol engine，确保 resume 后下一个 $q$ 完全一致。"""

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
