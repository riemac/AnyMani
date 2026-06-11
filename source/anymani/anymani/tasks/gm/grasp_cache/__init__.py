r"""Grasp cache contracts for generalized in-hand manipulation reset.

本包只固定 `gm` 任务如何“读取并消费”稳定 grasp cache；它不实现离线搜索、
Isaac Sim 批量 rollout、TRO-Grasp / IK 候选生成，也不接管 `distill` 的训练编排。

科研语义：grasp cache 是一个经验初始状态分布
$$
\mathcal{D}_{\text{grasp}}(q, T_{ho}\mid a, o, s, \rho),
$$
其中 $a$ 是 hand `asset_id`，$o$ 是 object id，$s$ 是 object scale bucket，
$\rho$ 是生成时采用的 pose distribution。在线 reset 只从这个经过验证的
分布采样，不再把 object pose 当作独立 DR 项后置扰动。
"""

from __future__ import annotations

from .sampler import GraspCacheResetRequest, GraspCacheSampleBatch, GraspCacheSampler
from .schema import GraspCacheKey, GraspCacheMetadata, GraspCacheTensorSpec
from .store import GraspCacheStore

__all__ = [
    "GraspCacheKey",
    "GraspCacheMetadata",
    "GraspCacheResetRequest",
    "GraspCacheSampleBatch",
    "GraspCacheSampler",
    "GraspCacheStore",
    "GraspCacheTensorSpec",
]
