r"""Sampling contract for cache-driven reset in `gm`.

本文件固定 reset 语义：cache reset 与普通 object pose DR 是互斥路径。若环境
选择 cache reset，则 object 初始位姿来自 validated cache entry 中的
$T^h_o$；普通 `reset_root_state_uniform` 只属于 no-cache baseline / ablation。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schema import GraspCacheKey
from .store import GraspCacheStore


@dataclass(frozen=True)
class GraspCacheResetRequest:
    r"""Describe one vectorized reset request served by a cache shard.

    Args:
        key (GraspCacheKey): 目标 cache shard；第一版同一 request 只服务同一 asset/object/scale。
        num_envs (int): 需要重置的 env 数量；采样时得到 $B=\texttt{num_envs}$ 条 entry。
        allow_tiny_online_jitter (bool): 是否允许未来加入极小在线扰动；第一版默认关闭。
    """

    key: GraspCacheKey  # 当前 reset batch 对应的 cache 主键
    num_envs: int  # reset batch size $B$，通常等于 `len(env_ids)`
    allow_tiny_online_jitter: bool = False  # 第一版不启用在线扰动，避免破坏稳定接触


@dataclass(frozen=True)
class GraspCacheSampleBatch:
    r"""Hold sampled cache entries before conversion to Isaac Lab root states.

    该 dataclass 先使用 `Any` 占位 tensor 类型，避免 scaffold 阶段强行导入
    `torch` 或 Isaac Lab。实现时应把字段约束为 torch tensor，并明确形状：
    `joint_pos: [B, dof]`，`object_pose_h: [B, 7]` 或 `[B, 4, 4]`。

    Args:
        joint_pos (Any): hand joint position batch，单位 rad，形状 `[B, dof]`。
        object_pose_h (Any): object pose relative to `{h}`，语义 $T^h_o$。
        sample_indices (Any): cache row indices，便于 debug 和失败样本回溯。
    """

    joint_pos: Any  # `[B, dof]`，hand 关节初始位置，单位 rad
    object_pose_h: Any  # `[B, 7]` 或 `[B, 4, 4]`，object 相对 hand semantic frame 的位姿
    sample_indices: Any  # `[B]`，被采样 cache row index，用于回放和排错


class GraspCacheSampler:
    r"""Sample validated grasp states for cache-driven reset.

    采样器的职责只是从已验证分布
    $\mathcal{D}_{\text{grasp}}(q,T^h_o\mid a,o,s,\rho)$ 中取样；它不做在线
    grasp synthesis，也不在 reset 后追加强 pose disturbance。若需要无 cache
    baseline，应由 env cfg 选择普通 random reset event，而不是让本 sampler
    退化成随机 object pose DR。
    """

    def __init__(self, store: GraspCacheStore) -> None:
        r"""Create a sampler backed by a cache store.

        Args:
            store (GraspCacheStore): cache shard 定位与加载接口。
        """

        # 只保存 store 引用；lazy load / device placement 交给后续实现阶段决定。
        self.store = store  # cache artifact store，负责按 key 定位 shard

    def sample(self, request: GraspCacheResetRequest) -> GraspCacheSampleBatch:
        r"""Sample a batch of cache entries for vectorized reset.

        Args:
            request (GraspCacheResetRequest): 指定 cache key、batch size 与扰动策略。

        Returns:
            GraspCacheSampleBatch: sampled `joint_pos`、`object_pose_h` 与 row indices。

        Raises:
            NotImplementedError: 当前阶段只落契约，不实现随机采样和 device 管理。
        """

        # TODO: 实现时应支持按 env_ids 采样，并与 Isaac Lab reset buffer/device 对齐。
        raise NotImplementedError("grasp cache sampling is a contract scaffold.")
