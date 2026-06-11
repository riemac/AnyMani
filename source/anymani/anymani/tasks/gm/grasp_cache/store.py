r"""Storage contract for `gm` grasp cache artifacts.

本文件只描述“cache 应如何被定位、加载、核对”，不决定离线生成算法，也不
实际绑定某一种磁盘格式。后续可以选择 `.pt`、HDF5、Zarr 或 npz；只要能恢复
`GraspCacheMetadata` 与 `GraspCacheTensorSpec`，在线 reset 语义就不变。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import GraspCacheKey, GraspCacheMetadata


@dataclass(frozen=True)
class GraspCacheStore:
    r"""Locate grasp-cache shards under a self-contained root.

    短期约定 cache 产物可以放在 `tasks/gm/grasp_cache/artifacts/` 下，满足用户
    当前“先自包含”的要求。长期应迁移到统一 artifact root；因此代码契约只依赖
    `root`，不要在训练配置里硬编码当前临时目录。

    Args:
        root (Path): cache artifact 根目录，下面按 `GraspCacheKey.as_posix_path()` 分层。
        shard_suffix (str): 单个 shard 的文件后缀；第一版建议 `.pt`，但本骨架不强制。
    """

    root: Path  # artifact 根目录；短期可为 `.../tasks/gm/grasp_cache/artifacts`
    shard_suffix: str = ".pt"  # 建议默认 `.pt`，便于保存 torch tensor 与 metadata

    def shard_path(self, key: GraspCacheKey) -> Path:
        r"""Return the expected path for one cache shard.

        Args:
            key (GraspCacheKey): asset/object/scale/pose-distribution 主键。

        Returns:
            Path: 由 `root/key.as_posix_path()/cache{suffix}` 组成的确定性路径。
        """

        # 使用 key 的层级路径而非 hash，优先服务科研核对和人工排错。
        return self.root / key.as_posix_path() / f"cache{self.shard_suffix}"

    def metadata_path(self, key: GraspCacheKey) -> Path:
        r"""Return the expected human-readable metadata path for one shard.

        Args:
            key (GraspCacheKey): asset/object/scale/pose-distribution 主键。

        Returns:
            Path: `metadata.json` 的确定性路径，后续用于人工审查和 manifest 索引。
        """

        # metadata 与 tensor shard 同目录，保证复制单个 shard 时不会丢失实验语义。
        return self.root / key.as_posix_path() / "metadata.json"

    def load_metadata(self, key: GraspCacheKey) -> GraspCacheMetadata:
        r"""Load and validate metadata for one cache shard.

        Args:
            key (GraspCacheKey): 待加载 shard 的主键。

        Returns:
            GraspCacheMetadata: 经过 schema 解析后的 metadata。

        Raises:
            NotImplementedError: 当前阶段只落契约，不实现 JSON 解析。
        """

        # TODO: 实现时应检查 metadata 内部的 key 与路径 key 完全一致，避免错配 cache。
        raise NotImplementedError("grasp cache metadata loading is a contract scaffold.")

    def load_tensors(self, key: GraspCacheKey) -> dict[str, Any]:
        r"""Load raw tensors for one cache shard.

        Args:
            key (GraspCacheKey): 待加载 shard 的主键。

        Returns:
            dict[str, Any]: 至少包含 `joint_pos` 与 `object_pose_h` 的 tensor 字典。

        Raises:
            NotImplementedError: 当前阶段只落契约，不绑定 torch / h5py / zarr 实现。
        """

        # TODO: 实现时必须用 metadata 的 `GraspCacheTensorSpec` 校验 tensor 形状与 frame。
        raise NotImplementedError("grasp cache tensor loading is a contract scaffold.")
