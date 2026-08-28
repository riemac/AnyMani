r"""异构 RL 基础设施阶段的确定性冻结 $Z$ provider。

真实 SSL stage 尚未交付 retained artifact 时，下游仍需要验证固定接口、资产路由、策略网络、rollout、
PPO backward/update 与 checkpoint。本模块为每个 `(asset, owner)` 生成一个稳定的 128 维接口替身：

* 输入 identity 只来自 dataset/manifest/physical hash、owner index、算法版本与 seed；
* 有效 owner 向量做零均值、单位均方归一化，数值尺度接近 final LayerNorm token；
* ghost owner 精确为零；mask 与 graph 必须由 canonical manifest 交付，provider 不推断物理拓扑；
* 全部 table 注册为 buffer，不产生 trainable parameter，也不接收梯度。

该 $Z$ 只能证明 RL infrastructure 消费接口正确，不能解释为学习到的几何表征。后续真实 SSL artifact
只替换 provider，``EmbodimentPolicyInput.geometry_entities`` 与 PPO 合同保持不变。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from anymani.distill.models.policy import CANONICAL_JOINT_COUNT, CANONICAL_OWNER_COUNT

if TYPE_CHECKING:
    from anymani.assets.canonical_runtime import CanonicalHandArtifact

FROZEN_Z_ALGORITHM_VERSION = "sha256-counter-zero-mean-unit-rms-v1"
"""跨 PyTorch/NumPy 版本稳定的 counter-hash 向量算法。"""


def _stable_digest(payload: Any) -> str:
    r"""对 JSON-compatible identity 计算 canonical SHA-256。"""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_hash_vector(identity: str, width: int) -> torch.Tensor:
    r"""由 SHA-256 counter stream 生成零均值、单位均方的 float32 向量。

    Args:
        identity (str): 当前 asset/owner 的完整稳定身份。
        width (int): 输出特征宽度 $D$，当前正式接口为 128。

    Returns:
        torch.Tensor: `[D]` float32 向量，均值约 0，RMS 约 1。
    """

    if width < 2:
        raise ValueError("frozen Z width must be at least two")
    values: list[float] = []  # Python 整数/浮点路径避免 torch RNG 跨版本变化
    counter = 0
    while len(values) < width:
        block = hashlib.sha256(f"{identity}:{counter}".encode()).digest()  # 32 deterministic bytes
        for offset in range(0, len(block), 4):
            integer = int.from_bytes(block[offset : offset + 4], byteorder="big", signed=False)
            values.append((integer + 0.5) / float(2**32) * 2.0 - 1.0)  # uint32 -> open-ish $(-1,1)$
            if len(values) == width:
                break
        counter += 1
    vector = torch.tensor(values, dtype=torch.float64)  # 先用 float64 完成归一化，减少平台舍入差异
    vector = vector - vector.mean()  # 每个 owner token 精确去均值
    rms = torch.sqrt(torch.mean(vector.square()))  # $\sqrt{D^{-1}\sum_i z_i^2}$
    if not torch.isfinite(rms) or float(rms) <= 0.0:
        raise RuntimeError("deterministic frozen Z hash produced an invalid RMS")
    return (vector / rms).to(dtype=torch.float32)  # final token dtype/尺度 `[D]`


@dataclass(frozen=True)
class FrozenZBatch:
    r"""按 runtime asset row gather 后的策略静态输入。"""

    geometry_entities: torch.Tensor
    """冻结 $Z$，形状 `[B,21,D]`。"""

    owner_valid_mask: torch.Tensor
    """PALM/JOINT/TIP owner mask，bool `[B,21]`。"""

    joint_valid_mask: torch.Tensor
    """真实 active joint mask，bool `[B,16]`。"""

    shortest_path: torch.Tensor
    """真实 owner graph shortest-path buckets，long `[B,21,21]`。"""

    parent_direction: torch.Tensor
    """真实 owner graph parent-direction buckets，long `[B,21,21]`。"""

    child_direction: torch.Tensor
    """真实 owner graph child-direction buckets，long `[B,21,21]`。"""


class FrozenZProvider(nn.Module):
    r"""只读 `[asset,owner]` $Z$/mask/graph lookup table。

    provider 是策略 network 的子模块，因此 buffers 会随 actor checkpoint 保存。``identity`` 同时写入
    run config/checkpoint metadata，restore 时用于在加载 state dict 前核对 dataset 和 manifest。
    """

    z_table: torch.Tensor
    owner_valid_mask: torch.Tensor
    joint_valid_mask: torch.Tensor
    shortest_path: torch.Tensor
    parent_direction: torch.Tensor
    child_direction: torch.Tensor
    _identity_payload: dict[str, Any]

    def __init__(
        self,
        *,
        asset_ids: tuple[str, ...],
        physical_geometry_hashes: tuple[str, ...],
        owner_valid_mask: torch.Tensor,
        joint_valid_mask: torch.Tensor,
        shortest_path: torch.Tensor,
        parent_direction: torch.Tensor,
        child_direction: torch.Tensor,
        dataset_digest: str,
        manifest_digest: str,
        seed: int = 0,
        width: int = 128,
        algorithm_version: str = FROZEN_Z_ALGORITHM_VERSION,
    ) -> None:
        r"""构造确定性冻结 bank，并验证 manifest 交付的所有 row/graph 轴。

        Args:
            asset_ids: 有序资产 ID，长度 $A$。
            physical_geometry_hashes: 与资产 row 对齐的真实几何 hash。
            owner_valid_mask: bool `[A,21]`。
            joint_valid_mask: bool `[A,16]`。
            shortest_path: long `[A,21,21]`。
            parent_direction: long `[A,21,21]`。
            child_direction: long `[A,21,21]`。
            dataset_digest: 原始 `ppo.yaml` bytes identity。
            manifest_digest: canonical runtime manifest identity。
            seed: 接口替身 seed；不复用 PPO sampling seed。
            width: $Z$ 特征宽度，当前正式值为 128。
            algorithm_version: hash-to-vector 公式版本。
        """

        super().__init__()
        asset_count = len(asset_ids)  # canonical asset row 数 $A$
        if asset_count < 1 or len(physical_geometry_hashes) != asset_count:
            raise ValueError("asset IDs and physical hashes must be non-empty and row-aligned")
        if len(set(asset_ids)) != asset_count:
            raise ValueError("frozen Z asset IDs must be unique")
        expected_owner = (asset_count, CANONICAL_OWNER_COUNT)
        expected_joint = (asset_count, CANONICAL_JOINT_COUNT)
        expected_graph = (asset_count, CANONICAL_OWNER_COUNT, CANONICAL_OWNER_COUNT)
        if owner_valid_mask.shape != expected_owner or owner_valid_mask.dtype != torch.bool:
            raise ValueError(f"owner_valid_mask must be bool {expected_owner}")
        if joint_valid_mask.shape != expected_joint or joint_valid_mask.dtype != torch.bool:
            raise ValueError(f"joint_valid_mask must be bool {expected_joint}")
        for name, graph in (
            ("shortest_path", shortest_path),
            ("parent_direction", parent_direction),
            ("child_direction", child_direction),
        ):
            if graph.shape != expected_graph or graph.dtype not in {torch.int32, torch.int64, torch.long}:
                raise ValueError(f"{name} must be integer {expected_graph}")
        if not torch.all(owner_valid_mask[:, 0]):
            raise ValueError("PALM owner must be valid for every asset")
        if not torch.equal(owner_valid_mask[:, 1 : 1 + CANONICAL_JOINT_COUNT], joint_valid_mask):
            raise ValueError("JOINT owner mask must exactly equal active joint mask")
        if seed < 0 or width < 2 or not dataset_digest or not manifest_digest or not algorithm_version:
            raise ValueError("frozen Z identity fields, non-negative seed and width>=2 are required")

        self.asset_ids = tuple(asset_ids)  # host provenance，不作为 tensor checkpoint payload
        self.physical_geometry_hashes = tuple(physical_geometry_hashes)
        self.dataset_digest = str(dataset_digest)
        self.manifest_digest = str(manifest_digest)
        self.seed = int(seed)
        self.width = int(width)
        self.algorithm_version = str(algorithm_version)

        # 每个有效 owner 独立生成向量；identity 显式包含 physical hash 与 owner index。
        z_table = torch.zeros(asset_count, CANONICAL_OWNER_COUNT, width, dtype=torch.float32)  # `[A,21,D]`
        for asset_row, physical_hash in enumerate(self.physical_geometry_hashes):
            for owner_index in range(CANONICAL_OWNER_COUNT):
                if bool(owner_valid_mask[asset_row, owner_index]):
                    owner_identity = _stable_digest(
                        {
                            "algorithm_version": self.algorithm_version,
                            "dataset_digest": self.dataset_digest,
                            "manifest_digest": self.manifest_digest,
                            "physical_geometry_hash": physical_hash,
                            "owner_index": owner_index,
                            "seed": self.seed,
                        }
                    )
                    z_table[asset_row, owner_index] = _normalized_hash_vector(owner_identity, width)

        # 在 CPU materialization 阶段只计算一次 table digest；checkpoint 周期不得触发 GPU→CPU copy。
        table_sha256 = hashlib.sha256(z_table.contiguous().numpy().tobytes()).hexdigest()  # frozen table bytes identity
        identity_payload = {
            "algorithm_version": self.algorithm_version,
            "seed": self.seed,
            "width": self.width,
            "dataset_digest": self.dataset_digest,
            "manifest_digest": self.manifest_digest,
            "asset_ids": list(self.asset_ids),
            "physical_geometry_hashes": list(self.physical_geometry_hashes),
            "z_table_sha256": table_sha256,
        }
        self._identity_payload = {**identity_payload, "identity_digest": _stable_digest(identity_payload)}

        self.register_buffer("z_table", z_table, persistent=True)  # actor checkpoint 保存冻结接口输入
        self.register_buffer("owner_valid_mask", owner_valid_mask.clone(), persistent=True)
        self.register_buffer("joint_valid_mask", joint_valid_mask.clone(), persistent=True)
        self.register_buffer("shortest_path", shortest_path.to(dtype=torch.long).clone(), persistent=True)
        self.register_buffer("parent_direction", parent_direction.to(dtype=torch.long).clone(), persistent=True)
        self.register_buffer("child_direction", child_direction.to(dtype=torch.long).clone(), persistent=True)

    @property
    def identity(self) -> dict[str, Any]:
        r"""返回 checkpoint/run metadata 使用的冻结 $Z$ 身份，不同步 device-resident table。"""

        # 返回新的 list，避免外部 YAML/checkpoint 处理器改写 provider 内部有序资产身份。
        return {
            **self._identity_payload,
            "asset_ids": list(self.asset_ids),
            "physical_geometry_hashes": list(self.physical_geometry_hashes),
        }

    def resolve(self, asset_row: torch.Tensor) -> FrozenZBatch:
        r"""按 `[B]` runtime asset row gather冻结 $Z$、mask 与真实 graph。

        Args:
            asset_row (torch.Tensor): integer `[B]`，由环境 observation/routing 交付。

        Returns:
            FrozenZBatch: 与 policy batch 对齐的只读 tensors。
        """

        if asset_row.ndim != 1 or asset_row.dtype not in {torch.int32, torch.int64, torch.long}:
            raise ValueError("asset_row must be a rank-1 integer tensor")
        if asset_row.device != self.z_table.device:
            asset_row = asset_row.to(device=self.z_table.device)  # provider buffers 与 gather index 同 device
        if torch.any(asset_row < 0) or torch.any(asset_row >= self.z_table.shape[0]):
            raise IndexError("asset_row contains a row outside the frozen Z bank")
        return FrozenZBatch(
            geometry_entities=self.z_table[asset_row].detach(),
            owner_valid_mask=self.owner_valid_mask[asset_row],
            joint_valid_mask=self.joint_valid_mask[asset_row],
            shortest_path=self.shortest_path[asset_row],
            parent_direction=self.parent_direction[asset_row],
            child_direction=self.child_direction[asset_row],
        )


def build_frozen_z_provider_from_canonical_artifacts(
    artifacts: Sequence[CanonicalHandArtifact],
    *,
    dataset_digest: str,
    manifest_digest: str,
    seed: int = 0,
    width: int = 128,
    max_graph_distance: int = 8,
) -> FrozenZProvider:
    r"""从 canonical routing manifest 构造 frozen $Z$/mask/owner-graph provider。

    该入口不读取 raw mesh、StaticGeometryEvidence 或 SSL encoder。canonical v1 owner axis 固定为
    ``PALM0 + JOINT1:17 + TIP17:21``；每根手指的 active joint slots 按 proximal depth 组成链：

    $$
    PALM\to j_0\to\cdots\to j_{d-1}\to TIP.
    $$

    ghost owners invalid，图 bucket 取 ``max_graph_distance``；valid owner 的 shortest/parent/child
    距离由真实 active chain 计算。由此 provider 的图只依赖 assets manifest，不依赖 SSL 会话。

    Args:
        artifacts (Sequence[CanonicalHandArtifact]): dataset row 顺序的 canonical artifacts。
        dataset_digest (str): ``ppo.yaml`` bytes SHA-256。
        manifest_digest (str): 有序 canonical group manifest SHA-256。
        seed (int): frozen接口 seed，与 PPO sampling seed 分离。
        width (int): 每个有效 owner 的 $Z$ 宽度，正式值 128。
        max_graph_distance (int): 不可达/截断 graph bucket，需与 policy config 一致。

    Returns:
        FrozenZProvider: checkpoint-persistent `[A,21,128]` lookup bank。
    """

    rows = tuple(artifacts)  # 固化 dataset row 顺序，禁止 generator/lazy iterator 二次变化
    if not rows:
        raise ValueError("frozen Z construction requires at least one canonical artifact")
    if max_graph_distance < 1:
        raise ValueError("max_graph_distance must be positive")
    asset_count = len(rows)  # $A=2048$ formal train rows
    joint_mask = torch.tensor(
        [artifact.routing.active_joint_mask for artifact in rows],
        dtype=torch.bool,
    )  # `[A,16]`，canonical depth-major PhysX joint axis
    tip_mask = torch.tensor(
        [artifact.routing.active_tip_mask for artifact in rows],
        dtype=torch.bool,
    )  # `[A,4]`，index/middle/ring/thumb
    owner_mask = torch.zeros(asset_count, CANONICAL_OWNER_COUNT, dtype=torch.bool)  # `[A,21]`
    owner_mask[:, 0] = True  # PALM 对所有资产有效
    owner_mask[:, 1:17] = joint_mask  # JOINT owner 与 action mask 同一真源
    owner_mask[:, 17:21] = tip_mask  # TIP owner 来自 artifact routing
    graph_shape = (asset_count, CANONICAL_OWNER_COUNT, CANONICAL_OWNER_COUNT)
    shortest = torch.full(graph_shape, max_graph_distance, dtype=torch.long)
    parent = torch.full_like(shortest, max_graph_distance)

    # 每行只有 21 owners；Python BFS 的工作量约 $2048\times21^2$，发生于一次性 startup。
    for asset_row in range(asset_count):
        adjacency: list[list[int]] = [[] for _ in range(CANONICAL_OWNER_COUNT)]
        parent_index = [-1] * CANONICAL_OWNER_COUNT  # PALM root 与 invalid owners 均为 -1
        for finger_index in range(4):
            finger_joint_owners = [
                1 + depth * 4 + finger_index
                for depth in range(4)
                if bool(joint_mask[asset_row, depth * 4 + finger_index])
            ]  # proximal→distal active canonical slots
            expected_prefix = list(range(len(finger_joint_owners)))
            actual_depths = [(owner_index - 1 - finger_index) // 4 for owner_index in finger_joint_owners]
            if actual_depths != expected_prefix:
                raise ValueError(f"asset row {asset_row} has a non-compact active finger chain")
            tip_owner = 17 + finger_index
            if bool(tip_mask[asset_row, finger_index]) != bool(finger_joint_owners):
                raise ValueError(f"asset row {asset_row} TIP mask disagrees with active finger chain")
            chain = [0, *finger_joint_owners, tip_owner] if finger_joint_owners else [0]
            for ancestor, child in zip(chain, chain[1:]):
                adjacency[ancestor].append(child)
                adjacency[child].append(ancestor)
                parent_index[child] = ancestor

        valid_owners = torch.nonzero(owner_mask[asset_row], as_tuple=False).flatten().tolist()
        for source in valid_owners:
            shortest[asset_row, source, source] = 0
            parent[asset_row, source, source] = 0
            frontier = [source]
            visited = {source}
            distance = 0
            while frontier:
                next_frontier: list[int] = []
                for node in frontier:
                    shortest[asset_row, source, node] = min(distance, max_graph_distance)
                    for neighbor in adjacency[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            next_frontier.append(neighbor)
                frontier = next_frontier
                distance += 1
            ancestor = parent_index[source]
            ancestor_distance = 1
            while ancestor >= 0:
                parent[asset_row, source, ancestor] = min(ancestor_distance, max_graph_distance)
                ancestor = parent_index[ancestor]
                ancestor_distance += 1
    child = parent.transpose(1, 2).contiguous()  # child-direction 是 parent relation 的转置
    return FrozenZProvider(
        asset_ids=tuple(artifact.asset_id for artifact in rows),
        physical_geometry_hashes=tuple(artifact.physical_geometry_hash for artifact in rows),
        owner_valid_mask=owner_mask,
        joint_valid_mask=joint_mask,
        shortest_path=shortest,
        parent_direction=parent,
        child_direction=child,
        dataset_digest=dataset_digest,
        manifest_digest=manifest_digest,
        seed=seed,
        width=width,
    )


__all__ = [
    "FROZEN_Z_ALGORITHM_VERSION",
    "FrozenZBatch",
    "FrozenZProvider",
    "build_frozen_z_provider_from_canonical_artifacts",
]
