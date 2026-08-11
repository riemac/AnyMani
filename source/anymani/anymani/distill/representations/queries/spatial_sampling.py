r"""GPU 在线空间 query 采样；不查询 mesh、不读取 target 误差。

静态 owner boundary、surface normals 与 workspace bank 由 ``robots.owner_geometry`` 离线物化；
本模块只把它们与当前 $q$ 的 owner $SE(3)$ 位姿组合，生成 `{h}` 中的 query。这样训练时不移动
mesh、不重建 BVH，也不在采样分布中泄漏最近点、field prediction 或未来动态状态。

每个 owner 的 query 轴严格按：

$$
N_Q=N_W+N_S+N_A,
\qquad
(N_W,N_S,N_A)=(0.50,0.25,0.25)N_Q.
$$

workspace bank 在所有 $q$ 之间复用；owner-shell 和 adjacent/间隙 query 随当前 owner 位姿变化。
``query_stratum`` 只进入 provenance 与分层 loss，不进入 decoder。

这三层的测度目的不同：

- workspace 让模型看到跨构型共享的绝对米制工作空间，防止只在表面附近记忆局部形状；
- owner-shell 同时在 boundary 外侧和 solid 内侧取样，约束 Gaussian 壳层的两侧衰减；
- adjacent 在 owner 图的一跳邻接体之间取近表面插值，强化指间间隙、掌指挂载和 distal 接近区域。

workspace query 必须在构型之间复用。若每个 owner 的 query 都随该 owner 一起刚体共动，模型可以
通过 query 坐标的共动模式猜测当前 q，而不是恢复真实表面变化；这会使隐式场重建误差看起来很好，
却不能证明 hand-frame physical field 被学习。shell/adjacent 则刻意随 q 更新，因为它们的物理
意义是“当前 owner boundary 的局部邻域”。

采样器的输入边界：

```text
允许：q、robots owner transforms、离线 home surface points/normals、静态 workspace bank、固定 seed
禁止：closest point、distance、density、prediction error、contact/object state、历史或动作
```

因此本模块可以在 target backend 之前运行，并可在 query 采样后对 query coordinates 做停止梯度。
训练模型只收到 query coordinates 作为 decoder 的空间条件；query_stratum 和 adjacent owner index
只留在 target/provenance，不能成为 decoder 的类别输入。

所有长度使用 m。shell offset 是沿 owner-local boundary normal 旋转到 `{h}` 后的米制偏移；外侧
使用正号，内侧使用负号。对非光滑 mesh corner，normal 来自 union face provenance，故该 query
本身可以被 target backend 屏蔽，不应在 sampler 中偷偷改变物理面。

adjacent query 不把不同 owner Boolean 合并。它从 owner graph 的一跳邻居选取两组当前 surface
候选，比较少量候选 pair 的 hand-frame 距离，选局部最近的一对，再在两点之间取 25% 到 75% 的
插值。它是间隙 query 的 Monte-Carlo 近似，不是最近点教师；严格距离仍由 Warp owner-local BVH
查询给出。若 owner 图没有邻居，配置直接失败，不能把 workspace 点冒充 adjacent。

随机数分为静态 realization seed 与 online query seed。静态 home surface/anchor seed 改变 cache
identity；online seed 只改变本批 Monte-Carlo query，不改变 asset content hash。相同 q、静态 cache、
workspace bank 和 online seed 必须逐元素复现，便于有限差分与 target backend 对照。

`query_count=64`、$(32,16,16)$ 是首个 smoke/calibration preset。它不等价于正式实验选择；正式
网格应在 generated-only 训练中比较 query count、shell offsets、workspace padding、anchor count
和 bandwidth，并在固定 held-out bank 上报告按 stratum 的误差。fraction 不能通过四舍五入静默改成
另一测度，配置校验会拒绝无法整数分解的 query_count。

该模块只产生坐标与 provenance，不产生 field label，也不假设 owner surface 是凸体。凸 hull 只会
填充真实凹槽，不能作为 workspace/shell/adjacent 的 geometry source。所有 boundary 真值来自
robots 的严格 owner union，所有动态位姿来自 robots 的 POE lowering。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from anymani.robots.geometry_kinematics import EmbodimentGeometrySpec, forward_owner_transforms
from anymani.robots.owner_geometry import HomeSurfaceSamples, OwnerGeometryCache

from ..targets.field_samples import QueryStratum


@dataclass(frozen=True)
class SpatialQuerySamplerCfg:
    r"""50/25/25 query mixture与局部壳层数值配置。"""

    query_count: int = 64  # 每个 owner 的 $N_Q$
    workspace_fraction: float = 0.50  # $N_W/N_Q$
    owner_shell_fraction: float = 0.25  # $N_S/N_Q$
    adjacent_fraction: float = 0.25  # $N_A/N_Q$
    shell_offset_min_m: float = 0.0005  # 双侧壳层最小偏移，m
    shell_offset_max_m: float = 0.004  # 双侧壳层最大偏移，m
    adjacent_candidate_count: int = 4  # 每个间隙 query 比较的候选 surface pair 数

    def __post_init__(self) -> None:
        """验证比例和整数 query 分解，拒绝静默改变监督测度。"""

        if self.query_count < 4:
            raise ValueError("query_count must leave at least one query in each stratum")
        fractions = (self.workspace_fraction, self.owner_shell_fraction, self.adjacent_fraction)
        if any(fraction < 0.0 for fraction in fractions) or not np.isclose(sum(fractions), 1.0):
            raise ValueError("query stratum fractions must be non-negative and sum to one")
        counts = tuple(round(self.query_count * fraction) for fraction in fractions)
        if sum(counts) != self.query_count or any(count < 1 for count in counts):
            raise ValueError(
                f"query_count={self.query_count} cannot represent configured 50/25/25 mixture as integers"
            )
        if not 0.0 < self.shell_offset_min_m <= self.shell_offset_max_m:
            raise ValueError("shell offsets must be positive and ordered")
        if self.adjacent_candidate_count < 1:
            raise ValueError("adjacent_candidate_count must be positive")

    @property
    def stratum_counts(self) -> tuple[int, int, int]:
        """返回 $(N_W,N_S,N_A)$。"""

        return tuple(round(self.query_count * fraction) for fraction in (
            self.workspace_fraction,
            self.owner_shell_fraction,
            self.adjacent_fraction,
        ))


@dataclass(frozen=True)
class SpatialQueryBatch:
    r"""当前构型 query 与来源 provenance。"""

    query_points_h: torch.Tensor  # `[B,G,N_Q,3]`，`{h}`，m
    query_stratum: torch.Tensor  # `[B,G,N_Q]`，QueryStratum 整数
    adjacent_owner_index: torch.Tensor  # `[B,G,N_Q]`，非 adjacent 位置为 -1


def build_workspace_query_bank(
    cache: OwnerGeometryCache,
    spec: EmbodimentGeometrySpec,
    home_surface: HomeSurfaceSamples,
    *,
    query_count: int,
    padding_m: float = 0.03,
    sampling_seed: int = 0,
) -> torch.Tensor:
    r"""从基准 owner boundary 的 hand-frame 包围盒构造固定 workspace bank。

    该 bank 只使用静态 geometry，不随 $q$ 更新；它被所有构型重复广播，避免共动 query 造成
    “查询点随手指一起移动”的静态标签退化。
    """

    if query_count < 1 or padding_m < 0.0:
        raise ValueError("query_count must be positive and padding_m must be non-negative")
    owner_transforms = spec.owner_home_transforms.detach().cpu()
    local_points = torch.as_tensor(home_surface.points_owner_local_m, dtype=owner_transforms.dtype)
    hand_points = _transform_owner_points(owner_transforms.unsqueeze(0), local_points).reshape(-1, 3).numpy()
    lower = hand_points.min(axis=0) - float(padding_m)
    upper = hand_points.max(axis=0) + float(padding_m)
    rng = np.random.default_rng(int(sampling_seed))
    workspace = rng.uniform(lower, upper, size=(query_count, 3)).astype(np.float32)
    return torch.from_numpy(workspace)


def sample_spatial_queries(
    q: torch.Tensor,
    spec: EmbodimentGeometrySpec,
    cache: OwnerGeometryCache,
    home_surface: HomeSurfaceSamples,
    workspace_query_bank_h: torch.Tensor,
    *,
    config: SpatialQuerySamplerCfg = SpatialQuerySamplerCfg(),
    sampling_seed: int = 0,
) -> SpatialQueryBatch:
    r"""在 GPU 上生成当前构型的 50/25/25 query batch。

    Args:
        q (torch.Tensor): ``[B,N_J]`` 当前物理关节角，rad；这里只读取 ``q.detach()`` 生成 query。
        spec (EmbodimentGeometrySpec): robots lower 的动态运动学规格。
        cache (OwnerGeometryCache): owner-local boundary 与 component provenance。
        home_surface (HomeSurfaceSamples): 固定 owner-local boundary samples 和 face index。
        workspace_query_bank_h (torch.Tensor): 固定 ``[N_W,3]`` hand-frame bank，m。
        config (SpatialQuerySamplerCfg): query 比例和壳层参数。
        sampling_seed (int): 当前 query 抽样 seed；不依赖模型输出。

    Returns:
        SpatialQueryBatch: `[B,G,N_Q,3]` query 与分层 provenance。
    """

    if q.ndim != 2 or q.shape[1] != spec.space_screws.shape[0]:
        raise ValueError("q must have shape [B,N_J]")
    if workspace_query_bank_h.shape != (config.stratum_counts[0], 3):
        raise ValueError("workspace_query_bank_h must have shape [N_W,3]")
    if len(cache.records) != spec.owner_home_transforms.shape[0]:
        raise ValueError("cache/spec owner axes must match")

    batch_size = q.shape[0]
    owner_count = spec.owner_home_transforms.shape[0]
    workspace_count, shell_count, adjacent_count = config.stratum_counts
    generator = torch.Generator(device=q.device)
    generator.manual_seed(int(sampling_seed))
    owner_transforms = forward_owner_transforms(spec, q.detach())  # query sampling不保留q梯度
    home_points = torch.as_tensor(home_surface.points_owner_local_m, device=q.device, dtype=q.dtype)
    home_normals = _home_surface_normals(cache, home_surface, device=q.device, dtype=q.dtype)
    current_surface_points = _transform_owner_points(owner_transforms, home_points)
    current_surface_normals = _transform_owner_vectors(owner_transforms, home_normals)

    workspace = workspace_query_bank_h.to(device=q.device, dtype=q.dtype)
    workspace = workspace.view(1, 1, workspace_count, 3).expand(batch_size, owner_count, -1, -1)
    shell = _sample_owner_shell_queries(
        current_surface_points,
        current_surface_normals,
        shell_count=shell_count,
        config=config,
        generator=generator,
    )
    adjacent, adjacent_owner_index = _sample_adjacent_queries(
        current_surface_points,
        spec,
        adjacent_count=adjacent_count,
        candidate_count=config.adjacent_candidate_count,
        generator=generator,
    )
    query_points = torch.cat((workspace, shell, adjacent), dim=2).detach()
    query_stratum = torch.cat(
        (
            torch.full((batch_size, owner_count, workspace_count), int(QueryStratum.WORKSPACE), device=q.device),
            torch.full((batch_size, owner_count, shell_count), int(QueryStratum.OWNER_SHELL), device=q.device),
            torch.full((batch_size, owner_count, adjacent_count), int(QueryStratum.ADJACENT), device=q.device),
        ),
        dim=2,
    )
    adjacent_index = torch.cat(
        (
            torch.full((batch_size, owner_count, workspace_count + shell_count), -1, device=q.device),
            adjacent_owner_index,
        ),
        dim=2,
    )
    return SpatialQueryBatch(query_points, query_stratum, adjacent_index)


def _home_surface_normals(
    cache: OwnerGeometryCache,
    home_surface: HomeSurfaceSamples,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """从固定 face provenance 读取 owner-local boundary normals。"""

    normals = np.stack(
        [record.mesh.face_normals[face_indices] for record, face_indices in zip(cache.records, home_surface.face_indices)],
        axis=0,
    )
    return torch.as_tensor(normals, device=device, dtype=dtype)


def _sample_owner_shell_queries(
    surface_points: torch.Tensor,
    surface_normals: torch.Tensor,
    *,
    shell_count: int,
    config: SpatialQuerySamplerCfg,
    generator: torch.Generator,
) -> torch.Tensor:
    """沿 boundary normal 生成内外双侧近表面 query。"""

    batch_size, owner_count, home_count, _ = surface_points.shape
    indices = torch.randint(
        home_count,
        (batch_size, owner_count, shell_count),
        generator=generator,
        device=surface_points.device,
    )
    gather_index = indices.unsqueeze(-1).expand(-1, -1, -1, 3)
    points = torch.gather(surface_points, 2, gather_index)
    normals = torch.gather(surface_normals, 2, gather_index)
    signs = torch.ones((batch_size, owner_count, shell_count, 1), device=surface_points.device)
    signs[..., : shell_count // 2, :] = -1.0
    offsets = torch.rand(
        (batch_size, owner_count, shell_count, 1), generator=generator, device=surface_points.device
    )
    offsets = config.shell_offset_min_m + offsets * (config.shell_offset_max_m - config.shell_offset_min_m)
    return points + signs * offsets * normals


def _sample_adjacent_queries(
    surface_points: torch.Tensor,
    spec: EmbodimentGeometrySpec,
    *,
    adjacent_count: int,
    candidate_count: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从 owner 图邻接体的近表面候选对中采样间隙 query。"""

    batch_size, owner_count, home_count, _ = surface_points.shape
    if spec.owner_graph_shortest is None:
        raise ValueError("spec must contain owner graph distances for adjacent query sampling")
    neighbors = [
        torch.where(spec.owner_graph_shortest[owner_index] == 1)[0]
        for owner_index in range(owner_count)
    ]
    if any(len(values) == 0 for values in neighbors):
        raise ValueError("every owner must have at least one graph neighbor for adjacent queries")
    neighbor_index = torch.tensor([int(values[0]) for values in neighbors], device=surface_points.device)
    selected_neighbor = neighbor_index.view(1, owner_count, 1).expand(batch_size, -1, adjacent_count)

    left_index = torch.randint(
        home_count,
        (batch_size, owner_count, adjacent_count, candidate_count),
        generator=generator,
        device=surface_points.device,
    )
    right_index = torch.randint(
        home_count,
        (batch_size, owner_count, adjacent_count, candidate_count),
        generator=generator,
        device=surface_points.device,
    )
    left = _gather_surface_points(surface_points, left_index)
    neighbor_gather = neighbor_index.view(1, owner_count, 1, 1).expand(
        batch_size, owner_count, home_count, 3
    )
    neighbor_surface = torch.gather(surface_points, 1, neighbor_gather)  # `[B,G,M,3]`
    right = _gather_surface_points(neighbor_surface, right_index)
    distances = torch.linalg.vector_norm(left - right, dim=-1)
    best = torch.argmin(distances, dim=-1)
    best_index = best.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, 3)
    left_best = torch.gather(left, 3, best_index).squeeze(3)
    right_best = torch.gather(right, 3, best_index).squeeze(3)
    interpolation = 0.25 + 0.5 * torch.rand(
        (batch_size, owner_count, adjacent_count, 1), generator=generator, device=surface_points.device
    )
    return (1.0 - interpolation) * left_best + interpolation * right_best, selected_neighbor


def _gather_surface_points(surface_points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """按 `[B,G,...]` 索引读取 owner-local transformed surface points。"""

    gather_index = indices.unsqueeze(-1).expand(*indices.shape, 3)
    return torch.gather(surface_points.unsqueeze(2).expand(-1, -1, indices.shape[2], -1, -1), 3, gather_index)


def _transform_owner_points(owner_transforms: torch.Tensor, local_points: torch.Tensor) -> torch.Tensor:
    """批量把 `[G,M,3]` 或 `[B,G,M,3]` 点变换到 `{h}`。"""

    rotation = owner_transforms[..., :3, :3]
    translation = owner_transforms[..., :3, 3]
    if local_points.ndim == 3:
        local_points = local_points.unsqueeze(0).expand(owner_transforms.shape[0], -1, -1, -1)
    return torch.einsum("bgij,bgmj->bgmi", rotation, local_points) + translation.unsqueeze(-2)


def _transform_owner_vectors(owner_transforms: torch.Tensor, local_vectors: torch.Tensor) -> torch.Tensor:
    """批量旋转 owner-local 法向，不叠加平移。"""

    rotation = owner_transforms[..., :3, :3]
    if local_vectors.ndim == 3:
        local_vectors = local_vectors.unsqueeze(0).expand(owner_transforms.shape[0], -1, -1, -1)
    return torch.einsum("bgij,bgmj->bgmi", rotation, local_vectors)


__all__ = [
    "SpatialQueryBatch",
    "SpatialQuerySamplerCfg",
    "build_workspace_query_bank",
    "sample_spatial_queries",
]
