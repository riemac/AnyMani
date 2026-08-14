r"""GPU 在线空间 query 采样；只读取静态三角形测度，不查询 target 误差。

静态 owner boundary triangles、face normals、area CDF 与 physical anchors 在每项资产 materialization
阶段固定；本模块每个同资产 q 子批次在线采样 triangle/barycentric/anchor offset，并与当前 owner
$SE(3)$ 位姿组合成 `{h}` query。训练时不移动 mesh、不重建 BVH，也不在采样分布中泄漏最近点、
field prediction 或未来动态状态。

每个 owner 的 query 轴严格按：

$$
N_Q=N_W+N_S+N_A,
\qquad
(N_W,N_S,N_A)=(0.50,0.25,0.25)N_Q.
$$

workspace realization 在同资产 q 子批次内跨多个 $q$ 复用、下一子批次重采；owner-shell 与
adjacent/间隙 query 对每个 $q$ 独立重采并随当前 owner 位姿变化。``query_stratum`` 与来源索引只进入
provenance 和分层 loss，不进入 decoder。

这三层的测度目的不同：

- workspace 从固定 physical anchors 周围的 `{h}` 球云采样，使同一 q 子批次看到可比较的绝对空间；
- owner-shell 同时在 boundary 外侧和 solid 内侧取样，约束 Gaussian 壳层的两侧衰减；
- adjacent 在 owner 图的一跳邻接体之间取近表面插值，强化指间间隙、掌指挂载和 distal 接近区域。

workspace query distribution 必须与 $q$ 独立；其 anchor centers 与 offsets 都表达于 `{h}`，不经过
owner transform。同一 q 子批次共享 realization 只用于降低跨 q 比较方差，下一子批次重采以扩大覆盖。
shell/adjacent 则刻意读取当前 owner boundary；即使 surface point identity 每次重采，它们的局部测度
仍会随 owner 共动，因此不能把 shell 误写成主要 q-sensitive 证据。

采样器的输入边界：

```text
允许：q、owner transforms、静态 owner triangles/area/normals、physical anchors、固定 seed
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

随机数分为静态 realization seed 与 online query seed。静态 home surface/anchor seed 改变 retained
evidence realization；online seed 只改变本批 Monte-Carlo query，不改变 asset content hash。相同 q、
anchors、surface sampling cache 与 online seed 必须逐元素复现，便于有限差分与 target backend 对照。

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
from anymani.robots.owner_geometry import OwnerGeometryCache

from ..targets.field_samples import QueryStratum

SURFACE_QUERY_SAMPLING_VERSION = "owner-triangle-area-barycentric-v1"


@dataclass(frozen=True)
class SpatialQuerySamplerCfg:
    r"""50/25/25 query mixture、anchor workspace 与局部壳层数值配置。"""

    query_count: int = 64  # 每个 owner 的 $N_Q$
    workspace_fraction: float = 0.50  # $N_W/N_Q$
    owner_shell_fraction: float = 0.25  # $N_S/N_Q$
    adjacent_fraction: float = 0.25  # $N_A/N_Q$
    workspace_radius_m: float = 0.05  # 固定 anchor 周围三维均匀球云半径 $R_W$，m
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
        if counts[1] % 2:
            raise ValueError("owner-shell query count must be even for an exact 50/50 inside/outside split")
        if self.workspace_radius_m <= 0.0:
            raise ValueError("workspace anchor-cloud radius must be positive")
        if not 0.0 < self.shell_offset_min_m <= self.shell_offset_max_m:
            raise ValueError("shell offsets must be positive and ordered")
        if self.adjacent_candidate_count < 1:
            raise ValueError("adjacent_candidate_count must be positive")

    @property
    def stratum_counts(self) -> tuple[int, int, int]:
        """返回 $(N_W,N_S,N_A)$。"""

        return (
            round(self.query_count * self.workspace_fraction),
            round(self.query_count * self.owner_shell_fraction),
            round(self.query_count * self.adjacent_fraction),
        )


@dataclass(frozen=True)
class SpatialQueryBatch:
    r"""当前构型 query 与仅供采样审计的来源 provenance。

    该对象由 resolved ``SpatialQuerySamplerCfg`` 的总运行时消费，而不是配置类本身；coordinates 与
    provenance 在一次同资产 q 子批次构造后冻结。``workspace_anchor_index``/``adjacent_owner_index``
    不进入 decoder，防止类别或离散 ID 成为场重建捷径。
    """

    query_points_h: torch.Tensor  # `[B,G,N_Q,3]`，`{h}`，m
    query_stratum: torch.Tensor  # `[B,G,N_Q]`，QueryStratum 整数
    adjacent_owner_index: torch.Tensor  # `[B,G,N_Q]`，非 adjacent 位置为 -1
    workspace_anchor_index: torch.Tensor  # `[B,G,N_Q]`，非 workspace 位置为 -1


@dataclass(frozen=True)
class OwnerSurfaceSamplingCache:
    r"""每项资产在一个 device 上固定的 owner-local 三角表面采样测度。

    每个 owner 保留真实 union surface 的 ``[F_g,3,3]`` triangles、单位 face normals 与归一化面积
    CDF。owner 只做刚体运动，故 triangle area 不随 q 改变；在线阶段只需随机 face、均匀 barycentric
    point 和当前 $T_{hg}(q)$，无需移动 mesh 或重建 BVH。
    """

    triangles_owner_local_m: tuple[torch.Tensor, ...]  # 每项 `[F_g,3,3]`，m
    face_normals_owner_local: tuple[torch.Tensor, ...]  # 每项 `[F_g,3]`，单位向量
    face_area_cdf: tuple[torch.Tensor, ...]  # 每项 `[F_g]`，严格递增并以 1 结尾


def materialize_owner_surface_sampling_cache(
    cache: OwnerGeometryCache,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> OwnerSurfaceSamplingCache:
    r"""把 CPU owner union surface 物化为 GPU 在线采样所需的最小静态测度。

    Args:
        cache (OwnerGeometryCache): 每个 owner 的真实 union surface 与规范顺序。
        device (torch.device | str): 训练/Warp 所在 CUDA device。
        dtype (torch.dtype): 与 q/model 相同的浮点 dtype。

    Returns:
        OwnerSurfaceSamplingCache: owner-local triangle、normal 与 area CDF tuples。
    """

    triangles: list[torch.Tensor] = []  # owner 之间 face 数可变，不能伪 padding 后再采样
    normals: list[torch.Tensor] = []  # face normal 与 triangle 轴逐项同序
    cdfs: list[torch.Tensor] = []  # 面积 categorical 的逆 CDF 路径
    for record in cache.records:
        surface = record.surface_mesh  # CPU float64 真值，只在 device materialization 边界读取
        triangle = torch.as_tensor(np.array(surface.triangles, copy=True), device=device, dtype=dtype)
        normal = torch.as_tensor(np.array(surface.face_normals, copy=True), device=device, dtype=dtype)
        area = torch.as_tensor(np.array(surface.area_faces, copy=True), device=device, dtype=dtype)
        if triangle.ndim != 3 or triangle.shape[1:] != (3, 3) or torch.any(area <= 0.0):
            raise ValueError(f"owner {record.owner_id!r} surface sampling cache requires positive triangles")
        cdf = torch.cumsum(area / area.sum(), dim=0)  # $P(f\le j)=\sum_{i\le j}A_i/\sum_iA_i$
        cdf[-1] = 1.0  # 消除浮点累计误差，保证 $u<1$ 总能映射到最后一个 face
        triangles.append(triangle.contiguous())  # `[F_g,3,3]` owner-local，m
        normals.append(normal.contiguous())  # `[F_g,3]` owner-local 单位向量
        cdfs.append(cdf.contiguous())  # `[F_g]` 归一化面积 CDF
    return OwnerSurfaceSamplingCache(tuple(triangles), tuple(normals), tuple(cdfs))


def sample_spatial_queries(
    q: torch.Tensor,
    spec: EmbodimentGeometrySpec,
    surface_sampling: OwnerSurfaceSamplingCache,
    anchors_hand_m: torch.Tensor,
    *,
    config: SpatialQuerySamplerCfg = SpatialQuerySamplerCfg(),
    sampling_seed: int = 0,
) -> SpatialQueryBatch:
    r"""为一个同资产 q 子批次生成在线 anchor/surface/adjacent query。

    workspace 先均匀有放回选择 anchor，再在半径 $R_W$ 的三维球内按体积均匀采 offset；同一 q
    子批次共享该 realization。shell/adjacent 对每个 q 从完整 owner surface 面积测度重新采 triangle
    与 barycentric point。所有随机路径由 ``sampling_seed`` 唯一复现，输出坐标立即停止梯度。

    Args:
        q (torch.Tensor): ``[B,N_J]`` 当前物理关节角，rad；这里只读取 ``q.detach()`` 生成 query。
        spec (EmbodimentGeometrySpec): robots lower 的动态运动学规格。
        surface_sampling (OwnerSurfaceSamplingCache): GPU owner-local triangle、normal 与 area CDF。
        anchors_hand_m (torch.Tensor): ``[K,3]`` 固定 physical anchors，`{h}`，m。
        config (SpatialQuerySamplerCfg): query 比例和壳层参数。
        sampling_seed (int): 当前 query 抽样 seed；不依赖模型输出。

    Returns:
        SpatialQueryBatch: `[B,G,N_Q,3]` query 与分层 provenance。
    """

    if q.ndim != 2 or q.shape[1] != spec.space_screws.shape[0]:
        raise ValueError("q must have shape [B,N_J]")
    if anchors_hand_m.ndim != 2 or anchors_hand_m.shape[-1] != 3 or anchors_hand_m.shape[0] < 1:
        raise ValueError("anchors_hand_m must have non-empty shape [K,3]")
    if len(surface_sampling.triangles_owner_local_m) != spec.owner_home_transforms.shape[0]:
        raise ValueError("surface sampling cache/spec owner axes must match")

    batch_size = q.shape[0]
    owner_count = spec.owner_home_transforms.shape[0]
    workspace_count, shell_count, adjacent_count = config.stratum_counts
    generator = torch.Generator(device=q.device)
    generator.manual_seed(int(sampling_seed))
    owner_transforms = forward_owner_transforms(spec, q.detach())  # query sampling不保留q梯度
    workspace, workspace_anchor = _sample_anchor_workspace_queries(
        anchors_hand_m.to(device=q.device, dtype=q.dtype),
        batch_size=batch_size,
        owner_count=owner_count,
        workspace_count=workspace_count,
        radius_m=config.workspace_radius_m,
        generator=generator,
    )
    shell = _sample_owner_shell_queries(
        owner_transforms,
        surface_sampling,
        shell_count=shell_count,
        config=config,
        generator=generator,
    )
    adjacent, adjacent_owner_index = _sample_adjacent_queries(
        owner_transforms,
        surface_sampling,
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
    workspace_index = torch.cat(
        (
            workspace_anchor,
            torch.full((batch_size, owner_count, shell_count + adjacent_count), -1, device=q.device),
        ),
        dim=2,
    )
    return SpatialQueryBatch(query_points, query_stratum, adjacent_index, workspace_index)


def _sample_anchor_workspace_queries(
    anchors_hand_m: torch.Tensor,
    *,
    batch_size: int,
    owner_count: int,
    workspace_count: int,
    radius_m: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""在固定 physical anchors 周围采一次 q-independent `{h}` 球云并广播到 q/owner。

    三维均匀体积球不能令半径直接服从均匀分布；若 $U\sim\mathcal U(0,1)$，应使用
    $r=R_WU^{1/3}$，方向由各向同性标准高斯归一化得到。anchor index 均匀有放回采样，使固定
    $N_W$ 不随手型的 anchor 数 $K$ 改变。
    """

    device = anchors_hand_m.device  # workspace 与 q/model 位于同一 device
    anchor_index = torch.randint(  # 每个 workspace slot 独立均匀选择一个 physical anchor
        anchors_hand_m.shape[0],
        (workspace_count,),
        generator=generator,
        device=device,
    )
    direction = torch.randn((workspace_count, 3), generator=generator, device=device)  # 各向同性方向 proposal
    direction = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min(1.0e-12)
    radial = radius_m * torch.rand(  # $r=R_WU^{1/3}$ 才对应球内均匀体积测度
        (workspace_count, 1), generator=generator, device=device
    ).pow(1.0 / 3.0)
    realization = anchors_hand_m.index_select(0, anchor_index) + radial * direction  # `[N_W,3]`，`{h}`，m
    workspace = realization.view(1, 1, workspace_count, 3).expand(batch_size, owner_count, -1, -1)
    provenance = anchor_index.view(1, 1, workspace_count).expand(batch_size, owner_count, -1)
    return workspace, provenance  # q 子批次与 owner 轴共享同一 absolute workspace realization


def _sample_owner_shell_queries(
    owner_transforms: torch.Tensor,
    surface_sampling: OwnerSurfaceSamplingCache,
    *,
    shell_count: int,
    config: SpatialQuerySamplerCfg,
    generator: torch.Generator,
) -> torch.Tensor:
    r"""从完整 current owner surface 面积测度采内外双侧近表面 query。

    每个 query 都重新抽 face 与 triangle interior point，不再从固定 $M_g$ home points 中选索引。
    采样点和法向先在 owner-local 构造，再由当前 $T_{hg}(q)$ 变换到 `{h}`。
    """

    points, normals = _sample_current_owner_surface(
        owner_transforms,
        surface_sampling,
        sample_count=shell_count,
        generator=generator,
    )
    batch_size, owner_count = owner_transforms.shape[:2]  # 同资产 q 子批次与规范 owner 轴
    signs = torch.ones((batch_size, owner_count, shell_count, 1), device=owner_transforms.device)
    signs[..., : shell_count // 2, :] = -1.0
    offsets = torch.rand(
        (batch_size, owner_count, shell_count, 1), generator=generator, device=owner_transforms.device
    )
    offsets = config.shell_offset_min_m + offsets * (config.shell_offset_max_m - config.shell_offset_min_m)
    return points + signs * offsets * normals  # $x=p\pm\delta n$，`{h}`，m


def _sample_adjacent_queries(
    owner_transforms: torch.Tensor,
    surface_sampling: OwnerSurfaceSamplingCache,
    spec: EmbodimentGeometrySpec,
    *,
    adjacent_count: int,
    candidate_count: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""均匀覆盖全部一跳 graph neighbors，并从 current surfaces 近候选对采间隙 query。"""

    batch_size, owner_count = owner_transforms.shape[:2]  # `[B,G]` current owner pose 轴
    if spec.owner_graph_shortest is None:
        raise ValueError("spec must contain owner graph distances for adjacent query sampling")
    neighbors = [
        torch.where(spec.owner_graph_shortest[owner_index] == 1)[0]
        for owner_index in range(owner_count)
    ]
    if any(len(values) == 0 for values in neighbors):
        raise ValueError("every owner must have at least one graph neighbor for adjacent queries")
    selected_neighbor = torch.empty(  # 每个 adjacent slot 独立从当前 owner 的全部一跳邻居均匀抽样
        (batch_size, owner_count, adjacent_count), device=owner_transforms.device, dtype=torch.long
    )
    for owner_index, values in enumerate(neighbors):
        choice = torch.randint(  # 有放回 Monte-Carlo；跨 step 长期覆盖全部 parent/child neighbors
            len(values),
            (batch_size, adjacent_count),
            generator=generator,
            device=owner_transforms.device,
        )
        selected_neighbor[:, owner_index] = values.to(device=owner_transforms.device).index_select(0, choice.reshape(-1)).reshape(
            batch_size, adjacent_count
        )

    sample_count = adjacent_count * candidate_count  # 每个 query 比较 $C_A$ 个 current-surface pairs
    left, _ = _sample_current_owner_surface(  # target owner 的面积加权连续 surface candidates
        owner_transforms, surface_sampling, sample_count=sample_count, generator=generator
    )
    right_all, _ = _sample_current_owner_surface(  # 每个 potential neighbor 自己的独立 candidates
        owner_transforms, surface_sampling, sample_count=sample_count, generator=generator
    )
    left = left.reshape(batch_size, owner_count, adjacent_count, candidate_count, 3)
    right_all = right_all.reshape(batch_size, owner_count, adjacent_count, candidate_count, 3)
    neighbor_gather = selected_neighbor.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, candidate_count, 3)
    right = torch.gather(right_all, 1, neighbor_gather)  # `[B,G,N_A,C_A,3]` selected neighbor candidates
    distances = torch.linalg.vector_norm(left - right, dim=-1)
    best = torch.argmin(distances, dim=-1)
    best_index = best.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, 3)
    left_best = torch.gather(left, 3, best_index).squeeze(3)
    right_best = torch.gather(right, 3, best_index).squeeze(3)
    interpolation = 0.25 + 0.5 * torch.rand(
        (batch_size, owner_count, adjacent_count, 1), generator=generator, device=owner_transforms.device
    )
    return (1.0 - interpolation) * left_best + interpolation * right_best, selected_neighbor


def _sample_current_owner_surface(
    owner_transforms: torch.Tensor,
    surface_sampling: OwnerSurfaceSamplingCache,
    *,
    sample_count: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""从每个 owner 的完整面积测度采 current `{h}` surface point 与 face normal。

    triangle interior 使用 $u,v\sim\mathcal U(0,1)$、$a=\sqrt u$ 与 barycentric weights
    $(1-a,a(1-v),av)$，保证三角形面积均匀。owner face 数可变，因此只沿 $G$ 做小循环，所有
    q/sample 轴保持 GPU 向量化。
    """

    batch_size, owner_count = owner_transforms.shape[:2]  # 同资产 q 子批次与规范 owner 数
    points = torch.empty(  # `[B,G,N,3]` current surface points，`{h}`，m
        (batch_size, owner_count, sample_count, 3), device=owner_transforms.device, dtype=owner_transforms.dtype
    )
    normals = torch.empty_like(points)  # `[B,G,N,3]` current face unit normals
    for owner_index in range(owner_count):
        cdf = surface_sampling.face_area_cdf[owner_index]  # `[F_g]`，面积 categorical CDF
        face_uniform = torch.rand(  # `[B,N]`，每个 q/query 独立重采 face identity
            (batch_size, sample_count), generator=generator, device=owner_transforms.device
        )
        face_index = torch.searchsorted(cdf, face_uniform.contiguous())  # 面积加权 face selector
        triangles = surface_sampling.triangles_owner_local_m[owner_index][face_index]  # `[B,N,3,3]`
        root = torch.sqrt(torch.rand(  # $a=\sqrt u$ 消除 naive barycentric 的中心偏置
            (batch_size, sample_count, 1), generator=generator, device=owner_transforms.device
        ))
        second = torch.rand(  # $v$ 在与 $a$ 正交的 simplex 轴上均匀
            (batch_size, sample_count, 1), generator=generator, device=owner_transforms.device
        )
        barycentric = torch.cat((1.0 - root, root * (1.0 - second), root * second), dim=-1)
        local_point = torch.sum(triangles * barycentric.unsqueeze(-1), dim=-2)  # `[B,N,3]`，owner-local，m
        local_normal = surface_sampling.face_normals_owner_local[owner_index][face_index]  # `[B,N,3]`
        rotation = owner_transforms[:, owner_index, :3, :3]  # owner-local -> `{h}` 当前旋转
        translation = owner_transforms[:, owner_index, :3, 3]  # owner reference origin，`{h}`，m
        points[:, owner_index] = torch.einsum("bij,bnj->bni", rotation, local_point) + translation.unsqueeze(1)
        normals[:, owner_index] = torch.einsum("bij,bnj->bni", rotation, local_normal)  # 方向不叠加平移
    return points, normals


__all__ = [
    "OwnerSurfaceSamplingCache",
    "SURFACE_QUERY_SAMPLING_VERSION",
    "SpatialQueryBatch",
    "SpatialQuerySamplerCfg",
    "materialize_owner_surface_sampling_cache",
    "sample_spatial_queries",
]
