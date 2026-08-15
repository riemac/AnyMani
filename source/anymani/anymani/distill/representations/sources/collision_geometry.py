r"""按 owner 构造 collision surface、可选 solid 与基准表面采样缓存。

输入是 ``assets.bank.HandContainer`` 的类型化几何语义，以及 source 层已 lower 的
``EmbodimentGeometrySpec``。每个碰撞片先从自身局部坐标严格变换到 owner reference link。surface
与 solid 是两类不同的物理证据：unsigned UDF、最近点和 home point 只要求三角表面；同 owner 多个
闭合 solid 才调用 ``trimesh.boolean.union(..., engine="manifold")``。任何 Boolean 失败均拒绝，
不使用补洞、convex hull、包围盒或删除近似重叠面作为替代真值。

Boolean union 的边界自然删除完全埋藏的内部面，同时保留与外部连通的开放凹槽内壁。不同 owner
永不合并，因为跨 owner 最近点来源和一阶 Jacobian 必须保持可辨认。

静态缓存与在线缓存的生命周期严格分开：

```text
assets.bank.HandContainer
    -> materialize_owner_geometry_cache       # CPU/offline, once per asset hash
    -> sample_owner_home_surfaces              # CPU/offline, fixed realization
    -> sample_palm_anchor_supports              # CPU/offline, fixed realization
    -> materialize_warp_owner_geometry_cache   # GPU, once per asset/device
    -> distill query/target                    # online q-dependent transforms and BVH queries
```

CPU 侧使用 trimesh 是因为这里的主任务是读取 mesh、严格检查 volume、执行一次 Manifold Boolean
和按真实面积产生固定证据；这些操作不是每个训练 step 的批量最近点查询。`manifold3d` 是唯一的
多 solid union 后端，版本写入 cache provenance。Blender、convex hull、voxel hull 与包围盒都不
能替代失败的 union，因为它们会改变真实凹槽、埋藏面和 surface area。

owner-local 变换的物理方向是：component collision frame 先经过
`component_owner_local_transforms[c]` 到 owner reference link，再由当前
$T_{hg}(q)$ 到 `{h}`。缓存 mesh 永远不随 q 移动；online query 将 `{h}` query 反变换到 local
BVH。这个方向同时保证同一 owner 的多个 component 可以在统一 local frame Boolean，而不产生
每 batch mesh transform 或 BVH rebuild。

`OwnerSurfaceRecord.surface_mesh` 只要求有限、非空的三角表面；它定义
$d_g(x;q)=\inf_{y\in S_g(q)}\|x-y\|_2$，不需要 inside/outside。`solid_mesh` 只有在所有 owner
component 都是合法 volume 时存在；Boolean union 与 palm interior anchor 必须显式读取它。开放表面
不得补洞或构造伪 solid。STL 的逐面重复顶点会在分类前确定性焊接，避免把表示冗余误判为真实破洞。

表面采样的两步测度不能混淆：第一步按三角面面积采候选，使大面按物理面积获得更多样本；第二步
在候选中做确定性最远点子采样，改善有限 $M_g$ 预算的空间覆盖。输出保存 owner-local point、
union face index 与 barycentric 坐标，法向只供 query shell 采样，不进入 retained encoder。改变
mesh tessellation 或 oversample factor 必须通过 cache version/provenance 明确记录。

anchor 与 home surface 的语义不同。home points 永远在 owner boundary 上，不能混入 solid interior；
anchor 允许从 palm surface/interior 支持取得，因为 anchor 的作用是提供 mount-conditioned
reference constellation，而不是声称它们是 collision surface。每根 finger 的 seed 只进入
`finger_names/seed_ids` provenance；`anchors_hand_m` 输出成完整无序 $K$ 集合。

Warp cache 持有 `wp.Mesh`、points、indices、face altitudes 与 CPU-face 映射的强引用。上传前按实际
float32 顶点过滤零边长/零面积三角形；删除面积比例超过 $10^{-8}$ 时严格失败。cache key 包含实际
surface hash、surface-processing version 与 device，并通过 lease/release 让 GPU asset window 驱逐
全局强引用。在线 Warp 失败必须暴露为 backend error，不能静默切换 trimesh。

当前 Warp face margin 定义为最近投影点到选中三角形三条边的最小物理距离：若重心坐标为
$(b_0,b_1,b_2)$、对应高为 $(h_0,h_1,h_2)$，则

$$
m_{feature}=\min_i b_i h_i.
$$

它只证明当前三角面 feature 的局部 interior 余量，不等价于全局第二近表面间隔；target backend
必须把这一点写进 provenance，并在需要严格 medial-axis 证据时使用独立 reference。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import trimesh

from anymani.assets.asset_schema_geometry import (
    CollisionComponentSemanticsCfg,
    HandGeometrySemanticsCfg,
)
from anymani.assets.bank import HandContainer

from .kinematics import EmbodimentGeometrySpec


@dataclass(frozen=True)
class OwnerSurfaceRecord:
    r"""一个 owner 的三角表面、可选闭合体及其碰撞片 provenance。"""

    owner_id: str  # 与语义实体轴一致的稳定 ID
    owner_index: int  # 规范实体轴索引
    role: str  # palm/joint/tip
    component_ids: tuple[str, ...]  # 参与并集的碰撞片
    surface_mesh: trimesh.Trimesh  # owner reference link 中的 UDF 表面 $S_g$，单位 m
    solid_mesh: trimesh.Trimesh | None  # 合法闭合体；开放 surface owner 为 None
    boolean_applied: bool  # 仅所有 component 都是 solid 且数量大于一时为 True


@dataclass(frozen=True)
class OwnerGeometryCache:
    r"""一项资产可复用的 owner-local 静态几何缓存。"""

    asset_id: str
    asset_content_hash: str
    boolean_backend: str
    records: tuple[OwnerSurfaceRecord, ...]
    surface_geometry_hash: str = ""  # 实际 owner-local vertices/faces 的内容哈希
    surface_processing_version: str = "owner-surface-v2"  # surface/solid 与 float32 清理合同版本


@dataclass(frozen=True)
class HomeSurfaceSamples:
    r"""按真实表面积候选再均匀化得到的固定 owner 基准表面点。"""

    owner_ids: tuple[str, ...]
    points_owner_local_m: np.ndarray  # `[G,M,3]`，只含 boundary point
    face_indices: np.ndarray  # `[G,M]`，指向 union mesh face
    barycentric: np.ndarray  # `[G,M,3]`，供重建与后端对照
    sampling_seed: int
    oversample_factor: int


@dataclass(frozen=True)
class AnchorSamples:
    r"""每个资产固定一次的 palm 支持锚点 realization。"""

    anchors_hand_m: np.ndarray  # `[K,3]`，统一 `{h}` 坐标，m
    finger_names: tuple[str, ...]  # `[K]` provenance；不进入网络分组
    seed_ids: tuple[str, ...]  # `[K]` provenance；只用于重现与审计
    surface_mask: np.ndarray  # `[K]`，surface/interior 采样来源
    radial_support_radius_m: float
    radial_decay_scale_m: float  # 截断 Gaussian 的 $\tau_a$，m；只改变支持球内的候选测度
    surface_fraction: float
    sampling_seed: int
    algorithm_version: str  # 修改 proposal/acceptance/selection 语义时必须显式升级


@dataclass(frozen=True)
class WarpOwnerMeshHandle:
    r"""一个 owner 的 GPU-resident Warp BVH 及其底层数组强引用。"""

    owner_id: str
    mesh: object  # wp.Mesh；object 避免普通 import 强制加载/初始化 Warp
    points: object  # wp.array[vec3]；必须与 mesh 同生命周期
    indices: object  # wp.array[int32]；必须与 mesh 同生命周期
    source_face_indices: object  # wp.array[int32]；filtered BVH face -> CPU surface face
    face_altitudes: object  # wp.array[vec3]；每个重心坐标对应的三角形高，单位 m
    face_count: int
    surface_audit: WarpSurfaceAudit  # float32 退化面删除数量与面积守恒证据


@dataclass(frozen=True)
class WarpOwnerGeometryCache:
    r"""一项资产全部 owner-local mesh 的 GPU BVH 缓存。"""

    asset_id: str
    asset_content_hash: str
    surface_geometry_hash: str
    surface_processing_version: str
    device: str
    handles: tuple[WarpOwnerMeshHandle, ...]


@dataclass(frozen=True)
class WarpSurfaceAudit:
    r"""一个 owner 在 float64 CPU surface 到 float32 Warp view 间的面积审计。"""

    input_face_count: int  # CPU surface 三角形数
    output_face_count: int  # float32 有效三角形数
    removed_face_count: int  # float32 中零边长或零面积三角形数
    input_area_m2: float  # CPU float64 总面积，$\mathrm{m}^2$
    removed_area_m2: float  # 被删面在 CPU float64 下的总面积，$\mathrm{m}^2$
    removed_area_fraction: float  # removed/input，无量纲


@dataclass(frozen=True)
class WarpSurfaceView:
    r"""实际上传 Warp 的 float32 表面及 CPU face provenance。"""

    vertices: np.ndarray  # `[V,3]` float32，m
    faces: np.ndarray  # `[F_valid,3]` int32
    source_face_indices: np.ndarray  # `[F_valid]`，指向 `surface_mesh.faces`
    audit: WarpSurfaceAudit  # 面积守恒和删除数量


@dataclass(frozen=True)
class GeometryIdentity:
    r"""一项已物化资产的物理映射身份与构型采样域身份。

    `physical_geometry_hash` 覆盖 frame、$q_{home}$、空间旋量、owner home 位姿、
    祖先/图关系、component-to-owner 变换和实际 owner surface；明确排除 joint limits。
    `configuration_domain_hash` 只覆盖规范 joint names 与合法角域，因此 limit-only
    variants 会共享物理 group，但保留不同 Sobol 采样域 provenance。
    """

    physical_geometry_hash: str  # 学习映射 leakage group 的 SHA-256
    configuration_domain_hash: str  # `joint names + limits` 的 SHA-256


_WarpCacheKey = tuple[str, str, str]
_WARP_OWNER_CACHE: dict[_WarpCacheKey, WarpOwnerGeometryCache] = {}
"""按 ``(surface_geometry_hash, surface_processing_version, device)`` 复用 BVH。"""

_WARP_OWNER_CACHE_LEASES: dict[_WarpCacheKey, int] = {}
"""每项 GPU cache 的活跃 resident-window lease 数；归零即驱逐全局强引用。"""


def materialize_owner_geometry_cache(
    container: HandContainer,
    spec: EmbodimentGeometrySpec,
) -> OwnerGeometryCache:
    r"""把一个 bank container 的全部碰撞片物化为逐 owner 真实并集边界。

    Args:
        container (HandContainer): 含类型化 ``geometry_semantics`` 与 mesh 虚拟路径映射的资产。
        spec (EmbodimentGeometrySpec): 同一语义 lower 的 component-to-owner 变换。

    Returns:
        OwnerGeometryCache: 规范 owner 顺序的闭合三角网格边界。
    """

    semantics = container.geometry_semantics
    if semantics is None:
        raise ValueError("HandContainer must be resolved with require_geometry_semantics=True")
    if spec.owner_ids != tuple(owner.owner_id for owner in semantics.owners):
        raise ValueError("EmbodimentGeometrySpec owner axis does not match container geometry semantics")
    if spec.component_owner_local_transforms is None:
        raise ValueError("EmbodimentGeometrySpec is missing component_owner_local_transforms")
    if spec.component_owner_local_transforms.shape[0] != len(semantics.components):
        raise ValueError("component transform axis does not match geometry semantics")

    transformed_by_owner: dict[str, list[trimesh.Trimesh]] = {
        owner.owner_id: [] for owner in semantics.owners
    }
    component_index_by_id = {
        component.component_id: component_index
        for component_index, component in enumerate(semantics.components)
    }
    for component_index, component in enumerate(semantics.components):
        mesh = _component_mesh(component, container=container)
        transform = spec.component_owner_local_transforms[component_index].detach().cpu().numpy()
        mesh.apply_transform(transform)  # collision local -> owner reference link
        _require_surface(mesh, context=f"component '{component.component_id}'")  # UDF 只需真实三角表面
        transformed_by_owner[component.owner_id].append(mesh)

    records: list[OwnerSurfaceRecord] = []
    for owner in semantics.owners:
        component_meshes = transformed_by_owner[owner.owner_id]
        if len(component_meshes) != len(owner.component_ids):
            raise ValueError(f"owner '{owner.owner_id}' component materialization is incomplete")
        all_components_are_solid = all(_is_volume(mesh) for mesh in component_meshes)  # solid 可用性不能靠路径猜测
        solid_mesh = (
            strict_owner_union(component_meshes, owner_id=owner.owner_id)
            if all_components_are_solid
            else None
        )  # 只有全部 component 都是 volume 时，Boolean 才有集合并语义
        surface_mesh = (
            solid_mesh.copy()
            if solid_mesh is not None
            else _concatenate_owner_surfaces(component_meshes, owner_id=owner.owner_id)
        )  # 开放 component 保留原三角边界，不补洞、不 hull
        records.append(
            OwnerSurfaceRecord(
                owner_id=owner.owner_id,
                owner_index=owner.owner_index,
                role=owner.role,
                component_ids=tuple(owner.component_ids),
                surface_mesh=surface_mesh,
                solid_mesh=solid_mesh,
                boolean_applied=solid_mesh is not None and len(component_meshes) > 1,
            )
        )

    # 额外检查 sidecar 组件顺序和运动学 component 轴仍完全一致，避免缓存串包。
    if tuple(component_index_by_id) != tuple(component.component_id for component in semantics.components):
        raise ValueError("component provenance order is inconsistent")
    surface_geometry_hash = _owner_surface_geometry_hash(tuple(records))  # cache identity 来自实际物理表面
    return OwnerGeometryCache(
        asset_id=container.asset_id,
        asset_content_hash=semantics.content_hash,
        boolean_backend=f"manifold3d=={version('manifold3d')}",
        records=tuple(records),
        surface_geometry_hash=surface_geometry_hash,
    )


def materialize_warp_owner_geometry_cache(
    cache: OwnerGeometryCache,
    *,
    device: str = "cuda:0",
) -> WarpOwnerGeometryCache:
    r"""把 CPU 上验证过的 owner union 一次性上传为 Warp BVH。

    该函数只允许显式调用，不在模块 import 或 bank resolve 时初始化 CUDA。训练循环复用返回缓存，
    不得每个 batch 重建 mesh/BVH。失败直接抛出；在线主路径不自动回退 trimesh。
    """

    surface_hash = cache.surface_geometry_hash or _owner_surface_geometry_hash(cache.records)
    key = (surface_hash, cache.surface_processing_version, device)
    existing = _WARP_OWNER_CACHE.get(key)
    if existing is not None:
        _WARP_OWNER_CACHE_LEASES[key] = int(_WARP_OWNER_CACHE_LEASES.get(key, 0)) + 1
        return existing

    try:
        import warp as wp
    except Exception as exc:
        raise RuntimeError("Warp is required for the online owner-surface target backend") from exc
    wp.init()
    resolved_device = wp.get_device(device)
    if resolved_device.is_cuda and not wp.is_cuda_available():
        raise RuntimeError(f"Warp CUDA device is unavailable: {device}")

    handles: list[WarpOwnerMeshHandle] = []
    for record in cache.records:
        surface_view = prepare_warp_surface_view(record.surface_mesh, owner_id=record.owner_id)
        vertices = surface_view.vertices  # `[V,3]` float32，和 Warp 实际计算完全一致
        faces_2d = surface_view.faces  # `[F_valid,3]`，已删除 float32 退化面
        faces = faces_2d.reshape(-1)
        points = wp.array(vertices, dtype=wp.vec3, device=device)
        indices = wp.array(faces, dtype=wp.int32, device=device)
        source_face_indices = wp.array(surface_view.source_face_indices, dtype=wp.int32, device=device)
        triangles = vertices[faces_2d]  # `[F_valid,3,3]`，float32 上传视图
        doubled_area = np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
            axis=-1,
        )
        opposite_edges = np.stack(
            (
                np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=-1),
                np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=-1),
                np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=-1),
            ),
            axis=-1,
        )
        if np.any(opposite_edges <= 0.0) or np.any(doubled_area <= 0.0):
            raise RuntimeError(f"owner '{record.owner_id}' float32 surface filtering left a degenerate triangle")
        face_altitudes = wp.array(
            (doubled_area[:, None] / opposite_edges).astype(np.float32),
            dtype=wp.vec3,
            device=device,
        )  # $h_i=2A/|e_i|$，与重心坐标乘积得到到对边的物理距离
        mesh = wp.Mesh(points, indices)
        handles.append(
            WarpOwnerMeshHandle(
                owner_id=record.owner_id,
                mesh=mesh,
                points=points,
                indices=indices,
                source_face_indices=source_face_indices,
                face_altitudes=face_altitudes,
                face_count=len(faces_2d),
                surface_audit=surface_view.audit,
            )
        )
    result = WarpOwnerGeometryCache(
        asset_id=cache.asset_id,
        asset_content_hash=cache.asset_content_hash,
        surface_geometry_hash=surface_hash,
        surface_processing_version=cache.surface_processing_version,
        device=device,
        handles=tuple(handles),
    )
    _WARP_OWNER_CACHE[key] = result
    _WARP_OWNER_CACHE_LEASES[key] = 1
    return result


def release_warp_owner_geometry_cache(cache: WarpOwnerGeometryCache) -> bool:
    r"""释放一项 GPU owner cache lease，并在归零时驱逐全局强引用。

    Returns:
        bool: 本次释放是否使 lease 归零并完成 registry eviction。调用方随后还需
            丢弃自身 `WarpOwnerGeometryCache`/device-state 引用，Python 引用计数归零时
            `wp.Mesh.__del__` 才会销毁底层 BVH。
    """

    key = (cache.surface_geometry_hash, cache.surface_processing_version, cache.device)
    leases = _WARP_OWNER_CACHE_LEASES.get(key)
    if leases is None:
        raise KeyError("Warp owner geometry cache is not resident or has already been released")
    if leases > 1:
        _WARP_OWNER_CACHE_LEASES[key] = leases - 1
        return False
    _WARP_OWNER_CACHE_LEASES.pop(key, None)
    _WARP_OWNER_CACHE.pop(key, None)
    return True


def warp_owner_geometry_cache_stats() -> dict[str, int]:
    r"""返回当前全局 GPU owner cache 的 entry、owner BVH 与 lease 数。"""

    return {
        "entry_count": len(_WARP_OWNER_CACHE),
        "owner_mesh_count": sum(len(cache.handles) for cache in _WARP_OWNER_CACHE.values()),
        "lease_count": sum(_WARP_OWNER_CACHE_LEASES.values()),
    }


def geometry_identity(
    semantics: HandGeometrySemanticsCfg,
    spec: EmbodimentGeometrySpec,
    cache: OwnerGeometryCache,
) -> GeometryIdentity:
    r"""计算不含路径、时间戳和资产 ID 的物理/构型双重身份。

    物理 hash 对应映射 $(q,x,g)\mapsto d_g(x;q)$，所以包含定义 owner 运动与
    表面的全部量；limits 只改变训练会从哪个 $q$ 域采样，不改变该映射。调用方应使用
    同一精度 lowering（正式 manifest 使用 CPU float64），避免 dtype 选择混入身份。

    Args:
        semantics (HandGeometrySemanticsCfg): 已验证的 frame、owner 与 joint 名义语义。
        spec (EmbodimentGeometrySpec): 对应资产的 CPU 运动学 lowering。
        cache (OwnerGeometryCache): 已焊接/并集后的实际 owner surface。

    Returns:
        GeometryIdentity: 物理映射 hash 与 configuration-domain hash。
    """

    if spec.joint_limits is None:
        raise ValueError("geometry identity requires explicit joint limits for configuration-domain provenance")
    if tuple(spec.joint_names) != tuple(semantics.active_joint_names):
        raise ValueError("geometry identity joint axis does not match geometry semantics")
    if tuple(spec.owner_ids) != tuple(owner.owner_id for owner in semantics.owners):
        raise ValueError("geometry identity owner axis does not match geometry semantics")
    surface_hash = cache.surface_geometry_hash or _owner_surface_geometry_hash(cache.records)

    physical = hashlib.sha256()
    physical.update(b"physical-geometry-v1\0")
    _hash_strings(physical, spec.joint_names)  # q 坐标身份与规范顺序
    _hash_strings(physical, spec.owner_ids)  # owner/field 输出身份与规范顺序
    _hash_strings(physical, tuple(owner.role for owner in semantics.owners))  # PALM/JOINT/TIP 语义
    _hash_tensor(physical, spec.space_screws, floating=True)  # `[N_J,6]`，{h} 空间旋量
    _hash_tensor(physical, spec.q_home, floating=True)  # `[N_J]`，rad reference
    _hash_tensor(physical, spec.owner_home_transforms, floating=True)  # `[G,4,4]`，owner local -> {h}
    _hash_tensor(physical, spec.owner_ancestor_mask, floating=False)  # `[G,N_J]` 动力学结构零
    _hash_tensor(physical, spec.joint_ancestor_mask, floating=False)  # `[N_J,N_J]` 分支结构
    for optional in (
        spec.owner_parent_indices,
        spec.owner_graph_shortest,
        spec.owner_graph_parent,
        spec.owner_graph_child,
        spec.component_owner_indices,
    ):
        _hash_optional_tensor(physical, optional, floating=False)
    _hash_optional_tensor(physical, spec.component_owner_local_transforms, floating=True)
    physical.update(bytes.fromhex(surface_hash))  # 实际三角表面，已排除 mesh path/asset ID

    domain = hashlib.sha256()
    domain.update(b"configuration-domain-v1\0")
    _hash_strings(domain, spec.joint_names)  # limit 每一行对应的 q 坐标
    _hash_tensor(domain, spec.joint_limits, floating=True)  # `[N_J,2]`，rad
    return GeometryIdentity(
        physical_geometry_hash=physical.hexdigest(),
        configuration_domain_hash=domain.hexdigest(),
    )


def prepare_warp_surface_view(
    surface_mesh: trimesh.Trimesh,
    *,
    owner_id: str,
    max_area_loss_fraction: float = 1.0e-8,
) -> WarpSurfaceView:
    r"""构造 Warp 实际消费的 float32 surface，并审计退化面面积损失。

    一个 CPU float64 三角形可能因顶点间距低于 float32 分辨率而坍缩。过滤判据必须
    在转换后的实际顶点上计算：任一边长为零或叉积面积为零即删除。被删面的物理面积
    仍用 CPU float64 三角形计算，要求

    $$
    \frac{\sum_{f\in\mathcal D} A_f^{64}}{\sum_f A_f^{64}}\le 10^{-8}.
    $$

    该操作只定义 GPU surface view，不修改资产文件、CPU surface 或可选 solid。
    """

    if not 0.0 <= max_area_loss_fraction <= 1.0:
        raise ValueError("max_area_loss_fraction must lie in [0,1]")
    _require_surface(surface_mesh, context=f"owner '{owner_id}' surface")
    vertices64 = np.asarray(surface_mesh.vertices, dtype=np.float64)
    faces = np.asarray(surface_mesh.faces, dtype=np.int32)
    triangles64 = vertices64[faces]
    cross64 = np.cross(triangles64[:, 1] - triangles64[:, 0], triangles64[:, 2] - triangles64[:, 0])
    area64 = 0.5 * np.linalg.norm(cross64, axis=-1)  # `[F]`，CPU 物理面积 $\mathrm{m}^2$
    total_area = float(area64.sum())
    if not np.isfinite(total_area) or total_area <= 0.0:
        raise ValueError(f"owner '{owner_id}' surface has zero or non-finite total area")

    vertices32 = vertices64.astype(np.float32)
    triangles32 = vertices32[faces]
    edges32 = np.stack(
        (
            triangles32[:, 1] - triangles32[:, 0],
            triangles32[:, 2] - triangles32[:, 1],
            triangles32[:, 0] - triangles32[:, 2],
        ),
        axis=1,
    )
    squared_edge_lengths32 = np.sum(edges32 * edges32, axis=-1)  # `[F,3]`，$\mathrm{m}^2$
    doubled_area32 = np.linalg.norm(np.cross(edges32[:, 0], -edges32[:, 2]), axis=-1)  # `[F]`，$2A$
    valid = np.all(squared_edge_lengths32 > 0.0, axis=-1) & (doubled_area32 > 0.0)
    removed_area = float(area64[~valid].sum())
    removed_fraction = removed_area / total_area
    if removed_fraction > max_area_loss_fraction:
        raise ValueError(
            f"owner '{owner_id}' float32 surface area-loss budget exceeded: "
            f"removed_fraction={removed_fraction:.12g}, budget={max_area_loss_fraction:.12g}"
        )
    if not np.any(valid):
        raise ValueError(f"owner '{owner_id}' has no valid float32 triangles after surface filtering")

    audit = WarpSurfaceAudit(
        input_face_count=len(faces),
        output_face_count=int(valid.sum()),
        removed_face_count=int((~valid).sum()),
        input_area_m2=total_area,
        removed_area_m2=removed_area,
        removed_area_fraction=removed_fraction,
    )
    return WarpSurfaceView(
        vertices=np.ascontiguousarray(vertices32),
        faces=np.ascontiguousarray(faces[valid]),
        source_face_indices=np.ascontiguousarray(np.flatnonzero(valid).astype(np.int32)),
        audit=audit,
    )


def strict_owner_union(meshes: list[trimesh.Trimesh], *, owner_id: str) -> trimesh.Trimesh:
    r"""对一个 owner 的闭合 solids 做严格 Manifold Boolean union。

    单组件直接复制，不进行无意义重网格；多组件 union 输入与输出都必须满足 ``is_volume``。
    """

    if not meshes:
        raise ValueError(f"owner '{owner_id}' has no collision solids")
    for component_index, mesh in enumerate(meshes):
        _require_volume(mesh, context=f"owner '{owner_id}' component[{component_index}]")
    if len(meshes) == 1:
        return meshes[0].copy()

    try:
        union = trimesh.boolean.union(meshes, engine="manifold", check_volume=True)
    except Exception as exc:
        raise ValueError(f"strict Boolean union failed for owner '{owner_id}': {exc}") from exc
    if not isinstance(union, trimesh.Trimesh):
        raise ValueError(f"strict Boolean union for owner '{owner_id}' did not return a single Trimesh")
    union.remove_unreferenced_vertices()
    _require_volume(union, context=f"owner '{owner_id}' union")
    return union


def _concatenate_owner_surfaces(meshes: list[trimesh.Trimesh], *, owner_id: str) -> trimesh.Trimesh:
    r"""把含开放 component 的 owner 表面组成一个可查询三角集合。

    开放 surface 没有内部集合，因而不能送入 solid Boolean。这里的几何对象是
    $S_g=\bigcup_c S_{g,c}$：最近点对三角集合取下确界；即使三角形相交或重叠，
    unsigned distance 仍有定义。函数不补洞、不删除所谓 buried faces。
    """

    if not meshes:
        raise ValueError(f"owner '{owner_id}' has no collision surfaces")
    surface = meshes[0].copy() if len(meshes) == 1 else trimesh.util.concatenate(tuple(meshes))
    surface.remove_unreferenced_vertices()
    _require_surface(surface, context=f"owner '{owner_id}' concatenated surface")
    return surface


def sample_owner_home_surfaces(
    cache: OwnerGeometryCache,
    *,
    points_per_owner: int,
    sampling_seed: int,
    oversample_factor: int = 8,
) -> HomeSurfaceSamples:
    r"""从每个 owner 的真实并集边界按面积采候选，再做确定性最远点均匀化。

    面积采样保证不同大小三角面按物理面积进入候选池；最远点子采样只改善有限预算的覆盖均匀性，
    不引入实体内部点或表面法向作为模型输入。
    """

    if points_per_owner < 1:
        raise ValueError("points_per_owner must be positive")
    if oversample_factor < 1:
        raise ValueError("oversample_factor must be positive")

    sampled_points: list[np.ndarray] = []
    sampled_faces: list[np.ndarray] = []
    sampled_barycentric: list[np.ndarray] = []
    for record in cache.records:
        owner_seed = _stable_owner_seed(sampling_seed, record.owner_id)
        candidate_count = points_per_owner * oversample_factor
        sampled_surface = trimesh.sample.sample_surface(
            record.surface_mesh,
            candidate_count,
            seed=owner_seed,
        )
        candidates = sampled_surface[0]  # trimesh 版本可附加颜色；前两项稳定为 points/face indices
        candidate_faces = sampled_surface[1]
        selected = _farthest_point_indices(candidates, points_per_owner)
        points = candidates[selected]
        faces = candidate_faces[selected]
        triangles = record.surface_mesh.triangles[faces]
        barycentric = trimesh.triangles.points_to_barycentric(triangles, points)
        sampled_points.append(points)
        sampled_faces.append(faces)
        sampled_barycentric.append(barycentric)

    return HomeSurfaceSamples(
        owner_ids=tuple(record.owner_id for record in cache.records),
        points_owner_local_m=np.stack(sampled_points, axis=0),
        face_indices=np.stack(sampled_faces, axis=0),
        barycentric=np.stack(sampled_barycentric, axis=0),
        sampling_seed=sampling_seed,
        oversample_factor=oversample_factor,
    )


def sample_palm_anchor_supports(
    cache: OwnerGeometryCache,
    semantics: HandGeometrySemanticsCfg,
    spec: EmbodimentGeometrySpec,
    *,
    anchors_per_finger: int,
    sampling_seed: int,
    radial_support_radius_m: float = 0.05,
    radial_decay_scale_m: float | None = None,
    surface_fraction: float = 0.5,
) -> AnchorSamples:
    r"""从每根手指挂载 seed 的径向衰减 PALM 支持域采 surface/interior anchors。

    对 seed-local 距离 $r=\|p-p_{seed}\|_2$，支持域限制为 $r\le R_a$，候选接受权重为：

    $$
    w_a(r)=\exp\left(-\frac{r^2}{2\tau_a^2}\right).
    $$

    surface proposal 按 PALM union 的真实三角形面积采样，interior proposal 在
    ``sphere(seed,R_a) ∩ palm solid`` 内按体积采样；径向接受后继续用确定性最远点选择，兼顾
    挂载点附近的概率偏好与有限 $K$ 下的点间分离。seed/finger 只属于采样 provenance，不进入网络。

    数值锚点：$R_a=0.05\,\mathrm m$、$\tau_a=R_a/2=0.025\,\mathrm m$、每指 10 点、
    surface/interior 各半。它们是首个可运行主线配置，不是已经由消融接受的算法常数。
    """

    if anchors_per_finger < 1:
        raise ValueError("anchors_per_finger must be positive")
    radial_decay_scale_m = (  # 独立调用覆盖 $R_a$ 时，未声明的 $\tau_a$ 始终保持 $R_a/2$ 关系
        0.5 * radial_support_radius_m if radial_decay_scale_m is None else radial_decay_scale_m
    )
    if radial_support_radius_m <= 0.0 or not 0.0 < radial_decay_scale_m <= radial_support_radius_m:
        raise ValueError("anchor support radius and radial decay scale must satisfy 0 < tau_a <= R_a")
    if not 0.0 <= surface_fraction <= 1.0:
        raise ValueError("surface_fraction must lie in [0,1]")
    if spec.owner_ids != tuple(owner.owner_id for owner in semantics.owners):
        raise ValueError("anchor semantics/spec owner axes do not match")
    palm_index = next(owner.owner_index for owner in semantics.owners if owner.owner_id == "palm")
    palm_record = cache.records[palm_index]
    palm_transform = spec.owner_home_transforms[palm_index].detach().cpu().numpy()
    hand_rotation = np.asarray(semantics.asset_to_hand_rotation, dtype=np.float64).reshape(3, 3)
    hand_translation = np.asarray(semantics.asset_to_hand_translation_m, dtype=np.float64)
    inverse_palm = np.linalg.inv(palm_transform)

    all_points: list[np.ndarray] = []
    all_finger_names: list[str] = []
    all_seed_ids: list[str] = []
    all_surface_mask: list[bool] = []
    surface_count = int(round(anchors_per_finger * surface_fraction))
    interior_count = anchors_per_finger - surface_count
    for seed in semantics.anchor_seeds:
        seed_hand = hand_rotation @ np.asarray(seed.position_a_m) + hand_translation
        seed_homogeneous = np.append(seed_hand, 1.0)
        seed_local = (inverse_palm @ seed_homogeneous)[:3]
        sampled_surface = trimesh.sample.sample_surface(
            palm_record.surface_mesh,
            max(anchors_per_finger * 64, 256),
            seed=_stable_owner_seed(sampling_seed, seed.seed_id),
        )
        local_surface = sampled_surface[0]  # 可选第三返回值不属于 anchor 几何合同
        local_surface = _within_radius(local_surface, seed_local, radial_support_radius_m)
        local_surface = _radial_decay_candidates(  # 面积 proposal 经 $w_a(r)$ 接受后偏向真实挂载 seed
            local_surface,
            seed_local,
            radial_decay_scale_m,
            seed=_stable_owner_seed(sampling_seed + 2, seed.seed_id),
        )
        if len(local_surface) < surface_count:
            raise ValueError(
                f"anchor seed '{seed.seed_id}' has only {len(local_surface)} palm surface candidates "
                f"after radial decay within radius {radial_support_radius_m} m; need {surface_count}"
            )
        selected_surface = (
            local_surface[_farthest_point_indices(local_surface, surface_count)] if surface_count else np.empty((0, 3))
        )

        if interior_count and palm_record.solid_mesh is None:
            raise ValueError(
                "palm interior anchors require OwnerSurfaceRecord.solid_mesh; "
                "an open surface cannot define inside support"
            )
        interior_candidates = max(anchors_per_finger * 64, 256)  # 大候选池使径向接受后仍可做覆盖选择
        local_interior = _sample_interior_support(
            palm_record.solid_mesh,
            seed_local,
            radial_support_radius_m,
            interior_candidates if interior_count else 0,
            seed=_stable_owner_seed(sampling_seed + 1, seed.seed_id),
        ) if palm_record.solid_mesh is not None else np.empty((0, 3))
        local_interior = _radial_decay_candidates(  # 体积 proposal 使用与 surface 相同的物理衰减尺度
            local_interior,
            seed_local,
            radial_decay_scale_m,
            seed=_stable_owner_seed(sampling_seed + 3, seed.seed_id),
        )
        if len(local_interior) < interior_count:
            raise ValueError(
                f"anchor seed '{seed.seed_id}' has only {len(local_interior)} palm interior candidates "
                f"after radial decay within radius {radial_support_radius_m} m; need {interior_count}"
            )
        selected_interior = (
            local_interior[_farthest_point_indices(local_interior, interior_count)]
            if interior_count
            else np.empty((0, 3))
        )
        local_points = np.concatenate((selected_surface, selected_interior), axis=0)
        hand_points = (palm_transform @ np.concatenate((local_points, np.ones((len(local_points), 1))), axis=1).T).T[:, :3]
        all_points.append(hand_points)
        all_finger_names.extend([seed.finger_name] * anchors_per_finger)
        all_seed_ids.extend([seed.seed_id] * anchors_per_finger)
        all_surface_mask.extend([True] * surface_count + [False] * interior_count)

    return AnchorSamples(
        anchors_hand_m=np.concatenate(all_points, axis=0),
        finger_names=tuple(all_finger_names),
        seed_ids=tuple(all_seed_ids),
        surface_mask=np.asarray(all_surface_mask, dtype=bool),
        radial_support_radius_m=float(radial_support_radius_m),
        radial_decay_scale_m=float(radial_decay_scale_m),
        surface_fraction=float(surface_fraction),
        sampling_seed=int(sampling_seed),
        algorithm_version="palm-seed-radial-gaussian-fps-v1",
    )


def _component_mesh(
    component: CollisionComponentSemanticsCfg,
    *,
    container: HandContainer,
) -> trimesh.Trimesh:
    """把 sidecar 几何 payload 恢复为 collision-local 三角网格。"""

    payload = component.geometry_payload
    kind = component.geometry_kind
    if kind == "box":
        mesh = trimesh.creation.box(extents=np.asarray(payload["size"], dtype=np.float64))
    elif kind == "cylinder":
        mesh = trimesh.creation.cylinder(
            radius=float(payload["radius"]),
            height=float(payload["length"]),
            sections=64,
        )
    elif kind == "elliptic_cylinder":
        mesh = _elliptic_cylinder_mesh(
            radius_x=float(payload["radius_x"]),
            radius_z=float(payload["radius_z"]),
            length=float(payload["length"]),
        )
    elif kind == "sphere":
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=float(payload["radius"]))
    elif kind == "mesh":
        mesh_path = _resolve_component_mesh_path(str(payload["file_path"]), container=container)
        # STL 常逐面复制相同坐标顶点；`process=True` 的确定性焊接只恢复拓扑邻接，
        # 不补洞、不构造 hull，也不改变真实开放边界。后续仍独立判断 solid 可用性。
        loaded = trimesh.load(mesh_path, force="mesh", process=True)
        if not isinstance(loaded, trimesh.Trimesh):
            raise ValueError(f"mesh component '{component.component_id}' did not load as one Trimesh")
        mesh = loaded.copy()
        mesh.apply_scale(np.asarray(payload.get("scale", (1.0, 1.0, 1.0)), dtype=np.float64))
    else:
        raise ValueError(f"unsupported collision geometry kind={kind!r} for '{component.component_id}'")
    _clean_surface_topology(mesh)  # 所有 primitive/mesh 统一去重，保证 hash 与 Warp 输入稳定
    _require_surface(mesh, context=f"component '{component.component_id}'")
    return mesh


def _elliptic_cylinder_mesh(*, radius_x: float, radius_z: float, length: float) -> trimesh.Trimesh:
    """复现 exporter 的局部 +y 主轴椭圆柱。"""

    mesh = trimesh.creation.cylinder(radius=1.0, height=1.0, sections=64)
    source = mesh.vertices.copy()  # trimesh canonical cylinder 主轴为 z
    mesh.vertices[:, 0] = radius_x * source[:, 0]
    mesh.vertices[:, 1] = length * source[:, 2]
    mesh.vertices[:, 2] = -radius_z * source[:, 1]  # 负号保持变换手性与面绕序
    return mesh


def _resolve_component_mesh_path(raw_path: str, *, container: HandContainer) -> Path:
    """通过 container 虚拟视图解析 absolute/URDF-relative mesh 路径。"""

    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        if not candidate.is_file():
            raise FileNotFoundError(f"collision mesh does not exist: {candidate}")
        return candidate

    for mesh_ref in container.mesh_refs:
        if raw_path == mesh_ref.raw_uri or PurePosixPath(raw_path).name == mesh_ref.virtual_path.name:
            return mesh_ref.real_path
    virtual_candidate = PurePosixPath("meshes") / PurePosixPath(raw_path).name
    try:
        return container.real_path(virtual_candidate)
    except KeyError as exc:
        raise FileNotFoundError(f"collision mesh path {raw_path!r} is not part of asset {container.asset_id!r}") from exc


def _clean_surface_topology(mesh: trimesh.Trimesh) -> None:
    r"""焊接重复顶点并删除重复、精确零面积和无引用元素。

    这里只去除不携带几何测度的表示冗余。非零面积三角形无论是否形成 watertight
    surface 都保留；因此开放槽、边界边和真实薄片不会被修复或删除。
    """

    mesh.merge_vertices()  # 坐标相同顶点共享拓扑索引，恢复 STL 闭合体邻接
    if len(mesh.faces):
        unique = mesh.unique_faces()  # 相同顶点集合的重复三角形不增加表面集合
        mesh.update_faces(unique)
    if len(mesh.faces):
        triangles = np.asarray(mesh.triangles, dtype=np.float64)
        doubled_area = np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
            axis=-1,
        )
        mesh.update_faces(np.isfinite(doubled_area) & (doubled_area > 0.0))
    mesh.remove_unreferenced_vertices()


def _require_surface(mesh: trimesh.Trimesh, *, context: str) -> None:
    r"""要求有限、非空且含正面积三角形的 surface，不要求 watertight。"""

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or faces.ndim != 2 or faces.shape[1:] != (3,):
        raise ValueError(f"{context} must be a triangle surface")
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError(f"{context} must contain vertices and faces")
    if not np.all(np.isfinite(vertices)):
        raise ValueError(f"{context} contains non-finite vertices")
    triangles = vertices[faces]
    doubled_area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=-1,
    )
    if not np.all(np.isfinite(doubled_area)) or np.any(doubled_area <= 0.0):
        raise ValueError(f"{context} contains zero-area or non-finite triangles")


def _is_volume(mesh: trimesh.Trimesh) -> bool:
    r"""判断 surface 是否同时提供可信 inside/outside solid 语义。"""

    return bool(mesh.is_watertight and mesh.is_winding_consistent and mesh.is_volume)


def _require_volume(mesh: trimesh.Trimesh, *, context: str) -> None:
    """拒绝非闭合、绕序不一致或零体积网格。"""

    if not _is_volume(mesh):
        raise ValueError(
            f"{context} must be a watertight consistently-wound volume; "
            f"watertight={mesh.is_watertight}, winding={mesh.is_winding_consistent}, volume={mesh.is_volume}"
        )


def _owner_surface_geometry_hash(records: tuple[OwnerSurfaceRecord, ...]) -> str:
    r"""哈希 owner 顺序、ID 与实际 float64 surface vertices/faces。

    路径、时间戳、asset ID、joint limits 与 provenance 不参与；相同物理表面在相同
    owner-local frame 下得到相同 cache identity。该 hash 当前只覆盖 surface cache，
    完整 split 所需运动学/`q_home`/component transform hash 在 assets 层另行定义。
    """

    digest = hashlib.sha256()
    digest.update(b"owner-surface-v2\0")
    for record in records:
        vertices = np.ascontiguousarray(record.surface_mesh.vertices, dtype="<f8")
        faces = np.ascontiguousarray(record.surface_mesh.faces, dtype="<i4")
        digest.update(record.owner_id.encode("utf-8") + b"\0")
        digest.update(np.asarray(vertices.shape, dtype="<i8").tobytes())
        digest.update(vertices.tobytes())
        digest.update(np.asarray(faces.shape, dtype="<i8").tobytes())
        digest.update(faces.tobytes())
    return digest.hexdigest()


def _hash_strings(digest: Any, values: tuple[str, ...]) -> None:
    r"""以长度前缀编码字符串轴，避免连接歧义。"""

    digest.update(len(values).to_bytes(8, "little", signed=False))
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little", signed=False))
        digest.update(encoded)


def _hash_tensor(digest: Any, value: Any, *, floating: bool) -> None:
    r"""把 tensor 规约成 CPU little-endian 连续数组后写入 shape、dtype 与数值。"""

    tensor = value.detach().cpu()
    array = np.asarray(tensor, dtype="<f8" if floating else "<i8")
    array = np.ascontiguousarray(array)
    digest.update(b"f8" if floating else b"i8")
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())


def _hash_optional_tensor(digest: Any, value: Any | None, *, floating: bool) -> None:
    r"""为可选物理关系张量加入 presence bit 后复用规范 tensor 编码。"""

    if value is None:
        digest.update(b"\x00")
        return
    digest.update(b"\x01")
    _hash_tensor(digest, value, floating=floating)


def _stable_owner_seed(seed: int, owner_id: str) -> int:
    """把全局种子和稳定 owner ID 混成 NumPy 接受的 32-bit 种子。"""

    owner_hash = 2166136261
    for byte in owner_id.encode("utf-8"):
        owner_hash = ((owner_hash ^ byte) * 16777619) & 0xFFFFFFFF
    return (int(seed) ^ owner_hash) & 0xFFFFFFFF


def _within_radius(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """返回球形支持邻域内的候选点。"""

    return points[np.linalg.norm(points - center[None, :], axis=-1) <= radius]


def _radial_decay_candidates(
    points: np.ndarray,
    center: np.ndarray,
    scale: float,
    *,
    seed: int,
) -> np.ndarray:
    r"""按截断 Gaussian 径向权重接受 PALM surface/volume proposal。

    输入 proposal 已经由 ``_within_radius`` 或 ``_sample_interior_support`` 限制在支持球内；本函数
    只实施 $w_a(r)=\exp(-r^2/(2\tau_a^2))$。由于 $w_a(0)=1$，可直接把权重作为接受概率，
    无需未知归一化常数。后续最远点选择负责有限 anchor 数下的点间分离。

    Args:
        points (np.ndarray): ``[N,3]`` PALM-local surface 或 interior proposal，单位 m。
        center (np.ndarray): ``[3]`` first-active mount seed，PALM-local，单位 m。
        scale (float): 截断 Gaussian 衰减尺度 $\tau_a>0$，单位 m。
        seed (int): 独立、可复现的候选接受随机种子。

    Returns:
        np.ndarray: 保持原 proposal 顺序的接受点，形状 ``[N_{accept},3]``。
    """

    if len(points) == 0:  # 空 surface/solid proposal 原样返回，由 caller 给出带 seed 的失败信息
        return points
    squared_radius = np.sum((points - center[None, :]) ** 2, axis=-1)  # $r^2$，单位 $\mathrm m^2$
    acceptance = np.exp(-squared_radius / (2.0 * scale * scale))  # $w_a(r)\in(0,1]$，无量纲
    rng = np.random.default_rng(seed)  # 每个 seed/source 独立，修改一类候选不扰动另一类
    return points[rng.random(len(points)) < acceptance]  # rejection sampling 保留原始面积/体积基测度


def _sample_interior_support(
    mesh: trimesh.Trimesh,
    center: np.ndarray,
    radius: float,
    count: int,
    *,
    seed: int,
) -> np.ndarray:
    """在 sphere∩solid 中 rejection sample 固定数量的内部点。"""

    if count == 0:
        return np.empty((0, 3), dtype=np.float64)
    rng = np.random.default_rng(seed)
    accepted: list[np.ndarray] = []
    attempts = 0
    max_attempts = max(10000, count * 10000)
    while sum(len(batch) for batch in accepted) < count and attempts < max_attempts:
        batch_size = max(256, (count - sum(len(batch) for batch in accepted)) * 32)
        candidate = rng.uniform(-radius, radius, size=(batch_size, 3)) + center[None, :]
        candidate = candidate[np.linalg.norm(candidate - center[None, :], axis=-1) <= radius]
        if len(candidate):
            inside = mesh.contains(candidate)
            accepted.append(candidate[inside])
        attempts += batch_size
    result = np.concatenate(accepted, axis=0) if accepted else np.empty((0, 3), dtype=np.float64)
    if len(result) < count:
        raise ValueError(f"palm solid has only {len(result)} interior support candidates; need {count}")
    return result[:count]


def _farthest_point_indices(points: np.ndarray, count: int) -> np.ndarray:
    """在面积候选池中做确定性欧氏最远点子采样。"""

    if count > len(points):
        raise ValueError(f"cannot select {count} points from {len(points)} candidates")
    centroid = points.mean(axis=0)
    first = int(np.argmax(np.sum((points - centroid) ** 2, axis=-1)))
    selected = np.empty(count, dtype=np.int64)
    selected[0] = first
    minimum_squared_distance = np.sum((points - points[first]) ** 2, axis=-1)
    for output_index in range(1, count):
        next_index = int(np.argmax(minimum_squared_distance))
        selected[output_index] = next_index
        next_distance = np.sum((points - points[next_index]) ** 2, axis=-1)
        minimum_squared_distance = np.minimum(minimum_squared_distance, next_distance)
    return selected


__all__ = [
    "HomeSurfaceSamples",
    "AnchorSamples",
    "GeometryIdentity",
    "OwnerGeometryCache",
    "OwnerSurfaceRecord",
    "WarpOwnerGeometryCache",
    "WarpOwnerMeshHandle",
    "WarpSurfaceAudit",
    "WarpSurfaceView",
    "materialize_owner_geometry_cache",
    "materialize_warp_owner_geometry_cache",
    "prepare_warp_surface_view",
    "geometry_identity",
    "release_warp_owner_geometry_cache",
    "sample_owner_home_surfaces",
    "sample_palm_anchor_supports",
    "strict_owner_union",
    "warp_owner_geometry_cache_stats",
]
