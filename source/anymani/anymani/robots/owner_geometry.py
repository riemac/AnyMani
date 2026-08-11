r"""按 owner 构造真实 collision-solid 并集与基准表面采样缓存。

输入是 ``assets.bank.HandContainer`` 的类型化几何语义，以及 ``robots`` 已 lower 的
``EmbodimentGeometrySpec``。每个碰撞片先从自身局部坐标严格变换到 owner reference link；同 owner
有多个 solid 时调用 ``trimesh.boolean.union(..., engine="manifold")``。输入/输出必须是闭合体，
任何 Boolean 失败均拒绝，不使用 convex hull、包围盒或删除近似重叠面作为替代真值。

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

`OwnerSurfaceRecord.mesh` 必须满足 watertight、winding-consistent 和 `is_volume`。这一要求既是
inside/outside 的必要条件，也是 manifold Boolean 的输入前提。单 component 仍要做 volume 检查，
只是无需 union；多 component 必须检查 union 输出，不能因为输入合法就假设后端输出合法。

表面采样的两步测度不能混淆：第一步按三角面面积采候选，使大面按物理面积获得更多样本；第二步
在候选中做确定性最远点子采样，改善有限 $M_g$ 预算的空间覆盖。输出保存 owner-local point、
union face index 与 barycentric 坐标，法向只供 query shell 采样，不进入 retained encoder。改变
mesh tessellation 或 oversample factor 必须通过 cache version/provenance 明确记录。

anchor 与 home surface 的语义不同。home points 永远在 owner boundary 上，不能混入 solid interior；
anchor 允许从 palm surface/interior 支持取得，因为 anchor 的作用是提供 mount-conditioned
reference constellation，而不是声称它们是 collision surface。每根 finger 的 seed 只进入
`finger_names/seed_ids` provenance；`anchors_hand_m` 输出成完整无序 $K$ 集合。

Warp cache 持有 `wp.Mesh`、points、indices、face altitudes 的强引用，避免底层 device array 被
Python 垃圾回收。cache key 至少包含 asset content hash 与 device；GPU BVH 在训练生命周期内只构造
一次。在线 Warp 失败必须暴露为 backend error，不能静默切换 trimesh；CPU reference 如需运行，
由显式 oracle 命令选择并单独记录吞吐/数值差异。

当前 Warp face margin 定义为最近投影点到选中三角形三条边的最小物理距离：若重心坐标为
$(b_0,b_1,b_2)$、对应高为 $(h_0,h_1,h_2)$，则

$$
m_{feature}=\min_i b_i h_i.
$$

它只证明当前三角面 feature 的局部 interior 余量，不等价于全局第二近表面间隔；target backend
必须把这一点写进 provenance，并在需要严格 medial-axis 证据时使用独立 reference。
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path, PurePosixPath

import numpy as np
import trimesh

from anymani.assets.asset_schema_geometry import (
    CollisionComponentSemanticsCfg,
    HandGeometrySemanticsCfg,
)
from anymani.assets.bank import HandContainer

from .geometry_kinematics import EmbodimentGeometrySpec


@dataclass(frozen=True)
class OwnerSurfaceRecord:
    r"""一个 owner 的真实并集边界及其碰撞片 provenance。"""

    owner_id: str  # 与语义实体轴一致的稳定 ID
    owner_index: int  # 规范实体轴索引
    role: str  # palm/joint/tip
    component_ids: tuple[str, ...]  # 参与并集的碰撞片
    mesh: trimesh.Trimesh  # owner reference link 中的闭合并集边界，单位 m
    boolean_applied: bool  # 单片 owner 为 False，多片严格并集为 True


@dataclass(frozen=True)
class OwnerGeometryCache:
    r"""一项资产可复用的 owner-local 静态几何缓存。"""

    asset_id: str
    asset_content_hash: str
    boolean_backend: str
    records: tuple[OwnerSurfaceRecord, ...]


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
    surface_fraction: float
    sampling_seed: int


@dataclass(frozen=True)
class WarpOwnerMeshHandle:
    r"""一个 owner 的 GPU-resident Warp BVH 及其底层数组强引用。"""

    owner_id: str
    mesh: object  # wp.Mesh；object 避免普通 import 强制加载/初始化 Warp
    points: object  # wp.array[vec3]；必须与 mesh 同生命周期
    indices: object  # wp.array[int32]；必须与 mesh 同生命周期
    face_altitudes: object  # wp.array[vec3]；每个重心坐标对应的三角形高，单位 m
    face_count: int


@dataclass(frozen=True)
class WarpOwnerGeometryCache:
    r"""一项资产全部 owner-local mesh 的 GPU BVH 缓存。"""

    asset_id: str
    asset_content_hash: str
    device: str
    handles: tuple[WarpOwnerMeshHandle, ...]


_WARP_OWNER_CACHE: dict[tuple[str, str], WarpOwnerGeometryCache] = {}
"""按 ``(asset_content_hash, device)`` 复用静态 owner-local BVH。"""


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
        _require_volume(mesh, context=f"component '{component.component_id}'")
        transformed_by_owner[component.owner_id].append(mesh)

    records: list[OwnerSurfaceRecord] = []
    for owner in semantics.owners:
        component_meshes = transformed_by_owner[owner.owner_id]
        if len(component_meshes) != len(owner.component_ids):
            raise ValueError(f"owner '{owner.owner_id}' component materialization is incomplete")
        union = strict_owner_union(component_meshes, owner_id=owner.owner_id)
        records.append(
            OwnerSurfaceRecord(
                owner_id=owner.owner_id,
                owner_index=owner.owner_index,
                role=owner.role,
                component_ids=tuple(owner.component_ids),
                mesh=union,
                boolean_applied=len(component_meshes) > 1,
            )
        )

    # 额外检查 sidecar 组件顺序和运动学 component 轴仍完全一致，避免缓存串包。
    if tuple(component_index_by_id) != tuple(component.component_id for component in semantics.components):
        raise ValueError("component provenance order is inconsistent")
    return OwnerGeometryCache(
        asset_id=container.asset_id,
        asset_content_hash=semantics.content_hash,
        boolean_backend=f"manifold3d=={version('manifold3d')}",
        records=tuple(records),
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

    key = (cache.asset_content_hash, device)
    existing = _WARP_OWNER_CACHE.get(key)
    if existing is not None:
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
        vertices = np.asarray(record.mesh.vertices, dtype=np.float32)
        faces = np.asarray(record.mesh.faces, dtype=np.int32).reshape(-1)
        points = wp.array(vertices, dtype=wp.vec3, device=device)
        indices = wp.array(faces, dtype=wp.int32, device=device)
        triangles = np.asarray(record.mesh.triangles, dtype=np.float32)
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
            raise ValueError(f"owner '{record.owner_id}' union contains a degenerate triangle")
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
                face_altitudes=face_altitudes,
                face_count=len(record.mesh.faces),
            )
        )
    result = WarpOwnerGeometryCache(
        asset_id=cache.asset_id,
        asset_content_hash=cache.asset_content_hash,
        device=device,
        handles=tuple(handles),
    )
    _WARP_OWNER_CACHE[key] = result
    return result


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
            record.mesh,
            candidate_count,
            seed=owner_seed,
        )
        candidates = sampled_surface[0]  # trimesh 版本可附加颜色；前两项稳定为 points/face indices
        candidate_faces = sampled_surface[1]
        selected = _farthest_point_indices(candidates, points_per_owner)
        points = candidates[selected]
        faces = candidate_faces[selected]
        triangles = record.mesh.triangles[faces]
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
    surface_fraction: float = 0.5,
) -> AnchorSamples:
    r"""从每根手指 seed 邻域与 PALM solid 交集采 surface/interior anchors。

    seed 只控制 palm 支持邻域，绝不把 finger index 编入锚点特征。surface 候选按面积抽样，
    interior 候选在 solid 的 AABB rejection sample 中取得；两者随后都以确定性最远点方式均匀化。
    输出锚点统一变换到 `{h}`，并保留 provenance 供审计，不改变网络中的完整 $K$ 集合语义。
    """

    if anchors_per_finger < 1:
        raise ValueError("anchors_per_finger must be positive")
    if radial_support_radius_m <= 0.0:
        raise ValueError("radial_support_radius_m must be positive")
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
            palm_record.mesh,
            max(anchors_per_finger * 64, 256),
            seed=_stable_owner_seed(sampling_seed, seed.seed_id),
        )
        local_surface = sampled_surface[0]  # 可选第三返回值不属于 anchor 几何合同
        local_surface = _within_radius(local_surface, seed_local, radial_support_radius_m)
        if len(local_surface) < surface_count:
            raise ValueError(
                f"anchor seed '{seed.seed_id}' has only {len(local_surface)} palm surface candidates "
                f"within radius {radial_support_radius_m} m; need {surface_count}"
            )
        selected_surface = (
            local_surface[_farthest_point_indices(local_surface, surface_count)] if surface_count else np.empty((0, 3))
        )

        local_interior = _sample_interior_support(
            palm_record.mesh,
            seed_local,
            radial_support_radius_m,
            interior_count,
            seed=_stable_owner_seed(sampling_seed + 1, seed.seed_id),
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
        surface_fraction=float(surface_fraction),
        sampling_seed=int(sampling_seed),
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
        loaded = trimesh.load(mesh_path, force="mesh", process=False)
        if not isinstance(loaded, trimesh.Trimesh):
            raise ValueError(f"mesh component '{component.component_id}' did not load as one Trimesh")
        mesh = loaded.copy()
        mesh.apply_scale(np.asarray(payload.get("scale", (1.0, 1.0, 1.0)), dtype=np.float64))
    else:
        raise ValueError(f"unsupported collision geometry kind={kind!r} for '{component.component_id}'")
    mesh.remove_unreferenced_vertices()
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


def _require_volume(mesh: trimesh.Trimesh, *, context: str) -> None:
    """拒绝非闭合、绕序不一致或零体积网格。"""

    if not mesh.is_watertight or not mesh.is_winding_consistent or not mesh.is_volume:
        raise ValueError(
            f"{context} must be a watertight consistently-wound volume; "
            f"watertight={mesh.is_watertight}, winding={mesh.is_winding_consistent}, volume={mesh.is_volume}"
        )


def _stable_owner_seed(seed: int, owner_id: str) -> int:
    """把全局种子和稳定 owner ID 混成 NumPy 接受的 32-bit 种子。"""

    owner_hash = 2166136261
    for byte in owner_id.encode("utf-8"):
        owner_hash = ((owner_hash ^ byte) * 16777619) & 0xFFFFFFFF
    return (int(seed) ^ owner_hash) & 0xFFFFFFFF


def _within_radius(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """返回球形支持邻域内的候选点。"""

    return points[np.linalg.norm(points - center[None, :], axis=-1) <= radius]


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
    "OwnerGeometryCache",
    "OwnerSurfaceRecord",
    "WarpOwnerGeometryCache",
    "WarpOwnerMeshHandle",
    "materialize_owner_geometry_cache",
    "materialize_warp_owner_geometry_cache",
    "sample_owner_home_surfaces",
    "sample_palm_anchor_supports",
    "strict_owner_union",
]
