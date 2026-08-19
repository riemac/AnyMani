r"""custom mesh collision 的 signed-distance 查询后端。

本模块服务于 post-mutate validator 的一个很具体的科研语义：

1. custom fingertip 已经从规则 primitive 扩展到真实 STL/OBJ mesh；
2. `min_finger_spacing` 不能继续依赖 `sdf_proxy` 这种外包盒近似；
3. validator 需要直接回答“某个 surface point 是否进入另一根 finger 的真实 mesh 内部”。

核心公式
--------

给定 mesh body $M$ 和 world 点 $\mathbf{x}_w$，先把点变到 mesh local frame：

$$
\mathbf{x}_l = R^\top(\mathbf{x}_w-\mathbf{t}).
$$

然后查询 mesh signed distance：

$$
d_M(\mathbf{x}_w)=\operatorname{SDF}_M(\mathbf{x}_l),
$$

符号约定与 primitive SDF 完全一致：

- outside: $d>0$；
- surface: $d=0$；
- inside / penetration: $d<0$。

工程路线
--------

- 默认优先使用 NVIDIA Warp：`wp.Mesh` 建 BVH，`wp.mesh_query_point()` 查询最近点和 sign；
- 若 `backend="auto"` 且 Warp/CUDA 不可用或运行失败，回退到 `trimesh.ProximityQuery`；
- mesh 必须在 `trimesh` 预处理后满足 watertight 与 winding-consistent，否则 fail-hard。

这里刻意不把 `sdf_proxy` 放进 fallback。proxy 是上一版为了让 STL 临时进入 primitive
SDF 的折中近似，不再代表当前 validator 的科研证书。
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..asset_schema_core import MeshGeometryCfg, Vector3
from ._collision_geometry import apply_inverse_pose, apply_pose

if TYPE_CHECKING:
    from ._collision_geometry import CollisionBodyRecord


MeshSdfBackend = Literal["auto", "warp", "trimesh"]
"""mesh signed-distance 后端选择。

`"auto"` 是科研生产默认值：优先 GPU/Warp，失败回 CPU/trimesh；
`"warp"` 用于强制验证 GPU 路线；
`"trimesh"` 用于显式 CPU fallback / debug。
"""


@dataclass
class MeshSdfQueryStats:
    r"""一次 SDF clearance 评估中的 mesh 后端证书信息。

    Attributes:
        requested_backend: 用户配置的后端策略，决定是否允许自动回退。
        actual_backend: 实际查询过 mesh 的后端；无 mesh 时保持 `"none"`。
        mesh_query_count: 参与 signed-distance 查询的 mesh body 次数。
        mesh_sample_count: 为 source mesh 生成 surface samples 的次数。
        fallback_events: GPU/Warp 自动回退 CPU/trimesh 时留下的诊断记录。
    """

    requested_backend: MeshSdfBackend = "auto"
    actual_backend: str = "none"
    mesh_query_count: int = 0
    mesh_sample_count: int = 0
    fallback_events: list[str] = field(default_factory=list)

    def record_backend(self, backend: str) -> None:
        r"""记录一次实际 mesh 查询后端。

        同一轮评估中可能有些 mesh 走 Warp、有些 mesh 因 `auto` 回退走 trimesh。
        此时证书写 `"mixed"`，避免最后一次查询覆盖前面的后端事实。
        """

        if self.actual_backend == "none":
            self.actual_backend = backend
        elif self.actual_backend != backend:
            self.actual_backend = "mixed"

    def to_dict(self) -> dict[str, object]:
        r"""转成 certificate 可序列化字段。"""

        return {
            "requested_backend": self.requested_backend,
            "actual_backend": self.actual_backend,
            "mesh_query_count": self.mesh_query_count,
            "mesh_sample_count": self.mesh_sample_count,
            "fallback_events": list(self.fallback_events),
        }


@dataclass(frozen=True)
class _WarpMeshHandle:
    r"""Warp mesh 运行时句柄。

    `wp.Mesh` 内部只保存 device 侧 BVH 与输入 array 的引用语义，因此这里必须同时持有
    `points`、`indices` 与 `mesh`，避免 Python GC 提前释放底层 buffer。
    """

    mesh: Any
    points: Any
    indices: Any


_WARP_MESH_CACHE_MAXSIZE = 128
_WARP_MESH_CACHE: OrderedDict[
    tuple[str, str, tuple[float, float, float], str],
    _WarpMeshHandle,
] = OrderedDict()
"""Warp BVH cache。

key = `(resolved_path, content_sha256, scale_xyz, device)`。同一个 custom tip 会在批量
post-mutate 中反复出现，缓存能避免每次 pairwise clearance 都重建 BVH；content hash
阻止同路径 mesh 被改写后复用旧 BVH。LRU 上限 128 使单 GPU service 的显存占用不随
连续 mother 数量无界增长。
"""


def is_mesh_body(body: CollisionBodyRecord) -> bool:
    r"""判断 collision body 是否为真实 mesh geometry。"""

    return isinstance(body.geometry, MeshGeometryCfg)


def sample_mesh_surface(body: CollisionBodyRecord, *, sample_count: int) -> list[Vector3]:
    r"""在 mesh local surface 上采样，并变换到 world frame。

    Args:
        body: collision body 记录；其 `geometry` 必须是 `MeshGeometryCfg`。
        sample_count: 每个 mesh body 的 surface 采样点数，默认由上层配置给出 4096。

    Returns:
        list[Vector3]: world-space surface samples，用于近似 $S_F$。

    Raises:
        TypeError: 若 body 不是 mesh。
        ValueError: 若 mesh 文件不存在或 mesh 不是可靠闭合壳。
    """

    if not isinstance(body.geometry, MeshGeometryCfg):
        raise TypeError(f"sample_mesh_surface expects MeshGeometryCfg, got {type(body.geometry).__name__}")

    # 读取的是已经按 `geometry.scale` 烘焙过的 mesh，因此采样点处在 collision local frame。
    mesh_path, mesh_sha256 = _mesh_identity(body.geometry.file_path)
    mesh = _load_checked_trimesh(mesh_path, _scale_tuple(body.geometry.scale), mesh_sha256)

    # `trimesh.sample_surface` 支持 seed；这里用 path/body 的稳定 hash，避免测试和批量生成随机漂移。
    seed = _stable_seed(body.body_path, mesh_path, mesh_sha256, repr(body.geometry.scale))
    from trimesh.sample import sample_surface

    sampled = sample_surface(mesh, sample_count, seed=seed)
    local_points = sampled[0]  # [N, 3]，mesh local 坐标；其余返回值是 face/barycentric diagnostics

    # 将局部 surface samples 推到 palm/world frame，进入 finger-finger clearance 统一坐标系。
    return [apply_pose(body.world_pose, _to_vector3(point)) for point in local_points]


def signed_distance_to_mesh_body(
    point_world: Vector3,
    body: CollisionBodyRecord,
    *,
    backend: MeshSdfBackend,
    device: str,
    stats: MeshSdfQueryStats | None = None,
) -> float:
    r"""查询单个 world 点到 mesh body 的 signed distance。

    这是 batch 版本的标量包装，主要用于 source union surface filtering。
    """

    return float(
        signed_distance_to_mesh_body_batch(
            [point_world],
            body,
            backend=backend,
            device=device,
            stats=stats,
        )[0]
    )


def signed_distance_to_mesh_body_batch(
    points_world: list[Vector3],
    body: CollisionBodyRecord,
    *,
    backend: MeshSdfBackend,
    device: str,
    stats: MeshSdfQueryStats | None = None,
) -> np.ndarray:
    r"""批量查询 world points 到 mesh body 的 signed distance。

    Args:
        points_world: world-space 查询点，形状语义 $N\times3$，单位 meter。
        body: target mesh collision body。
        backend: `"auto"` / `"warp"` / `"trimesh"`。
        device: 上层 SDF 设备解析结果，`"cuda"` 时才尝试 Warp。
        stats: 可选证书统计对象。

    Returns:
        np.ndarray: shape `[N]`，outside positive / inside negative。
    """

    if not isinstance(body.geometry, MeshGeometryCfg):
        raise TypeError(f"signed_distance_to_mesh_body_batch expects MeshGeometryCfg, got {type(body.geometry).__name__}")
    if not points_world:
        return np.empty((0,), dtype=np.float64)

    # mesh SDF 在 local frame 查询；world→local 的变换必须与 primitive SDF 共享同一 pose 约定。
    points_local = np.asarray([apply_inverse_pose(body.world_pose, point) for point in points_world], dtype=np.float32)

    # `backend="auto"` 的默认科研生产路线：CUDA 可用时优先 Warp，否则直接 CPU/trimesh。
    if backend in {"auto", "warp"} and device == "cuda":
        try:
            return _signed_distance_warp(points_local, body.geometry, stats=stats)
        except Exception as exc:
            if backend == "warp":
                raise RuntimeError(f"Warp mesh SDF failed for {body.body_path}: {exc}") from exc
            if stats is not None:
                stats.fallback_events.append(f"{body.body_path}: warp→trimesh fallback because {type(exc).__name__}: {exc}")

    # CPU fallback 不代表低可信度；只要 mesh watertight/winding consistent，signed distance 语义仍成立。
    return _signed_distance_trimesh(points_local, body.geometry, stats=stats)


def _signed_distance_trimesh(
    points_local: np.ndarray,
    geometry: MeshGeometryCfg,
    *,
    stats: MeshSdfQueryStats | None,
) -> np.ndarray:
    r"""用 `trimesh.ProximityQuery` 计算 CPU signed distance。

    `trimesh` 的符号约定与本项目相反：inside positive / outside negative。
    因此返回前必须取负，统一成 outside positive / inside negative。
    """

    mesh_path, mesh_sha256 = _mesh_identity(geometry.file_path)
    query = _get_trimesh_proximity_query(mesh_path, _scale_tuple(geometry.scale), mesh_sha256)
    if stats is not None:
        stats.record_backend("trimesh")
        stats.mesh_query_count += 1
    return -np.asarray(query.signed_distance(points_local), dtype=np.float64)


@lru_cache(maxsize=128)
def _get_trimesh_proximity_query(
    file_path: str,
    scale: tuple[float, float, float],
    content_sha256: str,
):
    r"""缓存 CPU proximity query。

    source union filtering 会对同一个 mesh 进行许多单点查询；若每次都重新构造
    `ProximityQuery`，批量 post-mutate 的 Python 开销会明显放大。
    """

    from trimesh.proximity import ProximityQuery

    return ProximityQuery(_load_checked_trimesh(file_path, scale, content_sha256))


def _signed_distance_warp(
    points_local: np.ndarray,
    geometry: MeshGeometryCfg,
    *,
    stats: MeshSdfQueryStats | None,
) -> np.ndarray:
    r"""用 Warp/CUDA 计算 mesh signed distance。

    Warp 的 `mesh_query_point()` 返回最近三角面、重心坐标和 sign；最近距离由
    `mesh_eval_position()` 得到的最近点显式计算。该路径不做梯度，只服务 validator。
    """

    wp = _require_warp_cuda()
    device = "cuda:0"
    handle = _get_warp_mesh_handle(geometry, device=device)
    query_points = wp.array(points_local, dtype=wp.vec3, device=device)  # [N, 3] local query points
    distances = wp.zeros(points_local.shape[0], dtype=float, device=device)  # [N] signed distances

    # kernel 定义在文件顶层，Warp 能稳定拿到源码并缓存编译产物。
    wp.launch(_warp_mesh_signed_distance_kernel, dim=points_local.shape[0], inputs=[handle.mesh.id, query_points, distances], device=device)
    wp.synchronize_device(device)

    if stats is not None:
        stats.record_backend("warp")
        stats.mesh_query_count += 1
    return np.asarray(distances.numpy(), dtype=np.float64)


def _get_warp_mesh_handle(geometry: MeshGeometryCfg, *, device: str) -> _WarpMeshHandle:
    r"""构造或复用 Warp mesh BVH。"""

    wp = _require_warp_cuda()
    path, mesh_sha256 = _mesh_identity(geometry.file_path)
    scale = _scale_tuple(geometry.scale)
    key = (path, mesh_sha256, scale, device)
    if key in _WARP_MESH_CACHE:
        _WARP_MESH_CACHE.move_to_end(key)
        return _WARP_MESH_CACHE[key]

    mesh = _load_checked_trimesh(path, scale, mesh_sha256)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32).reshape(-1)
    points = wp.array(vertices, dtype=wp.vec3, device=device)
    indices = wp.array(faces, dtype=wp.int32, device=device)
    handle = _WarpMeshHandle(mesh=wp.Mesh(points, indices), points=points, indices=indices)
    _remember_warp_mesh_handle(key, handle)
    return handle


def _remember_warp_mesh_handle(
    key: tuple[str, str, tuple[float, float, float], str],
    handle: _WarpMeshHandle,
) -> None:
    r"""按最近使用顺序保存 BVH，并在超过 128 项时释放最旧 Python owner。"""

    _WARP_MESH_CACHE[key] = handle
    _WARP_MESH_CACHE.move_to_end(key)
    if len(_WARP_MESH_CACHE) > _WARP_MESH_CACHE_MAXSIZE:
        _WARP_MESH_CACHE.popitem(last=False)


@lru_cache(maxsize=128)
def _load_checked_trimesh(
    file_path: str,
    scale: tuple[float, float, float],
    content_sha256: str,
):
    r"""加载、缩放并验证 mesh。

    `process=True` 是刻意选择：STL 常把每个 triangle 的顶点重复写入，若不合并顶点，
    一个几何上闭合的 tip 会被误判为非 watertight。处理后仍不闭合或 winding 不一致，
    才说明 signed distance 的 inside/outside 证书不可信。
    """

    import trimesh

    path = _resolve_mesh_path(file_path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != content_sha256:
        raise ValueError(f"mesh bytes changed while constructing SDF cache entry: {path}")
    mesh = trimesh.load(path, force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"mesh SDF expects a triangle mesh, got {type(mesh).__name__}: {path}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"mesh SDF got empty mesh: {path}")

    mesh = mesh.copy()
    mesh.apply_scale(scale)
    if not mesh.is_watertight:
        raise ValueError(f"mesh SDF requires watertight mesh after trimesh processing: {path}")
    if not mesh.is_winding_consistent:
        raise ValueError(f"mesh SDF requires winding-consistent mesh after trimesh processing: {path}")
    return mesh


def _mesh_identity(file_path: str) -> tuple[str, str]:
    r"""返回 cache 使用的规范路径与 byte-level mesh identity。

    路径区分不同资产来源，SHA-256 证明该路径当前承载的三角网格字节。这样同一路径在
    长时间 dataset build 中被重写时会形成新 cache key，而不会继续使用旧 device BVH。
    """

    path = _resolve_mesh_path(file_path)
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_mesh_path(file_path: str) -> Path:
    r"""解析 mesh 路径。

    生成资产通常写绝对路径；若未来 adapter 写相对路径，则按当前工作目录和 assets
    根目录两级兜底解析。解析失败必须 fail-hard，因为 validator 不能跳过未知几何。
    """

    raw_path = Path(file_path).expanduser()
    if raw_path.is_absolute() and raw_path.exists():
        return raw_path.resolve()
    if raw_path.exists():
        return raw_path.resolve()
    assets_root = Path(__file__).resolve().parents[1]
    candidate = assets_root / raw_path
    if candidate.exists():
        return candidate.resolve()
    raise ValueError(f"mesh file does not exist for SDF validator: {file_path!r}")


def _require_warp_cuda():
    r"""导入并初始化 Warp CUDA 后端。"""

    try:
        import warp as wp
    except Exception as exc:
        raise RuntimeError("warp-lang is not importable") from exc

    wp.init()
    if not wp.is_cuda_available():
        raise RuntimeError("Warp CUDA device is unavailable")
    return wp


def _scale_tuple(scale: Vector3) -> tuple[float, float, float]:
    r"""把 schema scale 转为 cache key 友好的三元组。"""

    return (float(scale[0]), float(scale[1]), float(scale[2]))


def _stable_seed(*parts: str) -> int:
    r"""从稳定字符串生成 `trimesh.sample()` 可接受的随机种子。"""

    digest = hashlib.sha256("::".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _to_vector3(point: np.ndarray) -> Vector3:
    r"""把 numpy 点转成项目 schema 使用的 `Vector3`。"""

    return (float(point[0]), float(point[1]), float(point[2]))


try:
    import warp as wp

    @wp.kernel
    def _warp_mesh_signed_distance_kernel(
        mesh: wp.uint64,  # pyright: ignore[reportInvalidTypeForm]
        points: wp.array(dtype=wp.vec3),  # pyright: ignore[reportInvalidTypeForm]
        distances: wp.array(dtype=float),  # pyright: ignore[reportInvalidTypeForm]
    ):
        r"""Warp kernel：批量查询点到单个 mesh 的 signed distance。"""

        tid = wp.tid()
        point = points[tid]
        query = wp.mesh_query_point(mesh, point, 1.0e8)  # pyright: ignore[reportArgumentType]
        if not query.result:  # pyright: ignore[reportAttributeAccessIssue]
            distances[tid] = 3.4028234663852886e38
            return
        closest = wp.mesh_eval_position(  # pyright: ignore[reportAttributeAccessIssue]
            mesh,
            query.face,  # pyright: ignore[reportAttributeAccessIssue]
            query.u,  # pyright: ignore[reportAttributeAccessIssue]
            query.v,  # pyright: ignore[reportAttributeAccessIssue]
        )
        distance = wp.length(closest - point)
        if query.sign >= 0.0:  # pyright: ignore[reportAttributeAccessIssue]
            distances[tid] = distance
        else:
            distances[tid] = -distance

except Exception:
    # 没装 Warp 时仍允许 import 本模块；真正请求 Warp 后端时 `_require_warp_cuda()` 会报错。
    _warp_mesh_signed_distance_kernel = None  # pyright: ignore[reportAssignmentType]


__all__ = [
    "MeshSdfBackend",
    "MeshSdfQueryStats",
    "is_mesh_body",
    "sample_mesh_surface",
    "signed_distance_to_mesh_body",
    "signed_distance_to_mesh_body_batch",
]
