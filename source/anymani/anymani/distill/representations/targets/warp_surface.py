r"""Warp GPU 主端的 owner-local 最近三角面查询。

本模块只消费 ``robots`` 已构造好的 GPU-resident owner BVH 和当前 owner $SE(3)$ 位姿；它不读取
URDF、sidecar、mesh 文件，也不参与网络前向。给定固定于 `{h}` 的 query $x^h$ 与 owner 当前位姿
$T_{hg}(q)$，先计算：

$$
x^g=T_{hg}(q)^{-1}x^h,
$$

再在 owner-local BVH 中调用 ``wp.mesh_query_point``。Warp kernel 返回最近 face、三角面参数
$(u,v)$、最近点与 sign；最近点回变 `{h}`，供 Gaussian distance、closest-source provenance
和点 Jacobian 的一阶教师使用。query 坐标与 mesh 都不参与 autograd，只有训练模型对 $q$ 的预测
路径保留梯度。

生产路径明确要求 CUDA/Warp；不自动回退 CPU。CPU/trimesh 只在独立 reference backend 中用于
数值对照，避免在线训练吞吐因 Python proximity 查询或 host-device 往返退化。

张量轴与缓存关系：

```text
query_points_h      [B, G, N_Q, 3]   # `{h}`, m
owner_transforms    [B, G, 4, 4]     # owner local -> `{h}`
Warp handles        G                 # 每个 owner 一个静态 BVH
distance            [B, G, N_Q]      # unsigned, m
closest_point_h     [B, G, N_Q, 3]   # `{h}`, m
face_index          [B, G, N_Q]      # owner union face
barycentric         [B, G, N_Q, 3]   # 和为 1
feature_margin      [B, G, N_Q]      # m
```

Python 只沿静态 owner 轴做 $G$ 次 kernel launch；每次 launch 内的 $B N_Q$ query 完全并行。BVH、
vertices、indices 与 face altitudes 在训练前上传并驻留 GPU，调用期间不读取 trimesh、不创建 mesh、
不发生 host-to-device copy。未来若 profile 证明 $G$ 次 launch 是瓶颈，可以评估多 mesh handle 的
融合 kernel，但必须先保持 face provenance 与 owner axis 完全一致。

Warp 与 PyTorch 通过 ``wp.from_torch`` 零拷贝共享当前 CUDA tensor，并用当前 PyTorch stream
创建 ``wp.ScopedStream``。因此 kernel 输出后的 PyTorch $SE(3)$ 变换与后续 target 公式保持 stream
顺序，不需要每个 owner 强制全设备 synchronize。测试在读取标量证据前显式 synchronize；训练代码
让正常 CUDA dependency 管理顺序。

``mesh_query_point`` 返回 owner-local 最近点。kernel 不把局部点直接当 `{h}` 输出，而是在全部
owner 查询完成后用当前 $T_{hg}(q)$ 统一变换；遗漏这一步会让距离数值仍正确，却让点 Jacobian 和
frame provenance 全部错误。distance 因刚体变换保持不变，不需要重新计算。

重心坐标约定为 $(1-u-v,u,v)$，与 union face 的三个 vertex 顺序一致。face index 的稳定性只在
同一 asset content hash、Manifold 版本与 cache realization 内保证；跨 remeshing 不能比较裸 face
整数，必须连同 asset/cache hash 解释。

feature margin 使用离线预计算的三角形三条高。它筛除最近投影落在 triangle edge/vertex 附近的
一阶样本，降低离散 face source 切换对 $\kappa$ 的污染。它不检测两个远距离 surface patches 的
全局等距 medial axis，因此结果字段刻意命名为 ``feature_margin_m``，不能在论文或日志中直接改名
为“全局唯一性间隔”。

sign 由 Warp winding query 返回，当前 Gaussian distance shell 只消费 unsigned distance。保留 sign
是为了 inside/outside 诊断和未来 occupancy 候选，不能在当前目标中暗中把 unsigned shell 改成
signed field。任何 field 语义变更都必须在 representations/fields 与 objective 中显式完成。

该 backend 是 training-only privileged teacher。它不要求 autograd，也不应注册成模型模块；PPO
部署不构建 Warp BVH、不调用 kernel，也不保存 closest point/face/barycentric。density/κ 双训练目标只对
模型参数执行普通反向，Warp 的离散最近面与解析标签始终停止梯度。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from anymani.distill.representations.sources.collision_geometry import WarpOwnerGeometryCache


@dataclass(frozen=True)
class WarpSurfaceQueryResult:
    r"""全部 owner/query 的最近面结果。"""

    distance_m: torch.Tensor  # `[B,G,N_Q]`，unsigned distance，m
    closest_point_h_m: torch.Tensor  # `[B,G,N_Q,3]`，最近点，`{h}`，m
    face_index: torch.Tensor  # `[B,G,N_Q]`，owner union mesh face
    barycentric: torch.Tensor  # `[B,G,N_Q,3]`，最近点三角面重心坐标
    feature_margin_m: torch.Tensor  # `[B,G,N_Q]`，投影点到当前三角形边界的最小物理距离
    sign: torch.Tensor  # `[B,G,N_Q]`，Warp inside/outside sign；unsigned field 不直接消费


def query_owner_surfaces_warp(
    query_points_h: torch.Tensor,
    owner_transforms_hg: torch.Tensor,
    warp_cache: WarpOwnerGeometryCache,
) -> WarpSurfaceQueryResult:
    r"""在 GPU 上批量查询所有 owner 的最近三角面。

    Args:
        query_points_h (torch.Tensor): ``[B,G,N_Q,3]`` 固定 `{h}` 查询点，m，CUDA float tensor。
        owner_transforms_hg (torch.Tensor): ``[B,G,4,4]`` 当前 owner reference link 位姿。
        warp_cache (WarpOwnerGeometryCache): 与 owner 轴及资产 hash 对齐的 GPU BVH。

    Returns:
        WarpSurfaceQueryResult: distance、最近点、face/barycentric provenance 与 sign。

    Raises:
        ValueError: shape、device、owner 数或 dtype 不符合在线主端合同。
        RuntimeError: Warp 不可用或 CUDA 不可用。
    """

    if query_points_h.ndim != 4 or query_points_h.shape[-1] != 3:
        raise ValueError("query_points_h must have shape [B,G,N_Q,3]")
    if owner_transforms_hg.shape != (query_points_h.shape[0], query_points_h.shape[1], 4, 4):
        raise ValueError("owner_transforms_hg must have shape [B,G,4,4] matching query points")
    if not query_points_h.is_cuda or not owner_transforms_hg.is_cuda:
        raise RuntimeError("Warp online surface query requires CUDA-resident tensors")
    if query_points_h.dtype != torch.float32 or owner_transforms_hg.dtype != torch.float32:
        raise ValueError("Warp surface query currently requires float32 CUDA tensors")
    if query_points_h.device != torch.device(warp_cache.device):
        raise ValueError("query tensor device does not match Warp owner cache device")

    _ensure_warp_kernel()
    import warp as wp

    batch_size, owner_count, query_count, _ = query_points_h.shape
    if len(warp_cache.handles) != owner_count:
        raise ValueError("Warp owner cache axis does not match query owner axis")
    device = query_points_h.device
    # Owner-major storage makes each per-owner `[B,N_Q,...]` slice contiguous. With `[B,G,...]`,
    # `tensor[:, owner_index].reshape(-1)` allocates a copy when B>1, so Warp writes would never reach
    # the original output and leave uninitialized distances in the teacher tensor.
    distance = torch.empty((owner_count, batch_size, query_count), device=device, dtype=torch.float32)
    closest_local = torch.empty((owner_count, batch_size, query_count, 3), device=device, dtype=torch.float32)
    face_index = torch.empty((owner_count, batch_size, query_count), device=device, dtype=torch.int32)
    barycentric = torch.empty((owner_count, batch_size, query_count, 3), device=device, dtype=torch.float32)
    feature_margin = torch.empty((owner_count, batch_size, query_count), device=device, dtype=torch.float32)
    sign = torch.empty((owner_count, batch_size, query_count), device=device, dtype=torch.float32)
    stream = wp.stream_from_torch(torch.cuda.current_stream(device=device))
    with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
        for owner_index, handle in enumerate(warp_cache.handles):
            inverse_transform = torch.linalg.inv(owner_transforms_hg[:, owner_index])
            points_h = query_points_h[:, owner_index].reshape(-1, 3)
            points_local = (
                torch.einsum("bij,bnj->bni", inverse_transform[:, :3, :3], points_h.reshape(batch_size, -1, 3))
                + inverse_transform[:, :3, 3].unsqueeze(1)
            ).reshape(-1, 3).contiguous()
            distance_flat = distance[owner_index].reshape(-1)
            closest_flat = closest_local[owner_index].reshape(-1, 3)
            face_flat = face_index[owner_index].reshape(-1)
            barycentric_flat = barycentric[owner_index].reshape(-1, 3)
            feature_margin_flat = feature_margin[owner_index].reshape(-1)
            sign_flat = sign[owner_index].reshape(-1)
            wp.launch(
                _warp_owner_surface_query_kernel,
                dim=points_local.shape[0],
                inputs=[
                    handle.mesh.id,
                    handle.face_altitudes,
                    handle.source_face_indices,
                    wp.from_torch(points_local, dtype=wp.vec3),
                    wp.from_torch(distance_flat, dtype=wp.float32),
                    wp.from_torch(closest_flat, dtype=wp.vec3),
                    wp.from_torch(face_flat, dtype=wp.int32),
                    wp.from_torch(barycentric_flat, dtype=wp.vec3),
                    wp.from_torch(feature_margin_flat, dtype=wp.float32),
                    wp.from_torch(sign_flat, dtype=wp.float32),
                ],
                device=warp_cache.device,
            )
    distance = distance.permute(1, 0, 2).contiguous()
    closest_local = closest_local.permute(1, 0, 2, 3).contiguous()
    face_index = face_index.permute(1, 0, 2).contiguous()
    barycentric = barycentric.permute(1, 0, 2, 3).contiguous()
    feature_margin = feature_margin.permute(1, 0, 2).contiguous()
    sign = sign.permute(1, 0, 2).contiguous()
    closest_h = (
        torch.einsum("bgij,bgnj->bgni", owner_transforms_hg[..., :3, :3], closest_local)
        + owner_transforms_hg[..., :3, 3].unsqueeze(-2)
    )
    return WarpSurfaceQueryResult(distance, closest_h, face_index, barycentric, feature_margin, sign)


def _ensure_warp_kernel() -> None:
    """延迟检查 Warp kernel 是否成功注册。"""

    if _warp_owner_surface_query_kernel is None:
        raise RuntimeError("Warp mesh query kernel is unavailable in this environment")


try:
    import warp as wp

    @wp.kernel
    def _warp_owner_surface_query_kernel(
        mesh: wp.uint64,
        face_altitudes: wp.array(dtype=wp.vec3),
        source_face_indices: wp.array(dtype=wp.int32),
        points_local: wp.array(dtype=wp.vec3),
        distance: wp.array(dtype=float),
        closest_point: wp.array(dtype=wp.vec3),
        face_index: wp.array(dtype=wp.int32),
        barycentric: wp.array(dtype=wp.vec3),
        feature_margin: wp.array(dtype=float),
        sign: wp.array(dtype=float),
    ):
        """Warp kernel：最近面、最近点、重心坐标和内外 sign。"""

        thread = wp.tid()
        query = wp.mesh_query_point(mesh, points_local[thread], 1.0e8)
        if not query.result:
            distance[thread] = 3.4028234663852886e38
            closest_point[thread] = wp.vec3(0.0, 0.0, 0.0)
            face_index[thread] = -1
            barycentric[thread] = wp.vec3(0.0, 0.0, 0.0)
            feature_margin[thread] = 0.0
            sign[thread] = 0.0
            return
        closest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        closest_point[thread] = closest
        distance[thread] = wp.length(closest - points_local[thread])
        face_index[thread] = source_face_indices[query.face]
        bary = wp.vec3(1.0 - query.u - query.v, query.u, query.v)
        barycentric[thread] = bary
        altitudes = face_altitudes[query.face]
        feature_margin[thread] = wp.min(
            bary[0] * altitudes[0],
            wp.min(bary[1] * altitudes[1], bary[2] * altitudes[2]),
        )
        sign[thread] = query.sign

except Exception:
    _warp_owner_surface_query_kernel = None


__all__ = ["WarpSurfaceQueryResult", "query_owner_surfaces_warp"]
