"""Warp closest-surface queries for physical geometry targets."""


from __future__ import annotations

from dataclasses import dataclass

import torch

from anymani.distill.representations.sources.collision_geometry import WarpOwnerGeometryCache


@dataclass(frozen=True)
class WarpSurfaceQueryResult:


    distance_m: torch.Tensor
    closest_point_h_m: torch.Tensor
    face_index: torch.Tensor
    barycentric: torch.Tensor
    feature_margin_m: torch.Tensor
    sign: torch.Tensor


def query_owner_surfaces_warp(
    query_points_h: torch.Tensor,
    owner_transforms_hg: torch.Tensor,
    warp_cache: WarpOwnerGeometryCache,
) -> WarpSurfaceQueryResult:


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
