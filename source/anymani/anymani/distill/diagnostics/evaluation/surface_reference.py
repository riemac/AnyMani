r"""Warp 在线距离后端与可选 Kaolin 0.18.0 reference 的数值/吞吐对照。

Kaolin 不进入 teacher 主路径，也不参与自动 fallback。该函数只在显式诊断时导入
``kaolin.metrics.trianglemesh.point_to_mesh_distance``，比较其 squared unsigned distance 与 Warp
``mesh_query_point`` 的 unsigned distance。Kaolin face vertices 在计时前变换到当前 ``{h}``，因此报告
只比较 point-to-triangle kernel，不把两种 API 不同的数据准备成本伪装成算法差异。
"""

from __future__ import annotations

from collections.abc import Callable  # 被计时 operation 的无参数 GPU 调用合同
from dataclasses import dataclass  # 数值/吞吐报告冻结为不可变证据
from statistics import median  # latency 使用中位数抵抗偶发调度抖动

import torch  # GPU faces、queries、CUDA Event 与数值误差

from anymani.distill.representations.targets.warp_surface import (  # 正式 online teacher backend
    query_owner_surfaces_warp,
)
from anymani.robots.owner_geometry import (  # 同一资产 CPU triangles 与 GPU BVH
    OwnerGeometryCache,
    WarpOwnerGeometryCache,
)


@dataclass(frozen=True)
class SurfaceReferenceReport:
    r"""同一批 query 的 unsigned distance 数值误差与 kernel latency。

    Warp latency 保留主路径的 `{h}` 到 owner-local 变换；Kaolin faces 预先变换到 `{h}`，只计
    point-to-triangle kernel。该报告用于后端审计，不代表完整 trainer 吞吐。
    """

    maximum_absolute_error_m: float  # $\\max |d_{Warp}-d_{Kaolin}|$，m
    mean_absolute_error_m: float  # 全部 $B\\times G\\times N_Q$ 点平均，m
    warp_median_ms: float  # Warp query 含 query `{h}` -> owner-local 变换
    kaolin_median_ms: float  # Kaolin point-to-triangle；已排除 face transform
    warmup_iterations: int  # 两端各自排除 lazy load 的次数
    measured_iterations: int  # 两端各自 CUDA Event 样本数


def compare_warp_and_kaolin_distances(
    query_points_h: torch.Tensor,  # `[B,G,N_Q,3]`，CUDA float32，`{h}`，m
    owner_transforms_hg: torch.Tensor,  # `[B,G,4,4]`，owner-local -> `{h}`
    owner_cache: OwnerGeometryCache,  # CPU strict union triangles
    warp_cache: WarpOwnerGeometryCache,  # GPU Warp BVHs
    *,
    warmup_iterations: int = 10,  # 每端预热次数
    measured_iterations: int = 30,  # 每端正式计时次数
) -> SurfaceReferenceReport:
    r"""在相同 GPU query 上比较 Warp distance 与 Kaolin 0.18.0 reference。

    Args:
        query_points_h (torch.Tensor): ``[B,G,N_Q,3]`` CUDA float32，``{h}``，m。
        owner_transforms_hg (torch.Tensor): ``[B,G,4,4]`` owner-local 到 ``{h}`` 当前位姿。
        owner_cache (OwnerGeometryCache): CPU Manifold union triangles，与 owner 轴同序。
        warp_cache (WarpOwnerGeometryCache): 同一内容哈希的 GPU Warp BVH。
        warmup_iterations (int): 两端 kernel 各自预热次数；默认 10。
        measured_iterations (int): CUDA Event 重复计时次数；默认 30。

    Returns:
        SurfaceReferenceReport: 米制绝对误差和两端 median kernel latency。

    Raises:
        RuntimeError: CUDA 或 Kaolin 不可用时抛出；绝不回退 CPU。
        ValueError: owner/hash/shape 轴不一致或计时次数非法时抛出。
    """

    # reference 仅接受主路径相同的 CUDA float32 数据，避免 dtype/device 差异污染误差归因。
    if not query_points_h.is_cuda or query_points_h.dtype != torch.float32:  # 与正式 Warp 主路径相同
        raise RuntimeError("surface reference requires CUDA float32 query_points_h")  # 不回退 CPU/float64
    if owner_transforms_hg.shape != (*query_points_h.shape[:2], 4, 4):  # `[B,G]` pose 轴闭合
        raise ValueError("owner_transforms_hg must align with [B,G] query axes")  # 不广播 owner pose
    if warmup_iterations < 0 or measured_iterations < 1:  # timing 离散域
        raise ValueError("reference timing iterations are invalid")  # 至少一个正式样本
    if owner_cache.asset_content_hash != warp_cache.asset_content_hash:  # CPU/GPU 必须同资产
        raise ValueError("CPU owner cache and Warp cache content hashes differ")  # 防止串 cache
    if len(owner_cache.records) != query_points_h.shape[1]:  # owner 数 $G$
        raise ValueError("owner cache axis does not match query owner axis")  # 不截断/重复

    # Kaolin 是显式 optional diagnostic dependency；缺失时报告清楚，不改变 Warp teacher 行为。
    try:  # 仅显式调用时触发 optional import
        from kaolin.metrics.trianglemesh import point_to_mesh_distance  # squared unsigned distance
    except Exception as exc:  # 缺包、ABI/CUDA 不兼容都在诊断边界报告
        raise RuntimeError(  # 不改变 Warp teacher backend
            "Kaolin 0.18.0 is required only for this explicit reference diagnostic"
        ) from exc

    transformed_faces: list[torch.Tensor] = []  # 每项 `[B,F_g,3,3]`，当前 `{h}`，m
    for owner_index, record in enumerate(owner_cache.records):  # $g=0,...,G-1$
        local_faces = torch.as_tensor(  # CPU numpy triangles -> GPU tensor
            record.mesh.triangles,  # `[F_g,3,3]` owner-local union boundary，m
            device=query_points_h.device,  # 与 query 同 GPU
            dtype=torch.float32,  # 与 Warp/Kaolin kernel 同 dtype
        )  # `[F_g,3,3]` owner-local union boundary
        transform = owner_transforms_hg[:, owner_index]  # `[B,4,4]`，owner-local -> `{h}`
        faces_h = (  # $p_h=R_{hg}p_g+t_{hg}$
            torch.einsum(  # `[B,3,3] x [F_g,3,3] -> [B,F_g,3,3]`
                "bij,fvj->bfvi", transform[:, :3, :3], local_faces
            )
            + transform[:, None, None, :3, 3]  # broadcast translation `[B,1,1,3]`
        )  # `[B,F_g,3,3]`；计时前物化，隔离 API 数据准备差异
        transformed_faces.append(faces_h.contiguous())  # Kaolin kernel 连续 face buffer

    def warp_call() -> torch.Tensor:
        r"""返回 ``[B,G,N_Q]`` Warp unsigned distance，单位 m。"""

        return query_owner_surfaces_warp(  # 正式主路径完整 surface query
            query_points_h,  # `{h}` queries
            owner_transforms_hg,  # 当前 owner poses
            warp_cache,  # owner-local BVHs
        ).distance_m  # 只取 `[B,G,N_Q]` distance，m

    def kaolin_call() -> torch.Tensor:
        r"""逐 owner 调 Kaolin squared-distance kernel，再恢复米制 distance。"""

        distances: list[torch.Tensor] = []  # 每项 `[B,N_Q]`
        for owner_index, faces_h in enumerate(transformed_faces):  # Kaolin API 每次处理一个 owner
            squared_distance, _, _ = point_to_mesh_distance(  # 返回 `[B,N_Q]` squared m²
                query_points_h[:, owner_index],  # `[B,N_Q,3]`，`{h}`
                faces_h,  # `[B,F_g,3,3]`，`{h}`
            )
            distances.append(torch.sqrt(torch.clamp_min(squared_distance, 0.0)))  # $d=\\sqrt{d^2}$，m
        return torch.stack(distances, dim=1)  # `[B,G,N_Q]`

    # 先各执行一次取得数值；CUDA synchronize 保证后续 CPU 标量读取不跨未完成 kernel。
    warp_distance = warp_call()  # `[B,G,N_Q]`，m
    kaolin_distance = kaolin_call()  # `[B,G,N_Q]`，m
    torch.cuda.synchronize(query_points_h.device)  # 完成两端 kernels 后才读误差标量
    absolute_error = torch.abs(warp_distance - kaolin_distance)  # 同一 unsigned surface distance 误差，m
    warp_latency = _cuda_median_ms(  # 正式 Warp operation median
        warp_call,  # `{h}` -> local + BVH query
        device=query_points_h.device,  # 当前 GPU
        warmup_iterations=warmup_iterations,  # 排除 lazy load
        measured_iterations=measured_iterations,  # Event 样本数
    )
    kaolin_latency = _cuda_median_ms(  # Kaolin kernel median
        kaolin_call,  # faces 已预变换
        device=query_points_h.device,  # 当前 GPU
        warmup_iterations=warmup_iterations,  # 排除 lazy load
        measured_iterations=measured_iterations,  # Event 样本数
    )
    return SurfaceReferenceReport(  # GPU 标量同步转 Python evidence
        maximum_absolute_error_m=float(absolute_error.max()),  # worst query，m
        mean_absolute_error_m=float(absolute_error.mean()),  # 全体平均，m
        warp_median_ms=warp_latency,  # 主路径 kernel 边界
        kaolin_median_ms=kaolin_latency,  # reference kernel 边界
        warmup_iterations=warmup_iterations,  # 复现 timing protocol
        measured_iterations=measured_iterations,  # 复现 timing protocol
    )


def _cuda_median_ms(
    operation: Callable[[], torch.Tensor],  # 无参数、在当前 stream 提交 GPU kernels
    *,
    device: torch.device,  # CUDA Event 所在 GPU
    warmup_iterations: int,  # 不计时预热次数
    measured_iterations: int,  # 正式 Event 样本数
) -> float:
    r"""使用逐次 CUDA Event 测量 operation median latency，单位 ms。

    Returns:
        float: 正式样本的 device elapsed time 中位数，单位 ms。
    """

    # 预热排除 lazy kernel/module load；返回张量不搬回 host，也不把同步计入 measured interval。
    for _ in range(warmup_iterations):  # lazy module/kernel/cache warmup
        operation()  # 返回张量保持 GPU resident
    torch.cuda.synchronize(device)  # 清空预热工作后开始正式计时
    samples_ms: list[float] = []  # 每次完整 operation 的 device elapsed time
    stream = torch.cuda.current_stream(device)  # 两个 Event 与 operation 使用同一 CUDA stream
    for _ in range(measured_iterations):  # 每次独立 Event pair
        start = torch.cuda.Event(enable_timing=True)  # 当前 stream 起点
        end = torch.cuda.Event(enable_timing=True)  # 当前 stream 终点
        start.record(stream)  # operation 前 stream event
        operation()  # 提交待测 kernels；不做 host copy
        end.record(stream)  # operation 后 stream event
        end.synchronize()  # 只同步当前测量终点
        samples_ms.append(float(start.elapsed_time(end)))  # device elapsed ms
    return float(median(samples_ms))  # 抵抗偶发 outlier


__all__ = [  # 可选 reference 公开面
    "SurfaceReferenceReport",  # typed report
    "compare_warp_and_kaolin_distances",  # explicit diagnostic
]
