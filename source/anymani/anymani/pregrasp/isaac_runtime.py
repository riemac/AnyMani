r"""Pregrasp search与ManagerBased reset共享的Isaac runtime物理原件。

该模块必须在``AppLauncher``之后导入；它不由``anymani.pregrasp.__init__``重导出，从而保持schema/cache/provider
可在普通Python合同测试中使用。核心frame chain为：

$$
T_{wh}=T_{wa}T_{ah},\qquad T_{wo}=T_{wh}T_{ho},\qquad T_{ho}=T_{wh}^{-1}T_{wo}.
$$

``semantic_R_ha/semantic_p_ha``定义$T_{ha}$，因此必须先求逆得到$T_{ah}$；直接把$p_{ha}$当作$p_{ah}$
会在非零平移标定时产生符号与frame错误。
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import isaaclab.utils.math as math_utils
import torch

from .mvp80_strict_search import deepest_contact_normal_from_buffers


def file_sha256(path: Path | str) -> str:
    r"""流式计算已解析本地object bytes的SHA-256。"""

    resolved = Path(path).expanduser().resolve()  # identity绑定实际bytes，不绑定Nucleus URL文本
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)  # 1 MiB block限制host峰值内存
    return digest.hexdigest()


def hand_semantic_pose_w(
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    semantic_R_ha: Sequence[float],
    semantic_p_ha: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""由raw asset root pose和$T_{ha}$标定计算hand semantic world pose。

    Args:
        root_pos_w (torch.Tensor): $p_{wa}$，形状``[B,3]``，单位m。
        root_quat_w (torch.Tensor): $q_{wa}$，形状``[B,4]``，顺序wxyz。
        semantic_R_ha (Sequence[float]): row-major $R_{ha}$，9个标量。
        semantic_p_ha (Sequence[float]): $p_{ha}$，3个标量，单位m。

    Returns:
        tuple[torch.Tensor, torch.Tensor]: $(p_{wh},q_{wh})$，形状``[B,3]``与``[B,4]``。
    """

    batch_size = root_pos_w.shape[0]  # vectorized env batch $B$
    if root_pos_w.shape != (batch_size, 3) or root_quat_w.shape != (batch_size, 4):
        raise ValueError("root pose must have shapes [B,3] and [B,4]")
    r_ha = torch.as_tensor(semantic_R_ha, dtype=root_pos_w.dtype, device=root_pos_w.device).reshape(1, 3, 3)
    p_ha = torch.as_tensor(semantic_p_ha, dtype=root_pos_w.dtype, device=root_pos_w.device).reshape(1, 3)
    q_ha = math_utils.quat_from_matrix(r_ha)  # $q_{ha}$，wxyz
    q_ah = math_utils.quat_inv(q_ha).expand(batch_size, -1)  # $R_{ah}=R_{ha}^{T}$
    p_ah = math_utils.quat_apply(q_ah, -p_ha.expand(batch_size, -1))  # $p_{ah}=-R_{ah}p_{ha}$
    return math_utils.combine_frame_transforms(root_pos_w, root_quat_w, p_ah, q_ah)  # $T_{wh}=T_{wa}T_{ah}$


def object_pose_h_from_world(
    hand_pos_w: torch.Tensor,
    hand_quat_w: torch.Tensor,
    object_pos_w: torch.Tensor,
    object_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""把object world pose变换为candidate hand-frame pose $T_{ho}$。"""

    return math_utils.subtract_frame_transforms(
        hand_pos_w,
        hand_quat_w,
        object_pos_w,
        object_quat_w,
    )  # $T_{ho}=T_{wh}^{-1}T_{wo}$


def object_pose_w_from_hand(
    hand_pos_w: torch.Tensor,
    hand_quat_w: torch.Tensor,
    object_pos_h: torch.Tensor,
    object_quat_h: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""把candidate $T_{ho}$组合为simulator写入所需world pose。"""

    return math_utils.combine_frame_transforms(
        hand_pos_w,
        hand_quat_w,
        object_pos_h,
        object_quat_h,
    )  # $T_{wo}=T_{wh}T_{ho}$


def contact_separation_summary(sensor: Any, physics_dt: float) -> dict[str, Any]:
    r"""解包RigidContactView并返回force/separation/penetration充分统计量。

    PhysX flat buffer由``count/start``描述每个``(env,body,filter)``分组。负separation表示几何重叠，
    penetration depth定义为$\max(0,-\min d_{sep})$。Normal force可能因contact normal方向为负，
    因此同时返回signed sum和absolute sum；接触激活仍应使用vector magnitude而不是该signed scalar。
    """

    forces, _, _, separations, counts, starts = sensor.contact_physx_view.get_contact_data(dt=float(physics_dt))
    flat_counts = counts.reshape(-1).to(dtype=torch.long)  # 每个pair group的contact point数量
    flat_starts = starts.reshape(-1).to(dtype=torch.long)  # 每组在flat buffer中的起点
    row_ids = torch.repeat_interleave(torch.arange(flat_counts.numel(), device=flat_counts.device), flat_counts)
    if row_ids.numel() == 0:
        return {
            "contact_points": 0,
            "normal_force_sum_N": 0.0,
            "normal_force_abs_sum_N": 0.0,
            "min_separation_m": None,
            "penetration_depth_m": 0.0,
        }
    block_starts = flat_counts.cumsum(0) - flat_counts  # packed row_ids中每组的局部起点
    offsets = torch.arange(row_ids.numel(), device=row_ids.device) - block_starts.repeat_interleave(flat_counts)
    indices = flat_starts[row_ids] + offsets  # PhysX flat buffer中的真实contact indices
    valid_separations = separations.reshape(-1).index_select(0, indices)
    valid_forces = forces.reshape(-1).index_select(0, indices)
    minimum = float(valid_separations.min().item())  # m；负数表示penetration
    return {
        "contact_points": int(indices.numel()),
        "normal_force_sum_N": float(valid_forces.sum().item()),
        "normal_force_abs_sum_N": float(valid_forces.abs().sum().item()),
        "min_separation_m": minimum,
        "penetration_depth_m": max(0.0, -minimum),
    }


def contact_penetration_depth_per_env(sensor: Any, physics_dt: float) -> torch.Tensor:
    r"""返回每个environment在一个filtered sensor view中的最大penetration depth。

    Returns:
        torch.Tensor: ``[B]`` non-negative penetration depth，单位m；无contact的env为0。
    """

    _, _, _, separations, counts, starts = sensor.contact_physx_view.get_contact_data(dt=float(physics_dt))
    flat_counts = counts.reshape(-1).to(dtype=torch.long)  # group轴为env×body×filter
    flat_starts = starts.reshape(-1).to(dtype=torch.long)
    group_ids = torch.repeat_interleave(torch.arange(flat_counts.numel(), device=flat_counts.device), flat_counts)
    environment_count = sensor.body_physx_view.count // sensor.num_bodies  # rigid body view按env-major排列
    if group_ids.numel() == 0:
        return torch.zeros(environment_count, dtype=torch.float32, device=flat_counts.device)
    block_starts = flat_counts.cumsum(0) - flat_counts
    offsets = torch.arange(group_ids.numel(), device=group_ids.device) - block_starts.repeat_interleave(flat_counts)
    indices = flat_starts[group_ids] + offsets
    valid_separations = separations.reshape(-1).index_select(0, indices)
    group_minimum = torch.full(
        (flat_counts.numel(),),
        torch.inf,
        dtype=valid_separations.dtype,
        device=valid_separations.device,
    )  # 无contact group保持+inf，后续不产生penetration
    group_minimum.scatter_reduce_(0, group_ids, valid_separations, reduce="amin", include_self=True)
    pair_minimum = group_minimum.reshape(environment_count, sensor.num_bodies, -1).amin(dim=(1, 2))
    return torch.where(torch.isfinite(pair_minimum), torch.clamp(-pair_minimum, min=0.0), 0.0)


def deepest_contact_normal_per_env(sensor: Any, physics_dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    r"""返回每个environment最深contact的penetration与PhysX world normal。

    Normal方向保持PhysX原定义，不在通用helper中假设应移动sensor还是filter object。Pregrasp搜索中sensor是
    robot link、filter是object；若要给object做depenetration proposal，应由调用方显式验证并选择normal符号。

    Returns:
        tuple: penetration depth `[B]`，m；对应world normal `[B,3]`。无penetration rows返回全零。
    """

    _, _, normals, separations, counts, starts = sensor.contact_physx_view.get_contact_data(dt=float(physics_dt))
    environment_count = sensor.body_physx_view.count // sensor.num_bodies
    return deepest_contact_normal_from_buffers(
        normals,
        separations,
        counts,
        starts,
        environment_count=environment_count,
        body_count=sensor.num_bodies,
    )


__all__ = [
    "contact_separation_summary",
    "contact_penetration_depth_per_env",
    "deepest_contact_normal_per_env",
    "file_sha256",
    "hand_semantic_pose_w",
    "object_pose_h_from_world",
    "object_pose_w_from_hand",
]
