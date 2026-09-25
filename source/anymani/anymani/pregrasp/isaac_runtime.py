(
    'Isaac runtime primitives shared by pregrasp search and ManagerBased reset. '
    'Import only after AppLauncher; keep it out of anymani.pregrasp exports so '
    'schema/cache/provider remain usable in plain Python tests. Frame chain: '
    'T_wh=T_wa*T_ah, T_wo=T_wh*T_ho, T_ho=inverse(T_wh)*T_wo. semantic_R_ha/p_ha '
    'define T_ha, so invert them to obtain T_ah; using p_ha as p_ah gives a '
    'sign/frame error when calibration translation is nonzero.'
)

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import isaaclab.utils.math as math_utils
import torch

from .mvp80_strict_search import deepest_contact_normal_from_buffers


def file_sha256(path: Path | str) -> str:
    'Stream SHA-256 over resolved local object bytes.'

    resolved = Path(path).expanduser().resolve()  # Bind identity to actual bytes, not Nucleus URL text.
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)  # Use 1 MiB blocks to bound host memory.
    return digest.hexdigest()


def hand_semantic_pose_w(
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    semantic_R_ha: Sequence[float],
    semantic_p_ha: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    (
        'Compute hand semantic world pose from raw asset root pose and T_ha '
        'calibration. Inputs: root position [B,3] m, root quaternion [B,4] wxyz, '
        'row-major R_ha, and p_ha [3] m. Return (p_wh,q_wh) with shapes [B,3] and '
        '[B,4].'
    )

    batch_size = root_pos_w.shape[0]  # Vectorized environment batch B.
    if root_pos_w.shape != (batch_size, 3) or root_quat_w.shape != (batch_size, 4):
        raise ValueError("root pose must have shapes [B,3] and [B,4]")
    r_ha = torch.as_tensor(semantic_R_ha, dtype=root_pos_w.dtype, device=root_pos_w.device).reshape(1, 3, 3)
    p_ha = torch.as_tensor(semantic_p_ha, dtype=root_pos_w.dtype, device=root_pos_w.device).reshape(1, 3)
    q_ha = math_utils.quat_from_matrix(r_ha)  # Quaternion q_ha, order wxyz.
    q_ah = math_utils.quat_inv(q_ha).expand(batch_size, -1)  # Inverse rotation R_ah = transpose(R_ha).
    p_ah = math_utils.quat_apply(q_ah, -p_ha.expand(batch_size, -1))  # Inverse translation p_ah = -R_ah*p_ha.
    return math_utils.combine_frame_transforms(root_pos_w, root_quat_w, p_ah, q_ah)  # World hand transform T_wh = T_wa*T_ah.


def object_pose_h_from_world(
    hand_pos_w: torch.Tensor,
    hand_quat_w: torch.Tensor,
    object_pos_w: torch.Tensor,
    object_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    'Transform object world pose into candidate hand-frame pose T_ho.'

    return math_utils.subtract_frame_transforms(
        hand_pos_w,
        hand_quat_w,
        object_pos_w,
        object_quat_w,
    )  # Object pose in hand frame: T_ho = inverse(T_wh)*T_wo.


def object_pose_w_from_hand(
    hand_pos_w: torch.Tensor,
    hand_quat_w: torch.Tensor,
    object_pos_h: torch.Tensor,
    object_quat_h: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    'Compose candidate T_ho into the world pose written to the simulator.'

    return math_utils.combine_frame_transforms(
        hand_pos_w,
        hand_quat_w,
        object_pos_h,
        object_quat_h,
    )  # World object pose: T_wo = T_wh*T_ho.


def contact_separation_summary(sensor: Any, physics_dt: float) -> dict[str, Any]:
    (
        'Unpack RigidContactView and return force/separation/penetration statistics. '
        'PhysX count/start describe each (env,body,filter) group; negative separation '
        'means overlap and depth is max(0,-min(separation)). Normal force may be '
        'negative with contact-normal direction, so return signed and absolute sums. '
        'Use vector magnitude, not signed scalar, to activate contact.'
    )

    forces, _, _, separations, counts, starts = sensor.contact_physx_view.get_contact_data(dt=float(physics_dt))
    flat_counts = counts.reshape(-1).to(dtype=torch.long)  # Contact-point count per pair group.
    flat_starts = starts.reshape(-1).to(dtype=torch.long)  # Start index of each group in the flat buffer.
    row_ids = torch.repeat_interleave(torch.arange(flat_counts.numel(), device=flat_counts.device), flat_counts)
    if row_ids.numel() == 0:
        return {
            "contact_points": 0,
            "normal_force_sum_N": 0.0,
            "normal_force_abs_sum_N": 0.0,
            "min_separation_m": None,
            "penetration_depth_m": 0.0,
        }
    block_starts = flat_counts.cumsum(0) - flat_counts  # Group-local start in packed row_ids.
    offsets = torch.arange(row_ids.numel(), device=row_ids.device) - block_starts.repeat_interleave(flat_counts)
    indices = flat_starts[row_ids] + offsets  # Actual contact indices in the PhysX flat buffer.
    valid_separations = separations.reshape(-1).index_select(0, indices)
    valid_forces = forces.reshape(-1).index_select(0, indices)
    minimum = float(valid_separations.min().item())  # Meters; negative separation means penetration.
    return {
        "contact_points": int(indices.numel()),
        "normal_force_sum_N": float(valid_forces.sum().item()),
        "normal_force_abs_sum_N": float(valid_forces.abs().sum().item()),
        "min_separation_m": minimum,
        "penetration_depth_m": max(0.0, -minimum),
    }


def contact_penetration_depth_per_env(sensor: Any, physics_dt: float) -> torch.Tensor:
    'Return maximum penetration depth per env in one filtered sensor view, shape [B], meters; no contact gives zero.'

    _, _, _, separations, counts, starts = sensor.contact_physx_view.get_contact_data(dt=float(physics_dt))
    flat_counts = counts.reshape(-1).to(dtype=torch.long)  # Group axes are env x body x filter.
    flat_starts = starts.reshape(-1).to(dtype=torch.long)
    group_ids = torch.repeat_interleave(torch.arange(flat_counts.numel(), device=flat_counts.device), flat_counts)
    environment_count = sensor.body_physx_view.count // sensor.num_bodies  # Rigid-body view is env-major.
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
    )  # No-contact groups stay at +inf and cannot create false penetration.
    group_minimum.scatter_reduce_(0, group_ids, valid_separations, reduce="amin", include_self=True)
    pair_minimum = group_minimum.reshape(environment_count, sensor.num_bodies, -1).amin(dim=(1, 2))
    return torch.where(torch.isfinite(pair_minimum), torch.clamp(-pair_minimum, min=0.0), 0.0)


def deepest_contact_normal_per_env(sensor: Any, physics_dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    (
        'Return deepest contact penetration and PhysX world normal per env. Preserve '
        'PhysX normal direction; the helper does not assume whether sensor or filter '
        'object should move. Callers must choose and validate the sign for a '
        'depenetration proposal. Return depth [B] m and normal [B,3]; zero rows have '
        'no penetration.'
    )

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
