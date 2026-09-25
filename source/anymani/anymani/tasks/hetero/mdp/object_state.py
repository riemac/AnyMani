(
    'Pure-Torch hand-frame geometry for privileged object/task feature blocks. '
    'The object block describes physical state; the task block carries command, '
    'error, and progress. Root translation affects only the object-position frame '
    'chain. Orientation and velocity use R_wh = R_wa * transpose(R_ha).'
)

from __future__ import annotations

import torch

from .task_math import quaternion_to_matrix_wxyz


def rotation_matrix_to_rot6d(rotation: torch.Tensor) -> torch.Tensor:
    'Stack the first two rotation-matrix columns into continuous rot6d [r1;r2].'

    if rotation.shape[-2:] != (3, 3) or not bool(torch.isfinite(rotation).all().item()):
        raise ValueError("rotation must be finite with shape [...,3,3]")
    return torch.cat((rotation[..., :, 0], rotation[..., :, 1]), dim=-1)


def object_state_in_hand_frame(
    *,
    root_quat_wxyz: torch.Tensor,
    semantic_R_ha: torch.Tensor,
    object_pos_w: torch.Tensor,
    object_quat_wxyz: torch.Tensor,
    position_anchor_w: torch.Tensor,
    object_linear_velocity_w: torch.Tensor,
    object_angular_velocity_w: torch.Tensor,
) -> torch.Tensor:
    (
        'Build the object block with shape [B,1,15]. Position is hand-frame and '
        'anchor-relative; orientation, linear velocity, and angular velocity use the '
        'same hand frame. Units are meters, dimensionless, m/s, and rad/s.'
    )

    batch_size = object_pos_w.shape[0]
    vectors = (object_pos_w, position_anchor_w, object_linear_velocity_w, object_angular_velocity_w)
    if any(vector.shape != (batch_size, 3) for vector in vectors):
        raise ValueError("object position/anchor/velocities must share [B,3]")
    if root_quat_wxyz.shape != (batch_size, 4) or object_quat_wxyz.shape != (batch_size, 4):
        raise ValueError("root/object quaternions must share [B,4]")
    if semantic_R_ha.shape != (3, 3):
        raise ValueError("semantic_R_ha must have shape [3,3]")
    rotation_wa = quaternion_to_matrix_wxyz(root_quat_wxyz)
    rotation_hw = semantic_R_ha.unsqueeze(0) @ rotation_wa.transpose(-1, -2)  # Hand-to-world rotation: R_hw = R_ha * R_aw.
    rotation_wo = quaternion_to_matrix_wxyz(object_quat_wxyz)
    relative_position_h = torch.einsum(
        "bij,bj->bi", rotation_hw, object_pos_w - position_anchor_w
    )  # Position relative to the anchor, meters.
    rotation_ho = rotation_hw @ rotation_wo
    rot6d = rotation_matrix_to_rot6d(rotation_ho)
    linear_velocity_h = torch.einsum("bij,bj->bi", rotation_hw, object_linear_velocity_w)
    angular_velocity_h = torch.einsum("bij,bj->bi", rotation_hw, object_angular_velocity_w)
    return torch.cat((relative_position_h, rot6d, linear_velocity_h, angular_velocity_h), dim=-1).unsqueeze(1)


def task_state(
    axis_h: torch.Tensor,
    goal_error_so3_h_rad: torch.Tensor,
    net_rotation_rad: torch.Tensor,
    max_positive_net_rotation_rad: torch.Tensor,
) -> torch.Tensor:
    (
        'Build the task block [B,1,8] from hand axis, goal log error, positive '
        'rotation frontier, and signed net rotation. Net rotation alone cannot '
        'determine frontier reward after rollback, so retain the episode maximum. The '
        '30-degree interval is fixed task identity, not an input feature.'
    )

    if axis_h.ndim != 2 or axis_h.shape[1] != 3 or goal_error_so3_h_rad.shape != axis_h.shape:
        raise ValueError("axis_h and goal error must share [B,3]")
    if net_rotation_rad.shape != axis_h.shape[:1] or max_positive_net_rotation_rad.shape != axis_h.shape[:1]:
        raise ValueError("net rotation and frontier maximum must have shape [B]")
    axis_norm = torch.linalg.vector_norm(axis_h, dim=-1, keepdim=True)
    if bool((axis_norm < 1.0e-12).any().item()):
        raise ValueError("task axis must be non-zero")
    normalized_axis = axis_h / axis_norm
    return torch.cat(
        (
            normalized_axis,
            goal_error_so3_h_rad,
            max_positive_net_rotation_rad.unsqueeze(-1),
            net_rotation_rad.unsqueeze(-1),
        ),
        dim=-1,
    ).unsqueeze(1)


__all__ = ["object_state_in_hand_frame", "rotation_matrix_to_rot6d", "task_state"]
