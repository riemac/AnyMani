r"""Privileged object/task structured blocks的纯Torch hand-frame几何。

Object block描述当前物理状态，task block描述command/error/progress；二者不合并成旧15D切片。Root translation
只影响object position frame chain，velocity与orientation只使用$R_{wh}=R_{wa}R_{ha}^{\mathsf T}$。
"""

from __future__ import annotations

import math

import torch

from .task_math import quaternion_to_matrix_wxyz


def rotation_matrix_to_rot6d(rotation: torch.Tensor) -> torch.Tensor:
    r"""把$R\in SO(3)$前两列按列拼成continuous rot6d$[r_1;r_2]$。"""

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
    r"""构造$O^c_{t,\mathrm{obj}}\in\mathbb R^{B\times1\times15}$。

    $$
    O^c_{t,\mathrm{obj}}=
    [R_{hw}(p_{wo}-p_{anchor}),\operatorname{rot6d}(R_{hw}R_{wo}),R_{hw}v^w,R_{hw}\omega^w].
    $$

    Position单位m，rot6d无量纲，linear velocity单位m/s，angular velocity单位rad/s。
    """

    batch_size = object_pos_w.shape[0]
    vectors = (object_pos_w, position_anchor_w, object_linear_velocity_w, object_angular_velocity_w)
    if any(vector.shape != (batch_size, 3) for vector in vectors):
        raise ValueError("object position/anchor/velocities must share [B,3]")
    if root_quat_wxyz.shape != (batch_size, 4) or object_quat_wxyz.shape != (batch_size, 4):
        raise ValueError("root/object quaternions must share [B,4]")
    if semantic_R_ha.shape != (3, 3):
        raise ValueError("semantic_R_ha must have shape [3,3]")
    rotation_wa = quaternion_to_matrix_wxyz(root_quat_wxyz)
    rotation_hw = semantic_R_ha.unsqueeze(0) @ rotation_wa.transpose(-1, -2)  # $R_{hw}=R_{ha}R_{aw}$
    rotation_wo = quaternion_to_matrix_wxyz(object_quat_wxyz)
    relative_position_h = torch.einsum(
        "bij,bj->bi", rotation_hw, object_pos_w - position_anchor_w
    )  # anchor-relative，单位m
    rotation_ho = rotation_hw @ rotation_wo
    rot6d = rotation_matrix_to_rot6d(rotation_ho)
    linear_velocity_h = torch.einsum("bij,bj->bi", rotation_hw, object_linear_velocity_w)
    angular_velocity_h = torch.einsum("bij,bj->bi", rotation_hw, object_angular_velocity_w)
    return torch.cat((relative_position_h, rot6d, linear_velocity_h, angular_velocity_h), dim=-1).unsqueeze(1)


def task_state(
    axis_h: torch.Tensor,
    goal_error_so3_h_rad: torch.Tensor,
    net_rotation_rad: torch.Tensor,
    *,
    subgoal_angle_rad: float = math.pi / 6.0,
) -> torch.Tensor:
    r"""构造$O^c_{t,\mathrm{task}}=[\hat k^h,\phi^h,\theta_{goal},\Psi]$，形状$[B,1,8]$。"""

    if axis_h.ndim != 2 or axis_h.shape[1] != 3 or goal_error_so3_h_rad.shape != axis_h.shape:
        raise ValueError("axis_h and goal error must share [B,3]")
    if net_rotation_rad.shape != axis_h.shape[:1]:
        raise ValueError("net_rotation_rad must have shape [B]")
    axis_norm = torch.linalg.vector_norm(axis_h, dim=-1, keepdim=True)
    if bool((axis_norm < 1.0e-12).any().item()):
        raise ValueError("task axis must be non-zero")
    normalized_axis = axis_h / axis_norm
    subgoal = torch.full_like(net_rotation_rad, subgoal_angle_rad)
    return torch.cat(
        (
            normalized_axis,
            goal_error_so3_h_rad,
            subgoal.unsqueeze(-1),
            net_rotation_rad.unsqueeze(-1),
        ),
        dim=-1,
    ).unsqueeze(1)


__all__ = ["object_state_in_hand_frame", "rotation_matrix_to_rot6d", "task_state"]
