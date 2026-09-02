r"""Privileged object/task block的frame、shape与字段分离合同。"""

from __future__ import annotations

import math

import torch

from anymani.tasks.hetero.mdp.object_state import object_state_in_hand_frame, task_state
from anymani.tasks.hetero.mdp.task_math import quaternion_from_angle_axis_wxyz


def _quat(axis: tuple[float, float, float], angle: float) -> torch.Tensor:
    r"""构造FP64单batch quaternion。"""

    return quaternion_from_angle_axis_wxyz(
        torch.tensor((angle,), dtype=torch.float64), torch.tensor((axis,), dtype=torch.float64)
    )


def test_object_state_uses_nonidentity_hand_frame_and_preserves_units() -> None:
    r"""非identity$R_{ha}/R_{wa}$下验证position、orientation与twist均转入hand frame。"""

    root_quat = _quat((0.0, 0.0, 1.0), math.pi / 2.0)  # $R_{wa}=R_z(90°)$
    object_quat = root_quat.clone()  # raw root与object orientation一致
    semantic_R_ha = torch.tensor(
        ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)), dtype=torch.float64
    )  # $R_{ha}=R_x(90°)$
    anchor = torch.zeros(1, 3, dtype=torch.float64)
    position = torch.tensor(((0.0, 1.0, 0.0),), dtype=torch.float64)
    linear = torch.tensor(((0.0, 2.0, 0.0),), dtype=torch.float64)
    angular = torch.tensor(((0.0, 0.0, 3.0),), dtype=torch.float64)
    state = object_state_in_hand_frame(
        root_quat_wxyz=root_quat,
        semantic_R_ha=semantic_R_ha,
        object_pos_w=position,
        object_quat_wxyz=object_quat,
        position_anchor_w=anchor,
        object_linear_velocity_w=linear,
        object_angular_velocity_w=angular,
    )
    assert state.shape == (1, 1, 15)
    assert torch.allclose(state[0, 0, :3], torch.tensor((1.0, 0.0, 0.0), dtype=torch.float64), atol=1.0e-10)
    assert torch.allclose(state[0, 0, 9:12], torch.tensor((2.0, 0.0, 0.0), dtype=torch.float64), atol=1.0e-10)
    assert torch.allclose(state[0, 0, 12:15], torch.tensor((0.0, -3.0, 0.0), dtype=torch.float64), atol=1.0e-10)


def test_object_orientation_is_current_pose_not_goal_error() -> None:
    r"""Object rot6d随当前$R_{ho}$变化；goal error只属于独立task block。"""

    identity = _quat((0.0, 0.0, 1.0), 0.0)
    rotated = _quat((0.0, 0.0, 1.0), math.pi / 2.0)
    zero = torch.zeros(1, 3, dtype=torch.float64)
    semantic = torch.eye(3, dtype=torch.float64)
    first = object_state_in_hand_frame(
        root_quat_wxyz=identity,
        semantic_R_ha=semantic,
        object_pos_w=zero,
        object_quat_wxyz=identity,
        position_anchor_w=zero,
        object_linear_velocity_w=zero,
        object_angular_velocity_w=zero,
    )
    second = object_state_in_hand_frame(
        root_quat_wxyz=identity,
        semantic_R_ha=semantic,
        object_pos_w=zero,
        object_quat_wxyz=rotated,
        position_anchor_w=zero,
        object_linear_velocity_w=zero,
        object_angular_velocity_w=zero,
    )
    assert not torch.equal(first[..., 3:9], second[..., 3:9])
    task = task_state(
        torch.tensor(((0.0, 0.0, 1.0),), dtype=torch.float64),
        torch.tensor(((0.1, -0.2, 0.3),), dtype=torch.float64),
        torch.tensor((-0.4,), dtype=torch.float64),
    )
    assert task.shape == (1, 1, 8)
    assert torch.equal(task[0, 0, 3:6], torch.tensor((0.1, -0.2, 0.3), dtype=torch.float64))
    assert float(task[0, 0, 7].item()) == -0.4  # signed progress保留负值
