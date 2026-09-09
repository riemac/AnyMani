r"""Heterogeneous tactile-rotation纯$SO(3)$/reward/metric边界测试。"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from anymani.tasks.hetero.mdp.task_math import (
    contact_role_reward,
    equal_asset_mean,
    full_pose_keypoint_reward,
    goal_errors_and_success,
    hand_axis_to_world,
    impulse_to_rate,
    moving_goal_quaternion,
    projected_space_rotation_delta,
    quaternion_from_angle_axis_wxyz,
    rotation_frontier_update,
    task_termination_flags,
)


def _axis_quaternion(axis: tuple[float, float, float], angle: float, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    r"""构造单batch angle-axis fixture。"""

    return quaternion_from_angle_axis_wxyz(
        torch.tensor((angle,), dtype=dtype),
        torch.tensor((axis,), dtype=dtype),
    )


def test_signed_rotation_is_quaternion_sign_invariant_and_roundtrip_cancels() -> None:
    r"""正/负progress保留符号，$q\sim-q$，往返净转角为零。"""

    identity = _axis_quaternion((0.0, 0.0, 1.0), 0.0)
    positive = _axis_quaternion((0.0, 0.0, 1.0), 0.2)
    axis = torch.tensor(((0.0, 0.0, 1.0),), dtype=torch.float64)
    forward = projected_space_rotation_delta(identity, positive, axis)
    backward = projected_space_rotation_delta(positive, identity, axis)
    flipped = projected_space_rotation_delta(-identity, positive, axis)
    assert torch.allclose(forward, torch.tensor((0.2,), dtype=torch.float64), atol=1.0e-10)
    assert torch.allclose(backward, torch.tensor((-0.2,), dtype=torch.float64), atol=1.0e-10)
    assert torch.allclose(flipped, forward, atol=1.0e-10)
    assert torch.allclose(forward + backward, torch.zeros_like(forward), atol=1.0e-10)
    assert float((backward / (2.0 * math.pi)).item()) < 0.0  # signed turns不能positive clamp


def test_nonidentity_hand_frame_and_root_transform_fixed_axis() -> None:
    r"""Counterfactual同时使用非identity$R_{ha}$与$R_{wa}$，禁止把hand +z当world +z。"""

    axis_h = torch.tensor(((0.0, 0.0, 1.0),), dtype=torch.float64)
    # $R_{ha}=R_x(90^\circ)$，故$v_a=R_{ha}^Tv_h=(0,1,0)$。
    semantic_R_ha = torch.tensor(
        ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)), dtype=torch.float64
    )
    # $R_{wa}=R_z(90^\circ)$，将asset +y映成world -x。
    root_quaternion = _axis_quaternion((0.0, 0.0, 1.0), math.pi / 2.0)
    axis_w = hand_axis_to_world(axis_h, root_quaternion, semantic_R_ha)
    assert torch.allclose(axis_w, torch.tensor(((-1.0, 0.0, 0.0),), dtype=torch.float64), atol=1.0e-10)


def test_rotation_frontier_counts_only_new_positive_30_degree_intervals() -> None:
    r"""往返重越旧阈值不重复计数；单步跨两档时$\Delta K=2$而非一个布尔近似。"""

    interval = math.pi / 6.0  # 物理前沿宽度$\delta=30^\circ$
    maximum = torch.zeros(1, dtype=torch.float64)  # $M_0=0$ rad
    count = torch.zeros(1, dtype=torch.long)  # $K_0=0$
    observed: list[tuple[int, int, bool]] = []

    # 依次：首次越过一档、退回、重越旧档、再跨两档、反向；预期只在新历史前沿产生增量。
    for angle in (interval, 0.1, interval + 0.01, 3.0 * interval + 0.01, -0.5):
        maximum, count, delta, pulse = rotation_frontier_update(
            torch.tensor((angle,), dtype=torch.float64),
            maximum,
            count,
            frontier_interval_rad=interval,
        )
        observed.append((int(count.item()), int(delta.item()), bool(pulse.item())))

    assert observed == [(1, 1, True), (1, 0, False), (1, 0, False), (3, 2, True), (3, 0, False)]
    assert torch.allclose(maximum, torch.tensor((3.0 * interval + 0.01,), dtype=torch.float64))


def test_moving_goal_is_left_multiplied_from_current_pose() -> None:
    r"""30°goal是world-space left multiplication，不从旧goal累积。"""

    current = _axis_quaternion((1.0, 0.0, 0.0), 0.4)
    axis_w = torch.tensor(((0.0, 0.0, 1.0),), dtype=torch.float64)
    goal = moving_goal_quaternion(current, axis_w)
    delta = projected_space_rotation_delta(current, goal, axis_w)
    assert torch.allclose(delta, torch.tensor((math.pi / 6.0,), dtype=torch.float64), atol=1.0e-10)


def test_success_uses_strict_orientation_and_position_boundaries() -> None:
    r"""Orientation`<5 mm`与position`<25 mm`均为严格门。"""

    dtype = torch.float64
    identity = _axis_quaternion((0.0, 0.0, 1.0), 0.0, dtype=dtype)
    # z rotation六点平均距离$d=(4r/3)\sin(\theta/2)$，反解目标$d$。
    radius = 0.05
    angle_below = 2.0 * math.asin(3.0 * 0.00499 / (4.0 * radius))
    angle_equal = 2.0 * math.asin(3.0 * 0.00500 / (4.0 * radius))
    goal_below = _axis_quaternion((0.0, 0.0, 1.0), angle_below, dtype=dtype)
    goal_equal = _axis_quaternion((0.0, 0.0, 1.0), angle_equal, dtype=dtype)
    anchor = torch.zeros(1, 3, dtype=dtype)
    position_below = torch.tensor(((0.02499, 0.0, 0.0),), dtype=dtype)
    position_equal = torch.tensor(((0.02500, 0.0, 0.0),), dtype=dtype)
    orientation_error, _, _, success = goal_errors_and_success(
        position_below, identity, anchor, goal_below
    )
    assert float(orientation_error.item()) < 0.005
    assert bool(success.item())
    orientation_error, _, _, success = goal_errors_and_success(
        position_below, identity, anchor, goal_equal
    )
    assert torch.allclose(orientation_error, torch.tensor((0.005,), dtype=dtype), atol=1.0e-12)
    assert not bool(success.item())
    _, position_error, _, success = goal_errors_and_success(position_equal, identity, anchor, identity)
    assert torch.equal(position_error, torch.tensor((0.025,), dtype=dtype))
    assert not bool(success.item())


def test_termination_uses_closed_drop_and_signed_normal_alignment() -> None:
    r"""7 cm边界终止；反向normal不能被absolute alignment放过。"""

    position_error = torch.tensor((0.06999, 0.07000))
    alignment = torch.tensor((1.0, -1.0))
    drop, axis_failure = task_termination_flags(position_error, alignment)
    assert drop.tolist() == [False, True]
    assert axis_failure.tolist() == [False, True]


def test_full_pose_reward_and_impulse_rate_have_expected_units() -> None:
    r"""Exact goal的shape reward为1；one-step impulse除以policy dt形成rate。"""

    identity = _axis_quaternion((0.0, 0.0, 1.0), 0.0, dtype=torch.float32)
    position = torch.zeros(1, 3)
    reward = full_pose_keypoint_reward(position, identity, position, identity)
    assert torch.allclose(reward, torch.ones_like(reward), atol=1.0e-7)
    rate = impulse_to_rate(torch.tensor((True, False)), 0.05)
    assert torch.equal(rate, torch.tensor((20.0, 0.0)))


def _pose_reward_wrapper(command: SimpleNamespace):
    r"""提取真实reward wrapper并绑定可控物体状态，避免纯数学测试加载Isaac Articulation。"""

    path = Path(__file__).resolve().parents[1] / "mdp" / "rewards.py"  # 实际MDP wrapper
    definition = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "pose_keypoint_reward"
    )  # 仅提取待验证函数
    namespace = {
        "torch": torch,
        "ManagerBasedRLEnv": object,
        "get_rotation_command": lambda _env, _name: command,
        "full_pose_keypoint_reward": full_pose_keypoint_reward,
    }  # 全局依赖指向真实纯数学kernel；只替代环境状态读取
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)  # 执行原函数体
    return namespace["pose_keypoint_reward"]  # 两个测试共享同一生产wrapper加载方法


def test_position_only_kernel_is_rotation_invariant_and_decreases_with_center_distance() -> None:
    r"""位置核应等于$\operatorname{sech}^2(25d)$，对姿态自由、对中心偏移单调。

    $d$单位m，系数25单位$m^{-1}$；此处返回无量纲kernel，RewardManager随后乘weight与policy dt。
    90°旋转在位置模式可获满分，但strict goal继续以0.05 m半径判定姿态误差。
    """

    dtype = torch.float64  # 用高精度隔离测量公式的差异
    distance = torch.tensor([0.0, 0.02, 0.07], dtype=dtype)  # 从中心到物理位置终止尺度，m
    anchor = torch.zeros(3, 3, dtype=dtype)  # 三个环境的共同位置anchor
    position = anchor.clone()  # 构造纯平移误差，无坐标框架变换
    position[:, 0] = distance  # 世界x方向偏移，范数恰为d
    identity = _axis_quaternion((0.0, 0.0, 1.0), 0.0).expand(3, -1)  # 目标四元数wxyz
    rotated = _axis_quaternion((0.0, 0.0, 1.0), math.pi / 2).expand(3, -1)  # 同一90°姿态误差
    command = SimpleNamespace(
        object=SimpleNamespace(data=SimpleNamespace(root_pos_w=position, root_quat_w=rotated)),
        position_anchor_w=anchor,
        goal_quat_w=identity,
        cfg=SimpleNamespace(keypoint_radius_m=0.05),
    )  # strict goal与reward共享状态，但各自使用规定的测量半径
    reward = _pose_reward_wrapper(command)  # 真实wrapper与纯math实现
    measured = reward(None, "goal_pose", position_only=True)  # 新模式：只有中心位置误差
    torch.testing.assert_close(
        measured, 1.0 / torch.cosh(25.0 * distance).square(), atol=1e-14, rtol=1e-14
    )  # 核函数解析式
    assert measured[0] == 1 and bool((measured[:-1] > measured[1:]).all())  # 中心满分，距离增大则奖励下降
    command.object.data.root_quat_w = identity  # 仅改变物体姿态，位置数据逐值保留
    torch.testing.assert_close(
        reward(None, "goal_pose", position_only=True), measured, atol=0, rtol=0
    )  # 姿态不影响位置核
    assert command.cfg.keypoint_radius_m == 0.05  # reward调用不改变strict goal配置
    _, _, _, success = goal_errors_and_success(
        position, rotated, anchor, identity, keypoint_radius_m=command.cfg.keypoint_radius_m
    )  # 原姿态门
    assert not bool(success.any())  # 位置模式的满分不等于strict姿态成功


def test_default_full_pose_wrapper_preserves_existing_values() -> None:
    r"""默认模式与显式False都逐值恢复原full-pose核，正半径继续测量位置与姿态。"""

    position = torch.tensor([[0.01, -0.02, 0.005]], dtype=torch.float64)  # 非零位置误差，m
    anchor = torch.zeros_like(position)  # 固定世界anchor
    identity = _axis_quaternion((0.0, 0.0, 1.0), 0.0)  # 目标姿态
    rotated = _axis_quaternion((1.0, 0.0, 0.0), 0.3)  # 绕x轴姿态偏差，rad
    command = SimpleNamespace(
        object=SimpleNamespace(data=SimpleNamespace(root_pos_w=position, root_quat_w=rotated)),
        position_anchor_w=anchor,
        goal_quat_w=identity,
        cfg=SimpleNamespace(keypoint_radius_m=0.05),
    )  # 非退化六点半径
    reward = _pose_reward_wrapper(command)  # 默认函数应保持旧任务语义
    expected = full_pose_keypoint_reward(position, rotated, anchor, identity, keypoint_radius_m=0.05)  # 已有数学定义
    torch.testing.assert_close(reward(None, "goal_pose"), expected, atol=0, rtol=0)  # 隐式默认
    torch.testing.assert_close(reward(None, "goal_pose", position_only=False), expected, atol=0, rtol=0)  # 显式默认


def test_contact_reward_masks_ghost_roles_and_excludes_palm() -> None:
    r"""Ghost TIP/non-tip不进入count；palm不属于该函数的坏接触输入。"""

    tip_bits = torch.tensor(((True, True, True, False),))
    tip_mask = torch.tensor(((True, False, True, False),))  # 第二个contact是ghost
    non_tip_bits = torch.tensor(((False, True, True),))
    non_tip_mask = torch.tensor(((True, False, False),))  # 两个bad bits都在ghost slots
    good, bad = contact_role_reward(tip_bits, tip_mask, non_tip_bits, non_tip_mask)
    assert torch.equal(good, torch.tensor((1.0,)))
    assert torch.equal(bad, torch.tensor((0.0,)))


def test_equal_asset_mean_is_not_episode_weighted() -> None:
    r"""Asset A均值2、B均值10时equal-asset结果为6，而非$14/3$。"""

    metric_sum = torch.tensor((4.0, 10.0))
    episode_count = torch.tensor((2.0, 1.0))
    result = equal_asset_mean(metric_sum, episode_count)
    assert torch.equal(result, torch.tensor(6.0))
    assert not torch.isclose(result, metric_sum.sum() / episode_count.sum())
