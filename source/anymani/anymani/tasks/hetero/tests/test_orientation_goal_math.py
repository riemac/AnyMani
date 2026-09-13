r"""姿态推进与位置奖金的独立性、倒数奖励单步尺度以及固定目标序列。"""

import __future__
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from anymani.tasks.hetero.mdp.task_math import (
    axis_angle_from_quaternion_wxyz,
    hand_axis_to_world,
    inverse_orientation_step_reward,
    moving_goal_quaternion,
    orientation_tracking_flags,
)


def test_position_error_blocks_bonus_but_not_orientation_advance():
    r"""同一角度达标的两个位置，一个合格、另一个继续推进但不领奖金。"""
    advance, bonus = orientation_tracking_flags(torch.tensor([.1, .1, .3]), torch.tensor([.01, .04, .01]))
    assert advance.tolist() == [True, True, False]
    assert bonus.tolist() == [True, False, False]
    value = inverse_orientation_step_reward(torch.tensor([0., .1, .4]), .1)
    torch.testing.assert_close(value, torch.tensor([10., 5., 2.]))  # 尚未加权的真实每步奖励。


def test_previous_goal_sequence_retains_fixed_angles():
    r"""目标从30°累加到60°，不继承实际物体尚余的跟踪误差。"""
    q0 = torch.tensor([[1., 0., 0., 0.]])
    axis = torch.tensor([[0., 0., 1.]])
    first = moving_goal_quaternion(q0, axis, subgoal_angle_rad=math.pi / 6)
    second = moving_goal_quaternion(first, axis, subgoal_angle_rad=math.pi / 6)
    torch.testing.assert_close(axis_angle_from_quaternion_wxyz(second).norm(dim=-1), torch.tensor([math.pi / 3]))


def _command_method(name):
    r"""执行真实command方法的AST，省去仅为测试状态机而启动Isaac。"""
    path = Path(__file__).resolve().parents[1] / 'mdp/commands.py'
    cls = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef) and node.name == 'HeterogeneousRotationCommand')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {'torch': torch, 'math': math, 'hand_axis_to_world': hand_axis_to_world, 'moving_goal_quaternion': moving_goal_quaternion}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec', flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


def test_actual_command_advances_angle_only_and_pays_qualified_goal_once():
    r"""直接调用真实推进逻辑；位置不合格的第二环境照常换目标但不获得奖金计数。"""
    calls = []
    state = SimpleNamespace(goal_advance_pulse=torch.tensor([True, True, False]),
                            goal_success_pulse=torch.tensor([True, False, False]),
                            goal_success_count=torch.zeros(3), goal_advance_count=torch.zeros(3),
                            subgoal_throughput_per_horizon_s=torch.zeros(3), command_counter=torch.zeros(3), time_left=torch.zeros(3),
                            cfg=SimpleNamespace(horizon_s=120., resampling_time_range=(1e6, 1e6)),
                            _resample_command=lambda ids: calls.append(ids.tolist()))
    update = _command_method('_update_command')
    update(state)
    update(state)
    assert calls == [[0, 1]]  # 同一达标事件不重复推进或发奖金。
    assert state.goal_success_count.tolist() == [1., 0., 0.]
    assert state.goal_advance_count.tolist() == [1., 1., 0.]


def test_actual_resample_uses_previous_goal_not_current_object():
    r"""物体仅转20度时，旧30度目标仍推进到60度，避免目标跟踪误差积累。"""
    q0 = torch.tensor([[1., 0., 0., 0.]])
    axis = torch.tensor([[0., 0., 1.]])
    old_goal = moving_goal_quaternion(q0, axis, subgoal_angle_rad=math.pi / 6)
    actual = moving_goal_quaternion(q0, axis, subgoal_angle_rad=math.pi / 9)
    state = SimpleNamespace(_as_ids=lambda ids: ids, goal_quat_w=old_goal, axis_w=torch.zeros_like(axis),
                            axis_h=axis, semantic_R_ha=torch.eye(3), robot=SimpleNamespace(data=SimpleNamespace(root_quat_w=q0)),
                            object=SimpleNamespace(data=SimpleNamespace(root_quat_w=actual)),
                            cfg=SimpleNamespace(goal_reference='previous_goal', subgoal_angle_rad=math.pi / 6),
                            _refresh_goal_state=lambda ids: None)
    _command_method('_resample_command')(state, torch.tensor([0]))
    torch.testing.assert_close(axis_angle_from_quaternion_wxyz(state.goal_quat_w).norm(dim=-1), torch.tensor([math.pi / 3]))
