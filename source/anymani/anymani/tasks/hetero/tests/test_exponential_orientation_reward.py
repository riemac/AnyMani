r"""指数姿态核的数值锚点、递减性、真实Manager步长补偿与配置往返。"""

import __future__

import ast
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from anymani.tasks.hetero.mdp.task_math import exponential_orientation_step_reward


def _load_definitions(filename, names, namespace):
    r"""执行生产函数/配置类的原AST；仅隔离Isaac导入，不复制其数学实现。"""
    path = Path(__file__).resolve().parents[1] / "mdp" / filename  # 真实任务源码。
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace  # 交付实际函数体，参数与条件分支保持生产语义。


def test_exponential_kernel_values_and_monotonicity():
    r"""固定0、容差、首目标与下一目标起始误差；目标接近时奖励严格增加。"""
    angles = torch.tensor([0., .2, math.pi / 6, math.pi / 6 + .2])  # rad。
    expected = torch.tensor([.9090909091, .4300074825, .1216466963, .0550280019])  # 权重1的单步奖励。
    torch.testing.assert_close(exponential_orientation_step_reward(angles), expected)
    grid = torch.linspace(.001, math.pi, 1000, requires_grad=True)  # SO(3)全部非零角误差。
    values = exponential_orientation_step_reward(grid)
    values.sum().backward()  # 检查真实连续函数在整个误差域的导数符号。
    assert bool(torch.isfinite(values).all()) and bool((values > 0).all())
    assert grid.grad is not None and bool((grid.grad < 0).all())


def test_runtime_dt_compensation_and_configuration_round_trip():
    r"""核选择能经checkpoint字典往返；其余配置不变，Manager积分后数值正确。"""
    theta = torch.tensor([math.pi / 6, .2])  # 模拟command已经测得的两种实时误差。
    command = SimpleNamespace(orientation_error_rad=theta)
    scope = {"__name__": __name__, "dataclass": dataclass, "asdict": asdict, "math": math,
             "exponential_orientation_step_reward": exponential_orientation_step_reward,
             "get_rotation_command": lambda env, name: command}
    _load_definitions("rewards.py", {"track_orientation_exponential"}, scope)
    reward = scope["track_orientation_exponential"]  # 真实Manager接口中的dt补偿。
    torch.testing.assert_close(reward(SimpleNamespace(step_dt=.05), "goal_pose") * .05,
                               exponential_orientation_step_reward(theta))
    _load_definitions("orientation_goal.py", {"OrientationGoalCfg", "configure_orientation_goal"}, scope)
    cfg_type = scope["OrientationGoalCfg"]
    baseline = cfg_type().to_dict()  # 倒数核的规范默认配置。
    candidate = cfg_type(kernel="exponential").to_dict()
    assert {k for k in candidate if candidate[k] != baseline[k]} == {"kernel"}
    assert cfg_type(**candidate).to_dict() == candidate  # 固定评价按checkpoint恢复同一数学形式。
    scope.update(RewardTermCfg=SimpleNamespace, rewards=SimpleNamespace(track_orientation_exponential=reward))
    terms = {name: SimpleNamespace() for name in ("goal_success", "failure", "bad_finger_non_tip_contact", "joint_pose_anchor")}
    terms["speed_band"] = SimpleNamespace(params={})  # 当前配置函数修改的速度带参数。
    env = SimpleNamespace(rewards=SimpleNamespace(**terms), commands=SimpleNamespace(goal_pose=SimpleNamespace()))
    scope["configure_orientation_goal"](env, cfg_type(**candidate), training=True, adr=None)
    assert env.rewards.orientation_tracking.func is reward and env.rewards.orientation_tracking.weight == 1
    assert env.rewards.orientation_tracking.params == {"command_name": "goal_pose", "slope_rad_inv": 4., "denominator_epsilon": .1}
    assert env.commands.goal_pose.orientation_success_threshold_rad == .2 and env.rewards.goal_success.weight == 250
