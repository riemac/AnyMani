r"""Official LEAP reward 的 tensor-level 数值 contract。

这些测试不启动 Isaac Sim，而是构造最小 ManagerBased fake env，直接核对七个 reward 分量、阈值闭开区间
与 IsaacLab ``RewardManager`` 的 $\Delta t$ 缩放语义。这样测试针对的是进入 PPO 的数学量，而不是仅检查
源码里是否出现过某个函数名。
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

REWARDS_PATH = Path(__file__).resolve().parents[1] / "mdp" / "rewards.py"
r"""被测 reward 源文件；path import 避免 contract test 触发完整 Isaac/Kit/USD runtime。"""

LEAPHAND_OFFICIAL_ADR_CFG_PATH = REWARDS_PATH.parents[1] / "config" / "leaphand" / "leaphand_adr_env_cfg.py"
r"""主线 official ADR 配置；数值测试从其 AST 读取实际 scalar params。"""


def _official_reward_scalar_params() -> dict[str, Any]:
    r"""从主线 ``RewTerm.params`` 提取可字面求值的 official reward 参数。

    ``object_cfg=SceneEntityCfg("object")`` 是 runtime selector 而非标量，因此保留被测函数的同名默认值；
    其余字符串、权重、阈值、圈数与 dt 开关均直接来自当前配置 AST。这样配置值漂移会进入数值调用，
    不会被测试函数自身的默认参数掩盖。

    Returns:
        dict[str, Any]: 可直接展开给 ``OfficialLeapReward.__call__`` 的 scalar params。
    """

    tree = ast.parse(LEAPHAND_OFFICIAL_ADR_CFG_PATH.read_text(encoding="utf-8"))  # 只解析源码，不 import Kit。
    params_node: ast.Dict | None = None  # 目标是 ``LeapHandOfficialADRRewardsCfg.official_reward.params``。

    # 精确定位配置类与 ``official_reward = RewTerm(..., params={...})``，避免误读其他 reward group。
    for node in tree.body:
        if not (isinstance(node, ast.ClassDef) and node.name == "LeapHandOfficialADRRewardsCfg"):
            continue
        for statement in node.body:
            if not (
                isinstance(statement, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "official_reward" for target in statement.targets)
                and isinstance(statement.value, ast.Call)
            ):
                continue
            params_keyword = next((keyword for keyword in statement.value.keywords if keyword.arg == "params"), None)
            if params_keyword is not None and isinstance(params_keyword.value, ast.Dict):
                params_node = params_keyword.value

    assert params_node is not None, "official reward params dict not found in LEAP ADR cfg"
    scalar_params: dict[str, Any] = {}
    for key_node, value_node in zip(params_node.keys, params_node.values, strict=True):
        if not (isinstance(key_node, ast.Constant) and isinstance(key_node.value, str)):
            continue
        try:
            scalar_params[key_node.value] = ast.literal_eval(value_node)  # 只接受 literal，不执行配置代码。
        except (TypeError, ValueError):
            continue  # ``SceneEntityCfg(...)`` 由函数的等价默认 selector 承担。

    required = {
        "action_term_name",
        "command_name",
        "dist_reward_scale",
        "rot_reward_scale",
        "rot_eps",
        "action_penalty_scale",
        "pose_diff_penalty_scale",
        "success_tolerance",
        "position_success_threshold",
        "reach_goal_bonus",
        "fall_dist",
        "fall_penalty",
        "z_rotation_steps",
        "divide_by_step_dt",
    }
    assert scalar_params.keys() >= required  # 配置缺项时立即失败，而不是静默退回实现默认值。
    return scalar_params


def _load_rewards_module() -> types.ModuleType:
    r"""以最小 IsaacLab 类型壳加载 ``rewards.py``，保留全部 torch 数值逻辑。

    ``rewards.py`` 的公式只依赖 torch；``RigidObject``、``SceneEntityCfg`` 与 ``ManagerTermBase`` 在本测试
    中只承担类型/容器职责。对这些接口做窄 stub 可验证 reward 数学而不导入 ``pxr``，因此仍属于默认
    contract suite，而不是 Isaac Sim runtime smoke。

    Returns:
        types.ModuleType: 隔离加载的 official reward 模块。
    """

    # 四元数角距离在每个 test 内按预设 batch monkeypatch；这里先提供可替换的模块属性。
    math_stub = types.ModuleType("isaaclab.utils.math")
    setattr(math_stub, "quat_error_magnitude", lambda *_args: None)

    # ManagerTermBase 只需保存 cfg/env；这正是被测 class 构造器和 ``_env.step_dt`` 所依赖的接口。
    class ManagerTermBase:
        r"""测试用 ManagerTermBase 最小壳。"""

        def __init__(self, cfg: SimpleNamespace, env: SimpleNamespace):
            self.cfg = cfg  # 保留 IsaacLab manager term 的配置引用。
            self._env = env  # combined reward 用该引用读取 policy step duration。

    class SceneEntityCfg:
        r"""只保留 entity name 的测试用 scene selector。"""

        def __init__(self, name: str):
            self.name = name  # scene mapping key，例如 ``object``。

    isaaclab_stub = types.ModuleType("isaaclab")
    utils_stub = types.ModuleType("isaaclab.utils")
    setattr(utils_stub, "math", math_stub)
    assets_stub = types.ModuleType("isaaclab.assets")
    setattr(assets_stub, "RigidObject", object)
    managers_stub = types.ModuleType("isaaclab.managers")
    setattr(managers_stub, "ManagerTermBase", ManagerTermBase)
    setattr(managers_stub, "SceneEntityCfg", SceneEntityCfg)

    # 临时覆盖 import graph，执行完目标文件后恢复进程原状态，避免影响同一 pytest session 的其他测试。
    replacements = {
        "isaaclab": isaaclab_stub,
        "isaaclab.assets": assets_stub,
        "isaaclab.managers": managers_stub,
        "isaaclab.utils": utils_stub,
        "isaaclab.utils.math": math_stub,
    }
    previous = {name: sys.modules.get(name) for name in replacements}
    try:
        sys.modules.update(replacements)
        spec = importlib.util.spec_from_file_location("_inhand_official_rewards_for_test", REWARDS_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


rewards = _load_rewards_module()  # 单次加载后，各 test 只替换模块内的 SO(3) 距离函数。
OFFICIAL_REWARD_PARAMS = _official_reward_scalar_params()  # 数值 contract 绑定当前主线 cfg，而非函数 defaults。


class _Scene(dict):
    r"""同时提供 mapping 访问与 per-env origin 的最小 scene stub。"""

    def __init__(self, object_asset: SimpleNamespace, env_origins: torch.Tensor):
        super().__init__({"object": object_asset})  # reward 通过 ``scene[object_cfg.name]`` 读取刚体。
        self.env_origins = env_origins  # 世界系到环境系的平移原点，形状 $[N,3]$，单位 m。


def _fake_env(step_dt: float) -> tuple[SimpleNamespace, torch.Tensor]:
    r"""构造覆盖 success/fall/z-spin 边界及非零正则项的四环境 batch。

    Batch 四行分别覆盖：

    1. success 的双 ``<=`` 边界，同时令 $\omega_z=0.25$ 验证 spin 下界为开区间；
    2. fall 的 ``>=`` 边界，同时令 $\omega_z=1.5$ 验证 spin 上界为开区间；
    3. 正常非 success/fall 样本，且角速度位于 spin 开区间内部；
    4. 零姿态误差与非零 action/pregrasp 正则样本。

    Args:
        step_dt (float): policy step 时长 $\Delta t$，单位 s。

    Returns:
        tuple[SimpleNamespace, torch.Tensor]: fake env 与预设 SO(3) 角距离，batch 维均为 4。
    """

    # 目标位置统一为环境原点；物体只沿 x 轴偏移，因此 L2 距离就是下列标量。
    goal_dist = torch.tensor([0.025, 0.070, 0.040, 0.010], dtype=torch.float32)  # $d_p$，单位 m。
    object_pos_w = torch.zeros(4, 3, dtype=torch.float32)  # 物体世界系位置，形状 $[N,3]$。
    object_pos_w[:, 0] = goal_dist  # env origin 为零，所以 $p_o^e=p_o^w$。

    # 姿态张量只负责满足接口；角距离由 monkeypatch 固定，以精确测试 0.2 rad 边界。
    identity_quat = torch.zeros(4, 4, dtype=torch.float32)  # wxyz 四元数，形状 $[N,4]$。
    identity_quat[:, 0] = 1.0  # 单位四元数 $(1,0,0,0)$。
    rot_dist = torch.tensor([0.200, 0.100, 0.210, 0.000], dtype=torch.float32)  # $d_{SO(3)}$，单位 rad。

    # z-spin 的前两行恰好落在开区间边界，后两行位于奖励窗口内部。
    root_ang_vel_w = torch.zeros(4, 3, dtype=torch.float32)  # 世界系角速度，形状 $[N,3]$，单位 rad/s。
    root_ang_vel_w[:, 2] = torch.tensor([0.25, 1.50, 0.50, 1.00])  # $\omega_z$，单位 rad/s。

    # 选择非平凡 executed action，使 $\lVert a_t^{exec}\rVert_2^2$ 在每行都可独立核对。
    executed_actions = torch.tensor(
        [[1.0, 2.0], [0.0, 1.0], [2.0, -1.0], [0.5, -0.5]], dtype=torch.float32
    )  # 规范化 action，形状 $[N,2]$。
    current_targets = torch.tensor(
        [[1.0, 0.0], [0.0, 2.0], [1.0, -1.0], [0.5, 0.5]], dtype=torch.float32
    )  # 当前 command target $q_t^{cmd}$，单位 rad。
    pregrasp_targets = torch.zeros_like(current_targets)  # pregrasp anchor $q^{pregrasp}$，单位 rad。

    # 把 rigid object、command term 与 action term 按 reward 的公开 runtime contract 组装起来。
    object_asset = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=object_pos_w,
            root_quat_w=identity_quat.clone(),
            root_ang_vel_w=root_ang_vel_w,
        )
    )
    command_term = SimpleNamespace(pos_command_e=torch.zeros(4, 3), quat_command_w=identity_quat.clone())
    action_term = SimpleNamespace(
        executed_actions=executed_actions,
        current_targets=current_targets,
        pregrasp_targets=pregrasp_targets,
    )
    env = SimpleNamespace(
        num_envs=4,
        device="cpu",
        step_dt=float(step_dt),
        scene=_Scene(object_asset, torch.zeros(4, 3)),
        command_manager=SimpleNamespace(get_term=lambda _name: command_term),
        action_manager=SimpleNamespace(get_term=lambda _name: action_term),
    )
    return env, rot_dist


def _combined_term(env: SimpleNamespace) -> Any:
    r"""用主线配置参数初始化 combined term，保持 reset-time 缓存接口与 runtime 一致。"""

    return rewards.OfficialLeapReward(SimpleNamespace(params=OFFICIAL_REWARD_PARAMS), env)


def test_combined_reward_matches_all_seven_terms_and_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Combined raw reward 应逐行等于七项公式，并保持 success/fall/spin 的边界语义。"""

    env, rot_dist = _fake_env(step_dt=0.05)  # 20 Hz policy step，$\Delta t=0.05$ s。
    monkeypatch.setattr(rewards.math_utils, "quat_error_magnitude", lambda *_args: rot_dist)

    call_params = {**OFFICIAL_REWARD_PARAMS, "divide_by_step_dt": False}  # 只翻转 N031 dt ablation。
    raw_reward = _combined_term(env)(env, **call_params)  # $r_t^{official}$，尚未除以 $\Delta t$。
    object_asset = env.scene["object"]  # 读取 fake batch 以独立构造解析解。
    action_term = env.action_manager.get_term("hand_joint_pos")  # action/pregrasp 数值锚点。
    goal_dist = object_asset.data.root_pos_w[:, 0]  # 本 fixture 仅沿 x 偏移，故等于位置 L2 距离。
    action_l2 = torch.sum(action_term.executed_actions**2, dim=-1)  # $\lVert a_t^{exec}\rVert_2^2$。
    pregrasp_l2 = torch.sum(action_term.current_targets**2, dim=-1)  # $\lVert q_t^{cmd}-q^{pre}\rVert_2^2$。

    # 显式验证闭/开边界，避免公式总和偶然相等却掩盖 indicator 判据漂移。
    success = (rot_dist <= float(call_params["success_tolerance"])) & (
        goal_dist <= float(call_params["position_success_threshold"])
    )  # success 两个阈值都为闭区间。
    fall = goal_dist >= float(call_params["fall_dist"])  # fall 的 7 cm 阈值为闭区间。
    omega_z = object_asset.data.root_ang_vel_w[:, 2]  # 世界系 z 轴角速度，单位 rad/s。
    z_spin = (omega_z > 0.25) & (omega_z < 1.5)  # spin 奖励窗口两端均为开区间。
    assert success.tolist() == [True, False, False, True]
    assert fall.tolist() == [False, True, False, False]
    assert z_spin.tolist() == [False, False, True, True]

    expected = (
        float(call_params["dist_reward_scale"]) * goal_dist
        + float(call_params["rot_reward_scale"]) / (torch.abs(rot_dist) + float(call_params["rot_eps"]))
        + float(call_params["action_penalty_scale"]) * action_l2
        + float(call_params["pose_diff_penalty_scale"]) * pregrasp_l2
        + float(call_params["reach_goal_bonus"]) * success.float()
        + float(call_params["fall_penalty"]) * fall.float()
        + z_spin.float()
    )  # 七项 official DirectRLEnv per-step reward；所有可配置系数均来自主线 cfg AST。
    torch.testing.assert_close(raw_reward, expected)


@pytest.mark.parametrize("step_dt", [0.05, 1.0 / 30.0])
def test_combined_dt_switch_matches_reward_manager_scaling(
    monkeypatch: pytest.MonkeyPatch,
    step_dt: float,
) -> None:
    r"""dt 开关应只决定 term 是否预先除 $\Delta t$，再由 RewardManager 统一乘回 $\Delta t$。"""

    env, rot_dist = _fake_env(step_dt=step_dt)  # 分别覆盖 20 Hz 与 30 Hz policy step。
    monkeypatch.setattr(rewards.math_utils, "quat_error_magnitude", lambda *_args: rot_dist)
    term = _combined_term(env)  # 同一个公式对象，只翻转 N030/N031 的 dt 开关。

    raw = term(env, **{**OFFICIAL_REWARD_PARAMS, "divide_by_step_dt": False})  # N031 输出 $r_t^{official}$。
    aligned = term(
        env, **{**OFFICIAL_REWARD_PARAMS, "divide_by_step_dt": True}
    )  # official 输出 $r_t^{official}/\Delta t$。
    torch.testing.assert_close(aligned, raw / step_dt)  # callable 层只相差 $1/\Delta t$。
    manager_aligned = aligned * step_dt  # official manager 输出恢复 DirectRLEnv 单步量 $r_t$。
    manager_unaligned = raw * step_dt  # N031 manager 输出保留 $\Delta t r_t$ 语义。
    torch.testing.assert_close(manager_aligned, raw)
    torch.testing.assert_close(manager_unaligned, manager_aligned * step_dt)


@pytest.mark.parametrize("step_dt", [0.05, 1.0 / 30.0])
def test_weighted_split_terms_recompose_aligned_combined_reward(
    monkeypatch: pytest.MonkeyPatch,
    step_dt: float,
) -> None:
    r"""七个 split terms 按 official 权重求和后应等于 ``divide_by_step_dt=True`` combined term。"""

    env, rot_dist = _fake_env(step_dt=step_dt)  # 同时证伪 split helper 是否误写死为 20 Hz scale。
    monkeypatch.setattr(rewards.math_utils, "quat_error_magnitude", lambda *_args: rot_dist)
    params = OFFICIAL_REWARD_PARAMS  # 权重、阈值与 term 名称均来自 active LEAP config。

    split_sum = (
        float(params["dist_reward_scale"])
        * rewards.official_goal_distance(env, command_name=str(params["command_name"]))
        + float(params["rot_reward_scale"])
        * rewards.official_orientation(env, command_name=str(params["command_name"]), rot_eps=float(params["rot_eps"]))
        + float(params["action_penalty_scale"])
        * rewards.official_action_l2(env, action_term_name=str(params["action_term_name"]))
        + float(params["pose_diff_penalty_scale"])
        * rewards.official_pregrasp_l2(env, action_term_name=str(params["action_term_name"]))
        + float(params["reach_goal_bonus"])
        * rewards.official_success_bonus(
            env,
            command_name=str(params["command_name"]),
            success_tolerance=float(params["success_tolerance"]),
            position_success_threshold=float(params["position_success_threshold"]),
        )
        + float(params["fall_penalty"])
        * rewards.official_fall_penalty(
            env, command_name=str(params["command_name"]), fall_dist=float(params["fall_dist"])
        )
        + rewards.official_z_spin_bonus(env)
    )  # 每项均先除 $\Delta t$，权重与阈值绑定 active config。
    combined = _combined_term(env)(env, **params)  # active cfg 的 $r_t^{official}/\Delta t$。
    torch.testing.assert_close(split_sum, combined)
