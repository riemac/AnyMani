r"""IsaacSim smoke for official LEAP raw-root pose and `{h}` semantic-frame calibration.

本文件是运行时 smoke，不属于默认 `pytest` contract suite。它必须通过显式路径运行，
因为模块导入阶段会启动 `AppLauncher(headless=True)`：

```bash
cd /home/hac/isaac/AnyMani
source /home/hac/isaac/env_isaaclab/bin/activate
timeout --kill-after=20s 240s /home/hac/isaac/IsaacLab/isaaclab.sh -p -m pytest \
  source/anymani/anymani/smokes/isaacsim/test_leap_hand_frame_calibration.py -q -s
```

该 smoke 只验证一个 sim2sim 先行语义：official LEAP 的 raw asset/root frame `{a}`
仍由 `InitialStateCfg.pos/rot` 摆到任务姿态，而 `{h}` 只是固定附着在 `{a}` 上的
hand semantic frame。测试不训练、不 reset MDP、不改 reward / obs / action，只检查
IsaacLab runtime 看到的 root pose 是否与纯数学工具
`tools/leap_hand_frame_calibration.py` 中的 $T_{ah}$ / $T_{ea}^{init}$ 推导一致。
"""

from __future__ import annotations

# ruff: noqa: I001
# IsaacLab smoke 必须先创建 `AppLauncher`，再导入 `omni` / Isaac runtime 相关模块；
# 因此本文件有意不服从普通 isort 的“所有 import 必须在文件顶部”排序模型。

from isaaclab.app import AppLauncher

# headless smoke 不打开 GUI；若需要肉眼看轴，可后续在 viewer 脚本里画 frame marker。
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import pytest
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass

from anymani.robots.leap import LEAP_HAND_CFG
from anymani.tools.leap_hand_frame_calibration import (
    USER_P_AH,
    USER_ROOT_POS_E,
    USER_ROOT_QUAT_WXYZ,
    USER_X_H_IN_A,
    USER_Y_H_IN_A,
    USER_Z_H_IN_A,
    calibrate_frame,
    matrix_from_axis_columns,
    quat_wxyz_to_matrix,
)

SMOKE_NUM_ENVS = 2
r"""最小 cloned env 数；两个 env 可验证 `{ENV_REGEX_NS}/Robot` 展开和 env origin 平移。"""

SMOKE_ENV_SPACING = 0.75
r"""与现有 LEAP viewer / GM env 接近的 env spacing，单位 m。"""

POSE_ATOL = 1.0e-5
r"""root pose / semantic frame 数值比较容差；这里只比较静态 spawn/reset 结果。"""


def _make_leap_root_pose_cfg(prim_path: str) -> ArticulationCfg:
    r"""构造只覆盖 root pose 的 official LEAP articulation cfg。

    Args:
        prim_path (str): IsaacLab asset prim path，可为 `{ENV_REGEX_NS}/Robot` 模板。

    Returns:
        ArticulationCfg: 保留 `LEAP_HAND_CFG` 的 USD / actuator / collision 语义，只覆盖 root pose。
    """

    return LEAP_HAND_CFG.replace(
        prim_path=prim_path,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=USER_ROOT_POS_E,  # $p_{ea}^{init}$：只负责把 raw `{a}` 摆到任务姿态
            rot=USER_ROOT_QUAT_WXYZ,  # $R_{ea}^{init}$：IsaacLab `(w,x,y,z)` root quaternion
            joint_pos={"a_.*": 0.0},  # smoke 不验证 grasp pose，只验证 root/frame 语义
            joint_vel={"a_.*": 0.0},  # 静态 smoke：关节速度归零
        ),
    )


@configclass
class LeapFrameSmokeSceneCfg(InteractiveSceneCfg):
    r"""只包含 official LEAP 手、地板和灯光的最小 scene。"""

    robot: ArticulationCfg = _make_leap_root_pose_cfg("{ENV_REGEX_NS}/Robot")
    r"""official LEAP hand；运行时展开为 `/World/envs/env_i/Robot`。"""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(size=(2.0, 2.0), color=(0.45, 0.45, 0.45)),
    )
    r"""全局地板，只服务 runtime scene sanity，不参与 frame 数学。"""

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.85, 0.85, 0.85)),
    )
    r"""全局 DomeLight；headless 下无视觉必要，但保持和 viewer 场景接近。"""


def teardown_module() -> None:
    r"""关闭 IsaacSim app，避免显式 smoke 结束后遗留 Kit 进程。"""

    simulation_app.close()


@pytest.mark.isaacsim
def test_leap_root_pose_and_calibrated_hand_frame_match_math_helper() -> None:
    r"""验证 IsaacSim runtime 中的 LEAP root pose 与 `{h}` 标定推导一致。

    科研语义：
        用户在纯数学工具中给出 $T_{ah}$，LEAP env cfg 给出 $T_{ea}^{init}$。
        运行时应满足
        $$
        T_{eh}^{init}=T_{ea}^{init}T_{ah},
        $$
        且 cloned env 只额外引入平移 `env_origins`，不改变 `{h}` 的方向。
    """

    sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, render_interval=2, device="cuda:0"))
    scene = InteractiveScene(
        LeapFrameSmokeSceneCfg(num_envs=SMOKE_NUM_ENVS, env_spacing=SMOKE_ENV_SPACING, replicate_physics=False)
    )
    sim.reset()

    # 按 IsaacLab InteractiveScene 语义把每个 env 的 root state 写到 world frame。
    _reset_scene_to_default_root_pose(scene)
    sim.step()
    scene.update(sim.get_physics_dt())

    # 该 smoke 明确验证 IsaacLab env 层级，而不是 world 直挂单体资产。
    # 用户人工标定和后续 GM env 都应理解为 `/World/envs/env_0/Robot` 下的 raw root `{a}`。
    _assert_env_robot_prims_exist(scene)

    # 先验证 raw asset/root frame `{a}` 的 runtime pose 没被 `{h}` 标定逻辑偷偷改掉。
    robot = scene["robot"]
    expected_root_pos_w = scene.env_origins + _tensor3(USER_ROOT_POS_E, device=robot.device).unsqueeze(0)
    assert torch.allclose(robot.data.root_pos_w, expected_root_pos_w, atol=POSE_ATOL)
    _assert_quat_batch_equivalent(robot.data.root_quat_w, USER_ROOT_QUAT_WXYZ)

    # 再验证 `{h}` 是由 runtime root pose 与工具标定 $T_{ah}$ 复合得到，而不是另一个隐式 frame。
    R_ah = matrix_from_axis_columns(USER_X_H_IN_A, USER_Y_H_IN_A, USER_Z_H_IN_A)
    calibration = calibrate_frame(R_ah, USER_P_AH, root_pos_e=USER_ROOT_POS_E, root_quat_wxyz=USER_ROOT_QUAT_WXYZ)
    runtime_p_eh_w, runtime_R_eh = _runtime_hand_frame_pose_w(robot.data.root_pos_w, robot.data.root_quat_w, R_ah)
    expected_p_eh_w = scene.env_origins + _tensor3(calibration.p_eh_init, device=robot.device).unsqueeze(0)
    expected_R_eh = _matrix_tensor(calibration.R_eh_init, device=robot.device).unsqueeze(0).repeat(SMOKE_NUM_ENVS, 1, 1)

    assert torch.allclose(runtime_p_eh_w, expected_p_eh_w, atol=POSE_ATOL)
    assert torch.allclose(runtime_R_eh, expected_R_eh, atol=POSE_ATOL)

    _print_frame_summary(calibration.R_eh_init, calibration.p_eh_init)


def _assert_env_robot_prims_exist(scene: InteractiveScene) -> None:
    r"""确认 smoke 运行在 `/World/envs/env_i/Robot` 层级下。

    Args:
        scene (InteractiveScene): 当前 smoke scene，需提供 stage 与 cloned env paths。
    """

    stage = scene.stage  # USD stage；InteractiveScene 已展开 `{ENV_REGEX_NS}`
    for env_prim_path in scene.env_prim_paths:
        robot_prim_path = f"{env_prim_path}/Robot"  # 例如 `/World/envs/env_0/Robot`
        assert stage.GetPrimAtPath(robot_prim_path).IsValid(), f"missing env robot prim: {robot_prim_path}"


def _reset_scene_to_default_root_pose(scene: InteractiveScene) -> None:
    r"""把 LEAP hand 写回配置中的 root / joint default state。

    Args:
        scene (InteractiveScene): 当前 smoke scene，包含 `scene["robot"]`。
    """

    robot = scene["robot"]  # official LEAP articulation
    root_state = robot.data.default_root_state.clone()  # `[B,13]`，env-frame default root state
    root_state[:, :3] += scene.env_origins  # cloned env 的 world pose = env origin + env-frame root pose
    robot.write_root_pose_to_sim(root_state[:, :7])  # 写入 $p_{wa},q_{wa}$
    robot.write_root_velocity_to_sim(root_state[:, 7:])  # 静态 smoke：root velocity 归零
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    scene.reset()  # 激活 IsaacLab scene buffers


def _runtime_hand_frame_pose_w(
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    R_ah,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""由 runtime `{a}` pose 与 $T_{ah}$ 计算 `{h}` 在 world/env 中的 pose。

    Args:
        root_pos_w (torch.Tensor): `{a}` origin 的 world 位置，形状 `[B,3]`。
        root_quat_w (torch.Tensor): `{a}` orientation，IsaacLab `(w,x,y,z)`，形状 `[B,4]`。
        R_ah: `{h}` 轴在 `{a}` 中的方向矩阵。

    Returns:
        tuple[torch.Tensor, torch.Tensor]: `(p_wh, R_wh)`，形状分别为 `[B,3]` 与 `[B,3,3]`。
    """

    p_ah = _tensor3(USER_P_AH, device=root_pos_w.device)  # `{h}` origin in `{a}`，单位 m
    R_ah_tensor = _matrix_tensor(R_ah, device=root_pos_w.device)  # `{h}` axes expressed in `{a}`
    R_wa = torch.stack(
        [_matrix_tensor(quat_wxyz_to_matrix(tuple(float(v) for v in quat)), device=root_pos_w.device) for quat in root_quat_w]
    )  # `[B,3,3]`，runtime root orientation $R_{wa}$
    p_wh = root_pos_w + torch.einsum("bij,j->bi", R_wa, p_ah)  # $p_{wh}=p_{wa}+R_{wa}p_{ah}$
    R_wh = torch.einsum("bij,jk->bik", R_wa, R_ah_tensor)  # $R_{wh}=R_{wa}R_{ah}$
    return p_wh, R_wh


def _assert_quat_batch_equivalent(actual_quat_wxyz: torch.Tensor, expected_quat_wxyz: tuple[float, ...]) -> None:
    r"""按 quaternion 双覆盖语义比较 runtime root orientation。

    Args:
        actual_quat_wxyz (torch.Tensor): runtime quaternion batch，形状 `[B,4]`。
        expected_quat_wxyz (tuple[float, ...]): 配置中的 IsaacLab `(w,x,y,z)` quaternion。
    """

    expected = torch.tensor(expected_quat_wxyz, dtype=actual_quat_wxyz.dtype, device=actual_quat_wxyz.device)
    expected = expected / torch.linalg.norm(expected)  # 容忍配置四元数存在轻微非单位误差
    dots = torch.sum(actual_quat_wxyz * expected.unsqueeze(0), dim=-1).abs()  # $q$ 与 $-q$ 表示同一 SO(3)
    assert torch.allclose(dots, torch.ones_like(dots), atol=POSE_ATOL)


def _tensor3(values: tuple[float, float, float], *, device: torch.device | str) -> torch.Tensor:
    r"""构造三维 float tensor。"""

    return torch.tensor(values, dtype=torch.float32, device=device)


def _matrix_tensor(matrix, *, device: torch.device | str) -> torch.Tensor:
    r"""构造 $3\times3$ float tensor。"""

    return torch.tensor(matrix, dtype=torch.float32, device=device)


def _print_frame_summary(R_eh, p_eh) -> None:
    r"""打印 `{h}` 在 env 中的方向，供 smoke 日志人眼核对。"""

    x_h_e = (R_eh[0][0], R_eh[1][0], R_eh[2][0])  # 第一列：$x_h^e$
    y_h_e = (R_eh[0][1], R_eh[1][1], R_eh[2][1])  # 第二列：$y_h^e$
    z_h_e = (R_eh[0][2], R_eh[1][2], R_eh[2][2])  # 第三列：$z_h^e$
    print("\nLEAP hand frame calibration smoke summary")
    print(f"p_h in env: {tuple(float(v) for v in p_eh)}")
    print(f"x_h in env: {tuple(float(v) for v in x_h_e)}")
    print(f"y_h in env: {tuple(float(v) for v in y_h_e)}")
    print(f"z_h in env: {tuple(float(v) for v in z_h_e)}")
