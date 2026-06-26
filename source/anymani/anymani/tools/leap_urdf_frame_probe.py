#!/usr/bin/env python3
r"""只把 LEAP 官方 USD 手本体正常显示在 Isaac Sim 场景中。

启动命令：

```bash
cd /home/hac/isaac/AnyMani
source /home/hac/isaac/env_isaaclab/bin/activate
/home/hac/isaac/IsaacLab/isaaclab.sh -p source/anymani/anymani/tools/leap_urdf_frame_probe.py
```

查看 IsaacLab env 层级：

```bash
cd /home/hac/isaac/AnyMani
source /home/hac/isaac/env_isaaclab/bin/activate
/home/hac/isaac/IsaacLab/isaaclab.sh -p source/anymani/anymani/tools/leap_urdf_frame_probe.py \
  --scene-mode env --num-envs 2
```

自动冒烟测试：

```bash
cd /home/hac/isaac/AnyMani
source /home/hac/isaac/env_isaaclab/bin/activate
timeout --kill-after=10s 240s /home/hac/isaac/IsaacLab/isaaclab.sh -p \
  source/anymani/anymani/tools/leap_urdf_frame_probe.py --headless --smoke-seconds 2
```

这个脚本只服务一个肉眼检查目标：官方 LEAP USD 在 world identity pose 下长什么样。
它不加载 URDF，不创建 RL env，不画 frame marker，不放坐标轴 visual prim。
场景里只保留手、本应存在的地板和照明；不会额外放方块、球体或坐标轴。

实现方式刻意贴近 IsaacLab 官方 `scripts/demos/hands.py`：
用 `Articulation(ArticulationCfg)` 生成机器人，而不是手写 USD stage 引用。
`--scene-mode env` 进一步贴近 `scripts/tutorials/02_scene/create_scene.py`：
用 `InteractiveScene` 展开 `{ENV_REGEX_NS}/Robot` 到 `/World/envs/env_i/Robot`。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="只打开 LEAP 官方 USD，放在 world identity pose。")
parser.add_argument(
    "--scene-mode",
    choices=("world", "env"),
    default="world",
    help="world: /World/LeapHand 单体资产；env: /World/envs/env_i/Robot 训练式层级。",
)
parser.add_argument("--num-envs", type=int, default=2, help="env 模式下创建的环境数量。")
parser.add_argument("--env-spacing", type=float, default=0.75, help="env 模式下相邻环境的间距。")
parser.add_argument("--smoke-seconds", type=float, default=None, help="可选：运行若干秒后自动退出。")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from anymani.robots.leap import LEAP_HAND_CFG  # noqa: E402

ROOT_POS_W = (0.0, 0.0, 0.0)
"""手根节点在 world 坐标系下的位置；这里故意取原点，方便肉眼判断 USD 自身朝向。"""

ROOT_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)
"""手根节点在 world 坐标系下的姿态；identity quaternion 表示不额外旋转资产。"""


WORLD_ROBOT_PRIM_PATH = "/World/LeapHand"
"""world 模式下的手根 prim；运行后 Stage 面板应能看到并展开这个节点。"""

ENV_ROBOT_PRIM_EXPR = "{ENV_REGEX_NS}/Robot"
"""env 模式下的手根 prim 模板；IsaacLab 会展开成 `/World/envs/env_i/Robot`。"""


def _make_leap_identity_cfg(prim_path: str) -> ArticulationCfg:
    r"""构造 LEAP identity-root 配置。

    Args:
        prim_path (str): IsaacLab asset prim path，可为真实路径或 `{ENV_REGEX_NS}` 模板。

    Returns:
        ArticulationCfg: 只覆盖 root pose / joint seed 的 LEAP 配置。
    """

    return LEAP_HAND_CFG.replace(
        prim_path=prim_path,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=ROOT_POS_W,
            rot=ROOT_QUAT_WXYZ,
            joint_pos={"a_.*": 0.0},
            joint_vel={"a_.*": 0.0},
        ),
    )


@configclass
class LeapEnvViewerSceneCfg(InteractiveSceneCfg):
    r"""只用于查看 `/World/envs/env_i/Robot` 层级的轻量 scene。"""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(size=(2.0, 2.0), color=(0.45, 0.45, 0.45)),
    )
    """全局地板；不属于任何单独 env。"""

    robot: ArticulationCfg = _make_leap_identity_cfg(ENV_ROBOT_PRIM_EXPR)
    """官方 LEAP 手；运行时展开到 `/World/envs/env_i/Robot`。"""

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.85, 0.85, 0.85)),
    )
    """全局 DomeLight；不表达任何 frame 语义。"""


def design_world_scene() -> Articulation:
    r"""按 IsaacLab 官方 hands demo 的方式创建 world 直挂 LEAP Articulation。"""

    # IsaacLab 官方 demo 默认保留地面；这里也保留地板，方便肉眼判断手的空间姿态。
    ground_cfg = sim_utils.GroundPlaneCfg(size=(2.0, 2.0), color=(0.45, 0.45, 0.45))
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    # 默认 viewport 光照可能让手看起来接近全黑；灯光不是可见几何 marker。
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.85, 0.85, 0.85))
    light_cfg.func("/World/Light", light_cfg)

    # 复用项目里已验证的 LEAP 配置，只覆盖当前肉眼检查需要的根路径和 identity pose。
    return Articulation(_make_leap_identity_cfg(WORLD_ROBOT_PRIM_PATH))


def design_env_scene() -> InteractiveScene:
    r"""按 IsaacLab InteractiveScene 方式创建 `/World/envs/env_i/Robot` 层级。"""

    if args_cli.num_envs < 1:
        raise ValueError(f"--num-envs must be positive, got {args_cli.num_envs}.")
    if args_cli.env_spacing <= 0.0:
        raise ValueError(f"--env-spacing must be positive, got {args_cli.env_spacing}.")

    scene_cfg = LeapEnvViewerSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=args_cli.env_spacing,
        replicate_physics=False,
    )
    return InteractiveScene(scene_cfg)


def _frame_prims(prim_paths: list[str]) -> None:
    r"""在 GUI 里 frame 到手，避免打开窗口后视角仍停留在空白区域。"""

    # GUI experience 才一定有 viewport extension；headless smoke 不应因为没有 viewport 模块失败。
    try:
        from omni.kit.viewport.utility import frame_viewport_prims
    except ModuleNotFoundError:
        return

    # Viewport frame 到手本体；如果当前 GUI 尚未创建 active viewport，失败也不影响手的 spawn。
    frame_viewport_prims(prims=prim_paths)


def _print_world_summary(robot: Articulation) -> None:
    r"""打印 world 模式最小运行时信息，防止空窗口被误认为加载成功。"""

    print("\n" + "=" * 88)
    print("LEAP hand viewer 已启动：world mode")
    print(f"robot prim: {WORLD_ROBOT_PRIM_PATH}")
    print(f"world pose: pos={ROOT_POS_W}, quat_wxyz={ROOT_QUAT_WXYZ}")
    print(f"num_joints: {robot.num_joints}")
    print(f"joint_names: {robot.joint_names}")
    print("显示手、地板和灯光；没有坐标轴、frame marker、方块或球体。")
    print("=" * 88 + "\n")


def _print_env_summary(scene: InteractiveScene) -> None:
    r"""打印 env 模式最小运行时信息，直接暴露 env_i/Robot 层级。"""

    robot = scene["robot"]
    robot_prim_paths = [f"{env_path}/Robot" for env_path in scene.env_prim_paths]

    print("\n" + "=" * 88)
    print("LEAP hand viewer 已启动：env mode")
    print(f"scene env prims: {scene.env_prim_paths}")
    print(f"robot prim expression: {ENV_ROBOT_PRIM_EXPR}")
    print(f"robot concrete prims: {robot_prim_paths}")
    print(f"env_origins: {scene.env_origins.detach().cpu().tolist()}")
    print(f"root pose in each env frame: pos={ROOT_POS_W}, quat_wxyz={ROOT_QUAT_WXYZ}")
    print(f"num_joints: {robot.num_joints}")
    print(f"joint_names: {robot.joint_names}")
    print("显示手、地板和灯光；没有坐标轴、frame marker、方块或球体。")
    print("=" * 88 + "\n")


def _smoke_start_time() -> float | None:
    r"""返回 smoke 起始时间；GUI 常规运行时返回 None。"""

    return time.monotonic() if args_cli.smoke_seconds is not None else None


def _maybe_exit_smoke(smoke_start: float | None) -> None:
    r"""smoke 模式达到指定秒数后立即退出 Isaac Sim。"""

    if smoke_start is None:
        return
    if time.monotonic() - smoke_start >= float(args_cli.smoke_seconds):
        print(f"[leap_urdf_frame_probe] smoke completed after {args_cli.smoke_seconds:.2f}s.")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def run_world_simulator(sim: SimulationContext, robot: Articulation) -> None:
    r"""保持 world 模式 LEAP 手在默认构型并推进仿真。"""

    sim_dt = sim.get_physics_dt()
    smoke_start = _smoke_start_time()

    while simulation_app.is_running():
        # 每帧把目标设为默认关节位置；这和官方 hands demo 的“保持/驱动手”模式一致。
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.write_data_to_sim()

        sim.step()
        robot.update(sim_dt)

        _maybe_exit_smoke(smoke_start)


def run_env_simulator(sim: SimulationContext, scene: InteractiveScene) -> None:
    r"""保持 env 模式每个 LEAP 手在默认构型并推进仿真。"""

    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    smoke_start = _smoke_start_time()

    while simulation_app.is_running():
        # InteractiveScene 会把同一批 joint targets 写到所有 env 的 Robot。
        robot.set_joint_position_target(robot.data.default_joint_pos)
        scene.write_data_to_sim()

        sim.step()
        scene.update(sim_dt)

        _maybe_exit_smoke(smoke_start)


def reset_world_robot(robot: Articulation) -> None:
    r"""把 world 模式手写回默认 root / joint state。"""

    root_state = robot.data.default_root_state.clone()
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    robot.reset()


def reset_env_scene(scene: InteractiveScene) -> None:
    r"""按 IsaacLab 官方 scene tutorial，把每个 env 的 root state 写到 world frame。"""

    robot = scene["robot"]
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    scene.reset()


def main() -> None:
    r"""启动最小可视化场景。"""

    # 官方 demos 均先创建 SimulationContext，再创建资产，再 reset 激活 PhysX/articulation handles。
    sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, render_interval=2, device=args_cli.device))

    if args_cli.scene_mode == "world":
        sim.set_camera_view([0.45, -0.55, 0.35], [0.0, 0.0, 0.08])
        robot = design_world_scene()

        sim.reset()

        reset_world_robot(robot)
        _frame_prims([WORLD_ROBOT_PRIM_PATH])
        _print_world_summary(robot)
        run_world_simulator(sim, robot)
    else:
        sim.set_camera_view([0.9, -1.0, 0.6], [0.0, 0.0, 0.08])
        scene = design_env_scene()

        sim.reset()

        reset_env_scene(scene)
        _frame_prims([f"{env_path}/Robot" for env_path in scene.env_prim_paths])
        _print_env_summary(scene)
        run_env_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
