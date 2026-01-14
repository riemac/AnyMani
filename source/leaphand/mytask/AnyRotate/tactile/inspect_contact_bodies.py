"""调试脚本：检查LeapHand USD中哪些body可以作为ContactSensor的prim_path

Usage:
    python inspect_contact_bodies.py --headless
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect LeapHand contact sensor bodies")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import omni.usd
from pxr import UsdPhysics, PhysxSchema

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from leaphand.tasks.manager_based.leaphand.inhand_base_env_cfg import InHandSceneCfg


def find_contact_bodies(prim, indent=0):
    """递归查找所有带RigidBody和ContactReportAPI的prim"""
    has_rb = prim.HasAPI(UsdPhysics.RigidBodyAPI)
    has_contact = prim.HasAPI(PhysxSchema.PhysxContactReportAPI)
    
    if has_rb:
        marker = "✅ CONTACT_SENSOR_READY" if has_contact else "⚠️ NO_CONTACT_API (需要activate_contact_sensors=True)"
        print(f"{'  ' * indent}{prim.GetPath()} [{marker}]")
    
    for child in prim.GetChildren():
        find_contact_bodies(child, indent + 1)


def main():
    """Main function."""
    # 创建仿真上下文
    sim_cfg = SimulationCfg(device="cuda:0", dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # 设置主相机视角
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    
    # 创建场景
    scene_cfg = InHandSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # 重置仿真
    sim.reset()
    
    print("\n" + "="*80)
    print("LeapHand机器人Body名称列表")
    print("="*80)
    robot = scene["robot"]
    for i, name in enumerate(robot.body_names):
        print(f"[{i:2d}] {name}")
    
    print("\n" + "="*80)
    print("搜索USD中所有RigidBody（可用于ContactSensor的prim）")
    print("="*80)
    stage = omni.usd.get_context().get_stage()
    robot_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    find_contact_bodies(robot_prim)
    
    print("\n" + "="*80)
    print("推荐的ContactSensor配置")
    print("="*80)
    print("基于以上输出，在InHandSceneCfg中添加：\n")
    print("contact_xxx = ContactSensorCfg(")
    print("    prim_path=\"{ENV_REGEX_NS}/Robot/<YOUR_BODY_PATH>\",")
    print("    filter_prim_paths_expr=[\"{ENV_REGEX_NS}/object\"],")
    print("    update_period=0.0,")
    print("    history_length=3,")
    print("    debug_vis=True,")
    print(")")
    print("\n注意：只能使用标记为 [✅ CONTACT_SENSOR_READY] 的路径！")
    print("="*80)
    
    # 关闭仿真
    simulation_app.close()


if __name__ == "__main__":
    main()
