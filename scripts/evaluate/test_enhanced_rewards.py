#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
测试LeapHand环境中的增强奖励函数：
1. 改进的旋转速度奖励（目标角速度型）
2. 指尖距离惩罚
3. 扭矩惩罚
4. 旋转轴对齐奖励

使用方法:
python scripts/evaluate/test_enhanced_rewards.py --num_envs 4 --rotation_axis_mode random

NOTE: 本脚本可用于观察探索阶段奖励函数的效果，从而调整奖励函数的参数。
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

import torch

from isaaclab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(description="测试LeapHand增强奖励函数")
parser.add_argument("--num_envs", type=int, default=4, help="环境数量")
parser.add_argument("--rotation_axis_mode", type=str, default="random", 
                   choices=["z_axis", "x_axis", "y_axis", "random"],
                   help="旋转轴模式")
parser.add_argument("--headless", action="store_true", help="无头模式运行")

# 解析参数并启动应用
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sim import SimulationContext

# 导入环境配置
from anymani.tasks.inhand.config.leaphand.leaphand_continuous_rot_env_cfg import (
    LeaphandContinuousRotEnvCfg
)

def test_enhanced_rewards():
    """测试增强奖励函数"""
    
    print("=" * 80)
    print("LeapHand连续旋转环境 - 增强奖励函数测试")
    print("=" * 80)
    
    # 创建环境配置
    env_cfg = LeaphandContinuousRotEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # 设置旋转轴模式
    env_cfg.commands.rotation_axis.rotation_axis_mode = args_cli.rotation_axis_mode
    
    # 确保启用可视化
    env_cfg.commands.rotation_axis.debug_vis = True
    
    print(f"环境数量: {env_cfg.scene.num_envs}")
    print(f"旋转轴模式: {args_cli.rotation_axis_mode}")
    print("增强奖励函数:")
    print("  ✅ 旋转速度奖励 (目标角速度型)")
    print("  ✅ 指尖距离惩罚")
    print("  ✅ 扭矩惩罚")
    print("  ✅ 旋转轴对齐奖励")
    
    # 创建环境
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    print("\n✅ 环境创建成功！")
    print("📊 奖励函数详情：")
    
    # 打印奖励管理器信息
    reward_terms = env.reward_manager._term_names
    reward_weights = [term_cfg.weight for term_cfg in env.reward_manager._term_cfgs]
    
    for i, (name, weight) in enumerate(zip(reward_terms, reward_weights)):
        print(f"   {i+1:2d}. {name:<25} 权重: {weight:+8.4f}")
    
    try:
        # 重置环境
        env.reset()
        
        print(f"\n🔄 开始测试奖励函数...")
        
        # 运行仿真步骤
        step_count = 0
        max_steps = 500 if args_cli.headless else 1000
        
        # 记录奖励统计
        reward_stats = {name: [] for name in reward_terms}
        total_rewards = []
        
        while step_count < max_steps:
            # 执行随机动作
            actions = torch.randn(env.num_envs, env.action_manager.total_action_dim, device=env.device)
            actions = torch.clamp(actions, -1.0, 1.0)
            
            # 执行步骤
            obs, rewards, terminated, truncated, info = env.step(actions)
            step_count += 1
            
            # 记录奖励统计
            total_rewards.append(rewards.mean().item())
            
            # 记录各项奖励
            if hasattr(env.reward_manager, '_step_reward'):
                for i, name in enumerate(reward_terms):
                    reward_value = env.reward_manager._step_reward[:, i].mean().item()
                    reward_stats[name].append(reward_value)
            
            # 每100步打印一次统计信息
            if step_count % 100 == 0:
                print(f"\n📈 步骤 {step_count:4d}/{max_steps}:")
                print(f"   总奖励: {rewards.mean().item():+8.4f}")
                
                # 打印各项奖励的当前值
                if hasattr(env.reward_manager, '_step_reward'):
                    for i, name in enumerate(reward_terms):
                        current_reward = env.reward_manager._step_reward[:, i].mean().item()
                        print(f"   {name:<25}: {current_reward:+8.4f}")
        
        print(f"\n✅ 测试完成！共运行 {step_count} 步")
        
        # 计算并显示奖励统计
        if reward_stats:
            print("\n📊 奖励统计摘要:")
            print(f"   总奖励平均值: {sum(total_rewards)/len(total_rewards):+8.4f}")
            print(f"\n   {'奖励项':<25} {'权重':<8} {'平均值':<10} {'范围'}")
            print(f"   {'-'*25} {'-'*8} {'-'*10} {'-'*25}")
            
            for i, name in enumerate(reward_terms):
                if reward_stats[name]:
                    weight = reward_weights[i]
                    avg_reward = sum(reward_stats[name]) / len(reward_stats[name])
                    min_reward = min(reward_stats[name])
                    max_reward = max(reward_stats[name])
                    print(f"   {name:<25} {weight:+7.1f} {avg_reward:+9.4f} [{min_reward:+7.4f}, {max_reward:+7.4f}]")
        
        if not args_cli.headless:
            print("\n⏸️  测试已完成，但环境仍在运行")
            print("   💡 您可以继续在Isaac Sim中观察可视化效果")
            print("   💡 按Ctrl+C退出程序")
            
            # 保持环境运行，让用户观察
            try:
                while True:
                    actions = torch.randn(env.num_envs, env.action_manager.total_action_dim, device=env.device)
                    actions = torch.clamp(actions, -0.3, 0.3)  # 使用较小的动作幅度
                    env.step(actions)
            except KeyboardInterrupt:
                print("\n⏹️  用户中断，正在退出...")
    
    except KeyboardInterrupt:
        print("\n⏹️  测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 关闭环境
        env.close()
        print("\n✅ 环境已关闭！")

def main():
    """主函数"""
    try:
        test_enhanced_rewards()
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return 1
    return 0

if __name__ == "__main__":
    # 运行测试
    exit_code = main()
    # 关闭仿真应用
    simulation_app.close()
    sys.exit(exit_code)
