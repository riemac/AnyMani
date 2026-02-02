#!/usr/bin/env python3
"""
快速检查奖励权重脚本

这个脚本快速验证：
1. 环境初始化时的奖励权重
2. 模拟不同步数下的权重变化
3. 为课程学习配置调整提供参考数据
"""

import argparse
import torch
import gymnasium as gym

# 导入Isaac Lab相关模块
from isaaclab.app import AppLauncher

# 解析命令行参数
parser = argparse.ArgumentParser(description="快速检查奖励权重")
parser.add_argument("--task", type=str, default="Isaac-Leaphand-ContinuousRot-Manager-v0", help="任务名称")
parser.add_argument("--num_envs", type=int, default=100, help="环境数量")
parser.add_argument("--device", type=str, default="cuda:0", help="设备")
parser.add_argument("--headless", action="store_true", help="无头模式运行")
parser.add_argument("--expected_steps", type=int, default=23976000, help="预期的checkpoint步数")

args_cli = parser.parse_args()

# 启动Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 导入其他模块
import anymani.tasks.manager_based.leaphand  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def print_reward_weights(env, step_count, title="奖励权重"):
    """打印奖励权重"""
    print(f"\n📊 {title} (步数: {step_count:,}):")
    print("-" * 70)
    
    if hasattr(env.unwrapped, 'reward_manager'):
        reward_manager = env.unwrapped.reward_manager
        weights = {}
        
        for term_name in reward_manager.active_terms:
            term_cfg = reward_manager.get_term_cfg(term_name)
            weight = term_cfg.weight
            weights[term_name] = weight
            print(f"  {term_name:<35} : {weight:>10.4f}")
        
        print("-" * 70)
        return weights
    else:
        print("  ⚠️  环境没有reward_manager")
        return None


def simulate_step_progression(env, step_values):
    """模拟不同步数下的权重变化"""
    print(f"\n🔄 模拟不同训练步数下的权重变化:")
    print("=" * 80)
    
    original_counter = env.unwrapped.common_step_counter
    weight_progression = {}
    
    for step_count in step_values:
        # 临时设置步数
        env.unwrapped.common_step_counter = step_count
        
        # 如果有课程学习管理器，重新计算
        if hasattr(env.unwrapped, 'curriculum_manager') and env.unwrapped.curriculum_manager is not None:
            env.unwrapped.curriculum_manager.compute()
        
        # 记录权重
        weights = print_reward_weights(env, step_count, f"步数 {step_count:,}")
        if weights:
            weight_progression[step_count] = weights
    
    # 恢复原始计数器
    env.unwrapped.common_step_counter = original_counter
    if hasattr(env.unwrapped, 'curriculum_manager') and env.unwrapped.curriculum_manager is not None:
        env.unwrapped.curriculum_manager.compute()
    
    return weight_progression


def analyze_weight_changes(weight_progression):
    """分析权重变化"""
    if len(weight_progression) < 2:
        return
    
    print(f"\n📈 权重变化分析:")
    print("=" * 80)
    
    step_values = sorted(weight_progression.keys())
    first_step = step_values[0]
    last_step = step_values[-1]
    
    first_weights = weight_progression[first_step]
    last_weights = weight_progression[last_step]
    
    print(f"分析区间: {first_step:,} → {last_step:,} 步")
    print("-" * 80)
    
    for term_name in first_weights.keys():
        first_val = first_weights[term_name]
        last_val = last_weights[term_name]
        change = last_val - first_val
        change_pct = (change / first_val * 100) if first_val != 0 else 0
        
        status = "📈" if change > 0 else "📉" if change < 0 else "➡️"
        print(f"  {status} {term_name:<30} : {first_val:>8.4f} → {last_val:>8.4f} ({change:+.4f}, {change_pct:+.1f}%)")


def generate_config_suggestions(expected_steps, weight_progression):
    """生成配置调整建议"""
    print(f"\n💡 配置调整建议:")
    print("=" * 80)
    
    print(f"1. 当前问题:")
    print(f"   - checkpoint包含 {expected_steps:,} 步的训练")
    print(f"   - 但env.common_step_counter从0重新开始")
    print(f"   - 导致课程学习权重基于错误的步数计算")
    
    print(f"\n2. 解决方案选项:")
    print(f"   选项A: 修改任务配置中的课程学习阈值")
    print(f"   选项B: 在训练脚本中手动同步步数计数器")
    
    print(f"\n3. 选项A - 调整课程学习配置:")
    print(f"   在 leaphand_continuous_rot_env_cfg.py 中:")
    print(f"   将所有课程学习的步数阈值减少 {expected_steps:,}")
    print(f"   例如: 如果原来是 mid_step=600_000")
    print(f"        调整为: mid_step=max(0, 600_000-{expected_steps})")
    
    print(f"\n4. 选项B - 训练脚本中同步步数:")
    print(f"   在创建环境后添加:")
    print(f"   env.unwrapped.common_step_counter = {expected_steps}")
    print(f"   if hasattr(env.unwrapped, 'curriculum_manager'):")
    print(f"       env.unwrapped.curriculum_manager.compute()")


def main():
    """主函数"""
    print("🔍 快速检查奖励权重")
    print("=" * 80)
    
    # 解析环境配置
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    
    # 创建环境
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    print(f"✅ 环境创建成功: {args_cli.task}")
    print(f"   环境数量: {env.unwrapped.num_envs}")
    print(f"   初始步数: {env.unwrapped.common_step_counter:,}")
    
    # 检查是否有课程学习
    has_curriculum = hasattr(env.unwrapped, 'curriculum_manager') and env.unwrapped.curriculum_manager is not None
    print(f"   课程学习: {'✅ 启用' if has_curriculum else '❌ 未启用'}")
    
    # 打印初始权重
    initial_weights = print_reward_weights(env, 0, "初始权重")
    
    if has_curriculum:
        # 模拟不同步数下的权重变化
        test_steps = [
            0,                    # 初始状态
            600_000,             # 早期阶段
            1_200_000,           # 中期阶段
            args_cli.expected_steps,  # 预期checkpoint步数
            args_cli.expected_steps + 240_000,  # checkpoint后继续训练
        ]
        
        weight_progression = simulate_step_progression(env, test_steps)
        
        # 分析权重变化
        analyze_weight_changes(weight_progression)
        
        # 生成配置建议
        generate_config_suggestions(args_cli.expected_steps, weight_progression)
    
    else:
        print(f"\n⚠️  环境没有启用课程学习，权重不会随步数变化")
    
    # 关闭环境
    env.close()
    print("\n✅ 检查完成")


if __name__ == "__main__":
    main()
    simulation_app.close()
