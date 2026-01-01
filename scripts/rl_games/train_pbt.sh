#!/bin/bash
# Population-Based Training (PBT) 启动脚本
# 
# PBT工作原理：每个策略是一个独立进程，通过共享目录交换checkpoint
# 因此必须启动N个进程，每个进程设置不同的policy_idx
#
# 用法: ./scripts/rl_games/train_pbt.sh [num_policies] [num_envs]
# 示例: ./scripts/rl_games/train_pbt.sh 6 2048

set -e

# 默认参数
NUM_POLICIES=${1:-4}    # 种群大小，默认6
NUM_ENVS=${2:-1000}     # 每个策略的环境数，默认2048
TASK="Template-Leaphand-Rot-Manager-v0"

# 获取脚本所在目录（AnyRotate）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANYROTATE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
ISAACLAB_DIR="$(cd "$ANYROTATE_DIR/../IsaacLab" && pwd)"

# 共享目录配置
PBT_DIR="$ANYROTATE_DIR/logs/pbt"
PBT_WORKSPACE="leaphand_pbt_$(date +%Y-%m-%d_%H-%M-%S)"

# 创建PBT目录
mkdir -p "$PBT_DIR"

echo "=============================================="
echo "        Population-Based Training"
echo "=============================================="
echo "Task:         $TASK"
echo "Num Policies: $NUM_POLICIES (每个策略一个独立进程)"
echo "Num Envs:     $NUM_ENVS (每个进程的并行环境数)"
echo "总环境数:     $((NUM_POLICIES * NUM_ENVS))"
echo "----------------------------------------------"
echo "AnyRotate:    $ANYROTATE_DIR"
echo "IsaacLab:     $ISAACLAB_DIR"
echo "PBT Dir:      $PBT_DIR"
echo "Workspace:    $PBT_WORKSPACE"
echo "=============================================="
echo ""
echo "⚠️  注意：PBT需要为每个策略启动独立进程！"
echo "    将启动 $NUM_POLICIES 个后台进程..."
echo ""

# 切换到AnyRotate目录
cd "$ANYROTATE_DIR"

# 激活虚拟环境
source "$ANYROTATE_DIR/../env_isaaclab/bin/activate"

# 启动每个策略进程
PIDS=()
for ((i=0; i<NUM_POLICIES; i++)); do
    echo "🚀 Starting Policy $i..."
    
    # 使用nohup在后台运行，日志输出到文件
    nohup python "$ANYROTATE_DIR/scripts/rl_games/train.py" \
        --task="$TASK" \
        --num_envs="$NUM_ENVS" \
        --headless \
        --seed="$i" \
        agent.pbt.enabled=True \
        agent.pbt.num_policies="$NUM_POLICIES" \
        agent.pbt.policy_idx="$i" \
        agent.pbt.directory="$PBT_DIR" \
        agent.pbt.workspace="$PBT_WORKSPACE" \
        > "$PBT_DIR/policy_${i}.log" 2>&1 &
    
    PIDS+=($!)
    echo "   PID: ${PIDS[-1]} | Log: $PBT_DIR/policy_${i}.log"
    
    # 等待足够时间让前一个进程完成GPU初始化，避免CUDA资源竞争
    # Isaac Sim初始化通常需要60-90秒
    if [ $i -lt $((NUM_POLICIES - 1)) ]; then
        echo "   ⏳ 等待30秒让进程完成初始化..."
        sleep 30
    fi
done

echo ""
echo "=============================================="
echo "✅ All $NUM_POLICIES policies started!"
echo "=============================================="
echo ""
echo "📊 监控命令："
echo "   tail -f $PBT_DIR/policy_*.log"
echo ""
echo "📋 查看进程："
echo "   ps aux | grep 'train.py.*$TASK'"
echo ""
echo "🛑 停止所有训练："
echo "   kill ${PIDS[*]}"
echo "   # 或: pkill -f 'train.py.*$TASK'"
echo ""
echo "📁 PBT工作区："
echo "   ls -la $PBT_DIR/$PBT_WORKSPACE/"
echo ""
