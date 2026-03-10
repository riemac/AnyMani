#!/usr/bin/env bash
set -euo pipefail

# LeapHand direct reorientation 训练快捷脚本。
# 用法：
#   ./scripts/rl_games/train_leaphand_direct.sh
#   ./scripts/rl_games/train_leaphand_direct.sh 4096 5000
#   ./scripts/rl_games/train_leaphand_direct.sh 2048 2000 --seed 7

NUM_ENVS=${1:-4096}
MAX_ITERS=${2:-5000}
shift $(( $# >= 1 ? 1 : 0 )) || true
shift $(( $# >= 1 ? 1 : 0 )) || true
EXTRA_ARGS=("$@")

TASK="AnyMani-LeapHand-Direct-v0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANYMANI_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_DIR="$(cd "$ANYMANI_DIR/.." && pwd)"
ISAACLAB_DIR="$WORKSPACE_DIR/IsaacLab"
VENV_DIR="$WORKSPACE_DIR/env_isaaclab"
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
RUN_NAME="direct_${TIMESTAMP}"
LOG_DIR="$ANYMANI_DIR/logs/rl_games/manual_runs"
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"

mkdir -p "$LOG_DIR"

if [[ ! -f "$ISAACLAB_DIR/isaaclab.sh" ]]; then
    echo "[ERROR] 找不到 Isaac Lab 启动脚本: $ISAACLAB_DIR/isaaclab.sh" >&2
    exit 1
fi

if [[ -f "$VENV_DIR/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
fi

cd "$ANYMANI_DIR"

echo "=============================================="
echo " LeapHand Direct Training Launcher"
echo "=============================================="
echo "Task:        $TASK"
echo "Num envs:    $NUM_ENVS"
echo "Max iters:   $MAX_ITERS"
echo "Run name:    $RUN_NAME"
echo "Log file:    $LOG_FILE"
if (( ${#EXTRA_ARGS[@]} > 0 )); then
    echo "Extra args:   ${EXTRA_ARGS[*]}"
fi
echo "=============================================="

"$ISAACLAB_DIR/isaaclab.sh" -p scripts/rl_games/train.py \
    --task "$TASK" \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERS" \
    --headless \
    "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
