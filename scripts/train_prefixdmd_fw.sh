#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_ROOT"

CONFIG="${1:-configs/prefix_dmd_framewise.yaml}"
LOGDIR="${2:-/bht/PrefixDMD/training_outputs/prefix_dmd_framewise_onpolicy}"
# LOGDIR="${2:-/bht/PrefixDMD/training_outputs/prefix_dmd_framewise}/$(date +%Y%m%d_%H%M%S)"

NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29531}"

WANDB="${WANDB:-0}"
PROXY="${PROXY:-0}"
PROXY_URL="${PROXY_URL:-http://127.0.0.1:18889}"

WANDB_ARGS=()
if [[ "$WANDB" == "0" ]]; then
    WANDB_ARGS+=(--disable-wandb)
fi

if [[ "$PROXY" == "1" ]]; then
    export HTTP_PROXY="$PROXY_URL"
    export HTTPS_PROXY="$PROXY_URL"
    export WANDB_HTTP_PROXY="$PROXY_URL"
    export WANDB_HTTPS_PROXY="$PROXY_URL"
    export WANDB_MODE=online
fi

[[ -f "$CONFIG" ]] || { echo "ERROR: missing config: $CONFIG" >&2; exit 1; }
[[ -d wan_models/Wan2.1-T2V-1.3B ]] || {
  echo "ERROR: missing raw Wan base at wan_models/Wan2.1-T2V-1.3B" >&2
  exit 1
}

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

mkdir -p "$LOGDIR"

echo "===================================================="
echo "Teacher Forcing DMD Frame-wise"
echo "Config:     $CONFIG"
echo "Logdir:     $LOGDIR"
echo "GPUs:       $NUM_GPUS"
echo "Init:       RAW Wan2.1-T2V-1.3B"
echo "WandB:      $WANDB"
echo "Proxy:      $PROXY"
echo "===================================================="

torchrun \
  --nproc_per_node="$NUM_GPUS" \
  --master_port="$MASTER_PORT" \
  train.py \
  --config_path "$CONFIG" \
  --logdir "$LOGDIR" \
  --no_visualize \
  "${WANDB_ARGS[@]}" \
  2>&1 | tee "$LOGDIR/train_shell.log"

  # --disable-wandb