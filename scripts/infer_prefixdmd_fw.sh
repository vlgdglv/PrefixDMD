#!/usr/bin/env bash
set -euo pipefail

#----------------------------------------
#
#          Edit RUN NAME Here!!
#
#----------------------------------------
RUN_NAME=prefix_dmd_framewise_onpolicy/checkpoint_model_000800

NUM_SAMPLES=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_ROOT"

CONFIG="${CONFIG:-configs/infer_framewise_common.yaml}"
CKPT="${1:-training_outputs/${RUN_NAME}/model.pt}"

[[ $# -gt 0 ]] && shift

PROMPTS="${1:-prompts/test_prompt.txt}"

promptset="$(basename "$PROMPTS" .txt)"
OUTPUT_FOLDER=outputs/visual_check/${RUN_NAME}/
mkdir -p "$OUTPUT_FOLDER"

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN="${TORCHRUN:-torchrun}"
SEED="${SEED:-0}"
NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-21}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/self_gradient_forcing}"

USE_EMA="${USE_EMA:-1}"
EMA_ARGS=()
[[ "$USE_EMA" == "1" ]] && EMA_ARGS=(--use_ema)

[[ -f "$CONFIG" ]] || { echo "ERROR: config not found: $CONFIG" >&2; exit 1; }
[[ -f "$CKPT" ]] || { echo "ERROR: checkpoint not found: $CKPT" >&2; exit 1; }
[[ -f "$PROMPTS" ]] || { echo "ERROR: prompt file not found: $PROMPTS" >&2; exit 1; }

GPU_COUNT="$($PYTHON_BIN - <<'PY'
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(0)
PY
)"
if (( GPU_COUNT >= 8 )); then
    RUN_GPUS=8
else
    RUN_GPUS=1
fi

export PYTHONPATH="${PYTHONPATH:-$PROJECT_ROOT}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

cat <<INFO
==========================================
Self Gradient Forcing inference
Variant:           framewise
Config:            $CONFIG
Checkpoint:        $CKPT
Prompts:           $PROMPTS
Visible GPUs:      $GPU_COUNT
Launch GPUs:       $RUN_GPUS
Output frames:     $NUM_OUTPUT_FRAMES latent frames
Output:            $OUTPUT_FOLDER
==========================================
INFO

COMMON_ARGS=(
    inference.py
    --config_path "$CONFIG"
    --checkpoint_path "$CKPT"
    --data_path "$PROMPTS"
    --output_folder "$OUTPUT_FOLDER"
    --num_output_frames "$NUM_OUTPUT_FRAMES"
    --num_samples $NUM_SAMPLES
    --name_length 100
    --seed "$SEED"
    "${EMA_ARGS[@]}"
)

if (( RUN_GPUS > 1 )); then
    unset RANK WORLD_SIZE MASTER_ADDR MASTER_PORT NODE_RANK LOCAL_WORLD_SIZE GROUP_RANK
    "$TORCHRUN" --standalone --nnodes=1 --master_port=28439 --nproc_per_node="$RUN_GPUS" "${COMMON_ARGS[@]}" \
        2>&1 | tee "$OUTPUT_FOLDER/infer.log"
else
    "$PYTHON_BIN" "${COMMON_ARGS[@]}" 2>&1 | tee "$OUTPUT_FOLDER/infer.log"
fi

echo
echo "[done] Videos written to: $OUTPUT_FOLDER"
