#!/bin/bash
# Run TP+FSDP2 smoke test on RunPod
#
# Usage:
#   ./scripts/test_tp_runpod.sh           # Default: 2 GPUs, TP=2, 30 steps
#   ./scripts/test_tp_runpod.sh 4         # 4 GPUs, TP=4, 30 steps
#   ./scripts/test_tp_runpod.sh 8 4       # 8 GPUs, TP=4 (DP=2), 30 steps

set -e

NUM_GPUS=${1:-2}
TP_SIZE=${2:-$NUM_GPUS}
STEPS=${3:-30}
REBALANCE_STEP=${4:-15}

echo "========================================"
echo "TP+FSDP2 Smoke Test Configuration"
echo "========================================"
echo "Number of GPUs: $NUM_GPUS"
echo "TP Size: $TP_SIZE"
echo "DP Size: $((NUM_GPUS / TP_SIZE))"
echo "Steps: $STEPS"
echo "Rebalance Step: $REBALANCE_STEP"
echo "========================================"

# Change to training package directory
cd "$(dirname "$0")/.."

# Activate environment if needed
if [ -f "../../.venv/bin/activate" ]; then
    source "../../.venv/bin/activate"
fi

# Run the smoke test
echo "Starting torchrun..."
torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS \
    scripts/test_tp_smoke.py \
    --tp-size $TP_SIZE \
    --steps $STEPS \
    --rebalance-step $REBALANCE_STEP \
    --batch-size 4 \
    --seq-length 256 \
    --check-sync

echo "========================================"
echo "Smoke Test Complete"
echo "========================================"
