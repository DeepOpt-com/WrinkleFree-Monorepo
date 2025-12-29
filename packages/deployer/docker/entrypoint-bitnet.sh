#!/bin/bash
set -e

echo "Starting BitNet inference server..."
echo "Model: ${MODEL_PATH}"
echo "Host: ${HOST}:${PORT}"
echo "Threads: ${NUM_THREADS} (0 = auto)"
echo "Context: ${CONTEXT_SIZE}"

cd /opt/bitnet

# Run inference server
exec python run_inference.py \
    --model "${MODEL_PATH}" \
    --n-threads "${NUM_THREADS:-0}" \
    --ctx-size "${CONTEXT_SIZE:-4096}" \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8080}" \
    "$@"
