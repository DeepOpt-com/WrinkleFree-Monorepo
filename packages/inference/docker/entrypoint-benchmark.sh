#!/bin/bash
set -e

echo "=== WrinkleFree BitNet Cost Benchmark Container ==="
echo "Model Path: ${MODEL_PATH:-/models/model.gguf}"
echo "Port: ${PORT:-8080}"
echo "Threads: ${NUM_THREADS:-0} (0=auto)"
echo "Context Size: ${CONTEXT_SIZE:-4096}"
echo "Benchmark Mode: ${BENCHMARK_MODE:-false}"
echo ""

# Check if model exists
if [ ! -f "${MODEL_PATH}" ]; then
    # Try default native BitNet model
    if [ -f "/opt/bitnet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf" ]; then
        echo "Using pre-downloaded BitNet-b1.58-2B-4T model"
        MODEL_PATH="/opt/bitnet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
    else
        echo "ERROR: Model not found at ${MODEL_PATH}"
        echo "Please mount a model volume or set MODEL_PATH"
        exit 1
    fi
fi

echo "Starting BitNet inference server..."
echo ""

# Start the server
cd /opt/bitnet

if [ -f "build/bin/llama-server" ]; then
    # Use native llama-server for best performance
    exec ./build/bin/llama-server \
        -m "${MODEL_PATH}" \
        -c ${CONTEXT_SIZE:-4096} \
        -t ${NUM_THREADS:-0} \
        --host ${HOST:-0.0.0.0} \
        --port ${PORT:-8080}
elif [ -f "run_inference_server.py" ]; then
    # Fall back to Python wrapper
    exec python run_inference_server.py \
        -m "${MODEL_PATH}" \
        -c ${CONTEXT_SIZE:-4096} \
        -t ${NUM_THREADS:-0} \
        --host ${HOST:-0.0.0.0} \
        --port ${PORT:-8080}
else
    echo "ERROR: No inference server found"
    exit 1
fi
