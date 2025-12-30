#!/bin/bash
# Simple BitNet inference server
# Uses 4 CPU cores and GGUF model
#
# Usage:
#   ./scripts/start_server.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration
MODEL="${MODEL:-/home/lev/models/ggml-model-i2_s.gguf}"
PORT="${PORT:-8080}"
THREADS="${THREADS:-4}"
CTX_SIZE="${CTX_SIZE:-2048}"

# llama.cpp from sglang-bitnet
LLAMA_DIR="${PROJECT_DIR}/extern/sglang-bitnet/3rdparty/llama.cpp"
SERVER="${LLAMA_DIR}/build/bin/llama-server"

# Library path for shared libs
export LD_LIBRARY_PATH="${LLAMA_DIR}/build/src:${LLAMA_DIR}/build/ggml/src:${LD_LIBRARY_PATH:-}"

# Verify binary exists
if [[ ! -x "$SERVER" ]]; then
    echo "Error: llama-server not found. Build with:"
    echo "  cd extern/sglang-bitnet/3rdparty/llama.cpp"
    echo "  cmake -B build && cmake --build build -j4"
    exit 1
fi

# Verify model exists
if [[ ! -f "$MODEL" ]]; then
    echo "Error: Model not found at $MODEL"
    exit 1
fi

echo "=== BitNet Inference Server ==="
echo "Model:   $MODEL"
echo "Threads: $THREADS (limited to cores 0-$((THREADS-1)))"
echo "Port:    $PORT"
echo ""
echo "API: http://localhost:${PORT}/v1/chat/completions"
echo ""

# Run with limited cores
exec taskset -c 0-$((THREADS-1)) "$SERVER" \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --threads "$THREADS" \
    --ctx-size "$CTX_SIZE"
