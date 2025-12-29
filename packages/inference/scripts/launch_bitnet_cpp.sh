#!/bin/bash
# Launch BitNet.cpp server for BitNet models
#
# Usage:
#   ./scripts/launch_bitnet_cpp.sh [model_path] [port]
#
# Examples:
#   ./scripts/launch_bitnet_cpp.sh                          # Default model at extern/BitNet
#   ./scripts/launch_bitnet_cpp.sh /path/to/model.gguf 8080
#
# Environment variables:
#   BITNET_MODEL - Model path (GGUF file)
#   BITNET_PORT - Server port (default: 8080)
#   BITNET_HOST - Server host (default: 0.0.0.0)

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BITNET_DIR="${PROJECT_DIR}/extern/BitNet"

# Default model path
DEFAULT_MODEL="${BITNET_DIR}/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"

# Configuration
MODEL="${1:-${BITNET_MODEL:-$DEFAULT_MODEL}}"
PORT="${2:-${BITNET_PORT:-8080}}"
HOST="${BITNET_HOST:-0.0.0.0}"

# Check if llama-server exists
if [[ ! -x "${BITNET_DIR}/build/bin/llama-server" ]]; then
    echo "Error: llama-server not found. Build BitNet.cpp first:"
    echo "  cd extern/BitNet"
    echo "  cmake -B build -DBITNET_X86_TL2=ON"
    echo "  cmake --build build --config Release -j4"
    exit 1
fi

# Check if model exists
if [[ ! -f "$MODEL" ]]; then
    echo "Error: Model not found at $MODEL"
    echo "Download with:"
    echo "  huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir ${BITNET_DIR}/models/BitNet-b1.58-2B-4T"
    exit 1
fi

echo "=== BitNet.cpp Server ==="
echo "Model: $MODEL"
echo "Host:  $HOST"
echo "Port:  $PORT"
echo ""
echo "API endpoint: http://${HOST}:${PORT}/v1/chat/completions"
echo ""
echo "Performance: ~26 tok/s (1.6x faster than sglang)"
echo "Features:    KV cache reuse for repeated prompts"
echo ""

# Launch server with optimizations
# --cache-reuse 64: Reuse KV cache for prompts sharing 64+ token prefix
# --n-gpu-layers 0: CPU-only (BitNet is optimized for CPU)
# taskset limits to 8 cores to prevent system freeze
exec taskset -c 0-7 "${BITNET_DIR}/build/bin/llama-server" \
    -m "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --cache-reuse 64
