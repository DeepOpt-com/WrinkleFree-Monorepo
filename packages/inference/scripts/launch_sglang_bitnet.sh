#!/bin/bash
# Launch SGLang-BitNet server for BitNet models
#
# Usage:
#   ./scripts/launch_sglang_bitnet.sh [model_path] [port]
#
# Examples:
#   ./scripts/launch_sglang_bitnet.sh                          # Default: microsoft/bitnet-b1.58-2B-4T
#   ./scripts/launch_sglang_bitnet.sh /path/to/model 8000      # Custom model and port
#
# Environment variables:
#   SGLANG_MODEL - Model path (HuggingFace ID or local path)
#   SGLANG_PORT - Server port
#   SGLANG_HOST - Server host

set -euo pipefail

# Ensure uv is in PATH
export PATH="$HOME/.local/bin:$PATH"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration
MODEL="${1:-${SGLANG_MODEL:-microsoft/bitnet-b1.58-2B-4T}}"
PORT="${2:-${SGLANG_PORT:-30000}}"
HOST="${SGLANG_HOST:-0.0.0.0}"

echo "=== SGLang-BitNet Server ==="
echo "Model: $MODEL"
echo "Host:  $HOST"
echo "Port:  $PORT"
echo ""

# Check if sglang is available via uv
if ! uv run python -c "import sglang" 2>/dev/null; then
    echo "Error: sglang not found. Install with:"
    echo "  cd extern/sglang-bitnet && uv pip install -e python/"
    exit 1
fi

# Check if sgl-kernel BitNet is available
uv run python -c "
from sgl_kernel.quantization import bitnet_check_kernel_available
if bitnet_check_kernel_available():
    print('BitNet native kernels: AVAILABLE (AVX2/AVX512)')
else:
    print('WARNING: BitNet native kernels NOT available - using fallback')
"

echo ""
echo "Starting server..."
echo "API endpoint: http://${HOST}:${PORT}/v1/chat/completions"
echo ""

# Launch server with uv
# NOTE: Do NOT use --trust-remote-code, transformers 4.57+ has native BitNet support
# NOTE: --enable-torch-compile reduces Python overhead by ~10-15%
# NOTE: taskset limits to 8 cores to prevent system freeze
exec taskset -c 0-7 uv run python -m sglang.launch_server \
    --model-path "$MODEL" \
    --port "$PORT" \
    --host "$HOST" \
    --device cpu \
    --dtype bfloat16 \
    --enable-torch-compile
