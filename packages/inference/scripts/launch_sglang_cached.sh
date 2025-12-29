#!/bin/bash
# Launch SGLang-BitNet with GCS caching
#
# Usage:
#   ./scripts/launch_sglang_cached.sh [model_path] [port]
#
# Environment variables:
#   SGLANG_MODEL - Model path (HuggingFace ID or local path)
#   SGLANG_PORT - Server port
#   SKIP_GCS_CACHE - Set to "1" to skip GCS caching

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration
MODEL="${1:-${SGLANG_MODEL:-microsoft/bitnet-b1.58-2B-4T}}"
PORT="${2:-${SGLANG_PORT:-30000}}"
HOST="${SGLANG_HOST:-0.0.0.0}"
SKIP_GCS="${SKIP_GCS_CACHE:-0}"

echo "=== SGLang-BitNet Server (Cached) ==="
echo "Source Model: $MODEL"
echo "Host:  $HOST"
echo "Port:  $PORT"
echo ""

cd "$PROJECT_DIR"

# Run the caching logic
CACHE_ARGS=""
if [ "$SKIP_GCS" = "1" ]; then
    CACHE_ARGS="--skip-gcs"
fi

echo "Checking cache..."
CACHED_PATH=$(uv run python -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
from wrinklefree_inference.cache import get_cached_or_convert
path = get_cached_or_convert('$MODEL', skip_gcs=$( [ \"$SKIP_GCS\" = \"1\" ] && echo True || echo False ))
print(path)
")

echo ""
echo "Using model path: $CACHED_PATH"
echo ""

# Launch SGLang with cached model
exec uv run python -m sglang.launch_server \
    --model-path "$CACHED_PATH" \
    --port "$PORT" \
    --host "$HOST" \
    --device cpu \
    --dtype bfloat16
