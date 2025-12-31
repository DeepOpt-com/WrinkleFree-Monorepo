#!/bin/bash
# Native BitNet server limited to 4 CPUs (for local development).
#
# Usage:
#   ./scripts/serve_native_4cpu.sh models/dlm-bitnet-2b
#   ./scripts/serve_native_4cpu.sh models/dlm-bitnet-2b --port 8080
#
# This script:
#   1. Converts checkpoint to .bin format if needed (one-time)
#   2. Starts the native server pinned to 4 CPU cores
#
# For full performance, use serve_native.sh on a server with more cores.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Check arguments
if [ -z "$1" ]; then
    echo "Usage: $0 <checkpoint_path> [--port PORT]"
    echo ""
    echo "Examples:"
    echo "  $0 models/dlm-bitnet-2b"
    echo "  $0 models/dlm-bitnet-2b --port 8080"
    exit 1
fi

CHECKPOINT="$1"
shift  # Remove first arg, remaining args passed to server

# Handle both absolute and relative paths
if [[ "$CHECKPOINT" != /* ]]; then
    CHECKPOINT="$PWD/$CHECKPOINT"
fi

# Determine .bin path (alongside checkpoint directory)
if [[ "$CHECKPOINT" == *.bin ]]; then
    BIN_PATH="$CHECKPOINT"
    TOKENIZER_PATH="${CHECKPOINT%.bin}"
else
    BIN_PATH="${CHECKPOINT}.bin"
    TOKENIZER_PATH="$CHECKPOINT"
fi

# Check if checkpoint exists
if [ ! -e "$TOKENIZER_PATH" ] && [ ! -e "$BIN_PATH" ]; then
    echo "Error: Checkpoint not found: $TOKENIZER_PATH"
    echo ""
    echo "To download from GCS (model files only, excludes optimizer state):"
    echo "  mkdir -p models/dlm-bitnet-2b"
    echo "  gcloud storage cp \\"
    echo "      'gs://wrinklefree-checkpoints/dlm/bitnet-b1.58-2B-4T-bf16/checkpoint-step-2800/*.json' \\"
    echo "      'gs://wrinklefree-checkpoints/dlm/bitnet-b1.58-2B-4T-bf16/checkpoint-step-2800/*.safetensors' \\"
    echo "      'gs://wrinklefree-checkpoints/dlm/bitnet-b1.58-2B-4T-bf16/checkpoint-step-2800/*.jinja' \\"
    echo "      models/dlm-bitnet-2b/"
    exit 1
fi

# Convert to .bin if needed
if [ ! -f "$BIN_PATH" ]; then
    echo "Converting checkpoint to packed format..."
    echo "  Input:  $TOKENIZER_PATH"
    echo "  Output: $BIN_PATH"
    echo ""
    uv run python scripts/convert_to_sglkernel.py "$TOKENIZER_PATH" "$BIN_PATH"
    echo ""
    echo "Conversion complete!"
    echo ""
fi

# Start server with 4 CPU limit
echo "Starting native BitNet server (4 CPU limit)..."
echo "  Model:     $BIN_PATH"
echo "  Tokenizer: $TOKENIZER_PATH"
echo "  CPUs:      0-3 (4 cores)"
echo ""

exec taskset -c 0-3 uv run python scripts/serve_bitnet_native.py \
    --model "$BIN_PATH" \
    --tokenizer "$TOKENIZER_PATH" \
    "$@"
