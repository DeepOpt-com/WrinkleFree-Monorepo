#!/bin/bash
# Simple wrapper to start native BitNet inference server.
#
# Usage:
#   ./scripts/serve_native.sh models/dlm-bitnet-2b
#   ./scripts/serve_native.sh models/dlm-bitnet-2b --port 8080
#
# This script:
#   1. Converts checkpoint to .bin format if needed (one-time)
#   2. Starts the native server with optimized TL2 kernels
#
# Performance: ~29 tok/s on GCP c3d-standard-32 (AMD EPYC Genoa)

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
    echo "  $0 /path/to/my-checkpoint"
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
    # Already a .bin file
    BIN_PATH="$CHECKPOINT"
    TOKENIZER_PATH="${CHECKPOINT%.bin}"
else
    # Checkpoint directory - derive .bin path
    BIN_PATH="${CHECKPOINT}.bin"
    TOKENIZER_PATH="$CHECKPOINT"
fi

# Check if checkpoint exists
if [ ! -e "$TOKENIZER_PATH" ] && [ ! -e "$BIN_PATH" ]; then
    echo "Error: Checkpoint not found: $TOKENIZER_PATH"
    echo ""
    echo "To download from GCS:"
    echo "  mkdir -p models/dlm-bitnet-2b"
    echo "  gcloud storage cp -r gs://wrinklefree-checkpoints/dlm/bitnet-b1.58-2B-4T-bf16/checkpoint-step-2800/* models/dlm-bitnet-2b/"
    echo ""
    echo "Or use a HuggingFace model:"
    echo "  python -c \"from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('microsoft/BitNet-b1.58-2B-4T')\""
    exit 1
fi

# Convert to .bin if needed
if [ ! -f "$BIN_PATH" ]; then
    echo "Converting checkpoint to packed format..."
    echo "  Input:  $TOKENIZER_PATH"
    echo "  Output: $BIN_PATH"
    echo ""
    python scripts/convert_to_sglkernel.py "$TOKENIZER_PATH" "$BIN_PATH"
    echo ""
    echo "Conversion complete!"
    echo ""
fi

# Start server
echo "Starting native BitNet server..."
echo "  Model:     $BIN_PATH"
echo "  Tokenizer: $TOKENIZER_PATH"
echo ""

exec python scripts/serve_bitnet_native.py \
    --model "$BIN_PATH" \
    --tokenizer "$TOKENIZER_PATH" \
    "$@"
