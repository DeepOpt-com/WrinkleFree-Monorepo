#!/bin/bash
# Launch the Rust wf-inference engine with native BitNet inference.
#
# This uses pure Rust SIMD kernels for optimal performance.
#
# Usage:
#   ./scripts/launch_rust_gateway.sh [--native] [--port PORT]
#
# Options:
#   --native       Use native C++ inference (default if available)
#   --grpc         Use Python gRPC backend (fallback)
#   --port PORT    Server port (default: 30000)
#   --model PATH   Model path (default: microsoft/bitnet-b1.58-2B-4T)
#
# Performance comparison:
#   - Native inference: ~26 tok/s (matches BitNet.cpp)
#   - gRPC to Python:   ~19 tok/s (49ms overhead)

set -euo pipefail

# Source cargo if available
if [[ -f "$HOME/.cargo/env" ]]; then
    source "$HOME/.cargo/env"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
GATEWAY_DIR="${PROJECT_DIR}/rust"

# Default configuration
PORT="${PORT:-30000}"
HOST="${HOST:-0.0.0.0}"
MODEL="${MODEL:-microsoft/bitnet-b1.58-2B-4T}"
BACKEND="${BACKEND:-native}"  # native or grpc

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --native)
            BACKEND="native"
            shift
            ;;
        --grpc)
            BACKEND="grpc"
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== WrinkleFree Inference Engine ==="
echo "Backend: $BACKEND"
echo "Model:   $MODEL"
echo "Host:    $HOST"
echo "Port:    $PORT"
echo ""

cd "$GATEWAY_DIR"

# Build with appropriate features
FEATURES=""
if [[ "$BACKEND" == "native" ]]; then
    FEATURES="--features native-inference"
    echo "Building with native inference (C++ SIMD kernels)..."
else
    echo "Building with gRPC backend (Python scheduler)..."
fi

# Build in release mode (limit to 4 jobs to prevent system freeze)
cargo build --release $FEATURES -j4

# Set library path for llama.cpp shared libraries (for dlm_server)
export LD_LIBRARY_PATH="${PROJECT_DIR}/extern/llama.cpp/build/src:${PROJECT_DIR}/extern/llama.cpp/build/ggml/src:${LD_LIBRARY_PATH:-}"

# Run the gateway
echo ""
echo "Starting gateway..."
echo "API endpoint: http://${HOST}:${PORT}/v1/chat/completions"
echo ""

if [[ "$BACKEND" == "native" ]]; then
    # Native mode: use the native_server binary with C++ inference
    # Find the GGUF model file
    GGUF_PATH=""
    if [[ -f "$MODEL" ]]; then
        GGUF_PATH="$MODEL"
    elif [[ -d "$MODEL" ]]; then
        # Look for GGUF in directory
        GGUF_PATH=$(find "$MODEL" -name "*.gguf" | head -1)
    else
        # Check in models directory (project root)
        GGUF_PATH="${PROJECT_DIR}/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
    fi

    if [[ ! -f "$GGUF_PATH" ]]; then
        echo "Model not found at: $GGUF_PATH"
        echo "Downloading microsoft/BitNet-b1.58-2B-4T-gguf..."
        mkdir -p "${PROJECT_DIR}/models"
        huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir "${PROJECT_DIR}/models/BitNet-b1.58-2B-4T"
        GGUF_PATH="${PROJECT_DIR}/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
        if [[ ! -f "$GGUF_PATH" ]]; then
            echo "Error: Download failed"
            exit 1
        fi
    fi

    # Ensure tokenizer is present
    TOKENIZER_PATH="${PROJECT_DIR}/models/BitNet-b1.58-2B-4T/tokenizer.json"
    if [[ ! -f "$TOKENIZER_PATH" ]]; then
        echo "Downloading tokenizer..."
        huggingface-cli download microsoft/BitNet-b1.58-2B-4T tokenizer.json tokenizer_config.json --local-dir "${PROJECT_DIR}/models/BitNet-b1.58-2B-4T"
    fi

    echo "Model file: $GGUF_PATH"
    exec cargo run --release $FEATURES --bin native_server -- \
        --host "$HOST" \
        --port "$PORT" \
        --model-path "$GGUF_PATH"
else
    # gRPC mode: connect to Python scheduler
    # User needs to start Python scheduler separately
    echo "Note: Start the Python scheduler first with:"
    echo "  ./scripts/launch_sglang_bitnet.sh"
    echo ""
    exec cargo run --release -- \
        --host "$HOST" \
        --port "$PORT" \
        --grpc-endpoint "http://localhost:50051"
fi
