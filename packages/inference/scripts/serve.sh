#!/bin/bash
# Launch inference server + Streamlit chat UI
#
# Usage:
#   ./scripts/serve.sh                  # Start both server and UI (default backend)
#   ./scripts/serve.sh --backend rust   # Use Rust gateway with native inference
#   ./scripts/serve.sh --backend bitnet # Use BitNet.cpp
#   ./scripts/serve.sh --backend sglang # Use SGLang Python server
#   ./scripts/serve.sh --server         # Start only server
#   ./scripts/serve.sh --ui             # Start only UI (assumes server running)
#
# Backends (from fastest to slowest):
#   rust   - Rust gateway + C++ kernels (26+ tok/s, no Python)
#   bitnet - BitNet.cpp (26+ tok/s, llama.cpp based)
#   sglang - SGLang Python (16-19 tok/s, most features)
#
# Environment variables:
#   SGLANG_PORT - Server port (default: 30000, BitNet.cpp uses 8080)
#   STREAMLIT_PORT - Streamlit UI port (default: 7860)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Configuration
MODEL="${SGLANG_MODEL:-microsoft/bitnet-b1.58-2B-4T}"
SERVER_PORT="${SGLANG_PORT:-30000}"
STREAMLIT_PORT="${STREAMLIT_PORT:-7860}"
HOST="0.0.0.0"
BACKEND="${BACKEND:-sglang}"  # rust, bitnet, or sglang

# Parse args
START_SERVER=true
START_UI=true
while [[ $# -gt 0 ]]; do
    case $1 in
        --server)
            START_UI=false
            shift
            ;;
        --ui)
            START_SERVER=false
            shift
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--server|--ui] [--backend rust|bitnet|sglang]"
            exit 1
            ;;
    esac
done

# Adjust port for BitNet.cpp
if [[ "$BACKEND" == "bitnet" ]]; then
    SERVER_PORT="${BITNET_PORT:-8080}"
fi

cleanup() {
    echo ""
    echo "Shutting down..."
    pkill -f "sglang.launch_server.*$SERVER_PORT" 2>/dev/null || true
    pkill -f "llama-server.*$SERVER_PORT" 2>/dev/null || true
    pkill -f "sgl-model-gateway.*$SERVER_PORT" 2>/dev/null || true
    pkill -f "streamlit.*serve_sglang.py" 2>/dev/null || true
}
trap cleanup EXIT

# Start server
if $START_SERVER; then
    echo "=== Starting Server ==="
    echo "Backend: $BACKEND"
    echo "Model:   $MODEL"
    echo "Port:    $SERVER_PORT"
    echo ""

    case "$BACKEND" in
        rust)
            echo "Using Rust gateway with native C++ inference"
            echo "Performance: ~26 tok/s (no Python overhead)"
            echo ""
            "$SCRIPT_DIR/launch_rust_gateway.sh" --native --port "$SERVER_PORT" &
            SERVER_PID=$!
            ;;

        bitnet)
            echo "Using BitNet.cpp server"
            echo "Performance: ~26 tok/s"
            echo ""
            "$SCRIPT_DIR/launch_bitnet_cpp.sh" "$MODEL" "$SERVER_PORT" &
            SERVER_PID=$!
            ;;

        sglang)
            echo "Using SGLang Python server"
            echo "Performance: ~16-19 tok/s"
            echo ""
            # Kill any existing server on this port
            pkill -f "sglang.launch_server.*$SERVER_PORT" 2>/dev/null || true
            sleep 1

            uv run python -m sglang.launch_server \
                --model-path "$MODEL" \
                --port "$SERVER_PORT" \
                --host "$HOST" \
                --device cpu \
                --dtype bfloat16 &
            SERVER_PID=$!
            ;;

        *)
            echo "Error: Unknown backend '$BACKEND'"
            echo "Valid backends: rust, bitnet, sglang"
            exit 1
            ;;
    esac

    # Wait for server to be ready
    echo "Waiting for server to start..."
    for i in {1..120}; do
        # Try health endpoint first, fall back to models
        if curl -s "http://127.0.0.1:$SERVER_PORT/health" >/dev/null 2>&1 || \
           curl -s "http://127.0.0.1:$SERVER_PORT/v1/models" >/dev/null 2>&1; then
            echo "Server ready!"
            break
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "Error: Server process died"
            exit 1
        fi
        sleep 1
    done
fi

# Start Streamlit UI
if $START_UI; then
    echo ""
    echo "=== Starting Streamlit UI ==="
    echo "URL: http://$HOST:$STREAMLIT_PORT"

    # Set backend URL for Streamlit
    export SGLANG_URL="http://127.0.0.1:$SERVER_PORT"

    # Set BITNET_BACKEND based on backend type
    case "$BACKEND" in
        rust|sglang)
            export BITNET_BACKEND=sglang  # Use sglang protocol (OpenAI-compatible)
            ;;
        bitnet)
            export BITNET_BACKEND=bitnet_cpp
            export BITNET_URL="http://127.0.0.1:$SERVER_PORT"
            ;;
    esac

    uv run streamlit run demo/serve_sglang.py \
        --server.port "$STREAMLIT_PORT" \
        --server.address "$HOST" \
        --server.headless true
else
    # Keep server running
    echo ""
    echo "Server running at http://$HOST:$SERVER_PORT"
    echo "Press Ctrl+C to stop"
    wait
fi
