#!/bin/bash
# Download and build llama.cpp for GGUF conversion and dlm_server
#
# Usage:
#   ./scripts/setup_llama_cpp.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LLAMA_DIR="${PROJECT_DIR}/extern/llama.cpp"

# Check if already built
if [ -d "$LLAMA_DIR" ] && [ -f "$LLAMA_DIR/build/src/libllama.so" ]; then
    echo "llama.cpp already built at $LLAMA_DIR"
    exit 0
fi

# Clone if not present
if [ ! -d "$LLAMA_DIR" ]; then
    echo "Cloning llama.cpp..."
    git clone --depth 1 --branch merge-dev \
        https://github.com/Eddie-Wang1120/llama.cpp.git "$LLAMA_DIR"
fi

# Build
echo "Building llama.cpp..."
cd "$LLAMA_DIR"
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
cmake --build build -j$(nproc)

echo ""
echo "llama.cpp built successfully!"
echo "Libraries: $LLAMA_DIR/build/src/libllama.so"
