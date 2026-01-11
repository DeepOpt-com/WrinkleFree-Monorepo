#!/bin/bash
# Remote setup script for Rust BitNet server.
#
# Builds everything with -march=native for maximum SIMD optimization.
# NO PYTHON - uses Rust + C++ only for inference.
#
# Called by deploy_vultr.sh after code is synced.

set -euo pipefail

INSTALL_DIR="/opt/wrinklefree"
cd "$INSTALL_DIR"

log() {
    echo "[$(date '+%H:%M:%S')] $1"
}

log "=== Rust BitNet Server Setup ==="

# 1. Install system dependencies
log "Installing system dependencies..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
    build-essential \
    clang \
    cmake \
    ninja-build \
    pkg-config \
    libssl-dev \
    curl \
    git \
    jq \
    nodejs \
    npm \
    protobuf-compiler

# 2. Install Rust (if not present)
if ! command -v cargo &> /dev/null; then
    log "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
source "$HOME/.cargo/env"
rustup update stable

# 3. Install pm2 (if not present)
if ! command -v pm2 &> /dev/null; then
    log "Installing pm2..."
    npm install -g pm2
fi

# 4. Build llama.cpp with native SIMD optimization
log "Building llama.cpp with -march=native (AVX512)..."
LLAMA_DIR="$INSTALL_DIR/packages/inference/extern/sglang-bitnet/3rdparty/llama.cpp"

if [ ! -d "$LLAMA_DIR" ]; then
    log "ERROR: llama.cpp submodule not found at $LLAMA_DIR"
    log "Make sure submodules are synced: git submodule update --init --recursive"
    exit 1
fi

cd "$LLAMA_DIR"

# Create build-info files (workaround for missing cmake submodule)
mkdir -p common/cmake
cat > common/cmake/build-info-gen-cpp.cmake << 'CMAKE_EOF'
file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/common/build-info.cpp"
"int LLAMA_BUILD_NUMBER = 0;
char const *LLAMA_COMMIT = \"unknown\";
char const *LLAMA_COMPILER = \"clang\";
char const *LLAMA_BUILD_TARGET = \"x86_64-linux-gnu\";
")
CMAKE_EOF

cat > common/build-info.cpp << 'CPP_EOF'
int LLAMA_BUILD_NUMBER = 0;
char const *LLAMA_COMMIT = "unknown";
char const *LLAMA_COMPILER = "clang";
char const *LLAMA_BUILD_TARGET = "x86_64-linux-gnu";
CPP_EOF

rm -rf build
mkdir -p build
cd build

# Use clang for better SIMD codegen
cmake .. \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-march=native -mtune=native -O3" \
    -DCMAKE_CXX_FLAGS="-march=native -mtune=native -O3" \
    -DLLAMA_NATIVE=ON \
    -DGGML_NATIVE=ON \
    -G Ninja

ninja -j$(nproc)

log "llama.cpp built successfully"

# 5. Build Rust gateway with native inference
log "Building Rust gateway with native-inference feature..."
GATEWAY_DIR="$INSTALL_DIR/packages/inference/extern/sglang-bitnet/sgl-model-gateway"
cd "$GATEWAY_DIR"

# Set environment for native SIMD in both Rust and C++
export NATIVE_SIMD=1
export RUSTFLAGS="-C target-cpu=native"

# Build with release profile and native-inference feature
cargo build --release --features native-inference -j$(nproc)

log "Rust gateway built successfully"

# 6. Download model (GGUF format for llama.cpp)
log "Downloading BitNet model..."
MODEL_DIR="$INSTALL_DIR/models/BitNet-b1.58-2B-4T"
mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_DIR/ggml-model-i2_s.gguf" ]; then
    # Download from HuggingFace
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf \
            --local-dir "$MODEL_DIR" \
            --include "*.gguf" "*.json"
    else
        # Fallback: use curl
        log "Installing huggingface_hub..."
        pip3 install --quiet huggingface_hub
        huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf \
            --local-dir "$MODEL_DIR" \
            --include "*.gguf" "*.json"
    fi
fi

# Also download tokenizer files
if [ ! -f "$MODEL_DIR/tokenizer.json" ]; then
    log "Downloading tokenizer..."
    pip3 install --quiet huggingface_hub 2>/dev/null || true
    huggingface-cli download microsoft/BitNet-b1.58-2B-4T \
        tokenizer.json tokenizer_config.json \
        --local-dir "$MODEL_DIR"
fi

log "Model downloaded: $MODEL_DIR"

# 7. Create start script for pm2
log "Creating start script..."
cat > "$INSTALL_DIR/packages/inference/deploy/start_rust_server.sh" << 'EOF'
#!/bin/bash
# Start Rust BitNet server with native inference
cd /opt/wrinklefree

# Source cargo
source "$HOME/.cargo/env"

# Set library paths for llama.cpp
export LD_LIBRARY_PATH="/opt/wrinklefree/packages/inference/extern/sglang-bitnet/3rdparty/llama.cpp/build/src:/opt/wrinklefree/packages/inference/extern/sglang-bitnet/3rdparty/llama.cpp/build/ggml/src:${LD_LIBRARY_PATH:-}"

# Use all available CPUs
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)

# Model paths
MODEL_GGUF="/opt/wrinklefree/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"

# Run the Rust server
exec /opt/wrinklefree/packages/inference/extern/sglang-bitnet/sgl-model-gateway/target/release/native_server \
    --model-path "$MODEL_GGUF" \
    --host 0.0.0.0 \
    --port 30000
EOF
chmod +x "$INSTALL_DIR/packages/inference/deploy/start_rust_server.sh"

# 8. Print CPU info for verification
log ""
log "=== CPU Information ==="
lscpu | grep -E "Model name|CPU MHz|Flags" | head -5
grep -o 'avx512[^ ]*' /proc/cpuinfo | sort -u | head -5 || echo "No AVX512 detected"

# 8. Open firewall ports
log "Opening firewall ports..."
ufw allow 30000/tcp > /dev/null 2>&1 || true
ufw allow 7860/tcp > /dev/null 2>&1 || true

# 9. Install streamlit dependencies
log "Installing Streamlit dependencies..."
pip3 install --break-system-packages -q streamlit requests

log ""
log "=== Setup Complete ==="
log "Binary: $GATEWAY_DIR/target/release/native_server"
log "Model:  $MODEL_DIR/ggml-model-i2_s.gguf"
