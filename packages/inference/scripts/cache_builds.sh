#!/bin/bash
# Build caching script for sglang-bitnet and BitNet.cpp
# Uses GCS bucket to cache build artifacts based on source hash

set -e

GCS_BUCKET="gs://wrinklefree-build-cache"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Calculate hash of source files
calculate_hash() {
    local dir="$1"
    find "$dir" -type f \
        -not -path "*/.git/*" \
        -not -path "*/__pycache__/*" \
        -not -path "*/build/*" \
        -not -path "*/models/*" \
        -not -path "*/*.so" \
        -not -path "*/*.o" \
        \( -name "*.cpp" -o -name "*.h" -o -name "*.py" -o -name "*.cmake" -o -name "CMakeLists.txt" \) \
        -print0 2>/dev/null | xargs -0 sha256sum 2>/dev/null | sort | sha256sum | cut -d' ' -f1
}

# Check and download cache
check_cache() {
    local name="$1"
    local hash="$2"
    local target="$3"

    echo "Checking cache for $name (hash: ${hash:0:12}...)"
    if gsutil -q stat "$GCS_BUCKET/$name/$hash/marker" 2>/dev/null; then
        echo "Cache HIT for $name"
        mkdir -p "$(dirname "$target")"
        gsutil -m cp -r "$GCS_BUCKET/$name/$hash/build" "$target" 2>/dev/null && return 0
    fi
    echo "Cache MISS for $name"
    return 1
}

# Upload to cache
upload_cache() {
    local name="$1"
    local hash="$2"
    local source="$3"

    echo "Uploading $name to cache (hash: ${hash:0:12}...)"
    gsutil -m cp -r "$source" "$GCS_BUCKET/$name/$hash/build" 2>/dev/null || true
    echo "uploaded" | gsutil cp - "$GCS_BUCKET/$name/$hash/marker" 2>/dev/null || true
}

echo "=== Build Cache Script ==="
echo "Project root: $PROJECT_ROOT"

# --- sglang-bitnet ---
echo ""
echo "=== sglang-bitnet ==="
SGLANG_DIR="extern/sglang-bitnet/sgl-kernel"
SGLANG_HASH=$(calculate_hash "$SGLANG_DIR")
SGLANG_BUILD="$SGLANG_DIR/build"
SGLANG_SO_DIR="$SGLANG_DIR/python/sgl_kernel"

if [ -f "$SGLANG_SO_DIR/common_ops.cpython"*".so" ]; then
    echo "sglang-bitnet already built locally"
elif check_cache "sglang-bitnet" "$SGLANG_HASH" "$SGLANG_BUILD"; then
    echo "Restored sglang-bitnet from cache"
    # Also need to install the package
    pip install -e "$SGLANG_DIR" --no-build-isolation 2>/dev/null || true
else
    echo "Building sglang-bitnet..."
    pip install scikit-build-core cmake ninja 2>/dev/null || true
    pip install -e "$SGLANG_DIR" --no-build-isolation
    # Copy .so to source dir for editable install
    cp .venv/lib/python*/site-packages/sgl_kernel/common_ops.*.so "$SGLANG_SO_DIR/" 2>/dev/null || true
    upload_cache "sglang-bitnet" "$SGLANG_HASH" "$SGLANG_BUILD"
fi

# --- BitNet.cpp ---
echo ""
echo "=== BitNet.cpp ==="
BITNET_DIR="extern/BitNet"
BITNET_HASH=$(calculate_hash "$BITNET_DIR")
BITNET_BUILD="$BITNET_DIR/build"
BITNET_SERVER="$BITNET_BUILD/bin/llama-server"

if [ -f "$BITNET_SERVER" ]; then
    echo "BitNet.cpp already built locally"
elif check_cache "bitnet-cpp" "$BITNET_HASH" "$BITNET_BUILD"; then
    echo "Restored BitNet.cpp from cache"
else
    echo "Building BitNet.cpp..."
    cd "$BITNET_DIR"

    # Download model if not present
    if [ ! -d "models/BitNet-b1.58-2B-4T" ]; then
        echo "Downloading GGUF model..."
        pip install huggingface_hub 2>/dev/null || true
        huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf \
            --local-dir models/BitNet-b1.58-2B-4T \
            --include "*.gguf" 2>/dev/null || true
    fi

    # Build
    python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s

    cd "$PROJECT_ROOT"
    upload_cache "bitnet-cpp" "$BITNET_HASH" "$BITNET_BUILD"
fi

echo ""
echo "=== Build complete ==="
[ -f "$SGLANG_SO_DIR/common_ops"*".so" ] && echo "sglang-bitnet: OK" || echo "sglang-bitnet: MISSING"
[ -f "$BITNET_SERVER" ] && echo "BitNet.cpp: OK" || echo "BitNet.cpp: MISSING"
