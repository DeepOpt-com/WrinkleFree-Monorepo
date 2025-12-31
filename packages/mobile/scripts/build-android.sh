#!/bin/bash
# Build BitNet inference library for Android with TL1 kernels
#
# Prerequisites:
#   - Android NDK installed via Android Studio SDK Manager
#   - CMake and Ninja: sudo apt install cmake ninja-build
#
# Usage:
#   ./scripts/build-android.sh
#
# Output:
#   build-android/lib/libllmcore.so

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOBILE_DIR="$(dirname "$SCRIPT_DIR")"
INFERENCE_DIR="$MOBILE_DIR/../inference"

# Default NDK path (Android Studio default location)
ANDROID_SDK="${ANDROID_SDK:-$HOME/Android/Sdk}"
ANDROID_NDK="${ANDROID_NDK:-$ANDROID_SDK/ndk/27.0.11718014}"

if [ ! -d "$ANDROID_NDK" ]; then
    # Try to find any NDK version
    NDK_BASE="$ANDROID_SDK/ndk"
    if [ -d "$NDK_BASE" ]; then
        ANDROID_NDK=$(ls -d "$NDK_BASE"/* 2>/dev/null | head -1)
    fi
fi

# Use bundled CMake if available
if [ -d "$ANDROID_SDK/cmake/3.22.1/bin" ]; then
    export PATH="$ANDROID_SDK/cmake/3.22.1/bin:$PATH"
fi

if [ ! -d "$ANDROID_NDK" ]; then
    echo "Error: Android NDK not found."
    echo "Please install via Android Studio SDK Manager or set ANDROID_NDK env var."
    exit 1
fi

echo "Using Android NDK: $ANDROID_NDK"

# Build BitNet.cpp for Android with TL1 kernels
# BitNet submodule is at repo root, not inside inference package
REPO_ROOT="$(dirname "$(dirname "$INFERENCE_DIR")")"
BITNET_DIR="$REPO_ROOT/extern/BitNet"
BUILD_DIR="$MOBILE_DIR/build-android"

echo "Building BitNet for Android (arm64-v8a) with TL1 kernels..."
mkdir -p "$BUILD_DIR"

# Use i2_s quantization (simpler kernels, works on all ARM)
# TL1 requires model-specific kernel generation
cmake -B "$BUILD_DIR" -S "$BITNET_DIR" \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-28 \
    -DCMAKE_C_FLAGS="-march=armv8.4a+dotprod -O3" \
    -DCMAKE_CXX_FLAGS="-march=armv8.4a+dotprod -O3" \
    -DGGML_OPENMP=OFF \
    -DGGML_LLAMAFILE=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -G Ninja

# Build with limited parallelism to avoid system freeze
ninja -C "$BUILD_DIR" -j4

echo ""
echo "Build complete!"
echo "Binary: $BUILD_DIR/bin/llama-cli"
echo ""
echo "To test on device:"
echo "  adb push $BUILD_DIR/bin/llama-cli /data/local/tmp/"
echo "  adb push models/dlm-bitnet-2b-tl1.gguf /data/local/tmp/"
echo "  adb shell 'cd /data/local/tmp && ./llama-cli -m dlm-bitnet-2b-tl1.gguf -p \"Hello\" -n 50'"
