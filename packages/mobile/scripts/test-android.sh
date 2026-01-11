#!/bin/bash
# Test BitNet inference on Android device via ADB
#
# Prerequisites:
#   - Android device connected via USB with USB debugging enabled
#   - Model file in models/ directory
#
# Usage:
#   ./scripts/test-android.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOBILE_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$MOBILE_DIR/build-android"

# Add platform-tools to PATH
export PATH="$HOME/Android/Sdk/platform-tools:$PATH"

# Check for device
echo "Checking for connected Android device..."
DEVICES=$(adb devices | grep -v "List" | grep -v "^$" | wc -l)
if [ "$DEVICES" -eq 0 ]; then
    echo "Error: No Android device connected."
    echo "Please connect your device via USB and enable USB debugging."
    exit 1
fi

echo "Device found!"
adb devices

# Create directory on device
echo ""
echo "Setting up test directory on device..."
adb shell "mkdir -p /data/local/tmp/bitnet"

# Push binary and shared libraries
echo "Pushing llama-cli binary..."
adb push "$BUILD_DIR/bin/llama-cli" /data/local/tmp/bitnet/

echo "Pushing shared libraries..."
adb push "$BUILD_DIR/3rdparty/llama.cpp/src/libllama.so" /data/local/tmp/bitnet/
adb push "$BUILD_DIR/3rdparty/llama.cpp/ggml/src/libggml.so" /data/local/tmp/bitnet/

# Make executable
adb shell "chmod +x /data/local/tmp/bitnet/llama-cli"

# Push model (takes a while for 1.2GB)
MODEL_FILE="$MOBILE_DIR/models/ggml-model-i2_s.gguf"
if [ -f "$MODEL_FILE" ]; then
    echo ""
    echo "Pushing model file (1.2GB, this may take a few minutes)..."
    adb push "$MODEL_FILE" /data/local/tmp/bitnet/
else
    echo "Error: Model file not found at $MODEL_FILE"
    echo "Download with: huggingface-cli download microsoft/bitnet-b1.58-2B-4T-gguf ggml-model-i2_s.gguf --local-dir models/"
    exit 1
fi

# Run inference test
echo ""
echo "Running inference test..."
echo "============================================"
adb shell "cd /data/local/tmp/bitnet && export LD_LIBRARY_PATH=. && ./llama-cli -m ggml-model-i2_s.gguf -p 'The meaning of life is' -n 50 --threads 4"
echo "============================================"
echo ""
echo "Test complete!"
echo "To run more tests manually:"
echo "  adb shell"
echo "  cd /data/local/tmp/bitnet"
echo "  export LD_LIBRARY_PATH=."
echo "  ./llama-cli -m ggml-model-i2_s.gguf -p \"Your prompt here\" -n 100 --threads 4"
