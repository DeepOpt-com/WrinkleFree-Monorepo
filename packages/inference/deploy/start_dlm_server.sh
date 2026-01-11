#!/bin/bash
# Start Rust DLM server with Fast-dLLM v2 block diffusion
# NO PYTHON - uses native C++ inference via Rust FFI
cd /opt/wrinklefree

# Set library paths for llama.cpp
export LD_LIBRARY_PATH="/opt/wrinklefree/packages/inference/extern/sglang-bitnet/3rdparty/llama.cpp/build/src:/opt/wrinklefree/packages/inference/extern/sglang-bitnet/3rdparty/llama.cpp/build/ggml/src:${LD_LIBRARY_PATH:-}"

# Use all available CPUs
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)

# Model path - use DLM GGUF model
MODEL_GGUF="${DLM_MODEL:-/opt/wrinklefree/models/dlm-bitnet-2b.gguf}"

# Fallback to BitNet model if DLM model doesn't exist
if [ ! -f "$MODEL_GGUF" ]; then
    echo "DLM model not found at $MODEL_GGUF"
    MODEL_GGUF="/opt/wrinklefree/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
    echo "Falling back to: $MODEL_GGUF"
fi

# Run the DLM server with Fast-dLLM v2 block diffusion
exec /opt/wrinklefree/packages/inference/extern/sglang-bitnet/sgl-model-gateway/target/release/dlm_server \
    --model-path "$MODEL_GGUF" \
    --host 0.0.0.0 \
    --port 30000 \
    --block-size 32 \
    --threshold 0.95
